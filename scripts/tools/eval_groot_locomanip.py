# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause
"""Closed-loop GR00T evaluation for locomanipulation tasks.

Uses the Isaac-PickPlace-Locomanipulation-Camera-G1-Abs-v0 environment with:
- Agile RL locomotion policy embedded as ActionTerm (runs inside env.step())
- JointPositionActionCfg for upper body (replaces Pink IK)
- GR00T model via PolicyClient for stereo cameras + joint state → actions

Usage:
    # Terminal 1: Start GR00T server
    cd /home/mats/Bot/Nvidia/Isaac-GR00T
    uv run python -c "
    exec(open('examples/G1TriHand/g1_trihand_stereo_config.py').read())
    from gr00t.eval.run_gr00t_server import main, ServerConfig
    import tyro; main(tyro.cli(ServerConfig))
    " --model-path /home/mats/Bot/Models/g1_locomanip_finetune/checkpoint-3000 \
      --embodiment-tag NEW_EMBODIMENT --port 5555

    # Terminal 2: Run this eval script
    cd /home/mats/Bot/unitree/IsaacLab
    ./isaaclab.sh -p scripts/tools/eval_groot_locomanip.py \
        --task Isaac-PickPlace-Locomanipulation-Camera-G1-Abs-v0 \
        --policy_port 5555 --action_horizon 8 \
        --num_envs 1 --enable_cameras
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Evaluate GR00T locomanip model in Isaac Lab.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments.")
parser.add_argument(
    "--task",
    type=str,
    default="Isaac-PickPlace-Locomanipulation-Camera-G1-Abs-v0",
    help="Task name.",
)
parser.add_argument("--policy_host", type=str, default="localhost", help="GR00T server host.")
parser.add_argument("--policy_port", type=int, default=5555, help="GR00T server port.")
parser.add_argument(
    "--task_description",
    type=str,
    default="pick up the wheel with left hand, grab with right hand, walk right, and place it in the basket",
    help="Language instruction for the task.",
)
parser.add_argument("--action_horizon", type=int, default=8, help="Actions to execute per query.")
parser.add_argument("--max_steps", type=int, default=1000, help="Maximum steps per episode.")
parser.add_argument("--save_video", action="store_true", default=False, help="Save video.")

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# Add Isaac-GR00T to path
sys.path.insert(0, "/home/mats/Bot/Nvidia/Isaac-GR00T")

# Launch simulator
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import contextlib
import time
from collections import OrderedDict

import cv2
import gymnasium as gym
import numpy as np
import torch

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg
from isaaclab.envs.mdp.actions.actions_cfg import JointPositionActionCfg

from gr00t.policy.server_client import PolicyClient


def make_groot_eval_compatible(env_cfg):
    """Replace Pink IK with JointPositionActionCfg for GR00T eval.

    Keeps AgileBasedLowerBodyActionCfg intact so the agile RL locomotion
    policy runs inside env.step(). Disables recorders and terminations.
    """
    env_cfg.actions.upper_body_ik = JointPositionActionCfg(
        asset_name="robot",
        joint_names=[
            "waist_.*_joint",
            ".*_shoulder_.*_joint",
            ".*_elbow_joint",
            ".*_wrist_.*_joint",
            ".*_hand_.*_joint",
        ],
        scale=1.0,
        use_default_offset=False,  # GR00T outputs absolute positions
    )

    if hasattr(env_cfg, "recorders") and env_cfg.recorders is not None:
        env_cfg.recorders = None

    # Disable all termination terms (object_dropping, success, time_out)
    # so the eval runs open-ended without mid-episode resets
    if hasattr(env_cfg, "terminations") and env_cfg.terminations is not None:
        for attr in list(vars(env_cfg.terminations).keys()):
            if not attr.startswith("_"):
                delattr(env_cfg.terminations, attr)

    env_cfg.episode_length_s = 99999.0


class LocomanipGR00TAdapter:
    """Adapter between IsaacLab locomanip env and GR00T policy server.

    Action tensor layout for env.step():
      [upper_body(31), lower_body(4)]
    - upper_body: waist(3) + arms(14) + hands(14) in URDF order
    - lower_body: [vx, vy, wz, hip_height] → agile RL policy
    """

    # Joint patterns per state group (URDF-ordered within each)
    STATE_GROUPS = OrderedDict([
        ("left_leg", ["left_hip_.*_joint", "left_knee_joint", "left_ankle_.*_joint"]),
        ("right_leg", ["right_hip_.*_joint", "right_knee_joint", "right_ankle_.*_joint"]),
        ("waist", ["waist_.*_joint"]),
        ("left_arm", ["left_shoulder_.*_joint", "left_elbow_joint", "left_wrist_.*_joint"]),
        ("left_hand", ["left_hand_.*_joint"]),
        ("right_arm", ["right_shoulder_.*_joint", "right_elbow_joint", "right_wrist_.*_joint"]),
        ("right_hand", ["right_hand_.*_joint"]),
    ])

    # Joint patterns per upper-body action group
    ACTION_GROUPS = OrderedDict([
        ("left_arm", ["left_shoulder_.*_joint", "left_elbow_joint", "left_wrist_.*_joint"]),
        ("right_arm", ["right_shoulder_.*_joint", "right_elbow_joint", "right_wrist_.*_joint"]),
        ("left_hand", ["left_hand_.*_joint"]),
        ("right_hand", ["right_hand_.*_joint"]),
        ("waist", ["waist_.*_joint"]),
    ])

    def __init__(self, policy_client, task_description, env):
        self.policy = policy_client
        self.task_description = task_description
        self._saved_debug = False

        robot = env.scene["robot"]
        device = robot.device

        # ── State DOF index maps ──
        self.state_dof_indices = {}
        total_state = 0
        for group, patterns in self.STATE_GROUPS.items():
            ids, names = robot.find_joints(patterns)
            self.state_dof_indices[group] = torch.tensor(ids, device=device, dtype=torch.long)
            total_state += len(ids)
            print(f"[Adapter] State '{group}': {len(ids)} DOF → {names}")
        print(f"[Adapter] Total state DOF: {total_state}")

        # ── Action index maps ──
        upper_term = env.action_manager._terms["upper_body_ik"]
        upper_joint_names = list(upper_term._joint_names)
        self.upper_dim = upper_term.action_dim
        self.lower_dim = 4  # [vx, vy, wz, height]

        name_to_idx = {name: i for i, name in enumerate(upper_joint_names)}

        self.action_group_indices = {}
        for group, patterns in self.ACTION_GROUPS.items():
            _, names = robot.find_joints(patterns)
            indices = [name_to_idx[n] for n in names]
            self.action_group_indices[group] = indices
            print(f"[Adapter] Action '{group}': {len(indices)} joints → indices {indices}")

        print(f"[Adapter] Action tensor: [{self.upper_dim} upper, {self.lower_dim} lower]"
              f" = {self.upper_dim + self.lower_dim} total")

    def obs_to_groot(self, env, env_id=0):
        """Build GR00T observation from current sim state."""
        robot = env.scene["robot"]
        joint_pos = robot.data.joint_pos[env_id]

        # Per-group joint positions
        state_dict = {}
        for group, idx in self.state_dof_indices.items():
            values = joint_pos[idx].cpu().numpy().astype(np.float32)
            state_dict[group] = values[np.newaxis, np.newaxis, :]  # (1, 1, N)

        # Stereo cameras
        cam_left = env.scene["left_high_camera"].data.output["rgb"][env_id].cpu().numpy()
        cam_right = env.scene["right_high_camera"].data.output["rgb"][env_id].cpu().numpy()

        if cam_left.shape[-1] == 4:
            cam_left = cam_left[..., :3]
        if cam_right.shape[-1] == 4:
            cam_right = cam_right[..., :3]
        if cam_left.dtype != np.uint8:
            cam_left = (cam_left * 255).clip(0, 255).astype(np.uint8)
        if cam_right.dtype != np.uint8:
            cam_right = (cam_right * 255).clip(0, 255).astype(np.uint8)

        video_dict = {
            "cam_left_high": cam_left[np.newaxis, np.newaxis, ...],   # (1, 1, H, W, 3)
            "cam_right_high": cam_right[np.newaxis, np.newaxis, ...],
        }

        language_dict = {
            "annotation.human.task_description": [[self.task_description]],
        }

        # Debug: first frame
        if not self._saved_debug:
            self._saved_debug = True
            cv2.imwrite("/tmp/locomanip_cam_left.png", cv2.cvtColor(cam_left, cv2.COLOR_RGB2BGR))
            cv2.imwrite("/tmp/locomanip_cam_right.png", cv2.cvtColor(cam_right, cv2.COLOR_RGB2BGR))
            print(f"[Adapter] Debug images → /tmp/locomanip_cam_*.png")
            print(f"[Adapter] cam_left: {cam_left.shape}, cam_right: {cam_right.shape}")
            for k, v in state_dict.items():
                print(f"[Adapter]   state.{k}: {v.shape} = {v[0, 0]}")

        return {"video": video_dict, "state": state_dict, "language": language_dict}

    def decode_action(self, action_chunk, timestep=0):
        """Extract one timestep from the action chunk."""
        result = {}
        for key in action_chunk:
            arr = np.asarray(action_chunk[key])
            result[key] = arr[0, timestep]
        return result

    def build_action_tensor(self, action_dict, device, dtype):
        """Map GR00T action groups → env.step() action tensor.

        Returns tensor of shape (1, upper_dim + 4).

        The action tensor is split by the action manager:
          action[:, :upper_dim]  → JointPositionActionCfg (arms/hands/waist)
          action[:, upper_dim:]  → AgileBasedLowerBodyAction ([vx, vy, wz, height])
        """
        upper = torch.zeros(1, self.upper_dim, device=device, dtype=dtype)
        for group in ["left_arm", "right_arm", "left_hand", "right_hand", "waist"]:
            if group in action_dict:
                values = torch.tensor(action_dict[group], device=device, dtype=dtype)
                for i, idx in enumerate(self.action_group_indices[group]):
                    upper[0, idx] = values[i]

        nav = action_dict.get("navigate_command", np.zeros(3, dtype=np.float32))
        height = action_dict.get("base_height_command", np.zeros(1, dtype=np.float32))
        lower = torch.tensor(
            [[nav[0], nav[1], nav[2], height[0]]], device=device, dtype=dtype
        )
        return torch.cat([upper, lower], dim=-1)


def main():
    print(f"Connecting to GR00T server at {args_cli.policy_host}:{args_cli.policy_port}")
    policy_client = PolicyClient(host=args_cli.policy_host, port=args_cli.policy_port)
    if not policy_client.ping():
        raise RuntimeError("Failed to connect to GR00T server.")
    print("Connected to GR00T server")

    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
    make_groot_eval_compatible(env_cfg)

    env = gym.make(args_cli.task, cfg=env_cfg).unwrapped
    adapter = LocomanipGR00TAdapter(policy_client, args_cli.task_description, env)

    device = env.device
    dtype = env.scene["robot"].data.joint_pos.dtype

    obs_dict, _ = env.reset()

    video_frames = [] if args_cli.save_video else None
    step_count = 0
    action_queue = []

    print(f"\nStarting GR00T locomanip evaluation")
    print(f"  Task: {args_cli.task_description}")
    print(f"  Action horizon: {args_cli.action_horizon}")
    print(f"  Max steps: {args_cli.max_steps}")

    with contextlib.suppress(KeyboardInterrupt), torch.inference_mode():
        while simulation_app.is_running() and not simulation_app.is_exiting():

            if len(action_queue) == 0:
                t0 = time.perf_counter()
                groot_obs = adapter.obs_to_groot(env)
                action_chunk, info = adapter.policy.get_action(groot_obs)
                latency_ms = (time.perf_counter() - t0) * 1000

                horizon = np.asarray(action_chunk["left_arm"]).shape[1]
                for t in range(min(args_cli.action_horizon, horizon)):
                    action_queue.append(adapter.decode_action(action_chunk, t))

                if step_count == 0:
                    a0 = action_queue[0]
                    print(f"\n[Eval] First chunk: horizon={horizon}, latency={latency_ms:.0f}ms")
                    for k, v in a0.items():
                        print(f"  {k}: shape={v.shape}, values={v[:4]}...")
                elif step_count % 50 == 0:
                    print(f"[Eval] Step {step_count}, latency={latency_ms:.0f}ms")

            action_dict = action_queue.pop(0)
            action_tensor = adapter.build_action_tensor(action_dict, device, dtype)

            obs_dict, _, terminated, truncated, _ = env.step(action_tensor)

            if video_frames is not None:
                cam = env.scene["left_high_camera"].data.output["rgb"][0].cpu().numpy()
                if cam.shape[-1] == 4:
                    cam = cam[..., :3]
                if cam.dtype != np.uint8:
                    cam = (cam * 255).clip(0, 255).astype(np.uint8)
                video_frames.append(cam)

            step_count += 1
            if step_count % 100 == 0:
                print(f"Step {step_count}/{args_cli.max_steps}")
            if step_count >= args_cli.max_steps:
                print(f"\nReached max steps ({args_cli.max_steps})")
                break

    if video_frames and len(video_frames) > 0:
        video_path = "/tmp/groot_locomanip_eval.mp4"
        h, w = video_frames[0].shape[:2]
        writer = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*"mp4v"), 20.0, (w, h))
        for frame in video_frames:
            writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        writer.release()
        print(f"Saved {len(video_frames)} frames to {video_path}")

    print(f"\n{'='*50}")
    print(f"Evaluation complete: {step_count} steps")
    print(f"{'='*50}")
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
