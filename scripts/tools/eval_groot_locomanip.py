# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause
"""Closed-loop GR00T evaluation for locomanipulation tasks.

Uses the Isaac-PickPlace-Locomanipulation-Camera-G1-Abs-v0 environment with:
- Agile RL locomotion policy embedded as ActionTerm (runs inside env.step())
- JointPositionActionCfg for upper body (replaces Pink IK)
- GR00T model via PolicyClient for ego_view camera + joint state → actions
- Optional: Cosmos VLM subtask monitor for dynamic task description switching

Usage:
    # Terminal 1: Start GR00T server
    cd /home/mats/Bot/Nvidia/Isaac-GR00T
    conda activate gr00t
    uv run python gr00t/eval/run_gr00t_server.py \
        --model-path /home/mats/Bot/Models/g1_locomanip_finetune_3/checkpoint-4000 \
        --embodiment-tag UNITREE_G1 --port 5555

    # Terminal 2 (optional): Start Cosmos VLM subtask monitor
    conda activate cosmos
    python scripts/tools/run_cosmos_vlm_server.py \
        --model_path ~/Bot/Nvidia/Cosmos-Reason2-2B \
        --task "grab wheel, walk right, place in basket, walk left" \
        --subtasks "grab the wheel" "walk right to the basket" \
                   "place the wheel in the basket" "walk left" \
        --port 5556

    # Terminal 3: Run this eval script
    ./isaaclab.sh -p scripts/tools/eval_groot_locomanip.py \
        --task Isaac-PickPlace-Locomanipulation-Camera-G1-Abs-v0 \
        --policy_port 5555 --action_horizon 8 \
        --num_envs 1 --enable_cameras

    # With dynamic subtask switching:
    ./isaaclab.sh -p scripts/tools/eval_groot_locomanip.py \
        --task Isaac-PickPlace-Locomanipulation-Camera-G1-Abs-v0 \
        --policy_port 5555 --action_horizon 8 \
        --subtasks "grab the wheel" "walk right to the basket" \
                   "place the wheel in the basket" "walk left" \
        --cosmos_port 5556
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
    default="pick up the wheel, walk right, and place it in the basket",
    help="Language instruction for the task.",
)
parser.add_argument("--action_horizon", type=int, default=8, help="Actions to execute per query.")
parser.add_argument("--num_episodes", type=int, default=10, help="Number of episodes to evaluate.")
parser.add_argument("--save_video", action="store_true", default=False, help="Save video.")

# Cosmos VLM subtask monitor (optional — enables dynamic task description switching)
parser.add_argument(
    "--subtasks", nargs="+", default=None,
    help="Ordered subtask labels. Enables hierarchical mode with Cosmos VLM monitor.",
)
parser.add_argument(
    "--transitions", nargs="+", default=None,
    help="Yes/no questions for each subtask transition. One per subtask. "
         "When the VLM answers 'yes', advance to next subtask. "
         "E.g.: 'Is the robot holding the wheel?' 'Is the basket visible?' 'Is the wheel in the basket?'",
)
parser.add_argument("--cosmos_host", type=str, default="localhost", help="Cosmos VLM server host.")
parser.add_argument("--cosmos_port", type=int, default=5556, help="Cosmos VLM server port.")
parser.add_argument(
    "--cosmos_interval", type=int, default=25,
    help="Push a frame to Cosmos every N env steps (25 = 0.5s at 50Hz).",
)
parser.add_argument(
    "--debounce", type=int, default=3,
    help="Consecutive VLM votes needed before switching subtask (default 3).",
)

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# Add Isaac-GR00T to path
sys.path.insert(0, "/home/mats/Bot/Nvidia/Isaac-GR00T")

# Launch simulator
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import contextlib
import io
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


# ---------------------------------------------------------------------------
# Cosmos VLM subtask monitor client
# ---------------------------------------------------------------------------

class SubtaskMonitorClient:
    """Thin ZMQ client for the Cosmos VLM subtask monitor server.

    Sends ego_view frames and receives subtask classifications.
    VLM is the authority — subtask can go forward or backward.
    Debouncing prevents flickering between subtasks.
    """

    def __init__(self, host: str, port: int, subtasks: list[str],
                 timeout_ms: int = 5000, debounce: int = 3):
        import msgpack as _msgpack
        import zmq as _zmq
        self._msgpack = _msgpack
        self._zmq = _zmq

        self.subtasks = subtasks
        self.current_index = 0
        self.transition_log = []  # [(step, from_idx, to_idx)]

        # Debounce: require N consecutive votes for a different subtask before switching
        self._debounce = debounce
        self._candidate_idx = None
        self._candidate_count = 0

        self._context = _zmq.Context()
        self._socket = self._context.socket(_zmq.REQ)
        self._socket.setsockopt(_zmq.RCVTIMEO, timeout_ms)
        self._socket.setsockopt(_zmq.SNDTIMEO, timeout_ms)
        self._socket.connect(f"tcp://{host}:{port}")

    def _send(self, request: dict) -> dict:
        self._socket.send(self._msgpack.packb(request, use_bin_type=True))
        return self._msgpack.unpackb(self._socket.recv(), raw=False)

    def ping(self) -> bool:
        try:
            resp = self._send({"endpoint": "ping"})
            return resp.get("status") == "ok"
        except Exception:
            return False

    def _rgb_to_jpeg(self, rgb: np.ndarray) -> bytes:
        from PIL import Image
        img = Image.fromarray(rgb)
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=85)
        return buf.getvalue()

    @staticmethod
    def _strip_think(text: str) -> str:
        import re
        return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()

    def push_frame(self, rgb: np.ndarray, step: int) -> str:
        """Send a frame to Cosmos, get back the current subtask."""
        jpeg = self._rgb_to_jpeg(rgb)
        try:
            resp = self._send({
                "endpoint": "push_frame",
                "data": {"jpeg": jpeg},
            })
        except Exception as e:
            print(f"  [Cosmos] Push failed: {e}")
            return self.subtasks[self.current_index]

        raw_response = self._strip_think(resp.get("response", ""))
        if not raw_response:
            return self.subtasks[self.current_index]

        matched_idx = self._match_subtask(raw_response)
        if matched_idx is None or matched_idx == self.current_index:
            self._candidate_idx = None
            self._candidate_count = 0
            return self.subtasks[self.current_index]

        # Debounce
        if matched_idx == self._candidate_idx:
            self._candidate_count += 1
        else:
            self._candidate_idx = matched_idx
            self._candidate_count = 1

        if self._candidate_count >= self._debounce:
            old_idx = self.current_index
            self.current_index = matched_idx
            self.transition_log.append((step, old_idx, matched_idx))
            direction = "→" if matched_idx > old_idx else "←"
            print(f"  [Subtask] step {step}: '{self.subtasks[old_idx]}' {direction} "
                  f"'{self.subtasks[matched_idx]}' ({self._debounce}x confirmed)")
            self._candidate_idx = None
            self._candidate_count = 0
        else:
            pass

        return self.subtasks[self.current_index]

    def _match_subtask(self, response: str) -> int | None:
        import re
        response_lower = response.strip().lower()

        for i, st in enumerate(self.subtasks):
            if st.lower() == response_lower:
                return i

        num_match = re.match(r"^(\d+)", response_lower)
        if num_match:
            idx = int(num_match.group(1)) - 1
            if 0 <= idx < len(self.subtasks):
                return idx

        for i, st in enumerate(self.subtasks):
            if st.lower() in response_lower or response_lower in st.lower():
                return i

        stop_words = {"the", "a", "an", "to", "in", "is", "of", "and", "with"}
        resp_words = set(response_lower.split()) - stop_words
        best_idx, best_score = None, 0
        for i, st in enumerate(self.subtasks):
            st_words = set(st.lower().split()) - stop_words
            overlap = len(resp_words & st_words)
            if overlap > best_score:
                best_score = overlap
                best_idx = i

        if best_score >= 2:
            return best_idx

        print(f"  [Cosmos] Could not match: '{response[:80]}'")
        return None

    def reset(self):
        self.current_index = 0
        self.transition_log.clear()
        self._candidate_idx = None
        self._candidate_count = 0

    def close(self):
        self._socket.close()
        self._context.term()


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

        # Ego-view camera
        cam_ego = env.scene["ego_view_camera"].data.output["rgb"][env_id].cpu().numpy()

        if cam_ego.shape[-1] == 4:
            cam_ego = cam_ego[..., :3]
        if cam_ego.dtype != np.uint8:
            cam_ego = (cam_ego * 255).clip(0, 255).astype(np.uint8)

        video_dict = {
            "ego_view": cam_ego[np.newaxis, np.newaxis, ...],   # (1, 1, H, W, 3)
        }

        language_dict = {
            "annotation.human.task_description": [[self.task_description]],
        }

        # Debug: first frame
        if not self._saved_debug:
            self._saved_debug = True
            cv2.imwrite("/tmp/locomanip_cam_ego.png", cv2.cvtColor(cam_ego, cv2.COLOR_RGB2BGR))
            print(f"[Adapter] Debug image → /tmp/locomanip_cam_ego.png")
            print(f"[Adapter] cam_ego: {cam_ego.shape}")
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

    # -- Optional: Cosmos VLM subtask monitor --
    cosmos_monitor = None
    if args_cli.subtasks:
        print(f"Connecting to Cosmos VLM at {args_cli.cosmos_host}:{args_cli.cosmos_port}")
        cosmos_monitor = SubtaskMonitorClient(
            host=args_cli.cosmos_host,
            port=args_cli.cosmos_port,
            subtasks=args_cli.subtasks,
            debounce=args_cli.debounce,
        )
        if not cosmos_monitor.ping():
            raise RuntimeError("Failed to connect to Cosmos VLM server.")
        print(f"Connected to Cosmos VLM — subtasks: {args_cli.subtasks}")
        initial_task = args_cli.subtasks[0]
    else:
        initial_task = args_cli.task_description

    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
    make_groot_eval_compatible(env_cfg)

    env = gym.make(args_cli.task, cfg=env_cfg).unwrapped
    adapter = LocomanipGR00TAdapter(policy_client, initial_task, env)

    device = env.device
    dtype = env.scene["robot"].data.joint_pos.dtype

    num_episodes = args_cli.num_episodes
    results = []

    print(f"\nStarting GR00T locomanip evaluation")
    print(f"  Task: {initial_task}")
    if cosmos_monitor:
        print(f"  Mode: hierarchical (Cosmos VLM subtask monitor)")
        print(f"  Cosmos interval: every {args_cli.cosmos_interval} steps ({args_cli.cosmos_interval * 0.02:.1f}s)")
    else:
        print(f"  Mode: static task description")
    print(f"  Action horizon: {args_cli.action_horizon}")
    print(f"  Episodes: {num_episodes}")

    episode_idx = 0
    with contextlib.suppress(KeyboardInterrupt), torch.inference_mode():
        while episode_idx < num_episodes and simulation_app.is_running() and not simulation_app.is_exiting():
            obs_dict, _ = env.reset()
            action_queue = []
            step_count = 0
            video_frames = [] if args_cli.save_video else None

            # Reset subtask monitor for new episode
            if cosmos_monitor:
                cosmos_monitor.reset()
                adapter.task_description = args_cli.subtasks[0]

            print(f"\n--- Episode {episode_idx + 1}/{num_episodes} ---")
            if cosmos_monitor:
                print(f"  [Subtask] starting: '{adapter.task_description}'")

            while simulation_app.is_running() and not simulation_app.is_exiting():
                # -- Cosmos subtask update --
                if cosmos_monitor and step_count > 0 and step_count % args_cli.cosmos_interval == 0:
                    cam_rgb = env.scene["ego_view_camera"].data.output["rgb"][0].cpu().numpy()
                    if cam_rgb.shape[-1] == 4:
                        cam_rgb = cam_rgb[..., :3]
                    if cam_rgb.dtype != np.uint8:
                        cam_rgb = (cam_rgb * 255).clip(0, 255).astype(np.uint8)
                    new_subtask = cosmos_monitor.push_frame(cam_rgb, step_count)
                    adapter.task_description = new_subtask

                if len(action_queue) == 0:
                    t0 = time.perf_counter()
                    groot_obs = adapter.obs_to_groot(env)
                    action_chunk, info = adapter.policy.get_action(groot_obs)
                    latency_ms = (time.perf_counter() - t0) * 1000

                    horizon = np.asarray(action_chunk["left_arm"]).shape[1]
                    for t in range(min(args_cli.action_horizon, horizon)):
                        action_queue.append(adapter.decode_action(action_chunk, t))

                    if step_count == 0:
                        print(f"  First chunk: horizon={horizon}, latency={latency_ms:.0f}ms")
                    elif step_count % 100 == 0:
                        subtask_info = f", subtask='{adapter.task_description}'" if cosmos_monitor else ""
                        print(f"  Step {step_count}, latency={latency_ms:.0f}ms{subtask_info}")

                action_dict = action_queue.pop(0)
                action_tensor = adapter.build_action_tensor(action_dict, device, dtype)

                obs_dict, _, terminated, truncated, infos = env.step(action_tensor)

                if video_frames is not None:
                    cam = env.scene["ego_view_camera"].data.output["rgb"][0].cpu().numpy()
                    if cam.shape[-1] == 4:
                        cam = cam[..., :3]
                    if cam.dtype != np.uint8:
                        cam = (cam * 255).clip(0, 255).astype(np.uint8)
                    video_frames.append(cam)

                step_count += 1

                if terminated.any() or truncated.any():
                    fired = {}
                    tm = env.termination_manager
                    for i, name in enumerate(tm._term_names):
                        if tm._term_dones[0, i]:
                            fired[name] = True

                    outcome = ", ".join(fired.keys()) if fired else ("truncated" if truncated.any() else "terminated")
                    ep_result = {"outcome": fired, "steps": step_count}
                    if cosmos_monitor:
                        ep_result["subtask_transitions"] = list(cosmos_monitor.transition_log)
                        ep_result["final_subtask"] = cosmos_monitor.current_index
                    results.append(ep_result)
                    print(f"  >> {outcome} at step {step_count} ({step_count * 0.02:.1f}s)")
                    if cosmos_monitor:
                        print(f"  >> Final subtask: '{args_cli.subtasks[cosmos_monitor.current_index]}' "
                              f"({len(cosmos_monitor.transition_log)} transitions)")
                    break

            # Save per-episode video
            if video_frames and len(video_frames) > 0:
                video_path = f"/tmp/groot_locomanip_eval_ep{episode_idx:02d}.mp4"
                h, w = video_frames[0].shape[:2]
                writer = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*"mp4v"), 50.0, (w, h))
                for frame in video_frames:
                    writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
                writer.release()
                print(f"  Saved {len(video_frames)} frames → {video_path}")

            episode_idx += 1

    # Summary
    total = len(results)
    print(f"\n{'='*50}")
    print(f"  EVALUATION RESULTS ({total} episodes)")
    if cosmos_monitor:
        print(f"  Mode: hierarchical ({len(args_cli.subtasks)} subtasks)")
    print(f"{'='*50}")

    term_counts = {}
    for r in results:
        for name in r["outcome"]:
            term_counts[name] = term_counts.get(name, 0) + 1

    for name, count in sorted(term_counts.items()):
        print(f"  {name:20s}: {count}/{total} ({100*count/total:.0f}%)")

    print(f"{'='*50}")
    for i, r in enumerate(results):
        terms = ", ".join(r["outcome"].keys())
        subtask_str = ""
        if "final_subtask" in r:
            subtask_str = f"  subtask={r['final_subtask']}/{len(args_cli.subtasks)-1}"
        print(f"  Ep {i+1:2d}: {terms:20s} @ {r['steps']:4d} steps ({r['steps']*0.02:.1f}s){subtask_str}")

    if cosmos_monitor:
        cosmos_monitor.close()

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
