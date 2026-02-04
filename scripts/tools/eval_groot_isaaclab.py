# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""Script to evaluate GR00T model with Isaac Lab environments.

This script runs GR00T inference in the exact same visual environment
used for training data collection, ensuring visual domain consistency.

Usage:
    # Terminal 1: Start GR00T server
    cd /home/mats/Bot/Nvidia/Isaac-GR00T
    uv run python gr00t/eval/run_gr00t_server.py \
        --model-path /path/to/checkpoint \
        --embodiment-tag G1_INSPIRE

    # Terminal 2: Run this eval script
    cd /home/mats/Bot/Nvidia/IsaacLab
    ./isaaclab.sh -p scripts/tools/eval_groot_isaaclab.py \
        --task Isaac-Pick-Place-Target-G1-Inspire-v0 \
        --dataset_file ./datasets/your_training_demos.hdf5 \
        --policy_host localhost --policy_port 5555 \
        --num_envs 1 --enable_pinocchio --enable_cameras
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Evaluate GR00T model in Isaac Lab environments.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments.")
parser.add_argument("--task", type=str, default="Isaac-Pick-Place-Target-G1-Inspire-v0", help="Task name.")
parser.add_argument(
    "--select_episodes",
    type=int,
    nargs="+",
    default=[],
    help="A list of episode indices to use for initial states. Empty = use first episode.",
)
parser.add_argument("--dataset_file", type=str, default=None, help="Dataset file for initial states (optional).")
parser.add_argument("--policy_host", type=str, default="localhost", help="GR00T policy server host.")
parser.add_argument("--policy_port", type=int, default=5555, help="GR00T policy server port.")
parser.add_argument(
    "--task_description",
    type=str,
    default="Pick red cube and place in yellow zone",
    help="Task description for language conditioning.",
)
parser.add_argument("--action_horizon", type=int, default=1, help="Number of actions to execute per inference.")
parser.add_argument("--max_steps", type=int, default=1000, help="Maximum steps per episode.")
parser.add_argument(
    "--enable_pinocchio",
    action="store_true",
    default=False,
    help="Enable Pinocchio.",
)
parser.add_argument(
    "--validate_success",
    action="store_true",
    default=False,
    help="Track success rate using environment termination criteria.",
)
parser.add_argument("--save_video", action="store_true", default=False, help="Save video of evaluation.")

# append AppLauncher cli args (includes --enable_cameras)
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

if args_cli.enable_pinocchio:
    import pinocchio  # noqa: F401

# Add Isaac-GR00T to path
sys.path.insert(0, "/home/mats/Bot/Nvidia/Isaac-GR00T")

# launch the simulator
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import contextlib
import os
import time

import cv2
import gymnasium as gym
import numpy as np
import torch

from isaaclab.devices import Se3Keyboard, Se3KeyboardCfg
from isaaclab.utils.datasets import EpisodeData, HDF5DatasetFileHandler

if args_cli.enable_pinocchio:
    import isaaclab_tasks.manager_based.manipulation.pick_place  # noqa: F401

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg

# Import PolicyClient
from gr00t.policy.server_client import PolicyClient

is_paused = False


def play_cb():
    global is_paused
    is_paused = False


def pause_cb():
    global is_paused
    is_paused = True


class G1InspireGR00TAdapter:
    """Adapter between IsaacLab observations and GR00T VLA format.

    Handles joint index mapping between IsaacLab's full 57-joint G1 robot
    and GR00T's 26-DOF format (14 arm + 12 hand joints).
    """

    # Joint name patterns for GR00T format (must match training data order)
    # Training data order from convert_unitree_to_lerobot_v3.py FEATURE_NAMES_26DOF
    ARM_JOINT_NAMES = [
        # Left arm (7 DOF): shoulder_pitch, shoulder_roll, shoulder_yaw, elbow, wrist_roll, wrist_pitch, wrist_yaw
        "left_shoulder_pitch_joint",
        "left_shoulder_roll_joint",
        "left_shoulder_yaw_joint",
        "left_elbow_joint",
        "left_wrist_roll_joint",   # Testing: maybe IsaacLab uses roll, pitch, yaw order?
        "left_wrist_pitch_joint",
        "left_wrist_yaw_joint",
        # Right arm (7 DOF): same pattern
        "right_shoulder_pitch_joint",
        "right_shoulder_roll_joint",
        "right_shoulder_yaw_joint",
        "right_elbow_joint",
        "right_wrist_roll_joint",
        "right_wrist_pitch_joint",
        "right_wrist_yaw_joint",
    ]

    HAND_JOINT_NAMES = [
        # Left hand (6 DOF): pinky, ring, middle, index, thumb_pitch, thumb_yaw
        "L_pinky_proximal_joint",
        "L_ring_proximal_joint",
        "L_middle_proximal_joint",
        "L_index_proximal_joint",
        "L_thumb_proximal_pitch_joint",
        "L_thumb_proximal_yaw_joint",
        # Right hand (6 DOF): same pattern
        "R_pinky_proximal_joint",
        "R_ring_proximal_joint",
        "R_middle_proximal_joint",
        "R_index_proximal_joint",
        "R_thumb_proximal_pitch_joint",
        "R_thumb_proximal_yaw_joint",
    ]

    def __init__(self, policy_client: PolicyClient, task_description: str, robot=None):
        self.policy = policy_client
        self.task_description = task_description
        self._saved_debug_img = False

        # Joint index mapping (populated when robot is set)
        self.arm_joint_indices = None
        self.hand_joint_indices = None
        self.all_joint_indices = None

        if robot is not None:
            self.setup_joint_mapping(robot)

    def setup_joint_mapping(self, robot):
        """Setup joint index mapping from robot's joint names.

        Args:
            robot: IsaacLab articulation asset
        """
        # Get joint names from robot
        joint_names = robot.data.joint_names

        print(f"[DEBUG] Robot has {len(joint_names)} joints")
        print(f"[DEBUG] First 30 joint names: {joint_names[:30]}")

        # Find indices for arm joints
        self.arm_joint_indices = []
        for name in self.ARM_JOINT_NAMES:
            if name in joint_names:
                self.arm_joint_indices.append(joint_names.index(name))
            else:
                print(f"[WARNING] Arm joint not found: {name}")
                self.arm_joint_indices.append(-1)

        # Find indices for hand joints
        self.hand_joint_indices = []
        for name in self.HAND_JOINT_NAMES:
            if name in joint_names:
                self.hand_joint_indices.append(joint_names.index(name))
            else:
                print(f"[WARNING] Hand joint not found: {name}")
                self.hand_joint_indices.append(-1)

        self.all_joint_indices = self.arm_joint_indices + self.hand_joint_indices

        print(f"[DEBUG] Arm joint indices: {self.arm_joint_indices}")
        print(f"[DEBUG] Hand joint indices: {self.hand_joint_indices}")

    def _add_batch_time_dims(self, obs: dict) -> dict:
        """Add (B=1, T=1) dimensions to match GR00T server expectations."""
        result = {}
        for key, val in obs.items():
            if isinstance(val, np.ndarray):
                result[key] = val[np.newaxis, np.newaxis, ...]
            elif isinstance(val, dict):
                result[key] = self._add_batch_time_dims(val)
            else:
                result[key] = [[val]]
        return result

    def extract_26dof_state(self, full_joint_pos: np.ndarray) -> np.ndarray:
        """Extract 26 DOF state from full robot joint positions.

        Args:
            full_joint_pos: Full robot joint positions (57 DOF for G1)

        Returns:
            26 DOF state: [left_arm(7), right_arm(7), left_hand(6), right_hand(6)]
        """
        if self.all_joint_indices is None:
            # Fallback: assume first 26 joints (may not be correct)
            print("[WARNING] Joint mapping not set up, using first 26 joints")
            return full_joint_pos[:26]

        state_26dof = np.array([full_joint_pos[i] if i >= 0 else 0.0 for i in self.all_joint_indices])
        return state_26dof.astype(np.float32)

    def obs_to_policy_inputs(self, obs_dict: dict, env_id: int = 0) -> dict:
        """Convert IsaacLab observation dict into GR00T VLA input format.

        Args:
            obs_dict: Observation dict from env.observation_manager.compute()
            env_id: Environment index (for batched envs)

        Returns:
            Dict formatted for GR00T PolicyClient
        """
        model_obs = {}

        # (1) Video: cam_left and cam_right
        # IsaacLab images are (B, H, W, C) uint8 when normalize=False
        cam_left = obs_dict["policy"]["head_rgb_left"][env_id].cpu().numpy()
        cam_right = obs_dict["policy"]["head_rgb_right"][env_id].cpu().numpy()

        # Ensure uint8 and (H, W, C) format
        if cam_left.dtype != np.uint8:
            cam_left = (cam_left * 255).astype(np.uint8)
        if cam_right.dtype != np.uint8:
            cam_right = (cam_right * 255).astype(np.uint8)

        # Debug: Save first frame
        if not self._saved_debug_img:
            self._saved_debug_img = True
            cv2.imwrite("/tmp/isaaclab_cam_left.png", cv2.cvtColor(cam_left, cv2.COLOR_RGB2BGR))
            cv2.imwrite("/tmp/isaaclab_cam_right.png", cv2.cvtColor(cam_right, cv2.COLOR_RGB2BGR))
            print(f"[DEBUG] Saved IsaacLab cameras to /tmp/isaaclab_cam_*.png")
            print(f"[DEBUG] cam_left shape={cam_left.shape}, dtype={cam_left.dtype}")

        model_obs["video"] = {
            "cam_left": cam_left,
            "cam_right": cam_right,
        }

        # (2) State: Extract 26 DOF from full robot state
        full_joint_pos = obs_dict["policy"]["robot_joint_pos"][env_id].cpu().numpy()
        state_26dof = self.extract_26dof_state(full_joint_pos)

        # Split into arm + hand groups
        left_arm = state_26dof[:7]
        right_arm = state_26dof[7:14]
        left_hand = state_26dof[14:20]
        right_hand = state_26dof[20:26]

        model_obs["state"] = {
            "left_arm": left_arm,
            "right_arm": right_arm,
            "left_hand": left_hand,
            "right_hand": right_hand,
        }

        # (3) Language instruction
        model_obs["language"] = {
            "annotation.human.task_description": self.task_description,
        }

        # (4) Add batch and time dimensions
        model_obs = self._add_batch_time_dims(model_obs)

        return model_obs

    def decode_action_chunk(self, action_chunk: dict, timestep: int = 0) -> np.ndarray:
        """Decode action chunk at given timestep into flat 26 DOF action."""
        left_arm = np.array(action_chunk["left_arm"])[0, timestep]  # (7,)
        right_arm = np.array(action_chunk["right_arm"])[0, timestep]  # (7,)
        left_hand = np.array(action_chunk["left_hand"])[0, timestep]  # (6,)
        right_hand = np.array(action_chunk["right_hand"])[0, timestep]  # (6,)
        return np.concatenate([left_arm, right_arm, left_hand, right_hand], axis=0)

    def apply_action_to_robot(self, robot, action_26dof: np.ndarray, env_id: int = 0):
        """Apply 26 DOF action to robot's joint position targets.

        Args:
            robot: IsaacLab articulation asset
            action_26dof: 26 DOF action [left_arm(7), right_arm(7), left_hand(6), right_hand(6)]
            env_id: Environment index
        """
        if self.all_joint_indices is None:
            raise RuntimeError("Joint mapping not set up. Call setup_joint_mapping first.")

        # Get current joint position targets
        current_targets = robot.data.joint_pos[env_id].clone()

        # Convert action to torch tensor on same device
        action_tensor = torch.tensor(action_26dof, device=current_targets.device, dtype=current_targets.dtype)

        # Apply 26 DOF action to the relevant joints
        for i, joint_idx in enumerate(self.all_joint_indices):
            if joint_idx >= 0:
                current_targets[joint_idx] = action_tensor[i]

        # Set joint position targets
        robot.set_joint_position_target(current_targets.unsqueeze(0), env_ids=torch.tensor([env_id], device=current_targets.device))

    def get_action_chunk(self, obs_dict: dict, env_id: int = 0) -> tuple:
        """Get full action chunk from policy server."""
        model_input = self.obs_to_policy_inputs(obs_dict, env_id)
        action_chunk, info = self.policy.get_action(model_input)
        return action_chunk, info

    def get_action(self, obs_dict: dict, env_id: int = 0, timestep: int = 0) -> np.ndarray:
        """Get single action from policy server."""
        action_chunk, info = self.get_action_chunk(obs_dict, env_id)
        return self.decode_action_chunk(action_chunk, timestep)


def joint_pos_to_pink_action(joint_pos: np.ndarray, env) -> torch.Tensor:
    """Convert joint position targets to Pink IK action format.

    Pink IK action format for G1 Inspire:
    - Left wrist pose (3 pos + 4 quat = 7)
    - Right wrist pose (3 pos + 4 quat = 7)
    - Left hand joints (12)
    - Right hand joints (12)
    Total: 38

    For now, we use a workaround: directly set joint positions via
    the scene robot, bypassing the action space.
    """
    # This is a placeholder - the actual implementation would need
    # forward kinematics to compute wrist poses from joint positions.
    # For evaluation, we'll use direct joint control instead.
    raise NotImplementedError("Use direct joint control instead")


def main():
    """Evaluate GR00T model in IsaacLab environment."""
    global is_paused

    # Connect to GR00T policy server
    print(f"Connecting to GR00T Policy Server at {args_cli.policy_host}:{args_cli.policy_port}")
    policy_client = PolicyClient(host=args_cli.policy_host, port=args_cli.policy_port, strict=False)

    if policy_client.ping():
        print("Successfully connected to GR00T Policy Server")
    else:
        raise RuntimeError("Failed to connect to GR00T Policy Server")

    # Load dataset for initial states (optional)
    dataset_file_handler = None
    episode_count = 0
    if args_cli.dataset_file and os.path.exists(args_cli.dataset_file):
        print(f"Loading dataset from {args_cli.dataset_file}")
        dataset_file_handler = HDF5DatasetFileHandler()
        dataset_file_handler.open(args_cli.dataset_file)
        episode_count = dataset_file_handler.get_num_episodes()
        print(f"Found {episode_count} episodes in dataset")

    # Parse environment config
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)

    # Extract success checking function
    success_term = None
    if args_cli.validate_success:
        if hasattr(env_cfg.terminations, "success"):
            success_term = env_cfg.terminations.success
            env_cfg.terminations.success = None
        else:
            print("No success termination term found in environment.")

    # Disable recorders and terminations for evaluation
    env_cfg.recorders = {}
    env_cfg.terminations = {}

    # Create environment
    env = gym.make(args_cli.task, cfg=env_cfg).unwrapped

    # Get robot and setup adapter with joint mapping
    robot = env.scene["robot"]
    adapter = G1InspireGR00TAdapter(policy_client, task_description=args_cli.task_description, robot=robot)

    # Keyboard controls
    teleop_interface = Se3Keyboard(Se3KeyboardCfg(pos_sensitivity=0.1, rot_sensitivity=0.1))
    teleop_interface.add_callback("N", play_cb)
    teleop_interface.add_callback("B", pause_cb)
    print('Press "B" to pause and "N" to resume.')

    # Get idle action
    if hasattr(env_cfg, "idle_action"):
        idle_action = env_cfg.idle_action.repeat(args_cli.num_envs, 1)
    else:
        idle_action = torch.zeros(env.action_space.shape)

    # Reset environment
    env.reset()
    teleop_interface.reset()

    # If using dataset, reset to initial state from episode
    if dataset_file_handler and episode_count > 0:
        episode_names = list(dataset_file_handler.get_episode_names())
        episode_idx = args_cli.select_episodes[0] if args_cli.select_episodes else 0
        episode_data = dataset_file_handler.load_episode(episode_names[episode_idx], env.device)
        initial_state = episode_data.get_initial_state()
        env.reset_to(initial_state, torch.tensor([0], device=env.device), is_relative=True)
        print(f"Reset to initial state from episode {episode_idx}")

    # Video recording
    video_frames = [] if args_cli.save_video else None

    # Main evaluation loop
    step_count = 0
    success_count = 0
    total_episodes = 1
    action_queue = []

    print(f"\nStarting GR00T evaluation (max {args_cli.max_steps} steps per episode)")
    print(f"Task: {args_cli.task_description}")

    with contextlib.suppress(KeyboardInterrupt) and torch.inference_mode():
        while simulation_app.is_running() and not simulation_app.is_exiting():
            # Pause handling
            while is_paused:
                env.sim.render()
                continue

            # Get observations
            obs_dict = env.observation_manager.compute()

            # Debug: print observation keys and joint info on first step
            if step_count == 0:
                print(f"\n[DEBUG] Observation keys: {obs_dict['policy'].keys()}")
                joint_pos = obs_dict["policy"]["robot_joint_pos"][0].cpu().numpy()
                print(f"[DEBUG] Full joint pos shape: {joint_pos.shape}")

                # Extract and show 26 DOF state
                state_26dof = adapter.extract_26dof_state(joint_pos)
                print(f"[DEBUG] Extracted 26 DOF state:")
                print(f"  Left arm (7):  {state_26dof[:7]}")
                print(f"  Right arm (7): {state_26dof[7:14]}")
                print(f"  Left hand (6): {state_26dof[14:20]}")
                print(f"  Right hand (6): {state_26dof[20:26]}")

            # Get action from GR00T
            if len(action_queue) == 0:
                # Query server for new action chunk
                action_chunk, info = adapter.get_action_chunk(obs_dict, env_id=0)

                # Fill queue with actions from chunk
                horizon = np.array(action_chunk["left_arm"]).shape[1]
                for t in range(min(args_cli.action_horizon, horizon)):
                    action_queue.append(adapter.decode_action_chunk(action_chunk, t))

                # Debug: Print action on first query
                if step_count == 0:
                    print(f"[DEBUG] Action chunk horizon: {horizon}")
                    action_0 = adapter.decode_action_chunk(action_chunk, 0)
                    print(f"[DEBUG] Action (t=0):")
                    print(f"  Left arm:  {action_0[:7]}")
                    print(f"  Right arm: {action_0[7:14]}")
                    print(f"  Left hand: {action_0[14:20]}")
                    print(f"  Right hand: {action_0[20:26]}")

            # Get next action from queue
            action_np = action_queue.pop(0)

            # Apply action to robot using proper joint mapping
            adapter.apply_action_to_robot(robot, action_np, env_id=0)

            # Write joint targets to simulation
            robot.write_data_to_sim()

            # Step physics
            env.sim.step(render=True)

            # Update scene state (reads back from simulation)
            env.scene.update(env.sim.get_physics_dt())

            # Step interval events (e.g., reveal_target_on_lift)
            if hasattr(env, 'event_manager') and "interval" in env.event_manager.available_modes:
                env.event_manager.apply(mode="interval", dt=env.step_dt)

            # Save video frame
            if video_frames is not None:
                cam_left = obs_dict["policy"]["head_rgb_left"][0].cpu().numpy()
                if cam_left.dtype != np.uint8:
                    cam_left = (cam_left * 255).astype(np.uint8)
                video_frames.append(cam_left)

            # Check success
            if success_term is not None:
                is_success = bool(success_term.func(env, **success_term.params)[0])
                if is_success:
                    success_count += 1
                    print(f"\nSUCCESS at step {step_count}!")
                    break

            step_count += 1

            # Print progress
            if step_count % 100 == 0:
                print(f"Step {step_count}/{args_cli.max_steps}")

            # Check max steps
            if step_count >= args_cli.max_steps:
                print(f"\nReached max steps ({args_cli.max_steps})")
                break

    # Save video
    if video_frames and len(video_frames) > 0:
        video_path = "/tmp/groot_isaaclab_eval.mp4"
        print(f"\nSaving video to {video_path}")
        h, w = video_frames[0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(video_path, fourcc, 20.0, (w, h))
        for frame in video_frames:
            writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        writer.release()
        print(f"Saved {len(video_frames)} frames")

    # Print results
    print(f"\n{'=' * 50}")
    print(f"Evaluation Results:")
    print(f"  Total steps: {step_count}")
    if args_cli.validate_success:
        print(f"  Success: {success_count}/{total_episodes}")
    print(f"{'=' * 50}")

    # Cleanup
    if dataset_file_handler:
        dataset_file_handler.close()
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
