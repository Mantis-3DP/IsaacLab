# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Streaming recorder that writes data to disk in real-time.

This recorder bypasses the normal EpisodeData accumulation and HDF5 batch writing,
instead streaming images as JPEG and state/action data as incremental JSON.

Output format is compatible with Unitree's unitree_IL_lerobot converter.

Benefits:
- No memory buildup during recording
- Near-instant episode saves (just close file handles)
- 10-20x smaller storage via JPEG compression
- Optional rerun.io live visualization
- Direct compatibility with unitree_IL_lerobot converter
"""

from __future__ import annotations

import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

from isaaclab.managers.recorder_manager import RecorderTerm
from isaaclab.managers.manager_term_cfg import RecorderTermCfg
from isaaclab.utils import configclass
from isaaclab.utils.datasets import StreamingEpisodeWriter

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


# G1 Inspire joint indices in IsaacLab (from robot_joint_pos)
# Order matches Pink IK processed_actions for consistency between state and action
G1_INSPIRE_JOINT_MAPPING = {
    # Left arm (7 DOF) - indices in robot_joint_pos
    "left_arm": {
        "joint_names": [
            "left_shoulder_pitch_joint",
            "left_shoulder_roll_joint",
            "left_shoulder_yaw_joint",
            "left_elbow_joint",
            "left_wrist_roll_joint",
            "left_wrist_pitch_joint",
            "left_wrist_yaw_joint",
        ],
        "indices": None,
    },
    # Right arm (7 DOF)
    "right_arm": {
        "joint_names": [
            "right_shoulder_pitch_joint",
            "right_shoulder_roll_joint",
            "right_shoulder_yaw_joint",
            "right_elbow_joint",
            "right_wrist_roll_joint",
            "right_wrist_pitch_joint",
            "right_wrist_yaw_joint",
        ],
        "indices": None,
    },
    # Left hand (6 DOF) - order matches Unitree/LeRobot format
    # [pinky, ring, middle, index, thumb_pitch, thumb_yaw]
    "left_ee": {
        "joint_names": [
            "L_pinky_proximal_joint",
            "L_ring_proximal_joint",
            "L_middle_proximal_joint",
            "L_index_proximal_joint",
            "L_thumb_proximal_pitch_joint",
            "L_thumb_proximal_yaw_joint",
        ],
        "indices": None,
    },
    # Right hand (6 DOF) - order matches Unitree/LeRobot format
    "right_ee": {
        "joint_names": [
            "R_pinky_proximal_joint",
            "R_ring_proximal_joint",
            "R_middle_proximal_joint",
            "R_index_proximal_joint",
            "R_thumb_proximal_pitch_joint",
            "R_thumb_proximal_yaw_joint",
        ],
        "indices": None,
    },
}

# Camera name mapping: IsaacLab name -> Unitree format
CAMERA_MAPPING = {
    "head_rgb_left": "color_0",
    "head_rgb_right": "color_1",
    "cam_left_high": "color_0",
    "cam_right_high": "color_1",
    "cam_left_wrist": "color_2",
    "cam_right_wrist": "color_3",
}

# Fixed action indices for Pink IK controller's processed_actions
# The Pink IK controller outputs: 14 arm joints + 24 hand joints = 38 total
# Structure: [left_arm(7), right_arm(7), hand_joints(24)]
#
# Hand joint order in processed_actions (from env config hand_joint_names):
#   14-17: L_index, L_middle, L_pinky, L_ring (proximal)
#   18: L_thumb_yaw, 19-22: R_index, R_middle, R_pinky, R_ring, 23: R_thumb_yaw
#   28: L_thumb_pitch, 33: R_thumb_pitch
#
# Output order matches Unitree/LeRobot format: [pinky, ring, middle, index, thumb_pitch, thumb_yaw]
PINK_IK_ACTION_INDICES = {
    "left_arm": list(range(0, 7)),       # indices 0-6
    "right_arm": list(range(7, 14)),     # indices 7-13
    # Left hand reordered: pinky(16), ring(17), middle(15), index(14), thumb_pitch(28), thumb_yaw(18)
    "left_ee": [16, 17, 15, 14, 28, 18],
    # Right hand reordered: pinky(21), ring(22), middle(20), index(19), thumb_pitch(33), thumb_yaw(23)
    "right_ee": [21, 22, 20, 19, 33, 23],
}


@configclass
class StreamingRecorderCfg(RecorderTermCfg):
    """Configuration for streaming recorder term."""

    # NOTE: class_type is set after StreamingRecorder class is defined (end of file)

    task_dir: str = "/tmp/isaaclab/streaming_dataset"
    """Directory to save streaming episodes."""

    frequency: float = 30.0
    """Target recording frequency in Hz (for metadata)."""

    jpeg_quality: int = 95
    """JPEG compression quality (1-100)."""

    enable_rerun: bool = True
    """Enable rerun.io live visualization."""

    rerun_memory_limit: str = "300MB"
    """Memory limit for rerun viewer."""

    image_obs_keys: list[str] | None = None
    """Observation keys that contain images. If None, auto-detect by name pattern."""

    state_obs_keys: list[str] | None = None
    """Observation keys to record as state. If None, record all non-image obs."""

    unitree_format: bool = True
    """Output in Unitree format compatible with unitree_IL_lerobot converter."""

    capture_scene_state: bool = False
    """Capture full scene state for mimic annotation compatibility. Works with unitree_format=True or False."""


class StreamingRecorder(RecorderTerm):
    """Recorder term that streams data to disk in real-time.

    This recorder writes images as JPEG and state/action data as JSON
    directly to disk during recording, avoiding memory buildup.

    Output format is compatible with Unitree's unitree_IL_lerobot converter
    when unitree_format=True (default).

    Usage:
        Add this to your env config's recorders:

        ```python
        from isaaclab.envs.mdp.recorders.streaming_recorder import StreamingRecorderCfg

        @configclass
        class MyRecordersCfg:
            streaming = StreamingRecorderCfg(
                task_dir="./datasets/my_task",
                enable_rerun=True,
            )
        ```
    """

    cfg: StreamingRecorderCfg

    def __init__(self, cfg: StreamingRecorderCfg, env: ManagerBasedEnv):
        """Initialize the streaming recorder.

        Args:
            cfg: Configuration for the streaming recorder.
            env: The environment instance.
        """
        print(f"[StreamingRecorder] __init__ called with task_dir={cfg.task_dir}")
        super().__init__(cfg, env)

        # Create streaming writer
        task_info = {
            "goal": f"Recording from {env.cfg.env_name if hasattr(env.cfg, 'env_name') else 'unknown'}",
            "desc": "IsaacLab streaming recording",
            "steps": "Collected via teleoperation",
        }

        self._writer = StreamingEpisodeWriter(
            task_dir=cfg.task_dir,
            task_info=task_info,
            frequency=cfg.frequency,
            jpeg_quality=cfg.jpeg_quality,
            enable_rerun=cfg.enable_rerun,
            rerun_memory_limit=cfg.rerun_memory_limit,
        )

        self._recording = False
        self._current_env_id: int | None = None

        # Determine which observation keys are images vs states
        self._image_keys: set[str] = set()
        self._state_keys: set[str] = set()
        self._keys_initialized = False

        # Joint indices for Unitree format (populated from robot)
        self._joint_indices: dict[str, list[int]] = {}  # For robot state (robot joint indices)
        self._action_indices: dict[str, list[int]] = {}  # For actions (action space indices)
        self._joint_indices_initialized = False

    def _initialize_joint_indices(self):
        """Initialize joint indices for Unitree format conversion."""
        if self._joint_indices_initialized:
            return

        # Get robot from scene
        robot = None
        if hasattr(self._env, "scene"):
            for asset_name in self._env.scene.articulations:
                asset = self._env.scene.articulations[asset_name]
                if "robot" in asset_name.lower() or "g1" in asset_name.lower():
                    robot = asset
                    break

        if robot is None:
            print("[StreamingRecorder] Warning: Could not find robot in scene")
            self._joint_indices_initialized = True
            return

        # Get joint names from robot
        joint_names = robot.joint_names
        print(f"[StreamingRecorder] Robot joint names: {joint_names}")

        # Build index mapping for each group
        for group_name, group_info in G1_INSPIRE_JOINT_MAPPING.items():
            indices = []
            for joint_name in group_info["joint_names"]:
                # Try exact match first
                if joint_name in joint_names:
                    indices.append(joint_names.index(joint_name))
                else:
                    # Try partial match
                    found = False
                    for i, name in enumerate(joint_names):
                        if joint_name in name or name in joint_name:
                            indices.append(i)
                            found = True
                            break
                    if not found:
                        print(f"[StreamingRecorder] Warning: Could not find joint {joint_name}")
                        indices.append(-1)  # Placeholder

            self._joint_indices[group_name] = indices
            print(f"[StreamingRecorder] {group_name} indices: {indices}")

        # Get action indices dynamically from Pink IK action term
        # This ensures action indices match the actual processed_actions ordering
        self._initialize_action_indices()

        self._joint_indices_initialized = True

    def _initialize_action_indices(self):
        """Initialize action indices dynamically from Pink IK action term.

        This replaces the hardcoded PINK_IK_ACTION_INDICES with dynamic lookup
        based on the actual joint names in the Pink IK controller.
        """
        # Try to get action term with controlled joint names
        if hasattr(self._env, "action_manager"):
            for term_name in self._env.action_manager.active_terms:
                term = self._env.action_manager.get_term(term_name)
                if hasattr(term, "_controlled_joint_names"):
                    # Get the actual joint names from Pink IK (in processed_actions order)
                    controlled_joint_names = list(term._controlled_joint_names)
                    print(f"[StreamingRecorder] Pink IK controlled joints ({len(controlled_joint_names)}): {controlled_joint_names}")

                    # Build action index mapping using same name matching as state extraction
                    for group_name, group_info in G1_INSPIRE_JOINT_MAPPING.items():
                        indices = []
                        for joint_name in group_info["joint_names"]:
                            # Find index in controlled_joint_names
                            found = False
                            for i, name in enumerate(controlled_joint_names):
                                if joint_name == name or joint_name in name or name in joint_name:
                                    indices.append(i)
                                    found = True
                                    break
                            if not found:
                                print(f"[StreamingRecorder] Warning: Could not find action joint {joint_name}")
                                indices.append(-1)  # Placeholder

                        self._action_indices[group_name] = indices
                        print(f"[StreamingRecorder] Action {group_name} indices: {indices}")
                    return

        # Fallback to hardcoded indices if no Pink IK term found
        print("[StreamingRecorder] Warning: No Pink IK term found, using hardcoded PINK_IK_ACTION_INDICES")
        self._action_indices = {k: list(v) for k, v in PINK_IK_ACTION_INDICES.items()}

    def _initialize_keys(self):
        """Auto-detect image vs state observation keys."""
        if self._keys_initialized:
            return

        # Get observation keys from the policy group
        if hasattr(self._env, "observation_manager"):
            obs_terms = self._env.observation_manager.active_terms.get("policy", [])
            for term_name in obs_terms:
                term_name_lower = term_name.lower()
                # Check if this is an image observation
                is_image = any(
                    pattern in term_name_lower
                    for pattern in ["rgb", "image", "camera", "depth", "color"]
                )
                if is_image:
                    self._image_keys.add(term_name)
                else:
                    self._state_keys.add(term_name)

        # Override with explicit config if provided
        if self.cfg.image_obs_keys is not None:
            self._image_keys = set(self.cfg.image_obs_keys)
        if self.cfg.state_obs_keys is not None:
            self._state_keys = set(self.cfg.state_obs_keys)

        self._keys_initialized = True

    def reset(self, env_ids: Sequence[int] | None = None):
        """Handle reset - discard current episode if recording (manual reset).

        This is called by recorder_manager.reset() BEFORE env.reset().
        For manual reset (R key), this discards the incomplete episode.
        For success reset, record_pre_reset() already saved the episode,
        so _recording will be False and this does nothing.
        """
        if self._recording:
            # Manual reset - discard the current episode
            self._writer.discard_episode()
            self._recording = False
            print("[StreamingRecorder] Episode discarded (manual reset)")

    def record_pre_reset(self, env_ids: Sequence[int] | None) -> tuple[str | None, torch.Tensor | dict | None]:
        """Called before reset - save current episode if recording.

        Note: Success status is handled by record_demos.py since the success term
        is extracted from termination_manager before recording.
        """
        if self._recording and self._current_env_id in (env_ids or []):
            # Save episode - success status will be set by record_demos.py via
            # set_success_to_episodes() after this call
            self._writer.save_episode(success=None)
            self._recording = False
            print(f"[StreamingRecorder] Episode saved to {self._writer.episode_dir}")

            # Increment counter manually since StreamingRecorder bypasses self._episodes
            # The RecorderManager.export_episodes() counter logic won't run for us
            env_id = self._current_env_id
            rm = self._env.recorder_manager
            if not hasattr(rm, "_exported_successful_episode_count"):
                rm._exported_successful_episode_count = {}
            rm._exported_successful_episode_count[env_id] = rm._exported_successful_episode_count.get(env_id, 0) + 1

        return None, None

    def record_post_reset(self, env_ids: Sequence[int] | None) -> tuple[str | None, torch.Tensor | dict | None]:
        """Called after reset - start new episode."""
        if env_ids is not None and len(env_ids) > 0:
            # Start recording for first environment
            self._current_env_id = env_ids[0] if isinstance(env_ids, (list, tuple)) else env_ids[0].item()
            if self._writer.create_episode():
                self._recording = True

        return None, None

    def record_pre_step(self) -> tuple[str | None, torch.Tensor | dict | None]:
        """Called before step - record actions."""
        if not self._recording:
            return None, None

        # Actions will be captured in record_post_step along with states
        return None, None

    def record_post_step(self) -> tuple[str | None, torch.Tensor | dict | None]:
        """Called after step - stream data to disk."""
        if not self._recording or self._current_env_id is None:
            return None, None

        self._initialize_keys()
        if self.cfg.unitree_format:
            self._initialize_joint_indices()

        env_id = self._current_env_id

        # Collect images
        colors = {}
        if hasattr(self._env, "obs_buf") and "policy" in self._env.obs_buf:
            obs = self._env.obs_buf["policy"]
            if isinstance(obs, dict):
                for key in self._image_keys:
                    if key in obs:
                        img = obs[key][env_id]
                        # Map camera name if using Unitree format
                        if self.cfg.unitree_format:
                            unitree_key = CAMERA_MAPPING.get(key, key)
                            colors[unitree_key] = img
                        else:
                            colors[key] = img

        # Collect states
        states = {}
        actions = {}
        raw_actions = None  # Raw Pink IK input for replay/annotate compatibility

        if self.cfg.unitree_format:
            # Get joint positions from robot
            robot = None
            if hasattr(self._env, "scene"):
                for asset_name in self._env.scene.articulations:
                    asset = self._env.scene.articulations[asset_name]
                    if "robot" in asset_name.lower() or "g1" in asset_name.lower():
                        robot = asset
                        break

            if robot is not None:
                joint_pos = robot.data.joint_pos[env_id]

                # Extract joint groups for Unitree format
                for group_name, indices in self._joint_indices.items():
                    if indices:
                        # Filter out invalid indices
                        valid_indices = [i for i in indices if i >= 0]
                        if valid_indices:
                            states[group_name] = {"qpos": joint_pos[valid_indices]}

            # Get actions from action_manager (commanded positions, not current positions)
            # This is critical for imitation learning - action should be the target,
            # not the current state (otherwise delta = action - state = 0)
            if hasattr(self._env, "action_manager"):
                for term_name in self._env.action_manager.active_terms:
                    term = self._env.action_manager.get_term(term_name)
                    if hasattr(term, "processed_actions") and term.processed_actions is not None:
                        processed = term.processed_actions[env_id]
                        # Map processed actions to joint groups using action indices
                        # (not robot joint indices - they have different ordering)
                        for group_name, indices in self._action_indices.items():
                            if indices:
                                valid_indices = [i for i in indices if 0 <= i < len(processed)]
                                if valid_indices:
                                    actions[group_name] = {"qpos": processed[valid_indices]}
                        break  # Use first action term with processed_actions

            # Also capture raw actions (Pink IK input) for replay/annotate compatibility
            raw_actions = None
            if hasattr(self._env, "action_manager"):
                raw_action = self._env.action_manager.action
                if raw_action is not None:
                    raw_actions = raw_action[env_id].cpu().tolist()

            # Capture scene state if configured (for mimic annotation compatibility)
            if self.cfg.capture_scene_state and hasattr(self._env, "scene"):
                scene_state = self._env.scene.get_state(is_relative=True)

                def extract_env_state(state_dict, env_id):
                    result = {}
                    for k, v in state_dict.items():
                        if isinstance(v, dict):
                            result[k] = extract_env_state(v, env_id)
                        elif isinstance(v, torch.Tensor):
                            result[k] = v[env_id].cpu().tolist() if v.numel() > 0 else None
                    return result

                states["scene"] = extract_env_state(scene_state, env_id)
        else:
            # Original format
            if hasattr(self._env, "obs_buf") and "policy" in self._env.obs_buf:
                obs = self._env.obs_buf["policy"]
                if isinstance(obs, dict):
                    for key in self._state_keys:
                        if key in obs:
                            states[key] = obs[key][env_id]

            # Collect scene state (robot joints, etc.) if configured
            if self.cfg.capture_scene_state and hasattr(self._env, "scene"):
                scene_state = self._env.scene.get_state(is_relative=True)
                # Extract just the env_id slice
                def extract_env_state(state_dict, env_id):
                    result = {}
                    for k, v in state_dict.items():
                        if isinstance(v, dict):
                            result[k] = extract_env_state(v, env_id)
                        elif isinstance(v, torch.Tensor):
                            result[k] = v[env_id]
                    return result
                states["scene"] = extract_env_state(scene_state, env_id)

            # Collect actions
            if hasattr(self._env, "action_manager"):
                action = self._env.action_manager.action
                if action is not None:
                    actions["raw"] = action[env_id]

                # Also get processed actions
                for term_name in self._env.action_manager.active_terms:
                    term = self._env.action_manager.get_term(term_name)
                    if hasattr(term, "processed_actions"):
                        actions[term_name] = term.processed_actions[env_id]

        # Stream to disk
        self._writer.add_item(
            colors=colors,
            states=states,
            actions=actions,
            raw_actions=raw_actions,
        )

        return None, None

    def close(self, file_path: str):
        """Clean up streaming writer."""
        if self._recording:
            self._writer.save_episode()
        self._writer.close()


# Set class_type after class is defined
StreamingRecorderCfg.class_type = StreamingRecorder
