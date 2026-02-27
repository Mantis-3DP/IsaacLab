# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Streaming recorder that writes data to disk in real-time.

This recorder bypasses the normal EpisodeData accumulation and HDF5 batch writing,
instead streaming images as JPEG and state/action data as incremental JSON.

Joint structure is discovered dynamically from the env's action terms at runtime,
so it works with any robot (Inspire, TriHand, etc.) without hardcoded mappings.

Benefits:
- No memory buildup during recording
- Near-instant episode saves (just close file handles)
- 10-20x smaller storage via JPEG compression
- Optional rerun.io live visualization
- Robot-agnostic: discovers joint groups from Pink IK action term
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


# Camera name mapping: IsaacLab name -> Unitree format
CAMERA_MAPPING = {
    "head_rgb_left": "color_0",
    "head_rgb_right": "color_1",
    "cam_left_high": "color_0",
    "cam_right_high": "color_1",
    "cam_left_wrist": "color_2",
    "cam_right_wrist": "color_3",
    "ego_view": "color_0",
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

    Joint structure is discovered dynamically from the env's action terms,
    so it works with any robot without hardcoded joint mappings.

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
        self._metadata_written = False

        # Determine which observation keys are images vs states
        self._image_keys: set[str] = set()
        self._state_keys: set[str] = set()
        self._keys_initialized = False

        # Dynamic joint structure (discovered from action terms)
        self._joint_indices: dict[str, list[int]] = {}       # group -> robot joint indices
        self._action_indices: dict[str, list[int]] = {}      # group -> processed_actions indices
        self._joint_names_map: dict[str, list[str]] = {}     # group -> joint name list
        self._eef_link_indices: dict[str, int] = {}           # task_name -> body index
        self._eef_link_names: dict[str, str] = {}             # task_name -> link name
        self._robot = None                                     # cached robot reference
        self._joint_structure_initialized = False

    # ------------------------------------------------------------------
    # Robot and joint discovery
    # ------------------------------------------------------------------

    def _find_robot(self):
        """Find and cache the robot articulation from the scene."""
        if self._robot is not None:
            return self._robot
        if hasattr(self._env, "scene"):
            for asset_name in self._env.scene.articulations:
                asset = self._env.scene.articulations[asset_name]
                if "robot" in asset_name.lower() or "g1" in asset_name.lower():
                    self._robot = asset
                    return self._robot
        return None

    def _discover_joint_structure(self):
        """Discover joint groups, indices, and EEF links from action terms.

        Queries the Pink IK action term (if present) for arm joints, hand joints,
        and EEF link names. Splits into left/right groups by name pattern.
        Falls back to recording all robot joints as a flat group if no IK term found.
        """
        if self._joint_structure_initialized:
            return

        robot = self._find_robot()
        if robot is None:
            print("[StreamingRecorder] Warning: Could not find robot in scene")
            self._joint_structure_initialized = True
            return

        # --- Find Pink IK action term ---
        ik_term = None
        if hasattr(self._env, "action_manager"):
            for term_name in self._env.action_manager.active_terms:
                term = self._env.action_manager.get_term(term_name)
                if hasattr(term, "_isaaclab_controlled_joint_names"):
                    ik_term = term
                    break

        if ik_term is None:
            # Fallback: record ALL robot joints as a single flat group
            print("[StreamingRecorder] No Pink IK term found, recording all joints as flat group")
            all_names = list(robot.joint_names)
            self._joint_indices["joints"] = list(range(len(all_names)))
            self._joint_names_map["joints"] = all_names
            self._joint_structure_initialized = True
            return

        # --- Split arm joints into left/right/torso ---
        arm_names = list(ik_term._isaaclab_controlled_joint_names)
        arm_ids = list(ik_term._isaaclab_controlled_joint_ids)

        left_arm_names, left_arm_robot_ids, left_arm_action_ids = [], [], []
        right_arm_names, right_arm_robot_ids, right_arm_action_ids = [], [], []
        other_arm_names, other_arm_robot_ids, other_arm_action_ids = [], [], []

        for i, (name, robot_idx) in enumerate(zip(arm_names, arm_ids)):
            if "left" in name.lower() or name.startswith("L_") or name.startswith("l_"):
                left_arm_names.append(name)
                left_arm_robot_ids.append(robot_idx)
                left_arm_action_ids.append(i)
            elif "right" in name.lower() or name.startswith("R_") or name.startswith("r_"):
                right_arm_names.append(name)
                right_arm_robot_ids.append(robot_idx)
                right_arm_action_ids.append(i)
            else:
                # Torso joints (e.g., waist_yaw, waist_pitch, waist_roll)
                other_arm_names.append(name)
                other_arm_robot_ids.append(robot_idx)
                other_arm_action_ids.append(i)

        # --- Split hand joints into left/right ---
        hand_names = list(ik_term._hand_joint_names)
        hand_ids = list(ik_term._hand_joint_ids)
        n_arm_joints = len(arm_names)

        left_hand_names, left_hand_robot_ids, left_hand_action_ids = [], [], []
        right_hand_names, right_hand_robot_ids, right_hand_action_ids = [], [], []

        for i, (name, robot_idx) in enumerate(zip(hand_names, hand_ids)):
            action_idx = n_arm_joints + i  # offset into processed_actions
            if "left" in name.lower() or name.startswith("L_") or name.startswith("l_"):
                left_hand_names.append(name)
                left_hand_robot_ids.append(robot_idx)
                left_hand_action_ids.append(action_idx)
            elif "right" in name.lower() or name.startswith("R_") or name.startswith("r_"):
                right_hand_names.append(name)
                right_hand_robot_ids.append(robot_idx)
                right_hand_action_ids.append(action_idx)
            else:
                print(f"[StreamingRecorder] Warning: Cannot determine side for hand joint '{name}', skipping")

        # --- Store discovered groups ---
        if left_arm_robot_ids:
            self._joint_indices["left_arm"] = left_arm_robot_ids
            self._action_indices["left_arm"] = left_arm_action_ids
            self._joint_names_map["left_arm"] = left_arm_names

        if right_arm_robot_ids:
            self._joint_indices["right_arm"] = right_arm_robot_ids
            self._action_indices["right_arm"] = right_arm_action_ids
            self._joint_names_map["right_arm"] = right_arm_names

        if other_arm_robot_ids:
            self._joint_indices["torso"] = other_arm_robot_ids
            self._action_indices["torso"] = other_arm_action_ids
            self._joint_names_map["torso"] = other_arm_names

        if left_hand_robot_ids:
            self._joint_indices["left_hand"] = left_hand_robot_ids
            self._action_indices["left_hand"] = left_hand_action_ids
            self._joint_names_map["left_hand"] = left_hand_names

        if right_hand_robot_ids:
            self._joint_indices["right_hand"] = right_hand_robot_ids
            self._action_indices["right_hand"] = right_hand_action_ids
            self._joint_names_map["right_hand"] = right_hand_names

        # --- Discover EEF links from Pink IK config ---
        if hasattr(ik_term, "cfg") and hasattr(ik_term.cfg, "target_eef_link_names"):
            eef_map: dict[str, str] = ik_term.cfg.target_eef_link_names
            body_names = list(robot.data.body_names)
            for task_name, link_name in eef_map.items():
                if link_name in body_names:
                    self._eef_link_indices[task_name] = body_names.index(link_name)
                    self._eef_link_names[task_name] = link_name
                else:
                    print(f"[StreamingRecorder] Warning: EEF link '{link_name}' not found in body_names")

        self._joint_structure_initialized = True
        self._print_discovered_structure()

    def _print_discovered_structure(self):
        """Print the discovered joint structure for debugging."""
        print("[StreamingRecorder] === Discovered Joint Structure ===")
        for group_name in self._joint_indices:
            n_joints = len(self._joint_indices[group_name])
            names = self._joint_names_map.get(group_name, [])
            action_ids = self._action_indices.get(group_name, [])
            print(f"  {group_name}: {n_joints} joints, robot_ids={self._joint_indices[group_name]}, "
                  f"action_ids={action_ids}")
            if names:
                print(f"    names: {names}")
        if self._eef_link_indices:
            print("  EEF links:")
            for task_name, body_idx in self._eef_link_indices.items():
                print(f"    {task_name}: body_idx={body_idx}, link='{self._eef_link_names[task_name]}'")
        print("[StreamingRecorder] ================================")

    # ------------------------------------------------------------------
    # Observation key discovery
    # ------------------------------------------------------------------

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
                    for pattern in ["rgb", "image", "camera", "depth", "color", "ego", "view"]
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

    # ------------------------------------------------------------------
    # Episode lifecycle
    # ------------------------------------------------------------------

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
                self._metadata_written = False

        return None, None

    def record_pre_step(self) -> tuple[str | None, torch.Tensor | dict | None]:
        """Called before step - record actions."""
        if not self._recording:
            return None, None

        # Actions will be captured in record_post_step along with states
        return None, None

    # ------------------------------------------------------------------
    # Main recording logic
    # ------------------------------------------------------------------

    def record_post_step(self) -> tuple[str | None, torch.Tensor | dict | None]:
        """Called after step - stream data to disk."""
        if not self._recording or self._current_env_id is None:
            return None, None

        self._initialize_keys()
        if self.cfg.unitree_format:
            self._discover_joint_structure()

        env_id = self._current_env_id

        # Write joint metadata to episode on first frame
        if self.cfg.unitree_format and not self._metadata_written:
            self._write_metadata_to_episode()
            self._metadata_written = True

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
            robot = self._find_robot()

            if robot is not None:
                joint_pos = robot.data.joint_pos[env_id]

                # Extract joint states using discovered groups
                for group_name, indices in self._joint_indices.items():
                    if indices:
                        states[group_name] = {"qpos": joint_pos[indices]}

                # Record measured EEF poses (from forward kinematics)
                if self._eef_link_indices:
                    body_pos = robot.data.body_pos_w[env_id]
                    body_quat = robot.data.body_quat_w[env_id]
                    env_origin = self._env.scene.env_origins[env_id]
                    for task_name, body_idx in self._eef_link_indices.items():
                        # Position in env-origin frame (not world frame)
                        states[f"{task_name}_pos"] = {"qpos": (body_pos[body_idx] - env_origin).cpu().tolist()}
                        # Quaternion (wxyz convention from IsaacLab)
                        states[f"{task_name}_quat"] = {"qpos": body_quat[body_idx].cpu().tolist()}

            # Get actions from action_manager (commanded positions, not current positions)
            if hasattr(self._env, "action_manager"):
                for term_name in self._env.action_manager.active_terms:
                    term = self._env.action_manager.get_term(term_name)
                    if hasattr(term, "processed_actions") and term.processed_actions is not None:
                        processed = term.processed_actions[env_id]
                        # Map processed actions to joint groups using discovered action indices
                        for group_name, indices in self._action_indices.items():
                            if indices:
                                valid = [i for i in indices if 0 <= i < len(processed)]
                                if valid:
                                    actions[group_name] = {"qpos": processed[valid]}
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

    # ------------------------------------------------------------------
    # Metadata
    # ------------------------------------------------------------------

    def _write_metadata_to_episode(self):
        """Write joint group metadata to the episode for downstream converters.

        Stores metadata on the writer so it gets serialized into the episode JSON.
        This tells downstream converters the joint names and ordering for each group.
        """
        metadata = {
            "joint_groups": {k: list(v) for k, v in self._joint_names_map.items()},
            "eef_links": dict(self._eef_link_names),
        }
        # Store on the writer instance for serialization
        self._writer.episode_metadata = metadata

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def close(self, file_path: str):
        """Clean up streaming writer."""
        if self._recording:
            self._writer.save_episode()
        self._writer.close()


# Set class_type after class is defined
StreamingRecorderCfg.class_type = StreamingRecorder
