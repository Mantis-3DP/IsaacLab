#!/usr/bin/env python3
"""
Convert StreamingRecorder JPEG/JSON dataset to HDF5 format for annotate_demos.py.

This script converts recorded demonstrations from the streaming format
(JPEG images + JSON state/action data) to HDF5 format used by Isaac Lab's
annotation and imitation learning tools.

Usage:
    python convert_streaming_to_hdf5.py \
        --input /path/to/streaming_dataset \
        --output ./datasets/converted.hdf5 \
        --task Isaac-PickPlaceTarget-Cube-G1-Inspire-v0

Requirements:
    pip install h5py numpy pillow tqdm
"""

import argparse
import json
import os
from pathlib import Path

import h5py
import numpy as np
from PIL import Image
from tqdm import tqdm


# Streaming 26 DOF layout (Unitree format)
STREAMING_GROUPS = {
    "left_arm": {"dof": 7},
    "right_arm": {"dof": 7},
    "left_ee": {"dof": 6},  # Left hand
    "right_ee": {"dof": 6},  # Right hand
}

# Robot 53 DOF layout (IsaacLab G1 Inspire)
ROBOT_53DOF_INDICES = {
    "torso": (0, 3),
    "left_leg": (3, 9),
    "right_leg": (9, 15),
    "head": (15, 17),
    "left_arm": (17, 24),
    "right_arm": (24, 31),
    "left_hand": (31, 42),  # 11 joints
    "right_hand": (42, 53),  # 11 joints
}

# Mapping from streaming hand joints (6) to IsaacLab hand joints (11)
# Streaming has: pinky, ring, middle, index, thumb_pitch, thumb_yaw
# IsaacLab has: 11 joints including intermediate/distal
HAND_6_TO_11_MAPPING = {
    # Input index (in 6 DOF) -> Output index (in 11 DOF)
    0: 2,   # pinky_proximal
    1: 3,   # ring_proximal
    2: 1,   # middle_proximal
    3: 0,   # index_proximal
    4: 9,   # thumb_pitch (proximal_pitch)
    5: 4,   # thumb_yaw
}

# Camera mapping: Unitree format -> IsaacLab format
CAMERA_UNITREE_TO_ISAACLAB = {
    "color_0": "head_rgb_left",
    "color_1": "head_rgb_right",
    "color_2": "cam_left_wrist",
    "color_3": "cam_right_wrist",
}


def list_episodes(input_dir: Path) -> list[Path]:
    """Find all episode directories in input."""
    episodes = []
    for entry in sorted(input_dir.iterdir()):
        if entry.is_dir() and entry.name.startswith("episode_"):
            data_json = entry / "data.json"
            if data_json.exists():
                episodes.append(entry)
    return episodes


def load_episode(episode_dir: Path) -> dict | None:
    """Load episode data from streaming format."""
    data_json = episode_dir / "data.json"

    with open(data_json, "r") as f:
        content = f.read()

    # Fix Python None -> JSON null
    content = content.replace('"success": none', '"success": null')
    content = content.replace('"success": None', '"success": null')

    try:
        episode_data = json.loads(content)
    except json.JSONDecodeError as e:
        print(f"Warning: Invalid JSON in {episode_dir}: {e}")
        return None

    if "data" not in episode_data or len(episode_data.get("data", [])) == 0:
        print(f"Warning: Empty episode {episode_dir.name}, skipping")
        return None

    return episode_data


def extract_26dof_from_groups(groups: dict) -> np.ndarray:
    """Extract 26 DOF vector from grouped format.

    Args:
        groups: Dict with keys like 'left_arm', 'right_arm', 'left_ee', 'right_ee'

    Returns:
        26 DOF array: [left_arm(7), right_arm(7), left_hand(6), right_hand(6)]
    """
    dof_26 = np.zeros(26, dtype=np.float32)

    # Left arm (7 DOF) at indices 0-6
    if "left_arm" in groups and "qpos" in groups["left_arm"]:
        qpos = groups["left_arm"]["qpos"]
        dof_26[0:7] = qpos[:7]

    # Right arm (7 DOF) at indices 7-13
    if "right_arm" in groups and "qpos" in groups["right_arm"]:
        qpos = groups["right_arm"]["qpos"]
        dof_26[7:14] = qpos[:7]

    # Left hand (6 DOF) at indices 14-19
    if "left_ee" in groups and "qpos" in groups["left_ee"]:
        qpos = groups["left_ee"]["qpos"]
        dof_26[14:20] = qpos[:6]

    # Right hand (6 DOF) at indices 20-25
    if "right_ee" in groups and "qpos" in groups["right_ee"]:
        qpos = groups["right_ee"]["qpos"]
        dof_26[20:26] = qpos[:6]

    return dof_26


def expand_26dof_to_53dof(dof_26: np.ndarray) -> np.ndarray:
    """Expand 26 DOF to 53 DOF robot state.

    The 26 DOF format only has arm and hand joints. We fill in zeros for
    torso, legs, and head, and expand 6 DOF hands to 11 DOF.

    Args:
        dof_26: 26 DOF array [left_arm(7), right_arm(7), left_hand(6), right_hand(6)]

    Returns:
        53 DOF array matching full G1 Inspire robot state
    """
    dof_53 = np.zeros(53, dtype=np.float32)

    # Left arm (7 DOF) -> indices 17-24
    dof_53[17:24] = dof_26[0:7]

    # Right arm (7 DOF) -> indices 24-31
    dof_53[24:31] = dof_26[7:14]

    # Left hand: expand 6 DOF to 11 DOF
    left_hand_6 = dof_26[14:20]
    for src_idx, dst_idx in HAND_6_TO_11_MAPPING.items():
        dof_53[31 + dst_idx] = left_hand_6[src_idx]

    # Right hand: expand 6 DOF to 11 DOF
    right_hand_6 = dof_26[20:26]
    for src_idx, dst_idx in HAND_6_TO_11_MAPPING.items():
        dof_53[42 + dst_idx] = right_hand_6[src_idx]

    return dof_53


def build_initial_state(first_state_26: np.ndarray, scene_state: dict | None = None) -> dict:
    """Build initial_state structure for reset_to.

    Args:
        first_state_26: First frame's 26 DOF state
        scene_state: Optional scene state dict from StreamingRecorder (with capture_scene_state=True)

    Returns:
        Dict in InteractiveScene.get_state() format
    """
    # Expand to 53 DOF
    joint_pos_53 = expand_26dof_to_53dof(first_state_26)

    # Default robot root pose
    robot_root_pose = np.array([[0, 0, 0.84, 1, 0, 0, 0]], dtype=np.float32)

    # Extract robot root pose from scene state if available
    if scene_state:
        articulation = scene_state.get("articulation", {})
        robot_state = articulation.get("robot", {})
        if "root_pose" in robot_state:
            rp = robot_state["root_pose"]
            if isinstance(rp, list):
                robot_root_pose = np.array([rp], dtype=np.float32)

    # Create state structure
    initial_state = {
        "articulation": {
            "robot": {
                "root_pose": robot_root_pose,
                "root_velocity": np.zeros((1, 6), dtype=np.float32),
                "joint_position": joint_pos_53.reshape(1, -1),
                "joint_velocity": np.zeros((1, 53), dtype=np.float32),
            }
        },
        "rigid_object": {},
    }

    # Extract object poses from scene state if available
    if scene_state:
        rigid_objects = scene_state.get("rigid_object", {})
        for obj_name, obj_state in rigid_objects.items():
            if "root_pose" in obj_state:
                rp = obj_state["root_pose"]
                if isinstance(rp, list):
                    initial_state["rigid_object"][obj_name] = {
                        "root_pose": np.array([rp], dtype=np.float32),
                        "root_velocity": np.zeros((1, 6), dtype=np.float32),
                    }

    return initial_state


def convert_episode_to_hdf5(
    episode_dir: Path,
    episode_idx: int,
    hdf5_group: h5py.Group,
    skip_images: bool = False,
    use_state_as_action: bool = True,
) -> dict:
    """Convert a single episode to HDF5 format.

    Args:
        episode_dir: Path to episode directory
        episode_idx: Episode index
        hdf5_group: HDF5 data group to write to
        skip_images: Skip loading images
        use_state_as_action: If True, use state data for actions when action data is incomplete

    Returns:
        Episode info dict
    """
    episode_data = load_episode(episode_dir)
    if episode_data is None:
        return None

    data_items = episode_data.get("data", [])
    num_frames = len(data_items)
    is_success = episode_data.get("success", None)

    if num_frames == 0:
        return None

    # Create demo group
    demo_name = f"demo_{episode_idx}"
    demo_group = hdf5_group.create_group(demo_name)
    demo_group.attrs["num_samples"] = num_frames
    if is_success is not None:
        demo_group.attrs["success"] = is_success

    # Collect state and action arrays
    states_26 = []
    actions_26 = []
    raw_actions_list = []  # Raw Pink IK actions for replay/annotate
    camera_frames = {}  # cam_key -> list of frames
    has_raw_actions = False
    first_frame_scene_state = None  # Scene state from first frame for initial_state

    for idx, item in enumerate(data_items):
        # Capture scene state from first frame (for mimic annotation)
        if idx == 0:
            states_dict = item.get("states", {})
            first_frame_scene_state = states_dict.get("scene", None)
            if first_frame_scene_state:
                print(f"  Found scene state with objects: {list(first_frame_scene_state.get('rigid_object', {}).keys())}")
        # Extract state (26 DOF)
        states = item.get("states", {})
        state_26 = extract_26dof_from_groups(states)
        states_26.append(state_26)

        # Extract action
        actions = item.get("actions", {})
        if use_state_as_action or not actions:
            # Use state as action (for imitation learning, action = target state)
            action_26 = state_26.copy()
        else:
            # Build action from available groups
            action_26 = np.zeros(26, dtype=np.float32)
            # Check what's available in actions
            if "left_arm" in actions:
                action_26[0:7] = actions["left_arm"]["qpos"][:7]
            else:
                # Copy from state if not in actions
                action_26[0:7] = state_26[0:7]
            if "right_arm" in actions:
                action_26[7:14] = actions["right_arm"]["qpos"][:7]
            else:
                action_26[7:14] = state_26[7:14]
            if "left_ee" in actions:
                action_26[14:20] = actions["left_ee"]["qpos"][:6]
            else:
                action_26[14:20] = state_26[14:20]
            if "right_ee" in actions:
                action_26[20:26] = actions["right_ee"]["qpos"][:6]
            else:
                action_26[20:26] = state_26[20:26]
        actions_26.append(action_26)

        # Extract raw actions (Pink IK input) if available
        raw_action = item.get("raw_actions", None)
        if raw_action is not None:
            has_raw_actions = True
            raw_actions_list.append(np.array(raw_action, dtype=np.float32))
        else:
            # Placeholder - will be replaced if any raw_actions exist
            raw_actions_list.append(None)

        # Load images
        if not skip_images:
            colors_paths = item.get("colors", {})
            for cam_key, rel_path in colors_paths.items():
                if cam_key not in camera_frames:
                    camera_frames[cam_key] = []

                img_path = episode_dir / rel_path
                if img_path.exists():
                    img = np.array(Image.open(img_path).convert("RGB"))
                    camera_frames[cam_key].append(img)
                else:
                    print(f"Warning: Image not found: {img_path}")

    # Convert to arrays
    states_26 = np.array(states_26, dtype=np.float32)
    actions_26 = np.array(actions_26, dtype=np.float32)

    # Expand states to 53 DOF for robot_joint_pos
    states_53 = np.array([expand_26dof_to_53dof(s) for s in states_26], dtype=np.float32)

    # Create datasets
    # If raw_actions available, use them for HDF5 actions (for replay/annotate)
    # Otherwise fall back to 26 DOF joint positions
    if has_raw_actions:
        # Convert raw_actions list to array
        raw_actions_array = np.array(raw_actions_list, dtype=np.float32)
        demo_group.create_dataset("actions", data=raw_actions_array, compression="gzip")
        print(f"  Using raw_actions ({raw_actions_array.shape[1]} DOF) for replay compatibility")
    else:
        # Fall back: use 26 DOF joint positions (won't work for Pink IK replay)
        demo_group.create_dataset("actions", data=actions_26, compression="gzip")
        print(f"  Warning: No raw_actions found, using 26 DOF joint positions (Pink IK replay won't work)")

    # Store 26 DOF joint positions as processed_actions (for LeRobot conversion)
    demo_group.create_dataset("processed_actions", data=actions_26, compression="gzip")

    # Initial state for reset_to (includes object poses if scene state was recorded)
    initial_state = build_initial_state(states_26[0], first_frame_scene_state)

    def save_nested_dict(group, key, value):
        """Recursively save nested dict to HDF5."""
        if isinstance(value, dict):
            subgroup = group.create_group(key)
            for k, v in value.items():
                save_nested_dict(subgroup, k, v)
        else:
            group.create_dataset(key, data=value, compression="gzip")

    save_nested_dict(demo_group, "initial_state", initial_state)

    # Observations
    obs_group = demo_group.create_group("obs")

    # Robot joint positions (53 DOF)
    obs_group.create_dataset("robot_joint_pos", data=states_53, compression="gzip")

    # Also store 26 DOF version for reference
    obs_group.create_dataset("robot_joint_pos_26dof", data=states_26, compression="gzip")

    # Camera observations
    if not skip_images:
        for cam_key, frames in camera_frames.items():
            if frames:
                # Map to IsaacLab camera name
                isaaclab_name = CAMERA_UNITREE_TO_ISAACLAB.get(cam_key, cam_key)
                frames_array = np.array(frames, dtype=np.uint8)
                obs_group.create_dataset(isaaclab_name, data=frames_array, compression="gzip")

    return {
        "episode_index": episode_idx,
        "num_frames": num_frames,
        "success": is_success,
    }


def main():
    parser = argparse.ArgumentParser(description="Convert streaming dataset to HDF5 format")
    parser.add_argument("--input", "-i", type=str, required=True,
                        help="Input streaming dataset directory")
    parser.add_argument("--output", "-o", type=str, required=True,
                        help="Output HDF5 file path")
    parser.add_argument("--task", type=str, default="Isaac-PickPlaceTarget-Cube-G1-Inspire-v0",
                        help="Task/environment name")
    parser.add_argument("--skip-images", action="store_true",
                        help="Skip loading images (for testing)")
    parser.add_argument("--use-state-as-action", action="store_true", default=True,
                        help="Use state data as actions (default: True)")
    parser.add_argument("--max-episodes", type=int, default=None,
                        help="Limit number of episodes (for testing)")
    args = parser.parse_args()

    input_path = Path(args.input).expanduser()
    output_path = Path(args.output).expanduser()

    if not input_path.exists():
        raise FileNotFoundError(f"Input directory not found: {input_path}")

    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Find episodes
    episodes = list_episodes(input_path)
    print(f"Found {len(episodes)} episodes in {input_path}")

    if args.max_episodes:
        episodes = episodes[:args.max_episodes]
        print(f"Limiting to {len(episodes)} episodes")

    if not episodes:
        raise ValueError("No valid episodes found")

    # Create HDF5 file
    with h5py.File(output_path, "w") as hdf5_file:
        # Create data group
        data_group = hdf5_file.create_group("data")

        # Set environment info
        env_args = {
            "env_name": args.task,
            "type": 2,  # robomimic compatibility
        }
        data_group.attrs["env_args"] = json.dumps(env_args)
        data_group.attrs["total"] = 0

        # Convert episodes
        total_frames = 0
        successful_episodes = 0
        converted_count = 0

        for ep_idx, episode_dir in enumerate(tqdm(episodes, desc="Converting episodes")):
            ep_info = convert_episode_to_hdf5(
                episode_dir=episode_dir,
                episode_idx=converted_count,
                hdf5_group=data_group,
                skip_images=args.skip_images,
                use_state_as_action=args.use_state_as_action,
            )

            if ep_info is not None:
                total_frames += ep_info["num_frames"]
                if ep_info["success"]:
                    successful_episodes += 1
                converted_count += 1

        # Update total frames
        data_group.attrs["total"] = total_frames

    print(f"\nConversion complete!")
    print(f"  Episodes: {converted_count} (of {len(episodes)} total)")
    print(f"  Frames: {total_frames}")
    print(f"  Successful: {successful_episodes}")
    print(f"  Output: {output_path}")
    print(f"\nNote: Actions are stored as 26 DOF joint positions (Unitree format).")
    print(f"      States are stored as both 26 DOF and 53 DOF (expanded).")
    print(f"\nTo use with annotate_demos.py, you may need to adjust the action space")
    print(f"or use a joint position action mode instead of Pink IK.")


if __name__ == "__main__":
    main()
