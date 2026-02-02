#!/usr/bin/env python3
"""
Convert Isaac Lab HDF5 demonstration dataset to LeRobot v2.1 format for GR00T.

This script converts recorded demonstrations from Isaac Lab's HDF5 format to the
LeRobot v2.1 dataset format required by NVIDIA GR00T, which includes:
- Parquet files with concatenated action/state arrays
- MP4 videos for camera observations (AV1 codec)
- JSON metadata files including modality.json for GR00T

Usage:
    python convert_hdf5_to_lerobot.py \
        --input /path/to/demos.hdf5 \
        --output ~/.cache/huggingface/lerobot/dataset_name \
        --repo-id username/dataset_name

Requirements:
    pip install h5py pyarrow pandas imageio[ffmpeg] tqdm
"""

import argparse
import json
import os
import shutil
from datetime import datetime
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm

# Try to import imageio for video encoding
try:
    import imageio.v3 as iio
    HAS_IMAGEIO = True
except ImportError:
    HAS_IMAGEIO = False
    print("Warning: imageio not found. Install with: pip install imageio[ffmpeg]")


# G1 Inspire joint configuration (53 DOF total)
G1_JOINT_GROUPS = {
    "torso": {"start": 0, "end": 3, "joints": ["torso_yaw", "torso_pitch", "torso_roll"]},
    "left_leg": {"start": 3, "end": 9, "joints": ["left_hip_yaw", "left_hip_roll", "left_hip_pitch", "left_knee", "left_ankle_pitch", "left_ankle_roll"]},
    "right_leg": {"start": 9, "end": 15, "joints": ["right_hip_yaw", "right_hip_roll", "right_hip_pitch", "right_knee", "right_ankle_pitch", "right_ankle_roll"]},
    "head": {"start": 15, "end": 17, "joints": ["head_pitch", "head_yaw"]},
    "left_arm": {"start": 17, "end": 24, "joints": ["left_shoulder_pitch", "left_shoulder_roll", "left_shoulder_yaw", "left_elbow_pitch", "left_wrist_yaw", "left_wrist_pitch", "left_wrist_roll"]},
    "right_arm": {"start": 24, "end": 31, "joints": ["right_shoulder_pitch", "right_shoulder_roll", "right_shoulder_yaw", "right_elbow_pitch", "right_wrist_yaw", "right_wrist_pitch", "right_wrist_roll"]},
    "left_hand": {"start": 31, "end": 42, "joints": [f"left_finger_{i}" for i in range(11)]},
    "right_hand": {"start": 42, "end": 53, "joints": [f"right_finger_{i}" for i in range(11)]},
}

# Action dimension mapping (38 DOF for manipulation task)
G1_ACTION_GROUPS = {
    "left_arm": {"start": 0, "end": 7},
    "right_arm": {"start": 7, "end": 14},
    "left_hand": {"start": 14, "end": 26},
    "right_hand": {"start": 26, "end": 38},
}

# Unitree G1 Inspire 26 DOF action groups (for unitree_IL_lerobot compatibility)
UNITREE_G1_INSPIRE_ACTION_GROUPS = {
    "left_arm": {"start": 0, "end": 7},
    "right_arm": {"start": 7, "end": 14},
    "left_hand": {"start": 14, "end": 20},
    "right_hand": {"start": 20, "end": 26},
}

# Joint names for 26 DOF format
UNITREE_G1_INSPIRE_ACTION_NAMES = [
    # Left arm (7)
    "left_shoulder_pitch", "left_shoulder_roll", "left_shoulder_yaw",
    "left_elbow_pitch", "left_wrist_yaw", "left_wrist_pitch", "left_wrist_roll",
    # Right arm (7)
    "right_shoulder_pitch", "right_shoulder_roll", "right_shoulder_yaw",
    "right_elbow_pitch", "right_wrist_yaw", "right_wrist_pitch", "right_wrist_roll",
    # Left hand (6)
    "left_pinky", "left_ring", "left_middle", "left_index", "left_thumb_bend", "left_thumb_rot",
    # Right hand (6)
    "right_pinky", "right_ring", "right_middle", "right_index", "right_thumb_bend", "right_thumb_rot",
]

# IsaacLab 24 hand joints → Unitree 12 hand joints mapping
# Maps IsaacLab index to Unitree index (only drive joints, not intermediate/distal)
ISAACLAB_TO_UNITREE_HAND = {
    # Left hand
    2: 0,   # L_pinky_proximal → left_pinky
    3: 1,   # L_ring_proximal → left_ring
    1: 2,   # L_middle_proximal → left_middle
    0: 3,   # L_index_proximal → left_index
    14: 4,  # L_thumb_pitch → left_thumb_bend
    4: 5,   # L_thumb_yaw → left_thumb_rot
    # Right hand
    7: 6,   # R_pinky_proximal → right_pinky
    8: 7,   # R_ring_proximal → right_ring
    6: 8,   # R_middle_proximal → right_middle
    5: 9,   # R_index_proximal → right_index
    19: 10, # R_thumb_pitch → right_thumb_bend
    9: 11,  # R_thumb_yaw → right_thumb_rot
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert Isaac Lab HDF5 demos to LeRobot v2.1 format for GR00T"
    )
    parser.add_argument(
        "--input", "-i",
        type=str,
        required=True,
        help="Path to input HDF5 file"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        required=True,
        help="Output directory for LeRobot dataset"
    )
    parser.add_argument(
        "--repo-id",
        type=str,
        default=None,
        help="HuggingFace repo ID (e.g., 'username/dataset_name')"
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=20,
        help="Video/data FPS (default: 20, based on Isaac Lab dt=0.00833, decimation=6)"
    )
    parser.add_argument(
        "--video-codec",
        type=str,
        default="libx264",
        choices=["libx264", "libaom-av1", "libsvtav1"],
        help="Video codec (default: libx264, use libaom-av1 or libsvtav1 for AV1)"
    )
    parser.add_argument(
        "--video-quality",
        type=int,
        default=23,
        help="Video quality (CRF, lower=better, default: 23)"
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=1000,
        help="Max episodes per chunk (default: 1000)"
    )
    parser.add_argument(
        "--robot-type",
        type=str,
        default="unitree_g1_inspire",
        help="Robot type identifier (default: unitree_g1_inspire)"
    )
    parser.add_argument(
        "--task-description",
        type=str,
        default=None,
        help="Task description (auto-detected from HDF5 if not provided)"
    )
    parser.add_argument(
        "--skip-videos",
        action="store_true",
        help="Skip video generation (for testing)"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing output directory"
    )
    parser.add_argument(
        "--unitree-g1-inspire",
        action="store_true",
        help="Convert to Unitree G1 Inspire 26 DOF format (14 arm + 12 hand) for unitree_IL_lerobot"
    )
    return parser.parse_args()


def get_demo_keys(hdf5_file):
    """Get sorted list of demo keys from HDF5 file."""
    data_group = hdf5_file["data"]
    demo_keys = [k for k in data_group.keys() if k.startswith("demo_")]
    demo_keys.sort(key=lambda x: int(x.split("_")[1]))
    return demo_keys


def extract_env_info(hdf5_file):
    """Extract environment info from HDF5 attributes."""
    data_group = hdf5_file["data"]
    env_args_str = data_group.attrs.get("env_args", "{}")
    if isinstance(env_args_str, bytes):
        env_args_str = env_args_str.decode("utf-8")
    env_args = json.loads(env_args_str)
    return env_args


def get_observation_keys(hdf5_file, demo_key):
    """Get list of observation keys from a demo."""
    obs_group = hdf5_file[f"data/{demo_key}/obs"]
    return list(obs_group.keys())


def identify_camera_keys(obs_keys):
    """Identify which observation keys are camera images."""
    camera_keys = []
    for key in obs_keys:
        if any(cam_hint in key.lower() for cam_hint in ["rgb", "depth", "image", "camera"]):
            camera_keys.append(key)
    return camera_keys


def identify_state_keys(obs_keys, camera_keys):
    """Identify non-camera observation keys."""
    return [k for k in obs_keys if k not in camera_keys]


def convert_to_unitree_26dof(actions_38d, processed_actions):
    """Convert IsaacLab 38D format to Unitree G1 Inspire 26D format.

    Args:
        actions_38d: Original actions (14D arm poses + 24D hand joints)
        processed_actions: IK-solved joint positions (first 14 are arm joints)

    Returns:
        26D array: 14 arm joints + 12 hand joints (in radians)
    """
    # Extract 14 arm joints from IK-solved processed_actions
    arm_joints = processed_actions[:, :14]  # 7 left + 7 right

    # Extract and remap 24 hand joints to 12 drive joints
    hand_24 = actions_38d[:, 14:]
    hand_12 = np.zeros((len(hand_24), 12))
    for il_idx, ut_idx in ISAACLAB_TO_UNITREE_HAND.items():
        hand_12[:, ut_idx] = hand_24[:, il_idx]

    return np.concatenate([arm_joints, hand_12], axis=1)  # 26D


def create_directory_structure(output_dir, camera_keys, force=False):
    """Create LeRobot v2.1 directory structure for GR00T."""
    output_path = Path(output_dir).expanduser()

    if output_path.exists():
        if force:
            shutil.rmtree(output_path)
        else:
            raise FileExistsError(
                f"Output directory {output_path} already exists. Use --force to overwrite."
            )

    # Create directories
    (output_path / "meta").mkdir(parents=True)
    (output_path / "data" / "chunk-000").mkdir(parents=True)

    # Create video directories for each camera
    for cam_key in camera_keys:
        video_key = f"observation.images.{cam_key}"
        (output_path / "videos" / "chunk-000" / video_key).mkdir(parents=True)

    return output_path


def encode_video(frames, output_path, fps, codec="libx264", quality=23):
    """Encode frames to MP4 video."""
    if not HAS_IMAGEIO:
        raise RuntimeError("imageio required for video encoding")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Convert float32 [0,1] to uint8 [0,255] if needed
    if frames.dtype == np.float32 or frames.dtype == np.float64:
        frames = (np.clip(frames, 0, 1) * 255).astype(np.uint8)

    # Set output parameters based on codec
    if "av1" in codec:
        output_params = ["-crf", str(quality), "-pix_fmt", "yuv420p"]
    else:
        output_params = ["-crf", str(quality), "-pix_fmt", "yuv420p", "-preset", "fast"]

    iio.imwrite(
        str(output_path),
        frames,
        fps=fps,
        codec=codec,
        output_params=output_params
    )


def convert_episode(
    hdf5_file,
    demo_key,
    episode_idx,
    output_path,
    camera_keys,
    state_keys,
    fps,
    video_codec,
    video_quality,
    skip_videos,
    global_frame_idx_start,
    task_index=0,
    unitree_g1_inspire=False
):
    """Convert a single episode to LeRobot v2.1 format for GR00T."""
    demo_group = hdf5_file[f"data/{demo_key}"]
    obs_group = demo_group["obs"]

    num_frames = demo_group.attrs.get("num_samples", obs_group[state_keys[0]].shape[0])
    is_success = demo_group.attrs.get("success", False)

    # Build frame data with concatenated arrays (GR00T format)
    frame_data = {
        "episode_index": np.full(num_frames, episode_idx, dtype=np.int64),
        "frame_index": np.arange(num_frames, dtype=np.int64),
        "index": np.arange(global_frame_idx_start, global_frame_idx_start + num_frames, dtype=np.int64),
        "timestamp": np.arange(num_frames, dtype=np.float32) / fps,
        "task_index": np.full(num_frames, task_index, dtype=np.int64),
    }

    # Build concatenated observation.state array
    # For G1 Inspire: robot_joint_pos (53)
    state_arrays = []
    if "robot_joint_pos" in state_keys:
        state_arrays.append(obs_group["robot_joint_pos"][:])

    if state_arrays:
        state_concat = np.concatenate(state_arrays, axis=1)
        # Store as list of arrays for parquet
        frame_data["observation.state"] = [state_concat[i].tolist() for i in range(num_frames)]

    # Add other state observations as separate columns if needed
    for key in ["eef_pos", "eef_quat", "cube_positions", "cube_orientations"]:
        if key in state_keys:
            data = obs_group[key][:]
            frame_data[f"observation.{key}"] = [data[i].tolist() for i in range(num_frames)]

    # Build concatenated action array
    if unitree_g1_inspire:
        # Convert to Unitree G1 Inspire 26 DOF format
        if "actions" in demo_group and "processed_actions" in demo_group:
            actions_38d = demo_group["actions"][:]
            processed_actions = demo_group["processed_actions"][:]
            actions = convert_to_unitree_26dof(actions_38d, processed_actions)
        elif "processed_actions" in demo_group:
            # Fallback: use processed_actions directly (already IK-solved)
            processed_actions = demo_group["processed_actions"][:]
            # Try to get hand joints from actions if available
            if "actions" in demo_group:
                actions_38d = demo_group["actions"][:]
                actions = convert_to_unitree_26dof(actions_38d, processed_actions)
            else:
                actions = processed_actions[:, :26]  # Take first 26 if no raw actions
        else:
            actions = None
    else:
        # Original 38 DOF format
        if "actions" in demo_group:
            actions = demo_group["actions"][:]
        elif "processed_actions" in demo_group:
            actions = demo_group["processed_actions"][:]
        else:
            actions = None

    if actions is not None:
        frame_data["action"] = [actions[i].tolist() for i in range(num_frames)]

    # Add next.done column (True only on last frame)
    done = np.zeros(num_frames, dtype=bool)
    done[-1] = True
    frame_data["next.done"] = done

    # Add success indicator
    frame_data["next.success"] = np.full(num_frames, is_success, dtype=bool)

    # Process camera observations
    video_info = {}
    if not skip_videos and camera_keys:
        for cam_key in camera_keys:
            cam_frames = obs_group[cam_key][:]
            video_key = f"observation.images.{cam_key}"

            # Video path: videos/chunk-000/observation.images.<cam>/episode_000000.mp4
            video_path = output_path / "videos" / "chunk-000" / video_key / f"episode_{episode_idx:06d}.mp4"

            encode_video(
                cam_frames,
                video_path,
                fps,
                video_codec,
                video_quality
            )

            # Store video info
            video_info[video_key] = {
                "height": cam_frames.shape[1],
                "width": cam_frames.shape[2],
                "channels": cam_frames.shape[3] if cam_frames.ndim > 3 else 1,
            }

    # Create DataFrame and save as Parquet
    df = pd.DataFrame(frame_data)
    parquet_path = output_path / "data" / "chunk-000" / f"episode_{episode_idx:06d}.parquet"

    table = pa.Table.from_pandas(df, preserve_index=False)
    pq.write_table(table, parquet_path)

    return {
        "episode_index": episode_idx,
        "num_frames": num_frames,
        "length_s": num_frames / fps,
        "is_success": is_success,
        "video_info": video_info,
        "tasks": [task_index],
        "action_dim": actions.shape[1] if actions is not None else 0,
        "state_dim": state_concat.shape[1] if state_arrays else 0,
    }


def compute_statistics(hdf5_file, demo_keys):
    """Compute dataset statistics for normalization."""
    all_actions = []
    all_states = []

    for demo_key in demo_keys:
        demo_group = hdf5_file[f"data/{demo_key}"]
        obs_group = demo_group["obs"]

        # Collect actions
        if "actions" in demo_group:
            all_actions.append(demo_group["actions"][:])
        elif "processed_actions" in demo_group:
            all_actions.append(demo_group["processed_actions"][:])

        # Collect states
        if "robot_joint_pos" in obs_group:
            all_states.append(obs_group["robot_joint_pos"][:])

    stats = {}

    if all_actions:
        actions = np.concatenate(all_actions, axis=0)
        stats["action"] = {
            "mean": actions.mean(axis=0).tolist(),
            "std": actions.std(axis=0).tolist(),
            "min": actions.min(axis=0).tolist(),
            "max": actions.max(axis=0).tolist(),
            "q01": np.percentile(actions, 1, axis=0).tolist(),
            "q99": np.percentile(actions, 99, axis=0).tolist(),
        }

    if all_states:
        states = np.concatenate(all_states, axis=0)
        stats["observation.state"] = {
            "mean": states.mean(axis=0).tolist(),
            "std": states.std(axis=0).tolist(),
            "min": states.min(axis=0).tolist(),
            "max": states.max(axis=0).tolist(),
            "q01": np.percentile(states, 1, axis=0).tolist(),
            "q99": np.percentile(states, 99, axis=0).tolist(),
        }

    return stats


def create_modality_json(output_path, camera_keys, action_dim, state_dim, unitree_g1_inspire=False):
    """Create modality.json for GR00T with joint group mappings."""
    modality = {
        "state": {},
        "action": {},
        "video": {},
        "annotation": {
            "human.task_description": {
                "original_key": "task_index"
            }
        }
    }

    # Map state groups (G1 with 53 DOF)
    if state_dim >= 53:
        for group_name, group_info in G1_JOINT_GROUPS.items():
            modality["state"][group_name] = {
                "start": group_info["start"],
                "end": group_info["end"]
            }
    else:
        # Fallback: treat as single group
        modality["state"]["joints"] = {"start": 0, "end": state_dim}

    # Map action groups
    if unitree_g1_inspire:
        # Use Unitree G1 Inspire 26 DOF action groups
        for group_name, group_info in UNITREE_G1_INSPIRE_ACTION_GROUPS.items():
            modality["action"][group_name] = {
                "start": group_info["start"],
                "end": group_info["end"]
            }
    elif action_dim >= 38:
        # Use original 38 DOF action groups
        for group_name, group_info in G1_ACTION_GROUPS.items():
            modality["action"][group_name] = {
                "start": group_info["start"],
                "end": group_info["end"]
            }
    else:
        # Fallback: treat as single group
        modality["action"]["joints"] = {"start": 0, "end": action_dim}

    # Map camera keys
    for cam_key in camera_keys:
        video_key = f"observation.images.{cam_key}"
        # Create alias without special characters
        alias = cam_key.replace("_", ".")
        modality["video"][alias] = {
            "original_key": video_key
        }

    with open(output_path / "meta" / "modality.json", "w") as f:
        json.dump(modality, f, indent=2)


def numpy_to_python(obj):
    """Convert numpy types to Python native types for JSON serialization."""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: numpy_to_python(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [numpy_to_python(v) for v in obj]
    return obj


def create_metadata_files(
    output_path,
    episode_infos,
    camera_keys,
    fps,
    repo_id,
    robot_type,
    task_description,
    env_info,
    stats,
    unitree_g1_inspire=False
):
    """Create all metadata JSON files for LeRobot v2.1 / GR00T."""
    meta_dir = output_path / "meta"

    total_frames = int(sum(ep["num_frames"] for ep in episode_infos))
    total_episodes = int(len(episode_infos))
    successful_episodes = sum(1 for ep in episode_infos if ep["is_success"])

    action_dim = episode_infos[0]["action_dim"] if episode_infos else 0
    state_dim = episode_infos[0]["state_dim"] if episode_infos else 0

    # Get video info from first episode
    video_info = episode_infos[0].get("video_info", {}) if episode_infos else {}

    # 1. info.json - Main dataset metadata
    features = {
        "episode_index": {"dtype": "int64", "shape": [1], "names": None},
        "frame_index": {"dtype": "int64", "shape": [1], "names": None},
        "index": {"dtype": "int64", "shape": [1], "names": None},
        "timestamp": {"dtype": "float32", "shape": [1], "names": None},
        "task_index": {"dtype": "int64", "shape": [1], "names": None},
        "next.done": {"dtype": "bool", "shape": [1], "names": None},
        "next.success": {"dtype": "bool", "shape": [1], "names": None},
    }

    # Add action feature (concatenated array)
    if action_dim > 0:
        if unitree_g1_inspire:
            # Use Unitree G1 Inspire joint names for 26 DOF
            action_names = [f"{name}.pos" for name in UNITREE_G1_INSPIRE_ACTION_NAMES]
        else:
            # Use original G1 action group names for 38 DOF
            action_names = []
            for group_name, group_info in G1_ACTION_GROUPS.items():
                for i in range(group_info["end"] - group_info["start"]):
                    action_names.append(f"{group_name}.pos.{i}")
        features["action"] = {
            "dtype": "float32",
            "shape": [action_dim],
            "names": action_names[:action_dim]
        }

    # Add state feature (concatenated array)
    if state_dim > 0:
        state_names = []
        for group_name, group_info in G1_JOINT_GROUPS.items():
            for joint in group_info["joints"]:
                state_names.append(f"{joint}.pos")
        features["observation.state"] = {
            "dtype": "float32",
            "shape": [state_dim],
            "names": state_names[:state_dim]
        }

    # Add camera features
    for cam_key in camera_keys:
        video_key = f"observation.images.{cam_key}"
        info = video_info.get(video_key, {"height": 200, "width": 200, "channels": 3})
        features[video_key] = {
            "dtype": "video",
            "shape": [info["height"], info["width"], info["channels"]],
            "names": ["height", "width", "channels"],
            "info": {
                "video.height": info["height"],
                "video.width": info["width"],
                "video.codec": "av1",  # GR00T expects AV1
                "video.pix_fmt": "yuv420p",
                "video.is_depth_map": False,
                "video.fps": fps,
                "video.channels": info["channels"],
                "has_audio": False
            }
        }

    info = {
        "codebase_version": "v2.1",
        "robot_type": robot_type,
        "fps": fps,
        "total_episodes": total_episodes,
        "total_frames": total_frames,
        "total_tasks": 1,
        "total_chunks": 1,
        "chunks_size": 1000,
        "total_videos": len(camera_keys) * total_episodes if camera_keys else 0,
        "splits": {
            "train": f"0:{total_episodes}"
        },
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4" if camera_keys else None,
        "features": features,
    }

    if repo_id:
        info["repo_id"] = repo_id

    # Convert numpy types to Python native types
    info = numpy_to_python(info)

    with open(meta_dir / "info.json", "w") as f:
        json.dump(info, f, indent=2)

    # 2. episodes.jsonl - One line per episode
    task_desc = task_description or env_info.get("env_name", "manipulation_task")
    with open(meta_dir / "episodes.jsonl", "w") as f:
        for ep in episode_infos:
            ep_data = {
                "episode_index": int(ep["episode_index"]),
                "tasks": [task_desc],
                "length": int(ep["num_frames"]),
            }
            f.write(json.dumps(ep_data) + "\n")

    # 3. tasks.jsonl - Task descriptions
    with open(meta_dir / "tasks.jsonl", "w") as f:
        task_data = {
            "task_index": 0,
            "task": task_desc
        }
        f.write(json.dumps(task_data) + "\n")

    # 4. stats.json - Dataset statistics for normalization
    stats = numpy_to_python(stats)
    with open(meta_dir / "stats.json", "w") as f:
        json.dump(stats, f, indent=2)

    # 5. modality.json - GR00T specific modality mapping
    create_modality_json(output_path, camera_keys, action_dim, state_dim, unitree_g1_inspire)

    print(f"\nDataset statistics:")
    print(f"  Total episodes: {total_episodes}")
    print(f"  Total frames: {total_frames}")
    print(f"  Successful episodes: {successful_episodes} ({successful_episodes/total_episodes*100:.1f}%)")
    print(f"  Mean episode length: {total_frames/total_episodes:.1f} frames")
    print(f"  Action dimension: {action_dim}")
    print(f"  State dimension: {state_dim}")


def main():
    args = parse_args()

    print(f"Converting Isaac Lab HDF5 to LeRobot v2.1 format (GR00T compatible)")
    print(f"  Input:  {args.input}")
    print(f"  Output: {args.output}")
    if getattr(args, 'unitree_g1_inspire', False):
        print(f"  Mode:   Unitree G1 Inspire 26 DOF (for unitree_IL_lerobot)")

    # Open HDF5 file
    with h5py.File(args.input, "r") as hdf5_file:
        # Get demo keys and environment info
        demo_keys = get_demo_keys(hdf5_file)
        env_info = extract_env_info(hdf5_file)

        print(f"\nFound {len(demo_keys)} episodes")
        print(f"Environment: {env_info.get('env_name', 'unknown')}")

        # Identify observation types
        obs_keys = get_observation_keys(hdf5_file, demo_keys[0])
        camera_keys = identify_camera_keys(obs_keys)
        state_keys = identify_state_keys(obs_keys, camera_keys)

        print(f"\nCamera observations: {camera_keys}")
        print(f"State observations: {state_keys}")

        # Get dimensions
        unitree_mode = getattr(args, 'unitree_g1_inspire', False)
        if unitree_mode:
            action_dim = 26  # Unitree G1 Inspire: 14 arm + 12 hand
        else:
            action_dim = hdf5_file[f"data/{demo_keys[0]}/actions"].shape[1]
        state_dim = hdf5_file[f"data/{demo_keys[0]}/obs/robot_joint_pos"].shape[1] if "robot_joint_pos" in state_keys else 0
        print(f"Action dimension: {action_dim}" + (" (Unitree 26 DOF)" if unitree_mode else ""))
        print(f"State dimension: {state_dim}")

        # Create output directory
        output_path = create_directory_structure(args.output, camera_keys, args.force)
        print(f"\nCreated output directory: {output_path}")

        # Compute statistics
        print("\nComputing dataset statistics...")
        stats = compute_statistics(hdf5_file, demo_keys)

        # Get task description
        task_desc = args.task_description or env_info.get("env_name", "manipulation_task")

        # Convert episodes
        episode_infos = []
        global_frame_idx = 0

        print(f"\nConverting episodes...")
        for episode_idx, demo_key in enumerate(tqdm(demo_keys, desc="Episodes")):
            ep_info = convert_episode(
                hdf5_file,
                demo_key,
                episode_idx,
                output_path,
                camera_keys,
                state_keys,
                args.fps,
                args.video_codec,
                args.video_quality,
                args.skip_videos,
                global_frame_idx,
                task_index=0,
                unitree_g1_inspire=unitree_mode
            )
            episode_infos.append(ep_info)
            global_frame_idx += ep_info["num_frames"]

        # Create metadata files
        print("\nCreating metadata files...")
        create_metadata_files(
            output_path,
            episode_infos,
            camera_keys,
            args.fps,
            args.repo_id,
            args.robot_type,
            task_desc,
            env_info,
            stats,
            unitree_g1_inspire=unitree_mode
        )

    print(f"\n{'='*60}")
    print(f"Conversion complete!")
    print(f"Output: {output_path}")
    if unitree_mode:
        print(f"\nTo use with unitree_IL_lerobot:")
        print(f"  cd ~/Bot/unitree/unitree_IL_lerobot")
        print(f"  python train.py --dataset {output_path}")
    else:
        print(f"\nTo use with GR00T:")
        print(f"  # Verify with:")
        print(f"  python -c \"from gr00t.data.dataset import LeRobotSingleDataset; "
              f"ds = LeRobotSingleDataset('{output_path}', 'UNITREE_G1'); print(len(ds))\"")


if __name__ == "__main__":
    main()
