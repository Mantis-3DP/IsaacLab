#!/usr/bin/env python3
"""
Convert StreamingRecorder JPEG/JSON locomanipulation dataset to LeRobot v2.1 format.

Matches the NVIDIA GR00T-WBC reference format (PhysicalAI-Robotics-GR00T-X-Embodiment-Sim)
with 43D state/action, 14D EEF, navigate_command, and base_height_command.

Input:  Streaming JPEG/JSON from Isaac-PickPlace-Locomanipulation-Camera-G1-Abs-v0
Output: LeRobot v2.1 dataset compatible with GR00T fine-tuning

Usage:
    python convert_streaming_to_lerobot_locomanip.py \
        --input ~/Bot/Datasets/locomanip_camera_json_jpeg \
        --output ~/Bot/Datasets/PhysicalAI-Robotics-GR00T-X-Embodiment-Sim/unitree_g1.LocomanipPickPlace \
        --task-description "pick up the object, walk, and place it on the table"
"""

import argparse
import json
import os
import shutil
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image
from scipy.spatial.transform import Rotation
from tqdm import tqdm

try:
    import imageio.v3 as iio
    HAS_IMAGEIO = True
except ImportError:
    HAS_IMAGEIO = False
    print("Warning: imageio not found. Install with: pip install imageio[ffmpeg]")


# --- Joint index maps: G1 URDF (interleaved) → NVIDIA grouped format ---
# Scene state joint_position is 43D in URDF order (interleaved left/right).
# We de-interleave into NVIDIA's body-grouped order.

# URDF indices for each body group (verified by cross-referencing grouped states)
SCENE_IDX_LEFT_LEG = [0, 3, 6, 9, 13, 17]      # hip_pitch, hip_roll, hip_yaw, knee, ankle_pitch, ankle_roll
SCENE_IDX_RIGHT_LEG = [1, 4, 7, 10, 14, 18]
SCENE_IDX_WAIST = [2, 5, 8]                      # yaw, roll, pitch
SCENE_IDX_LEFT_ARM = [11, 15, 19, 21, 23, 25, 27]
SCENE_IDX_RIGHT_ARM = [12, 16, 20, 22, 24, 26, 28]
SCENE_IDX_LEFT_HAND = [29, 30, 31, 35, 36, 37, 41]
SCENE_IDX_RIGHT_HAND = [32, 33, 34, 38, 39, 40, 42]

# NVIDIA 43D state layout
# [0:6]   left_leg
# [6:12]  right_leg
# [12:15] waist
# [15:22] left_arm
# [22:29] left_hand
# [29:36] right_arm
# [36:43] right_hand

# Joint names matching NVIDIA format (for info.json)
STATE_JOINT_NAMES = [
    # Left leg (6)
    "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint",
    "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
    # Right leg (6)
    "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint",
    "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
    # Waist (3)
    "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",
    # Left arm (7)
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint",
    "left_elbow_joint", "left_wrist_roll_joint", "left_wrist_pitch_joint", "left_wrist_yaw_joint",
    # Left hand (7) — TriHand joints
    "left_hand_joint_0", "left_hand_joint_1", "left_hand_joint_2",
    "left_hand_joint_3", "left_hand_joint_4", "left_hand_joint_5", "left_hand_joint_6",
    # Right arm (7)
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint",
    "right_elbow_joint", "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint",
    # Right hand (7)
    "right_hand_joint_0", "right_hand_joint_1", "right_hand_joint_2",
    "right_hand_joint_3", "right_hand_joint_4", "right_hand_joint_5", "right_hand_joint_6",
]

# Camera mapping
CAMERA_MAPPING = {
    "color_0": "observation.images.cam_left_high",
    "color_1": "observation.images.cam_right_high",
}
VIDEO_MODALITY_KEYS = {
    "color_0": "cam_left_high",
    "color_1": "cam_right_high",
}


def list_episodes(input_dir: Path) -> list[Path]:
    episodes = []
    for entry in sorted(input_dir.iterdir()):
        if entry.is_dir() and entry.name.startswith("episode_"):
            if (entry / "data.json").exists():
                episodes.append(entry)
    return episodes


def load_episode(episode_dir: Path) -> dict | None:
    with open(episode_dir / "data.json", "r") as f:
        content = f.read()
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


def extract_43d_state(states: dict) -> np.ndarray:
    """Build 43D state in NVIDIA grouped order from scene joint_position."""
    scene = states.get("scene", {})
    robot = scene.get("articulation", {}).get("robot", {})
    jp = robot.get("joint_position", [])

    if len(jp) == 43:
        # De-interleave URDF order → NVIDIA grouped order
        state = np.zeros(43, dtype=np.float64)
        jp = np.array(jp)
        state[0:6] = jp[SCENE_IDX_LEFT_LEG]
        state[6:12] = jp[SCENE_IDX_RIGHT_LEG]
        state[12:15] = jp[SCENE_IDX_WAIST]
        state[15:22] = jp[SCENE_IDX_LEFT_ARM]
        state[22:29] = jp[SCENE_IDX_LEFT_HAND]
        state[29:36] = jp[SCENE_IDX_RIGHT_ARM]
        state[36:43] = jp[SCENE_IDX_RIGHT_HAND]
        return state

    # Fallback: build from grouped data (no legs)
    state = np.zeros(43, dtype=np.float64)
    if "torso" in states:
        state[12:15] = states["torso"]["qpos"]
    if "left_arm" in states:
        state[15:22] = states["left_arm"]["qpos"]
    if "left_hand" in states:
        state[22:29] = states["left_hand"]["qpos"]
    if "right_arm" in states:
        state[29:36] = states["right_arm"]["qpos"]
    if "right_hand" in states:
        state[36:43] = states["right_hand"]["qpos"]
    return state


def quat_wxyz_to_rot6d(quat_wxyz: np.ndarray) -> np.ndarray:
    """Convert quaternion (w,x,y,z) to 6D rotation (first two rows of rotation matrix)."""
    r = Rotation.from_quat([quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]])
    mat = r.as_matrix()
    return mat[:2, :].flatten()  # 6D


def extract_18d_eef_state(states: dict) -> np.ndarray:
    """Build 18D EEF state: [left_xyz(3)+rot6d(6), right_xyz(3)+rot6d(6)]."""
    eef = np.zeros(18, dtype=np.float64)
    if "left_wrist_pos" in states:
        eef[0:3] = states["left_wrist_pos"]["qpos"]
    if "left_wrist_quat" in states:
        eef[3:9] = quat_wxyz_to_rot6d(np.array(states["left_wrist_quat"]["qpos"]))
    if "right_wrist_pos" in states:
        eef[9:12] = states["right_wrist_pos"]["qpos"]
    if "right_wrist_quat" in states:
        eef[12:18] = quat_wxyz_to_rot6d(np.array(states["right_wrist_quat"]["qpos"]))
    return eef


def extract_43d_action(actions: dict, states: dict) -> np.ndarray:
    """Build 43D action in NVIDIA order.

    Upper body from action commands, legs from observed state (Agile RL output).
    """
    action = np.zeros(43, dtype=np.float64)

    # Legs: copy from state (Agile RL controlled these, not operator)
    scene = states.get("scene", {})
    robot = scene.get("articulation", {}).get("robot", {})
    jp = robot.get("joint_position", [])
    if len(jp) == 43:
        jp = np.array(jp)
        action[0:6] = jp[SCENE_IDX_LEFT_LEG]
        action[6:12] = jp[SCENE_IDX_RIGHT_LEG]

    # Upper body from action commands
    if "torso" in actions:
        action[12:15] = actions["torso"]["qpos"]
    if "left_arm" in actions:
        action[15:22] = actions["left_arm"]["qpos"]
    if "left_hand" in actions:
        action[22:29] = actions["left_hand"]["qpos"]
    if "right_arm" in actions:
        action[29:36] = actions["right_arm"]["qpos"]
    if "right_hand" in actions:
        action[36:43] = actions["right_hand"]["qpos"]
    return action


def extract_18d_action_eef(raw_actions: list) -> np.ndarray:
    """Extract 18D EEF action from raw_actions, converting quat→rot6d.

    raw_actions layout: [left_wrist(7: xyz+quat_wxyz), right_wrist(7), hands(14), loco(4)]
    Output: [left_xyz(3)+rot6d(6), right_xyz(3)+rot6d(6)]
    """
    eef = np.zeros(18, dtype=np.float64)
    if len(raw_actions) >= 14:
        ra = np.array(raw_actions, dtype=np.float64)
        eef[0:3] = ra[0:3]       # left xyz
        eef[3:9] = quat_wxyz_to_rot6d(ra[3:7])   # left quat → rot6d
        eef[9:12] = ra[7:10]     # right xyz
        eef[12:18] = quat_wxyz_to_rot6d(ra[10:14])  # right quat → rot6d
    return eef


def extract_navigate_command(raw_actions: list) -> np.ndarray:
    """Extract [vx, vy, wz] from raw_actions[28:31]."""
    if len(raw_actions) >= 31:
        return np.array(raw_actions[28:31], dtype=np.float64)
    return np.zeros(3, dtype=np.float64)


def extract_base_height_command(raw_actions: list) -> float:
    """Extract hip_height scalar from raw_actions[31]."""
    if len(raw_actions) >= 32:
        return float(raw_actions[31])
    return 0.0


def encode_video(frames, output_path: Path, fps: int, codec: str = "libx264", quality: int = 23):
    if not HAS_IMAGEIO:
        raise RuntimeError("imageio required. pip install imageio[ffmpeg]")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not frames:
        return
    frames_array = np.stack(frames, axis=0)
    iio.imwrite(
        str(output_path), frames_array, fps=fps, codec=codec,
        output_params=["-crf", str(quality), "-pix_fmt", "yuv420p", "-preset", "fast"],
    )


def convert_episode(
    episode_dir: Path, episode_idx: int, output_path: Path,
    fps: int, video_codec: str, video_quality: int, skip_videos: bool,
    global_frame_idx_start: int, task_index: int = 0,
) -> tuple[dict, int, set]:
    episode_data = load_episode(episode_dir)
    if episode_data is None:
        return {}, 0, set()

    data_items = episode_data.get("data", [])
    num_frames = len(data_items)
    if num_frames == 0:
        return {}, 0, set()

    # Frame metadata
    frame_data = {
        "episode_index": np.full(num_frames, episode_idx, dtype=np.int64),
        "frame_index": np.arange(num_frames, dtype=np.int64),
        "index": np.arange(global_frame_idx_start, global_frame_idx_start + num_frames, dtype=np.int64),
        "timestamp": np.arange(num_frames, dtype=np.float32) / fps,
        "task_index": np.full(num_frames, task_index, dtype=np.int64),
    }

    # Collect per-frame data
    states_43_list = []
    eef_states_list = []
    actions_43_list = []
    action_eef_list = []
    nav_cmd_list = []
    height_cmd_list = []
    camera_frames = {}
    camera_keys_used = set()

    for item in data_items:
        states = item.get("states", {})
        actions = item.get("actions", {})
        raw_actions = item.get("raw_actions", [])

        states_43_list.append(extract_43d_state(states).tolist())
        eef_states_list.append(extract_18d_eef_state(states).tolist())
        actions_43_list.append(extract_43d_action(actions, states).tolist())
        action_eef_list.append(extract_18d_action_eef(raw_actions).tolist())
        nav_cmd_list.append(extract_navigate_command(raw_actions).tolist())
        height_cmd_list.append(extract_base_height_command(raw_actions))

        # Load images
        if not skip_videos:
            for cam_key, rel_path in item.get("colors", {}).items():
                camera_keys_used.add(cam_key)
                if cam_key not in camera_frames:
                    camera_frames[cam_key] = []
                img_path = episode_dir / rel_path
                if img_path.exists():
                    camera_frames[cam_key].append(np.array(Image.open(img_path).convert("RGB")))

    frame_data["observation.state"] = states_43_list
    frame_data["observation.eef_state"] = eef_states_list
    frame_data["action"] = actions_43_list
    frame_data["action.eef"] = action_eef_list
    frame_data["teleop.navigate_command"] = nav_cmd_list
    frame_data["teleop.base_height_command"] = height_cmd_list

    # Encode videos
    if not skip_videos:
        for cam_key, frames in camera_frames.items():
            if frames:
                video_key = CAMERA_MAPPING.get(cam_key, f"observation.images.{cam_key}")
                video_path = output_path / "videos" / "chunk-000" / video_key / f"episode_{episode_idx:06d}.mp4"
                encode_video(frames, video_path, fps=fps, codec=video_codec, quality=video_quality)

    return frame_data, num_frames, camera_keys_used


def compute_stats(arrays: np.ndarray) -> dict:
    return {
        "mean": arrays.mean(axis=0).tolist(),
        "std": arrays.std(axis=0).tolist(),
        "min": arrays.min(axis=0).tolist(),
        "max": arrays.max(axis=0).tolist(),
        "q01": np.percentile(arrays, 1, axis=0).tolist(),
        "q99": np.percentile(arrays, 99, axis=0).tolist(),
    }


def main():
    parser = argparse.ArgumentParser(description="Convert locomanipulation streaming data to NVIDIA LeRobot v2.1")
    parser.add_argument("--input", "-i", type=str, required=True)
    parser.add_argument("--output", "-o", type=str, required=True)
    parser.add_argument("--task-description", type=str, default="pick up the object and place it on the table")
    parser.add_argument("--fps", type=int, default=20, help="Recording FPS (default: 20)")
    parser.add_argument("--video-codec", type=str, default="libx264")
    parser.add_argument("--video-quality", type=int, default=23)
    parser.add_argument("--skip-videos", action="store_true")
    parser.add_argument("--force", "-f", action="store_true")
    parser.add_argument("--max-episodes", type=int, default=None)
    args = parser.parse_args()

    input_path = Path(args.input).expanduser()
    output_path = Path(args.output).expanduser()

    episodes = list_episodes(input_path)
    print(f"Found {len(episodes)} episodes in {input_path}")
    if args.max_episodes:
        episodes = episodes[:args.max_episodes]

    if not episodes:
        raise ValueError("No valid episodes found")

    # Create output structure
    if output_path.exists():
        if args.force:
            shutil.rmtree(output_path)
        else:
            raise FileExistsError(f"{output_path} exists. Use --force to overwrite.")

    (output_path / "meta").mkdir(parents=True)
    (output_path / "data" / "chunk-000").mkdir(parents=True)

    # Scan first episode for camera keys
    first_ep = None
    for ep in episodes:
        first_ep = load_episode(ep)
        if first_ep is not None:
            break
    camera_keys = list(first_ep["data"][0].get("colors", {}).keys())
    for cam_key in camera_keys:
        video_key = CAMERA_MAPPING.get(cam_key, f"observation.images.{cam_key}")
        (output_path / "videos" / "chunk-000" / video_key).mkdir(parents=True)

    # Convert episodes
    global_frame_idx = 0
    total_frames = 0
    valid_ep_idx = 0
    all_states, all_eef, all_actions, all_action_eef, all_nav, all_height = [], [], [], [], [], []

    for episode_dir in tqdm(episodes, desc="Converting"):
        frame_data, num_frames, _ = convert_episode(
            episode_dir, valid_ep_idx, output_path, args.fps,
            args.video_codec, args.video_quality, args.skip_videos,
            global_frame_idx, task_index=0,
        )
        if num_frames > 0:
            df = pd.DataFrame(frame_data)
            pq.write_table(
                pa.Table.from_pandas(df),
                output_path / "data" / "chunk-000" / f"episode_{valid_ep_idx:06d}.parquet",
            )
            global_frame_idx += num_frames
            total_frames += num_frames

            all_states.append(np.array(frame_data["observation.state"]))
            all_eef.append(np.array(frame_data["observation.eef_state"]))
            all_actions.append(np.array(frame_data["action"]))
            all_action_eef.append(np.array(frame_data["action.eef"]))
            all_nav.append(np.array(frame_data["teleop.navigate_command"]))
            all_height.append(np.array(frame_data["teleop.base_height_command"]).reshape(-1, 1))
            valid_ep_idx += 1

    print(f"Wrote {valid_ep_idx} episodes, {total_frames} frames")

    # --- Write metadata ---

    # embodiment.json
    with open(output_path / "meta" / "embodiment.json", "w") as f:
        json.dump({
            "record_frequency": float(args.fps),
            "robot_name": "G1",
            "robot_type": "humanoid",
            "embodiment_tag": "unitree_g1_full_body_with_height_nav_cmd_sim",
        }, f, indent=4)

    # tasks.jsonl
    with open(output_path / "meta" / "tasks.jsonl", "w") as f:
        f.write(json.dumps({"task_index": 0, "task": args.task_description}) + "\n")

    # episodes.jsonl
    frame_start = 0
    with open(output_path / "meta" / "episodes.jsonl", "w") as f:
        for i in range(valid_ep_idx):
            ep_len = len(all_states[i])
            f.write(json.dumps({
                "episode_index": i,
                "tasks": [args.task_description],
                "length": ep_len,
            }) + "\n")
            frame_start += ep_len

    # TODO(human): Configure video features for your camera setup
    # info.json
    info = {
        "codebase_version": "v2.1",
        "robot_type": "humanoid",
        "total_episodes": valid_ep_idx,
        "total_frames": total_frames,
        "total_tasks": 1,
        "total_videos": len(camera_keys) * valid_ep_idx,
        "total_chunks": 1,
        "chunks_size": 1000,
        "fps": float(args.fps),
        "splits": {"train": f"0:{valid_ep_idx}"},
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
        "features": {
            "observation.state": {
                "dtype": "float64", "shape": [43], "names": STATE_JOINT_NAMES,
            },
            "observation.eef_state": {
                "dtype": "float64", "shape": [18],
                "names": ["left_xyz", "left_rot6d", "right_xyz", "right_rot6d"],
            },
            "action": {
                "dtype": "float64", "shape": [43], "names": STATE_JOINT_NAMES,
            },
            "action.eef": {
                "dtype": "float64", "shape": [18],
                "names": ["left_xyz", "left_rot6d", "right_xyz", "right_rot6d"],
            },
            "teleop.navigate_command": {
                "dtype": "float64", "shape": [3], "names": ["lin_vel_x", "lin_vel_y", "ang_vel_z"],
            },
            "teleop.base_height_command": {
                "dtype": "float64", "shape": [1], "names": "base_height_command",
            },
            "timestamp": {"dtype": "float32", "shape": [1], "names": None},
            "frame_index": {"dtype": "int64", "shape": [1], "names": None},
            "episode_index": {"dtype": "int64", "shape": [1], "names": None},
            "index": {"dtype": "int64", "shape": [1], "names": None},
            "task_index": {"dtype": "int64", "shape": [1], "names": None},
        },
    }
    for cam_key in camera_keys:
        video_key = CAMERA_MAPPING.get(cam_key, f"observation.images.{cam_key}")
        info["features"][video_key] = {
            "dtype": "video", "shape": [480, 640, 3],
            "names": ["height", "width", "channel"],
            "video_info": {
                "video.fps": float(args.fps), "video.codec": "h264",
                "video.pix_fmt": "yuv420p", "video.is_depth_map": False, "has_audio": False,
            },
        }
    with open(output_path / "meta" / "info.json", "w") as f:
        json.dump(info, f, indent=4)

    # modality.json — keys MUST match embodiment config modality_keys exactly
    # Data is now stored as rot6d (9D per wrist) in parquet
    modality = {
        "state": {
            "left_leg":   {"original_key": "observation.state", "start": 0,  "end": 6},
            "right_leg":  {"original_key": "observation.state", "start": 6,  "end": 12},
            "waist":      {"original_key": "observation.state", "start": 12, "end": 15},
            "left_arm":   {"original_key": "observation.state", "start": 15, "end": 22},
            "left_hand":  {"original_key": "observation.state", "start": 22, "end": 29},
            "right_arm":  {"original_key": "observation.state", "start": 29, "end": 36},
            "right_hand": {"original_key": "observation.state", "start": 36, "end": 43},
            "left_eef":   {"original_key": "observation.eef_state", "start": 0,  "end": 9},
            "right_eef":  {"original_key": "observation.eef_state", "start": 9,  "end": 18},
        },
        "action": {
            "left_eef":           {"original_key": "action.eef", "start": 0,  "end": 9},
            "right_eef":          {"original_key": "action.eef", "start": 9,  "end": 18},
            "left_hand":          {"original_key": "action", "start": 22, "end": 29},
            "right_hand":         {"original_key": "action", "start": 36, "end": 43},
            "navigate_command":   {"original_key": "teleop.navigate_command",    "start": 0, "end": 3},
            "base_height_command":{"original_key": "teleop.base_height_command", "start": 0, "end": 1},
        },
        "video": {},
        "annotation": {
            "human.task_description": {"original_key": "task_index"}
        },
    }
    for cam_key in camera_keys:
        video_key = CAMERA_MAPPING.get(cam_key, f"observation.images.{cam_key}")
        mod_key = VIDEO_MODALITY_KEYS.get(cam_key, cam_key)
        modality["video"][mod_key] = {"original_key": video_key}

    with open(output_path / "meta" / "modality.json", "w") as f:
        json.dump(modality, f, indent=4)

    # stats.json
    if all_states:
        cat_states = np.concatenate(all_states)
        cat_eef = np.concatenate(all_eef)
        cat_actions = np.concatenate(all_actions)
        cat_action_eef = np.concatenate(all_action_eef)
        cat_nav = np.concatenate(all_nav)
        cat_height = np.concatenate(all_height)

        stats = {
            "observation.state": compute_stats(cat_states),
            "observation.eef_state": compute_stats(cat_eef),
            "action": compute_stats(cat_actions),
            "action.eef": compute_stats(cat_action_eef),
            "teleop.navigate_command": compute_stats(cat_nav),
            "teleop.base_height_command": compute_stats(cat_height),
        }
        with open(output_path / "meta" / "stats.json", "w") as f:
            json.dump(stats, f, indent=2)

    print(f"\nDone! {valid_ep_idx} episodes, {total_frames} frames → {output_path}")


if __name__ == "__main__":
    main()
