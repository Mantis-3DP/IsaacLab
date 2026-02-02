#!/usr/bin/env python3
"""
Convert StreamingRecorder JPEG/JSON dataset to LeRobot v2.1 format for GR00T.

This script converts recorded demonstrations from the streaming format
(JPEG images + JSON state/action data) to LeRobot v2.1 format required by GR00T.

Output format:
- Parquet files with concatenated action/state arrays (26 DOF for G1 Inspire)
- MP4 videos for camera observations
- JSON metadata files including modality.json for GR00T

Usage:
    python convert_streaming_to_lerobot.py \
        --input /path/to/streaming_dataset \
        --output ~/lerobot_datasets/my_dataset \
        --task-description "Pick cube and place in target zone"

Requirements:
    pip install pyarrow pandas imageio[ffmpeg] tqdm pillow
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
from tqdm import tqdm

# Try to import imageio for video encoding
try:
    import imageio.v3 as iio
    HAS_IMAGEIO = True
except ImportError:
    HAS_IMAGEIO = False
    print("Warning: imageio not found. Install with: pip install imageio[ffmpeg]")


# G1 Inspire 26 DOF layout matching the modality config
UNITREE_G1_INSPIRE_LAYOUT = {
    "left_arm": {"start": 0, "end": 7},
    "right_arm": {"start": 7, "end": 14},
    "left_hand": {"start": 14, "end": 20},
    "right_hand": {"start": 20, "end": 26},
}

# Camera mapping from streaming format to LeRobot
CAMERA_MAPPING = {
    "color_0": "observation.images.color_0",  # head_rgb_left
    "color_1": "observation.images.color_1",  # head_rgb_right
    "color_2": "observation.images.color_2",  # cam_left_wrist
    "color_3": "observation.images.color_3",  # cam_right_wrist
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
    """Load episode data from streaming format.

    Returns None if episode is empty or invalid.
    """
    data_json = episode_dir / "data.json"

    with open(data_json, "r") as f:
        content = f.read()

    # Fix Python None -> JSON null (streaming writer bug)
    content = content.replace('"success": none', '"success": null')
    content = content.replace('"success": None', '"success": null')

    try:
        episode_data = json.loads(content)
    except json.JSONDecodeError as e:
        print(f"Warning: Invalid JSON in {episode_dir}: {e}")
        return None

    # Check if episode has actual data
    if "data" not in episode_data or len(episode_data.get("data", [])) == 0:
        print(f"Warning: Empty episode {episode_dir.name}, skipping")
        return None

    return episode_data


def extract_26dof_state(states: dict) -> np.ndarray:
    """Extract 26 DOF state vector from streaming format.

    Expected format from StreamingRecorder:
    {
        "left_arm": {"qpos": [7 floats]},
        "right_arm": {"qpos": [7 floats]},
        "left_ee": {"qpos": [6 floats]},
        "right_ee": {"qpos": [6 floats]}
    }
    """
    state_26 = np.zeros(26, dtype=np.float32)

    # Left arm (7 DOF)
    if "left_arm" in states and "qpos" in states["left_arm"]:
        qpos = states["left_arm"]["qpos"]
        if len(qpos) == 7:
            state_26[0:7] = qpos

    # Right arm (7 DOF)
    if "right_arm" in states and "qpos" in states["right_arm"]:
        qpos = states["right_arm"]["qpos"]
        if len(qpos) == 7:
            state_26[7:14] = qpos

    # Left hand (6 DOF) - stored as "left_ee" in streaming format
    if "left_ee" in states and "qpos" in states["left_ee"]:
        qpos = states["left_ee"]["qpos"]
        if len(qpos) == 6:
            state_26[14:20] = qpos

    # Right hand (6 DOF) - stored as "right_ee" in streaming format
    if "right_ee" in states and "qpos" in states["right_ee"]:
        qpos = states["right_ee"]["qpos"]
        if len(qpos) == 6:
            state_26[20:26] = qpos

    return state_26


def extract_26dof_action(actions: dict) -> np.ndarray:
    """Extract 26 DOF action vector from streaming format.

    Same format as state extraction.
    """
    return extract_26dof_state(actions)  # Same structure


def create_directory_structure(output_path: Path, camera_keys: list[str], force: bool = False):
    """Create LeRobot v2.1 directory structure."""
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
        video_key = CAMERA_MAPPING.get(cam_key, f"observation.images.{cam_key}")
        (output_path / "videos" / "chunk-000" / video_key).mkdir(parents=True)

    return output_path


def encode_video(frames: list[np.ndarray], output_path: Path, fps: int = 30, codec: str = "libx264", quality: int = 23):
    """Encode frames to MP4 video."""
    if not HAS_IMAGEIO:
        raise RuntimeError("imageio required for video encoding. Install with: pip install imageio[ffmpeg]")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not frames:
        return

    frames_array = np.stack(frames, axis=0)

    output_params = ["-crf", str(quality), "-pix_fmt", "yuv420p", "-preset", "fast"]

    iio.imwrite(
        str(output_path),
        frames_array,
        fps=fps,
        codec=codec,
        output_params=output_params
    )


def convert_episode(
    episode_dir: Path,
    episode_idx: int,
    output_path: Path,
    fps: int,
    video_codec: str,
    video_quality: int,
    skip_videos: bool,
    global_frame_idx_start: int,
    task_index: int = 0,
    use_next_state_as_action: bool = False,
) -> tuple[dict, int, set]:
    """Convert a single episode to LeRobot v2.1 format.

    Returns:
        (frame_data_dict, num_frames, camera_keys_used)
    """
    episode_data = load_episode(episode_dir)

    # Skip invalid/empty episodes
    if episode_data is None:
        return {}, 0, set()

    data_items = episode_data.get("data", [])
    num_frames = len(data_items)
    is_success = episode_data.get("success", None)

    if num_frames == 0:
        return {}, 0, set()

    # Build frame data
    frame_data = {
        "episode_index": np.full(num_frames, episode_idx, dtype=np.int64),
        "frame_index": np.arange(num_frames, dtype=np.int64),
        "index": np.arange(global_frame_idx_start, global_frame_idx_start + num_frames, dtype=np.int64),
        "timestamp": np.arange(num_frames, dtype=np.float32) / fps,
        "task_index": np.full(num_frames, task_index, dtype=np.int64),
        # GR00T expects annotation columns
        "annotation.human.task_description": np.full(num_frames, task_index, dtype=np.int64),
        "next.done": np.zeros(num_frames, dtype=bool),
        "next.reward": np.zeros(num_frames, dtype=np.float32),
    }
    # Mark last frame as done
    frame_data["next.done"][-1] = True

    # Collect states and actions
    states_list = []
    actions_list = []

    # Collect camera frames
    camera_frames = {}  # cam_key -> list of frames
    camera_keys_used = set()

    for item in data_items:
        # Extract state
        states = item.get("states", {})
        state_26 = extract_26dof_state(states)
        states_list.append(state_26.tolist())

        # Extract action
        actions = item.get("actions", {})
        action_26 = extract_26dof_action(actions)
        actions_list.append(action_26.tolist())

        # Load images
        if not skip_videos:
            colors_paths = item.get("colors", {})
            for cam_key, rel_path in colors_paths.items():
                camera_keys_used.add(cam_key)

                if cam_key not in camera_frames:
                    camera_frames[cam_key] = []

                # Load image
                img_path = episode_dir / rel_path
                if img_path.exists():
                    img = np.array(Image.open(img_path).convert("RGB"))
                    camera_frames[cam_key].append(img)
                else:
                    print(f"Warning: Image not found: {img_path}")

    frame_data["observation.state"] = states_list

    # Optionally shift actions: action[t] = state[t+1] for RELATIVE training
    # This ensures the model learns actual motion deltas instead of near-zero deltas
    if use_next_state_as_action and len(states_list) > 1:
        # action[t] = state[t+1], last action = last state (repeat)
        shifted_actions = states_list[1:] + [states_list[-1]]
        frame_data["action"] = shifted_actions
    else:
        frame_data["action"] = actions_list

    # Encode videos
    if not skip_videos:
        for cam_key, frames in camera_frames.items():
            if frames:
                video_key = CAMERA_MAPPING.get(cam_key, f"observation.images.{cam_key}")
                video_path = output_path / "videos" / "chunk-000" / video_key / f"episode_{episode_idx:06d}.mp4"
                encode_video(frames, video_path, fps=fps, codec=video_codec, quality=video_quality)

                # Add video path references to frame data
                col_name = f"{video_key}.video_path"
                if col_name not in frame_data:
                    frame_data[col_name] = []

                video_rel_path = f"videos/chunk-000/{video_key}/episode_{episode_idx:06d}.mp4"
                frame_data[col_name] = [video_rel_path] * num_frames

    return frame_data, num_frames, camera_keys_used


def main():
    parser = argparse.ArgumentParser(description="Convert streaming dataset to LeRobot v2.1 format")
    parser.add_argument("--input", "-i", type=str, required=True, help="Input streaming dataset directory")
    parser.add_argument("--output", "-o", type=str, required=True, help="Output LeRobot dataset directory")
    parser.add_argument("--task-description", type=str, default="Pick and place task", help="Task description for annotation")
    parser.add_argument("--fps", type=int, default=30, help="Video FPS (default: 30)")
    parser.add_argument("--video-codec", type=str, default="libx264", help="Video codec (default: libx264)")
    parser.add_argument("--video-quality", type=int, default=23, help="Video CRF quality (default: 23, lower=better)")
    parser.add_argument("--skip-videos", action="store_true", help="Skip video encoding (for testing)")
    parser.add_argument("--force", "-f", action="store_true", help="Overwrite existing output")
    parser.add_argument("--max-episodes", type=int, default=None, help="Limit number of episodes (for testing)")
    parser.add_argument("--use-next-state-as-action", action="store_true",
                        help="Use state[t+1] as action[t] for RELATIVE action training. "
                             "This ensures action-state deltas represent actual motion.")
    args = parser.parse_args()

    input_path = Path(args.input).expanduser()
    output_path = Path(args.output).expanduser()

    if not input_path.exists():
        raise FileNotFoundError(f"Input directory not found: {input_path}")

    # Find episodes
    episodes = list_episodes(input_path)
    print(f"Found {len(episodes)} episodes in {input_path}")

    if args.max_episodes:
        episodes = episodes[:args.max_episodes]
        print(f"Limiting to {len(episodes)} episodes")

    if not episodes:
        raise ValueError("No valid episodes found")

    # Scan first valid episode to determine camera keys
    first_episode = None
    for ep in episodes:
        first_episode = load_episode(ep)
        if first_episode is not None:
            break

    if first_episode is None:
        raise ValueError("No valid episodes with data found")

    first_item = first_episode.get("data", [{}])[0]
    camera_keys = list(first_item.get("colors", {}).keys())
    print(f"Found cameras: {camera_keys}")

    # Create output structure
    create_directory_structure(output_path, camera_keys, force=args.force)

    # Convert episodes - write individual parquet files per episode
    global_frame_idx = 0
    total_frames = 0
    successful_episodes = 0
    valid_episode_idx = 0

    for ep_idx, episode_dir in enumerate(tqdm(episodes, desc="Converting episodes")):
        frame_data, num_frames, _ = convert_episode(
            episode_dir=episode_dir,
            episode_idx=valid_episode_idx,  # Use valid episode index
            output_path=output_path,
            fps=args.fps,
            video_codec=args.video_codec,
            video_quality=args.video_quality,
            skip_videos=args.skip_videos,
            global_frame_idx_start=global_frame_idx,
            task_index=0,
            use_next_state_as_action=args.use_next_state_as_action,
        )

        if num_frames > 0:
            # Write individual parquet file for this episode
            df = pd.DataFrame(frame_data)
            parquet_path = output_path / "data" / "chunk-000" / f"episode_{valid_episode_idx:06d}.parquet"
            pq.write_table(pa.Table.from_pandas(df), parquet_path)

            global_frame_idx += num_frames
            total_frames += num_frames

            # Check success from episode
            episode_data = load_episode(episode_dir)
            if episode_data is not None and episode_data.get("success", None) is True:
                successful_episodes += 1

            valid_episode_idx += 1

    print(f"Wrote {valid_episode_idx} parquet files with {total_frames} total frames")

    # Write modality.json - keys must match the Python modality config
    # Video keys: head_rgb_left, head_rgb_right (mapped from color_0, color_1)
    video_key_mapping = {
        "color_0": "cam_left",
        "color_1": "cam_right",
        "color_2": "cam_left_wrist",
        "color_3": "cam_right_wrist",
    }

    modality = {
        "state": {
            "left_arm": {"start": 0, "end": 7},
            "right_arm": {"start": 7, "end": 14},
            "left_hand": {"start": 14, "end": 20},
            "right_hand": {"start": 20, "end": 26},
        },
        "action": {
            "left_arm": {"start": 0, "end": 7},
            "right_arm": {"start": 7, "end": 14},
            "left_hand": {"start": 14, "end": 20},
            "right_hand": {"start": 20, "end": 26},
        },
        "video": {},
        "annotation": {
            "human.task_description": {
                "original_key": "task_index"
            }
        }
    }

    # Add video modalities with proper key names
    for cam_key in camera_keys:
        video_key = CAMERA_MAPPING.get(cam_key, f"observation.images.{cam_key}")
        # Use GR00T-compatible keys (head_rgb_left, head_rgb_right, etc.)
        modality_key = video_key_mapping.get(cam_key, cam_key)
        modality["video"][modality_key] = {"original_key": video_key}

    with open(output_path / "meta" / "modality.json", "w") as f:
        json.dump(modality, f, indent=4)

    # Write info.json
    info = {
        "codebase_version": "v2.1",
        "robot_type": "unitree_g1_inspire",
        "fps": args.fps,
        "total_episodes": valid_episode_idx,
        "total_frames": total_frames,
        "total_tasks": 1,
        "total_chunks": 1,
        "chunks_size": 1000,
        "total_videos": len(camera_keys) * valid_episode_idx,
        "successful_episodes": successful_episodes,
        "splits": {"train": f"0:{valid_episode_idx}"},
        # GR00T LeRobot v2 path templates (required by lerobot_episode_loader)
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
        "features": {
            "observation.state": {
                "dtype": "float32",
                "shape": [26],
                "names": [
                    "left_shoulder_pitch", "left_shoulder_roll", "left_shoulder_yaw",
                    "left_elbow", "left_wrist_roll", "left_wrist_pitch", "left_wrist_yaw",
                    "right_shoulder_pitch", "right_shoulder_roll", "right_shoulder_yaw",
                    "right_elbow", "right_wrist_roll", "right_wrist_pitch", "right_wrist_yaw",
                    "left_pinky", "left_ring", "left_middle", "left_index", "left_thumb_pitch", "left_thumb_yaw",
                    "right_pinky", "right_ring", "right_middle", "right_index", "right_thumb_pitch", "right_thumb_yaw",
                ],
            },
            "action": {
                "dtype": "float32",
                "shape": [26],
                "names": [
                    "left_shoulder_pitch", "left_shoulder_roll", "left_shoulder_yaw",
                    "left_elbow", "left_wrist_roll", "left_wrist_pitch", "left_wrist_yaw",
                    "right_shoulder_pitch", "right_shoulder_roll", "right_shoulder_yaw",
                    "right_elbow", "right_wrist_roll", "right_wrist_pitch", "right_wrist_yaw",
                    "left_pinky", "left_ring", "left_middle", "left_index", "left_thumb_pitch", "left_thumb_yaw",
                    "right_pinky", "right_ring", "right_middle", "right_index", "right_thumb_pitch", "right_thumb_yaw",
                ],
            },
        },
        "created_at": datetime.now().isoformat(),
    }

    for cam_key in camera_keys:
        video_key = CAMERA_MAPPING.get(cam_key, f"observation.images.{cam_key}")
        info["features"][video_key] = {
            "dtype": "video",
            "shape": [480, 640, 3],  # Assuming standard resolution
            "names": ["height", "width", "channels"],
            "info": {
                "video.fps": args.fps,
                "video.codec": args.video_codec,
                "video.pix_fmt": "yuv420p",
            },
        }

    with open(output_path / "meta" / "info.json", "w") as f:
        json.dump(info, f, indent=4)

    # Write tasks.jsonl (JSON Lines format - one JSON object per line)
    with open(output_path / "meta" / "tasks.jsonl", "w") as f:
        f.write(json.dumps({"task_index": 0, "task": args.task_description}) + "\n")

    # Write episodes.jsonl AND episodes_stats.jsonl (JSON Lines format)
    # Reuse valid_episode_idx counter from above (already counted during parquet writing)
    frame_start = 0
    ep_counter = 0

    # Collect all stats for aggregate computation
    all_actions = []
    all_states = []
    episodes_stats_list = []

    with open(output_path / "meta" / "episodes.jsonl", "w") as f:
        for episode_dir in episodes:
            episode_data = load_episode(episode_dir)
            if episode_data is None:
                continue
            data_items = episode_data.get("data", [])
            num_frames = len(data_items)
            if num_frames == 0:
                continue

            # Collect states and actions for this episode
            ep_states = []
            ep_actions = []
            for item in data_items:
                states = item.get("states", {})
                actions = item.get("actions", {})
                ep_states.append(extract_26dof_state(states))
                ep_actions.append(extract_26dof_action(actions))

            ep_states = np.array(ep_states)
            ep_actions = np.array(ep_actions)

            # Compute per-episode stats
            ep_stats = {
                "action": {
                    "mean": ep_actions.mean(axis=0).tolist(),
                    "std": ep_actions.std(axis=0).tolist(),
                    "min": ep_actions.min(axis=0).tolist(),
                    "max": ep_actions.max(axis=0).tolist(),
                    "count": [num_frames],  # Single-element array (REQUIRED)
                },
                "observation.state": {
                    "mean": ep_states.mean(axis=0).tolist(),
                    "std": ep_states.std(axis=0).tolist(),
                    "min": ep_states.min(axis=0).tolist(),
                    "max": ep_states.max(axis=0).tolist(),
                    "count": [num_frames],  # Single-element array (REQUIRED)
                },
            }
            episodes_stats_list.append({"episode_index": ep_counter, "stats": ep_stats})

            # Collect for aggregate stats
            all_actions.append(ep_actions)
            all_states.append(ep_states)

            episode_entry = {
                "episode_index": ep_counter,
                "tasks": [args.task_description],
                "length": num_frames,
            }
            f.write(json.dumps(episode_entry) + "\n")
            frame_start += num_frames
            ep_counter += 1

    # Write episodes_stats.jsonl (REQUIRED for v2.1 -> v3.0 conversion)
    with open(output_path / "meta" / "episodes_stats.jsonl", "w") as f:
        for ep_stats_entry in episodes_stats_list:
            f.write(json.dumps(ep_stats_entry) + "\n")

    # Write stats.json (aggregate statistics for normalization)
    if all_actions and all_states:
        all_actions = np.concatenate(all_actions, axis=0)
        all_states = np.concatenate(all_states, axis=0)

        stats = {
            "action": {
                "mean": all_actions.mean(axis=0).tolist(),
                "std": all_actions.std(axis=0).tolist(),
                "min": all_actions.min(axis=0).tolist(),
                "max": all_actions.max(axis=0).tolist(),
                "q01": np.percentile(all_actions, 1, axis=0).tolist(),
                "q99": np.percentile(all_actions, 99, axis=0).tolist(),
            },
            "observation.state": {
                "mean": all_states.mean(axis=0).tolist(),
                "std": all_states.std(axis=0).tolist(),
                "min": all_states.min(axis=0).tolist(),
                "max": all_states.max(axis=0).tolist(),
                "q01": np.percentile(all_states, 1, axis=0).tolist(),
                "q99": np.percentile(all_states, 99, axis=0).tolist(),
            },
        }

        # Add video/image stats (standard RGB 0-255 range for lerobot compatibility)
        for cam_key in camera_keys:
            video_key = CAMERA_MAPPING.get(cam_key, f"observation.images.{cam_key}")
            stats[video_key] = {
                "min": [0.0, 0.0, 0.0],
                "max": [255.0, 255.0, 255.0],
                "mean": [127.5, 127.5, 127.5],
                "std": [72.0, 72.0, 72.0],
                "q01": [0.0, 0.0, 0.0],
                "q10": [25.0, 25.0, 25.0],
                "q50": [127.5, 127.5, 127.5],
                "q90": [230.0, 230.0, 230.0],
                "q99": [255.0, 255.0, 255.0],
            }

        with open(output_path / "meta" / "stats.json", "w") as f:
            json.dump(stats, f, indent=2)

    print(f"\nConversion complete!")
    print(f"  Episodes: {valid_episode_idx} (of {len(episodes)} total)")
    print(f"  Frames: {total_frames}")
    print(f"  Successful: {successful_episodes}")
    print(f"  Output: {output_path}")


if __name__ == "__main__":
    main()
