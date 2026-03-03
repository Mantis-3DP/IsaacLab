#!/usr/bin/env python3
"""Convert a stereo 20Hz LeRobot dataset into single-camera 50Hz ego_view datasets.

Takes unitree_g1.LocomanipPickPlace (30 episodes, 20Hz, cam_left_high + cam_right_high)
and produces two separate datasets where each stereo camera becomes 'ego_view' at 50Hz.
This doubles training data (60 effective episodes) while matching the UNITREE_G1 config.

Output:
    unitree_g1.LocomanipPickPlace_Left_50Hz/   (ego_view = cam_left_high)
    unitree_g1.LocomanipPickPlace_Right_50Hz/  (ego_view = cam_right_high)

Steps per episode:
    1. Video: ffmpeg upsamples 20fps → 50fps (frame duplication)
    2. Parquet: linear interpolation of state/action from 20Hz → 50Hz
    3. Meta: info.json, modality.json, tasks.jsonl, episodes.jsonl, embodiment.json

Usage:
    conda activate env_isaaclab
    python scripts/tools/convert_stereo_to_egoview_50hz.py \
        --source_dir ~/Bot/Datasets/.../unitree_g1.LocomanipPickPlace \
        --sides left right

    # Only convert left camera:
    python scripts/tools/convert_stereo_to_egoview_50hz.py \
        --source_dir ~/Bot/Datasets/.../unitree_g1.LocomanipPickPlace \
        --sides left
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SOURCE_FPS = 20.0
TARGET_FPS = 50.0

CAMERA_MAP = {
    "left": "observation.images.cam_left_high",
    "right": "observation.images.cam_right_high",
}

# Columns to linearly interpolate (continuous values)
INTERP_LIST_COLUMNS = [
    "observation.state",
    "observation.eef_state",
    "action",
    "action.eef",
    "teleop.navigate_command",
]
INTERP_SCALAR_COLUMNS = [
    "teleop.base_height_command",
]

# Discrete columns (nearest-neighbor)
DISCRETE_COLUMNS = ["task_index"]


# ---------------------------------------------------------------------------
# Video upsampling
# ---------------------------------------------------------------------------

def upsample_video(src_path: Path, dst_path: Path) -> int:
    """Upsample video from 20fps to 50fps using ffmpeg. Returns output frame count."""
    dst_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        "ffmpeg", "-y", "-i", str(src_path),
        "-r", str(int(TARGET_FPS)),
        "-vsync", "cfr",
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-an",  # no audio
        str(dst_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  ffmpeg error: {result.stderr[-500:]}", file=sys.stderr)
        raise RuntimeError(f"ffmpeg failed for {src_path}")

    # Count output frames
    probe_cmd = [
        "ffprobe", "-v", "quiet",
        "-count_frames",
        "-select_streams", "v:0",
        "-show_entries", "stream=nb_read_frames",
        "-print_format", "json",
        str(dst_path),
    ]
    probe = subprocess.run(probe_cmd, capture_output=True, text=True)
    info = json.loads(probe.stdout)
    n_frames = int(info["streams"][0]["nb_read_frames"])
    return n_frames


# ---------------------------------------------------------------------------
# Parquet upsampling
# ---------------------------------------------------------------------------

def upsample_parquet(src_path: Path, dst_path: Path, target_frames: int,
                     episode_index: int, global_index_offset: int) -> int:
    """Upsample a parquet file from 20Hz to 50Hz. Returns number of output rows."""
    table = pq.read_table(src_path)
    df = table.to_pandas()
    n_src = len(df)

    # Source and target time grids
    src_times = np.linspace(0, (n_src - 1) / SOURCE_FPS, n_src)
    dst_times = np.linspace(0, (target_frames - 1) / TARGET_FPS, target_frames)

    out = {}

    # Interpolate list columns (multi-dimensional)
    for col in INTERP_LIST_COLUMNS:
        if col not in df.columns:
            continue
        src_arr = np.array(df[col].tolist())  # (N, D)
        ndim = src_arr.shape[1]
        dst_arr = np.zeros((target_frames, ndim))
        for d in range(ndim):
            dst_arr[:, d] = np.interp(dst_times, src_times, src_arr[:, d])
        out[col] = [row.tolist() for row in dst_arr]

    # Interpolate scalar columns
    for col in INTERP_SCALAR_COLUMNS:
        if col not in df.columns:
            continue
        src_vals = df[col].values.astype(float)
        dst_vals = np.interp(dst_times, src_times, src_vals)
        out[col] = dst_vals

    # Discrete columns (nearest-neighbor via time mapping)
    src_indices = np.searchsorted(src_times, dst_times, side="right") - 1
    src_indices = np.clip(src_indices, 0, n_src - 1)
    for col in DISCRETE_COLUMNS:
        if col not in df.columns:
            continue
        src_vals = df[col].values
        out[col] = src_vals[src_indices]

    # Recomputed columns
    out["timestamp"] = dst_times.astype(np.float32)
    out["frame_index"] = np.arange(target_frames, dtype=np.int64)
    out["episode_index"] = np.full(target_frames, episode_index, dtype=np.int64)
    out["index"] = np.arange(global_index_offset, global_index_offset + target_frames, dtype=np.int64)

    # Build DataFrame and write
    out_df = pd.DataFrame(out)

    # Ensure column order matches source
    col_order = [c for c in df.columns
                 if c not in ("observation.images.cam_left_high",
                              "observation.images.cam_right_high")]
    # Add ego_view video reference column
    out_df["observation.images.ego_view"] = [
        {"path": f"videos/chunk-000/observation.images.ego_view/episode_{episode_index:06d}.mp4",
         "timestamp": float(t)}
        for t in dst_times
    ]

    # Reorder: keep non-video source columns + ego_view
    final_cols = []
    for c in col_order:
        if c in out_df.columns:
            final_cols.append(c)
    if "observation.images.ego_view" not in final_cols:
        final_cols.append("observation.images.ego_view")

    out_df = out_df[final_cols]

    dst_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_parquet(dst_path, index=False)

    return target_frames


# ---------------------------------------------------------------------------
# Meta file generation
# ---------------------------------------------------------------------------

def write_meta_files(output_dir: Path, source_dir: Path, episode_lengths: list[int],
                     side: str):
    """Write all meta files for the output dataset."""
    meta_dir = output_dir / "meta"
    meta_dir.mkdir(parents=True, exist_ok=True)

    n_episodes = len(episode_lengths)
    total_frames = sum(episode_lengths)

    # --- info.json ---
    with open(source_dir / "meta" / "info.json") as f:
        info = json.load(f)

    info["fps"] = TARGET_FPS
    info["total_episodes"] = n_episodes
    info["total_frames"] = total_frames
    info["total_videos"] = n_episodes  # 1 camera × N episodes

    # Replace stereo camera features with single ego_view
    features = info["features"]
    for cam_key in ["observation.images.cam_left_high", "observation.images.cam_right_high"]:
        features.pop(cam_key, None)
    features["observation.images.ego_view"] = {
        "dtype": "video",
        "shape": [480, 640, 3],
        "names": ["height", "width", "channel"],
        "video_info": {
            "video.fps": TARGET_FPS,
            "video.codec": "h264",
            "video.pix_fmt": "yuv420p",
            "video.is_depth_map": False,
            "has_audio": False,
        },
    }

    info["splits"] = {"train": f"0:{n_episodes}"}

    with open(meta_dir / "info.json", "w") as f:
        json.dump(info, f, indent=4)

    # --- modality.json (matches _2_Segmented format) ---
    modality = {
        "state": {
            "left_leg": {"original_key": "observation.state", "start": 0, "end": 6},
            "right_leg": {"original_key": "observation.state", "start": 6, "end": 12},
            "waist": {"original_key": "observation.state", "start": 12, "end": 15},
            "left_arm": {"original_key": "observation.state", "start": 15, "end": 22},
            "left_hand": {"original_key": "observation.state", "start": 22, "end": 29},
            "right_arm": {"original_key": "observation.state", "start": 29, "end": 36},
            "right_hand": {"original_key": "observation.state", "start": 36, "end": 43},
        },
        "action": {
            "left_arm": {"original_key": "action", "start": 15, "end": 22},
            "right_arm": {"original_key": "action", "start": 29, "end": 36},
            "left_hand": {"original_key": "action", "start": 22, "end": 29},
            "right_hand": {"original_key": "action", "start": 36, "end": 43},
            "waist": {"original_key": "action", "start": 12, "end": 15},
            "base_height_command": {"original_key": "teleop.base_height_command", "start": 0, "end": 1},
            "navigate_command": {"original_key": "teleop.navigate_command", "start": 0, "end": 3},
        },
        "video": {
            "ego_view": {"original_key": "observation.images.ego_view"},
        },
        "annotation": {
            "human.task_description": {"original_key": "task_index"},
        },
    }

    with open(meta_dir / "modality.json", "w") as f:
        json.dump(modality, f, indent=4)

    # --- tasks.jsonl ---
    with open(source_dir / "meta" / "tasks.jsonl") as f:
        tasks_lines = f.readlines()
    with open(meta_dir / "tasks.jsonl", "w") as f:
        f.writelines(tasks_lines)

    # --- episodes.jsonl ---
    # Read source tasks for the task description
    tasks_map = {}
    for line in tasks_lines:
        t = json.loads(line)
        tasks_map[t["task_index"]] = t["task"]

    with open(meta_dir / "episodes.jsonl", "w") as f:
        for i, length in enumerate(episode_lengths):
            entry = {
                "episode_index": i,
                "tasks": [tasks_map.get(0, "")],
                "length": length,
            }
            f.write(json.dumps(entry) + "\n")

    # --- embodiment.json ---
    embodiment = {
        "record_frequency": TARGET_FPS,
        "robot_name": "G1",
        "robot_type": "humanoid",
        "embodiment_tag": "unitree_g1_full_body_with_height_nav_cmd_sim",
    }
    with open(meta_dir / "embodiment.json", "w") as f:
        json.dump(embodiment, f, indent=4)

    print(f"  Meta files written to {meta_dir}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def convert_side(source_dir: Path, output_dir: Path, side: str):
    """Convert one camera side from the stereo dataset to an ego_view dataset."""
    cam_key = CAMERA_MAP[side]
    video_dir = source_dir / "videos" / "chunk-000" / cam_key
    data_dir = source_dir / "data" / "chunk-000"

    print(f"\n{'='*60}")
    print(f"Converting {side} camera ({cam_key}) → ego_view")
    print(f"  Source: {source_dir}")
    print(f"  Output: {output_dir}")
    print(f"{'='*60}")

    # Find all episodes
    parquet_files = sorted(data_dir.glob("episode_*.parquet"))
    n_episodes = len(parquet_files)
    print(f"  Found {n_episodes} episodes")

    episode_lengths = []
    global_index = 0

    for ep_idx, pq_file in enumerate(parquet_files):
        ep_num = int(pq_file.stem.split("_")[1])
        src_video = video_dir / f"episode_{ep_num:06d}.mp4"

        if not src_video.exists():
            print(f"  SKIP episode {ep_num}: video not found at {src_video}")
            continue

        print(f"\n  Episode {ep_num} ({ep_idx+1}/{n_episodes}):")

        # 1. Upsample video
        dst_video = (output_dir / "videos" / "chunk-000" / "observation.images.ego_view"
                     / f"episode_{ep_idx:06d}.mp4")
        print(f"    Video: {src_video.name} → {dst_video.name} ...", end=" ", flush=True)
        n_frames = upsample_video(src_video, dst_video)
        print(f"{n_frames} frames at {TARGET_FPS}fps")

        # 2. Upsample parquet
        dst_parquet = output_dir / "data" / "chunk-000" / f"episode_{ep_idx:06d}.parquet"
        print(f"    Parquet: {pq_file.name} → {dst_parquet.name} ...", end=" ", flush=True)
        n_rows = upsample_parquet(pq_file, dst_parquet, n_frames,
                                  episode_index=ep_idx,
                                  global_index_offset=global_index)
        print(f"{n_rows} rows")

        episode_lengths.append(n_frames)
        global_index += n_frames

    # 3. Write meta files
    print(f"\n  Writing meta files...")
    write_meta_files(output_dir, source_dir, episode_lengths, side)

    total_frames = sum(episode_lengths)
    print(f"\n  Done: {n_episodes} episodes, {total_frames} total frames at {TARGET_FPS}Hz")


def main():
    parser = argparse.ArgumentParser(
        description="Convert stereo 20Hz dataset to ego_view 50Hz datasets.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--source_dir", type=str, required=True,
        help="Path to source stereo dataset (unitree_g1.LocomanipPickPlace).",
    )
    parser.add_argument(
        "--output_dir", type=str, default=None,
        help="Output directory pattern. Use {side} placeholder. "
             "Default: source_dir/../unitree_g1.LocomanipPickPlace_{Side}_50Hz",
    )
    parser.add_argument(
        "--sides", nargs="+", choices=["left", "right"], default=["left", "right"],
        help="Which camera sides to convert (default: both).",
    )

    args = parser.parse_args()
    source_dir = Path(args.source_dir).expanduser().resolve()

    if not source_dir.exists():
        print(f"Error: source directory not found: {source_dir}", file=sys.stderr)
        sys.exit(1)

    for side in args.sides:
        if args.output_dir:
            output_dir = Path(args.output_dir.format(side=side)).expanduser().resolve()
        else:
            output_dir = source_dir.parent / f"unitree_g1.LocomanipPickPlace_{side.capitalize()}_50Hz"

        convert_side(source_dir, output_dir, side)

    print(f"\nAll done. Converted {len(args.sides)} dataset(s).")


if __name__ == "__main__":
    main()
