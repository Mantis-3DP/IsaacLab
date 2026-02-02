#!/usr/bin/env python3
"""
Combine place_g1_inspire_v3 and pick_place_g1_inspire_v30_fixed_v21 datasets.

v3: 99 episodes, cam_left/cam_right naming, HDF5 converted
v30: 25 episodes, color_0/color_1 naming, streaming recorded with commanded actions
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json

def combine_datasets():
    d1 = Path("/home/mats/Bot/Datasets/place_g1_inspire_v3")
    d2 = Path("/home/mats/Bot/Datasets/pick_place_g1_inspire_v30_fixed_v21")
    out = Path("/home/mats/Bot/Datasets/g1_inspire_combined")

    out_data = out / "data" / "chunk-000"
    out_data.mkdir(parents=True, exist_ok=True)

    # Get episode files
    d1_episodes = sorted((d1 / "data" / "chunk-000").glob("episode_*.parquet"))
    d2_episodes = sorted((d2 / "data" / "chunk-000").glob("episode_*.parquet"))

    print(f"Found {len(d1_episodes)} episodes in v3")
    print(f"Found {len(d2_episodes)} episodes in v30")

    total_episodes = 0
    global_index = 0

    # Process v3 dataset
    print("\nProcessing v3 dataset...")
    for ep_file in d1_episodes:
        df = pd.read_parquet(ep_file)

        # Add video path columns (v3 might be missing these)
        df["observation.images.cam_left.video_path"] = f"videos/chunk-000/observation.images.cam_left/episode_{total_episodes:06d}.mp4"
        df["observation.images.cam_right.video_path"] = f"videos/chunk-000/observation.images.cam_right/episode_{total_episodes:06d}.mp4"

        # Add annotation column if missing
        if "annotation.human.task_description" not in df.columns:
            df["annotation.human.task_description"] = 0

        # Add next.done and next.reward if missing
        if "next.done" not in df.columns:
            df["next.done"] = False
            df.loc[df.index[-1], "next.done"] = True
        if "next.reward" not in df.columns:
            df["next.reward"] = 0.0

        # Update indices
        df["episode_index"] = total_episodes
        df["index"] = range(global_index, global_index + len(df))

        # Save
        df.to_parquet(out_data / f"episode_{total_episodes:06d}.parquet")

        global_index += len(df)
        total_episodes += 1

        if total_episodes % 20 == 0:
            print(f"  Processed {total_episodes} episodes...")

    d1_count = total_episodes
    d1_frames = global_index
    print(f"v3: {d1_count} episodes, {d1_frames} frames")

    # Process v30 dataset
    print("\nProcessing v30 dataset...")
    for ep_file in d2_episodes:
        df = pd.read_parquet(ep_file)

        # Rename video path columns (color_0/1 → cam_left/right)
        df["observation.images.cam_left.video_path"] = f"videos/chunk-000/observation.images.cam_left/episode_{total_episodes:06d}.mp4"
        df["observation.images.cam_right.video_path"] = f"videos/chunk-000/observation.images.cam_right/episode_{total_episodes:06d}.mp4"

        # Drop old video path columns
        cols_to_drop = [c for c in df.columns if "color_0.video_path" in c or "color_1.video_path" in c]
        df = df.drop(columns=cols_to_drop, errors="ignore")

        # Update indices
        df["episode_index"] = total_episodes
        df["index"] = range(global_index, global_index + len(df))

        # Save
        df.to_parquet(out_data / f"episode_{total_episodes:06d}.parquet")

        global_index += len(df)
        total_episodes += 1

    d2_count = total_episodes - d1_count
    d2_frames = global_index - d1_frames
    print(f"v30: {d2_count} episodes, {d2_frames} frames")

    print(f"\n=== Combined Dataset ===")
    print(f"Total episodes: {total_episodes}")
    print(f"Total frames: {global_index}")

    return total_episodes, global_index


def create_metadata(total_episodes: int, total_frames: int):
    out = Path("/home/mats/Bot/Datasets/g1_inspire_combined")
    meta = out / "meta"
    meta.mkdir(parents=True, exist_ok=True)

    # Create modality.json
    modality = {
        "state": {
            "left_arm": {"start": 0, "end": 7},
            "right_arm": {"start": 7, "end": 14},
            "left_hand": {"start": 14, "end": 20},
            "right_hand": {"start": 20, "end": 26}
        },
        "action": {
            "left_arm": {"start": 0, "end": 7},
            "right_arm": {"start": 7, "end": 14},
            "left_hand": {"start": 14, "end": 20},
            "right_hand": {"start": 20, "end": 26}
        },
        "video": {
            "cam_left": {"original_key": "observation.images.cam_left"},
            "cam_right": {"original_key": "observation.images.cam_right"}
        },
        "annotation": {
            "human.task_description": {"original_key": "task_index"}
        }
    }

    with open(meta / "modality.json", "w") as f:
        json.dump(modality, f, indent=4)
    print(f"Created {meta / 'modality.json'}")

    # Create info.json
    info = {
        "codebase_version": "v2.1",
        "robot_type": "G1_INSPIRE",
        "total_episodes": total_episodes,
        "total_frames": total_frames,
        "total_tasks": 1,
        "total_videos": total_episodes * 2,
        "total_chunks": 1,
        "chunks_size": 1000,
        "fps": 30,
        "splits": {"train": f"0:{total_episodes}"},
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
        "features": {
            "observation.state": {
                "dtype": "float32",
                "shape": [26],
                "names": ["left_arm_0", "left_arm_1", "left_arm_2", "left_arm_3", "left_arm_4", "left_arm_5", "left_arm_6",
                         "right_arm_0", "right_arm_1", "right_arm_2", "right_arm_3", "right_arm_4", "right_arm_5", "right_arm_6",
                         "left_hand_0", "left_hand_1", "left_hand_2", "left_hand_3", "left_hand_4", "left_hand_5",
                         "right_hand_0", "right_hand_1", "right_hand_2", "right_hand_3", "right_hand_4", "right_hand_5"]
            },
            "action": {
                "dtype": "float32",
                "shape": [26],
                "names": ["left_arm_0", "left_arm_1", "left_arm_2", "left_arm_3", "left_arm_4", "left_arm_5", "left_arm_6",
                         "right_arm_0", "right_arm_1", "right_arm_2", "right_arm_3", "right_arm_4", "right_arm_5", "right_arm_6",
                         "left_hand_0", "left_hand_1", "left_hand_2", "left_hand_3", "left_hand_4", "left_hand_5",
                         "right_hand_0", "right_hand_1", "right_hand_2", "right_hand_3", "right_hand_4", "right_hand_5"]
            },
            "observation.images.cam_left": {
                "dtype": "video",
                "shape": [480, 640, 3],
                "names": ["height", "width", "channel"],
                "info": {
                    "video.height": 480,
                    "video.width": 640,
                    "video.codec": "av1",
                    "video.pix_fmt": "yuv420p",
                    "video.is_depth_map": False,
                    "video.fps": 30,
                    "video.channels": 3
                }
            },
            "observation.images.cam_right": {
                "dtype": "video",
                "shape": [480, 640, 3],
                "names": ["height", "width", "channel"],
                "info": {
                    "video.height": 480,
                    "video.width": 640,
                    "video.codec": "av1",
                    "video.pix_fmt": "yuv420p",
                    "video.is_depth_map": False,
                    "video.fps": 30,
                    "video.channels": 3
                }
            },
            "timestamp": {"dtype": "float32", "shape": [1], "names": None},
            "frame_index": {"dtype": "int64", "shape": [1], "names": None},
            "episode_index": {"dtype": "int64", "shape": [1], "names": None},
            "index": {"dtype": "int64", "shape": [1], "names": None},
            "task_index": {"dtype": "int64", "shape": [1], "names": None}
        }
    }

    with open(meta / "info.json", "w") as f:
        json.dump(info, f, indent=4)
    print(f"Created {meta / 'info.json'}")

    # Create tasks.json
    tasks = [{"task_index": 0, "task": "Pick cube and place in target zone"}]
    with open(meta / "tasks.json", "w") as f:
        json.dump(tasks, f, indent=4)
    print(f"Created {meta / 'tasks.json'}")


if __name__ == "__main__":
    total_episodes, total_frames = combine_datasets()
    create_metadata(total_episodes, total_frames)
    print("\nDone! Combined dataset saved to /home/mats/Bot/Datasets/g1_inspire_combined/")
