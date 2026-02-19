#!/usr/bin/env python3
"""Segment robot episode videos into subtasks using Cosmos Reason 2.

Analyzes LeRobot dataset videos with a local Cosmos Reason 2 VLM to detect
temporal boundaries between subtasks (e.g., "pick up", "walk", "place").
Outputs per-frame annotations that can be applied back to the dataset.

Usage:
    # Free-form: let Cosmos describe the phases it sees
    python segment_episodes_cosmos.py \
        --dataset_dir /path/to/lerobot/dataset \
        --episodes 0

    # Guided: classify into predefined subtask labels
    python segment_episodes_cosmos.py \
        --dataset_dir /path/to/lerobot/dataset \
        --subtasks "pick up the wheel with both hands" \
                   "walk to the right while holding the wheel" \
                   "place the wheel in the basket"

    # Apply annotations to dataset (after reviewing output JSON)
    python segment_episodes_cosmos.py \
        --dataset_dir /path/to/lerobot/dataset \
        --apply /path/to/annotations.json

Requirements:
    pip install transformers torch pyarrow
    Cosmos Reason 2 model at --model_path (default: /home/mats/Bot/Nvidia/Cosmos-Reason2-2B)
"""

import argparse
import json
import re
import sys
import time
from collections import Counter
from pathlib import Path

import pyarrow.parquet as pq
import pyarrow as pa
import torch
import transformers


DEFAULT_MODEL_PATH = "/home/mats/Bot/Nvidia/Cosmos-Reason2-2B"
DEFAULT_CAMERA = "observation.images.cam_left_high"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Segment episode videos into subtasks using Cosmos Reason 2."
    )
    parser.add_argument(
        "--dataset_dir", type=str, required=True,
        help="Path to LeRobot dataset directory.",
    )
    parser.add_argument(
        "--subtasks", type=str, nargs="+", default=None,
        help="Subtask descriptions to classify into. If omitted, Cosmos describes freely.",
    )
    parser.add_argument(
        "--model_path", type=str, default=DEFAULT_MODEL_PATH,
        help=f"Path to local Cosmos Reason 2 model (default: {DEFAULT_MODEL_PATH}).",
    )
    parser.add_argument(
        "--camera", type=str, default=DEFAULT_CAMERA,
        help=f"Camera key for video files (default: {DEFAULT_CAMERA}).",
    )
    parser.add_argument(
        "--episodes", type=int, nargs="*", default=None,
        help="Specific episode indices to process (default: all).",
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Output path for annotations JSON (default: <dataset_dir>/meta/subtask_annotations.json).",
    )
    parser.add_argument(
        "--apply", type=str, default=None,
        help="Apply annotations from JSON file to dataset (skip inference).",
    )
    parser.add_argument(
        "--fps", type=int, default=4,
        help="FPS to feed video to Cosmos (default: 4).",
    )
    parser.add_argument(
        "--video_fps", type=int, default=20,
        help="Original video FPS in the dataset (default: 20).",
    )
    return parser.parse_args()


def load_dataset_meta(dataset_dir: Path):
    """Load tasks.jsonl and episodes.jsonl."""
    tasks = []
    with open(dataset_dir / "meta" / "tasks.jsonl") as f:
        for line in f:
            tasks.append(json.loads(line.strip()))

    episodes = []
    with open(dataset_dir / "meta" / "episodes.jsonl") as f:
        for line in f:
            episodes.append(json.loads(line.strip()))

    return tasks, episodes


def find_video_files(dataset_dir: Path, camera: str):
    """Find all episode video files for a given camera."""
    video_dir = dataset_dir / "videos" / "chunk-000" / camera
    if not video_dir.exists():
        raise FileNotFoundError(f"Video directory not found: {video_dir}")
    videos = sorted(video_dir.glob("episode_*.mp4"))
    return {int(v.stem.split("_")[1]): v for v in videos}


def load_cosmos_model(model_path: str):
    """Load Cosmos Reason 2 model and processor."""
    print(f"Loading Cosmos Reason 2 from {model_path}...")
    t0 = time.perf_counter()

    model = transformers.Qwen3VLForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map="auto",
        attn_implementation="sdpa",
    )
    processor = transformers.AutoProcessor.from_pretrained(model_path)

    dt = time.perf_counter() - t0
    print(f"Model loaded in {dt:.1f}s")
    return model, processor


def build_guided_prompt(subtasks: list[str]) -> str:
    """Prompt for classifying segments into predefined subtask labels."""
    subtask_list = "\n".join(f"  {i+1}. \"{s}\"" for i, s in enumerate(subtasks))

    return f"""Analyze this robot manipulation video carefully. The robot performs a sequence of subtasks one after another.

Classify each time segment into exactly one of these subtasks:
{subtask_list}

For each subtask that appears in the video, provide the start and end timestamps in seconds.
Every frame must belong to exactly one subtask — there should be no gaps or overlaps.

Output your answer as a JSON array with this exact format:
```json
[
  {{"subtask": "exact subtask text from list above", "start_sec": 0.0, "end_sec": 5.2}},
  {{"subtask": "next subtask text", "start_sec": 5.2, "end_sec": 12.8}}
]
```

Important:
- Use the exact subtask text from the list above
- Segments must be contiguous (no gaps) and non-overlapping
- start_sec of each segment must equal end_sec of the previous segment
- The first segment must start at 0.0
- The last segment must end at the video duration"""


def build_freeform_prompt() -> str:
    """Prompt for Cosmos to freely describe the phases it observes."""
    return """This is a first-person video from a camera mounted on a humanoid robot's head. The robot performs a multi-step task involving both manipulation (grabbing, lifting, placing objects) and locomotion (walking left/right/forward). When the robot walks, the background shifts but its hands stay visible.

Break the video into sequential subtasks. For each subtask provide a short action label and timestamps.

Label guidelines:
- 3-10 words, imperative form
- Specify which hand when manipulating (left hand, right hand, both hands)
- Walking shows as background movement — look for the scene shifting sideways or forward
- Examples: "grab wheel with left hand", "walk to the right while holding wheel", "place wheel in basket"

Output ONLY a JSON array:
```json
[
  {"subtask": "description", "start_sec": 0.0, "end_sec": 4.5},
  {"subtask": "description", "start_sec": 4.5, "end_sec": 8.0}
]
```

Segments must be contiguous (no gaps), first starts at 0.0, last ends at video duration.
Be detailed — prefer more smaller segments over fewer large ones. A 20-second video typically has 5-10 distinct phases."""


def query_cosmos(model, processor, video_path: str, prompt: str, fps: int = 4):
    """Send a video + prompt to Cosmos and return the response text."""
    messages = [
        {
            "role": "system",
            "content": [{"type": "text", "text": "You are a precise video analysis assistant. Always output valid JSON."}],
        },
        {
            "role": "user",
            "content": [
                {"type": "video", "video": video_path, "fps": fps},
                {"type": "text", "text": prompt},
            ],
        },
    ]

    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    )
    inputs = inputs.to(model.device)

    with torch.inference_mode():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=2048,
            do_sample=False,
        )

    # Decode only the new tokens
    input_len = inputs["input_ids"].shape[-1]
    output_ids = generated_ids[:, input_len:]
    response = processor.batch_decode(output_ids, skip_special_tokens=True)[0]

    return response


def parse_cosmos_response(response: str, subtasks: list[str] | None = None) -> list[dict]:
    """Parse Cosmos response into structured segments.

    If subtasks is provided, validates against the list.
    If subtasks is None (free-form mode), accepts any description.
    """
    # Try to extract JSON from the response
    json_match = re.search(r"```(?:json)?\s*(\[.*?\])\s*```", response, re.DOTALL)
    if json_match:
        json_str = json_match.group(1)
    else:
        json_match = re.search(r"\[.*\]", response, re.DOTALL)
        if json_match:
            json_str = json_match.group(0)
        else:
            print(f"  WARNING: Could not find JSON in response:\n{response}")
            return []

    try:
        segments = json.loads(json_str)
    except json.JSONDecodeError as e:
        print(f"  WARNING: Failed to parse JSON: {e}\n  Raw: {json_str[:200]}")
        return []

    validated = []
    valid_subtasks = set(subtasks) if subtasks else None

    for seg in segments:
        if not isinstance(seg, dict):
            continue
        description = seg.get("subtask", "")

        if valid_subtasks is not None:
            # Guided mode: match against predefined labels
            if description not in valid_subtasks:
                # Try fuzzy match
                matched = False
                for valid in valid_subtasks:
                    if valid.lower() in description.lower() or description.lower() in valid.lower():
                        description = valid
                        matched = True
                        break
                if not matched:
                    print(f"  WARNING: Unknown subtask '{description}', skipping")
                    continue

        validated.append({
            "subtask": description,
            "start_sec": float(seg.get("start_sec", 0)),
            "end_sec": float(seg.get("end_sec", 0)),
        })

    return validated


def segments_to_frame_labels(segments: list[dict], num_frames: int,
                             video_fps: int, subtasks: list[str]) -> list[int]:
    """Convert time-based segments to per-frame subtask indices.

    Returns a list of length num_frames where each value is the subtask index
    (0-based, matching the order in the subtasks list).
    """
    subtask_to_idx = {s: i for i, s in enumerate(subtasks)}
    labels = [0] * num_frames

    for seg in segments:
        idx = subtask_to_idx.get(seg["subtask"], 0)
        start_frame = int(seg["start_sec"] * video_fps)
        end_frame = int(seg["end_sec"] * video_fps)
        start_frame = max(0, min(start_frame, num_frames))
        end_frame = max(0, min(end_frame, num_frames))
        for f in range(start_frame, end_frame):
            labels[f] = idx

    return labels


def segment_episodes(args):
    """Run Cosmos on each episode video and produce annotations."""
    dataset_dir = Path(args.dataset_dir)
    tasks, episodes = load_dataset_meta(dataset_dir)
    video_files = find_video_files(dataset_dir, args.camera)

    guided = args.subtasks is not None
    mode = "guided" if guided else "free-form"

    # Determine which episodes to process
    if args.episodes is not None:
        episode_indices = args.episodes
    else:
        episode_indices = [ep["episode_index"] for ep in episodes]

    episode_indices = [i for i in episode_indices if i in video_files]
    print(f"Processing {len(episode_indices)} episodes ({mode} mode)")

    model, processor = load_cosmos_model(args.model_path)

    if guided:
        prompt = build_guided_prompt(args.subtasks)
        print(f"\nSubtasks to detect:")
        for i, s in enumerate(args.subtasks):
            print(f"  {i}: {s}")
    else:
        prompt = build_freeform_prompt()
        print(f"\nFree-form mode: Cosmos will describe the phases it sees")
    print()

    annotations = {
        "mode": mode,
        "subtasks": args.subtasks,  # None for free-form
        "video_fps": args.video_fps,
        "cosmos_fps": args.fps,
        "camera": args.camera,
        "episodes": {},
    }

    for ep_idx in episode_indices:
        video_path = video_files[ep_idx]
        ep_meta = episodes[ep_idx]
        num_frames = ep_meta["length"]

        print(f"Episode {ep_idx:03d} ({num_frames} frames, {num_frames/args.video_fps:.1f}s)...")
        t0 = time.perf_counter()

        response = query_cosmos(model, processor, str(video_path), prompt, args.fps)
        dt = time.perf_counter() - t0

        segments = parse_cosmos_response(response, args.subtasks)

        print(f"  Inference: {dt:.1f}s")
        print(f"  Segments: {len(segments)}")
        for seg in segments:
            print(f"    [{seg['start_sec']:.1f}s - {seg['end_sec']:.1f}s] {seg['subtask']}")

        ep_annotation = {
            "segments": segments,
            "raw_response": response,
        }

        # For guided mode, also compute per-frame labels
        if guided:
            frame_labels = segments_to_frame_labels(
                segments, num_frames, args.video_fps, args.subtasks
            )
            label_counts = Counter(frame_labels)
            print(f"  Frame distribution: {dict(sorted(label_counts.items()))}")
            ep_annotation["frame_labels"] = frame_labels

        annotations["episodes"][str(ep_idx)] = ep_annotation

    # Save annotations
    output_path = args.output or str(dataset_dir / "meta" / "subtask_annotations.json")
    with open(output_path, "w") as f:
        json.dump(annotations, f, indent=2)
    print(f"\nAnnotations saved to {output_path}")

    # In free-form mode, print a summary of all unique descriptions across episodes
    if not guided:
        all_descriptions = []
        for ep_data in annotations["episodes"].values():
            for seg in ep_data["segments"]:
                all_descriptions.append(seg["subtask"])
        unique = sorted(set(all_descriptions))
        print(f"\nUnique descriptions found ({len(unique)}):")
        for d in unique:
            count = all_descriptions.count(d)
            print(f"  [{count}x] {d}")
        print(f"\nReview these, pick canonical labels, then re-run with --subtasks")

    return annotations


def apply_annotations(args):
    """Apply previously generated annotations to the dataset."""
    dataset_dir = Path(args.dataset_dir)

    with open(args.apply) as f:
        annotations = json.load(f)

    subtasks = annotations["subtasks"]
    if subtasks is None:
        print("Error: Cannot apply free-form annotations directly.")
        print("Re-run with --subtasks to produce guided annotations first.")
        sys.exit(1)

    print(f"Applying annotations with {len(subtasks)} subtasks:")
    for i, s in enumerate(subtasks):
        print(f"  {i}: {s}")

    tasks, episodes = load_dataset_meta(dataset_dir)

    # Add new subtask entries to tasks.jsonl
    existing_tasks = {t["task"] for t in tasks}
    next_task_idx = max(t["task_index"] for t in tasks) + 1
    subtask_to_task_idx = {}

    for subtask in subtasks:
        if subtask in existing_tasks:
            for t in tasks:
                if t["task"] == subtask:
                    subtask_to_task_idx[subtask] = t["task_index"]
                    break
        else:
            subtask_to_task_idx[subtask] = next_task_idx
            tasks.append({"task_index": next_task_idx, "task": subtask})
            next_task_idx += 1

    print(f"\nTask index mapping:")
    for subtask, idx in subtask_to_task_idx.items():
        print(f"  {idx}: {subtask}")

    label_to_task_idx = [subtask_to_task_idx[s] for s in subtasks]

    # Update parquet files
    updated_count = 0
    for ep_key, ep_data in annotations["episodes"].items():
        ep_idx = int(ep_key)
        frame_labels = ep_data["frame_labels"]
        task_indices = [label_to_task_idx[l] for l in frame_labels]

        parquet_path = dataset_dir / "data" / "chunk-000" / f"episode_{ep_idx:06d}.parquet"
        if not parquet_path.exists():
            print(f"  WARNING: {parquet_path} not found, skipping")
            continue

        table = pq.read_table(parquet_path)
        num_rows = len(table)

        if len(task_indices) != num_rows:
            print(f"  WARNING: Episode {ep_idx}: {len(task_indices)} labels vs {num_rows} rows, adjusting...")
            if len(task_indices) < num_rows:
                task_indices.extend([task_indices[-1]] * (num_rows - len(task_indices)))
            else:
                task_indices = task_indices[:num_rows]

        col_idx = table.column_names.index("task_index")
        table = table.set_column(col_idx, "task_index", pa.array(task_indices, type=pa.int64()))

        pq.write_table(table, parquet_path)
        updated_count += 1
        print(f"  Updated episode {ep_idx:03d} ({num_rows} frames)")

    # Update episodes.jsonl
    for ep_key, ep_data in annotations["episodes"].items():
        ep_idx = int(ep_key)
        frame_labels = ep_data["frame_labels"]
        seen = set()
        ep_subtasks = []
        for l in frame_labels:
            if l not in seen:
                seen.add(l)
                ep_subtasks.append(subtasks[l])
        episodes[ep_idx]["tasks"] = ep_subtasks

    with open(dataset_dir / "meta" / "tasks.jsonl", "w") as f:
        for t in tasks:
            f.write(json.dumps(t) + "\n")

    with open(dataset_dir / "meta" / "episodes.jsonl", "w") as f:
        for ep in episodes:
            f.write(json.dumps(ep) + "\n")

    print(f"\nApplied annotations to {updated_count} episodes")
    print(f"Updated tasks.jsonl ({len(tasks)} tasks)")
    print(f"Updated episodes.jsonl")


def main():
    args = parse_args()

    if args.apply:
        apply_annotations(args)
    else:
        segment_episodes(args)


if __name__ == "__main__":
    main()
