# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Streaming episode writer that saves data to disk in real-time.

Based on Unitree's xr_teleoperate EpisodeWriter approach:
- Background thread for non-blocking writes
- JPEG compression for images (10-20x smaller than raw)
- Incremental JSON for state/action data
- Near-instant episode save (just close file handles)
"""

from __future__ import annotations

import cv2
import json
import numpy as np
import os
import threading
import torch
from queue import Queue, Empty
from typing import Any

try:
    import rerun as rr
    import rerun.blueprint as rrb
    RERUN_AVAILABLE = True
except ImportError:
    RERUN_AVAILABLE = False


class StreamingEpisodeWriter:
    """Writes episode data to disk in real-time using background thread.

    This writer streams images as JPEG and state/action data as incremental JSON,
    avoiding the memory buildup that occurs with batch HDF5 writing.

    Directory structure per episode:
        episode_XXXX/
            colors/
                000000_cam_left.jpg
                000000_cam_right.jpg
                000001_cam_left.jpg
                ...
            data.json  (incrementally written)

    The data.json format matches Unitree's xr_teleoperate format for compatibility.
    """

    def __init__(
        self,
        task_dir: str,
        task_info: dict | None = None,
        frequency: float = 30.0,
        jpeg_quality: int = 95,
        enable_rerun: bool = False,
        rerun_memory_limit: str = "300MB",
    ):
        """Initialize the streaming episode writer.

        Args:
            task_dir: Base directory for saving episodes.
            task_info: Optional dict with task metadata (goal, desc, steps).
            frequency: Target recording frequency in Hz (for metadata).
            jpeg_quality: JPEG compression quality (1-100). Higher = better quality, larger files.
            enable_rerun: Enable rerun.io live visualization.
            rerun_memory_limit: Memory limit for rerun viewer.
        """
        self.task_dir = task_dir
        self.frequency = frequency
        self.jpeg_quality = jpeg_quality
        self.enable_rerun = enable_rerun and RERUN_AVAILABLE

        # Task metadata
        self.task_info = task_info or {
            "goal": "Task goal not specified",
            "desc": "Task description not specified",
            "steps": "Task steps not specified",
        }

        # Episode tracking
        self.episode_id = -1
        self.item_id = -1
        self.episode_dir = None
        self.color_dir = None
        self.json_path = None
        self.first_item = True

        # Thread-safe queue for background writing
        self.item_queue: Queue = Queue(maxsize=-1)  # Unlimited size
        self.stop_worker = False
        self.need_save = False
        self.is_available = True

        # Start background worker thread
        self.worker_thread = threading.Thread(target=self._process_queue, daemon=True)
        self.worker_thread.start()

        # Initialize rerun if enabled
        if self.enable_rerun:
            self._init_rerun(rerun_memory_limit)

        # Create task directory if needed
        os.makedirs(self.task_dir, exist_ok=True)

        # Find existing episodes to continue numbering
        self._scan_existing_episodes()

    def _scan_existing_episodes(self):
        """Scan for existing episodes to continue numbering."""
        if os.path.exists(self.task_dir):
            episode_dirs = [
                d for d in os.listdir(self.task_dir)
                if d.startswith("episode_") and os.path.isdir(os.path.join(self.task_dir, d))
            ]
            if episode_dirs:
                # Get highest episode number
                self.episode_id = max(
                    int(d.split("_")[-1]) for d in episode_dirs
                )

    def _init_rerun(self, memory_limit: str):
        """Initialize rerun.io viewer."""
        from datetime import datetime
        rr.init(datetime.now().strftime("IsaacLab_%Y%m%d_%H%M%S"))
        rr.spawn(memory_limit=memory_limit, hide_welcome_screen=True)

    def is_ready(self) -> bool:
        """Check if writer is ready for new episode."""
        return self.is_available

    def create_episode(self) -> bool:
        """Create a new episode for recording.

        Returns:
            True if episode created successfully, False if writer is busy.
        """
        if not self.is_available:
            return False

        self.episode_id += 1
        self.item_id = -1
        self.first_item = True

        # Create episode directories
        self.episode_dir = os.path.join(self.task_dir, f"episode_{self.episode_id:04d}")
        self.color_dir = os.path.join(self.episode_dir, "colors")
        self.json_path = os.path.join(self.episode_dir, "data.json")

        os.makedirs(self.episode_dir, exist_ok=True)
        os.makedirs(self.color_dir, exist_ok=True)

        # Start JSON file with metadata
        info = {
            "version": "1.0.0",
            "frequency": self.frequency,
            "image": {"format": "jpeg", "quality": self.jpeg_quality},
        }

        with open(self.json_path, "w", encoding="utf-8") as f:
            f.write('{\n')
            f.write('"info": ' + json.dumps(info, indent=4) + ',\n')
            f.write('"text": ' + json.dumps(self.task_info, indent=4) + ',\n')
            f.write('"data": [\n')

        self.is_available = False
        return True

    def add_item(
        self,
        colors: dict[str, np.ndarray | torch.Tensor] | None = None,
        states: dict[str, Any] | None = None,
        actions: dict[str, Any] | None = None,
        raw_actions: list[float] | None = None,
        success: bool | None = None,
    ):
        """Add a frame to the current episode (non-blocking).

        Args:
            colors: Dict of camera_name -> image (HWC format, uint8 or float).
            states: Dict of state data (will be JSON serialized).
            actions: Dict of action data (will be JSON serialized).
            raw_actions: Raw action input (e.g., Pink IK EEF poses) for replay compatibility.
            success: Optional success flag for this frame.
        """
        self.item_id += 1

        # Convert torch tensors to numpy for image data
        if colors:
            processed_colors = {}
            for key, img in colors.items():
                if isinstance(img, torch.Tensor):
                    img = img.cpu().numpy()
                # Convert float [0,1] to uint8 [0,255] if needed
                if img.dtype in [np.float32, np.float64]:
                    img = (img * 255).astype(np.uint8)
                processed_colors[key] = img
            colors = processed_colors

        # Convert state/action tensors to lists for JSON
        states = self._tensordict_to_json(states) if states else {}
        actions = self._tensordict_to_json(actions) if actions else {}

        item_data = {
            "idx": self.item_id,
            "colors": colors,
            "states": states,
            "actions": actions,
            "raw_actions": raw_actions,
            "success": success,
        }

        self.item_queue.put(item_data)

    def _tensordict_to_json(self, data: dict | torch.Tensor | np.ndarray | Any) -> Any:
        """Recursively convert tensors to JSON-serializable format."""
        if isinstance(data, dict):
            return {k: self._tensordict_to_json(v) for k, v in data.items()}
        elif isinstance(data, torch.Tensor):
            return data.cpu().tolist()
        elif isinstance(data, np.ndarray):
            return data.tolist()
        elif isinstance(data, (list, tuple)):
            return [self._tensordict_to_json(item) for item in data]
        else:
            return data

    def _process_queue(self):
        """Background thread: process queued items and write to disk."""
        while not self.stop_worker or not self.item_queue.empty():
            try:
                item_data = self.item_queue.get(timeout=0.1)
                self._write_item(item_data)
                self.item_queue.task_done()
            except Empty:
                pass

            # Check if save was triggered and queue is empty
            if self.need_save and self.item_queue.empty():
                self._finalize_episode()

    def _write_item(self, item_data: dict):
        """Write a single item to disk (called from background thread)."""
        idx = item_data["idx"]
        colors = item_data.get("colors", {})
        states = item_data.get("states", {})
        actions = item_data.get("actions", {})

        # Save images as JPEG
        color_paths = {}
        if colors:
            for cam_name, img in colors.items():
                filename = f"{idx:06d}_{cam_name}.jpg"
                filepath = os.path.join(self.color_dir, filename)

                # Convert RGB to BGR for OpenCV
                if len(img.shape) == 3 and img.shape[2] == 3:
                    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                else:
                    img_bgr = img

                cv2.imwrite(
                    filepath,
                    img_bgr,
                    [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality]
                )
                color_paths[cam_name] = os.path.join("colors", filename)

        # Build JSON entry
        json_entry = {
            "idx": idx,
            "colors": color_paths,
            "states": states,
            "actions": actions,
        }
        # Include raw_actions if available (for replay/annotate compatibility)
        if item_data.get("raw_actions") is not None:
            json_entry["raw_actions"] = item_data["raw_actions"]
        if item_data.get("success") is not None:
            json_entry["success"] = item_data["success"]

        # Append to JSON file
        with open(self.json_path, "a", encoding="utf-8") as f:
            if not self.first_item:
                f.write(",\n")
            f.write(json.dumps(json_entry, indent=4))
            self.first_item = False

        # Log to rerun if enabled
        if self.enable_rerun:
            self._log_to_rerun(item_data)

    def _log_to_rerun(self, item_data: dict):
        """Log item data to rerun.io viewer."""
        if not RERUN_AVAILABLE:
            return

        idx = item_data["idx"]
        rr.set_time_sequence("frame", idx)

        # Log images
        colors = item_data.get("colors", {})
        for cam_name, img in colors.items():
            rr.log(f"cameras/{cam_name}", rr.Image(img))

        # Log states as scalars
        states = item_data.get("states", {})
        self._log_nested_scalars(states, "states")

        # Log actions as scalars
        actions = item_data.get("actions", {})
        self._log_nested_scalars(actions, "actions")

    def _log_nested_scalars(self, data: dict, prefix: str):
        """Recursively log nested dict values as rerun scalars."""
        if not RERUN_AVAILABLE:
            return

        for key, value in data.items():
            path = f"{prefix}/{key}"
            if isinstance(value, dict):
                self._log_nested_scalars(value, path)
            elif isinstance(value, (list, tuple)):
                for i, v in enumerate(value):
                    if isinstance(v, (int, float)):
                        rr.log(f"{path}/{i}", rr.Scalar(v))
            elif isinstance(value, (int, float)):
                rr.log(path, rr.Scalar(value))

    def save_episode(self, success: bool | None = None) -> str:
        """Trigger episode save (non-blocking, waits for queue to drain).

        Args:
            success: Overall episode success flag.

        Returns:
            Path to the saved episode directory.
        """
        self.episode_success = success
        self.need_save = True
        return self.episode_dir

    def discard_episode(self):
        """Discard the current episode without saving (for manual reset).

        This clears the queue, deletes the episode directory, and resets
        the episode counter so the number can be reused.
        """
        import shutil

        # Clear the queue (don't write pending items)
        while not self.item_queue.empty():
            try:
                self.item_queue.get_nowait()
                self.item_queue.task_done()
            except Exception:
                break

        # Delete episode directory if it exists
        if self.episode_dir and os.path.exists(self.episode_dir):
            try:
                shutil.rmtree(self.episode_dir)
                print(f"[StreamingEpisodeWriter] Discarded episode: {self.episode_dir}")
            except Exception as e:
                print(f"[StreamingEpisodeWriter] Error discarding episode: {e}")

        # Decrement episode_id so we reuse this number
        self.episode_id -= 1
        self.is_available = True
        self.episode_dir = None
        self.color_dir = None
        self.json_path = None

    def _finalize_episode(self):
        """Finalize episode (called from background thread when queue is empty)."""
        # Close JSON array
        with open(self.json_path, "a", encoding="utf-8") as f:
            f.write("\n],\n")
            # Add episode-level metadata (use proper JSON null for None)
            success = getattr(self, "episode_success", None)
            if success is None:
                success_str = "null"
            else:
                success_str = "true" if success else "false"
            f.write(f'"success": {success_str}\n')
            f.write("}\n")

        self.need_save = False
        self.is_available = True

    def close(self):
        """Clean up resources and ensure all data is written."""
        # Wait for queue to drain
        self.item_queue.join()

        # Save any in-progress episode
        if not self.is_available:
            self.save_episode()
            # Wait for save to complete
            while not self.is_available:
                pass

        # Stop worker thread
        self.stop_worker = True
        self.worker_thread.join(timeout=5.0)

    def __del__(self):
        """Destructor."""
        try:
            self.close()
        except Exception:
            pass


class StreamingDatasetFileHandler:
    """Dataset file handler that uses StreamingEpisodeWriter.

    This is a drop-in replacement for HDF5DatasetFileHandler that uses
    the streaming directory format instead of HDF5.
    """

    def __init__(self):
        """Initialize the streaming dataset handler."""
        self._writer: StreamingEpisodeWriter | None = None
        self._demo_count = 0
        self._env_args = {}
        self._file_path = None

    def create(self, file_path: str, env_name: str | None = None):
        """Create a new streaming dataset.

        Args:
            file_path: Base path for the dataset (will create directory).
            env_name: Optional environment name for metadata.
        """
        # Remove .hdf5 extension if present
        if file_path.endswith(".hdf5"):
            file_path = file_path[:-5]

        self._file_path = file_path
        self._writer = StreamingEpisodeWriter(
            task_dir=file_path,
            task_info={"env_name": env_name or ""},
            enable_rerun=True,  # Enable rerun by default
        )
        self._demo_count = self._writer.episode_id + 1  # Continue from existing

    def add_env_args(self, env_args: dict):
        """Add environment arguments (stored in each episode's JSON)."""
        self._env_args.update(env_args)

    @property
    def demo_count(self) -> int:
        """Number of episodes recorded."""
        return self._demo_count

    def write_episode(self, episode, demo_id: int | None = None):
        """Write an episode using streaming writer.

        Note: This method converts from IsaacLab's EpisodeData format
        to the streaming format. For best performance, use the writer directly.
        """
        if episode.is_empty():
            return

        # Start new episode
        if not self._writer.create_episode():
            return

        # Get data from episode
        data = episode.data

        # Determine number of frames from actions
        num_frames = 0
        if "actions" in data:
            actions_data = data["actions"]
            if isinstance(actions_data, torch.Tensor):
                num_frames = len(actions_data)
            elif isinstance(actions_data, list):
                num_frames = len(actions_data)

        # Write each frame
        for i in range(num_frames):
            colors = {}
            states = {}
            actions = {}

            # Extract observations/images for this frame
            if "obs" in data:
                for key, value in data["obs"].items():
                    if "rgb" in key.lower() or "image" in key.lower() or "camera" in key.lower():
                        # This is an image observation
                        if isinstance(value, torch.Tensor):
                            colors[key] = value[i]
                        elif isinstance(value, list) and i < len(value):
                            colors[key] = value[i]
                    else:
                        # This is a state observation
                        if isinstance(value, torch.Tensor):
                            states[key] = value[i]
                        elif isinstance(value, list) and i < len(value):
                            states[key] = value[i]

            # Extract states for this frame
            if "states" in data:
                for key, value in data["states"].items():
                    if isinstance(value, torch.Tensor):
                        states[key] = value[i]
                    elif isinstance(value, dict):
                        states[key] = {
                            k: v[i] if isinstance(v, torch.Tensor) else v
                            for k, v in value.items()
                        }

            # Extract actions for this frame
            if "actions" in data:
                actions_val = data["actions"]
                if isinstance(actions_val, torch.Tensor):
                    actions = {"action": actions_val[i]}
                elif isinstance(actions_val, list) and i < len(actions_val):
                    actions = {"action": actions_val[i]}

            self._writer.add_item(colors=colors, states=states, actions=actions)

        # Save episode
        self._writer.save_episode(success=episode.success)
        self._demo_count += 1

    def flush(self):
        """Flush is handled automatically by streaming writer."""
        pass

    def close(self):
        """Close the streaming writer."""
        if self._writer is not None:
            self._writer.close()
            self._writer = None
