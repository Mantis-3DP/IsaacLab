# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Streaming recorder manager configuration.

This provides a drop-in replacement for ActionStateRecorderManagerCfg that
uses real-time streaming instead of batch HDF5 writing.
"""

from isaaclab.managers.recorder_manager import RecorderManagerBaseCfg, DatasetExportMode
from isaaclab.utils import configclass

from .streaming_recorder import StreamingRecorder, StreamingRecorderCfg


@configclass
class StreamingRecorderManagerCfg(RecorderManagerBaseCfg):
    """Recorder configuration that streams data to disk in real-time.

    This is a drop-in replacement for ActionStateRecorderManagerCfg that:
    - Writes images as JPEG immediately (no memory buildup)
    - Appends state/action JSON incrementally
    - Provides near-instant episode saves
    - Optionally shows live visualization via rerun.io

    Usage:
        ```python
        from isaaclab.envs.mdp.recorders import StreamingRecorderManagerCfg

        @configclass
        class MyEnvCfg(ManagerBasedRLEnvCfg):
            def __post_init__(self):
                self.recorders = StreamingRecorderManagerCfg(
                    dataset_export_dir_path="./datasets/my_task",
                    enable_rerun=True,
                )
        ```
    """

    # DISABLE HDF5 export completely - streaming recorder handles everything
    dataset_export_mode: DatasetExportMode = DatasetExportMode.EXPORT_NONE
    export_in_record_pre_reset: bool = False
    export_in_close: bool = False

    # Streaming-specific settings
    frequency: float = 30.0
    """Target recording frequency in Hz (for metadata)."""

    jpeg_quality: int = 95
    """JPEG compression quality (1-100). Higher = better quality, larger files."""

    enable_rerun: bool = True
    """Enable rerun.io live visualization."""

    rerun_memory_limit: str = "300MB"
    """Memory limit for rerun viewer."""

    capture_scene_state: bool = False
    """Capture full scene state (object poses) for mimic annotation compatibility."""

    # The streaming recorder term - initialized in __post_init__
    streaming_recorder: StreamingRecorderCfg | None = None

    def __post_init__(self):
        """Initialize the streaming recorder term with our settings."""
        print(f"[StreamingRecorderManagerCfg] __post_init__ called, task_dir={self.dataset_export_dir_path}")
        self.streaming_recorder = StreamingRecorderCfg(
            task_dir=self.dataset_export_dir_path,
            frequency=self.frequency,
            jpeg_quality=self.jpeg_quality,
            enable_rerun=self.enable_rerun,
            rerun_memory_limit=self.rerun_memory_limit,
            capture_scene_state=self.capture_scene_state,
        )
        # Explicitly set class_type (module-level assignment doesn't work with @configclass)
        self.streaming_recorder.class_type = StreamingRecorder
        print(f"[StreamingRecorderManagerCfg] streaming_recorder created with class_type={self.streaming_recorder.class_type}")
