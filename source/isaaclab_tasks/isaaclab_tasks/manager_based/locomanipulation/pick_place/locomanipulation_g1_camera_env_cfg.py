# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Locomanipulation G1 environment with stereo cameras.

Subclasses the base locomanipulation env and adds:
- Stereo head cameras (left_high_camera, right_high_camera) on torso d435 mount
- Image observation terms in the policy group
- Streaming recorder for dataset collection
"""

import isaaclab.envs.mdp as base_mdp
import isaaclab.sim as sim_utils
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import CameraCfg
from isaaclab.utils import configclass
from isaaclab.envs.mdp.recorders import StreamingRecorderManagerCfg

from isaaclab_tasks.manager_based.locomanipulation.pick_place.locomanipulation_g1_env_cfg import (
    LocomanipulationG1EnvCfg,
    ObservationsCfg,
)


@configclass
class CameraPolicyCfg(ObservationsCfg.PolicyCfg):
    """Policy observations extended with stereo camera images."""

    head_rgb_left = ObsTerm(
        func=base_mdp.image,
        params={"sensor_cfg": SceneEntityCfg("left_high_camera"), "data_type": "rgb", "normalize": False},
    )
    head_rgb_right = ObsTerm(
        func=base_mdp.image,
        params={"sensor_cfg": SceneEntityCfg("right_high_camera"), "data_type": "rgb", "normalize": False},
    )


@configclass
class CameraObservationsCfg(ObservationsCfg):
    """Observations with camera images added to the policy group."""

    policy: CameraPolicyCfg = CameraPolicyCfg()


@configclass
class LocomanipulationG1CameraEnvCfg(LocomanipulationG1EnvCfg):
    """Locomanipulation G1 environment with stereo cameras for data collection.

    Adds left/right head cameras mounted on the torso d435 link, matching the
    real G1 robot's RealSense D435 stereo camera placement.
    """

    observations: CameraObservationsCfg = CameraObservationsCfg()

    def __post_init__(self):
        super().__post_init__()

        # Stereo cameras on torso d435 mount (matches real robot RealSense D435)
        # TODO(human): Configure camera parameters for your setup
        self.scene.left_high_camera = CameraCfg(
            prim_path="{ENV_REGEX_NS}/Robot/torso_link/d435_link/left_high_cam",
            update_period=0.033,  # 30Hz
            height=480,
            width=640,
            data_types=["rgb"],
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=7.6, focus_distance=400.0, horizontal_aperture=20.0, clipping_range=(0.1, 2.0),
            ),
            offset=CameraCfg.OffsetCfg(pos=(0, 0.0315, 0), rot=(0.5, 0.5, -0.5, -0.5), convention="opengl"),
        )

        self.scene.right_high_camera = CameraCfg(
            prim_path="{ENV_REGEX_NS}/Robot/torso_link/d435_link/right_high_cam",
            update_period=0.033,  # 30Hz
            height=480,
            width=640,
            data_types=["rgb"],
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=7.6, focus_distance=400.0, horizontal_aperture=20.0, clipping_range=(0.1, 2.0),
            ),
            offset=CameraCfg.OffsetCfg(pos=(0, -0.0315, 0), rot=(0.5, 0.5, -0.5, -0.5), convention="opengl"),
        )

        # Streaming recorder for dataset collection
        self.recorders = StreamingRecorderManagerCfg(
            dataset_export_dir_path="/workspace/isaaclab/datasets/locomanip_camera",
            enable_rerun=False,
            jpeg_quality=85,
            frequency=20.0,
            capture_scene_state=True,
        )
