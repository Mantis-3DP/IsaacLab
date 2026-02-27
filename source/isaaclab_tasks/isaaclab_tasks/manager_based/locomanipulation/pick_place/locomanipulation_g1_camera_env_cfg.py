# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Locomanipulation G1 environment with ego-view camera.

Subclasses the base locomanipulation env and adds:
- Single RGB head camera on torso d435 mount matching the OAK-D W IMX378 placement
- Image observation term in the policy group (key: ego_view)
- Streaming recorder for dataset collection at 50Hz

Camera spec rationale:
- 224x224: matches GR00T input size exactly, avoids aspect ratio distortion on resize
- 50Hz: matches the sim control rate (decimation=4, physics=200Hz) and GR00T pre-training
- ego_view key: matches WBC real robot OAK sensor (oak.py, CameraMountPosition.EGO_VIEW)
- ~108° HFOV (focal_length=7.6, aperture=20.0): matches OAK-D W IMX378 real camera

NOTE: The streaming converter (convert_streaming_to_lerobot_locomanip_v21.py) must
map the streaming camera key to "ego_view" in the LeRobot dataset.
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
    """Policy observations extended with ego-view camera image."""

    ego_view = ObsTerm(
        func=base_mdp.image,
        params={"sensor_cfg": SceneEntityCfg("ego_view_camera"), "data_type": "rgb", "normalize": False},
    )


@configclass
class CameraObservationsCfg(ObservationsCfg):
    """Observations with camera image added to the policy group."""

    policy: CameraPolicyCfg = CameraPolicyCfg()


@configclass
class LocomanipulationG1CameraEnvCfg(LocomanipulationG1EnvCfg):
    """Locomanipulation G1 environment with single ego-view camera for data collection.

    Single RGB camera on the torso d435 mount, matching the OAK-D W IMX378 placement
    on the real robot. Uses key 'ego_view' matching the WBC oak.py default.
    """

    observations: CameraObservationsCfg = CameraObservationsCfg()

    def __post_init__(self):
        super().__post_init__()

        # Single ego-view camera on d435 head mount (matches OAK-D W IMX378 on real robot)
        self.scene.ego_view_camera = CameraCfg(
            prim_path="{ENV_REGEX_NS}/Robot/torso_link/d435_link/ego_view_cam",
            update_period=0.02,  # 50Hz — matches recording frequency and control rate
            height=224,
            width=224,
            data_types=["rgb"],
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=7.6, focus_distance=400.0, horizontal_aperture=20.0, clipping_range=(0.1, 2.0),
            ),
            offset=CameraCfg.OffsetCfg(pos=(0, 0, 0), rot=(0.5, 0.5, -0.5, -0.5), convention="opengl"),
        )

        # Streaming recorder for dataset collection
        self.recorders = StreamingRecorderManagerCfg(
            dataset_export_dir_path="/workspace/isaaclab/datasets/locomanip_camera",
            enable_rerun=False,
            jpeg_quality=85,
            frequency=50.0,  # 50Hz — matches control rate and GR00T pre-training
            capture_scene_state=True,
        )
