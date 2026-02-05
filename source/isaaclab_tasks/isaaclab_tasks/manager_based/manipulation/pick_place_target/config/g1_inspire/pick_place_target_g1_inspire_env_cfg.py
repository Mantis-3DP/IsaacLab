# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for G1 robot with Inspire hands in the pick-place-target task.

Task: Pick a cube and place it in the target zone (yellow box on table).
Based on stack_g1_inspire_env_cfg.py - inherits from StackEnvCfg with 2 cubes removed.
"""

import os
import tempfile
import torch

import carb

# Path to local assets (reuse from stack task)
G1_INSPIRE_ASSETS_DIR = os.path.join(
    os.path.dirname(__file__), "../../../stack/config/g1_inspire/assets"
)
from pink.tasks import DampingTask

from isaaclab.controllers.pink_ik.local_frame_task import LocalFrameTask

import isaaclab.controllers.utils as ControllerUtils
import isaaclab.envs.mdp as base_mdp
import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.controllers.pink_ik import NullSpacePostureTask, PinkIKControllerCfg
from isaaclab.devices.device_base import DevicesCfg
from isaaclab.devices.openxr import OpenXRDeviceCfg, XrCfg
from isaaclab.devices.openxr.retargeters.humanoid.unitree.inspire.g1_upper_body_retargeter import UnitreeG1RetargeterCfg
from isaaclab.envs.mdp.actions.pink_actions_cfg import PinkInverseKinematicsActionCfg
from isaaclab.envs.mdp.recorders import StreamingRecorderManagerCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.sensors import CameraCfg, FrameTransformerCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg
from isaaclab.utils import configclass

from isaaclab_tasks.manager_based.manipulation.stack import mdp
from isaaclab_tasks.manager_based.manipulation.stack.mdp import franka_stack_events
from isaaclab_tasks.manager_based.manipulation.stack.stack_env_cfg import StackEnvCfg
from isaaclab_tasks.manager_based.manipulation.pick_place_target import mdp as pick_place_target_mdp
from isaaclab_tasks.manager_based.manipulation.pick_place import mdp as pick_place_mdp

from isaaclab.markers.config import FRAME_MARKER_CFG  # isort: skip
from isaaclab_assets.robots.unitree import G1_INSPIRE_FTP_CFG  # isort: skip


# Target zone for placing the cube (in world coordinates)
# Yellow blocks in USD at local (0.1, 0.2, 0.95), table at (-4.3, -4.2, -0.2)
# World position: (-4.3+0.1, -4.2+0.2, -0.2+0.95) = (-4.2, -4.0, 0.75)
TARGET_ZONE = {
    "x": (-4.30, -4.10),  # 20cm wide, centered on yellow blocks
    "y": (-4.10, -3.90),  # 20cm deep, centered on y=-4.0
    "center": (-4.20, -4.00),
}


@configclass
class G1PickPlaceTargetObservationsCfg:
    """Observation specifications for G1 pick-place-target task (single cube)."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group with state values."""

        actions = ObsTerm(func=mdp.last_action)
        robot_joint_pos = ObsTerm(
            func=base_mdp.joint_pos,
            params={"asset_cfg": SceneEntityCfg("robot")},
        )
        robot_joint_vel = ObsTerm(
            func=base_mdp.joint_vel,
            params={"asset_cfg": SceneEntityCfg("robot")},
        )
        robot_root_pos = ObsTerm(func=base_mdp.root_pos_w, params={"asset_cfg": SceneEntityCfg("robot")})
        robot_root_rot = ObsTerm(func=base_mdp.root_quat_w, params={"asset_cfg": SceneEntityCfg("robot")})
        # Single cube observation (named cube_1 to match stack task pattern)
        cube_position = ObsTerm(func=base_mdp.root_pos_w, params={"asset_cfg": SceneEntityCfg("cube_1")})
        cube_orientation = ObsTerm(func=base_mdp.root_quat_w, params={"asset_cfg": SceneEntityCfg("cube_1")})
        # End-effector observations
        eef_pos = ObsTerm(func=mdp.ee_frame_pos)
        eef_quat = ObsTerm(func=mdp.ee_frame_quat)
        # Separate left/right EEF observations (for mimic env)
        left_eef_pos = ObsTerm(func=pick_place_mdp.get_eef_pos, params={"link_name": "left_wrist_yaw_link"})
        left_eef_quat = ObsTerm(func=pick_place_mdp.get_eef_quat, params={"link_name": "left_wrist_yaw_link"})
        right_eef_pos = ObsTerm(func=pick_place_mdp.get_eef_pos, params={"link_name": "right_wrist_yaw_link"})
        right_eef_quat = ObsTerm(func=pick_place_mdp.get_eef_quat, params={"link_name": "right_wrist_yaw_link"})
        # Camera observations
        head_rgb_left = ObsTerm(
            func=base_mdp.image,
            params={"sensor_cfg": SceneEntityCfg("left_high_camera"), "data_type": "rgb", "normalize": False},
        )
        head_rgb_right = ObsTerm(
            func=base_mdp.image,
            params={"sensor_cfg": SceneEntityCfg("right_high_camera"), "data_type": "rgb", "normalize": False},
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False

    policy: PolicyCfg = PolicyCfg()


@configclass
class PickPlaceTargetEventCfg:
    """Configuration for pick-place-target task events."""

    reset_g1_arm_pose = EventTerm(
        func=franka_stack_events.set_default_joint_pose,
        mode="reset",
        params={
            "default_pose": [0.0] * 57,
        },
    )

    # Robot position randomization: +/-5cm position, +/-10 degrees rotation
    reset_robot_position = EventTerm(
        func=base_mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {
                "x": (-0.05, 0.05),     # +/-5cm
                "y": (-0.05, 0.05),     # +/-5cm
                "yaw": (-0.175, 0.175), # +/-10 degrees (in radians)
            },
            "velocity_range": {},
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )

    randomize_cube_position = EventTerm(
        func=franka_stack_events.randomize_object_pose,
        mode="reset",
        params={
            # Cube spawns BEYOND target zone (further from robot)
            # Target zone is y: (-4.10, -3.90), so spawn at more negative Y
            "pose_range": {
                "x": (-4.50, -3.90),  # 20cm wide, same as target
                "y": (-4.20, -4.10),  # 20cm deep, 5cm closer to robot
                "z": (0.84, 0.84),
                "yaw": (-0.5, 0.5),
            },
            "min_separation": 0.0,
            "asset_cfgs": [SceneEntityCfg("cube_1")],
        },
    )

    # Hide yellow target cubes on episode reset
    reset_target_visibility = EventTerm(
        func=pick_place_target_mdp.reset_target_visibility,
        mode="reset",
        params={},
    )

    # Show target cubes when block is lifted
    check_reveal_target = EventTerm(
        func=pick_place_target_mdp.reveal_target_on_lift,
        mode="interval",
        interval_range_s=(0.05, 0.05),  # Check every 50ms
        params={
            "block_cfg": SceneEntityCfg("cube_1"),
            "height_threshold": 0.1,  # 10cm above initial
        },
    )


@configclass
class PickPlaceTargetTerminationsCfg:
    """Termination conditions for the pick-place-target task."""

    # Success: cube in target zone with hands open
    success = DoneTerm(
        func=pick_place_target_mdp.cube_in_target_zone,
        params={
            "robot_cfg": SceneEntityCfg("robot"),
            "cube_cfg": SceneEntityCfg("cube_1"),
            "target_x": TARGET_ZONE["x"],
            "target_y": TARGET_ZONE["y"],
        },
    )

    # Time out
    time_out = DoneTerm(func=base_mdp.time_out, time_out=True)


@configclass
class G1InspirePickPlaceTargetEnvCfg(StackEnvCfg):
    """Configuration for G1 Inspire pick-place-target task: pick cube -> place in target zone.

    Inherits from StackEnvCfg to get working scene setup, then simplifies to single cube.
    """

    # Override observations
    observations: G1PickPlaceTargetObservationsCfg = G1PickPlaceTargetObservationsCfg()

    # Override terminations
    terminations: PickPlaceTargetTerminationsCfg = PickPlaceTargetTerminationsCfg()

    # XR config
    xr: XrCfg = XrCfg(
        anchor_pos=(-4.2, -3.7, 0.0),
        anchor_rot=(0.0, 0.0, 0.0, 1.0),
    )

    # Temporary directory for URDF files
    temp_urdf_dir = tempfile.gettempdir()

    # Idle action for Pink IK controller
    idle_action = torch.tensor([
        -0.1487, 0.2038, 1.0952, 0.707, 0.0, 0.0, 0.707,
        0.1487, 0.2038, 1.0952, 0.707, 0.0, 0.0, 0.707,
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    ])

    def __post_init__(self):
        # Call parent post_init first
        super().__post_init__()

        # Increase env spacing for warehouse environment (default 2.5 is too small)
        self.scene.env_spacing = 7.0

        # Set events for pick-place-target task
        self.events = PickPlaceTargetEventCfg()

        # Set G1 robot with Inspire hands
        self.scene.robot = G1_INSPIRE_FTP_CFG.replace(
            prim_path="/World/envs/env_.*/Robot",
            init_state=ArticulationCfg.InitialStateCfg(
                pos=(-4.2, -3.7, 0.76),
                rot=(0.7071, 0, 0, -0.7071),
                joint_pos={
                    "right_shoulder_pitch_joint": -0.35,
                    "right_shoulder_roll_joint": -0.16,
                    "left_shoulder_pitch_joint": -0.35,
                    "left_shoulder_roll_joint": 0.16,
                    ".*_shoulder_yaw_joint": 0.0,
                    ".*_wrist_.*": 0.0,
                    "waist_.*": 0.0,
                    ".*_hip_.*": 0.0,
                    ".*_knee_.*": 0.0,
                    ".*_ankle_.*": 0.0,
                    ".*_thumb_.*": 0.0,
                    ".*_index_.*": 0.0,
                    ".*_middle_.*": 0.0,
                    ".*_ring_.*": 0.0,
                    ".*_pinky_.*": 0.0,
                },
                joint_vel={".*": 0.0},
            ),
        )
        self.scene.robot.spawn.semantic_tags = [("class", "robot")]

        # Remove the default table from parent config
        self.scene.table = None

        # Add warehouse room environment
        self.scene.room_walls = AssetBaseCfg(
            prim_path="/World/envs/env_.*/Room",
            init_state=AssetBaseCfg.InitialStateCfg(pos=[0.0, 0.0, 0.0], rot=[1.0, 0.0, 0.0, 0.0]),
            spawn=UsdFileCfg(usd_path=f"{G1_INSPIRE_ASSETS_DIR}/small_warehouse_digital_twin/small_warehouse_digital_twin.usd"),
        )

        # Add packing table with yellow box (target zone)
        # Yellow target cubes are hidden on reset and revealed when block is lifted
        self.scene.packing_table = AssetBaseCfg(
            prim_path="/World/envs/env_.*/PackingTable",
            init_state=AssetBaseCfg.InitialStateCfg(pos=[-4.3, -4.2, -0.2], rot=[1.0, 0.0, 0.0, 0.0]),
            spawn=UsdFileCfg(usd_path=f"{G1_INSPIRE_ASSETS_DIR}/table_with_yellowbox.usd"),
        )

        # Configure Pink IK controller
        self.actions.arm_action = PinkInverseKinematicsActionCfg(
            pink_controlled_joint_names=[
                ".*_shoulder_pitch_joint", ".*_shoulder_roll_joint", ".*_shoulder_yaw_joint",
                ".*_elbow_joint", ".*_wrist_yaw_joint", ".*_wrist_roll_joint", ".*_wrist_pitch_joint",
            ],
            hand_joint_names=[
                "L_index_proximal_joint", "L_middle_proximal_joint", "L_pinky_proximal_joint",
                "L_ring_proximal_joint", "L_thumb_proximal_yaw_joint", "R_index_proximal_joint",
                "R_middle_proximal_joint", "R_pinky_proximal_joint", "R_ring_proximal_joint",
                "R_thumb_proximal_yaw_joint", "L_index_intermediate_joint", "L_middle_intermediate_joint",
                "L_pinky_intermediate_joint", "L_ring_intermediate_joint", "L_thumb_proximal_pitch_joint",
                "R_index_intermediate_joint", "R_middle_intermediate_joint", "R_pinky_intermediate_joint",
                "R_ring_intermediate_joint", "R_thumb_proximal_pitch_joint", "L_thumb_intermediate_joint",
                "R_thumb_intermediate_joint", "L_thumb_distal_joint", "R_thumb_distal_joint",
            ],
            target_eef_link_names={
                "left_wrist": "left_wrist_yaw_link",
                "right_wrist": "right_wrist_yaw_link",
            },
            asset_name="robot",
            controller=PinkIKControllerCfg(
                articulation_name="robot",
                base_link_name="pelvis",
                num_hand_joints=24,
                show_ik_warnings=False,
                fail_on_joint_limit_violation=False,
                variable_input_tasks=[
                    LocalFrameTask("g1_29dof_rev_1_0_left_wrist_yaw_link", base_link_frame_name="g1_29dof_rev_1_0_pelvis",
                                   position_cost=8.0, orientation_cost=2.0, lm_damping=10, gain=0.5),
                    LocalFrameTask("g1_29dof_rev_1_0_right_wrist_yaw_link", base_link_frame_name="g1_29dof_rev_1_0_pelvis",
                                   position_cost=8.0, orientation_cost=2.0, lm_damping=10, gain=0.5),
                    DampingTask(cost=1.0),
                    NullSpacePostureTask(
                        cost=0.5, lm_damping=1,
                        controlled_frames=["g1_29dof_rev_1_0_left_wrist_yaw_link", "g1_29dof_rev_1_0_right_wrist_yaw_link"],
                        controlled_joints=[
                            "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint",
                            "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint",
                            "waist_yaw_joint", "waist_pitch_joint", "waist_roll_joint",
                        ],
                        gain=0.5,
                    ),
                ],
                fixed_input_tasks=[],
                xr_enabled=bool(carb.settings.get_settings().get("/app/xr/enabled")),
            ),
            enable_gravity_compensation=False,
        )

        self.actions.gripper_action = None

        # Gripper config for termination check
        self.gripper_joint_names = ["R_index_proximal_joint", "L_index_proximal_joint"]
        self.gripper_open_val = 0.0
        self.gripper_threshold = 0.3

        # SINGLE CUBE (instead of 3 cubes in stack task)
        # Named cube_1 and using Red_block prim path to match stack task pattern
        self.scene.cube_1 = RigidObjectCfg(
            prim_path="/World/envs/env_.*/Red_block",
            init_state=RigidObjectCfg.InitialStateCfg(
                pos=[-4.2, -4.2, 0.84],  # Beyond target zone (further from robot)
                rot=[1, 0, 0, 0],
            ),
            spawn=sim_utils.CuboidCfg(
                size=(0.05, 0.05, 0.05),
                rigid_props=sim_utils.RigidBodyPropertiesCfg(
                    disable_gravity=False,
                    retain_accelerations=False,
                ),
                mass_props=sim_utils.MassPropertiesCfg(mass=0.1),
                collision_props=sim_utils.CollisionPropertiesCfg(
                    collision_enabled=True,
                    contact_offset=0.01,
                    rest_offset=0.0,
                ),
                visual_material=sim_utils.PreviewSurfaceCfg(
                    diffuse_color=(1.0, 0.0, 0.0),  # Red
                    metallic=0,
                ),
                physics_material=sim_utils.RigidBodyMaterialCfg(
                    friction_combine_mode="max",
                    restitution_combine_mode="min",
                    static_friction=1.0,
                    dynamic_friction=0.5,
                    restitution=0.0,
                ),
            ),
        )

        # End-effector frame transformer
        marker_cfg = FRAME_MARKER_CFG.copy()
        marker_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
        marker_cfg.prim_path = "/Visuals/FrameTransformer"
        self.scene.ee_frame = FrameTransformerCfg(
            prim_path="{ENV_REGEX_NS}/Robot/pelvis",
            debug_vis=False,
            visualizer_cfg=marker_cfg,
            target_frames=[
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Robot/left_wrist_yaw_link",
                    name="left_end_effector",
                    offset=OffsetCfg(pos=[0.0, 0.0, 0.0]),
                ),
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Robot/right_wrist_yaw_link",
                    name="right_end_effector",
                    offset=OffsetCfg(pos=[0.0, 0.0, 0.0]),
                ),
            ],
        )

        # Cameras
        self.scene.left_high_camera = CameraCfg(
            prim_path="{ENV_REGEX_NS}/Robot/torso_link/d435_link/left_high_cam",
            update_period=0.033,  # 30Hz instead of every physics step
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
            update_period=0.033,  # 30Hz instead of every physics step
            height=480,
            width=640,
            data_types=["rgb"],
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=7.6, focus_distance=400.0, horizontal_aperture=20.0, clipping_range=(0.1, 2.0),
            ),
            offset=CameraCfg.OffsetCfg(pos=(0, -0.0315, 0), rot=(0.5, 0.5, -0.5, -0.5), convention="opengl"),
        )

        # Sim settings
        self.decimation = 6
        self.episode_length_s = 30
        self.sim.dt = 1 / 120
        self.sim.render_interval = 2

        # Convert USD to URDF for Pink IK controller
        temp_urdf_output_path, temp_urdf_meshes_output_path = ControllerUtils.convert_usd_to_urdf(
            self.scene.robot.spawn.usd_path, self.temp_urdf_dir, force_conversion=True
        )
        self.actions.arm_action.controller.urdf_path = temp_urdf_output_path
        self.actions.arm_action.controller.mesh_path = temp_urdf_meshes_output_path

        # Teleop devices
        self.teleop_devices = DevicesCfg(
            devices={
                "handtracking": OpenXRDeviceCfg(
                    retargeters=[
                        UnitreeG1RetargeterCfg(
                            enable_visualization=False,
                            num_open_xr_hand_joints=2 * 26,
                            sim_device=self.sim.device,
                            hand_joint_names=self.actions.arm_action.hand_joint_names,
                        ),
                    ],
                    sim_device=self.sim.device,
                    xr_cfg=self.xr,
                ),
            }
        )

        # Streaming recorder
        self.recorders = StreamingRecorderManagerCfg(
            dataset_export_dir_path="/workspace/isaaclab/datasets/pick_place_target_g1_inspire",
            enable_rerun=False,  # Disable for better performance
            jpeg_quality=85,     # Lower quality = faster encoding
            frequency=20.0,
            capture_scene_state=True,  # Enable for mimic annotation (captures object poses)
        )
