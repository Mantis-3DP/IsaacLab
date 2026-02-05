# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for G1 robot with Inspire hands in the cube stacking task."""

import os
import tempfile
import torch

import carb

# Path to local assets (copied from unitree_sim_isaaclab)
G1_INSPIRE_ASSETS_DIR = os.path.join(os.path.dirname(__file__), "assets")
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
from isaaclab.sensors import CameraCfg, FrameTransformerCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg
from isaaclab.utils import configclass

from isaaclab_tasks.manager_based.manipulation.stack import mdp
from isaaclab_tasks.manager_based.manipulation.stack.mdp import franka_stack_events
from isaaclab_tasks.manager_based.manipulation.stack.stack_env_cfg import StackEnvCfg
from isaaclab_tasks.manager_based.manipulation.pick_place import mdp as pick_place_mdp

from isaaclab.markers.config import FRAME_MARKER_CFG  # isort: skip
from isaaclab_assets.robots.unitree import G1_INSPIRE_FTP_CFG  # isort: skip




@configclass
class G1ObservationsCfg:
    """Observation specifications for G1 robot (compatible with 57 joints)."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group with state values."""

        actions = ObsTerm(func=mdp.last_action)
        # Use base_mdp.joint_pos instead of mdp.joint_pos_rel for G1's 57 joints
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
        # Cube observations
        cube_positions = ObsTerm(func=mdp.cube_positions_in_world_frame)
        cube_orientations = ObsTerm(func=mdp.cube_orientations_in_world_frame)
        # End-effector observations
        eef_pos = ObsTerm(func=mdp.ee_frame_pos)
        eef_quat = ObsTerm(func=mdp.ee_frame_quat)
        # Separate left/right EEF observations (for mimic env)
        left_eef_pos = ObsTerm(func=pick_place_mdp.get_eef_pos, params={"link_name": "left_wrist_yaw_link"})
        left_eef_quat = ObsTerm(func=pick_place_mdp.get_eef_quat, params={"link_name": "left_wrist_yaw_link"})
        right_eef_pos = ObsTerm(func=pick_place_mdp.get_eef_pos, params={"link_name": "right_wrist_yaw_link"})
        right_eef_quat = ObsTerm(func=pick_place_mdp.get_eef_quat, params={"link_name": "right_wrist_yaw_link"})
        # Camera observations (head stereo cameras) - normalize=False for raw RGB [0,1]
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

    @configclass
    class RGBCameraPolicyCfg(ObsGroup):
        """Observations for policy group with RGB images (head stereo only - matches real robot)."""

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

    @configclass
    class SubtaskCfg(ObsGroup):
        """Observations for subtask group."""

        grasp_1 = ObsTerm(
            func=mdp.object_grasped,
            params={
                "robot_cfg": SceneEntityCfg("robot"),
                "ee_frame_cfg": SceneEntityCfg("ee_frame"),
                "object_cfg": SceneEntityCfg("cube_2"),
            },
        )
        stack_1 = ObsTerm(
            func=mdp.object_stacked,
            params={
                "robot_cfg": SceneEntityCfg("robot"),
                "upper_object_cfg": SceneEntityCfg("cube_2"),
                "lower_object_cfg": SceneEntityCfg("cube_1"),
            },
        )
        grasp_2 = ObsTerm(
            func=mdp.object_grasped,
            params={
                "robot_cfg": SceneEntityCfg("robot"),
                "ee_frame_cfg": SceneEntityCfg("ee_frame"),
                "object_cfg": SceneEntityCfg("cube_3"),
            },
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False

    # observation groups
    policy: PolicyCfg = PolicyCfg()
    rgb_camera: RGBCameraPolicyCfg = RGBCameraPolicyCfg()
    subtask_terms: SubtaskCfg = SubtaskCfg()


@configclass
class EventCfg:
    """Configuration for events."""

    reset_g1_arm_pose = EventTerm(
        func=franka_stack_events.set_default_joint_pose,
        mode="reset",
        params={
            # Default pose for G1 arms (14 arm joints + waist + legs + 24 hand joints)
            # We set idle pose for all joints
            "default_pose": [0.0] * 57,  # All joints at 0 position
        },
    )

    randomize_cube_positions = EventTerm(
        func=franka_stack_events.randomize_object_pose,
        mode="reset",
        params={
            # Position range for cubes on the packing table in warehouse
            # Table center is around (-4.15, -4.08), z=0.84 is table height
            "pose_range": {
                "x": (-4.3, -4.0),
                "y": (-4.11, -3.91),  # Shifted 4cm right (+Y)
                "z": (0.84, 0.84),
                "yaw": (-0.5, 0.5),
            },
            "min_separation": 0.10,
            "asset_cfgs": [SceneEntityCfg("cube_1"), SceneEntityCfg("cube_2"), SceneEntityCfg("cube_3")],
        },
    )


@configclass
class G1InspireCubeStackEnvCfg(StackEnvCfg):
    """Configuration for the G1 robot with Inspire hands cube stacking task."""

    # Override observations as class attribute (required for camera entities to be linked)
    observations: G1ObservationsCfg = G1ObservationsCfg()

    # Position of the XR anchor in the world frame
    # Z=0 so user's physical height maps directly to simulation height
    xr: XrCfg = XrCfg(
        anchor_pos=(-4.2, -3.7, 0.0),
        anchor_rot=(0.0, 0.0, 0.0, 1.0),  # 180° around Z-axis
    )

    # Temporary directory for URDF files
    temp_urdf_dir = tempfile.gettempdir()

    # Idle action for Pink IK controller
    # Action format: [left arm pos (3), left arm quat (4), right arm pos (3), right arm quat (4),
    #                 left hand joint pos (12), right hand joint pos (12)]
    idle_action = torch.tensor([
        # Left arm EEF position and quaternion
        -0.1487, 0.2038, 1.0952, 0.707, 0.0, 0.0, 0.707,
        # Right arm EEF position and quaternion
        0.1487, 0.2038, 1.0952, 0.707, 0.0, 0.0, 0.707,
        # 24 hand joints (all zeros for open hands)
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    ])

    def __post_init__(self):
        # Call parent post_init first
        super().__post_init__()

        # Increase env spacing for warehouse environment (default 2.5 is too small)
        self.scene.env_spacing = 15.0

        # Set events for G1
        self.events = EventCfg()

        # Set G1 robot with Inspire hands - positioned in the warehouse environment
        self.scene.robot = G1_INSPIRE_FTP_CFG.replace(
            prim_path="/World/envs/env_.*/Robot",
            init_state=ArticulationCfg.InitialStateCfg(
                pos=(-4.2, -3.7, 0.76),  # Position in warehouse near the table
                rot=(0.7071, 0, 0, -0.7071),  # Facing the table
                joint_pos={
                    # Arms with natural posture bias (from G1_29DOF_CFG)
                    # Note: left/right shoulder_roll are mirrored (opposite signs)
                    "right_shoulder_pitch_joint": -0.35,
                    "right_shoulder_roll_joint": -0.16,  # Mirrored from left
                    "right_shoulder_yaw_joint": 0.0,
                    "right_wrist_yaw_joint": 0.0,
                    "right_wrist_roll_joint": 0.0,
                    "right_wrist_pitch_joint": 0.0,
                    "left_shoulder_pitch_joint": -0.35,
                    "left_shoulder_roll_joint": 0.16,  # Arm slightly outward
                    "left_shoulder_yaw_joint": 0.0,
                    "left_wrist_yaw_joint": 0.0,
                    "left_wrist_roll_joint": 0.0,
                    "left_wrist_pitch_joint": 0.0,
                    # Waist, legs, and hands at default
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
            init_state=AssetBaseCfg.InitialStateCfg(
                pos=[0.0, 0.0, 0.0],
                rot=[1.0, 0.0, 0.0, 0.0],
            ),
            spawn=UsdFileCfg(
                usd_path=f"{G1_INSPIRE_ASSETS_DIR}/small_warehouse_digital_twin/small_warehouse_digital_twin.usd",
            ),
        )

        # Add packing table with yellow box
        self.scene.packing_table = AssetBaseCfg(
            prim_path="/World/envs/env_.*/PackingTable",
            init_state=AssetBaseCfg.InitialStateCfg(
                pos=[-4.3, -4.2, -0.2],
                rot=[1.0, 0.0, 0.0, 0.0],
            ),
            spawn=UsdFileCfg(
                usd_path=f"{G1_INSPIRE_ASSETS_DIR}/table_with_yellowbox.usd",
            ),
        )

        # Configure actions for G1 with Pink IK controller
        self.actions.arm_action = PinkInverseKinematicsActionCfg(
            pink_controlled_joint_names=[
                ".*_shoulder_pitch_joint",
                ".*_shoulder_roll_joint",
                ".*_shoulder_yaw_joint",
                ".*_elbow_joint",
                ".*_wrist_yaw_joint",
                ".*_wrist_roll_joint",
                ".*_wrist_pitch_joint",
            ],
            hand_joint_names=[
                # All the drive and mimic joints, total 24 joints
                "L_index_proximal_joint",
                "L_middle_proximal_joint",
                "L_pinky_proximal_joint",
                "L_ring_proximal_joint",
                "L_thumb_proximal_yaw_joint",
                "R_index_proximal_joint",
                "R_middle_proximal_joint",
                "R_pinky_proximal_joint",
                "R_ring_proximal_joint",
                "R_thumb_proximal_yaw_joint",
                "L_index_intermediate_joint",
                "L_middle_intermediate_joint",
                "L_pinky_intermediate_joint",
                "L_ring_intermediate_joint",
                "L_thumb_proximal_pitch_joint",
                "R_index_intermediate_joint",
                "R_middle_intermediate_joint",
                "R_pinky_intermediate_joint",
                "R_ring_intermediate_joint",
                "R_thumb_proximal_pitch_joint",
                "L_thumb_intermediate_joint",
                "R_thumb_intermediate_joint",
                "L_thumb_distal_joint",
                "R_thumb_distal_joint",
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
                    LocalFrameTask(
                        "g1_29dof_rev_1_0_left_wrist_yaw_link",
                        base_link_frame_name="g1_29dof_rev_1_0_pelvis",
                        position_cost=8.0,
                        orientation_cost=2.0,
                        lm_damping=10,  # Higher damping prevents elbow flip
                        gain=0.5,
                    ),
                    LocalFrameTask(
                        "g1_29dof_rev_1_0_right_wrist_yaw_link",
                        base_link_frame_name="g1_29dof_rev_1_0_pelvis",
                        position_cost=8.0,
                        orientation_cost=2.0,
                        lm_damping=10,  # Higher damping prevents elbow flip
                        gain=0.5,
                    ),
                    DampingTask(
                        cost=1.0,  # Penalizes large velocity changes - prevents sudden elbow flip
                    ),
                    NullSpacePostureTask(
                        cost=0.5,
                        lm_damping=1,
                        controlled_frames=[
                            "g1_29dof_rev_1_0_left_wrist_yaw_link",
                            "g1_29dof_rev_1_0_right_wrist_yaw_link",
                        ],
                        controlled_joints=[
                            # Shoulders only - NO elbow joints (matches NVIDIA's official config)
                            # This lets IK determine elbow position naturally without fighting the null space
                            "left_shoulder_pitch_joint",
                            "left_shoulder_roll_joint",
                            "left_shoulder_yaw_joint",
                            "right_shoulder_pitch_joint",
                            "right_shoulder_roll_joint",
                            "right_shoulder_yaw_joint",
                            "waist_yaw_joint",
                            "waist_pitch_joint",
                            "waist_roll_joint",
                        ],
                        gain=0.5,  # Increased from 0.3 for faster posture adjustment
                    ),
                ],
                fixed_input_tasks=[],
                xr_enabled=bool(carb.settings.get_settings().get("/app/xr/enabled")),
            ),
            enable_gravity_compensation=False,
        )

        # For G1 Inspire, hand control is integrated into Pink IK - no separate gripper action needed
        self.actions.gripper_action = None

        # For dexterous hands, we don't use binary gripper - set dummy values for termination check
        # The termination function checks if gripper is open
        self.gripper_joint_names = ["R_index_proximal_joint", "L_index_proximal_joint"]
        self.gripper_open_val = 0.0  # Open position for Inspire hand fingers
        self.gripper_threshold = 0.3  # Wider threshold for dexterous hands

        # Define colored cuboid blocks on the packing table (matching unitree_sim_isaaclab positions)
        # Red block (cube_1)
        self.scene.cube_1 = RigidObjectCfg(
            prim_path="/World/envs/env_.*/Red_block",
            init_state=RigidObjectCfg.InitialStateCfg(
                pos=[-4.05, -4.0, 0.84],
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

        # Yellow block (cube_2)
        self.scene.cube_2 = RigidObjectCfg(
            prim_path="/World/envs/env_.*/Yellow_block",
            init_state=RigidObjectCfg.InitialStateCfg(
                pos=[-4.2, -4.0, 0.84],
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
                    diffuse_color=(1.0, 1.0, 0.0),  # Yellow
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

        # Green block (cube_3)
        self.scene.cube_3 = RigidObjectCfg(
            prim_path="/World/envs/env_.*/Green_block",
            init_state=RigidObjectCfg.InitialStateCfg(
                pos=[-4.12, -4.1, 0.84],
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
                    diffuse_color=(0.0, 1.0, 0.0),  # Green
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

        # End-effector frame transformer for observations
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

        # ==================== CAMERAS ====================
        # Head-mounted stereo cameras (following Franka visuomotor pattern)
        self.scene.left_high_camera = CameraCfg(
            prim_path="{ENV_REGEX_NS}/Robot/torso_link/d435_link/left_high_cam",
            update_period=0.0,
            height=480,
            width=640,
            data_types=["rgb"],
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=7.6,
                focus_distance=400.0,
                horizontal_aperture=20.0,
                clipping_range=(0.1, 2.0),
            ),
            offset=CameraCfg.OffsetCfg(
                pos=(0, 0.0315, 0),
                rot=(0.5, 0.5, -0.5, -0.5),
                convention="opengl",
            ),
        )

        self.scene.right_high_camera = CameraCfg(
            prim_path="{ENV_REGEX_NS}/Robot/torso_link/d435_link/right_high_cam",
            update_period=0.0,
            height=480,
            width=640,
            data_types=["rgb"],
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=7.6,
                focus_distance=400.0,
                horizontal_aperture=20.0,
                clipping_range=(0.1, 2.0),
            ),
            offset=CameraCfg.OffsetCfg(
                pos=(0, -0.0315, 0),
                rot=(0.5, 0.5, -0.5, -0.5),
                convention="opengl",
            ),
        )

        # Update episode length and sim settings
        self.decimation = 6
        self.episode_length_s = 45  
        self.sim.dt = 1 / 120  # 120Hz physics
        self.sim.render_interval = 2

        # Convert USD to URDF for Pink IK controller
        temp_urdf_output_path, temp_urdf_meshes_output_path = ControllerUtils.convert_usd_to_urdf(
            self.scene.robot.spawn.usd_path, self.temp_urdf_dir, force_conversion=True
        )
        self.actions.arm_action.controller.urdf_path = temp_urdf_output_path
        self.actions.arm_action.controller.mesh_path = temp_urdf_meshes_output_path

        # Configure teleop devices for hand tracking
        self.teleop_devices = DevicesCfg(
            devices={
                "handtracking": OpenXRDeviceCfg(
                    retargeters=[
                        UnitreeG1RetargeterCfg(
                            enable_visualization=True,
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

        # ==================== RECORDER ====================
        # Configure IsaacLab's streaming recording system (like Unitree xr_teleoperate)
        # Uses JPEG compression + incremental JSON for fast, low-memory recording
        # Benefits: No memory buildup, near-instant saves, 10-20x smaller storage, rerun.io viz
        self.recorders = StreamingRecorderManagerCfg(
            dataset_export_dir_path="/workspace/isaaclab/datasets/stack_g1_inspire",
            enable_rerun=True,
            jpeg_quality=95,
            frequency=20.0,  # Match control rate: 120Hz / 6 decimation = 20Hz
            capture_scene_state=True,  # Enable for mimic annotation (captures object poses)
        )
