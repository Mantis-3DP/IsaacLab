# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for G1 robot with Inspire hands for locomanipulation cube stacking task.

This combines the locomanipulation capabilities (Agile locomotion + upper body IK) with
the warehouse block stacking task, using Inspire 5-finger dexterous hands.

Architecture:
- Lower body: Controlled by Agile RL policy for locomotion
- Upper body + hands: Controlled by Pink IK for manipulation
- Scene: Warehouse environment with RGB/Yellow/Green blocks for stacking
"""

import os

import isaaclab.envs.mdp as base_mdp
import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.controllers.pink_ik.local_frame_task import LocalFrameTask
from isaaclab.controllers.pink_ik.null_space_posture_task import NullSpacePostureTask
from isaaclab.controllers.pink_ik.pink_ik_cfg import PinkIKControllerCfg
from isaaclab.devices.device_base import DevicesCfg
from isaaclab.devices.openxr import OpenXRDeviceCfg, XrCfg
from isaaclab.devices.openxr.retargeters.humanoid.unitree.g1_lower_body_standing import (
    G1LowerBodyStandingRetargeterCfg,
)
from isaaclab.devices.openxr.retargeters.humanoid.unitree.inspire.g1_upper_body_retargeter import (
    UnitreeG1RetargeterCfg,
)
from isaaclab.devices.openxr.xr_cfg import XrAnchorRotationMode
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.envs.mdp.actions.pink_actions_cfg import PinkInverseKinematicsActionCfg
from isaaclab.envs.mdp.recorders.recorders_cfg import ActionStateRecorderManagerCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.managers.recorder_manager import DatasetExportMode
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import CameraCfg, FrameTransformerCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR, retrieve_file_path

from isaaclab.markers.config import FRAME_MARKER_CFG  # isort: skip

from isaaclab_tasks.manager_based.locomanipulation.pick_place.configs.action_cfg import (
    AgileBasedLowerBodyActionCfg,
)
from isaaclab_tasks.manager_based.locomanipulation.pick_place.configs.agile_locomotion_observation_cfg import (
    AgileTeacherPolicyObservationsCfg,
)
from isaaclab_tasks.manager_based.manipulation.pick_place import mdp as manip_mdp
from isaaclab_tasks.manager_based.manipulation.stack import mdp
from isaaclab_tasks.manager_based.manipulation.stack.mdp import franka_stack_events

# Path to local assets (warehouse, table)
G1_INSPIRE_ASSETS_DIR = os.path.join(os.path.dirname(__file__), "assets")

# Path to G1 Inspire wholebody USD (floating base, supports locomanipulation)
# This USD is mounted from unitree_sim_isaaclab via docker-compose.unitree-assets.patch.yaml
# Mount target: /workspace/unitree_assets
G1_INSPIRE_WHOLEBODY_USD = (
    "/workspace/unitree_assets/robots/g1-29dof_wholebody_inspire/g1_29dof_with_inspire_rev_1_0.usd"
)

# Inspire hand joint names (24 joints total: 12 per hand)
INSPIRE_HAND_JOINT_NAMES = [
    # Left hand (12 joints)
    "L_index_proximal_joint",
    "L_index_intermediate_joint",
    "L_middle_proximal_joint",
    "L_middle_intermediate_joint",
    "L_ring_proximal_joint",
    "L_ring_intermediate_joint",
    "L_pinky_proximal_joint",
    "L_pinky_intermediate_joint",
    "L_thumb_proximal_yaw_joint",
    "L_thumb_proximal_pitch_joint",
    "L_thumb_intermediate_joint",
    "L_thumb_distal_joint",
    # Right hand (12 joints)
    "R_index_proximal_joint",
    "R_index_intermediate_joint",
    "R_middle_proximal_joint",
    "R_middle_intermediate_joint",
    "R_ring_proximal_joint",
    "R_ring_intermediate_joint",
    "R_pinky_proximal_joint",
    "R_pinky_intermediate_joint",
    "R_thumb_proximal_yaw_joint",
    "R_thumb_proximal_pitch_joint",
    "R_thumb_intermediate_joint",
    "R_thumb_distal_joint",
]


##
# Pink IK Controller Configuration for G1 with Inspire hands
##
G1_INSPIRE_UPPER_BODY_IK_CONTROLLER_CFG = PinkIKControllerCfg(
    articulation_name="robot",
    base_link_name="pelvis",
    num_hand_joints=24,  # Inspire has 24 hand joints (12 per hand)
    show_ik_warnings=True,
    fail_on_joint_limit_violation=False,
    variable_input_tasks=[
        LocalFrameTask(
            "g1_29dof_with_hand_rev_1_0_left_wrist_yaw_link",
            base_link_frame_name="g1_29dof_with_hand_rev_1_0_pelvis",
            position_cost=8.0,
            orientation_cost=2.0,
            lm_damping=10,
            gain=0.5,
        ),
        LocalFrameTask(
            "g1_29dof_with_hand_rev_1_0_right_wrist_yaw_link",
            base_link_frame_name="g1_29dof_with_hand_rev_1_0_pelvis",
            position_cost=8.0,
            orientation_cost=2.0,
            lm_damping=10,
            gain=0.5,
        ),
        NullSpacePostureTask(
            cost=0.5,
            lm_damping=1,
            controlled_frames=[
                "g1_29dof_with_hand_rev_1_0_left_wrist_yaw_link",
                "g1_29dof_with_hand_rev_1_0_right_wrist_yaw_link",
            ],
            controlled_joints=[
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
            gain=0.3,
        ),
    ],
    fixed_input_tasks=[],
)


##
# Pink IK Action Configuration for G1 with Inspire hands
##
G1_INSPIRE_UPPER_BODY_IK_ACTION_CFG = PinkInverseKinematicsActionCfg(
    pink_controlled_joint_names=[
        ".*_shoulder_pitch_joint",
        ".*_shoulder_roll_joint",
        ".*_shoulder_yaw_joint",
        ".*_elbow_joint",
        ".*_wrist_pitch_joint",
        ".*_wrist_roll_joint",
        ".*_wrist_yaw_joint",
        "waist_.*_joint",
    ],
    hand_joint_names=INSPIRE_HAND_JOINT_NAMES,
    target_eef_link_names={
        "left_wrist": "left_wrist_yaw_link",
        "right_wrist": "right_wrist_yaw_link",
    },
    asset_name="robot",
    controller=G1_INSPIRE_UPPER_BODY_IK_CONTROLLER_CFG,
)


##
# G1 Inspire Wholebody Robot Configuration
##
G1_INSPIRE_WHOLEBODY_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=G1_INSPIRE_WHOLEBODY_USD,
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=True,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=8,   # Increased from 4 for stability
            solver_velocity_iteration_count=4,   # Increased from 1 for stability
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(-4.2, -3.7, 0.76),  # Position in warehouse near the table
        rot=(0.7071, 0, 0, -0.7071),  # Facing the table (90° rotation)
        joint_pos={
            # Leg joints for standing
            ".*_hip_pitch_joint": -0.20,
            ".*_knee_joint": 0.42,
            ".*_ankle_pitch_joint": -0.23,
            # Arm joints
            ".*_elbow_joint": 0.87,
            "left_shoulder_roll_joint": 0.18,
            "left_shoulder_pitch_joint": 0.35,
            "right_shoulder_roll_joint": -0.18,
            "right_shoulder_pitch_joint": 0.35,
            # Inspire hand joints (all at 0)
            "L_.*_joint": 0.0,
            "R_.*_joint": 0.0,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.90,
    actuators={
        "legs": ImplicitActuatorCfg(
            joint_names_expr=[
                ".*_hip_yaw_joint",
                ".*_hip_roll_joint",
                ".*_hip_pitch_joint",
                ".*_knee_joint",
                ".*waist.*",
            ],
            stiffness={
                ".*_hip_yaw_joint": 150.0,
                ".*_hip_roll_joint": 150.0,
                ".*_hip_pitch_joint": 200.0,
                ".*_knee_joint": 200.0,
                ".*waist.*": 5000.0,  # Increased from 200 - waist needs high stiffness
            },
            damping={
                ".*_hip_yaw_joint": 5.0,
                ".*_hip_roll_joint": 5.0,
                ".*_hip_pitch_joint": 5.0,
                ".*_knee_joint": 5.0,
                ".*waist.*": 5.0,
            },
        ),
        "feet": ImplicitActuatorCfg(
            joint_names_expr=[".*_ankle_pitch_joint", ".*_ankle_roll_joint"],
            stiffness=20.0,
            damping=2.0,
        ),
        "shoulders": ImplicitActuatorCfg(
            joint_names_expr=[
                ".*_shoulder_pitch_joint",
                ".*_shoulder_roll_joint",
            ],
            stiffness=100.0,
            damping=2.0,
        ),
        "arms": ImplicitActuatorCfg(
            joint_names_expr=[
                ".*_shoulder_yaw_joint",
                ".*_elbow_joint",
            ],
            stiffness=50.0,
            damping=2.0,
        ),
        "wrist": ImplicitActuatorCfg(
            joint_names_expr=[".*_wrist_.*"],
            stiffness=40.0,
            damping=2.0,
        ),
        "hands": ImplicitActuatorCfg(
            joint_names_expr=[
                ".*_index_proximal_joint",
                ".*_index_intermediate_joint",
                ".*_middle_proximal_joint",
                ".*_middle_intermediate_joint",
                ".*_pinky_proximal_joint",
                ".*_pinky_intermediate_joint",
                ".*_ring_proximal_joint",
                ".*_ring_intermediate_joint",
                ".*_thumb_proximal_yaw_joint",
                ".*_thumb_proximal_pitch_joint",
                ".*_thumb_intermediate_joint",
                ".*_thumb_distal_joint",
            ],
            effort_limit=100.0,
            velocity_limit=50,
            stiffness=1000.0,
            damping=15.0,
        ),
    },
)


##
# Scene definition
##
@configclass
class LocomanipulationInspireStackSceneCfg(InteractiveSceneCfg):
    """Scene configuration for G1 Inspire locomanipulation block stacking in warehouse."""

    # G1 robot with Inspire hands - supports floating base for locomanipulation
    robot: ArticulationCfg = G1_INSPIRE_WHOLEBODY_CFG.replace(
        prim_path="/World/envs/env_.*/Robot",
    )

    # Warehouse room environment
    room_walls = AssetBaseCfg(
        prim_path="/World/envs/env_.*/Room",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[0.0, 0.0, 0.0], rot=[1.0, 0.0, 0.0, 0.0]),
        spawn=UsdFileCfg(
            usd_path=f"{G1_INSPIRE_ASSETS_DIR}/small_warehouse_digital_twin/small_warehouse_digital_twin.usd",
        ),
    )

    # Packing table - positioned in warehouse
    packing_table = AssetBaseCfg(
        prim_path="/World/envs/env_.*/PackingTable",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[-4.3, -4.2, -0.2], rot=[1.0, 0.0, 0.0, 0.0]),
        spawn=UsdFileCfg(usd_path=f"{G1_INSPIRE_ASSETS_DIR}/table_with_yellowbox.usd"),
    )

    # Blocks are defined in __post_init__ with proper physics - these are placeholders
    cube_1: RigidObjectCfg = None
    cube_2: RigidObjectCfg = None
    cube_3: RigidObjectCfg = None

    ground = AssetBaseCfg(prim_path="/World/GroundPlane", spawn=sim_utils.GroundPlaneCfg())
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )

    ee_frame: FrameTransformerCfg = None


##
# Action configuration
##
@configclass
class ActionsCfg:
    """Action specifications for G1 Inspire locomanipulation stacking task."""

    # Upper body IK (arms + waist + Inspire hands)
    upper_body_ik = G1_INSPIRE_UPPER_BODY_IK_ACTION_CFG

    # Lower body locomotion via Agile RL policy
    lower_body_joint_pos = AgileBasedLowerBodyActionCfg(
        asset_name="robot",
        joint_names=[".*_hip_.*_joint", ".*_knee_joint", ".*_ankle_.*_joint"],
        policy_output_scale=0.25,
        obs_group_name="lower_body_policy",
        policy_path=f"{ISAACLAB_NUCLEUS_DIR}/Policies/Agile/agile_locomotion.pt",
    )


##
# Observation configuration
##
@configclass
class ObservationsCfg:
    """Observation specifications for G1 Inspire locomanipulation stacking task."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy."""

        actions = ObsTerm(func=manip_mdp.last_action)
        robot_joint_pos = ObsTerm(
            func=base_mdp.joint_pos, params={"asset_cfg": SceneEntityCfg("robot")}
        )
        robot_joint_vel = ObsTerm(
            func=base_mdp.joint_vel, params={"asset_cfg": SceneEntityCfg("robot")}
        )
        robot_root_pos = ObsTerm(
            func=base_mdp.root_pos_w, params={"asset_cfg": SceneEntityCfg("robot")}
        )
        robot_root_rot = ObsTerm(
            func=base_mdp.root_quat_w, params={"asset_cfg": SceneEntityCfg("robot")}
        )
        cube_positions = ObsTerm(func=mdp.cube_positions_in_world_frame)
        cube_orientations = ObsTerm(func=mdp.cube_orientations_in_world_frame)
        eef_pos = ObsTerm(func=mdp.ee_frame_pos)
        eef_quat = ObsTerm(func=mdp.ee_frame_quat)
        # Inspire hand joint state (24 joints)
        hand_joint_state = ObsTerm(
            func=manip_mdp.get_robot_joint_state,
            params={"joint_names": ["L_.*_joint", "R_.*_joint"]},
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False

    @configclass
    class SubtaskCfg(ObsGroup):
        """Observations for subtask tracking."""

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

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False

    policy: PolicyCfg = PolicyCfg()
    subtask_terms: SubtaskCfg = SubtaskCfg()
    lower_body_policy: AgileTeacherPolicyObservationsCfg = AgileTeacherPolicyObservationsCfg()


##
# Event configuration
##
@configclass
class EventCfg:
    """Configuration for events.

    Note: Robot pose is NOT reset here - the standing pose from ArticulationCfg.InitialStateCfg
    is required for the Agile locomotion policy to maintain balance.
    """

    # Randomize cube positions on table in front of robot at (-4.2, -3.7)
    # Table surface at z~0.87, blocks spawn in reachable area
    randomize_cube_positions = EventTerm(
        func=franka_stack_events.randomize_object_pose,
        mode="reset",
        params={
            "pose_range": {"x": (-4.3, -4.05), "y": (-4.0, -3.85), "z": (0.87, 0.87), "yaw": (-0.5, 0.5)},
            "min_separation": 0.06,
            "asset_cfgs": [
                SceneEntityCfg("cube_1"),
                SceneEntityCfg("cube_2"),
                SceneEntityCfg("cube_3"),
            ],
        },
    )


##
# Termination configuration
##
@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    cube_dropped = DoneTerm(
        func=base_mdp.root_height_below_minimum,
        params={"minimum_height": 0.5, "asset_cfg": SceneEntityCfg("cube_1")},
    )


##
# Environment configuration
##
@configclass
class G1InspireLocomanipulationStackEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for G1 Inspire locomanipulation block stacking in warehouse.

    Uses G1 with Inspire 5-finger hands for locomanipulation. The Agile locomotion
    policy controls the lower body while Pink IK handles upper body manipulation.
    """

    scene: LocomanipulationInspireStackSceneCfg = LocomanipulationInspireStackSceneCfg(
        num_envs=1, env_spacing=5.0, replicate_physics=True
    )
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    events: EventCfg = EventCfg()
    terminations: TerminationsCfg = TerminationsCfg()

    commands = None
    rewards = None
    curriculum = None

    # XR anchor at robot position for VR control (z=0 so user height maps directly)
    xr: XrCfg = XrCfg(anchor_pos=(-4.2, -3.7, 0.0), anchor_rot=(0.0, 0.0, 0.0, 1.0))

    def __post_init__(self):
        """Post initialization."""
        self.decimation = 4
        self.episode_length_s = 45.0
        self.sim.dt = 1 / 200
        self.sim.render_interval = 2

        # Set URDF path for IK controller - use local Inspire kinematics URDF
        # This URDF has the correct joint names matching the Inspire USD
        urdf_path = "/workspace/unitree_assets/robots/g1-29dof_wholebody_inspire/g1_29dof_with_inspire_kinematics.urdf"
        self.actions.upper_body_ik.controller.urdf_path = urdf_path

        # Gripper configuration for grasp detection (uses index fingers as simplified gripper)
        self.gripper_joint_names = ["R_index_proximal_joint", "L_index_proximal_joint"]
        self.gripper_open_val = 0.0  # Open position for Inspire hand fingers
        self.gripper_threshold = 0.3  # Wider threshold for dexterous hands

        # Override blocks with proper physics properties (ensures they spawn correctly)
        # Robot at (-4.2, -3.7) - blocks positioned in front of robot within arm reach
        self.scene.cube_1 = RigidObjectCfg(
            prim_path="/World/envs/env_.*/Red_block",
            init_state=RigidObjectCfg.InitialStateCfg(pos=[-4.1, -3.95, 0.87], rot=[1, 0, 0, 0]),
            spawn=sim_utils.CuboidCfg(
                size=(0.05, 0.05, 0.05),
                rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=False, retain_accelerations=False),
                mass_props=sim_utils.MassPropertiesCfg(mass=0.1),
                collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True, contact_offset=0.01, rest_offset=0.0),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0), metallic=0),
                physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=1.0, dynamic_friction=0.5, restitution=0.0),
            ),
        )
        self.scene.cube_2 = RigidObjectCfg(
            prim_path="/World/envs/env_.*/Yellow_block",
            init_state=RigidObjectCfg.InitialStateCfg(pos=[-4.25, -3.92, 0.87], rot=[1, 0, 0, 0]),
            spawn=sim_utils.CuboidCfg(
                size=(0.05, 0.05, 0.05),
                rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=False, retain_accelerations=False),
                mass_props=sim_utils.MassPropertiesCfg(mass=0.1),
                collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True, contact_offset=0.01, rest_offset=0.0),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 1.0, 0.0), metallic=0),
                physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=1.0, dynamic_friction=0.5, restitution=0.0),
            ),
        )
        self.scene.cube_3 = RigidObjectCfg(
            prim_path="/World/envs/env_.*/Green_block",
            init_state=RigidObjectCfg.InitialStateCfg(pos=[-4.18, -3.98, 0.87], rot=[1, 0, 0, 0]),
            spawn=sim_utils.CuboidCfg(
                size=(0.05, 0.05, 0.05),
                rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=False, retain_accelerations=False),
                mass_props=sim_utils.MassPropertiesCfg(mass=0.1),
                collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True, contact_offset=0.01, rest_offset=0.0),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0), metallic=0),
                physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=1.0, dynamic_friction=0.5, restitution=0.0),
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

        # Teleop devices for VR control with Inspire hand retargeter
        self.teleop_devices = DevicesCfg(
            devices={
                "handtracking": OpenXRDeviceCfg(
                    retargeters=[
                        UnitreeG1RetargeterCfg(
                            enable_visualization=True,
                            num_open_xr_hand_joints=52,
                            sim_device=self.sim.device,
                            hand_joint_names=INSPIRE_HAND_JOINT_NAMES,
                        ),
                        G1LowerBodyStandingRetargeterCfg(sim_device=self.sim.device),
                    ],
                    sim_device=self.sim.device,
                    xr_cfg=self.xr,
                ),
            }
        )

        # Head camera
        self.scene.head_camera = CameraCfg(
            prim_path="/World/envs/env_.*/Robot/head_link/head_cam",
            update_period=0.02,
            height=480,
            width=640,
            data_types=["rgb"],
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=7.6, focus_distance=400.0, horizontal_aperture=20.0
            ),
            offset=CameraCfg.OffsetCfg(
                pos=(0.1, 0.0, 0.0), rot=(0.5, 0.5, -0.5, -0.5), convention="opengl"
            ),
        )

        # Recorder
        self.recorders = ActionStateRecorderManagerCfg(
            dataset_export_dir_path="/workspace/isaaclab/datasets/stack_g1_inspire_locomanipulation",
            dataset_filename="stack_episodes",
            dataset_export_mode=DatasetExportMode.EXPORT_ALL,
        )