# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
"""
Script to run Isaac Lab environments with DDS control for policy evaluation.

This script allows users to run any IsaacLab task with DDS functionality,
automatically converting IK-based tasks to direct joint control.

Works analogously to record_demos.py but adds DDS instead of recording.

required arguments:
    --task                    Name of the task.

optional arguments:
    -h, --help                Show this help message and exit
    --robot_type              Robot type (g129, h1_2, etc.). (default: g129)
    --enable_inspire_dds      Enable Inspire hand DDS communication.
    --enable_dex3_dds         Enable Dex3 hand DDS communication.
    --enable_dex1_dds         Enable Dex1 gripper DDS communication.
    --step_hz                 Control frequency in Hz. (default: 100)
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import os
import sys

# CRITICAL: Import pinocchio BEFORE any Isaac Sim/Lab imports!
# Isaac Sim bundles its own TinyXML libraries that conflict with pinocchio's urdfdom.
# If pinocchio loads after Isaac Sim, it links against the wrong library version.
# See: https://github.com/isaac-sim/IsaacLab/issues/1936
import pinocchio  # noqa: F401

# Isaac Lab AppLauncher (must come AFTER pinocchio import)
from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Run Isaac Lab environments with DDS control.")
parser.add_argument("--task", type=str, required=True, help="Name of the task.")
parser.add_argument("--robot_type", type=str, default="g129", help="Robot type (g129, h1_2, etc.)")
parser.add_argument("--enable_inspire_dds", action="store_true", help="Enable Inspire hand DDS")
parser.add_argument("--enable_dex3_dds", action="store_true", help="Enable Dex3 hand DDS")
parser.add_argument("--enable_dex1_dds", action="store_true", help="Enable Dex1 gripper DDS")
parser.add_argument("--enable_wholebody_dds", action="store_true", default=False, help="Enable wholebody DDS")
parser.add_argument("--step_hz", type=int, default=100, help="Control frequency in Hz")
parser.add_argument("--stats_interval", type=float, default=10.0, help="Statistics print interval (seconds)")
parser.add_argument("--action_source", type=str, default="dds", help="Action source (dds, file, etc.)")

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# Validate hand selection
hand_flags = [args_cli.enable_inspire_dds, args_cli.enable_dex3_dds, args_cli.enable_dex1_dds]
if sum(hand_flags) > 1:
    parser.error("Only one hand type can be enabled at a time")

# launch the simulator
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import contextlib
import signal
import time

import gymnasium as gym
import torch

# Add unitree_sim_isaaclab to path for DDS imports
UNITREE_SIM_PATH = os.path.expanduser("~/Bot/unitree/unitree_sim_isaaclab")
if UNITREE_SIM_PATH not in sys.path:
    sys.path.insert(0, UNITREE_SIM_PATH)

# IsaacLab imports
import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg

# DDS imports from unitree_sim_isaaclab
from dds.dds_create import create_dds_objects
from teleimager.image_server import ImageServer
from layeredcontrol.robot_control_system import RobotController, ControlConfig
from action_provider.create_action_provider import create_action_provider
from dds.reset_pose_dds import *
from dds.sim_state_dds import *
from tools.data_json_load import sim_state_to_json
from dds.g1_robot_dds import G1RobotDDS
from dds.inspire_dds import InspireDDS
from tasks.common_observations.camera_state import get_camera_image


def detect_cameras_from_env(env):
    """Detect camera sensors from the environment and build image server config.

    Returns:
        dict: Camera configuration dictionary for ImageServer
    """
    cam_config = {}
    base_zmq_port = 55555
    base_webrtc_port = 60001
    port_offset = 0

    # Get sensors from the environment scene
    sensors = getattr(env.scene, "sensors", {})

    for name, sensor in sensors.items():
        # Check if it's a camera sensor
        if "camera" not in name.lower():
            continue

        print(f"[DDS] Found camera: {name}")

        # Try to get camera dimensions from sensor config
        try:
            height = getattr(sensor.cfg, 'height', 480)
            width = getattr(sensor.cfg, 'width', 640)
        except:
            height, width = 480, 640

        # Determine if binocular based on name patterns
        binocular = "head" in name.lower() or "stereo" in name.lower()

        cam_config[name] = {
            "enable_zmq": True,
            "zmq_port": base_zmq_port + port_offset,
            "enable_webrtc": False,
            "webrtc_port": base_webrtc_port + port_offset,
            "webrtc_codec": "h264",
            "type": "isaacsim",
            "image_shape": [height, width],
            "binocular": binocular,
            "fps": 30,
        }
        port_offset += 1

    if not cam_config:
        print("[DDS] Warning: No cameras found in environment")
    else:
        print(f"[DDS] Configured {len(cam_config)} cameras: {list(cam_config.keys())}")

    return cam_config


def create_image_server_for_env(env):
    """Create an image server configured for the environment's cameras."""
    cam_config = detect_cameras_from_env(env)

    if not cam_config:
        # Fallback to default config if no cameras detected
        print("[DDS] Using fallback camera config")
        cam_config = {
            "front_camera": {
                "enable_zmq": True,
                "zmq_port": 55555,
                "enable_webrtc": False,
                "type": "isaacsim",
                "image_shape": [480, 640],
                "binocular": False,
                "fps": 30,
            }
        }

    server = ImageServer(cam_config, realsense_enable=False, camera_finder_verbose=False, isaacsim_enable=True)
    server.start()
    return server


def make_dds_compatible(env_cfg):
    """Transform env config for DDS compatibility.

    Modifications:
    1. Replace PinkInverseKinematicsActionCfg with JointPositionActionCfg
    2. Disable streaming recorders (avoid permission errors)
    3. Keep all scene/observations/terminations intact
    """
    from isaaclab.envs.mdp.actions.actions_cfg import JointPositionActionCfg

    # Check and replace IK with direct joint position control
    if hasattr(env_cfg, 'actions'):
        if hasattr(env_cfg.actions, 'arm_action'):
            action_type = type(env_cfg.actions.arm_action).__name__
            if 'InverseKinematics' in action_type or 'Pink' in action_type:
                print(f"[DDS] Converting {action_type} → JointPositionActionCfg")
                env_cfg.actions.arm_action = JointPositionActionCfg(
                    asset_name="robot",
                    joint_names=[".*"],
                    scale=1.0,
                    use_default_offset=False,
                )

        # Also check for generic 'action' attribute
        if hasattr(env_cfg.actions, 'action'):
            action_type = type(env_cfg.actions.action).__name__
            if 'InverseKinematics' in action_type or 'Pink' in action_type:
                print(f"[DDS] Converting {action_type} → JointPositionActionCfg")
                env_cfg.actions.action = JointPositionActionCfg(
                    asset_name="robot",
                    joint_names=[".*"],
                    scale=1.0,
                    use_default_offset=True,
                )

    # Disable streaming recorders to avoid permission errors
    if hasattr(env_cfg, 'recorders'):
        if env_cfg.recorders is not None:
            print("[DDS] Disabling recorders")
            env_cfg.recorders = None

    return env_cfg


# SDK's expected joint order for G1_29 (29 body joints)
# This matches the order in unitree_sdk2's G1_29_JointIndex enum
SDK_G1_29_JOINT_NAMES = [
    # Legs (0-11)
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
    # Waist (12-14)
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
    # Left arm (15-21)
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    # Right arm (22-28)
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
]

# Cache for joint mapping (avoid recomputing every frame)
_joint_mapping_cache = {
    "initialized": False,
    "sim_to_sdk_indices": None,  # sim_idx -> sdk_idx mapping
    "sim_joint_names": None,
}


def _build_joint_mapping(sim_joint_names):
    """Build mapping from simulation joint indices to SDK joint indices.

    The SDK expects joints in a specific order (SDK_G1_29_JOINT_NAMES).
    This function creates a mapping so we can reorder simulation joints
    to match the SDK's expected order.

    Returns:
        list: For each SDK index, the corresponding simulation index (or -1 if not found)
    """
    # Create lookup from joint name to simulation index
    sim_name_to_idx = {name: idx for idx, name in enumerate(sim_joint_names)}

    # Build mapping: sdk_idx -> sim_idx
    sdk_to_sim = []
    for sdk_idx, sdk_name in enumerate(SDK_G1_29_JOINT_NAMES):
        if sdk_name in sim_name_to_idx:
            sdk_to_sim.append(sim_name_to_idx[sdk_name])
        else:
            # Joint not found in simulation - will use 0.0
            sdk_to_sim.append(-1)

    return sdk_to_sim


def publish_robot_state_to_dds(env, g1_robot_dds):
    """Extract robot state from environment and publish via DDS.

    This is required for eval scripts (like eval_g1_gr00t.py) that subscribe
    to rt/lowstate to get current robot joint positions.

    IMPORTANT: The SDK expects joints in a specific order (see SDK_G1_29_JOINT_NAMES).
    This function remaps simulation joint positions to match the SDK's expected order.
    """
    global _joint_mapping_cache

    # G1_29 has 35 motor slots in LowState message
    G1_29_NUM_MOTORS = 35

    try:
        robot = env.scene["robot"]
        sim_joint_names = robot.data.joint_names

        # Build joint mapping on first call (or if joint names changed)
        if not _joint_mapping_cache["initialized"] or _joint_mapping_cache["sim_joint_names"] != sim_joint_names:
            _joint_mapping_cache["sim_to_sdk_indices"] = _build_joint_mapping(sim_joint_names)
            _joint_mapping_cache["sim_joint_names"] = sim_joint_names
            _joint_mapping_cache["initialized"] = True
            print(f"[DDS] Built joint mapping for {len(sim_joint_names)} simulation joints -> {len(SDK_G1_29_JOINT_NAMES)} SDK joints")

        sdk_to_sim = _joint_mapping_cache["sim_to_sdk_indices"]

        # Get joint positions, velocities from simulation
        joint_pos_sim = robot.data.joint_pos[0].cpu().numpy().astype(float)
        joint_vel_sim = robot.data.joint_vel[0].cpu().numpy().astype(float)

        # Remap to SDK order
        joint_pos = [0.0] * G1_29_NUM_MOTORS
        joint_vel = [0.0] * G1_29_NUM_MOTORS

        for sdk_idx, sim_idx in enumerate(sdk_to_sim):
            if sim_idx >= 0 and sim_idx < len(joint_pos_sim):
                joint_pos[sdk_idx] = float(joint_pos_sim[sim_idx])
                joint_vel[sdk_idx] = float(joint_vel_sim[sim_idx])
            # else: leave as 0.0 (joint not found in simulation)

        # Torques - use zeros (not critical for state feedback)
        joint_torque = [0.0] * G1_29_NUM_MOTORS

        # Get IMU data from robot root state
        root_quat = robot.data.root_quat_w[0].cpu().numpy().astype(float).tolist()
        root_ang_vel = robot.data.root_ang_vel_w[0].cpu().numpy().astype(float).tolist()

        # Construct IMU data array: [lin_acc(3), quat(4), lin_vel(3), ang_vel(3)]
        imu_data = [0.0, 0.0, 0.0,  # lin_acc placeholder
                    root_quat[0], root_quat[1], root_quat[2], root_quat[3],  # quat [w,x,y,z]
                    0.0, 0.0, 0.0,  # accelerometer placeholder
                    root_ang_vel[0], root_ang_vel[1], root_ang_vel[2]]  # gyroscope

        # Write to DDS shared memory
        g1_robot_dds.write_robot_state(
            joint_pos,
            joint_vel,
            joint_torque,
            imu_data
        )
    except Exception as e:
        # Log error on first occurrence, then silently fail
        if not getattr(publish_robot_state_to_dds, '_error_logged', False):
            print(f"[DDS] Error publishing robot state: {e}")
            publish_robot_state_to_dds._error_logged = True


# Inspire hand joint order expected by SDK's DDS message
# Order: Right hand (0-5), Left hand (6-11)
# Each hand: pinky, ring, middle, index, thumb_pitch, thumb_yaw
SDK_INSPIRE_JOINT_NAMES = [
    # Right hand (indices 0-5 in DDS message)
    "R_pinky_proximal_joint",
    "R_ring_proximal_joint",
    "R_middle_proximal_joint",
    "R_index_proximal_joint",
    "R_thumb_proximal_pitch_joint",
    "R_thumb_proximal_yaw_joint",
    # Left hand (indices 6-11 in DDS message)
    "L_pinky_proximal_joint",
    "L_ring_proximal_joint",
    "L_middle_proximal_joint",
    "L_index_proximal_joint",
    "L_thumb_proximal_pitch_joint",
    "L_thumb_proximal_yaw_joint",
]

# Cache for Inspire hand joint mapping
_inspire_mapping_cache = {
    "initialized": False,
    "sdk_to_sim_indices": None,
    "sim_joint_names": None,
}


def _build_inspire_mapping(sim_joint_names):
    """Build mapping from Inspire SDK joint indices to simulation joint indices."""
    sim_name_to_idx = {name: idx for idx, name in enumerate(sim_joint_names)}

    sdk_to_sim = []
    for sdk_name in SDK_INSPIRE_JOINT_NAMES:
        if sdk_name in sim_name_to_idx:
            sdk_to_sim.append(sim_name_to_idx[sdk_name])
        else:
            sdk_to_sim.append(-1)

    return sdk_to_sim


def publish_inspire_state_to_dds(env, inspire_dds):
    """Extract Inspire hand state from environment and publish via DDS.

    This is required for eval scripts that subscribe to rt/inspire/state
    to get current hand joint positions.

    Inspire hand has 12 joints (6 per hand):
    - Right hand: indices 0-5 (pinky, ring, middle, index, thumb_pitch, thumb_yaw)
    - Left hand: indices 6-11 (same order)
    """
    global _inspire_mapping_cache
    INSPIRE_NUM_MOTORS = 12  # 6 per hand

    try:
        robot = env.scene["robot"]
        sim_joint_names = robot.data.joint_names

        # Build joint mapping on first call
        if not _inspire_mapping_cache["initialized"] or _inspire_mapping_cache["sim_joint_names"] != sim_joint_names:
            _inspire_mapping_cache["sdk_to_sim_indices"] = _build_inspire_mapping(sim_joint_names)
            _inspire_mapping_cache["sim_joint_names"] = sim_joint_names
            _inspire_mapping_cache["initialized"] = True

            # Count how many joints were found
            found = sum(1 for idx in _inspire_mapping_cache["sdk_to_sim_indices"] if idx >= 0)
            print(f"[DDS] Built Inspire hand mapping: {found}/{len(SDK_INSPIRE_JOINT_NAMES)} joints found")

        sdk_to_sim = _inspire_mapping_cache["sdk_to_sim_indices"]

        # Get all joint positions/velocities
        joint_pos_full = robot.data.joint_pos[0].cpu().numpy()
        joint_vel_full = robot.data.joint_vel[0].cpu().numpy()

        # Remap to SDK order
        hand_positions = [0.0] * INSPIRE_NUM_MOTORS
        hand_velocities = [0.0] * INSPIRE_NUM_MOTORS

        for sdk_idx, sim_idx in enumerate(sdk_to_sim):
            if sim_idx >= 0 and sim_idx < len(joint_pos_full):
                hand_positions[sdk_idx] = float(joint_pos_full[sim_idx])
                hand_velocities[sdk_idx] = float(joint_vel_full[sim_idx])

        # Torques - use zeros (not critical for state feedback)
        hand_torques = [0.0] * INSPIRE_NUM_MOTORS

        # Write to DDS shared memory
        inspire_dds.write_inspire_state(
            hand_positions,
            hand_velocities,
            hand_torques
        )
    except Exception as e:
        # Log error on first occurrence, then silently fail
        if not getattr(publish_inspire_state_to_dds, '_error_logged', False):
            print(f"[DDS] Error publishing Inspire state: {e}")
            publish_inspire_state_to_dds._error_logged = True


def setup_signal_handlers(controller, dds_manager=None, image_server=None):
    """Set up signal handlers for clean shutdown."""
    def signal_handler(signum, frame):
        print(f"\nReceived signal {signum}, stopping...")
        try:
            controller.stop()
        except Exception as e:
            print(f"Failed to stop controller: {e}")
        try:
            if dds_manager is not None:
                dds_manager.stop_all_communication()
        except Exception as e:
            print(f"Failed to stop DDS: {e}")
        try:
            if image_server is not None:
                image_server.stop()
        except Exception as e:
            print(f"Failed to stop image server: {e}")

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)


def main():
    """Run the environment with DDS control."""
    print("=" * 60)
    print("run_dds.py - Run Any IsaacLab Task with DDS")
    print(f"Task: {args_cli.task}")
    print(f"Robot type: {args_cli.robot_type}")
    print("=" * 60)

    # Parse environment configuration
    try:
        env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=1)
        env_cfg.env_name = args_cli.task
    except Exception as e:
        print(f"Failed to parse environment configuration: {e}")
        return

    # Apply DDS compatibility transformation
    print("\n[DDS] Applying DDS compatibility transformation...")
    env_cfg = make_dds_compatible(env_cfg)

    # Create environment
    print("\nCreating environment...")
    try:
        env = gym.make(args_cli.task, cfg=env_cfg).unwrapped
        print("Environment created successfully")
    except Exception as e:
        print(f"Failed to create environment: {e}")
        return

    # Reset environment
    env.sim.reset()
    env.reset()

    # Create image server (auto-detects cameras from environment)
    print("\n[DDS] Creating image server...")
    try:
        image_server = create_image_server_for_env(env)
        print("[DDS] Image server created")
    except Exception as e:
        print(f"Failed to create image server: {e}")
        return

    # Create DDS objects
    print("\n[DDS] Creating DDS communication...")
    try:
        reset_pose_dds, sim_state_dds, dds_manager = create_dds_objects(args_cli, env)
        # Get reference to G1RobotDDS for state publishing
        g1_robot_dds = dds_manager.get_object("g129")
        # Get reference to InspireDDS for hand state publishing (if enabled)
        inspire_dds = dds_manager.get_object("inspire") if args_cli.enable_inspire_dds else None
        print("[DDS] DDS communication created")
        if inspire_dds:
            print("[DDS] Inspire hand DDS enabled")
    except Exception as e:
        print(f"Failed to create DDS: {e}")
        return

    # Create control config
    control_config = ControlConfig(
        step_hz=args_cli.step_hz,
        replay_mode=False
    )

    # Create action provider
    print("\n[DDS] Creating action provider...")
    try:
        action_provider = create_action_provider(env, args_cli)
        if action_provider is None:
            print("Action provider creation failed")
            return
        print("[DDS] Action provider created")
    except Exception as e:
        print(f"Failed to create action provider: {e}")
        return

    # Create controller
    controller = RobotController(env, control_config)
    controller.set_action_provider(action_provider)

    # Set up signal handlers
    setup_signal_handlers(controller, dds_manager, image_server)

    print("\n" + "=" * 60)
    print("DDS control ready!")
    print("The simulation is now accepting joint commands via DDS.")
    print("Press Ctrl+C to exit.")
    print("=" * 60 + "\n")

    try:
        # Start controller
        controller.start()

        # Main loop
        last_stats_time = time.time()
        loop_start_time = time.time()
        loop_count = 0

        with contextlib.suppress(KeyboardInterrupt), torch.inference_mode():
            while simulation_app.is_running() and controller.is_running:
                current_time = time.time()
                loop_count += 1

                # Publish robot state to DDS (required for eval scripts like eval_g1_gr00t.py)
                if g1_robot_dds:
                    publish_robot_state_to_dds(env, g1_robot_dds)

                # Publish Inspire hand state to DDS (required for hand control)
                if inspire_dds:
                    publish_inspire_state_to_dds(env, inspire_dds)

                # Write camera images to shared memory (for eval scripts)
                # This calls the camera_state observation function which writes to shared memory
                try:
                    get_camera_image(env)
                except Exception:
                    pass  # Camera observation is optional

                # Get and publish sim state
                try:
                    env_state = env.scene.get_state()
                    env_state_json = sim_state_to_json(env_state)
                    sim_state = {"init_state": env_state_json, "task_name": args_cli.task}
                    sim_state_dds.write_sim_state_data(sim_state)
                except Exception as e:
                    print(f"Failed to get/write env state: {e}")

                # Check for reset commands
                try:
                    reset_pose_cmd = reset_pose_dds.get_reset_pose_command()
                    if reset_pose_cmd is not None:
                        reset_category = reset_pose_cmd.get("reset_category")
                        if reset_category == '1':
                            print("[DDS] Reset object requested")
                            env_cfg.event_manager.trigger("reset_object_self", env)
                            reset_pose_dds.write_reset_pose_command(-1)
                        elif reset_category == '2':
                            print("[DDS] Reset all requested")
                            env_cfg.event_manager.trigger("reset_all_self", env)
                            reset_pose_dds.write_reset_pose_command(-1)
                except Exception as e:
                    print(f"Failed to process reset command: {e}")

                # Execute control step
                controller.step()

                # Check if simulation stopped
                if env.sim.is_stopped():
                    print("Simulation stopped")
                    break

    except KeyboardInterrupt:
        print("\nUser interrupted")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        print("\nCleaning up...")
        controller.cleanup()
        image_server.stop()
        env.close()
        print("Cleanup completed")


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
