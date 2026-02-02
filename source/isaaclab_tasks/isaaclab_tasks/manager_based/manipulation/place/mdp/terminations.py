# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Common functions that can be used to activate certain terminations for the place task.

The functions can be passed to the :class:`isaaclab.managers.TerminationTermCfg` object to enable
the termination introduced by the function.
"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def object_placed_upright(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg,
    object_cfg: SceneEntityCfg,
    target_height: float = 0.927,
    euler_xy_threshold: float = 0.10,
):
    """Check if an object placed upright by the specified robot."""

    robot: Articulation = env.scene[robot_cfg.name]
    object: RigidObject = env.scene[object_cfg.name]

    # Compute mug euler angles of X, Y axis, to check if it is placed upright
    object_euler_x, object_euler_y, _ = math_utils.euler_xyz_from_quat(object.data.root_quat_w)  # (N,4) [0, 2*pi]

    object_euler_x_err = torch.abs(math_utils.wrap_to_pi(object_euler_x))  # (N,)
    object_euler_y_err = torch.abs(math_utils.wrap_to_pi(object_euler_y))  # (N,)

    success = torch.logical_and(object_euler_x_err < euler_xy_threshold, object_euler_y_err < euler_xy_threshold)

    # Check if current mug height is greater than target height
    height_success = object.data.root_pos_w[:, 2] > target_height

    success = torch.logical_and(height_success, success)

    if hasattr(env.scene, "surface_grippers") and len(env.scene.surface_grippers) > 0:
        surface_gripper = env.scene.surface_grippers["surface_gripper"]
        suction_cup_status = surface_gripper.state.view(-1, 1)  # 1: closed, 0: closing, -1: open
        suction_cup_is_open = (suction_cup_status == -1).to(torch.float32)
        success = torch.logical_and(suction_cup_is_open, success)

    else:
        if hasattr(env.cfg, "gripper_joint_names"):
            gripper_joint_ids, _ = robot.find_joints(env.cfg.gripper_joint_names)
            success = torch.logical_and(
                success,
                torch.abs(torch.abs(robot.data.joint_pos[:, gripper_joint_ids[0]]) - env.cfg.gripper_open_val)
                < env.cfg.gripper_threshold,
            )
            success = torch.logical_and(
                success,
                torch.abs(torch.abs(robot.data.joint_pos[:, gripper_joint_ids[1]]) - env.cfg.gripper_open_val)
                < env.cfg.gripper_threshold,
            )
        else:
            raise ValueError("No gripper_joint_names found in environment config")

    return success


def object_a_is_into_b(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_a_cfg: SceneEntityCfg = SceneEntityCfg("object_a"),
    object_b_cfg: SceneEntityCfg = SceneEntityCfg("object_b"),
    xy_threshold: float = 0.03,  # xy_distance_threshold
    height_threshold: float = 0.04,  # height_distance_threshold
    height_diff: float = 0.0,  # expected height_diff
) -> torch.Tensor:
    """Check if an object a is put into another object b by the specified robot."""

    robot: Articulation = env.scene[robot_cfg.name]
    object_a: RigidObject = env.scene[object_a_cfg.name]
    object_b: RigidObject = env.scene[object_b_cfg.name]

    # check object a is into object b
    pos_diff = object_a.data.root_pos_w - object_b.data.root_pos_w
    height_dist = torch.linalg.vector_norm(pos_diff[:, 2:], dim=1)
    xy_dist = torch.linalg.vector_norm(pos_diff[:, :2], dim=1)

    success = torch.logical_and(xy_dist < xy_threshold, (height_dist - height_diff) < height_threshold)

    # Check gripper positions
    if hasattr(env.scene, "surface_grippers") and len(env.scene.surface_grippers) > 0:
        surface_gripper = env.scene.surface_grippers["surface_gripper"]
        suction_cup_status = surface_gripper.state.view(-1, 1)  # 1: closed, 0: closing, -1: open
        suction_cup_is_open = (suction_cup_status == -1).to(torch.float32)
        success = torch.logical_and(suction_cup_is_open, success)

    else:
        if hasattr(env.cfg, "gripper_joint_names"):
            gripper_joint_ids, _ = robot.find_joints(env.cfg.gripper_joint_names)
            assert len(gripper_joint_ids) == 2, "Terminations only support parallel gripper for now"

            success = torch.logical_and(
                success,
                torch.abs(torch.abs(robot.data.joint_pos[:, gripper_joint_ids[0]]) - env.cfg.gripper_open_val)
                < env.cfg.gripper_threshold,
            )
            success = torch.logical_and(
                success,
                torch.abs(torch.abs(robot.data.joint_pos[:, gripper_joint_ids[1]]) - env.cfg.gripper_open_val)
                < env.cfg.gripper_threshold,
            )
        else:
            raise ValueError("No gripper_joint_names found in environment config")

    return success


def cube_in_target_zone(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    cube_cfg: SceneEntityCfg = SceneEntityCfg("cube_1"),
    target_x: tuple[float, float] = (-4.35, -4.25),
    target_y: tuple[float, float] = (-3.85, -3.75),
    atol=0.0001,
    rtol=0.0001,
) -> torch.Tensor:
    """Check if cube is placed in target zone and optionally if arms are at rest.

    Args:
        env: The environment instance.
        robot_cfg: Configuration for the robot.
        cube_cfg: Configuration for the cube.
        target_x: X bounds (min, max) of target zone.
        target_y: Y bounds (min, max) of target zone.
        check_arms_at_rest: Whether to also check if arms returned to rest position.
        arm_rest_threshold: Max deviation from rest position (radians).

    Returns:
        Boolean tensor indicating success for each environment.
    """
    robot: Articulation = env.scene[robot_cfg.name]
    cube: RigidObject = env.scene[cube_cfg.name]

    # Check cube is in target zone
    cube_pos = cube.data.root_pos_w
    x_ok = (target_x[0] <= cube_pos[:, 0]) & (cube_pos[:, 0] <= target_x[1])
    y_ok = (target_y[0] <= cube_pos[:, 1]) & (cube_pos[:, 1] <= target_y[1])
    in_zone = x_ok & y_ok

    # Check gripper positions
    if hasattr(env.scene, "surface_grippers") and len(env.scene.surface_grippers) > 0:
        surface_gripper = env.scene.surface_grippers["surface_gripper"]
        suction_cup_status = surface_gripper.state.view(-1, 1)  # 1: closed, 0: closing, -1: open
        suction_cup_is_open = (suction_cup_status == -1).to(torch.float32)
        in_zone = torch.logical_and(suction_cup_is_open, in_zone)

    else:
        if hasattr(env.cfg, "gripper_joint_names"):
            gripper_joint_ids, _ = robot.find_joints(env.cfg.gripper_joint_names)
            assert len(gripper_joint_ids) == 2, "Terminations only support parallel gripper for now"

            in_zone = torch.logical_and(
                torch.isclose(
                    robot.data.joint_pos[:, gripper_joint_ids[0]],
                    torch.tensor(env.cfg.gripper_open_val, dtype=torch.float32).to(env.device),
                    atol=atol,
                    rtol=rtol,
                ),
                in_zone,
            )
            in_zone = torch.logical_and(
                torch.isclose(
                    robot.data.joint_pos[:, gripper_joint_ids[1]],
                    torch.tensor(env.cfg.gripper_open_val, dtype=torch.float32).to(env.device),
                    atol=atol,
                    rtol=rtol,
                ),
                in_zone,
            )
        else:
            raise ValueError("No gripper_joint_names found in environment config")

    return in_zone
