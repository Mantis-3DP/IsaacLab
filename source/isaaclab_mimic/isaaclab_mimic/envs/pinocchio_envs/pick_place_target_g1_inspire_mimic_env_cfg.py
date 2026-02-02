# Copyright (c) 2024-2026, The Isaac Lab Project Developers.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Mimic environment config for G1 Inspire pick-place-target task."""

from isaaclab.envs.mimic_env_cfg import MimicEnvCfg, SubTaskConfig
from isaaclab.utils import configclass

from isaaclab_tasks.manager_based.manipulation.pick_place_target.config.g1_inspire.pick_place_target_g1_inspire_env_cfg import (
    G1InspirePickPlaceTargetEnvCfg,
)


@configclass
class PickPlaceTargetG1InspireMimicEnvCfg(G1InspirePickPlaceTargetEnvCfg, MimicEnvCfg):
    """Mimic environment config for G1 Inspire pick-place-target task.

    This enables data augmentation via Isaac Lab Mimic for the pick-place-target task.
    Subtasks: grasp cube -> place in target zone
    """

    def __post_init__(self):
        # Call parent post-init
        super().__post_init__()

        # Override datagen config values for demonstration generation
        self.datagen_config.name = "g1_inspire_pick_place_target_D0"
        self.datagen_config.generation_guarantee = True
        self.datagen_config.generation_keep_failed = False
        self.datagen_config.generation_num_trials = 1000
        self.datagen_config.generation_select_src_per_subtask = False
        self.datagen_config.generation_select_src_per_arm = False
        self.datagen_config.generation_relative = False
        self.datagen_config.generation_joint_pos = False
        self.datagen_config.generation_transform_first_robot_pose = False
        self.datagen_config.generation_interpolate_from_last_target_pose = True
        self.datagen_config.max_num_failures = 25
        self.datagen_config.num_demo_to_render = 10
        self.datagen_config.num_fail_demo_to_render = 25
        self.datagen_config.seed = 1

        # Subtask configs for right arm (primary manipulation arm)
        # Task: grasp cube_1 -> place in target zone
        subtask_configs = []
        subtask_configs.append(
            SubTaskConfig(
                # Subtask 1: Grasp the cube
                object_ref="cube_1",
                # For manual annotation: press "S" when grasp is complete
                # Signal name used if automatic annotation is enabled
                subtask_term_signal="grasp",
                first_subtask_start_offset_range=(0, 0),
                subtask_term_offset_range=(0, 0),
                selection_strategy="nearest_neighbor_object",
                selection_strategy_kwargs={"nn_k": 3},
                action_noise=0.003,
                num_interpolation_steps=0,
                num_fixed_steps=0,
                apply_noise_during_interpolation=False,
            )
        )
        subtask_configs.append(
            SubTaskConfig(
                # Subtask 2: Place in target zone (final subtask)
                object_ref="cube_1",
                subtask_term_signal=None,  # Final subtask - no signal needed
                subtask_term_offset_range=(0, 0),
                selection_strategy="nearest_neighbor_object",
                selection_strategy_kwargs={"nn_k": 3},
                action_noise=0.003,
                num_interpolation_steps=3,
                num_fixed_steps=0,
                apply_noise_during_interpolation=False,
            )
        )
        self.subtask_configs["right"] = subtask_configs

        # Subtask configs for left arm (idle during this task)
        subtask_configs = []
        subtask_configs.append(
            SubTaskConfig(
                object_ref="cube_1",
                subtask_term_signal=None,
                subtask_term_offset_range=(0, 0),
                selection_strategy="nearest_neighbor_object",
                selection_strategy_kwargs={"nn_k": 3},
                action_noise=0.003,
                num_interpolation_steps=0,
                num_fixed_steps=0,
                apply_noise_during_interpolation=False,
            )
        )
        self.subtask_configs["left"] = subtask_configs
