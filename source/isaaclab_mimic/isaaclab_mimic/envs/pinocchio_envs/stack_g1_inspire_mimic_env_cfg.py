# Copyright (c) 2024-2026, The Isaac Lab Project Developers.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Mimic environment config for G1 Inspire stack task."""

from isaaclab.envs.mimic_env_cfg import MimicEnvCfg, SubTaskConfig
from isaaclab.utils import configclass

from isaaclab_tasks.manager_based.manipulation.stack.config.g1_inspire.stack_g1_inspire_env_cfg import (
    G1InspireCubeStackEnvCfg,
)


@configclass
class StackG1InspireMimicEnvCfg(G1InspireCubeStackEnvCfg, MimicEnvCfg):
    """Mimic environment config for G1 Inspire stack task.

    6-phase task (press S 5 times during annotation):
    1. grasp cube_1 → S
    2. place cube_1 in stacking position → S
    3. grasp cube_2 → S
    4. stack cube_2 on cube_1 → S
    5. grasp cube_3 → S
    6. stack cube_3 on cube_2 (final)
    """

    def __post_init__(self):
        super().__post_init__()

        # Datagen config
        self.datagen_config.name = "g1_inspire_stack_D0"
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

        # Subtask configs for right arm
        # 6-phase stack task: grasp → place → grasp → stack → grasp → stack
        subtask_configs = []

        # Phase 1: Grasp cube_1 (base cube)
        subtask_configs.append(
            SubTaskConfig(
                object_ref="cube_1",
                subtask_term_signal="grasp_0",  # Press S when cube_1 is grasped
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

        # Phase 2: Place cube_1 in stacking position
        subtask_configs.append(
            SubTaskConfig(
                object_ref="cube_1",
                subtask_term_signal="place_0",  # Press S when cube_1 is placed
                subtask_term_offset_range=(0, 0),
                selection_strategy="nearest_neighbor_object",
                selection_strategy_kwargs={"nn_k": 3},
                action_noise=0.003,
                num_interpolation_steps=3,
                num_fixed_steps=0,
                apply_noise_during_interpolation=False,
            )
        )

        # Phase 3: Grasp cube_2
        subtask_configs.append(
            SubTaskConfig(
                object_ref="cube_2",
                subtask_term_signal="grasp_1",  # Press S when cube_2 is grasped
                subtask_term_offset_range=(0, 0),
                selection_strategy="nearest_neighbor_object",
                selection_strategy_kwargs={"nn_k": 3},
                action_noise=0.003,
                num_interpolation_steps=3,
                num_fixed_steps=0,
                apply_noise_during_interpolation=False,
            )
        )

        # Phase 4: Stack cube_2 on cube_1
        subtask_configs.append(
            SubTaskConfig(
                object_ref="cube_1",
                subtask_term_signal="stack_1",  # Press S when cube_2 is stacked
                subtask_term_offset_range=(0, 0),
                selection_strategy="nearest_neighbor_object",
                selection_strategy_kwargs={"nn_k": 3},
                action_noise=0.003,
                num_interpolation_steps=3,
                num_fixed_steps=0,
                apply_noise_during_interpolation=False,
            )
        )

        # Phase 5: Grasp cube_3
        subtask_configs.append(
            SubTaskConfig(
                object_ref="cube_3",
                subtask_term_signal="grasp_2",  # Press S when cube_3 is grasped
                subtask_term_offset_range=(0, 0),
                selection_strategy="nearest_neighbor_object",
                selection_strategy_kwargs={"nn_k": 3},
                action_noise=0.003,
                num_interpolation_steps=3,
                num_fixed_steps=0,
                apply_noise_during_interpolation=False,
            )
        )

        # Phase 6: Stack cube_3 on cube_2 (final)
        subtask_configs.append(
            SubTaskConfig(
                object_ref="cube_2",
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

        # Left arm idle
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
