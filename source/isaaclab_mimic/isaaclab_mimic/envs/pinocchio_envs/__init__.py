# Copyright (c) 2024-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Sub-package with environment wrappers for Isaac Lab Mimic."""

import gymnasium as gym

gym.register(
    id="Isaac-PickPlace-GR1T2-Abs-Mimic-v0",
    entry_point=f"{__name__}.pickplace_gr1t2_mimic_env:PickPlaceGR1T2MimicEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.pickplace_gr1t2_mimic_env_cfg:PickPlaceGR1T2MimicEnvCfg",
    },
    disable_env_checker=True,
)

gym.register(
    id="Isaac-PickPlace-GR1T2-WaistEnabled-Abs-Mimic-v0",
    entry_point=f"{__name__}.pickplace_gr1t2_mimic_env:PickPlaceGR1T2MimicEnv",
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.pickplace_gr1t2_waist_enabled_mimic_env_cfg:PickPlaceGR1T2WaistEnabledMimicEnvCfg"
        ),
    },
    disable_env_checker=True,
)

gym.register(
    id="Isaac-NutPour-GR1T2-Pink-IK-Abs-Mimic-v0",
    entry_point=f"{__name__}.pickplace_gr1t2_mimic_env:PickPlaceGR1T2MimicEnv",
    kwargs={"env_cfg_entry_point": f"{__name__}.nutpour_gr1t2_mimic_env_cfg:NutPourGR1T2MimicEnvCfg"},
    disable_env_checker=True,
)

gym.register(
    id="Isaac-ExhaustPipe-GR1T2-Pink-IK-Abs-Mimic-v0",
    entry_point=f"{__name__}.pickplace_gr1t2_mimic_env:PickPlaceGR1T2MimicEnv",
    kwargs={"env_cfg_entry_point": f"{__name__}.exhaustpipe_gr1t2_mimic_env_cfg:ExhaustPipeGR1T2MimicEnvCfg"},
    disable_env_checker=True,
)

gym.register(
    id="Isaac-Locomanipulation-G1-Abs-Mimic-v0",
    entry_point=f"{__name__}.locomanipulation_g1_mimic_env:LocomanipulationG1MimicEnv",
    kwargs={"env_cfg_entry_point": f"{__name__}.locomanipulation_g1_mimic_env_cfg:LocomanipulationG1MimicEnvCfg"},
    disable_env_checker=True,
)

gym.register(
    id="Isaac-PickPlaceTarget-G1-Inspire-Mimic-v0",
    entry_point=f"{__name__}.pick_place_target_g1_inspire_mimic_env:PickPlaceTargetG1InspireMimicEnv",
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.pick_place_target_g1_inspire_mimic_env_cfg:PickPlaceTargetG1InspireMimicEnvCfg"
        ),
    },
    disable_env_checker=True,
)

gym.register(
    id="Isaac-Stack-G1-Inspire-Mimic-v0",
    entry_point=f"{__name__}.pick_place_target_g1_inspire_mimic_env:PickPlaceTargetG1InspireMimicEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.stack_g1_inspire_mimic_env_cfg:StackG1InspireMimicEnvCfg",
    },
    disable_env_checker=True,
)
