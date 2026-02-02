# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""G1 robot with Inspire hands configuration for cube stacking task."""

import gymnasium as gym

##
# Register Gym environments.
##

gym.register(
    id="Isaac-Stack-Cube-G1-Inspire-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.stack_g1_inspire_env_cfg:G1InspireCubeStackEnvCfg",
    },
    disable_env_checker=True,
)

gym.register(
    id="Isaac-Stack-Cube-G1-Inspire-Locomanipulation-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.stack_g1_locomanipulation_env_cfg:G1InspireLocomanipulationStackEnvCfg",
    },
    disable_env_checker=True,
)
