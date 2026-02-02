# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""G1 Inspire configurations for the place task."""

import gymnasium as gym

##
# Register Gym environments.
##

gym.register(
    id="Isaac-Place-Cube-G1-Inspire-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.place_g1_inspire_env_cfg:G1InspireCubePlaceEnvCfg",
    },
    disable_env_checker=True,
)
