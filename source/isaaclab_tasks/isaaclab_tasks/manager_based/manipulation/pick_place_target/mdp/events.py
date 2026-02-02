# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Custom event functions for pick-place-target task."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

import isaaclab.sim as sim_utils
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


# Prim paths for the yellow target cubes within the packing table USD
TARGET_CUBE_SUBPATHS = [
    "PackingTable_2/Cube",
    "PackingTable_2/Cube_01",
    "PackingTable_2/Cube_02",
    "PackingTable_2/Cube_03",
]


def _set_target_cubes_visibility(env: ManagerBasedEnv, env_idx: int, visible: bool):
    """Set visibility of the yellow target cubes in the packing table.

    Args:
        env: The environment instance.
        env_idx: The environment index.
        visible: Whether to make the cubes visible or hidden.
    """
    # Get the packing table base prim path for this environment
    base_path = f"/World/envs/env_{env_idx}/PackingTable"

    for subpath in TARGET_CUBE_SUBPATHS:
        full_path = f"{base_path}/{subpath}"
        prim = sim_utils.get_prim_at_path(full_path)
        if prim and prim.IsValid():
            sim_utils.set_prim_visibility(prim, visible=visible)


def reset_target_visibility(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
):
    """Hide target cubes on episode reset.

    Args:
        env: The environment instance.
        env_ids: The environment indices to reset.
    """
    for idx in env_ids.tolist():
        _set_target_cubes_visibility(env, idx, visible=False)


def reveal_target_on_lift(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    block_cfg: SceneEntityCfg,
    height_threshold: float = 0.1,
):
    """Show target cubes when block is lifted above threshold height.

    Args:
        env: The environment instance.
        env_ids: The environment indices (unused, checks all envs).
        block_cfg: The block asset configuration.
        height_threshold: Height above initial position to trigger reveal.
    """
    block = env.scene[block_cfg.name]

    # Check if block is above threshold height (picked up)
    block_z = block.data.root_pos_w[:, 2]
    initial_z = 0.84  # Table surface + half block height
    is_lifted = block_z > (initial_z + height_threshold)

    # Show target cubes for environments where block is lifted
    for idx in range(env.num_envs):
        if is_lifted[idx]:
            _set_target_cubes_visibility(env, idx, visible=True)


# Keep old functions for backwards compatibility
def reset_cover_visibility(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    cover_cfg: SceneEntityCfg,
):
    """Make cover visible on episode reset (legacy - uses grey cover asset).

    Args:
        env: The environment instance.
        env_ids: The environment indices to reset.
        cover_cfg: The cover asset configuration.
    """
    cover = env.scene[cover_cfg.name]
    for idx in env_ids.tolist():
        prim = cover.prims[idx]
        sim_utils.set_prim_visibility(prim, visible=True)
