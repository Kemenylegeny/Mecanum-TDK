# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

from isaaclab.terrains import TerrainImporter

from .rewards import goal_distance

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def terrain_levels_goal(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    command_name: str,
    success_radius: float,
    fail_radius: float,
    crash_terms: Sequence[str] = ("collision", "flipped"),
) -> torch.Tensor:
    """Terrain curriculum based on how the episode ended (more pillars on higher levels).

    Called before the reset, so the robot pose and the goal still belong to the finished episode:

    * move up: the robot ended the episode within ``success_radius`` of the goal,
    * move down: the robot crashed (``crash_terms``) or ended farther than ``fail_radius`` from the goal.
    """
    terrain: TerrainImporter = env.scene.terrain
    distance = goal_distance(env, command_name)[env_ids]
    move_up = distance < success_radius
    crashed = torch.zeros_like(move_up)
    for term in crash_terms:
        crashed |= env.termination_manager.get_term(term)[env_ids]
    move_down = ~move_up & (crashed | (distance > fail_radius))
    terrain.update_env_origins(env_ids, move_up, move_down)
    return torch.mean(terrain.terrain_levels.float())
