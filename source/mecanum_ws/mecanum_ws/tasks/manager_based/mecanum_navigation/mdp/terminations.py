# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.managers import SceneEntityCfg

from .observations import footprint_clearance

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def obstacle_collision(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    max_distance: float,
    half_extents: tuple[float, float],
    margin: float = 0.03,
) -> torch.Tensor:
    """Terminate when an obstacle is closer than ``margin`` to the robot footprint rectangle, measured by the lidar.

    This is robot-agnostic (no contact sensor / body names needed): the lidar sits at the base center and only
    sees the obstacle meshes. See :func:`footprint_clearance`.
    """
    return footprint_clearance(env, sensor_cfg, max_distance, half_extents).min(dim=1).values < margin
