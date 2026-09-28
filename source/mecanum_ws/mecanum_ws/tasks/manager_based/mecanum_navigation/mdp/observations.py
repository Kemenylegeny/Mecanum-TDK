# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import RayCaster

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv, ManagerBasedRLEnv


def lidar_distances(env: ManagerBasedEnv, sensor_cfg: SceneEntityCfg, max_distance: float) -> torch.Tensor:
    """Distance of every lidar ray to the first hit, clipped to ``max_distance``. Shape is (num_envs, num_rays).

    Rays that hit nothing are reported as ``max_distance``.
    """
    sensor: RayCaster = env.scene.sensors[sensor_cfg.name]
    dist = torch.linalg.norm(sensor.data.ray_hits_w - sensor.data.pos_w.unsqueeze(1), dim=-1)
    # missed rays are returned as inf
    return torch.nan_to_num(dist, nan=max_distance, posinf=max_distance).clamp(max=max_distance)


def footprint_clearance(
    env: ManagerBasedEnv, sensor_cfg: SceneEntityCfg, max_distance: float, half_extents: tuple[float, float]
) -> torch.Tensor:
    """Per lidar ray, free distance between the robot footprint rectangle and the first hit [m]. Shape (num_envs, rays).

    The lidar is yaw-aligned and sits at the base center, so the ray angle is relative to the robot heading and the
    rectangle boundary along a ray is ``min(hx / |cos a|, hy / |sin a|)``. Negative values mean penetration.
    """
    sensor: RayCaster = env.scene.sensors[sensor_cfg.name]
    directions = sensor.ray_directions[0, :, :2]
    directions = directions / torch.linalg.norm(directions, dim=1, keepdim=True)
    hx, hy = half_extents
    boundary = torch.minimum(hx / directions[:, 0].abs().clamp(min=1e-6), hy / directions[:, 1].abs().clamp(min=1e-6))
    return lidar_distances(env, sensor_cfg, max_distance) - boundary.unsqueeze(0)


def goal_position_b(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    """Goal position (x, y) in the yaw-aligned base frame. Shape is (num_envs, 2)."""
    return env.command_manager.get_command(command_name)[:, :2]


def time_left(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Remaining episode time as a fraction of the episode length. Shape is (num_envs, 1).

    The task reward is only given at the end of the episode, so the policy (and the critic) must know how much time
    is left to reach the goal.
    """
    return (1.0 - env.episode_length_buf.float() / env.max_episode_length).unsqueeze(1)
