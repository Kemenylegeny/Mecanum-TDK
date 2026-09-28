# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Rewards following Rudin et al., "Advanced Skills by Learning Locomotion and Local Navigation End-to-End" (2022).

The task reward is only given at the end of the episode, so the policy is free to choose its path and speed; the
penalties are provided throughout the episode and do not compete with the task as long as the goal is reached in time.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.assets import Articulation
from isaaclab.managers import ManagerTermBase, RewardTermCfg, SceneEntityCfg

from .observations import footprint_clearance

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def goal_distance(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    """Planar distance between the robot base and the goal of the pose command [m].

    Uses the world-frame goal and the current root position, so it is not lagging one step behind like the
    base-frame command (which is only updated after the rewards are computed).
    """
    command = env.command_manager.get_term(command_name)
    return torch.linalg.norm(command.pos_command_w[:, :2] - command.robot.data.root_pos_w[:, :2], dim=1)


def final_position(
    env: ManagerBasedRLEnv, command_name: str, reward_duration: float, std: float = 1.0
) -> torch.Tensor:
    """Task reward, eq. (1): ``1 / T_r * 1 / (1 + (d / std)^2)`` during the last ``T_r`` seconds, else 0.

    ``std = 1`` is the paper; a smaller ``std`` pays less for stopping short of the goal (at d = 1 m: 0.5 with
    std = 1, 0.2 with std = 0.5). Since the reward manager multiplies by the step time, its episode sum is at most 1
    (robot at the goal during the whole window), i.e. ``weight`` after weighting.
    """
    time = env.episode_length_buf * env.step_dt
    active = time > env.max_episode_length_s - reward_duration
    return active.float() / reward_duration / (1.0 + (goal_distance(env, command_name) / std) ** 2)


class exploration_bias(ManagerTermBase):
    """Exploration reward, eq. (3): cosine between the base velocity and the direction to the goal.

    Removed automatically once the task reward reaches ``threshold`` (50 %) of its maximum, measured as a running
    mean of the (unweighted) episode sums of :func:`final_position` over the finished episodes.
    """

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self._task_sum = torch.zeros(env.num_envs, device=env.device)
        self._task_mean = 0.0
        self.active = True

    def reset(self, env_ids=None):
        env_ids = slice(None) if env_ids is None else env_ids
        finished = self._task_sum[env_ids]
        if self.active and finished.numel() > 0:
            self._task_mean = 0.99 * self._task_mean + 0.01 * finished.mean().item()
            if self._task_mean >= self.cfg.params.get("threshold", 0.5):
                self.active = False
                print(f"[INFO] exploration_bias removed: task reward reached {self._task_mean:.2f} of its maximum.")
        self._task_sum[env_ids] = 0.0
        # visible in the training logs / W&B: whether the bias is still given, and the removal criterion
        # (running mean of the task reward as a fraction of its maximum; the bias is removed at `threshold`)
        log = self._env.extras.setdefault("log", {})
        log["Metrics/exploration_bias/active"] = float(self.active)
        log["Metrics/exploration_bias/task_reward_fraction"] = self._task_mean

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        command_name: str,
        reward_duration: float,
        threshold: float = 0.5,
        std: float = 1.0,
    ) -> torch.Tensor:
        self._task_sum += final_position(env, command_name, reward_duration, std) * env.step_dt
        if not self.active:
            return torch.zeros(env.num_envs, device=env.device)
        command = env.command_manager.get_term(command_name)
        vel = command.robot.data.root_lin_vel_w[:, :2]
        to_goal = command.pos_command_w[:, :2] - command.robot.data.root_pos_w[:, :2]
        return torch.sum(vel * to_goal, dim=1) / (torch.linalg.norm(vel, dim=1) * torch.linalg.norm(to_goal, dim=1) + 1e-6)


def stalling(
    env: ManagerBasedRLEnv, command_name: str, speed_threshold: float = 0.1, distance_threshold: float = 0.5
) -> torch.Tensor:
    """Stalling indicator, eq. (4): 1 while the robot is slower than 0.1 m/s farther than 0.5 m from the goal.

    Counteracts the discount factor, which otherwise makes the policy wait and rush to the goal at the last moment.
    """
    speed = torch.linalg.norm(env.command_manager.get_term(command_name).robot.data.root_lin_vel_w[:, :2], dim=1)
    return ((speed < speed_threshold) & (goal_distance(env, command_name) > distance_threshold)).float()


def wheel_slip_l2(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    wheel_radius: float,
    wheel_base_half_length: float,
    track_half_width: float,
    wheel_joint_signs: tuple[float, float, float, float] = (1.0, 1.0, 1.0, 1.0),
) -> torch.Tensor:
    """Sum of squared wheel slip speeds [m^2/s^2]: rim speed minus the rolling speed the base motion requires.

    The wheeled counterpart of the feet acceleration penalty: slipping wheels waste energy and wear the rollers.
    ``asset_cfg.joint_ids`` must be the wheels in the order FL, FR, RL, RR.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    v = asset.data.root_lin_vel_b
    wz = asset.data.root_ang_vel_b[:, 2]
    k = wheel_base_half_length + track_half_width
    rolling = torch.stack(
        [v[:, 0] - v[:, 1] - k * wz, v[:, 0] + v[:, 1] + k * wz, v[:, 0] + v[:, 1] - k * wz, v[:, 0] - v[:, 1] + k * wz],
        dim=1,
    )
    signs = torch.tensor(wheel_joint_signs, device=env.device)
    rim = asset.data.joint_vel[:, asset_cfg.joint_ids] * signs * wheel_radius
    return torch.sum((rim - rolling) ** 2, dim=1)


def obstacle_proximity(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    max_distance: float,
    half_extents: tuple[float, float],
    safe_clearance: float,
) -> torch.Tensor:
    """Penalty in [0, 1] growing linearly as the gap between the robot rectangle and the closest pillar gets below
    ``safe_clearance``."""
    clearance = footprint_clearance(env, sensor_cfg, max_distance, half_extents).min(dim=1).values
    return torch.clamp(safe_clearance - clearance, min=0.0, max=safe_clearance) / safe_clearance
