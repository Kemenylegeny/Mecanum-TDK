# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import MISSING
from typing import TYPE_CHECKING

import torch

from isaaclab.assets.articulation import Articulation
from isaaclab.managers.action_manager import ActionTerm, ActionTermCfg
from isaaclab.utils import configclass

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


class MecanumBaseVelocityAction(ActionTerm):
    r"""Maps a base twist action :math:`(v_x, v_y, \omega_z)` to the four mecanum wheel velocity targets.

    Uses the standard inverse kinematics of an "X"-configuration mecanum platform:

    .. math::

        \omega_{fl} = (v_x - v_y - (l_x + l_y)\,\omega_z) / r \\
        \omega_{fr} = (v_x + v_y + (l_x + l_y)\,\omega_z) / r \\
        \omega_{rl} = (v_x + v_y - (l_x + l_y)\,\omega_z) / r \\
        \omega_{rr} = (v_x - v_y + (l_x + l_y)\,\omega_z) / r

    The raw action is scaled by :attr:`MecanumBaseVelocityActionCfg.scale` (max. base velocities).
    """

    cfg: MecanumBaseVelocityActionCfg
    _asset: Articulation

    def __init__(self, cfg: MecanumBaseVelocityActionCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)

        self._joint_ids, self._joint_names = self._asset.find_joints(self.cfg.wheel_joint_names, preserve_order=True)
        if len(self._joint_ids) != 4:
            raise ValueError(
                f"Expected 4 wheel joints (FL, FR, RL, RR), got {len(self._joint_ids)}: {self._joint_names}"
            )

        self._raw_actions = torch.zeros(self.num_envs, self.action_dim, device=self.device)
        self._processed_actions = torch.zeros_like(self._raw_actions)
        self._scale = torch.tensor(self.cfg.scale, device=self.device).unsqueeze(0)

        # inverse kinematics matrix (4 x 3), including per-wheel signs and radius
        k = self.cfg.wheel_base_half_length + self.cfg.track_half_width
        ik = torch.tensor(
            [[1.0, -1.0, -k], [1.0, 1.0, k], [1.0, 1.0, -k], [1.0, -1.0, k]], device=self.device
        ) / self.cfg.wheel_radius
        signs = torch.tensor(self.cfg.wheel_joint_signs, device=self.device).unsqueeze(1)
        self._ik_matrix = signs * ik

    """
    Properties.
    """

    @property
    def action_dim(self) -> int:
        return 3

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._processed_actions

    """
    Operations.
    """

    def process_actions(self, actions: torch.Tensor):
        self._raw_actions[:] = actions
        self._processed_actions = torch.clamp(actions, -1.0, 1.0) * self._scale

    def apply_actions(self):
        wheel_vel = self._processed_actions @ self._ik_matrix.T
        wheel_vel = torch.clamp(wheel_vel, -self.cfg.max_wheel_speed, self.cfg.max_wheel_speed)
        self._asset.set_joint_velocity_target(wheel_vel, joint_ids=self._joint_ids)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        self._raw_actions[env_ids] = 0.0


@configclass
class MecanumBaseVelocityActionCfg(ActionTermCfg):
    """Configuration for :class:`MecanumBaseVelocityAction`."""

    class_type: type[ActionTerm] = MecanumBaseVelocityAction

    wheel_joint_names: list[str] = MISSING
    """Wheel joint names in the order front-left, front-right, rear-left, rear-right."""
    wheel_joint_signs: tuple[float, float, float, float] = (1.0, 1.0, 1.0, 1.0)
    """Per-wheel sign so that a positive joint velocity drives the robot forward."""
    wheel_radius: float = MISSING
    """Wheel radius [m]."""
    wheel_base_half_length: float = MISSING
    """Half distance between front and rear axles (lx) [m]."""
    track_half_width: float = MISSING
    """Half distance between left and right wheels (ly) [m]."""
    scale: tuple[float, float, float] = (1.0, 1.0, 1.0)
    """Max. base velocities (vx [m/s], vy [m/s], wz [rad/s]) corresponding to an action of 1."""
    max_wheel_speed: float = float("inf")
    """Wheel velocity targets are clipped to this value [rad/s]."""
