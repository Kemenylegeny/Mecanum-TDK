# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch

import isaaclab.sim as sim_utils
from isaaclab.envs.mdp import TerrainBasedPose2dCommand, TerrainBasedPose2dCommandCfg
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.utils import configclass
from isaaclab.utils.math import quat_from_euler_xyz


class GoalPositionCommand(TerrainBasedPose2dCommand):
    """Terrain-based goal position whose debug visualization shows the goal and, above every robot, an arrow
    pointing at its goal (instead of the heading of the goal pose, which the task does not use).

    The goal and arrow of ``cfg.highlight_env_index`` (the env the camera follows) use the second, highlighted marker.
    """

    cfg: GoalPositionCommandCfg

    def _resample_command(self, env_ids):
        super()._resample_command(env_ids)
        if self.cfg.min_distance <= 0.0 and self.cfg.max_distance <= 0.0:
            return
        # redraw goals closer than min_distance / farther than max_distance from the (already reset) robot
        env_ids = torch.as_tensor(env_ids, device=self.device)
        max_d = self.cfg.max_distance if self.cfg.max_distance > 0.0 else float("inf")
        for _ in range(self.cfg.max_redraws):
            dist = torch.linalg.norm(self.pos_command_w[env_ids, :2] - self.robot.data.root_pos_w[env_ids, :2], dim=1)
            bad = env_ids[(dist < self.cfg.min_distance) | (dist > max_d)]
            if bad.numel() == 0:
                break
            super()._resample_command(bad)
        if self.cfg.max_distance > 0.0:
            # still too far after the redraws: pull the goal towards the robot onto max_distance
            to_goal = self.pos_command_w[env_ids, :2] - self.robot.data.root_pos_w[env_ids, :2]
            dist = torch.linalg.norm(to_goal, dim=1, keepdim=True)
            scale = torch.clamp(self.cfg.max_distance / dist.clamp(min=1e-6), max=1.0)
            self.pos_command_w[env_ids, :2] = self.robot.data.root_pos_w[env_ids, :2] + to_goal * scale

    def _set_debug_vis_impl(self, debug_vis: bool):
        if debug_vis:
            if not hasattr(self, "goal_visualizer"):
                self.goal_visualizer = VisualizationMarkers(self.cfg.goal_visualizer_cfg)
                self.direction_visualizer = VisualizationMarkers(self.cfg.direction_visualizer_cfg)
            self.goal_visualizer.set_visibility(True)
            self.direction_visualizer.set_visibility(True)
        elif hasattr(self, "goal_visualizer"):
            self.goal_visualizer.set_visibility(False)
            self.direction_visualizer.set_visibility(False)

    def _debug_vis_callback(self, event):
        if not self.robot.is_initialized:
            return
        goal = self.pos_command_w.clone()
        goal[:, 2] = self._env.scene.env_origins[:, 2] + self.cfg.goal_height
        indices = torch.zeros(self.num_envs, dtype=torch.int32, device=self.device)
        if 0 <= self.cfg.highlight_env_index < self.num_envs:
            indices[self.cfg.highlight_env_index] = 1
        self.goal_visualizer.visualize(translations=goal, marker_indices=indices)
        root = self.robot.data.root_pos_w
        to_goal = self.pos_command_w[:, :2] - root[:, :2]
        yaw = torch.atan2(to_goal[:, 1], to_goal[:, 0])
        zeros = torch.zeros_like(yaw)
        arrow_pos = root.clone()
        arrow_pos[:, 2] += self.cfg.arrow_height
        self.direction_visualizer.visualize(
            translations=arrow_pos, orientations=quat_from_euler_xyz(zeros, zeros, yaw), marker_indices=indices
        )


@configclass
class GoalPositionCommandCfg(TerrainBasedPose2dCommandCfg):
    """Configuration for :class:`GoalPositionCommand`."""

    class_type: type = GoalPositionCommand

    goal_height: float = 0.15
    """Height of the goal marker above the ground [m]."""

    arrow_height: float = 0.35
    """Height of the direction arrow above the robot base [m]."""

    min_distance: float = 0.0
    """Minimum distance between the robot and a newly sampled goal [m] (goals are redrawn); 0 disables it."""

    max_distance: float = 0.0
    """Maximum distance between the robot and a newly sampled goal [m] (goals are redrawn, finally pulled in); 0
    disables it."""

    max_redraws: int = 20
    """Number of redraws for :attr:`min_distance` / :attr:`max_distance` before a goal is accepted."""

    highlight_env_index: int = -1
    """Env whose goal and arrow are highlighted (e.g. the one the viewer camera follows); -1 for none."""

    goal_visualizer_cfg: VisualizationMarkersCfg = VisualizationMarkersCfg(
        prim_path="/Visuals/Command/goal_position",
        markers={
            "goal": sim_utils.SphereCfg(
                radius=0.15, visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.9, 0.1, 0.1))
            ),
            "goal_highlight": sim_utils.SphereCfg(
                radius=0.25, visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.85, 0.0))
            ),
        },
    )
    """Marker at the goal position: red sphere, yellow (bigger) for the highlighted env."""

    direction_visualizer_cfg: VisualizationMarkersCfg = VisualizationMarkersCfg(
        prim_path="/Visuals/Command/goal_direction",
        markers={
            "arrow": sim_utils.UsdFileCfg(
                usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/UIElements/arrow_x.usd",
                scale=(0.4, 0.4, 0.6),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0)),
            ),
            "arrow_highlight": sim_utils.UsdFileCfg(
                usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/UIElements/arrow_x.usd",
                scale=(0.6, 0.6, 0.9),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.85, 0.0)),
            ),
        },
    )
    """Arrow above each robot pointing at its goal: green, yellow (bigger) for the highlighted env."""
