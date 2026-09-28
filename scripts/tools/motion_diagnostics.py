# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""How does a navigation policy move? One full episode per env, deterministic policy, then statistics and a plot.

Reports: arrival time, path length vs. straight line, speed, direction of motion in the body frame (forward /
diagonal / sideways / backwards), yaw rate while driving and at the goal, wheel command jitter (per-step change,
direction reversals), saturation, and whether the wheels stop at the goal. Writes ``motion.json`` and
``motion.png`` (trajectories with the robot heading, wheel commands and distance / speed of env 0).

Usage::

    python scripts/tools/motion_diagnostics.py --checkpoint logs/rsl_rl/mecanum_navigation_flat/<run>/model_1393.pt
"""

import argparse
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Motion diagnostics of a navigation policy.")
parser.add_argument("--task", default="Mecanum-Navigation-Flat-v0")
parser.add_argument("--checkpoint", required=True)
parser.add_argument("--num_envs", type=int, default=64)
parser.add_argument("--min_distance", type=float, default=2.0, help="Minimum start-goal distance [m].")
parser.add_argument("--seed", type=int, default=7)
parser.add_argument("--out_dir", default=None, help="Default: <checkpoint dir>/analysis_<checkpoint name>.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.headless = True
simulation_app = AppLauncher(args_cli).app

"""Rest everything follows."""

import importlib.metadata as metadata
import json
import os

import gymnasium as gym
import matplotlib
import numpy as np
import torch
from rsl_rl.runners import OnPolicyRunner

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg
from isaaclab_tasks.utils import load_cfg_from_registry, parse_env_cfg

import mecanum_ws.tasks  # noqa: F401
from mecanum_ws.robots import mecanum

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def rollout():
    cfg = parse_env_cfg(args_cli.task, num_envs=args_cli.num_envs)
    cfg.seed = args_cli.seed
    cfg.commands.goal_pose.min_distance = args_cli.min_distance
    cfg.observations.policy.enable_corruption = False
    agent = handle_deprecated_rsl_rl_cfg(
        load_cfg_from_registry(args_cli.task, "rsl_rl_cfg_entry_point"), metadata.version("rsl-rl-lib")
    )
    env = RslRlVecEnvWrapper(gym.make(args_cli.task, cfg=cfg))
    u = env.unwrapped
    runner = OnPolicyRunner(env, agent.to_dict(), log_dir=None, device=u.device)
    runner.load(args_cli.checkpoint)
    policy = runner.get_inference_policy(device=u.device)
    robot, goal = u.scene["robot"], u.command_manager.get_term("goal_pose")
    wheel_ids, _ = robot.find_joints(mecanum.WHEEL_JOINT_NAMES, preserve_order=True)
    # start a fresh episode in every env so that the logged episode is complete
    u.episode_length_buf[:] = u.max_episode_length
    env.step(torch.zeros(u.num_envs, 4, device=u.device))
    obs = env.get_observations()
    target, start = goal.pos_command_w[:, :2].clone(), robot.data.root_pos_w[:, :2].clone()
    rec = {k: [] for k in ("pos", "yaw", "vb", "wz", "cmd", "wheel", "d")}
    with torch.inference_mode():
        for _ in range(int(u.max_episode_length) - 2):
            actions = policy(obs)
            obs, _, _, _ = env.step(actions)
            rec["pos"].append(robot.data.root_pos_w[:, :2].clone())
            rec["yaw"].append(robot.data.heading_w.clone())
            rec["vb"].append(robot.data.root_lin_vel_b[:, :2].clone())
            rec["wz"].append(robot.data.root_ang_vel_b[:, 2].clone())
            rec["cmd"].append(actions.clamp(-1.0, 1.0) * mecanum.MAX_WHEEL_SPEED)
            rec["wheel"].append(robot.data.joint_vel[:, wheel_ids].clone())
            rec["d"].append(torch.linalg.norm(target - robot.data.root_pos_w[:, :2], dim=1))
    data = {k: torch.stack(v).cpu().numpy() for k, v in rec.items()}  # (T, N, ...)
    data.update(start=start.cpu().numpy(), goal=target.cpu().numpy(), dt=u.step_dt)
    return data


def statistics(r: dict) -> dict:
    dt, dist = r["dt"], r["d"]
    speed = np.linalg.norm(r["vb"], axis=2)
    moving = (speed > 0.1) & (dist > 0.5)
    at_goal = dist < 0.3
    direction = np.degrees(np.abs(np.arctan2(r["vb"][..., 1], r["vb"][..., 0])))[moving]
    arrival = np.array([np.argmax(dist[:, i] < 0.5) * dt if (dist[:, i] < 0.5).any() else np.nan for i in range(dist.shape[1])])
    straight = np.linalg.norm(r["goal"] - r["start"], axis=1)
    path = np.array([
        np.sum(np.linalg.norm(np.diff(r["pos"][: max(int(a / dt), 2), i], axis=0), axis=1)) if a == a else np.nan
        for i, a in enumerate(arrival)
    ])  # fmt: skip
    dcmd = np.diff(r["cmd"], axis=0)
    bins = [0, 20, 45, 70, 110, 135, 160, 180.1]
    hist = np.histogram(direction, bins=bins)[0]
    return {
        "arrived_frac": float(np.isfinite(arrival).mean()),
        "arrival_s_median": float(np.nanmedian(arrival)),
        "start_distance_m_median": float(np.median(straight)),
        "path_over_straight_median": float(np.nanmedian(path / straight)),
        "speed_while_moving_median": float(np.median(speed[moving])),
        "speed_p99": float(np.percentile(speed, 99)),
        "motion_direction_share": {
            f"{bins[i]:.0f}-{bins[i + 1]:.0f} deg": round(float(hist[i] / max(hist.sum(), 1)), 3) for i in range(len(hist))
        },
        "yaw_rate_while_moving_median": float(np.median(np.abs(r["wz"])[moving])),
        "yaw_rate_at_goal_median": float(np.median(np.abs(r["wz"])[at_goal])) if at_goal.any() else None,
        "wheel_cmd_change_per_step_median": float(np.median(np.abs(dcmd))),
        "wheel_cmd_change_per_step_p90": float(np.percentile(np.abs(dcmd), 90)),
        "wheel_cmd_reversals_per_s": float((np.diff(np.sign(dcmd), axis=0) != 0).mean() / dt),
        "wheel_cmd_saturated_frac": float((np.abs(r["cmd"]) > 0.95 * mecanum.MAX_WHEEL_SPEED).mean()),
        "wheel_speed_at_goal_mean_abs": float(np.abs(r["wheel"])[at_goal].mean()) if at_goal.any() else None,
    }


def plot(r: dict, path: str):
    dt = r["dt"]
    t = np.arange(r["d"].shape[0]) * dt
    fig = plt.figure(figsize=(16, 9))
    ax = fig.add_subplot(1, 2, 1)
    for i in range(min(6, r["d"].shape[1])):
        p, g = r["pos"][:, i] - r["start"][i], r["goal"][i] - r["start"][i]
        (line,) = ax.plot(p[:, 0], p[:, 1], lw=1.2, label=f"env {i}")
        k = np.arange(0, len(p), 25)
        ax.quiver(p[k, 0], p[k, 1], np.cos(r["yaw"][k, i]), np.sin(r["yaw"][k, i]), angles="xy", scale=25,
                  width=0.003, color=line.get_color())  # fmt: skip
        ax.plot(*g, "*", ms=14, color=line.get_color())
    ax.plot(0, 0, "ko")
    ax.set_aspect("equal")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    ax.set_title("paths (start = 0,0; star = goal; arrows = robot heading every 0.5 s)")
    ax2 = fig.add_subplot(3, 2, 2)
    ax2.plot(t, r["cmd"][:, 0, :], lw=0.8)
    ax2.set_ylabel("wheel command [rad/s]")
    ax2.set_title("env 0: wheel commands (FL, FR, RL, RR)")
    ax2.grid(alpha=0.3)
    ax3 = fig.add_subplot(3, 2, 4)
    ax3.plot(t, r["d"][:, 0], label="distance to goal [m]")
    ax3.plot(t, np.linalg.norm(r["vb"][:, 0], axis=1), lw=0.8, label="speed [m/s]")
    ax3.legend(fontsize=8)
    ax3.grid(alpha=0.3)
    ax4 = fig.add_subplot(3, 2, 6)
    sl = slice(200, 260)
    ax4.plot(t[sl], r["cmd"][sl, 0, 0], ".-", lw=0.8, label="FL command (zoom)")
    ax4.plot(t[sl], r["wheel"][sl, 0, 0], lw=0.8, label="FL actual")
    ax4.set_xlabel("time [s]")
    ax4.legend(fontsize=8)
    ax4.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=90)


def main():
    out_dir = args_cli.out_dir or os.path.join(
        os.path.dirname(os.path.abspath(args_cli.checkpoint)),
        "analysis_" + os.path.splitext(os.path.basename(args_cli.checkpoint))[0],
    )
    os.makedirs(out_dir, exist_ok=True)
    r = rollout()
    stats = statistics(r)
    with open(os.path.join(out_dir, "motion.json"), "w") as f:
        json.dump(stats, f, indent=2)
    plot(r, os.path.join(out_dir, "motion.png"))
    print(json.dumps(stats, indent=2))
    print(f"[INFO] Wrote {out_dir}/motion.json and motion.png")
    sys.stdout.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
