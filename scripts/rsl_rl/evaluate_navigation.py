# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Evaluate a navigation checkpoint: success, collisions (and why), speed profile over the episode, per terrain level.

The policy runs deterministically (mean action) on the training task with the curriculum frozen and the robots
spread uniformly over all terrain levels. Results are printed and written to ``<checkpoint dir>/eval_<name>.json``.

Usage::

    python scripts/rsl_rl/evaluate_navigation.py --checkpoint logs/rsl_rl/.../model_900.pt --num_envs 1024

Extra ``env.<path>=<value>`` arguments override the environment config like Hydra does in train.py
(e.g. ``env.episode_length_s=8.0``), so a policy is evaluated in the setup it was trained in.
"""

import argparse
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Evaluate a pillar navigation policy.")
parser.add_argument("--task", type=str, default="Mecanum-Navigation-Pillars-v0")
parser.add_argument("--checkpoint", type=str, required=True)
parser.add_argument("--num_envs", type=int, default=1024)
parser.add_argument("--episodes", type=int, default=2, help="Episodes per env.")
parser.add_argument("--seed", type=int, default=1)
AppLauncher.add_app_launcher_args(parser)
args_cli, overrides = parser.parse_known_args()
args_cli.headless = True
sys.argv = [sys.argv[0]]
simulation_app = AppLauncher(args_cli).app

"""Rest everything follows."""

import json
import math
import os

import ast
import importlib.metadata as metadata

import gymnasium as gym
import numpy as np
import torch
from rsl_rl.runners import OnPolicyRunner

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg
from isaaclab_tasks.utils import load_cfg_from_registry, parse_env_cfg

import mecanum_ws.tasks  # noqa: F401
from mecanum_ws.tasks.manager_based.mecanum_navigation.navigation_env_cfg import robot_module
from mecanum_ws.tasks.manager_based.mecanum_navigation import mdp

SUCCESS_RADIUS = 0.5
TIME_BINS = 10
# "started towards the goal": the goal distance dropped by at least START_PROGRESS_M within the first START_WINDOW_S
START_WINDOW_S = 2.0
START_PROGRESS_M = 0.5


def main():
    env_cfg = parse_env_cfg(args_cli.task, num_envs=args_cli.num_envs)
    env_cfg.seed = args_cli.seed
    apply_overrides(env_cfg, overrides)
    mecanum = robot_module(getattr(env_cfg, "robot_name", "fuji"))  # robot constants of the task
    # all difficulty levels, frozen
    env_cfg.scene.terrain.max_init_terrain_level = None
    env_cfg.curriculum.terrain_levels = None
    agent_cfg = load_cfg_from_registry(args_cli.task, "rsl_rl_cfg_entry_point")
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, metadata.version("rsl-rl-lib"))

    env = gym.make(args_cli.task, cfg=env_cfg)
    u = env.unwrapped
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=u.device)
    runner.load(args_cli.checkpoint)
    policy = runner.get_inference_policy(device=u.device)

    robot = u.scene["robot"]
    lidar = u.scene["lidar"]
    goal = u.command_manager.get_term("goal_pose")
    n, dev = u.num_envs, u.device
    # lidar ray angles in the yaw-aligned sensor frame (0 = robot front)
    ray_dirs = lidar.ray_directions[0, :, :2]
    ray_angle = torch.atan2(ray_dirs[:, 1], ray_dirs[:, 0])

    # per-episode accumulators
    ep_steps = torch.zeros(n, device=dev)
    path_len = torch.zeros(n, device=dev)
    start_dist = torch.full((n,), float("nan"), device=dev)
    arrival_step = torch.full((n,), float("nan"), device=dev)
    early_dist = torch.full((n,), float("nan"), device=dev)
    window_steps = round(START_WINDOW_S / u.step_dt)
    slow_steps = torch.zeros(n, device=dev)
    bin_speed = torch.zeros(n, TIME_BINS, device=dev)
    bin_count = torch.zeros(n, TIME_BINS, device=dev)
    prev_pos = robot.data.root_pos_w[:, :2].clone()
    episodes = []
    done_count = torch.zeros(n, dtype=torch.long, device=dev)
    max_len = u.max_episode_length
    # motion quality over all evaluated steps: the quantities the penalties act on (for the staged penalty study)
    wheels = mdp.SceneEntityCfg("robot", joint_names=mecanum.WHEEL_JOINT_NAMES, preserve_order=True)
    wheels.resolve(u.scene)
    motion = {k: 0.0 for k in ("torque2", "acc2", "action_rate2", "action2", "slip2", "stall", "steps")}
    spin_sum, spin_count = 0.0, 0.0
    prev_actions = torch.zeros(n, 4, device=dev)

    obs = env.get_observations()
    with torch.inference_mode():
        while (done_count < args_cli.episodes).any():
            # state before stepping (the reset inside step() overwrites it for finished envs)
            dist = mdp.goal_distance(u, "goal_pose")
            start_dist = torch.where(torch.isnan(start_dist), dist, start_dist)
            level = u.scene.terrain.terrain_levels.clone()
            # gap robot rectangle <-> pillars before the step: after a reset the lidar measures from the new spawn pose
            clearance = mdp.footprint_clearance(
                u, mdp.SceneEntityCfg("lidar"), 5.0, mecanum.FOOTPRINT_HALF_EXTENTS
            ).clone()
            actions = policy(obs)
            counting = (done_count < args_cli.episodes).float()
            obs, _, dones, _ = env.step(actions)
            # motion quality of this step (per env, summed over the wheels)
            torque2 = torch.sum(robot.data.applied_torque[:, wheels.joint_ids] ** 2, dim=1)
            acc2 = torch.sum(robot.data.joint_acc[:, wheels.joint_ids] ** 2, dim=1)
            slip2 = mdp.wheel_slip_l2(
                u, wheels, mecanum.WHEEL_RADIUS, mecanum.WHEEL_BASE_HALF_LENGTH, mecanum.TRACK_HALF_WIDTH,
                mecanum.WHEEL_JOINT_SIGNS,
            )
            for key, value in (
                ("torque2", torque2),
                ("acc2", acc2),
                ("action_rate2", torch.sum((actions - prev_actions) ** 2, dim=1)),
                ("action2", torch.sum(actions**2, dim=1)),
                ("slip2", slip2),
            ):
                motion[key] += float(torch.sum(value * counting))
            motion["steps"] += float(counting.sum())
            # quantities of the step that just happened (pose not yet reset for done envs is lost, so use terms)
            pos = robot.data.root_pos_w[:, :2]
            speed = torch.linalg.norm(robot.data.root_lin_vel_w[:, :2], dim=1)
            step_idx = ep_steps.clone()
            b = (step_idx / max_len * TIME_BINS).long().clamp(max=TIME_BINS - 1)
            bin_speed.scatter_add_(1, b.unsqueeze(1), speed.unsqueeze(1))
            bin_count.scatter_add_(1, b.unsqueeze(1), torch.ones_like(speed).unsqueeze(1))
            slow_steps += (speed < 0.1).float()
            ep_steps += 1
            done = dones.bool()
            path_len += torch.where(done, 0.0, torch.linalg.norm(pos - prev_pos, dim=1))
            dist_now = mdp.goal_distance(u, "goal_pose")
            arrived = (dist_now < SUCCESS_RADIUS) & torch.isnan(arrival_step) & ~done
            arrival_step = torch.where(arrived, ep_steps, arrival_step)
            early_dist = torch.where((ep_steps == window_steps) & ~done, dist_now, early_dist)
            # spinning at the goal, stalling far from it
            at_goal = (dist_now < 0.3) & (ep_steps > 50) & ~done & (counting > 0)
            spin_sum += float(robot.data.root_ang_vel_b[at_goal, 2].abs().sum())
            spin_count += float(at_goal.sum())
            motion["stall"] += float((((speed < 0.1) & (dist_now > 0.5) & ~done).float() * counting).sum())
            prev_actions = torch.where(done.unsqueeze(1), torch.zeros_like(actions), actions)

            if done.any():
                tm = u.termination_manager
                collided = tm.get_term("collision") | tm.get_term("flipped")
                # final distance: the reset already moved the goal, use the distance of the previous step
                final_dist = dist
                for i in done.nonzero().flatten().tolist():
                    if done_count[i] >= args_cli.episodes:
                        continue
                    rec = {
                        "level": int(level[i]),
                        "start_dist": float(start_dist[i]),
                        "final_dist": float(final_dist[i]),
                        "success": bool(final_dist[i] < SUCCESS_RADIUS) and not bool(collided[i]),
                        "collided": bool(collided[i]),
                        "time": float(ep_steps[i] * u.step_dt),
                        "path_len": float(path_len[i]),
                        "arrival_time": float(arrival_step[i] * u.step_dt),
                        # an episode that ended before the window (crash) counts with its last distance
                        "early_progress": float(start_dist[i] - (final_dist[i] if torch.isnan(early_dist[i]) else early_dist[i])),
                        "slow_frac": float(slow_steps[i] / ep_steps[i]),
                        "bin_speed": (bin_speed[i] / bin_count[i].clamp(min=1)).tolist(),
                        "bin_valid": (bin_count[i] > 0).tolist(),
                    }
                    if collided[i]:
                        # lidar of the last step before the crash (<= 1 policy step, a few cm, earlier)
                        k = int(clearance[i].argmin())
                        rec["crash_angle_deg"] = math.degrees(float(ray_angle[k]))
                        rec["crash_clearance"] = float(clearance[i, k])
                    episodes.append(rec)
                    done_count[i] += 1
                # reset accumulators
                for t in (ep_steps, path_len, slow_steps):
                    t[done] = 0.0
                start_dist[done] = float("nan")
                arrival_step[done] = float("nan")
                early_dist[done] = float("nan")
                bin_speed[done] = 0.0
                bin_count[done] = 0.0
            prev_pos = robot.data.root_pos_w[:, :2].clone()

    steps = max(motion.pop("steps"), 1.0)
    motion = {k: v / steps for k, v in motion.items()}
    motion["stall_frac"] = motion.pop("stall")
    motion["spin_at_goal"] = spin_sum / max(spin_count, 1.0)
    report(episodes, u.step_dt * max_len, motion)
    # Kit's shutdown can hang after long runs; the results are written, so exit hard
    sys.stdout.flush()
    os._exit(0)


def apply_overrides(env_cfg, overrides: list[str]):
    """Apply ``env.a.b.c=value`` overrides (other prefixes, e.g. ``agent.``, are ignored)."""
    for item in overrides:
        key, _, value = item.partition("=")
        if not key.startswith("env."):
            continue
        *path, leaf = key[len("env."):].split(".")
        obj = env_cfg
        for part in path:
            obj = obj[part] if isinstance(obj, dict) else getattr(obj, part)
        try:
            value = ast.literal_eval(value)
        except (ValueError, SyntaxError):
            pass
        if isinstance(obj, dict):
            obj[leaf] = value
        else:
            setattr(obj, leaf, value)
        print(f"[INFO] override env.{'.'.join(path + [leaf])} = {value!r}")


def report(episodes: list[dict], episode_s: float, motion: dict):
    e = episodes
    arr = lambda k: np.array([x[k] for x in e], dtype=float)  # noqa: E731
    succ, coll, lvl = arr("success"), arr("collided"), arr("level")
    print(f"\n===== {len(e)} episodes, episode length {episode_s:.1f} s")
    print(f"success (final dist < {SUCCESS_RADIUS} m, no crash): {succ.mean():.1%}   crashed: {coll.mean():.1%}")
    timeouts = ~coll.astype(bool)
    print(f"final distance of non-crashed episodes: median {np.median(arr('final_dist')[timeouts]):.2f} m")
    arrived = ~np.isnan(arr("arrival_time"))
    print(f"reached the goal at some point: {arrived.mean():.1%}; median arrival time {np.nanmedian(arr('arrival_time')):.1f} s "
          f"(of {episode_s:.0f} s); median start distance {np.median(arr('start_dist')):.2f} m")
    straight = arr("start_dist")
    print(f"mean speed while travelling: path / time = {np.nanmedian(arr('path_len') / arr('time')):.2f} m/s; "
          f"path / straight-line distance = {np.nanmedian(arr('path_len') / np.maximum(straight, 0.3)):.2f}")
    print(f"fraction of the episode slower than 0.1 m/s: {arr('slow_frac').mean():.1%}")
    bins = np.array([x["bin_speed"] for x in e]); valid = np.array([x["bin_valid"] for x in e])
    prof = [bins[valid[:, j], j].mean() if valid[:, j].any() else float("nan") for j in range(TIME_BINS)]
    print("mean speed per 10 % of episode time [m/s]:", " ".join(f"{p:.2f}" for p in prof))
    started = arr("early_progress") >= START_PROGRESS_M
    print(f"\nlevel  episodes  success  crashed  median final dist  started (>= {START_PROGRESS_M} m closer "
          f"in {START_WINDOW_S:g} s)  median progress in {START_WINDOW_S:g} s [m]")
    for L in sorted(set(lvl.astype(int))):
        m = lvl == L
        print(f"{L:5d} {m.sum():9d} {succ[m].mean():8.1%} {coll[m].mean():8.1%} {np.median(arr('final_dist')[m]):12.2f} "
              f"{started[m].mean():17.1%} {np.median(arr('early_progress')[m]):24.2f}")
    crashes = [x for x in e if x["collided"]]
    if crashes:
        ang = np.abs(np.array([x["crash_angle_deg"] for x in crashes]))
        clr = np.array([x["crash_clearance"] for x in crashes])
        t = np.array([x["time"] for x in crashes])
        print(f"\ncrashes: {len(crashes)}; closest pillar direction |angle| (0 = front, 90 = side, 180 = back):")
        hist, edges = np.histogram(ang, bins=[0, 30, 60, 120, 150, 180.1])
        print("   " + "  ".join(f"{int(edges[j])}-{int(edges[j + 1])}°: {hist[j]}" for j in range(len(hist))))
        print(f"   gap between pillar and the robot's rectangle at termination: median {np.median(clr):.2f} m, "
              f"{(clr > 0.05).mean():.0%} of the crashes had > 5 cm real clearance")
        print(f"   crash time: median {np.median(t):.1f} s")
    print("\nmotion quality (mean per policy step): " + ", ".join(
        f"{k} {v:.4g}" for k, v in motion.items()
    ) + "  [torque2: sum tau^2 Nm^2, acc2: sum qdd^2, action_rate2 / action2: raw actions, slip2: m^2/s^2, "
        "stall_frac: slow (< 0.1 m/s) > 0.5 m from the goal, spin_at_goal: |yaw rate| rad/s within 0.3 m]")
    out = os.path.join(os.path.dirname(os.path.abspath(args_cli.checkpoint)),
                       f"eval_{os.path.splitext(os.path.basename(args_cli.checkpoint))[0]}.json")
    with open(out, "w") as f:
        json.dump({"episodes": e, "success": float(succ.mean()), "crashed": float(coll.mean()), "motion": motion}, f)
    print(f"\n[INFO] Wrote {out}")


if __name__ == "__main__":
    main()
