# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Play a navigation checkpoint and record a video in which the followed robot AND its goal are always in view.

The camera looks at the midpoint between the robot of ``--env_index`` and its goal (yellow sphere) from a height that
grows with their distance (smoothed). Text overlay: time, distance to the goal, whether it is reached.

Usage::

    python scripts/tools/play_video.py --task Mecanum-Navigation-Flat-Own-Play-v0 --checkpoint <run>/model_399.pt \\
        --seconds 30 --headless --enable_cameras
"""

import argparse
import importlib.metadata as metadata
import os
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
parser.add_argument("--task", required=True)
parser.add_argument("--checkpoint", required=True)
parser.add_argument("--seconds", type=float, default=30.0)
parser.add_argument("--num_envs", type=int, default=32)
parser.add_argument("--env_index", type=int, default=1, help="Robot the camera follows (its goal is yellow).")
parser.add_argument("--fps", type=float, default=25.0)
parser.add_argument("--out", default=None, help="Default: <checkpoint dir>/videos/play_<checkpoint>_<seconds>s.mp4")
AppLauncher.add_app_launcher_args(parser)
args, overrides = parser.parse_known_args()
args.enable_cameras = True
sys.argv = [sys.argv[0]]
simulation_app = AppLauncher(args).app

"""Rest everything follows."""

import gymnasium as gym
import imageio.v2 as imageio
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from rsl_rl.runners import OnPolicyRunner

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg
from isaaclab_tasks.utils import load_cfg_from_registry, parse_env_cfg

import mecanum_ws.tasks  # noqa: F401


def main():
    cfg = parse_env_cfg(args.task, num_envs=args.num_envs)
    cfg.viewer.origin_type = "world"
    cfg.commands.goal_pose.highlight_env_index = args.env_index
    agent = handle_deprecated_rsl_rl_cfg(load_cfg_from_registry(args.task, "rsl_rl_cfg_entry_point"), metadata.version("rsl-rl-lib"))
    env = RslRlVecEnvWrapper(gym.make(args.task, cfg=cfg, render_mode="rgb_array"), clip_actions=agent.clip_actions)
    u = env.unwrapped
    runner = OnPolicyRunner(env, agent.to_dict(), log_dir=None, device=u.device)
    runner.load(args.checkpoint)
    policy = runner.get_inference_policy(device=u.device)
    robot, goal = u.scene["robot"], u.command_manager.get_term("goal_pose")
    i = args.env_index

    out = args.out or os.path.join(os.path.dirname(os.path.abspath(args.checkpoint)), "videos",
                                   f"play_{os.path.splitext(os.path.basename(args.checkpoint))[0]}_{args.seconds:g}s.mp4")  # fmt: skip
    os.makedirs(os.path.dirname(out), exist_ok=True)
    writer = imageio.get_writer(out, fps=args.fps, quality=8, macro_block_size=8)
    font = ImageFont.load_default()
    steps = int(round(args.seconds / u.step_dt))
    every = max(1, int(round(1.0 / (args.fps * u.step_dt))))
    look, height = None, None
    reached = 0
    obs = env.get_observations()
    with torch.inference_mode():
        for step in range(steps):
            obs, _, dones, _ = env.step(policy(obs))
            p = robot.data.root_pos_w[i, :3].cpu().numpy()
            g = goal.pos_command_w[i, :3].cpu().numpy()
            d = float(np.linalg.norm((g - p)[:2]))
            if bool(dones[i]):
                reached = 0
            elif d < 0.5:
                reached = 1
            mid = 0.5 * (p + g)
            mid[2] = 0.0
            h = max(1.6, 1.1 * d + 0.8)
            # smooth the camera, but jump after a reset (new goal)
            if look is None or bool(dones[i]):
                look, height = mid, h
            else:
                look, height = 0.9 * look + 0.1 * mid, 0.95 * height + 0.05 * h
            eye = look + np.array([-0.45 * height, -0.45 * height, height])
            u.viewport_camera_controller.update_view_location(eye.tolist(), look.tolist())
            if step % every:
                continue
            frame = u.render()
            if frame is None:
                continue
            img = Image.fromarray(np.asarray(frame)[..., :3].astype(np.uint8))
            draw = ImageDraw.Draw(img)
            t = float(u.episode_length_buf[i]) * u.step_dt
            text = (f"t = {step * u.step_dt:5.1f} s   epizod ido {t:4.1f} / {u.max_episode_length_s:.0f} s   "
                    f"tavolsag a (sarga) celtol: {d:4.2f} m   {'CELBA ERT' if reached else ''}")  # fmt: skip
            draw.rectangle([0, 0, img.width, 24], fill=(0, 0, 0))
            draw.text((10, 6), text, fill=(255, 255, 255), font=font)
            writer.append_data(np.asarray(img))
    writer.close()
    print(f"[play_video] wrote {out}", flush=True)
    sys.stdout.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
