# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Open-loop drive test with video: forward, sideways, diagonal, rotation in place (base twist -> mecanum IK -> wheels).

Flat ground with a checkerboard (visual only) for reference. The video shows a camera following the robot (fixed
orientation, so heading changes are visible) next to a top view of the whole path. Also writes a plot of commanded vs.
measured body velocity and a per-phase summary (steady state = last 2 s of every phase).

Usage::

    python scripts/tools/drive_test.py --robot own --headless --enable_cameras
"""

import argparse
import importlib
import os

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
parser.add_argument("--robot", choices=["own", "fuji"], default="own")
parser.add_argument("--speed", type=float, default=0.3, help="Linear speed of the test phases [m/s].")
parser.add_argument("--yaw_rate", type=float, default=1.0, help="Yaw rate of the rotation phase [rad/s].")
parser.add_argument("--phase_s", type=float, default=3.0)
parser.add_argument("--pause_s", type=float, default=1.0)
parser.add_argument("--fps", type=int, default=30)
parser.add_argument("--out_dir", default=None, help="Default: logs/drive_test/<robot>.")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.enable_cameras = True
simulation_app = AppLauncher(args).app

"""Rest everything follows."""

import json
import math

import imageio.v2 as imageio
import matplotlib
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.sensors import Camera, CameraCfg

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

rb = importlib.import_module("mecanum_ws.robots." + ("own_mecanum" if args.robot == "own" else "mecanum"))
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
V, W = args.speed, args.yaw_rate
PHASES = [  # name, (vx, vy, wz) in the body frame
    ("elore", (V, 0.0, 0.0)),
    ("oldalra (balra)", (0.0, V, 0.0)),
    ("atlosan (elore-balra)", (V, V, 0.0)),
    ("forgas helyben", (0.0, 0.0, W)),
]
T_SETTLE = 1.0


def twist_at(t: float) -> tuple[str, np.ndarray]:
    t -= T_SETTLE
    if t < 0:
        return "all", np.zeros(3)
    k, r = divmod(t, args.phase_s + args.pause_s)
    if k >= len(PHASES):
        return "all", np.zeros(3)
    name, tw = PHASES[int(k)]
    return (name, np.array(tw)) if r < args.phase_s else (name + " - megall", np.zeros(3))


def main():
    out_dir = os.path.join(ROOT, args.out_dir or f"logs/drive_test/{args.robot}")
    os.makedirs(out_dir, exist_ok=True)
    dt = 1.0 / 360.0
    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=dt, device=args.device, physics_material=rb.ROBOT_MATERIAL))
    ground = sim_utils.GroundPlaneCfg(physics_material=rb.ground_material(1.0), color=(0.55, 0.55, 0.58))
    ground.func("/World/ground", ground)
    light = sim_utils.DomeLightCfg(intensity=1500.0)
    light.func("/World/light", light)
    # checkerboard for visual reference (no collision)
    scale = 1.0 if args.robot == "own" else 3.0
    tile = 0.25 * scale
    lo, hi = -0.75 * scale, 2.75 * scale
    n = int(round((hi - lo) / tile))
    for i in range(n):
        for j in range(n):
            if (i + j) % 2:
                continue
            cfg = sim_utils.CuboidCfg(size=(tile, tile, 0.001), visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.25, 0.25, 0.3)))
            cfg.func(f"/World/tiles/t_{i}_{j}", cfg, translation=(lo + (i + 0.5) * tile, lo + (j + 0.5) * tile, 0.0006))
    robot = Articulation(rb.MECANUM_CFG.replace(prim_path="/World/Robot"))
    cam_follow = Camera(CameraCfg(prim_path="/World/cam_follow", width=640, height=480, data_types=["rgb"],
                                  spawn=sim_utils.PinholeCameraCfg(focal_length=18.0)))  # fmt: skip
    cam_top = Camera(CameraCfg(prim_path="/World/cam_top", width=640, height=480, data_types=["rgb"],
                               spawn=sim_utils.PinholeCameraCfg(focal_length=18.0)))  # fmt: skip
    sim.reset()
    dev = sim.device
    mid = 0.5 * (lo + hi)
    cam_top.set_world_poses_from_view(torch.tensor([[mid - 0.001, mid, 2.6 * scale]], device=dev), torch.tensor([[mid, mid, 0.0]], device=dev))
    wheel_ids, _ = robot.find_joints(rb.WHEEL_JOINT_NAMES, preserve_order=True)
    k = rb.WHEEL_BASE_HALF_LENGTH + rb.TRACK_HALF_WIDTH
    ik = torch.tensor([[1.0, -1.0, -k], [1.0, 1.0, k], [1.0, 1.0, -k], [1.0, -1.0, k]], device=dev) / rb.WHEEL_RADIUS
    signs = torch.tensor(rb.WHEEL_JOINT_SIGNS, device=dev)
    print(f"[drive_test] robot {args.robot}: bodies {robot.num_bodies}, joints {robot.num_joints}, "
          f"mass {float(robot.root_physx_view.get_masses().sum()):.2f} kg", flush=True)  # fmt: skip

    font = ImageFont.load_default()
    writer = imageio.get_writer(os.path.join(out_dir, "drive_test.mp4"), fps=args.fps, quality=8, macro_block_size=8)
    total = T_SETTLE + len(PHASES) * (args.phase_s + args.pause_s)
    steps = int(total / dt)
    every = int(round(1.0 / (args.fps * dt)))
    log = {k_: [] for k_ in ("t", "phase", "cmd", "v_b", "w_z", "pos", "yaw", "wheel_cmd", "wheel_vel")}
    for step in range(steps):
        t = step * dt
        phase, tw = twist_at(t)
        cmd = torch.tensor(tw, device=dev, dtype=torch.float32)
        wheel_cmd = torch.clamp(ik @ cmd, -rb.MAX_WHEEL_SPEED, rb.MAX_WHEEL_SPEED) * signs
        robot.set_joint_velocity_target(wheel_cmd.unsqueeze(0), joint_ids=wheel_ids)
        robot.write_data_to_sim()
        sim.step(render=False)
        robot.update(dt)
        if step % 4 == 0:
            d = robot.data
            log["t"].append(t)
            log["phase"].append(phase)
            log["cmd"].append(tw.tolist())
            log["v_b"].append(d.root_lin_vel_b[0, :2].tolist())
            log["w_z"].append(float(d.root_ang_vel_b[0, 2]))
            log["pos"].append(d.root_pos_w[0, :3].tolist())
            log["yaw"].append(float(d.heading_w[0]))
            log["wheel_cmd"].append(wheel_cmd.tolist())
            log["wheel_vel"].append(d.joint_vel[0, wheel_ids].tolist())
        if step % every == 0:
            pos = robot.data.root_pos_w[0]
            eye = pos + torch.tensor([-0.55 * scale, -0.75 * scale, 0.45 * scale], device=dev)
            cam_follow.set_world_poses_from_view(eye.unsqueeze(0), pos.unsqueeze(0))
            sim.render()
            cam_follow.update(dt, force_recompute=True)
            cam_top.update(dt, force_recompute=True)
            frames = []
            for cam in (cam_follow, cam_top):
                frames.append(cam.data.output["rgb"][0, ..., :3].cpu().numpy().astype(np.uint8))
            img = Image.fromarray(np.concatenate(frames, axis=1))
            draw = ImageDraw.Draw(img)
            v = robot.data.root_lin_vel_b[0]
            text = (f"t = {t:4.1f} s   {phase}   parancs: vx {tw[0]:+.2f}  vy {tw[1]:+.2f} m/s  wz {tw[2]:+.2f} rad/s   "
                    f"mert: vx {float(v[0]):+.2f}  vy {float(v[1]):+.2f}  wz {float(robot.data.root_ang_vel_b[0, 2]):+.2f}")  # fmt: skip
            draw.rectangle([0, 0, img.width, 22], fill=(0, 0, 0))
            draw.text((8, 5), text, fill=(255, 255, 255), font=font)
            writer.append_data(np.asarray(img))
    writer.close()

    # per-phase steady state (last 2 s of the drive part)
    t = np.array(log["t"])
    vb, wz = np.array(log["v_b"]), np.array(log["w_z"])
    summary = {"robot": args.robot, "phases": []}
    for i, (name, tw) in enumerate(PHASES):
        t0 = T_SETTLE + i * (args.phase_s + args.pause_s)
        sel = (t > t0 + args.phase_s - 2.0) & (t < t0 + args.phase_s)
        meas = np.array([vb[sel, 0].mean(), vb[sel, 1].mean(), wz[sel].mean()])
        tw = np.array(tw)
        main_axis = int(np.argmax(np.abs(tw)))
        err = {("vx", "vy", "wz")[j]: round(float(meas[j] - tw[j]), 4) for j in range(3)}
        summary["phases"].append({"phase": name, "command": tw.tolist(), "measured": meas.round(4).tolist(), "error": err,
                                  "rel_error_main_axis": round(float((meas[main_axis] - tw[main_axis]) / tw[main_axis]), 4)})  # fmt: skip
    with open(os.path.join(out_dir, "drive_test.json"), "w") as f:
        json.dump(summary, f, indent=2)

    fig, axs = plt.subplots(1, 2, figsize=(15, 5))
    cmd = np.array(log["cmd"])
    for j, (lab, col) in enumerate((("vx", "C0"), ("vy", "C1"), ("wz", "C2"))):
        axs[0].plot(t, cmd[:, j], "--", color=col, lw=1, label=f"{lab} parancs")
        axs[0].plot(t, vb[:, j] if j < 2 else wz, color=col, lw=1.2, label=f"{lab} mert")
    axs[0].set_xlabel("t [s]")
    axs[0].set_ylabel("m/s, rad/s (test koordinata-rendszer)")
    axs[0].legend(ncol=3, fontsize=8)
    axs[0].grid(alpha=0.3)
    pos, yaw = np.array(log["pos"]), np.array(log["yaw"])
    axs[1].plot(pos[:, 0], pos[:, 1], lw=1.2)
    kk = np.arange(0, len(pos), 45)
    axs[1].quiver(pos[kk, 0], pos[kk, 1], np.cos(yaw[kk]), np.sin(yaw[kk]), scale=30, width=0.003)
    axs[1].set_aspect("equal")
    axs[1].grid(alpha=0.3)
    axs[1].set_title("palya (nyil = robot iranya)")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "drive_test.png"), dpi=90)
    print(json.dumps(summary, indent=2), flush=True)
    print(f"[drive_test] wrote {out_dir}/drive_test.mp4, drive_test.png, drive_test.json", flush=True)
    os._exit(0)


if __name__ == "__main__":
    main()
