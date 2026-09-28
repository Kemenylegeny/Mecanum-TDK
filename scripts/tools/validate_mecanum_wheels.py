# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Open-loop validation of the roller mecanum robot (no RL).

Every environment runs one test case in parallel on flat ground with a known friction coefficient; all quantities
are logged at the physics rate from the ground truth state:

1. geometry & contact: chassis height ripple, roller contacts per wheel, non-ground contacts, load vs. weight
2. kinematics: forward / lateral / diagonal / rotation at two speeds and both signs vs. the ideal mecanum model,
   cross-coupling, symmetry; on the diagonal the two idle wheels are unpowered and must stay still
3. traction limit: step and ramp commands; slip onset and max. acceleration vs. mu*g/sqrt(2) and mu*g/2
4. numerics: run the script at several physics rates / solver iterations and compare the JSON outputs
   (``scripts/tools/summarize_validation.py``); vibration spectrum vs. roller passing frequency

Usage::

    python scripts/tools/validate_mecanum_wheels.py --headless --physics_hz 360 --out logs/validation/360hz.json
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Open-loop validation of the roller mecanum robot.")
parser.add_argument("--physics_hz", type=float, default=360.0)
parser.add_argument("--solver_iters", type=int, default=8, help="PhysX position iterations of the articulation.")
parser.add_argument("--mu", type=float, default=0.8, help="Roller-ground friction coefficient (static = dynamic).")
parser.add_argument("--vel_iters", type=int, default=None, help="PhysX velocity iterations (default: robot cfg).")
parser.add_argument("--contact_offset", type=float, default=None, help="Contact offset of the colliders [m].")
parser.add_argument("--roller_armature", type=float, default=None, help="Armature of the roller joints [kg m^2].")
parser.add_argument("--contact_stiffness", type=float, default=None, help="Roller contact stiffness [N/m], 0: rigid.")
parser.add_argument("--contact_damping", type=float, default=None, help="Roller contact damping [N s/m].")
parser.add_argument("--copies", type=int, default=2, help="Run every case this many times (checks many-env stability).")
parser.add_argument("--out", type=str, default=None, help="Path of the JSON result file (raw logs go to .npz).")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import json
import math
import os

import numpy as np
import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.utils import configclass

from mecanum_ws.robots import mecanum

G = 9.81
T_SETTLE = 0.5
T_END = 5.0
KIN_WINDOW = (2.5, 5.0)


@configclass
class ValidationSceneCfg(InteractiveSceneCfg):
    ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg(physics_material=mecanum.ground_material(args_cli.mu)),
    )
    robot: ArticulationCfg = mecanum.MECANUM_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    rollers = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*_roller_.*",
        filter_prim_paths_expr=["/World/ground/GroundPlane/CollisionPlane"],
    )
    chassis = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/" + mecanum.BASE_LINK_NAME)
    light = AssetBaseCfg(prim_path="/World/light", spawn=sim_utils.DomeLightCfg(intensity=1000.0))


##
# Test cases
##

K = mecanum.WHEEL_BASE_HALF_LENGTH + mecanum.TRACK_HALF_WIDTH
IK = torch.tensor([[1.0, -1.0, -K], [1.0, 1.0, K], [1.0, 1.0, -K], [1.0, -1.0, K]]) / mecanum.WHEEL_RADIUS
FK = torch.linalg.pinv(IK)
A_PRED = {"forward": args_cli.mu * G / math.sqrt(2.0), "lateral": args_cli.mu * G / math.sqrt(2.0)}
A_PRED["diagonal"] = args_cli.mu * G / 2.0
DIRS = {"forward": (1.0, 0.0), "lateral": (0.0, 1.0), "diagonal": (1 / math.sqrt(2), 1 / math.sqrt(2))}


def build_cases() -> list[dict]:
    cases = [{"name": "rest", "kind": "rest", "twist": (0.0, 0.0, 0.0)}]
    # kinematics: slow ramp to a constant twist
    for speed in (0.3, 0.8):
        for sign in (1, -1):
            s = sign * speed
            d = 1 / math.sqrt(2)
            cases += [
                {"name": f"forward_{s:+.1f}", "kind": "kin", "twist": (s, 0.0, 0.0)},
                {"name": f"lateral_{s:+.1f}", "kind": "kin", "twist": (0.0, s, 0.0)},
                # the IK commands 0 to two wheels (FL/RR resp. FR/RL): they must need no torque to stay still
                {"name": f"diag_xy_{s:+.1f}", "kind": "kin", "twist": (s * d, s * d, 0.0), "idle": (0, 3)},
                {"name": f"diag_x-y_{s:+.1f}", "kind": "kin", "twist": (s * d, -s * d, 0.0), "idle": (1, 2)},
            ]
    for wz in (0.6, 1.5):
        for sign in (1, -1):
            cases.append({"name": f"rotate_{sign * wz:+.1f}", "kind": "kin", "twist": (0.0, 0.0, sign * wz)})
    # traction: velocity step (motor torque >> traction) and acceleration ramps around the predicted limit
    for direction in ("forward", "lateral", "diagonal"):
        dx, dy = DIRS[direction]
        cases.append({"name": f"step_{direction}", "kind": "step", "dir": direction, "twist": (2 * dx, 2 * dy, 0.0)})
    for direction in ("forward", "diagonal"):
        dx, dy = DIRS[direction]
        for f in (0.5, 0.7, 0.9, 1.1, 1.3, 1.6):
            cases.append({
                "name": f"ramp_{direction}_{f:.1f}",
                "kind": "ramp",
                "dir": direction,
                "factor": f,
                "accel": f * A_PRED[direction],
                "twist": (1.5 * dx, 1.5 * dy, 0.0),
            })
    return cases


def twist_at(case: dict, t: float) -> np.ndarray:
    target = np.array(case["twist"])
    if t < T_SETTLE or case["kind"] == "rest":
        return np.zeros(3)
    if case["kind"] == "kin":
        return target * min(1.0, (t - T_SETTLE) / 1.0)
    if case["kind"] == "step":
        return target
    # ramp: constant commanded acceleration up to the target speed
    speed = np.linalg.norm(target[:2])
    return target * min(1.0, case["accel"] * (t - T_SETTLE) / speed)


##
# Main
##


def main():
    dt = 1.0 / args_cli.physics_hz
    # the robot colliders have no material bound, so they use the default (sim) material
    robot_material = mecanum.ROBOT_MATERIAL.replace(static_friction=1.0, dynamic_friction=1.0)
    if args_cli.contact_stiffness is not None:
        robot_material.compliant_contact_stiffness = args_cli.contact_stiffness
    if args_cli.contact_damping is not None:
        robot_material.compliant_contact_damping = args_cli.contact_damping
    sim_cfg = sim_utils.SimulationCfg(dt=dt, device=args_cli.device, physics_material=robot_material)

    sim = sim_utils.SimulationContext(sim_cfg)
    cases = build_cases()
    num_cases = len(cases)
    cases = cases * args_cli.copies
    scene_cfg = ValidationSceneCfg(num_envs=len(cases), env_spacing=10.0)
    scene_cfg.robot.spawn.activate_contact_sensors = True
    # compliant contact (roller elasticity) must be set on the ground material to take effect
    scene_cfg.ground.spawn.physics_material.compliant_contact_stiffness = robot_material.compliant_contact_stiffness
    scene_cfg.ground.spawn.physics_material.compliant_contact_damping = robot_material.compliant_contact_damping
    scene_cfg.robot.spawn.articulation_props.solver_position_iteration_count = args_cli.solver_iters
    if args_cli.vel_iters is not None:
        scene_cfg.robot.spawn.articulation_props.solver_velocity_iteration_count = args_cli.vel_iters
    if args_cli.contact_offset is not None:
        scene_cfg.robot.spawn.collision_props = sim_utils.CollisionPropertiesCfg(
            contact_offset=args_cli.contact_offset, rest_offset=0.0
        )
    if args_cli.roller_armature is not None:
        scene_cfg.robot.actuators["rollers"].armature = args_cli.roller_armature
    scene = InteractiveScene(scene_cfg)
    sim.reset()

    robot = scene["robot"]
    rollers = scene["rollers"]
    device = sim.device
    wheel_ids, _ = robot.find_joints(mecanum.WHEEL_JOINT_NAMES, preserve_order=True)
    ik, fk = IK.to(device), FK.to(device)
    signs = torch.tensor(mecanum.WHEEL_JOINT_SIGNS, device=device)
    roller_wheel = torch.tensor(
        [[n.startswith(w) for n in rollers.body_names] for w in ("front_left", "front_right", "rear_left", "rear_right")],
        device=device,
    ).float()  # (4, num_rollers)
    mass = robot.root_physx_view.get_masses()[0].sum().item()
    rollers_per_wheel = len(rollers.body_names) // 4

    n_steps = int(T_END / dt)
    log = {k: [] for k in ("v_b", "w_z", "z", "wheel_cmd", "wheel_vel", "wheel_torque", "roller_fz", "fz", "non_ground", "chassis")}
    for step in range(n_steps):
        t = step * dt
        twist = torch.tensor(np.stack([twist_at(c, t) for c in cases]), device=device, dtype=torch.float32)
        wheel_cmd = torch.clamp(twist @ ik.T, -mecanum.MAX_WHEEL_SPEED, mecanum.MAX_WHEEL_SPEED) * signs
        robot.set_joint_velocity_target(wheel_cmd, joint_ids=wheel_ids)
        scene.write_data_to_sim()
        sim.step(render=False)
        scene.update(dt)

        f_net = rollers.data.net_forces_w  # (N, R, 3)
        f_ground = rollers.data.force_matrix_w[:, :, 0]  # (N, R, 3)
        # logs stay on the GPU until the end (no per-step synchronization)
        log["v_b"].append(robot.data.root_lin_vel_b[:, :2].clone())
        log["w_z"].append(robot.data.root_ang_vel_b[:, 2].clone())
        log["z"].append(robot.data.root_pos_w[:, 2].clone())
        log["wheel_cmd"].append(wheel_cmd * signs)
        log["wheel_vel"].append(robot.data.joint_vel[:, wheel_ids] * signs)
        log["wheel_torque"].append(robot.data.applied_torque[:, wheel_ids] * signs)
        log["roller_fz"].append(f_net[..., 2].clone())
        log["fz"].append(f_net[..., 2].sum(dim=1))
        log["non_ground"].append(torch.linalg.norm(f_net - f_ground, dim=-1).max(dim=1).values)
        log["chassis"].append(torch.linalg.norm(scene["chassis"].data.net_forces_w[:, 0], dim=-1))
    log = {k: torch.stack(v).cpu().numpy() for k, v in log.items()}  # (T, N, ...)
    # rollers of a wheel carrying more than 0.5 N
    log["contacts"] = np.einsum("tnr,wr->tnw", (log["roller_fz"] > 0.5).astype(np.float32), roller_wheel.cpu().numpy())
    time = np.arange(n_steps) * dt
    v_fk = np.einsum("ij,tnj->tni", fk.cpu().numpy(), log["wheel_vel"])  # body twist from wheel speeds

    results = {
        "physics_hz": args_cli.physics_hz,
        "solver_iters": args_cli.solver_iters,
        "vel_iters": args_cli.vel_iters,
        "contact_offset": args_cli.contact_offset,
        "roller_armature": args_cli.roller_armature,
        "contact_stiffness": robot_material.compliant_contact_stiffness,
        "contact_damping": robot_material.compliant_contact_damping,
        "mu": args_cli.mu,
        "robot_mass": mass,
        "num_links": robot.num_bodies,
        "num_envs": len(cases),
        # copies of the same case must agree, and no robot may jump (many-env stability)
        "max_root_height": float(log["z"].max()),
        "copies_max_speed_diff": float(np.abs(log["v_b"][:, num_cases:] - np.tile(log["v_b"][:, :num_cases], (1, args_cli.copies - 1, 1))).max())
        if args_cli.copies > 1
        else 0.0,
        "cases": {},
    }
    win = (time >= KIN_WINDOW[0]) & (time < KIN_WINDOW[1])
    active = time >= T_SETTLE + 0.2
    for i, c in enumerate(cases[:num_cases]):
        r = {
            "min_contacts_per_wheel": int(log["contacts"][active, i].min()),
            "steps_with_wheel_off_ground": float((log["contacts"][active, i].min(axis=1) < 1).mean()),
            "max_non_ground_force": float(log["non_ground"][active, i].max()),
            "max_chassis_contact_force": float(log["chassis"][active, i].max()),
            "load_over_weight": float(log["fz"][active, i].mean() / (mass * G)),
        }
        cmd = np.array(c["twist"])
        if c["kind"] in ("kin", "rest"):
            v = log["v_b"][win, i].mean(axis=0)
            wz = log["w_z"][win, i].mean()
            meas = np.array([v[0], v[1], wz])
            r["measured_twist"] = meas.round(4).tolist()
            z = log["z"][win, i]
            r["height_ripple_mm"] = float((z.max() - z.min()) * 1e3)
            if c["kind"] == "kin":
                lin = np.linalg.norm(cmd[:2])
                if lin > 0:
                    u = cmd[:2] / lin
                    r["rel_error"] = float(np.linalg.norm(meas[:2] - cmd[:2]) / lin)
                    r["along_ratio"] = float(meas[:2] @ u / lin)
                    r["cross_track_ratio"] = float((meas[0] * -u[1] + meas[1] * u[0]) / lin)
                    r["yaw_rate_drift"] = float(wz)
                else:
                    r["rel_error"] = float(abs(wz - cmd[2]) / abs(cmd[2]))
                    r["along_ratio"] = float(wz / cmd[2])
                    r["linear_drift"] = float(np.linalg.norm(meas[:2]))
                r["slip_speed"] = float(np.abs(v_fk[win, i, :2] - log["v_b"][win, i]).max())
                if "idle" in c:
                    idle = list(c["idle"])
                    driven = [j for j in range(4) if j not in idle]
                    r["idle_wheel_speed"] = float(np.abs(log["wheel_vel"][win, i][:, idle]).mean())
                    r["idle_wheel_torque"] = float(np.abs(log["wheel_torque"][win, i][:, idle]).mean())
                    r["driven_wheel_speed"] = float(np.abs(log["wheel_vel"][win, i][:, driven]).mean())
                    r["driven_wheel_torque"] = float(np.abs(log["wheel_torque"][win, i][:, driven]).mean())
                # spectrum of the chassis vertical acceleration above the suspension modes (< 8 Hz), with the peaks
                # expressed as multiples of the roller passing frequency (rollers * wheel revolutions per second)
                acc = np.gradient(np.gradient(z, dt), dt)
                spec = np.abs(np.fft.rfft((acc - acc.mean()) * np.hanning(len(acc))))
                freq = np.fft.rfftfreq(len(acc), dt)
                band = (freq > 8.0) & (freq < 0.45 / dt)
                peaks = [j for j in range(1, len(spec) - 1) if band[j] and spec[j] >= spec[j - 1] and spec[j] >= spec[j + 1]]
                peaks = sorted(peaks, key=lambda j: -spec[j])[:3]
                wheel_rate = np.abs(log["wheel_vel"][win, i]).mean() / (2 * math.pi)
                r["roller_passing_hz"] = float(rollers_per_wheel * wheel_rate)
                r["vibration_peaks_hz"] = freq[peaks].round(2).tolist()
                r["vibration_peaks_rel_power"] = (spec[peaks] / spec[peaks[0]]).round(2).tolist()
                r["vibration_peaks_per_roller_passing"] = (freq[peaks] / r["roller_passing_hz"]).round(2).tolist()
        else:
            u = np.array(DIRS[c["dir"]])
            v_along = log["v_b"][:, i] @ u
            fk_along = v_fk[:, i, :2] @ u
            if c["kind"] == "step":
                sel = (time >= T_SETTLE + 0.05) & (time < T_SETTLE + 0.25)
                r["accel_measured"] = float(np.polyfit(time[sel], v_along[sel], 1)[0])
                r["accel_predicted"] = A_PRED[c["dir"]]
                r["accel_ratio"] = r["accel_measured"] / r["accel_predicted"]
            else:
                t_end = T_SETTLE + 1.2 / c["accel"]  # until the commanded speed reaches 1.2 m/s
                sel = (time >= T_SETTLE + 0.1) & (time < t_end)
                r["accel_commanded"] = c["accel"]
                r["factor"] = c["factor"]
                r["max_slip_speed"] = float(np.abs(fk_along[sel] - v_along[sel]).max())
                r["accel_achieved"] = float(np.polyfit(time[sel], v_along[sel], 1)[0])
        results["cases"][c["name"]] = r

    print(json.dumps(results, indent=1))
    if args_cli.out:
        os.makedirs(os.path.dirname(os.path.abspath(args_cli.out)), exist_ok=True)
        with open(args_cli.out, "w") as f:
            json.dump(results, f, indent=1)
        np.savez_compressed(os.path.splitext(args_cli.out)[0] + ".npz", time=time, **log)
        print(f"[INFO] Wrote {args_cli.out}")


if __name__ == "__main__":
    main()
    simulation_app.close()
