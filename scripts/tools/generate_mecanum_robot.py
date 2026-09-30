# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Generate a mecanum robot URDF with freely rotating rollers, sphere roller colliders and a pendulum front axle.

Wheel geometry is based on the FUJI FM202-205-15U mecanum wheel of `fuji_mecanum`_ (45° rollers, roller axis 0.0895 m
from the wheel axis, 13 mm roller radius, 102.5 mm envelope radius). As in the TIAGo Isaac integration (`tiago_isaac`_),
each roller collides through a row of spheres whose radii are chosen so that every sphere touches the cylindrical wheel
envelope; the hub has no collider, so the only contacts of a wheel are roller-ground contacts.

Deviations from the FUJI wheel / TIAGo robot, both forced by the simulator:

* 14 instead of 15 rollers (rollers 60 instead of 56 mm long to keep the roller overlap): PhysX GPU articulations with
  more than 64 links silently blow up when many environments are simulated (seen from ~64 envs), and
  1 chassis + 1 axle + 4 hubs + 4 * 15 rollers = 66 links.
* pendulum (rocker) front axle: a rigid 4-wheel chassis is statically indeterminate on rigid contacts and rocks between
  its diagonals, which unloads the driven wheels (e.g. on the diagonal only two wheels drive).

Wheel frame: x forward, y = wheel axle (joint axis), z up; a positive joint velocity drives the robot forward.
The roller at wheel angle ``th`` (0 = bottom) sits at ``Rc * r(th)`` with ``r(th) = (sin th, 0, -cos th)`` and its axis is
``u = cos(g) * t(th) + h * sin(g) * y``, with the tangent ``t(th) = (cos th, 0, sin th)``, the roller angle ``g`` and the
handedness ``h``. The bottom roller axis is thus ``(cos g, h sin g, 0)``, which gives the no-slip constraint
``w * R = vx + h * vy`` for that wheel (``h = -1`` on FL/RR and ``+1`` on FR/RL for the standard "X" configuration).

.. _fuji_mecanum: https://github.com/DaiGuard/fuji_mecanum
.. _tiago_isaac: https://github.com/AIS-Bonn/tiago_isaac

Usage::

    python scripts/tools/generate_mecanum_robot.py            # writes assets/robots/mecanum/robot.urdf
"""

import argparse
import json
import math
import os
from dataclasses import asdict, dataclass

import numpy as np
import trimesh


@dataclass
class WheelParams:
    num_rollers: int = 14
    roller_angle_deg: float = 45.0
    roller_center_radius: float = 0.0895
    """Distance of the roller axis (at its middle) from the wheel axis [m]."""
    roller_mid_radius: float = 0.013
    """Roller radius at its middle [m]; the wheel envelope radius is ``roller_center_radius + roller_mid_radius``."""
    roller_half_length: float = 0.030
    spheres_per_roller: int = 6
    roller_mass: float = 0.03
    roller_damping: float = 1.0e-5
    hub_radius: float = 0.075
    hub_width: float = 0.0644
    hub_mass: float = 1.0

    @property
    def radius(self) -> float:
        return self.roller_center_radius + self.roller_mid_radius


@dataclass
class RobotParams:
    chassis_size: tuple = (0.60, 0.40, 0.12)
    chassis_mass: float = 18.0
    wheel_x: float = 0.22
    """Half distance between front and rear axles (lx) [m]."""
    wheel_y: float = 0.26
    """Half distance between left and right wheel mid-planes (ly) [m]."""
    wheel_z: float = -0.02
    """Height of the wheel axes w.r.t. the chassis center [m]."""
    axle_mass: float = 1.0
    """Mass of the pendulum front axle [kg]."""
    axle_swing: float = 0.12
    """+- swing of the front axle around the x axis [rad] (~3 cm at the wheels)."""


WHEELS = {
    # name: (x sign, y sign, handedness h)
    "front_left": (1, 1, -1),
    "front_right": (1, -1, 1),
    "rear_left": (-1, 1, 1),
    "rear_right": (-1, -1, -1),
}


##
# Wheel geometry
##


def roller_profile(w: WheelParams, s: np.ndarray) -> np.ndarray:
    """Roller radius at axial coordinate ``s`` such that the roller surface lies on the wheel envelope."""
    c = math.cos(math.radians(w.roller_angle_deg))
    return w.radius - np.sqrt(w.roller_center_radius**2 + (s * c) ** 2)


def sphere_offsets(w: WheelParams) -> np.ndarray:
    """Axial positions of the collision spheres along a roller (spanning the full roller length)."""
    return np.linspace(-w.roller_half_length, w.roller_half_length, w.spheres_per_roller)


def roller_frame(w: WheelParams, th: float, h: int) -> tuple[np.ndarray, np.ndarray]:
    """Position and rotation matrix (columns = roller x, y, z axes) of a roller in the wheel frame."""
    g = math.radians(w.roller_angle_deg)
    radial = np.array([math.sin(th), 0.0, -math.cos(th)])
    tangent = np.array([math.cos(th), 0.0, math.sin(th)])
    axis = math.cos(g) * tangent + h * math.sin(g) * np.array([0.0, 1.0, 0.0])
    z = radial
    y = np.cross(z, axis)
    return w.roller_center_radius * radial, np.column_stack([axis, y, z])


def envelope_report(w: WheelParams, h: int = 1, samples: int = 721) -> dict:
    """Lowest point of the sphere colliders vs. wheel rotation over one roller pitch (roundness + contact handover)."""
    pitch = 2 * math.pi / w.num_rollers
    offsets = sphere_offsets(w)
    radii = roller_profile(w, offsets)
    depth = np.zeros(samples)
    owner = np.zeros(samples, dtype=int)
    for k, alpha in enumerate(np.linspace(0.0, pitch, samples)):
        best = -np.inf
        for i in range(w.num_rollers):
            pos, rot = roller_frame(w, i * pitch - alpha, h)
            centers = pos[None, :] + offsets[:, None] * rot[:, 0][None, :]
            d = (-centers[:, 2] + radii).max()  # distance of the lowest sphere point below the wheel axis
            if d > best:
                best, owner[k] = d, i
        depth[k] = best
    handovers = int(np.count_nonzero(np.diff(owner)))
    return {
        "envelope_radius": w.radius,
        "min_effective_radius": float(depth.min()),
        "max_effective_radius": float(depth.max()),
        "ripple_peak_to_peak_mm": float((depth.max() - depth.min()) * 1e3),
        "roller_handovers_per_pitch": handovers,
        "coverage_ok": bool(
            w.roller_half_length * math.cos(math.radians(w.roller_angle_deg)) / w.roller_center_radius
            >= math.tan(pitch / 2)
        ),
        "sphere_radii_mm": (radii * 1e3).round(2).tolist(),
    }


def roller_mesh(w: WheelParams, n_axial: int = 24, sections: int = 32) -> trimesh.Trimesh:
    """Barrel-shaped visual mesh of a roller, axis along x."""
    s = np.linspace(-w.roller_half_length, w.roller_half_length, n_axial)
    r = roller_profile(w, s)
    # closed profile in the (radius, axial) plane, revolved around z, then z -> x
    profile = np.column_stack([np.concatenate([[0.0], r, [0.0]]), np.concatenate([[s[0]], s, [s[-1]]])])
    mesh = trimesh.creation.revolve(profile, sections=sections)
    mesh.apply_transform(trimesh.transformations.rotation_matrix(math.pi / 2, [0, 1, 0]))
    return mesh


##
# URDF writing
##


def _fmt(v) -> str:
    return " ".join(f"{x:.6g}" for x in v)


def _rpy(rot: np.ndarray) -> tuple[float, float, float]:
    """URDF roll-pitch-yaw (R = Rz(yaw) Ry(pitch) Rx(roll)) of a rotation matrix."""
    pitch = math.asin(-max(-1.0, min(1.0, rot[2, 0])))
    roll = math.atan2(rot[2, 1], rot[2, 2])
    yaw = math.atan2(rot[1, 0], rot[0, 0])
    return roll, pitch, yaw


def _inertial(mass: float, ixx: float, iyy: float, izz: float) -> str:
    return (
        f'    <inertial><mass value="{mass:.6g}"/>'
        f'<inertia ixx="{ixx:.6g}" iyy="{iyy:.6g}" izz="{izz:.6g}" ixy="0" ixz="0" iyz="0"/></inertial>\n'
    )


def build_urdf(w: WheelParams, rb: RobotParams, roller_mesh_path: str) -> str:
    lx, ly, lz = rb.chassis_size
    m = rb.chassis_mass
    out = ['<?xml version="1.0"?>\n<robot name="mecanum_rollers">\n']
    out.append('  <material name="chassis"><color rgba="0.2 0.35 0.6 1"/></material>\n')
    out.append('  <material name="hub"><color rgba="0.15 0.15 0.15 1"/></material>\n')
    out.append('  <material name="roller"><color rgba="0.85 0.55 0.1 1"/></material>\n')
    out.append(
        '  <link name="base_link">\n'
        f'    <visual><geometry><box size="{_fmt(rb.chassis_size)}"/></geometry><material name="chassis"/></visual>\n'
        f'    <collision><geometry><box size="{_fmt(rb.chassis_size)}"/></geometry></collision>\n'
        + _inertial(m, m * (ly**2 + lz**2) / 12, m * (lx**2 + lz**2) / 12, m * (lx**2 + ly**2) / 12)
        + "  </link>\n"
    )
    # hub inertia: solid cylinder around y
    hr, hw, hm = w.hub_radius, w.hub_width, w.hub_mass
    hub_i = (hm * (3 * hr**2 + hw**2) / 12, hm * hr**2 / 2)
    # roller inertia: cylinder around its x axis
    rr, rl, rm = w.roller_mid_radius, 2 * w.roller_half_length, w.roller_mass
    roller_i = (rm * rr**2 / 2, rm * (3 * rr**2 + rl**2) / 12)
    offsets = sphere_offsets(w)
    radii = roller_profile(w, offsets)
    pitch = 2 * math.pi / w.num_rollers
    # pendulum front axle (swings around x): 3-point support, statically determinate
    am = rb.axle_mass
    out.append(
        '  <link name="front_axle">\n'
        f'    <visual><origin rpy="1.5708 0 0"/><geometry><cylinder radius="0.02" length="{2 * rb.wheel_y - hw:.4g}"/>'
        '</geometry><material name="hub"/></visual>\n'
        + _inertial(am, am * (2 * rb.wheel_y) ** 2 / 12, 1e-3, am * (2 * rb.wheel_y) ** 2 / 12)
        + "  </link>\n"
        '  <joint name="front_axle_joint" type="revolute">\n'
        '    <parent link="base_link"/><child link="front_axle"/>\n'
        f'    <origin xyz="{_fmt((rb.wheel_x, 0.0, rb.wheel_z))}"/><axis xyz="1 0 0"/>\n'
        f'    <limit lower="{-rb.axle_swing}" upper="{rb.axle_swing}" effort="1000" velocity="10"/>\n'
        "  </joint>\n"
    )
    for name, (sx, sy, h) in WHEELS.items():
        wheel = f"{name}_wheel"
        if sx > 0:
            parent, origin = "front_axle", (0.0, sy * rb.wheel_y, 0.0)
        else:
            parent, origin = "base_link", (sx * rb.wheel_x, sy * rb.wheel_y, rb.wheel_z)
        # hub: visual only, no collider (only roller-ground contacts)
        out.append(
            f'  <link name="{wheel}">\n'
            f'    <visual><origin rpy="1.5708 0 0"/><geometry><cylinder radius="{hr}" length="{hw}"/></geometry>'
            '<material name="hub"/></visual>\n'
            + _inertial(hm, hub_i[0], hub_i[1], hub_i[0])
            + "  </link>\n"
            f'  <joint name="{wheel}_joint" type="continuous">\n'
            f'    <parent link="{parent}"/><child link="{wheel}"/>\n'
            f'    <origin xyz="{_fmt(origin)}"/><axis xyz="0 1 0"/>\n'
            "  </joint>\n"
        )
        for i in range(w.num_rollers):
            roller = f"{name}_roller_{i:02d}"
            pos, rot = roller_frame(w, i * pitch, h)
            spheres = "".join(
                f'    <collision><origin xyz="{s:.6g} 0 0"/><geometry><sphere radius="{r:.6g}"/></geometry></collision>\n'
                for s, r in zip(offsets, radii)
            )
            out.append(
                f'  <link name="{roller}">\n'
                f'    <visual><geometry><mesh filename="{roller_mesh_path}"/></geometry><material name="roller"/></visual>\n'
                + spheres
                + _inertial(rm, roller_i[0], roller_i[1], roller_i[1])
                + "  </link>\n"
                f'  <joint name="{roller}_joint" type="continuous">\n'
                f'    <parent link="{wheel}"/><child link="{roller}"/>\n'
                f'    <origin xyz="{_fmt(pos)}" rpy="{_fmt(_rpy(rot))}"/><axis xyz="1 0 0"/>\n'
                f'    <dynamics damping="{w.roller_damping}"/>\n'
                "  </joint>\n"
            )
    out.append("</robot>\n")
    return "".join(out)


def main():
    default_out = os.path.join(os.path.dirname(__file__), "..", "..", "assets", "robots", "mecanum")
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--out_dir", default=os.path.abspath(default_out))
    parser.add_argument("--num_rollers", type=int, default=WheelParams.num_rollers)
    parser.add_argument("--spheres_per_roller", type=int, default=WheelParams.spheres_per_roller)
    parser.add_argument("--scale", type=float, default=1.0, help="Uniform scale of the FUJI wheel geometry.")
    parser.add_argument("--config", default=None,
                        help='JSON with "wheel" / "robot" fields overriding WheelParams / RobotParams (another robot).')  # fmt: skip
    args = parser.parse_args()

    w = WheelParams(num_rollers=args.num_rollers, spheres_per_roller=args.spheres_per_roller)
    for f in ("roller_center_radius", "roller_mid_radius", "roller_half_length", "hub_radius", "hub_width"):
        setattr(w, f, getattr(w, f) * args.scale)
    rb = RobotParams()
    if args.config:
        with open(args.config) as f:
            cfg = json.load(f)
        for obj, key in ((w, "wheel"), (rb, "robot")):
            for k, v in cfg.get(key, {}).items():
                if not hasattr(obj, k):
                    raise KeyError(f"unknown {key} parameter: {k}")
                setattr(obj, k, tuple(v) if isinstance(v, list) else v)

    os.makedirs(os.path.join(args.out_dir, "meshes"), exist_ok=True)
    roller_mesh(w).export(os.path.join(args.out_dir, "meshes", "roller.stl"))
    urdf = build_urdf(w, rb, "meshes/roller.stl")
    num_links = urdf.count("<link ")
    if num_links > 64:
        print(f"[WARN] {num_links} links: PhysX GPU articulations with > 64 links are unstable with many envs.")
    with open(os.path.join(args.out_dir, "robot.urdf"), "w") as f:
        f.write(urdf)
    report = {"wheel": asdict(w), "robot": asdict(rb), "num_links": num_links, "envelope": envelope_report(w)}
    with open(os.path.join(args.out_dir, "wheel_geometry.json"), "w") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report["envelope"], indent=2))
    print(f"[INFO] Wrote {args.out_dir}/robot.urdf ({num_links} links)")


if __name__ == "__main__":
    main()
