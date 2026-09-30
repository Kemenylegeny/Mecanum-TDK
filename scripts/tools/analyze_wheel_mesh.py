# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Measure the geometry of a mecanum wheel from a single-piece mesh (hub and rollers fused, e.g. a CAD STL export).

Steps: (1) wheel axis = the principal axis of inertia with the distinct moment (rotational symmetry), center =
center of mass; (2) the outermost surface points (radius > ``--outer_band`` below the maximum) fall apart into one
cluster per roller -> number of rollers; (3) per roller: a line fitted to the roller's surface points (PCA) gives the
roller axis, its angle to the wheel axis (45 deg for a standard mecanum wheel), the handedness, the distance ``Rc`` of
the roller axis from the wheel axis at the roller middle, the roller radius profile along the roller and its length.

The result (``wheel_mesh_analysis.json``) holds the parameters for ``generate_mecanum_robot.py`` (``WheelParams``).
Plots the radius profile and the clusters (``wheel_mesh_analysis.png``).

Usage::

    python scripts/tools/analyze_wheel_mesh.py assets/robots/own_mecanum/stl/kerek_jobb_egydarab.stl --unit 0.001
"""

import argparse
import json
import math
import os

import matplotlib
import numpy as np
import trimesh
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def wheel_frame(mesh: trimesh.Trimesh) -> tuple[np.ndarray, np.ndarray]:
    """Center and unit axis of the wheel: the principal axis whose moment differs from the other two."""
    center = mesh.center_mass
    moments, vectors = trimesh.inertia.principal_axis(mesh.moment_inertia)
    # the two moments perpendicular to the axis are (nearly) equal
    d = [abs(moments[1] - moments[2]), abs(moments[0] - moments[2]), abs(moments[0] - moments[1])]
    axis = vectors[int(np.argmin(d))]
    return center, axis / np.linalg.norm(axis)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("mesh")
    parser.add_argument("--unit", type=float, default=0.001, help="Mesh unit in meters (STL from CAD: mm).")
    parser.add_argument("--samples", type=int, default=400_000)
    parser.add_argument("--outer_band", type=float, default=0.0015, help="[m] band below the max. radius for clustering.")
    parser.add_argument("--out_dir", default=None)
    args = parser.parse_args()

    mesh = trimesh.load(args.mesh)
    mesh.apply_scale(args.unit)
    center, axis = wheel_frame(mesh)
    # wheel frame: y = axis, x / z perpendicular
    tmp = np.array([1.0, 0.0, 0.0]) if abs(axis[0]) < 0.9 else np.array([0.0, 0.0, 1.0])
    ex = np.cross(axis, tmp)
    ex /= np.linalg.norm(ex)
    ez = np.cross(ex, axis)
    pts, _ = trimesh.sample.sample_surface_even(mesh, args.samples, seed=0)
    local = np.stack([(pts - center) @ ex, (pts - center) @ axis, (pts - center) @ ez], axis=1)
    r = np.hypot(local[:, 0], local[:, 2])
    theta = np.arctan2(local[:, 0], -local[:, 2])
    R = float(r.max())
    width = float(local[:, 1].max() - local[:, 1].min())

    # rollers: the outer band splits into one connected cluster per roller
    outer = local[r > R - args.outer_band]
    tree = cKDTree(outer)
    pairs = tree.query_pairs(r=0.0012, output_type="ndarray")
    import scipy.sparse as sp

    graph = sp.coo_matrix((np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])), shape=(len(outer), len(outer)))
    n_comp, labels = connected_components(graph, directed=False)
    sizes = np.bincount(labels)
    big = [k for k in range(n_comp) if sizes[k] > 0.2 * np.median(sizes[sizes > 20])]
    rollers = []
    for k in big:
        c = outer[labels == k]
        # angular position of the cluster (center of the outer patch)
        th = math.atan2(c[:, 0].mean(), -c[:, 2].mean())
        rollers.append({"theta": th, "n": int(len(c))})
    rollers.sort(key=lambda x: x["theta"])
    n = len(rollers)

    # per roller: all surface points within the roller's angular sector and near the rim -> PCA line
    results = []
    for rl in rollers:
        dth = np.angle(np.exp(1j * (theta - rl["theta"])))
        sel = local[(np.abs(dth) < math.pi / n * 1.2) & (r > R * 0.55)]
        # iterate: fit line, keep points close to it (the roller), refit
        c0 = sel.mean(axis=0)
        for _ in range(3):
            u = np.linalg.svd(sel - c0, full_matrices=False)[2][0]
            s = (sel - c0) @ u
            dist = np.linalg.norm((sel - c0) - np.outer(s, u), axis=1)
            keep = dist < np.percentile(dist, 70)
            c0 = sel[keep].mean(axis=0)
        u = np.linalg.svd(sel[keep] - c0, full_matrices=False)[2][0]
        s = (sel - c0) @ u
        dist = np.linalg.norm((sel - c0) - np.outer(s, u), axis=1)
        roller_pts = dist < R * 0.4
        s, dist = s[roller_pts], dist[roller_pts]
        # the roller surface: outermost distance from the axis line per axial slice
        bins = np.linspace(s.min(), s.max(), 25)
        prof = [(0.5 * (a + b), dist[(s >= a) & (s < b)].max()) for a, b in zip(bins[:-1], bins[1:]) if np.any((s >= a) & (s < b))]
        prof = np.array(prof)
        # the roller length: where the profile is within 1 mm of the envelope's needs (ignore hub parts)
        mid_radius = float(prof[:, 1][np.argmin(np.abs(prof[:, 0]))])
        angle = math.degrees(math.acos(min(1.0, abs(u[1]))))  # angle between roller axis and wheel axis
        # handedness: sign of the axial component vs. the tangential component at the roller position
        tangent = np.array([math.cos(rl["theta"]), 0.0, math.sin(rl["theta"])])
        h = int(np.sign((u @ np.array([0.0, 1.0, 0.0])) * (u @ tangent)))
        rc = float(np.hypot(c0[0], c0[2]))
        results.append({"theta_deg": math.degrees(rl["theta"]), "angle_to_wheel_axis_deg": 90.0 - angle if angle > 45.5 else angle,
                        "tilt_deg": angle, "handedness": h, "roller_axis_radius_Rc": rc, "mid_radius": mid_radius,
                        "length_estimate": float(prof[:, 0].max() - prof[:, 0].min()), "profile": prof.round(5).tolist()})  # fmt: skip

    summary = {
        "mesh": os.path.abspath(args.mesh),
        "envelope_radius_R": R,
        "wheel_width": width,
        "num_rollers": n,
        "roller_angle_deg_median": float(np.median([x["tilt_deg"] for x in results])) if results else None,
        "handedness": int(np.sign(sum(x["handedness"] for x in results))) if results else None,
        "Rc_median": float(np.median([x["roller_axis_radius_Rc"] for x in results])) if results else None,
        "roller_mid_radius_median": float(np.median([x["mid_radius"] for x in results])) if results else None,
        "roller_length_median": float(np.median([x["length_estimate"] for x in results])) if results else None,
        "volume_cm3": float(mesh.volume * 1e6) if mesh.is_watertight else None,
    }
    out_dir = args.out_dir or os.path.dirname(os.path.abspath(args.mesh))
    with open(os.path.join(out_dir, "wheel_mesh_analysis.json"), "w") as f:
        json.dump({"summary": summary, "rollers": results}, f, indent=2)

    fig, axs = plt.subplots(1, 3, figsize=(16, 5))
    axs[0].scatter(outer[:, 0] * 1e3, outer[:, 2] * 1e3, s=0.3, c=labels, cmap="tab20")
    axs[0].set_aspect("equal")
    axs[0].set_title(f"outer band (r > R - {args.outer_band * 1e3:.1f} mm): {n} rollers")
    axs[1].scatter(theta, local[:, 1] * 1e3, s=0.05, c=r, cmap="viridis")
    axs[1].set_xlabel("wheel angle [rad]")
    axs[1].set_ylabel("axial [mm]")
    axs[1].set_title("surface points (color = radius)")
    for x in results:
        p = np.array(x["profile"])
        axs[2].plot(p[:, 0] * 1e3, p[:, 1] * 1e3, lw=0.8)
    axs[2].set_xlabel("along roller [mm]")
    axs[2].set_ylabel("roller radius [mm]")
    axs[2].set_title("roller radius profiles")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "wheel_mesh_analysis.png"), dpi=90)
    print(json.dumps(summary, indent=2))
    for x in results:
        print(f"  roller at {x['theta_deg']:7.1f} deg: tilt {x['tilt_deg']:.1f} deg, h {x['handedness']:+d}, Rc {x['roller_axis_radius_Rc'] * 1e3:.2f} mm, "
              f"mid radius {x['mid_radius'] * 1e3:.2f} mm, length ~{x['length_estimate'] * 1e3:.1f} mm")  # fmt: skip


if __name__ == "__main__":
    main()
