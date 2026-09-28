# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Summarize one or more outputs of ``validate_mecanum_wheels.py`` (plain Python, no simulator needed).

Usage::

    python scripts/tools/summarize_validation.py logs/validation/hz*.json
"""

import argparse
import json
import os

import numpy as np

# pass criteria (chosen for a model compared against the ideal kinematic model, not a real robot)
CRITERIA = {
    "kin_rel_error": 0.05,  # speed error vs. ideal kinematics (TIAGo: 3-8 % vs. the real robot)
    "kin_cross_track": 0.02,  # sideways drift / commanded speed
    "symmetry": 0.02,  # |+ - -| / commanded speed
    "height_ripple_mm": 1.0,
    "off_ground": 0.01,  # fraction of steps in which a wheel has no roller carrying > 0.5 N
    "accel_ratio": (0.8, 1.2),  # traction limited acceleration / prediction
}


def metrics(r: dict) -> dict:
    cases = r["cases"]
    kin = {n: c for n, c in cases.items() if "rel_error" in c}
    m = {}
    m["kin_rel_error_max"] = max(c["rel_error"] for c in kin.values())
    m["kin_rel_error_by_type"] = {
        t: max(c["rel_error"] for n, c in kin.items() if n.startswith(t))
        for t in ("forward", "lateral", "diag", "rotate")
    }
    m["kin_cross_track_max"] = max(abs(c.get("cross_track_ratio", 0.0)) for c in kin.values())
    sym = []
    for n, c in kin.items():
        if "+" in n:
            other = kin.get(n.replace("+", "-"))
            if other:
                sym.append(abs(abs(c["along_ratio"]) - abs(other["along_ratio"])))
    m["symmetry_max"] = max(sym)
    m["idle_wheel_torque_Nm"] = max(c["idle_wheel_torque"] for c in kin.values() if "idle_wheel_torque" in c)
    m["driven_wheel_torque_Nm"] = max(c["driven_wheel_torque"] for c in kin.values() if "driven_wheel_torque" in c)
    m["height_ripple_mm_max"] = max(c["height_ripple_mm"] for c in kin.values())
    m["off_ground_max"] = max(c["steps_with_wheel_off_ground"] for c in kin.values())
    m["off_ground_rest"] = cases["rest"]["steps_with_wheel_off_ground"]
    m["non_ground_force_max"] = max(c["max_non_ground_force"] for c in cases.values())
    m["chassis_force_max"] = max(c["max_chassis_contact_force"] for c in cases.values())
    m["load_over_weight"] = [round(min(c["load_over_weight"] for c in cases.values()), 3),
                             round(max(c["load_over_weight"] for c in cases.values()), 3)]
    m["accel_ratio"] = {n[5:]: round(c["accel_ratio"], 3) for n, c in cases.items() if n.startswith("step_")}
    m["slip_onset"] = {}
    for d in ("forward", "diagonal"):
        ramps = sorted((c["factor"], c["max_slip_speed"], c["accel_achieved"]) for n, c in cases.items()
                       if n.startswith(f"ramp_{d}"))
        m["slip_onset"][d] = [(f, round(s, 3), round(a, 2)) for f, s, a in ramps]
    vib = []
    for n, c in kin.items():
        if n.startswith("forward") or n.startswith("lateral"):
            vib.append((n, round(c["roller_passing_hz"], 1), c["vibration_peaks_per_roller_passing"],
                        c["vibration_peaks_rel_power"]))
    m["vibration"] = vib  # (case, roller passing Hz, top peaks / roller passing, relative power)
    return m


def verdicts(m: dict) -> dict:
    lo, hi = CRITERIA["accel_ratio"]
    return {
        "1 geometry/contact": m["height_ripple_mm_max"] < CRITERIA["height_ripple_mm"]
        and m["off_ground_max"] < CRITERIA["off_ground"]
        and m["non_ground_force_max"] < 1e-3
        and m["chassis_force_max"] < 1e-3,
        "2 kinematics": m["kin_rel_error_max"] < CRITERIA["kin_rel_error"]
        and m["kin_cross_track_max"] < CRITERIA["kin_cross_track"]
        and m["symmetry_max"] < CRITERIA["symmetry"],
        "3 traction": all(lo <= a <= hi for a in m["accel_ratio"].values()),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("files", nargs="+")
    args = parser.parse_args()
    rows = []
    for f in args.files:
        r = json.load(open(f))
        m = metrics(r)
        v = verdicts(m)
        rows.append((os.path.basename(f), m, v))
        print(f"\n=== {os.path.basename(f)}: {r['physics_hz']:.0f} Hz, pos iters {r['solver_iters']}, "
              f"vel iters {r.get('vel_iters')}, contact offset {r.get('contact_offset')}, "
              f"roller armature {r.get('roller_armature')}, mu {r['mu']}, mass {r['robot_mass']:.1f} kg, "
              f"{r.get('num_links')} links, {r.get('num_envs')} envs, max root height {r.get('max_root_height', 0):.3f} m, "
              f"copies max speed diff {r.get('copies_max_speed_diff', 0):.4f} m/s")
        for k, val in m.items():
            print(f"  {k:24s} {np.round(val, 4) if isinstance(val, float) else val}")
        print("  verdicts:", {k: ("PASS" if ok else "FAIL") for k, ok in v.items()})
    if len(rows) > 1:
        print("\n=== convergence")
        keys = ("kin_rel_error_max", "kin_cross_track_max", "height_ripple_mm_max", "off_ground_max")
        print(f"  {'run':28s}" + "".join(f"{k:>22s}" for k in keys) + "   accel_ratio (fwd/lat/diag)")
        for name, m, _ in rows:
            a = m["accel_ratio"]
            print(f"  {name:28s}" + "".join(f"{m[k]:22.4f}" for k in keys)
                  + f"   {a.get('forward')}/{a.get('lateral')}/{a.get('diagonal')}")


if __name__ == "__main__":
    main()
