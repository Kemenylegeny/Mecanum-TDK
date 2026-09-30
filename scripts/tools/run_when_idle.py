# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Queue a command: wait until no training / evaluation of this workspace runs and the GPU has enough free memory.

Idle = none of ``BUSY_PATTERNS`` in the process list (other than this queue and its own command) and at least
``--min_free_mb`` free GPU memory, for ``--confirm`` consecutive checks. Then runs the command and exits with its code.

Usage (detached)::

    setsid nohup python -u scripts/tools/run_when_idle.py -- python -u scripts/tools/train_eval_log.py ... \\
        > logs/queue/<name>.log 2>&1 < /dev/null &
"""

import argparse
import os
import subprocess
import sys
import time

BUSY_PATTERNS = [
    "scripts/rsl_rl/train.py", "scripts/rsl_rl/play.py", "scripts/rsl_rl/evaluate_navigation.py",
    "scripts/tools/train_eval_log.py", "scripts/tools/goal_weight_sweep.py", "scripts/tools/slip_sweep.py",
    "scripts/tools/penalty_study.py", "scripts/tools/staged_penalties.py", "scripts/tools/tune_rewards.py",
    "scripts/tools/motion_diagnostics.py", "scripts/tools/drive_test.py", "scripts/tools/validate_mecanum_wheels.py",
]  # fmt: skip


def log(msg: str):
    print(f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}", flush=True)


def busy_processes() -> list[str]:
    me = os.getpid()
    out = subprocess.run(["ps", "-eo", "pid=,ppid=,args="], capture_output=True, text=True).stdout.splitlines()
    busy = []
    for line in out:
        pid, ppid, cmd = line.strip().split(None, 2) if len(line.split()) >= 3 else (0, 0, "")
        if int(pid) == me or "run_when_idle.py" in cmd:
            continue
        if any(p in cmd for p in BUSY_PATTERNS):
            busy.append(f"{pid}: {cmd[:120]}")
    return busy


def free_gpu_mb() -> int:
    out = subprocess.run(["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"], capture_output=True, text=True)
    return int(out.stdout.split()[0]) if out.returncode == 0 else 0


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--poll", type=float, default=60.0, help="[s] between checks.")
    parser.add_argument("--confirm", type=int, default=2, help="Consecutive idle checks before starting.")
    parser.add_argument("--min_free_mb", type=int, default=11500, help="Free GPU memory needed [MiB].")
    parser.add_argument("command", nargs=argparse.REMAINDER, help="Command to run (after --).")
    args = parser.parse_args()
    cmd = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not cmd:
        parser.error("no command given")
    log(f"queued: {' '.join(cmd)}")
    idle, last = 0, None
    while idle < args.confirm:
        busy, free = busy_processes(), free_gpu_mb()
        state = f"{len(busy)} busy, {free} MiB free"
        if busy or free < args.min_free_mb:
            idle = 0
            if state != last:
                log(f"waiting ({state}): " + "; ".join(busy[:3]))
            last = state
        else:
            idle += 1
        time.sleep(args.poll if idle < args.confirm else 0)
    log("idle -> starting")
    code = subprocess.run(cmd).returncode
    log(f"finished with exit code {code}")
    sys.exit(code)


if __name__ == "__main__":
    main()
