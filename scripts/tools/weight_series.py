# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Train a series of runs that differ only in one reward weight, one after another (``train_eval_log.py`` each).

Every run: from scratch, training -> evaluation -> motion diagnostics -> dated EXPERIMENTS.md row (commit + push with
``--git``). Extra ``key=value`` arguments are passed to every run as Hydra overrides.

Usage (detached, after the running jobs)::

    setsid nohup python -u scripts/tools/run_when_idle.py -- python -u scripts/tools/weight_series.py \\
        --task Mecanum-Navigation-Flat-Own-v0 --term final_position --values 20 30 40 50 --name_prefix own_flat_goalw \\
        --wandb --video --git > logs/queue/own_flat_goalw.log 2>&1 < /dev/null &
"""

import argparse
import os
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(__file__))
from tune_rewards import PYTHON, ROOT, log  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--task", required=True)
    parser.add_argument("--term", required=True, help="Reward term whose weight changes.")
    parser.add_argument("--values", type=float, nargs="+", required=True)
    parser.add_argument("--name_prefix", required=True)
    parser.add_argument("--iterations", type=int, default=400)
    parser.add_argument("--num_envs", type=int, default=3072)
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--video", action="store_true")
    parser.add_argument("--git", action="store_true")
    parser.add_argument("--note", default="")
    args, overrides = parser.parse_known_args()
    for value in args.values:
        name = f"{args.name_prefix}{value:g}"
        cmd = [PYTHON, "-u", "scripts/tools/train_eval_log.py", "--task", args.task, "--name", name,
               "--iterations", str(args.iterations), "--num_envs", str(args.num_envs),
               "--note", f"{args.term} = {value:g} (sorozat: {args.name_prefix}*, {', '.join(f'{v:g}' for v in args.values)}); {args.note}".rstrip("; "),
               *(["--wandb"] if args.wandb else []), *(["--video"] if args.video else []), *(["--git"] if args.git else []),
               f"env.rewards.{args.term}.weight={value}", *overrides]  # fmt: skip
        log(f"[{name}] start: {args.term} = {value:g}")
        os.makedirs(os.path.join(ROOT, "logs", "runs"), exist_ok=True)
        with open(os.path.join(ROOT, "logs", "runs", f"{name}.log"), "w") as f:
            code = subprocess.run(cmd, cwd=ROOT, stdout=f, stderr=subprocess.STDOUT).returncode
        log(f"[{name}] finished (exit {code})")
        time.sleep(30)  # let the GPU memory be released
    log("series finished")


if __name__ == "__main__":
    main()
