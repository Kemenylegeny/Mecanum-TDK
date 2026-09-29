# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""One training run, unattended: train -> evaluate -> motion diagnostics -> EXPERIMENTS.md row -> git commit + push.

Extra ``key=value`` arguments are Hydra overrides for train.py (``env.`` ones are also used for the evaluation and
the diagnostics, e.g. the episode length).

Usage (detached)::

    WANDB_USERNAME=<entity> setsid nohup python -u scripts/tools/train_eval_log.py --task Mecanum-Navigation-Pillars-v0 \\
        --name pillars7s_torque_actionrate --note "..." --wandb --video --git env.episode_length_s=7.0 ... \\
        > logs/runs/pillars7s_torque_actionrate.log 2>&1 < /dev/null &
"""

import argparse
import glob
import json
import os
import re
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(__file__))
from tune_rewards import PYTHON, ROOT, latest_checkpoint, log, push_to_wandb, read_progress, run, summarize  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--task", required=True)
    parser.add_argument("--name", required=True, help="Run name (log folder suffix, W&B run name).")
    parser.add_argument("--note", default="", help="What this run tests (goes into EXPERIMENTS.md).")
    parser.add_argument("--iterations", type=int, default=400)
    parser.add_argument("--resume_from", default=None, help="Checkpoint to continue from (same experiment folder).")
    parser.add_argument("--checkpoint", default=None, help="Skip training: evaluate this checkpoint (its run's train.log).")
    parser.add_argument("--num_envs", type=int, default=3072)
    parser.add_argument("--eval_envs", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--video", action="store_true")
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb_project", default="mecanum-navigation")
    parser.add_argument("--git", action="store_true", help="Commit + push the EXPERIMENTS.md row.")
    # used by tune_rewards.summarize / push_to_wandb
    parser.add_argument("--crash_weight", type=float, default=0.5)
    parser.add_argument("--time_weight", type=float, default=0.01)
    args, overrides = parser.parse_known_args()
    args.study = args.name
    return args, overrides


def main():
    args, overrides = parse_args()
    out_dir = os.path.join(ROOT, "logs", "runs", args.name)
    os.makedirs(out_dir, exist_ok=True)
    env = dict(os.environ)
    cmd = [PYTHON, "-u", "scripts/rsl_rl/train.py", "--task", args.task, "--headless", "--num_envs", str(args.num_envs),
           "--max_iterations", str(args.iterations), "--seed", str(args.seed), "--run_name", args.name]  # fmt: skip
    if args.video:
        cmd += ["--video", "--video_length", "500", "--video_interval", str(48 * 100)]
    if args.wandb:
        cmd += ["--logger", "wandb", "--log_project_name", args.wandb_project]
    final = f"model_{args.iterations - 1}.pt"
    if args.resume_from:
        src_dir, src_ckpt = os.path.split(os.path.abspath(args.resume_from))
        cmd += ["--resume", "--load_run", f"^{re.escape(os.path.basename(src_dir))}$", "--checkpoint", f"^{re.escape(src_ckpt)}$"]
        start_iter = int(re.search(r"model_(\d+)\.pt", src_ckpt).group(1))
        final = f"model_{start_iter + args.iterations - 1}.pt"  # RSL-RL continues at start_iter, saves the last one
    cmd += overrides
    started = time.time()

    def run_dir():
        dirs = [d for d in glob.glob(os.path.join(ROOT, "logs", "rsl_rl", "*", f"*_{args.name}")) if os.path.getmtime(d) >= started - 5]
        return max(dirs, key=os.path.getmtime) if dirs else None

    def finished():
        d = run_dir()
        return d is not None and os.path.exists(os.path.join(d, final))

    log(f"[{args.name}] training {args.task}, {args.iterations} iterations"
        f"{' from ' + os.path.relpath(args.resume_from, ROOT) if args.resume_from else ''}: {overrides}")
    train_log = os.path.join(out_dir, "train.log")
    if args.checkpoint:
        ckpt = os.path.abspath(args.checkpoint)
        d = os.path.dirname(ckpt)
    else:
        run(cmd, train_log, env, done=finished)
        if not finished():
            log(f"[{args.name}] training failed, see {train_log}")
            return 1
        d = run_dir()
        ckpt = latest_checkpoint(d)
    env_overrides = [o for o in overrides if o.startswith("env.")]

    # evaluation (deterministic policy, all terrain levels, goals >= 1 m away)
    eval_json = os.path.join(d, f"eval_{os.path.splitext(os.path.basename(ckpt))[0]}.json")
    log(f"[{args.name}] evaluating {os.path.relpath(ckpt, ROOT)}")
    run([PYTHON, "-u", "scripts/rsl_rl/evaluate_navigation.py", "--task", args.task, "--checkpoint", ckpt,
         "--num_envs", str(args.eval_envs), "env.commands.goal_pose.min_distance=1.0", *env_overrides],
        os.path.join(out_dir, "eval.log"), env, done=lambda: os.path.exists(eval_json))  # fmt: skip
    # motion diagnostics
    diag_dir = os.path.join(d, "analysis_" + os.path.splitext(os.path.basename(ckpt))[0])
    log(f"[{args.name}] motion diagnostics")
    run([PYTHON, "-u", "scripts/tools/motion_diagnostics.py", "--task", args.task, "--checkpoint", ckpt, *env_overrides],
        os.path.join(out_dir, "motion.log"), env, done=lambda: os.path.exists(os.path.join(diag_dir, "motion.png")))  # fmt: skip

    result = {}
    if os.path.exists(eval_json):
        result = summarize(eval_json, args)
        with open(eval_json) as f:
            result["motion"] = json.load(f).get("motion", {})
    diag = {}
    if os.path.exists(os.path.join(diag_dir, "motion.json")):
        with open(os.path.join(diag_dir, "motion.json")) as f:
            diag = json.load(f)
    progress = read_progress(train_log)
    level = progress[max(progress)].get("level") if progress else None
    with open(os.path.join(out_dir, "result.json"), "w") as f:
        json.dump({"checkpoint": ckpt, "overrides": overrides, "eval": result, "motion_diagnostics": diag, "final_level": level}, f, indent=2)

    url = None
    if args.wandb and result:
        push_to_wandb(args, os.path.basename(d), {**result, "name": args.name, "iterations": args.iterations, "early_stopped": None})
        try:
            import wandb

            api = wandb.Api()
            runs = list(api.runs(f"{os.environ.get('WANDB_USERNAME') or api.default_entity}/{args.wandb_project}",
                                 filters={"display_name": os.path.basename(d)}))  # fmt: skip
            url = runs[0].url if runs else None
        except Exception:  # noqa: BLE001
            pass

    # EXPERIMENTS.md
    path = os.path.join(ROOT, "EXPERIMENTS.md")
    header = "## Egyedi tanítások (`scripts/tools/train_eval_log.py`)"
    text = open(path).read()
    if header not in text:
        text = text.rstrip() + f"""

{header}

| Dátum | Run (wandb) | Feladat | Beállítás (Hydra override) | Siker | Ütközés | Siker (7–9. szint) | Odaérés [s] | Végső curriculum-szint | Mozgás | Megjegyzés |
|---|---|---|---|---|---|---|---|---|---|---|
"""
    link = f"[{args.name}]({url})" if url else args.name
    with open(train_log, errors="ignore") as f:
        overflow = f.read().count("buffer overflow")
    note = args.note + (f" **PhysX overflow: {overflow}×**" if overflow else "")
    if result:
        mv = (f"sebesség {diag.get('speed_while_moving_median', float('nan')):.2f} m/s, előre "
              f"{diag.get('motion_direction_share', {}).get('0-20 deg', float('nan')):.0%}, pörgés a célban "
              f"{diag.get('yaw_rate_at_goal_median') or 0:.2f} rad/s") if diag else "-"  # fmt: skip
        row = (f"| {time.strftime('%Y-%m-%d %H:%M')} | {link} | `{args.task}` | {' '.join(f'`{o}`' for o in overrides)} "
               f"| {result['success']:.1%} | {result['crashed']:.1%} | {result['success_hard']:.1%} | {result['median_arrival_s']:.1f} "
               f"| {level if level is not None else '-'} | {mv} | {note} |")  # fmt: skip
    else:
        row = f"| {time.strftime('%Y-%m-%d %H:%M')} | {link} | `{args.task}` | {' '.join(overrides)} | kiértékelés sikertelen | | | | | | {note} |"
    with open(path, "w") as f:
        f.write(text.rstrip("\n") + "\n" + row + "\n")
    log(f"[{args.name}] {row}")
    if args.git:
        msg = f"{time.strftime('%Y-%m-%d')} run {args.name}: results\n\n{row}\n\nCo-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
        for c in (["git", "add", "EXPERIMENTS.md"], ["git", "commit", "-q", "-m", msg, "--", "EXPERIMENTS.md"],
                  ["timeout", "120", "git", "push", "-q"]):  # fmt: skip
            out = subprocess.run(c, cwd=ROOT, capture_output=True, text=True)
            if out.returncode != 0:
                log(f"[git] {' '.join(c[:3])} failed: {out.stderr.strip()[:200]}")
                break
    return 0


if __name__ == "__main__":
    sys.exit(main())
