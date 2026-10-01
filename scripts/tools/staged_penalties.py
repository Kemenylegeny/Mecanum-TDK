# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Re-introduce the motion penalties one at a time on top of the positive-reward-only flat navigation policy.

Starting point: a policy trained on ``Mecanum-Navigation-Flat-v0`` with only the task reward and the exploration
bias (the "base" run). The script

1. waits until the base training has finished, evaluates it and checks the gate (success >= ``--gate_success``),
2. for every stage in STAGES (in this order), switches on one more penalty on top of the weights accepted so far and
   fine-tunes the previously accepted policy (``--resume``) for ``--iterations`` iterations. The candidate weights are
   tried from the strongest to the weakest; the first one whose evaluation keeps
     success >= reference - ``--max_success_drop``, crashed <= reference + ``--max_crash_rise`` and
     median arrival time <= reference + ``--max_arrival_rise``
   is accepted (reference = the previously accepted policy) and becomes the start of the next stage,
3. stops with a report if no candidate of a stage passes ("this penalty breaks it"), or after the last stage.

Everything is written to ``logs/staged/<study>/`` (results.jsonl, report.md, logs of every run); an interrupted study
continues where it stopped (finished candidates are not re-run). Evaluations use the deterministic policy on the flat
task, the same for every candidate.

Usage (detached, survives an SSH logout)::

    WANDB_USERNAME=<entity> setsid nohup python -u scripts/tools/staged_penalties.py --study flat1 --wandb --video \\
        > logs/staged/flat1_study.log 2>&1 < /dev/null &
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
from tune_rewards import PYTHON, ROOT, fmt, latest_checkpoint, log, push_to_wandb, run, summarize  # noqa: E402

TASK = "Mecanum-Navigation-Flat-v0"
EXPERIMENT_DIR = os.path.join(ROOT, "logs", "rsl_rl", "mecanum_navigation_flat")

# order of re-introduction (most cited first, see README); weights from the strongest to the weakest candidate.
# metric: the evaluation quantity the penalty acts on (reported, lower = the penalty works)
STAGES = [
    # joint torques (Rudin et al. 2022, eq. 2) -> wheel motor torques; motor heat E_e (Xie et al. 2020)
    {"term": "wheel_torque_l2", "weights": [-8.0e-4, -2.0e-4, -5.0e-5], "metric": "torque2"},
    # abrupt action changes (Rudin eq. 2; Finke et al. 2026)
    {"term": "action_rate_l2", "weights": [-0.03, -0.01, -0.003], "metric": "action_rate2"},
    # joint accelerations (Rudin eq. 2) -> wheel accelerations; kinetic energy changes E_k (Xie)
    {"term": "wheel_acc_l2", "weights": [-4.0e-6, -1.0e-6, -2.5e-7], "metric": "acc2"},
    # stalling (Rudin eq. 4); ~ idle energy E_idle (Xie)
    {"term": "stalling", "weights": [-1.0, -0.5, -0.1], "metric": "stall_frac"},
    # wheel slip; friction dissipation E_f (Xie)
    {"term": "wheel_slip_l2", "weights": [-1.0, -0.5, -0.1], "metric": "slip2"},
    # action magnitude (Finke et al. 2026)
    {"term": "action_l2", "weights": [-0.2, -0.05, -0.01], "metric": "action2"},
]

# the same goal distribution for every evaluation (no trivially close goals)
EVAL_OVERRIDES = {"env.commands.goal_pose.min_distance": 1.0}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--study", required=True, help="Name of the study (folder under logs/staged).")
    parser.add_argument("--base_run", default=None, help="Run folder of the base policy (default: latest *flat_task_and_bias).")
    parser.add_argument("--base_checkpoint", default="model_499.pt")
    parser.add_argument("--iterations", type=int, default=150, help="Fine-tuning iterations per candidate.")
    parser.add_argument("--num_envs", type=int, default=3072)
    parser.add_argument("--eval_envs", type=int, default=1024)
    parser.add_argument("--gate_success", type=float, default=0.85, help="The base policy must reach this success.")
    parser.add_argument("--max_success_drop", type=float, default=0.03)
    parser.add_argument("--max_crash_rise", type=float, default=0.03)
    parser.add_argument("--max_arrival_rise", type=float, default=1.0, help="[s] median arrival time.")
    parser.add_argument("--stages", type=int, default=len(STAGES), help="Run only the first N stages.")
    parser.add_argument("--max_candidates", type=int, default=3, help="Try at most N weights per stage.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no_wait", action="store_true", help="Do not wait for a running training (tests only).")
    parser.add_argument("--video", action="store_true", help="Record a 10 s video every 100 iterations.")
    parser.add_argument("--wandb", action="store_true", help="Log every run to W&B (group = study name).")
    parser.add_argument("--wandb_project", default="mecanum-navigation")
    # used by tune_rewards.summarize / push_to_wandb
    parser.add_argument("--crash_weight", type=float, default=0.5)
    parser.add_argument("--time_weight", type=float, default=0.01)
    return parser.parse_args()


def training_running() -> bool:
    out = subprocess.run(["pgrep", "-f", "scripts/rsl_rl/train.py"], capture_output=True, text=True).stdout.split()
    return any(pid != str(os.getpid()) for pid in out)


def evaluate(args, study_dir: str, name: str, checkpoint: str) -> dict:
    """Deterministic evaluation of a checkpoint on the flat task; returns the summary incl. motion metrics."""
    eval_json = os.path.join(os.path.dirname(checkpoint), f"eval_{os.path.splitext(os.path.basename(checkpoint))[0]}.json")
    if not os.path.exists(eval_json):
        cmd = [
            PYTHON, "-u", "scripts/rsl_rl/evaluate_navigation.py", "--task", getattr(args, "task", TASK),
            "--checkpoint", checkpoint, "--num_envs", str(args.eval_envs),
        ] + [f"{k}={fmt(v)}" for k, v in {**getattr(args, "eval_overrides", {}), **EVAL_OVERRIDES}.items()]  # fmt: skip
        log(f"[{name}] evaluating {os.path.relpath(checkpoint, ROOT)}")
        run(cmd, os.path.join(study_dir, f"{name}_eval.log"), dict(os.environ), done=lambda: os.path.exists(eval_json))
    if not os.path.exists(eval_json):
        raise RuntimeError(f"evaluation of {checkpoint} failed, see {name}_eval.log")
    result = summarize(eval_json, args)
    with open(eval_json) as f:
        result["motion"] = json.load(f).get("motion", {})
    return result


def train(args, study_dir: str, name: str, overrides: dict, resume_from: str) -> tuple[str, int]:
    """Fine-tune ``resume_from`` (checkpoint path) with ``overrides``; returns (final checkpoint, iterations run)."""
    run_name = f"{args.study}_{name}"
    env = dict(os.environ)
    src_dir, src_ckpt = os.path.split(resume_from)
    cmd = [
        PYTHON, "-u", "scripts/rsl_rl/train.py", "--task", TASK, "--headless",
        "--num_envs", str(args.num_envs), "--max_iterations", str(args.iterations),
        "--seed", str(args.seed), "--run_name", run_name,
        "--resume", "--load_run", f"^{re.escape(os.path.basename(src_dir))}$",
        "--checkpoint", f"^{re.escape(src_ckpt)}$",
        "agent.save_interval=50",
    ]  # fmt: skip
    if args.video:
        cmd += ["--video", "--video_length", "500", "--video_interval", str(48 * 100)]
    if args.wandb:
        cmd += ["--logger", "wandb", "--log_project_name", args.wandb_project]
        env["WANDB_RUN_GROUP"] = args.study
    cmd += [f"{k}={fmt(v)}" for k, v in overrides.items()]

    start_iter = int(re.search(r"model_(\d+)\.pt", src_ckpt).group(1))
    final_ckpt = f"model_{start_iter + args.iterations}.pt"
    started = time.time()

    def run_dir():
        dirs = [d for d in glob.glob(os.path.join(EXPERIMENT_DIR, f"*_{run_name}")) if os.path.getmtime(d) >= started - 5]
        return max(dirs, key=os.path.getmtime) if dirs else None

    def finished():
        d = run_dir()
        return d is not None and (
            os.path.exists(os.path.join(d, final_ckpt)) or os.path.exists(os.path.join(d, f"model_{start_iter + args.iterations - 1}.pt"))
        )

    log(f"[{name}] fine-tuning {args.iterations} iterations from {os.path.relpath(resume_from, ROOT)}: {overrides}")
    status = run(cmd, os.path.join(study_dir, f"{name}_train.log"), env, done=finished)
    d = run_dir()
    if d is None or not finished():
        raise RuntimeError(f"training {name} failed ({status}), see {name}_train.log")
    return latest_checkpoint(d), args.iterations


def passes(args, result: dict, ref: dict) -> tuple[bool, str]:
    checks = [
        (result["success"] >= ref["success"] - args.max_success_drop,
         f"success {result['success']:.1%} (ref {ref['success']:.1%}, min {ref['success'] - args.max_success_drop:.1%})"),
        (result["crashed"] <= ref["crashed"] + args.max_crash_rise,
         f"crashed {result['crashed']:.1%} (ref {ref['crashed']:.1%})"),
        (result["median_arrival_s"] <= ref["median_arrival_s"] + args.max_arrival_rise,
         f"arrival {result['median_arrival_s']:.1f} s (ref {ref['median_arrival_s']:.1f} s)"),
    ]  # fmt: skip
    return all(ok for ok, _ in checks), "; ".join(("ok " if ok else "FAIL ") + text for ok, text in checks)


def write_report(study_dir: str, records: list[dict], status: str):
    lines = [f"# Staged penalties: {os.path.basename(study_dir)}", "", f"**Status:** {status}", "",
             "| stage | term | weight | accepted | success | crashed | median arrival [s] | mean speed [m/s] "
             "| targeted metric (ref -> new) | spin at goal [rad/s] | checks |",
             "|---|---|---|---|---|---|---|---|---|---|---|"]  # fmt: skip
    for r in records:
        if r.get("stage") == 0:
            m = r["motion"]
            lines.append(f"| 0 | base (task + exploration bias) | - | gate {'passed' if r['accepted'] else 'FAILED'} "
                         f"| {r['success']:.1%} | {r['crashed']:.1%} | {r['median_arrival_s']:.1f} | {r['mean_speed']:.2f} "
                         f"| - | {m.get('spin_at_goal', float('nan')):.2f} | |")  # fmt: skip
            continue
        if "error" in r:
            lines.append(f"| {r['stage']} | {r['term']} | {r['weight']:g} | error | | | | | | | {r['error']} |")
            continue
        m, key = r["motion"], r["metric"]
        lines.append(
            f"| {r['stage']} | {r['term']} | {r['weight']:g} | {'**yes**' if r['accepted'] else 'no'} | {r['success']:.1%} "
            f"| {r['crashed']:.1%} | {r['median_arrival_s']:.1f} | {r['mean_speed']:.2f} "
            f"| {key}: {r['ref_metric']:.4g} -> {m.get(key, float('nan')):.4g} | {m.get('spin_at_goal', float('nan')):.2f} "
            f"| {r['checks']} |"
        )
    accepted = {r["term"]: r["weight"] for r in records if r.get("stage", 0) > 0 and r.get("accepted")}
    lines += ["", "**Accepted weights (Hydra overrides for the final config):** "
              + (" ".join(f"env.rewards.{k}.weight={v:g}" for k, v in accepted.items()) or "none")]  # fmt: skip
    with open(os.path.join(study_dir, "report.md"), "w") as f:
        f.write("\n".join(lines) + "\n")


def main():
    args = parse_args()
    study_dir = os.path.join(ROOT, "logs", "staged", args.study)
    os.makedirs(study_dir, exist_ok=True)
    results_path = os.path.join(study_dir, "results.jsonl")
    records = []
    if os.path.exists(results_path):
        with open(results_path) as f:
            records = [json.loads(line) for line in f if line.strip()]

    def save(status: str):
        with open(results_path, "w") as f:
            f.writelines(json.dumps(r) + "\n" for r in records)
        write_report(study_dir, records, status)

    # -- base policy: wait for its training, evaluate, gate
    base_dir = args.base_run
    if base_dir is None:
        cands = sorted(glob.glob(os.path.join(EXPERIMENT_DIR, "*flat_task_and_bias")))
        if not cands:
            raise SystemExit("no base run (*flat_task_and_bias) found; pass --base_run")
        base_dir = cands[-1]
    base_dir = os.path.join(EXPERIMENT_DIR, base_dir) if not os.path.isabs(base_dir) else base_dir
    base_ckpt = os.path.join(base_dir, args.base_checkpoint)
    log(f"study {args.study}: base {os.path.relpath(base_ckpt, ROOT)}, {min(args.stages, len(STAGES))} stages")
    while not os.path.exists(base_ckpt) or (training_running() and not args.no_wait):
        log("waiting for the base training to finish ...")
        time.sleep(300)

    base = next((r for r in records if r.get("stage") == 0), None)
    if base is None:
        base = {"stage": 0, "checkpoint": base_ckpt, **evaluate(args, study_dir, "stage0_base", base_ckpt)}
        base["accepted"] = base["success"] >= args.gate_success
        records.append(base)
        save("base evaluated")
    log(f"[base] success {base['success']:.1%}, crashed {base['crashed']:.1%}, arrival {base['median_arrival_s']:.1f} s, "
        f"spin at goal {base['motion'].get('spin_at_goal', float('nan')):.2f} rad/s")  # fmt: skip
    if not base["accepted"]:
        save(f"stopped: base policy below the gate ({base['success']:.1%} < {args.gate_success:.0%})")
        log("base policy below the gate, stopping")
        return

    # -- stages
    ref, overrides = base, {}
    for k, stage in enumerate(STAGES[: args.stages], start=1):
        term = stage["term"]
        accepted = next((r for r in records if r.get("stage") == k and r.get("accepted")), None)
        if accepted is None:
            for w in stage["weights"][: args.max_candidates]:
                name = f"stage{k}_{term}_{abs(w):g}"
                done = next((r for r in records if r.get("stage") == k and r.get("weight") == w and "error" not in r), None)
                if done is None:
                    cand_overrides = {**overrides, f"env.rewards.{term}.weight": w}
                    try:
                        ckpt, its = train(args, study_dir, name, cand_overrides, ref["checkpoint"])
                        result = evaluate(args, study_dir, name, ckpt)
                    except RuntimeError as e:
                        records.append({"stage": k, "term": term, "weight": w, "error": str(e)})
                        save(f"stopped: {e}")
                        log(f"[{name}] ERROR {e}; stopping (not a rejection of the weight - rerun to continue)")
                        return
                    ok, checks = passes(args, result, ref)
                    done = {"stage": k, "term": term, "weight": w, "overrides": cand_overrides, "checkpoint": ckpt,
                            "iterations": its, "early_stopped": None, "metric": stage["metric"],
                            "ref_metric": ref["motion"].get(stage["metric"], float("nan")),
                            "accepted": ok, "checks": checks, **result}  # fmt: skip
                    records.append(done)
                    save(f"stage {k} ({term}) in progress")
                    log(f"[{name}] {'ACCEPTED' if ok else 'rejected'}: {checks}; "
                        f"{stage['metric']} {done['ref_metric']:.4g} -> {result['motion'].get(stage['metric'], float('nan')):.4g}")  # fmt: skip
                    if args.wandb:
                        push_to_wandb(args, os.path.basename(os.path.dirname(ckpt)), {**done, "name": name})
                if done["accepted"]:
                    accepted = done
                    break
        if accepted is None:
            save(f"stopped at stage {k}: no weight of {term} keeps the success / arrival time (see the table)")
            log(f"stage {k}: no weight of {term} passes, stopping")
            return
        ref, overrides = accepted, accepted["overrides"]
        log(f"stage {k} done: {term} = {accepted['weight']:g}")

    save(f"finished all {min(args.stages, len(STAGES))} stages")
    log("all stages finished")


if __name__ == "__main__":
    sys.exit(main())
