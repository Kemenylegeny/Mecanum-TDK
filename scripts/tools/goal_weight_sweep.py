# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Raise the task reward (``final_position`` weight) until the policy heads for the goal on every curriculum level.

Pillar task with the curriculum, penalties fixed (``BASE_OVERRIDES``). Attempt 0 evaluates an existing checkpoint
trained with the current weight (``--baseline_checkpoint``, optional; without it attempt 1 trains the start weight);
every further attempt multiplies the weight by ``--factor`` and trains from scratch. Each policy is evaluated on all levels (``evaluate_navigation.py``): an episode
"started towards the goal" if the goal distance dropped by >= 0.5 m within the first 2 s. The sweep stops at the first
weight where this holds for at least ``--min_started`` of the episodes on every level.

Every weight change is logged in EXPERIMENTS.md and committed + pushed (``--git``): once when its training starts,
once with the result. Records: ``logs/goal_sweep/<study>/records.json`` (a restarted sweep continues from there).

Usage (detached)::

    WANDB_USERNAME=<entity> setsid nohup python -u scripts/tools/goal_weight_sweep.py --wandb --video --git \\
        --baseline_checkpoint logs/rsl_rl/mecanum_navigation_pillars/<run>/model_200.pt \\
        > logs/goal_sweep/pillars7s_goalw.log 2>&1 < /dev/null &
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

TASK = "Mecanum-Navigation-Pillars-v0"
EXPERIMENT_DIR = os.path.join(ROOT, "logs", "rsl_rl", "mecanum_navigation_pillars")
# accepted in the flat7s penalty study (EXPERIMENTS.md), 7 s episodes, no entropy bonus
BASE_OVERRIDES = {
    "env.episode_length_s": 7.0,
    "env.rewards.wheel_torque_l2.weight": -0.00312,
    "env.rewards.action_rate_l2.weight": -18.0,
    "agent.algorithm.entropy_coef": 0.0,
}
WEIGHT_KEY = "env.rewards.final_position.weight"
EVAL_OVERRIDES = {"env.commands.goal_pose.min_distance": 1.0}
TRAIN_METRICS = {
    "level": "Curriculum/terrain_levels",
    "action_std": "Mean action std",
    "task_fraction": "Metrics/exploration_bias/task_reward_fraction",
    "bias_active": "Metrics/exploration_bias/active",
}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--study", default="pillars7s_goalw")
    parser.add_argument("--start_weight", type=float, default=10.0, help="Weight of the baseline (attempt 0).")
    parser.add_argument("--factor", type=float, default=2.0, help="Weight multiplier per attempt.")
    parser.add_argument("--max_attempts", type=int, default=5, help="Trainings (weights 10, 20, ... 160 without a baseline checkpoint).")
    parser.add_argument("--min_started", type=float, default=0.8, help="Share of episodes per level that must start.")
    parser.add_argument("--baseline_checkpoint", default=None, help="Checkpoint trained with --start_weight.")
    parser.add_argument("--baseline_train_log", default=None, help="Its training log (for the training metrics).")
    parser.add_argument("--iterations", type=int, default=400)
    parser.add_argument("--num_envs", type=int, default=3072)
    parser.add_argument("--eval_envs", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--video", action="store_true")
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb_project", default="mecanum-navigation")
    parser.add_argument("--git", action="store_true", help="Commit + push EXPERIMENTS.md after every change.")
    # used by tune_rewards.summarize
    parser.add_argument("--crash_weight", type=float, default=0.5)
    parser.add_argument("--time_weight", type=float, default=0.01)
    return parser.parse_args()


def train_metrics(train_log: str | None) -> dict:
    """Last logged value of the ``TRAIN_METRICS`` in an RSL-RL console log."""
    if not train_log or not os.path.exists(train_log):
        return {}
    with open(train_log, errors="ignore") as f:
        blocks = re.split(r"Learning iteration (\d+)/\d+", f.read())
    if len(blocks) < 3:
        return {}
    out = {"iterations": int(blocks[-2]) + 1, "physx_overflow": sum(len(re.findall("buffer overflow", b)) for b in blocks)}
    for key, label in TRAIN_METRICS.items():
        m = re.search(re.escape(label) + r":\s*(-?[\d.]+(?:e-?\d+)?)", blocks[-1])
        if m:
            out[key] = float(m.group(1))
    return out


def evaluate(args, study_dir: str, name: str, checkpoint: str, weight: float) -> dict:
    eval_json = os.path.join(os.path.dirname(checkpoint), f"eval_{os.path.splitext(os.path.basename(checkpoint))[0]}.json")
    overrides = {**BASE_OVERRIDES, **EVAL_OVERRIDES, WEIGHT_KEY: weight}
    cmd = [PYTHON, "-u", "scripts/rsl_rl/evaluate_navigation.py", "--task", TASK, "--checkpoint", checkpoint,
           "--num_envs", str(args.eval_envs)] + [f"{k}={fmt(v)}" for k, v in overrides.items() if k.startswith("env.")]  # fmt: skip
    log(f"[{name}] evaluating {os.path.relpath(checkpoint, ROOT)}")
    if os.path.exists(eval_json):
        os.remove(eval_json)  # an older evaluation (without the start metric) must not count as done
    run(cmd, os.path.join(study_dir, f"{name}_eval.log"), dict(os.environ), done=lambda: os.path.exists(eval_json))
    if not os.path.exists(eval_json):
        raise RuntimeError(f"evaluation of {checkpoint} failed, see {name}_eval.log")
    result = summarize(eval_json, args)
    with open(eval_json) as f:
        episodes = json.load(f)["episodes"]
    per_level = {}
    for L in sorted({e["level"] for e in episodes}):
        eps = [e for e in episodes if e["level"] == L]
        per_level[L] = sum(e["early_progress"] >= 0.5 for e in eps) / len(eps)
    result["started_per_level"] = per_level
    result["success_per_level"] = {
        L: sum(e["success"] for e in episodes if e["level"] == L) / sum(e["level"] == L for e in episodes) for L in per_level
    }
    result["started_min"] = min(per_level.values())
    result["started_mean"] = sum(e["early_progress"] >= 0.5 for e in episodes) / len(episodes)
    return result


def train(args, study_dir: str, name: str, weight: float) -> tuple[str, str]:
    """Train from scratch; returns (final checkpoint, W&B / log run name)."""
    run_name = f"{args.study}_{name}"
    env = dict(os.environ)
    cmd = [PYTHON, "-u", "scripts/rsl_rl/train.py", "--task", TASK, "--headless", "--num_envs", str(args.num_envs),
           "--max_iterations", str(args.iterations), "--seed", str(args.seed), "--run_name", run_name,
           "agent.save_interval=50"]  # fmt: skip
    if args.video:
        cmd += ["--video", "--video_length", "500", "--video_interval", str(48 * 100)]
    if args.wandb:
        cmd += ["--logger", "wandb", "--log_project_name", args.wandb_project]
        env["WANDB_RUN_GROUP"] = args.study
    cmd += [f"{k}={fmt(v)}" for k, v in {**BASE_OVERRIDES, WEIGHT_KEY: weight}.items()]
    final = f"model_{args.iterations - 1}.pt"
    started = time.time()

    def run_dir():
        dirs = [d for d in glob.glob(os.path.join(EXPERIMENT_DIR, f"*_{run_name}")) if os.path.getmtime(d) >= started - 5]
        return max(dirs, key=os.path.getmtime) if dirs else None

    def finished():
        d = run_dir()
        return d is not None and os.path.exists(os.path.join(d, final))

    log(f"[{name}] training {args.iterations} iterations from scratch, {WEIGHT_KEY}={weight:g}")
    status = run(cmd, os.path.join(study_dir, f"{name}_train.log"), env, done=finished)
    if not finished():
        raise RuntimeError(f"training {name} failed ({status}), see {name}_train.log")
    return latest_checkpoint(run_dir()), os.path.basename(run_dir())


def wandb_url(args, display_name: str | None) -> str | None:
    if not args.wandb or not display_name:
        return None
    try:
        import wandb

        api = wandb.Api()
        entity = os.environ.get("WANDB_USERNAME") or api.default_entity
        runs = list(api.runs(f"{entity}/{args.wandb_project}", filters={"display_name": display_name}))
        return runs[0].url if runs else None
    except Exception:  # noqa: BLE001
        return None


def render_row(rec: dict) -> str:
    name = f"[{rec['name']}]({rec['url']})" if rec.get("url") else rec["name"]
    head = f"| {rec['date']} | {name} | {rec['weight']:g} |"
    if rec["status"] == "running":
        return head + " | | | | | | | | | tanítás fut |"
    if rec["status"] == "error":
        return head + f" | | | | | | | | | hiba: {rec['error']} |"
    t = rec.get("train", {})

    def per_level(key):
        if key not in rec:
            return "–"
        return " ".join(f"{int(L)}:{v:.0%}" for L, v in sorted(rec[key].items(), key=lambda x: int(x[0])))

    overflow = f" **PhysX overflow: {t['physx_overflow']}×**" if t.get("physx_overflow") else ""
    return (f"{head} {rec['success']:.1%} | {rec['crashed']:.1%} | {rec['started_min']:.0%} / {rec['started_mean']:.0%} "
            f"| {per_level('started_per_level')} | {per_level('success_per_level')} "
            f"| {t.get('level', float('nan')):.2f} | {t.get('action_std', float('nan')):.2f} "
            f"| {t.get('task_fraction', float('nan')):.2f} ({'be' if t.get('bias_active', 1) else 'ki'}) | {rec['decision']}{overflow} |")  # fmt: skip


def write_experiments(args, records: list[dict], message: str):
    """(Re)write this sweep's section of EXPERIMENTS.md from the records, then commit + push."""
    path = os.path.join(ROOT, "EXPERIMENTS.md")
    header = f"## Célreward (`final_position`) súlyának emelése, oszlopok + curriculum (study `{args.study}`)"
    baseline = (f"A 0. próba egy meglévő checkpoint (`{os.path.relpath(args.baseline_checkpoint, ROOT)}`, súly {args.start_weight:g}). "
                if args.baseline_checkpoint else f"Az 1. próba súlya {args.start_weight:g}. ")  # fmt: skip
    intro = f"""{header}

`scripts/tools/goal_weight_sweep.py --study {args.study}`: az oszlopos feladat curriculummal, a flat7s-ben elfogadott
büntetésekkel ({", ".join(f"`{k.split('.')[-2] if 'rewards' in k else k.split('.')[-1]} = {v:g}`" for k, v in BASE_OVERRIDES.items())}).
{baseline}A súly próbánként {args.factor:g}-szeresére nő, minden próba nulláról tanít ({args.iterations} iteráció, {args.num_envs} env). Egy epizód akkor
„indul el a cél felé”, ha az első 2 s alatt legalább 0.5 m-rel közelebb kerül a célhoz; a súly emelése leáll, ha ez
**minden** szinten az epizódok legalább {args.min_started:.0%}-ára teljesül. Kiértékelés: determinisztikus policy, mind
a 10 szint, célok ≥ 1 m-re. A tanítási oszlopok a tanítás utolsó iterációjából.

| Dátum | Run (wandb) | `final_position` súly | Siker | Ütközés | Elindul: legrosszabb szint / összes | Elindul szintenként | Siker szintenként | Curriculum-szint (tanítás) | Akció std | Task reward arány (bias) | Döntés |
|---|---|---|---|---|---|---|---|---|---|---|---|
"""
    section = intro + "\n".join(render_row(r) for r in records) + "\n"
    with open(path) as f:
        text = f.read()
    if header in text:
        start = text.index(header)
        nxt = text.find("\n## ", start + len(header))
        text = text[:start] + section + (text[nxt:] if nxt >= 0 else "")
    else:
        text = text.rstrip() + "\n\n" + section
    with open(path, "w") as f:
        f.write(text)
    if args.git:
        msg = f"{time.strftime('%Y-%m-%d')} {args.study}: {message}\n\nCo-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
        for cmd in (["git", "add", "EXPERIMENTS.md"], ["git", "commit", "-q", "-m", msg, "--", "EXPERIMENTS.md"],
                    ["timeout", "120", "git", "push", "-q"]):  # fmt: skip
            out = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
            if out.returncode != 0:
                log(f"[git] {' '.join(cmd[:3])} failed: {out.stderr.strip()[:200]}")
                break


def main():
    args = parse_args()
    study_dir = os.path.join(ROOT, "logs", "goal_sweep", args.study)
    os.makedirs(study_dir, exist_ok=True)
    rec_path = os.path.join(study_dir, "records.json")
    records = []
    if os.path.exists(rec_path):
        with open(rec_path) as f:
            records = [r for r in json.load(f) if r["status"] != "running"]

    def save(message: str):
        with open(rec_path, "w") as f:
            json.dump(records, f, indent=2)
        write_experiments(args, records, message)

    def decide(rec: dict) -> bool:
        ok = rec["started_min"] >= args.min_started
        worst = min(rec["started_per_level"], key=rec["started_per_level"].get)
        rec["decision"] = ("elindul minden szinten -> emelés leáll" if ok else
                           f"nem minden szinten (legrosszabb: {worst}. szint) -> emelés")  # fmt: skip
        return ok

    # attempt 0: the policy trained with the current weight
    if not any(r["attempt"] == 0 for r in records) and args.baseline_checkpoint:
        ckpt = os.path.abspath(args.baseline_checkpoint)
        rec = {"attempt": 0, "name": "goalw_baseline", "weight": args.start_weight, "date": time.strftime("%Y-%m-%d %H:%M"),
               "checkpoint": ckpt, "status": "done", "train": train_metrics(args.baseline_train_log)}  # fmt: skip
        rec["url"] = wandb_url(args, os.path.basename(os.path.dirname(ckpt)))
        rec.update(evaluate(args, study_dir, rec["name"], ckpt, args.start_weight))
        decide(rec)
        records.append(rec)
        save(f"baseline final_position.weight = {args.start_weight:g}: started min {rec['started_min']:.0%} -> {rec['decision']}")
        log(f"[baseline] {render_row(rec)}")

    for attempt in range(1, args.max_attempts + 1):
        if any(r.get("decision", "").startswith("elindul minden") for r in records if r["attempt"] > 0):
            break
        if any(r["attempt"] == attempt and r["status"] == "done" for r in records):
            continue
        # with a baseline checkpoint, attempt 0 is the start weight; otherwise attempt 1 trains it
        weight = args.start_weight * args.factor ** (attempt if args.baseline_checkpoint else attempt - 1)
        name = f"goalw_{weight:g}"
        rec = {"attempt": attempt, "name": name, "weight": weight, "date": time.strftime("%Y-%m-%d %H:%M"), "status": "running"}
        records.append(rec)
        save(f"final_position.weight = {weight:g} -> training started")
        try:
            ckpt, display = train(args, study_dir, name, weight)
            rec["checkpoint"] = ckpt
            rec["train"] = train_metrics(os.path.join(study_dir, f"{name}_train.log"))
            rec.update(evaluate(args, study_dir, name, ckpt, weight))
            rec["url"] = wandb_url(args, display)
            rec["status"] = "done"
            done = decide(rec)
            if args.wandb:
                push_to_wandb(args, display, {**rec, "iterations": args.iterations, "early_stopped": None})
        except Exception as e:  # noqa: BLE001
            rec.update(status="error", error=str(e)[:200])
            done = False
        rec["date"] = time.strftime("%Y-%m-%d %H:%M")
        save(f"final_position.weight = {weight:g} -> {rec.get('decision', 'error')}")
        log(f"[{name}] {render_row(rec)}")
        if rec["status"] == "error" or done:  # an error stays in the log; a restarted sweep retries that weight
            break
    log("sweep finished")


if __name__ == "__main__":
    main()
