# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Add the motion penalties one at a time, every policy trained from scratch, with calibrated and tuned weights.

Stage 0 trains the flat navigation task (``Mecanum-Navigation-Flat-v0``) with only the positive rewards (task
reward + exploration bias) and checks the gate. Stage k then adds penalty k (STAGES order) on top of the weights
accepted so far and trains a new policy **from scratch** (no warm start).

Weight calibration: the evaluation of the previously accepted policy measures the per-step mean ``m`` of the
quantity the penalty acts on (e.g. sum of squared wheel torques). Since the reward manager scales every term by the
step time, that behavior would lose ``|w| * m * T`` per episode (T = episode length) to the penalty. The weight is
chosen so that this is the fraction ``f`` of the maximum task reward (``--task_max``, the task reward weight):

    w = -f * task_max / (m * T)

Tuning (at most ``--max_attempts`` trainings per stage, ``f`` starts at ``--share``):
  * success / crashes / arrival time got worse than the reference (previous accepted policy) -> weaker (``f`` down),
  * the penalty had no effect on the learned motion (targeted metric not reduced by ``--min_effect``) -> stronger,
  * once both a too weak and a too strong ``f`` are known, the next one is their geometric mean,
  * both fine -> accepted; the next stage starts from it. No attempt passes -> the study stops with a report.
A quantity that is already negligible for the reference (below the stage ``floor``) cannot be reduced further; the
effect check is then skipped and the stage is accepted on the success checks alone (noted in the report).

Every finished run is appended to ``EXPERIMENTS.md`` and, with ``--git``, committed and pushed (dated). Results,
report and logs: ``logs/penalty_study/<study>/``. An interrupted study continues where it stopped.

Usage (detached, survives an SSH logout)::

    WANDB_USERNAME=<entity> setsid nohup python -u scripts/tools/penalty_study.py --study flat7s --wandb --video --git \\
        > logs/penalty_study/flat7s_study.log 2>&1 < /dev/null &
"""

import argparse
import glob
import json
import math
import os
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(__file__))
from staged_penalties import EXPERIMENT_DIR, TASK, evaluate  # noqa: E402
from tune_rewards import PYTHON, ROOT, fmt, latest_checkpoint, log, push_to_wandb, run  # noqa: E402

# order of introduction (most cited first, see README / EXPERIMENTS.md); metric: evaluation quantity the penalty
# acts on (= per-step mean of the unweighted reward term); floor: below this there is nothing left to reduce
STAGES = [
    {"term": "wheel_torque_l2", "metric": "torque2", "floor": 1.0, "source": "Rudin eq. 2 joint torques; Xie E_e"},
    {"term": "action_rate_l2", "metric": "action_rate2", "floor": 1.0e-3, "source": "Rudin eq. 2; Finke 2026"},
    {"term": "wheel_acc_l2", "metric": "acc2", "floor": 10.0, "source": "Rudin eq. 2 joint accelerations; Xie E_k"},
    {"term": "stalling", "metric": "stall_frac", "floor": 0.02, "source": "Rudin eq. 4; Xie E_idle"},
    {"term": "wheel_slip_l2", "metric": "slip2", "floor": 1.0e-3, "source": "Xie E_f"},
    {"term": "action_l2", "metric": "action2", "floor": 1.0e-3, "source": "Finke 2026"},
]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--study", required=True, help="Name of the study (folder under logs/penalty_study).")
    parser.add_argument("--task", default=TASK, help="Navigation task (default: the FUJI robot on flat ground).")
    parser.add_argument("--experiment", default=os.path.basename(EXPERIMENT_DIR), help="RSL-RL experiment folder of --task.")
    parser.add_argument("--base", nargs="*", default=[],
                        help="key=value Hydra overrides of every training (env.* ones also for the evaluation).")
    parser.add_argument("--terms", nargs="*", default=None, help="Penalties to add, in this order (default: STAGES).")
    parser.add_argument("--floor", nargs="*", default=[], help="metric=value: robot-specific floors (see STAGES).")
    parser.add_argument("--iterations", type=int, default=400, help="PPO iterations per training (from scratch).")
    parser.add_argument("--num_envs", type=int, default=3072)
    parser.add_argument("--eval_envs", type=int, default=1024)
    parser.add_argument("--episode_s", type=float, default=7.0, help="Episode length of the flat task [s].")
    parser.add_argument("--task_max", type=float, default=10.0, help="Maximum task reward per episode (its weight).")
    parser.add_argument("--share", type=float, default=0.2, help="Initial penalty share f of the maximum task reward.")
    parser.add_argument("--min_effect", type=float, default=0.2, help="Targeted metric must drop by this fraction.")
    parser.add_argument("--max_attempts", type=int, default=4, help="Trainings per stage at most.")
    parser.add_argument("--gate_success", type=float, default=0.85, help="Stage 0 (no penalties) must reach this.")
    parser.add_argument("--max_success_drop", type=float, default=0.03)
    parser.add_argument("--max_crash_rise", type=float, default=0.03)
    parser.add_argument("--max_arrival_rise", type=float, default=1.0, help="[s] median arrival time.")
    parser.add_argument("--stages", type=int, default=len(STAGES), help="Run only the first N penalty stages.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--video", action="store_true", help="Record a 10 s video every 100 iterations.")
    parser.add_argument("--wandb", action="store_true", help="Log every run to W&B (group = study name).")
    parser.add_argument("--wandb_project", default="mecanum-navigation")
    parser.add_argument("--git", action="store_true", help="Commit + push EXPERIMENTS.md after every run.")
    # used by tune_rewards.summarize / push_to_wandb
    parser.add_argument("--crash_weight", type=float, default=0.5)
    parser.add_argument("--time_weight", type=float, default=0.01)
    args = parser.parse_args()
    args.base_overrides = {k: _value(v) for k, _, v in (item.partition("=") for item in args.base)}
    args.eval_overrides = {k: v for k, v in args.base_overrides.items() if k.startswith("env.")}
    args.experiment_dir = os.path.join(ROOT, "logs", "rsl_rl", args.experiment)
    return args


def _value(v: str):
    try:
        return float(v)
    except ValueError:
        return v


##
# Training / bookkeeping helpers
##


def train(args, study_dir: str, name: str, overrides: dict) -> str:
    """Train a policy from scratch with ``overrides``; returns its final checkpoint."""
    run_name = f"{args.study}_{name}"
    env = dict(os.environ)
    cmd = [
        PYTHON, "-u", "scripts/rsl_rl/train.py", "--task", args.task, "--headless",
        "--num_envs", str(args.num_envs), "--max_iterations", str(args.iterations),
        "--seed", str(args.seed), "--run_name", run_name, "agent.save_interval=50",
    ]  # fmt: skip
    if args.video:
        cmd += ["--video", "--video_length", "500", "--video_interval", str(48 * 100)]
    if args.wandb:
        cmd += ["--logger", "wandb", "--log_project_name", args.wandb_project]
        env["WANDB_RUN_GROUP"] = args.study
    cmd += [f"{k}={fmt(v)}" for k, v in overrides.items()]
    final = f"model_{args.iterations - 1}.pt"
    started = time.time()

    def run_dir():
        dirs = [d for d in glob.glob(os.path.join(args.experiment_dir, f"*_{run_name}")) if os.path.getmtime(d) >= started - 5]
        return max(dirs, key=os.path.getmtime) if dirs else None

    def finished():
        d = run_dir()
        return d is not None and os.path.exists(os.path.join(d, final))

    log(f"[{name}] training {args.iterations} iterations from scratch: {overrides or '(no penalties)'}")
    status = run(cmd, os.path.join(study_dir, f"{name}_train.log"), env, done=finished)
    if not finished():
        raise RuntimeError(f"training {name} failed ({status}), see {name}_train.log")
    return latest_checkpoint(run_dir())


def wandb_url(args, run_display_name: str) -> str | None:
    if not args.wandb:
        return None
    try:
        import wandb

        api = wandb.Api()
        entity = os.environ.get("WANDB_USERNAME") or api.default_entity
        runs = list(api.runs(f"{entity}/{args.wandb_project}", filters={"display_name": run_display_name}))
        return runs[0].url if runs else None
    except Exception:  # noqa: BLE001
        return None


def experiments_row(args, rec: dict):
    """Append the run to EXPERIMENTS.md (section of this study) and optionally commit + push it."""
    path = os.path.join(ROOT, "EXPERIMENTS.md")
    header = f"## Büntetések egyenként, nulláról tanítva, {args.episode_s:g} s epizód (study `{args.study}`)"
    with open(path) as f:
        text = f.read()
    if header not in text:
        text = text.rstrip() + f"""

{header}

`scripts/tools/penalty_study.py --study {args.study}` (`{args.task}`{', alap: ' + ' '.join(f'`{k}={fmt(v)}`' for k, v in args.base_overrides.items()) if args.base_overrides else ''}): minden policy nulláról, {args.iterations} iteráció, {args.num_envs} env.
Súly: `w = −f · {args.task_max:g} / (m_ref · {args.episode_s:g} s)`, ahol `m_ref` a büntetett mennyiség lépésenkénti
átlaga az előző elfogadott policynél (így a büntetés a fő jutalom `f`-szeresét vonná le). Elfogadás: a siker / ütközés /
odaérés nem romlik, és a célzott metrika legalább {args.min_effect:.0%}-kal csökken.

| Dátum | Run (wandb) | Lépés / próba | Új büntetés, súly (f) | Siker | Ütközés | Odaérés [s] | Célzott metrika: ref → új | Döntés |
|---|---|---|---|---|---|---|---|---|
"""
    run_link = f"[{rec['name']}]({rec['url']})" if rec.get("url") else rec["name"]
    if rec["stage"] == 0:
        new = "– (csak pozitív jutalom)"
        metric = "–"
    elif "error" in rec:
        new, metric = f"`{rec.get('term', '')}`", "–"
    else:
        new = f"`{rec['term']} = {rec['weight']:.3g}` (f = {rec['share']:.3g})"
        metric = f"{rec['metric']}: {rec['ref_metric']:.4g} → {rec['motion'].get(rec['metric'], float('nan')):.4g} ({-rec['reduction']:+.0%})"
    if "error" in rec:
        row = f"| {time.strftime('%Y-%m-%d %H:%M')} | {rec['name']} | {rec['stage']} / {rec.get('attempt', 0)} | {new} | | | | | hiba: {rec['error']} |"
    else:
        row = (f"| {time.strftime('%Y-%m-%d %H:%M')} | {run_link} | {rec['stage']} / {rec.get('attempt', 0)} | {new} "
               f"| {rec['success']:.1%} | {rec['crashed']:.1%} | {rec['median_arrival_s']:.1f} | {metric} | {rec['decision']} |")  # fmt: skip
    with open(path, "w") as f:
        f.write(text.rstrip("\n") + "\n" + row + "\n")
    if args.git:
        msg = (f"{time.strftime('%Y-%m-%d')} {args.study}: {rec['name']} -> {rec.get('decision', 'error')}\n\n"
               f"{row}\n\nCo-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>")  # fmt: skip
        for cmd in (["git", "add", "EXPERIMENTS.md"], ["git", "commit", "-q", "-m", msg, "--", "EXPERIMENTS.md"],
                    ["timeout", "120", "git", "push", "-q"]):  # fmt: skip
            out = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
            if out.returncode != 0:
                log(f"[git] {' '.join(cmd[:3])} failed: {out.stderr.strip()[:200]}")
                break


def passes(args, result: dict, ref: dict) -> tuple[bool, str]:
    checks = [
        (result["success"] >= ref["success"] - args.max_success_drop, f"success {result['success']:.1%} (ref {ref['success']:.1%})"),
        (result["crashed"] <= ref["crashed"] + args.max_crash_rise, f"crashed {result['crashed']:.1%} (ref {ref['crashed']:.1%})"),
        (result["median_arrival_s"] <= ref["median_arrival_s"] + args.max_arrival_rise,
         f"arrival {result['median_arrival_s']:.1f} s (ref {ref['median_arrival_s']:.1f} s)"),
    ]  # fmt: skip
    return all(ok for ok, _ in checks), "; ".join(("ok " if ok else "FAIL ") + text for ok, text in checks)


def write_report(study_dir: str, records: list[dict], status: str):
    lines = [f"# Penalty study: {os.path.basename(study_dir)}", "", f"**Status:** {status}", "",
             "| stage / attempt | term | weight | f | decision | success | crashed | arrival [s] | targeted metric (ref -> new) | checks |",
             "|---|---|---|---|---|---|---|---|---|---|"]  # fmt: skip
    for r in records:
        if "error" in r:
            lines.append(f"| {r['stage']} / {r.get('attempt', 0)} | {r.get('term', 'base')} | | | error | | | | | {r['error']} |")
        elif r["stage"] == 0:
            lines.append(f"| 0 | base | - | - | {r['decision']} | {r['success']:.1%} | {r['crashed']:.1%} | {r['median_arrival_s']:.1f} | - | |")
        else:
            m = r["motion"].get(r["metric"], float("nan"))
            lines.append(f"| {r['stage']} / {r['attempt']} | {r['term']} | {r['weight']:.3g} | {r['share']:.3g} | {r['decision']} "
                         f"| {r['success']:.1%} | {r['crashed']:.1%} | {r['median_arrival_s']:.1f} "
                         f"| {r['metric']}: {r['ref_metric']:.4g} -> {m:.4g} ({-r['reduction']:+.0%}) | {r['checks']} |")  # fmt: skip
    accepted = {r["term"]: r["weight"] for r in records if r.get("stage", 0) > 0 and r.get("decision", "").startswith("accepted")}
    lines += ["", "**Accepted weights:** " + (" ".join(f"env.rewards.{k}.weight={v:.3g}" for k, v in accepted.items()) or "none")]
    with open(os.path.join(study_dir, "report.md"), "w") as f:
        f.write("\n".join(lines) + "\n")


##
# Main
##


def main():
    args = parse_args()
    study_dir = os.path.join(ROOT, "logs", "penalty_study", args.study)
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

    def finish_record(rec: dict, name: str, ckpt: str):
        run_name = os.path.basename(os.path.dirname(ckpt))
        rec["url"] = wandb_url(args, run_name)
        if args.wandb:
            push_to_wandb(args, run_name, {**rec, "name": name, "iterations": args.iterations, "early_stopped": None})
        records.append(rec)
        experiments_row(args, rec)

    stages = STAGES if args.terms is None else [next(st for st in STAGES if st["term"] == t) for t in args.terms]
    stages = [dict(st) for st in stages[: args.stages]]
    floors = {m: float(v) for m, _, v in (item.partition("=") for item in args.floor)}
    for st in stages:
        st["floor"] = floors.get(st["metric"], st["floor"])
    log(f"study {args.study}: stage 0 (no penalties) + {len(stages)} penalty stages, {args.iterations} iterations each")

    # -- stage 0: positive rewards only
    base = next((r for r in records if r["stage"] == 0 and "error" not in r), None)
    if base is None:
        try:
            ckpt = train(args, study_dir, "stage0_base", dict(args.base_overrides))
            base = {"stage": 0, "name": "stage0_base", "checkpoint": ckpt, "overrides": dict(args.base_overrides),
                    **evaluate(args, study_dir, "stage0_base", ckpt)}  # fmt: skip
        except RuntimeError as e:
            records.append({"stage": 0, "name": "stage0_base", "error": str(e)})
            save(f"stopped: {e}")
            return
        base["decision"] = "accepted (gate)" if base["success"] >= args.gate_success else "gate FAILED"
        finish_record(base, "stage0_base", base["checkpoint"])
        save("stage 0 done")
    log(f"[stage0] success {base['success']:.1%}, arrival {base['median_arrival_s']:.1f} s, motion {json.dumps(base['motion'])}")
    if not base["decision"].startswith("accepted"):
        save(f"stopped: stage 0 below the gate ({base['success']:.1%} < {args.gate_success:.0%})")
        return

    ref = base
    for k, stage in enumerate(stages, start=1):
        term, metric, floor = stage["term"], stage["metric"], stage["floor"]
        ref_metric = ref["motion"].get(metric, 0.0)
        measurable = ref_metric >= floor
        f, lo, hi, accepted = args.share, None, None, None
        for attempt in range(args.max_attempts):
            weight = -f * args.task_max / (max(ref_metric, floor) * args.episode_s)
            weight = float(f"{weight:.3g}")
            name = f"stage{k}_{term}_a{attempt}"
            rec = next((r for r in records if r.get("name") == name and "error" not in r), None)
            if rec is None:
                overrides = {**ref["overrides"], f"env.rewards.{term}.weight": weight}
                try:
                    ckpt = train(args, study_dir, name, overrides)
                    result = evaluate(args, study_dir, name, ckpt)
                except RuntimeError as e:
                    err = {"stage": k, "attempt": attempt, "term": term, "name": name, "error": str(e)}
                    records.append(err)
                    experiments_row(args, err)
                    save(f"stopped: {e} (rerun the command to retry)")
                    log(f"[{name}] ERROR {e}; stopping")
                    return
                ok, checks = passes(args, result, ref)
                new_metric = result["motion"].get(metric, float("nan"))
                reduction = 1.0 - new_metric / ref_metric if ref_metric > 0 else 0.0
                effect = (not measurable) or reduction >= args.min_effect
                if ok and effect:
                    decision = "accepted" if measurable else "accepted (already negligible, effect not measurable)"
                elif not ok:
                    decision = "rejected: too strong (success / arrival worse) -> weaker"
                else:
                    decision = f"rejected: no effect ({-reduction:+.0%}) -> stronger"
                rec = {"stage": k, "attempt": attempt, "term": term, "name": name, "weight": weight, "share": f,
                       "overrides": overrides, "checkpoint": ckpt, "metric": metric, "ref_metric": ref_metric,
                       "reduction": reduction, "checks": checks, "decision": decision, **result}  # fmt: skip
                finish_record(rec, name, ckpt)
                save(f"stage {k} ({term}) in progress")
                log(f"[{name}] w={weight:.3g} (f={f:.3g}): {decision}; {checks}; {metric} {ref_metric:.4g} -> {new_metric:.4g}")
            if rec["decision"].startswith("accepted"):
                accepted = rec
                break
            if "too strong" in rec["decision"]:
                hi = f
                f = math.sqrt(lo * hi) if lo is not None else f / 2.0
            else:
                lo = f
                f = math.sqrt(lo * hi) if hi is not None else f * 2.0
        if accepted is None:
            save(f"stopped at stage {k}: no weight of {term} in {args.max_attempts} attempts (see the table)")
            log(f"stage {k}: no weight of {term} passes, stopping")
            return
        ref = accepted
        log(f"stage {k} done: {term} = {accepted['weight']:.3g}")

    save(f"finished all {len(stages)} penalty stages")
    log("all stages finished")


if __name__ == "__main__":
    sys.exit(main())
