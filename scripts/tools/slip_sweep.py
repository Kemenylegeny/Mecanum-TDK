# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tune the wheel slip penalty on the pillar task (9 s episodes): train from scratch, evaluate, adjust the weight.

The pillar task defaults carry the setup (9 s, final_position 10, torque -0.00312, action rate -18, entropy 0); only
``env.rewards.wheel_slip_l2.weight`` changes. The weight is calibrated like the other penalties,
``w = -f * 10 / (m_ref * 9 s)`` with ``m_ref`` the per-step slip2 of the latest policy without slip penalty.

Reference: the pillar policy without slip penalty (``--ref_checkpoint``, 7 s). A slip policy is accepted if, compared
with it, the success of no level group (0-3, 4-6, 7-9) drops by more than ``--max_group_drop``, the crash rate rises by
at most ``--max_crash_rise`` and slip2 drops by at least ``--min_effect``. Too strong (success / crashes worse): ``f``
is halved; no effect: doubled; once bracketed, the geometric mean. Stops at the first accepted weight or after
``--max_attempts`` trainings.

Attempt 1 can be an already running training (``--first_run_dir``, e.g. the from-scratch 800-iteration run with
f = 0.2): its checkpoint at ``--iterations`` is evaluated as soon as it exists, and the next training starts once
that run's process has finished (the GPU fits one training).

All policies are evaluated the same way (deterministic, all 10 levels, goals >= 1 m away, 256 envs x 8 episodes,
seed 1) plus ``motion_diagnostics.py``. Every weight is logged in EXPERIMENTS.md and committed + pushed (``--git``)
when its training starts and with its result. Records: ``logs/slip_sweep/<study>/records.json`` (restartable).
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
from goal_weight_sweep import train_metrics, wandb_url  # noqa: E402
from tune_rewards import PYTHON, ROOT, latest_checkpoint, log, push_to_wandb, run, summarize  # noqa: E402

TASK = "Mecanum-Navigation-Pillars-v0"
EXPERIMENT_DIR = os.path.join(ROOT, "logs", "rsl_rl", "mecanum_navigation_pillars")
TERM = "env.rewards.wheel_slip_l2.weight"
GROUPS = {"0-3": range(0, 4), "4-6": range(4, 7), "7-9": range(7, 10)}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--study", default="pillars9s_slipw")
    parser.add_argument("--ref_checkpoint", required=True, help="Pillar policy without slip penalty.")
    parser.add_argument("--ref_episode_s", type=float, default=7.0, help="Episode length the reference was trained with.")
    parser.add_argument("--m_ref", type=float, default=0.0160, help="Per-step slip2 used for the calibration.")
    parser.add_argument("--episode_s", type=float, default=9.0)
    parser.add_argument("--task_max", type=float, default=10.0)
    parser.add_argument("--share", type=float, default=0.2, help="Initial f.")
    parser.add_argument("--first_run_dir", default=None, help="Running training with f = --share (attempt 1).")
    parser.add_argument("--min_effect", type=float, default=0.2)
    parser.add_argument("--max_group_drop", type=float, default=0.05)
    parser.add_argument("--max_crash_rise", type=float, default=0.03)
    parser.add_argument("--max_attempts", type=int, default=4)
    parser.add_argument("--iterations", type=int, default=400)
    parser.add_argument("--num_envs", type=int, default=3072)
    parser.add_argument("--eval_envs", type=int, default=256)
    parser.add_argument("--eval_episodes", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--video", action="store_true")
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb_project", default="mecanum-navigation")
    parser.add_argument("--git", action="store_true")
    # used by tune_rewards.summarize
    parser.add_argument("--crash_weight", type=float, default=0.5)
    parser.add_argument("--time_weight", type=float, default=0.01)
    return parser.parse_args()


def weight_of(args, f: float) -> float:
    return -f * args.task_max / (args.m_ref * args.episode_s)


def busy(pattern: str) -> bool:
    return subprocess.run(["pgrep", "-f", pattern], capture_output=True).returncode == 0


def evaluate(args, study_dir: str, name: str, checkpoint: str, episode_s: float) -> dict:
    """Standard evaluation + motion diagnostics of ``checkpoint`` (linked into the study folder, results next to it)."""
    ev_dir = os.path.join(study_dir, "eval", name)
    os.makedirs(ev_dir, exist_ok=True)
    link = os.path.join(ev_dir, os.path.basename(checkpoint))
    if not os.path.exists(link):
        os.symlink(os.path.abspath(checkpoint), link)
    stem = os.path.splitext(os.path.basename(checkpoint))[0]
    eval_json = os.path.join(ev_dir, f"eval_{stem}.json")
    overrides = [f"env.episode_length_s={episode_s}", "env.commands.goal_pose.min_distance=1.0"]
    while busy("scripts/rsl_rl/evaluate_navigation.py"):  # one evaluation at a time next to a training
        time.sleep(30)
    if not os.path.exists(eval_json):
        log(f"[{name}] evaluating {os.path.relpath(checkpoint, ROOT)}")
        run([PYTHON, "-u", "scripts/rsl_rl/evaluate_navigation.py", "--task", TASK, "--checkpoint", link,
             "--num_envs", str(args.eval_envs), "--episodes", str(args.eval_episodes), "--seed", "1", *overrides],
            os.path.join(ev_dir, "eval.log"), dict(os.environ), done=lambda: os.path.exists(eval_json))  # fmt: skip
    if not os.path.exists(eval_json):
        raise RuntimeError(f"evaluation of {checkpoint} failed, see {ev_dir}/eval.log")
    diag = os.path.join(ev_dir, "motion.json")
    if not os.path.exists(diag):
        run([PYTHON, "-u", "scripts/tools/motion_diagnostics.py", "--task", TASK, "--checkpoint", link, "--out_dir", ev_dir,
             f"env.episode_length_s={episode_s}"], os.path.join(ev_dir, "motion.log"), dict(os.environ),
            done=lambda: os.path.exists(os.path.join(ev_dir, "motion.png")))  # fmt: skip
    result = summarize(eval_json, args)
    with open(eval_json) as f:
        data = json.load(f)
    result["slip2"] = data["motion"]["slip2"]
    result["spin_at_goal"] = data["motion"]["spin_at_goal"]
    eps = data["episodes"]
    result["groups"] = {
        g: sum(e["success"] for e in eps if e["level"] in lv) / max(sum(e["level"] in lv for e in eps), 1)
        for g, lv in GROUPS.items()
    }
    if os.path.exists(diag):
        with open(diag) as f:
            m = json.load(f)
        result["yaw_moving"] = m["yaw_rate_while_moving_median"]
        result["forward_share"] = m["motion_direction_share"]["0-20 deg"]
        result["speed_moving"] = m["speed_while_moving_median"]
    return result


def train(args, study_dir: str, name: str, weight: float) -> tuple[str, str]:
    run_name = f"{args.study}_{name}"
    env = dict(os.environ)
    cmd = [PYTHON, "-u", "scripts/rsl_rl/train.py", "--task", TASK, "--headless", "--num_envs", str(args.num_envs),
           "--max_iterations", str(args.iterations), "--seed", str(args.seed), "--run_name", run_name,
           f"{TERM}={weight:.4g}"]  # fmt: skip
    if args.video:
        cmd += ["--video", "--video_length", "500", "--video_interval", str(48 * 100)]
    if args.wandb:
        cmd += ["--logger", "wandb", "--log_project_name", args.wandb_project]
        env["WANDB_RUN_GROUP"] = args.study
    final = f"model_{args.iterations - 1}.pt"
    started = time.time()

    def run_dir():
        dirs = [d for d in glob.glob(os.path.join(EXPERIMENT_DIR, f"*_{run_name}")) if os.path.getmtime(d) >= started - 5]
        return max(dirs, key=os.path.getmtime) if dirs else None

    def finished():
        d = run_dir()
        return d is not None and os.path.exists(os.path.join(d, final))

    log(f"[{name}] training {args.iterations} iterations from scratch, {TERM}={weight:.4g}")
    status = run(cmd, os.path.join(study_dir, f"{name}_train.log"), env, done=finished)
    if not finished():
        raise RuntimeError(f"training {name} failed ({status}), see {name}_train.log")
    return os.path.join(run_dir(), final), os.path.basename(run_dir())


def decide(args, rec: dict, ref: dict) -> str:
    """'ok' / 'strong' / 'weak' + the reasons."""
    fails = [g for g in GROUPS if rec["groups"][g] < ref["groups"][g] - args.max_group_drop]
    crash_bad = rec["crashed"] > ref["crashed"] + args.max_crash_rise
    effect = 1.0 - rec["slip2"] / ref["slip2"]
    rec["effect"] = effect
    reasons = [f"siker romlik: {', '.join(fails)}. szint"] if fails else []
    if crash_bad:
        reasons.append(f"ütközés {rec['crashed']:.1%} (ref {ref['crashed']:.1%})")
    if fails or crash_bad:
        rec["verdict"], rec["decision"] = "strong", "túl erős (" + "; ".join(reasons) + ") -> gyengébb"
    elif effect < args.min_effect:
        rec["verdict"], rec["decision"] = "weak", f"nincs elég hatás (slip2 {-effect:+.0%}) -> erősebb"
    else:
        rec["verdict"], rec["decision"] = "ok", "elfogadva"
    return rec["verdict"]


def render_row(rec: dict, ref: dict) -> str:
    name = f"[{rec['name']}]({rec['url']})" if rec.get("url") else rec["name"]
    head = f"| {rec['date']} | {name} | {rec['weight']:.3g} (f = {rec['share']:.3g}) |"
    if rec.get("status") == "running":
        return head + " | | | | | | | tanítás fut |"
    if rec.get("status") == "error":
        return head + f" | | | | | | | hiba: {rec['error']} |"
    g = " / ".join(f"{rec['groups'][k]:.0%}" for k in GROUPS)
    slip = f"{rec['slip2']:.4f}" + (f" ({rec['slip2'] / ref['slip2'] - 1:+.0%})" if rec.get("name") != ref.get("name") else "")
    motion = (f"{rec.get('spin_at_goal', float('nan')):.2f} / {rec.get('yaw_moving', float('nan')):.2f} rad/s, előre "
              f"{rec.get('forward_share', float('nan')):.0%}, {rec.get('speed_moving', float('nan')):.2f} m/s")  # fmt: skip
    overflow = f" **PhysX overflow: {rec['train']['physx_overflow']}×**" if rec.get("train", {}).get("physx_overflow") else ""
    return (f"{head} {rec['success']:.1%} | {g} | {rec['crashed']:.1%} | {rec['median_arrival_s']:.1f} | {slip} "
            f"| {motion} | {rec.get('decision', '')}{overflow} |")  # fmt: skip


def write_experiments(args, ref: dict, records: list[dict], message: str):
    path = os.path.join(ROOT, "EXPERIMENTS.md")
    header = f"## Slip-büntetés súlya, oszlopok + curriculum, {args.episode_s:g} s (study `{args.study}`)"
    section = f"""{header}

`scripts/tools/slip_sweep.py --study {args.study}`: az oszlopos feladat alapbeállításával ({args.episode_s:g} s,
`final_position` 10, nyomaték −0.00312, akcióváltás −18, entropy 0), csak a `wheel_slip_l2` súlya változik; minden
próba nulláról, {args.iterations} iteráció. Súly: `w = −f · {args.task_max:g} / (m_ref · {args.episode_s:g} s)`,
`m_ref` = {args.m_ref:g} m²/s² (slip2 lépésenként a legutóbbi slip nélküli policynél). Referencia: slip nélküli
pillaros policy ({args.ref_episode_s:g} s). Elfogadás: egyik szintcsoport sikere sem romlik
{args.max_group_drop:.0%}-pontnál többet, az ütközés legfeljebb {args.max_crash_rise:.0%}-ponttal nő, a slip2 legalább
{args.min_effect:.0%}-kal csökken. Kiértékelés: determinisztikus, mind a 10 szint, célok ≥ 1 m, 256 env × 8 epizód,
seed 1. Mozgás (`motion_diagnostics.py`): forgás a célnál / menet közben, előre haladás aránya, sebesség.

| Dátum | Run (wandb) | Slip súly | Siker | Siker 0–3 / 4–6 / 7–9 | Ütközés | Odaérés [s] | slip2 (ref-hez) | Mozgás | Döntés |
|---|---|---|---|---|---|---|---|---|---|
{render_row({**ref, "weight": 0.0, "share": 0.0, "decision": "referencia (slip nélkül)"}, ref)}
""" + "\n".join(render_row(r, ref) for r in records) + "\n"
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
    study_dir = os.path.join(ROOT, "logs", "slip_sweep", args.study)
    os.makedirs(study_dir, exist_ok=True)
    rec_path = os.path.join(study_dir, "records.json")
    state = {"ref": None, "records": []}
    if os.path.exists(rec_path):
        with open(rec_path) as f:
            state = json.load(f)
        state["records"] = [r for r in state["records"] if r.get("status") != "running"]
    records = state["records"]

    def save(message: str):
        with open(rec_path, "w") as f:
            json.dump(state, f, indent=2)
        write_experiments(args, state["ref"], records, message)

    if state["ref"] is None:
        ref = evaluate(args, study_dir, "reference", args.ref_checkpoint, args.ref_episode_s)
        state["ref"] = {**ref, "name": "reference_no_slip", "date": time.strftime("%Y-%m-%d %H:%M"),
                        "checkpoint": os.path.abspath(args.ref_checkpoint)}  # fmt: skip
        state["ref"]["url"] = wandb_url(args, os.path.basename(os.path.dirname(os.path.abspath(args.ref_checkpoint))))
        save("reference (no slip penalty) evaluated")
    ref = state["ref"]
    log(f"[reference] {render_row({**ref, 'weight': 0.0, 'share': 0.0}, ref)}")

    f, lo, hi = args.share, None, None  # lo: largest too weak f, hi: smallest too strong f
    for r in records:
        if r.get("status") == "done":
            lo, hi = (max(lo or 0, r["share"]), hi) if r["verdict"] == "weak" else (lo, min(hi or 1e9, r["share"]))
    for attempt in range(1, args.max_attempts + 1):
        if any(r.get("verdict") == "ok" for r in records):
            break
        if any(r["attempt"] == attempt and r.get("status") == "done" for r in records):
            continue
        if attempt > 1 or not args.first_run_dir:
            if lo is not None and hi is not None:
                f = math.sqrt(lo * hi)
            elif hi is not None:
                f = hi / 2.0
            elif lo is not None:
                f = lo * 2.0
        weight = weight_of(args, f)
        name = f"slip_f{f:.3g}"
        rec = {"attempt": attempt, "name": name, "weight": weight, "share": f, "status": "running",
               "date": time.strftime("%Y-%m-%d %H:%M")}  # fmt: skip
        records.append(rec)
        try:
            if attempt == 1 and args.first_run_dir:
                d = os.path.abspath(args.first_run_dir)
                rec["name"] = os.path.basename(d).split("_", 2)[-1] + f"@{args.iterations}"
                save(f"slip weight {weight:.3g} (f = {f:.3g}): evaluating the running {os.path.basename(d)}")
                ckpt = os.path.join(d, f"model_{args.iterations}.pt")
                while not os.path.exists(ckpt):
                    time.sleep(60)
                time.sleep(30)  # let the checkpoint be written completely
                display = os.path.basename(d)
                log_path = glob.glob(os.path.join(ROOT, "logs", "runs", "*", "train.log"))
                log_path = [p for p in log_path if os.path.basename(os.path.dirname(p)) in display]
                rec["train"] = train_metrics(log_path[0]) if log_path else {}
            else:
                while busy("scripts/tools/train_eval_log.py") or busy("scripts/rsl_rl/train.py"):
                    time.sleep(60)  # the GPU fits one training
                save(f"slip weight {weight:.3g} (f = {f:.3g}) -> training started")
                ckpt, display = train(args, study_dir, name, weight)
                rec["train"] = train_metrics(os.path.join(study_dir, f"{name}_train.log"))
            rec["checkpoint"] = ckpt
            rec.update(evaluate(args, study_dir, rec["name"], ckpt, args.episode_s))
            rec["url"] = wandb_url(args, display)
            rec["status"] = "done"
            verdict = decide(args, rec, ref)
            if args.wandb and not (attempt == 1 and args.first_run_dir):
                push_to_wandb(args, display, {**rec, "iterations": args.iterations, "early_stopped": None})
        except Exception as e:  # noqa: BLE001
            rec.update(status="error", error=str(e)[:200])
            verdict = "error"
        rec["date"] = time.strftime("%Y-%m-%d %H:%M")
        save(f"slip weight {weight:.3g} (f = {f:.3g}) -> {rec.get('decision', 'error')}")
        log(f"[{rec['name']}] {render_row(rec, ref)}")
        if verdict == "error":
            break
        if verdict == "weak":
            lo = f if lo is None else max(lo, f)
        elif verdict == "strong":
            hi = f if hi is None else min(hi, f)
    log("slip sweep finished")


if __name__ == "__main__":
    main()
