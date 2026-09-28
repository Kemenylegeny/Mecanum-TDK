# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Reward / hyper-parameter search for the pillar navigation task, with early stopping.

Every trial is a set of Hydra overrides (reward weights, PPO parameters, ...). For each trial the script trains a
policy with ``scripts/rsl_rl/train.py``, stops it early when the mean training reward and the curriculum level have
flattened, evaluates the latest checkpoint with ``scripts/rsl_rl/evaluate_navigation.py`` (deterministic policy, all
terrain levels, the same goal distribution for every trial) and ranks the trials. Trials run one after the other
(one GPU); results are appended to ``logs/tuning/<study>/results.jsonl``, so an interrupted study continues where it
stopped (finished trials are skipped).

Score (higher is better): ``success - crash_weight * crashed - time_weight * median arrival time [s]``, with
success = within 0.5 m of the goal at the end of the episode without a crash.

Usage::

    # the predefined trials of TRIALS below (in this order)
    python scripts/tools/tune_rewards.py --study round1 --wandb --video

    # 8 random combinations of SEARCH_SPACE below
    python scripts/tools/tune_rewards.py --study random1 --random 8

    # own trials from a JSON file: {"name": {"hydra.key": value, ...}, ...}
    python scripts/tools/tune_rewards.py --study mine --trials my_trials.json

    # only print the leaderboard of a study
    python scripts/tools/tune_rewards.py --study round1 --report

Useful keys (``env.`` = environment cfg, ``agent.`` = PPO cfg)::

    env.rewards.<term>.weight                         e.g. env.rewards.wheel_slip_l2.weight=-0.1
    env.rewards.final_position.params.std             sharpness of the task reward (1.0 = paper)
    env.episode_length_s                              time to reach the goal [s]
    env.commands.goal_pose.min_distance               minimum start-goal distance [m] (0 = off)
    agent.algorithm.gamma / lam / entropy_coef / learning_rate
    agent.policy.init_noise_std
"""

import argparse
import glob
import json
import os
import random
import re
import signal
import subprocess
import sys
import time

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
PYTHON = os.environ.get("PYTHON", "/home/bence.farkas@egroup.hu/miniconda3/envs/env_isaaclab/bin/python")
EXPERIMENT_DIR = os.path.join(ROOT, "logs", "rsl_rl", "mecanum_navigation_pillars")

# lighter motion penalties: they are the only terms that distinguish a fast from a slow arrival
LIGHT_PENALTIES = {
    "env.rewards.wheel_slip_l2.weight": -0.1,
    "env.rewards.wheel_acc_l2.weight": -2.5e-7,
    "env.rewards.wheel_torque_l2.weight": -5.0e-5,
}

# predefined trials in priority order (the study runs them one after the other); each one targets the problems found
# in the evaluation of the first run (see README). The circle -> rectangle collision fix is in the base config.
TRIALS = {
    # reference: current config
    "baseline": {},
    # everything below together
    "combo": {
        "agent.algorithm.gamma": 0.995,
        "agent.algorithm.entropy_coef": 0.01,
        "env.rewards.final_position.params.std": 0.5,
        "env.commands.goal_pose.min_distance": 1.5,
        "env.episode_length_s": 10.0,
        **LIGHT_PENALTIES,
    },
    # (2) discount horizon: with gamma 0.99 (~2 s) the task reward at the end is weighted 0.99^500 ~ 0.007 at the
    # start of the episode -> slow cruising, then late rushing and crashing
    "gamma995": {"agent.algorithm.gamma": 0.995},
    # (3) no incentive for speed: less time and lighter motion penalties
    "speed": {"agent.algorithm.gamma": 0.995, "env.episode_length_s": 8.0, **LIGHT_PENALTIES},
    # (4) flat task reward: stopping 1 m short pays 0.2 instead of 0.5; (5) no trivial goals (4 % were < 1 m)
    "precise": {
        "agent.algorithm.gamma": 0.995,
        "env.rewards.final_position.params.std": 0.5,
        "env.commands.goal_pose.min_distance": 1.5,
    },
    # (2) even longer horizon on top of the combo
    "combo_gamma998": {
        "agent.algorithm.gamma": 0.998,
        "agent.algorithm.entropy_coef": 0.01,
        "env.rewards.final_position.params.std": 0.5,
        "env.commands.goal_pose.min_distance": 1.5,
        "env.episode_length_s": 10.0,
        **LIGHT_PENALTIES,
    },
    # (5) exploration: the action std dropped from 0.46 to 0.16 within 100 iterations
    "entropy": {"agent.algorithm.gamma": 0.995, "agent.algorithm.entropy_coef": 0.01},
}

# random search: every trial draws one value per key
SEARCH_SPACE = {
    "agent.algorithm.gamma": [0.995, 0.998],
    "agent.algorithm.entropy_coef": [0.005, 0.01],
    "env.rewards.final_position.weight": [10.0, 20.0],
    "env.rewards.final_position.params.std": [0.5, 0.7, 1.0],
    "env.rewards.stalling.weight": [-0.2, -0.5, -1.0],
    "env.rewards.wheel_slip_l2.weight": [-0.05, -0.2, -0.5],
    "env.rewards.wheel_acc_l2.weight": [-2.5e-7, -1.0e-6],
    "env.rewards.wheel_torque_l2.weight": [-5.0e-5, -2.0e-4],
    "env.rewards.obstacle_proximity.weight": [-0.5, -1.0, -2.0],
    "env.rewards.collision.weight": [-50.0, -100.0, -200.0],
    "env.episode_length_s": [8.0, 10.0, 12.0],
    "env.commands.goal_pose.min_distance": [0.0, 1.5],
}

# every trial is evaluated on the same goal distribution; only settings the policy depends on (its episode length,
# i.e. the meaning of the time-left observation) are taken over from the trial
EVAL_OVERRIDES = {"env.commands.goal_pose.min_distance": 1.0}
EVAL_TRIAL_KEYS = ("env.episode_length_s",)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--study", required=True, help="Name of the study (folder under logs/tuning).")
    parser.add_argument("--trials", help="JSON file {trial_name: {override: value}} instead of TRIALS.")
    parser.add_argument("--random", type=int, default=0, help="Number of random trials drawn from SEARCH_SPACE.")
    parser.add_argument("--only", nargs="*", help="Run only these trial names.")
    parser.add_argument("--iterations", type=int, default=500, help="Maximum PPO iterations per trial.")
    # 4096 envs + --video crash in Omniverse's renderer (cubric assertion -> CUDA illegal address) in the first
    # iteration (4096 robots x 62 links + 2 x 4096 markers = 2^18 rendered transforms); 3072 is stable
    parser.add_argument("--num_envs", type=int, default=3072)
    parser.add_argument("--eval_envs", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--video", action="store_true", help="Record a 10 s video every 100 iterations (~2.5 GB).")
    parser.add_argument("--wandb", action="store_true", help="Log every trial to W&B (group = study name).")
    parser.add_argument("--wandb_project", default="mecanum-navigation")
    # early stopping
    parser.add_argument("--min_iterations", type=int, default=150, help="Never stop before this iteration.")
    parser.add_argument("--es_window", type=int, default=75, help="Compare the mean of the last N iterations ...")
    parser.add_argument("--es_every", type=int, default=25, help="... with the N before, every this many iterations.")
    parser.add_argument("--es_rel_tol", type=float, default=0.03, help="Flat: mean reward improved less than this.")
    parser.add_argument("--es_level_tol", type=float, default=0.2, help="Flat: curriculum level rose less than this.")
    parser.add_argument("--es_patience", type=int, default=2, help="Stop after this many flat checks in a row.")
    parser.add_argument("--no_early_stop", action="store_true")
    # score
    parser.add_argument("--crash_weight", type=float, default=0.5)
    parser.add_argument("--time_weight", type=float, default=0.01)
    parser.add_argument("--report", action="store_true", help="Only print the leaderboard.")
    return parser.parse_args()


def fmt(value) -> str:
    return repr(value) if isinstance(value, float) else str(value)


def log(msg: str):
    print(f"{time.strftime('%H:%M:%S')} {msg}", flush=True)


##
# Training progress / early stopping
##

ITER_RE = re.compile(r"Learning iteration (\d+)/\d+")
VALUE_RE = {
    "reward": re.compile(r"Mean reward:\s*(-?[\d.]+(?:e-?\d+)?)"),
    "level": re.compile(r"Curriculum/terrain_levels:\s*(-?[\d.]+)"),
}


def read_progress(log_path: str) -> dict[int, dict]:
    """Per iteration: mean reward and curriculum level, parsed from the RSL-RL console output."""
    with open(log_path, errors="ignore") as f:
        text = f.read()
    progress = {}
    blocks = ITER_RE.split(text)  # [before, it, block, it, block, ...]
    for it, block in zip(blocks[1::2], blocks[2::2]):
        values = {k: float(m.group(1)) for k, r in VALUE_RE.items() if (m := r.search(block))}
        if "reward" in values:
            progress[int(it)] = values
    return progress


class EarlyStopper:
    """Stops when, ``patience`` checks in a row, both the mean reward and the curriculum level of the last
    ``window`` iterations barely improved over the ``window`` iterations before."""

    def __init__(self, args):
        self.args = args
        self.flat = 0
        self.next_check = max(args.min_iterations, 2 * args.es_window)
        self.reason = None

    def __call__(self, progress: dict[int, dict]) -> bool:
        a = self.args
        if a.no_early_stop or not progress:
            return False
        last = max(progress)
        while last >= self.next_check:
            it = self.next_check
            self.next_check += a.es_every
            recent = [progress[i] for i in range(it - a.es_window + 1, it + 1) if i in progress]
            before = [progress[i] for i in range(it - 2 * a.es_window + 1, it - a.es_window + 1) if i in progress]
            if len(recent) < a.es_window // 2 or len(before) < a.es_window // 2:
                continue
            mean = lambda xs, k: sum(x.get(k, 0.0) for x in xs) / len(xs)  # noqa: E731
            r_new, r_old = mean(recent, "reward"), mean(before, "reward")
            rel = (r_new - r_old) / max(abs(r_old), 1.0)
            dlevel = mean(recent, "level") - mean(before, "level")
            if rel < a.es_rel_tol and dlevel < a.es_level_tol:
                self.flat += 1
            else:
                self.flat = 0
            if self.flat >= a.es_patience:
                self.reason = (
                    f"flat at iteration {it}: mean reward {r_old:.2f} -> {r_new:.2f} ({rel:+.1%}), "
                    f"curriculum level {dlevel:+.2f} over the last {a.es_window} iterations"
                )
                return True
        return False


def stop(proc: subprocess.Popen, grace_s: float = 90.0):
    """SIGINT first (lets W&B finish the run), SIGKILL if Isaac Sim does not exit."""
    try:
        os.killpg(proc.pid, signal.SIGINT)
        proc.wait(timeout=grace_s)
    except (subprocess.TimeoutExpired, ProcessLookupError):
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        proc.wait()


def run(cmd: list[str], log_path: str, env: dict, done=None, should_stop=None, idle_kill_s: float = 90.0) -> str:
    """Run a command, logging to a file. Returns "exit" / "done" / "stopped".

    ``done()``: the work is finished; if the log then stays idle for ``idle_kill_s``, the process is killed
    (Isaac Sim's shutdown sometimes hangs). ``should_stop()``: stop the process now (early stopping)."""
    with open(log_path, "w") as f:
        proc = subprocess.Popen(cmd, cwd=ROOT, stdout=f, stderr=subprocess.STDOUT, env=env, start_new_session=True)
        try:
            while proc.poll() is None:
                time.sleep(15)
                if should_stop is not None and should_stop():
                    stop(proc)
                    return "stopped"
                if done is not None and done() and time.time() - os.path.getmtime(log_path) > idle_kill_s:
                    stop(proc, grace_s=5)
                    return "done"
        except KeyboardInterrupt:
            os.killpg(proc.pid, signal.SIGKILL)
            raise
    return "exit"


##
# Trials
##


def latest_checkpoint(run_dir: str) -> str | None:
    ckpts = glob.glob(os.path.join(run_dir, "model_*.pt"))
    return max(ckpts, key=lambda p: int(re.search(r"model_(\d+)\.pt", p).group(1))) if ckpts else None


def run_trial(args, study_dir: str, name: str, overrides: dict) -> dict:
    run_name = f"tune_{args.study}_{name}"
    env = dict(os.environ)
    train_cmd = [
        PYTHON, "-u", "scripts/rsl_rl/train.py",
        "--task", "Mecanum-Navigation-Pillars-v0", "--headless",
        "--num_envs", str(args.num_envs), "--max_iterations", str(args.iterations),
        "--seed", str(args.seed), "--run_name", run_name,
        # frequent checkpoints, so an early-stopped run is evaluated close to where it stopped
        "agent.save_interval=25",
    ]  # fmt: skip
    if args.video:
        train_cmd += ["--video", "--video_length", "500", "--video_interval", str(48 * 100)]
    if args.wandb:
        train_cmd += ["--logger", "wandb", "--log_project_name", args.wandb_project]
        env["WANDB_RUN_GROUP"] = args.study
    train_cmd += [f"{k}={fmt(v)}" for k, v in overrides.items()]

    started = time.time()
    final_ckpt = f"model_{args.iterations - 1}.pt"
    train_log = os.path.join(study_dir, f"{name}_train.log")

    def run_dir():
        dirs = [d for d in glob.glob(os.path.join(EXPERIMENT_DIR, f"*_{run_name}")) if os.path.getmtime(d) >= started - 5]
        return max(dirs, key=os.path.getmtime) if dirs else None

    def finished():
        d = run_dir()
        return d is not None and os.path.exists(os.path.join(d, final_ckpt))

    stopper = EarlyStopper(args)
    log(f"[{name}] training (max {args.iterations} iterations): {overrides or '(current config)'}")
    status = run(train_cmd, train_log, env, done=finished, should_stop=lambda: stopper(read_progress(train_log)))
    progress = read_progress(train_log)
    iterations = max(progress) + 1 if progress else 0
    d = run_dir()
    checkpoint = latest_checkpoint(d) if d else None
    if checkpoint is None or iterations < min(args.min_iterations, args.iterations - 1):
        return {"name": name, "overrides": overrides, "iterations": iterations,
                "error": f"training failed ({status}) after {iterations} iterations, see {name}_train.log"}  # fmt: skip
    if stopper.reason:
        log(f"[{name}] early stop: {stopper.reason}")

    # evaluation: same goal distribution for every trial, the trial's episode length
    eval_overrides = {**EVAL_OVERRIDES, **{k: v for k, v in overrides.items() if k in EVAL_TRIAL_KEYS}}
    eval_cmd = [
        PYTHON, "-u", "scripts/rsl_rl/evaluate_navigation.py",
        "--checkpoint", checkpoint, "--num_envs", str(args.eval_envs),
    ] + [f"{k}={fmt(v)}" for k, v in eval_overrides.items()]  # fmt: skip
    eval_json = os.path.join(d, f"eval_{os.path.splitext(os.path.basename(checkpoint))[0]}.json")
    log(f"[{name}] evaluating {os.path.relpath(checkpoint, ROOT)}")
    run(eval_cmd, os.path.join(study_dir, f"{name}_eval.log"), env, done=lambda: os.path.exists(eval_json))
    result = {
        "name": name,
        "overrides": overrides,
        "checkpoint": checkpoint,
        "iterations": iterations,
        "early_stopped": stopper.reason,
        "final_train_reward": progress[max(progress)]["reward"],
        "final_level": progress[max(progress)].get("level"),
    }
    if not os.path.exists(eval_json):
        return {**result, "error": f"evaluation failed, see {name}_eval.log"}
    result.update(summarize(eval_json, args))
    if args.wandb:
        push_to_wandb(args, os.path.basename(d), result)
    return result


def summarize(eval_json: str, args) -> dict:
    with open(eval_json) as f:
        episodes = json.load(f)["episodes"]
    n = len(episodes)
    arrival = sorted(e["arrival_time"] for e in episodes if e["arrival_time"] == e["arrival_time"])  # drop NaN
    median_arrival = arrival[len(arrival) // 2] if arrival else float("inf")
    hard = [e for e in episodes if e["level"] >= 7]
    m = {
        "success": sum(e["success"] for e in episodes) / n,
        "crashed": sum(e["collided"] for e in episodes) / n,
        "success_hard": sum(e["success"] for e in hard) / max(len(hard), 1),
        "median_arrival_s": median_arrival,
        "mean_speed": sum(e["path_len"] / e["time"] for e in episodes) / n,
    }
    m["score"] = m["success"] - args.crash_weight * m["crashed"] - args.time_weight * min(median_arrival, 60.0)
    return m


def push_to_wandb(args, run_display_name: str, result: dict):
    """Attach the evaluation to the trial's W&B run and mark it finished (an early-stopped training process is killed
    before it can close its run, which W&B would otherwise show as "crashed"). Best effort."""
    try:
        import wandb

        api = wandb.Api()
        entity = os.environ.get("WANDB_USERNAME") or api.default_entity
        for r in api.runs(f"{entity}/{args.wandb_project}", filters={"display_name": run_display_name}):
            tags = sorted(set(r.tags) | {f"study:{args.study}", "early_stopped" if result["early_stopped"] else "full"})
            run = wandb.init(entity=entity, project=args.wandb_project, id=r.id, resume="must", tags=tags,
                             settings=wandb.Settings(silent=True, init_timeout=300))  # fmt: skip
            for k in ("score", "success", "success_hard", "crashed", "median_arrival_s", "mean_speed", "iterations"):
                run.summary[f"eval/{k}"] = result[k]
            if result["early_stopped"]:
                run.summary["early_stop_reason"] = result["early_stopped"]
            run.finish()
    except Exception as e:  # noqa: BLE001
        log(f"[{result['name']}] W&B summary not updated: {e}")


##
# Report
##


def short(key: str) -> str:
    """env.rewards.wheel_slip_l2.weight -> wheel_slip_l2.weight, agent.algorithm.gamma -> gamma"""
    for prefix in ("env.rewards.", "agent.algorithm.", "agent.policy.", "env.commands.goal_pose.", "env."):
        if key.startswith(prefix):
            return key[len(prefix):]
    return key


def leaderboard(results: list[dict]) -> str:
    ok = sorted((r for r in results if "score" in r), key=lambda r: -r["score"])
    lines = [
        "| # | trial | score | success | success (levels 7-9) | crashed | median arrival [s] | mean speed [m/s] "
        "| iterations | overrides |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for i, r in enumerate(ok, 1):
        ov = ", ".join(f"{short(k)}={v}" for k, v in r["overrides"].items())
        its = f"{r['iterations']}{' (early stop)' if r.get('early_stopped') else ''}"
        lines.append(
            f"| {i} | {r['name']} | {r['score']:.3f} | {r['success']:.1%} | {r['success_hard']:.1%} | {r['crashed']:.1%} "
            f"| {r['median_arrival_s']:.1f} | {r['mean_speed']:.2f} | {its} | {ov or '-'} |"
        )
    for r in results:
        if "error" in r:
            lines.append(f"| - | {r['name']} | failed: {r['error']} | | | | | | | |")
    return "\n".join(lines)


def main():
    args = parse_args()
    study_dir = os.path.join(ROOT, "logs", "tuning", args.study)
    os.makedirs(study_dir, exist_ok=True)
    results_path = os.path.join(study_dir, "results.jsonl")
    results = []
    if os.path.exists(results_path):
        with open(results_path) as f:
            results = [json.loads(line) for line in f if line.strip()]

    if not args.report:
        if args.trials:
            with open(args.trials) as f:
                trials = json.load(f)
        elif args.random:
            rng = random.Random(args.seed)
            trials = {f"rand{i:02d}": {k: rng.choice(v) for k, v in SEARCH_SPACE.items()} for i in range(args.random)}
        else:
            trials = TRIALS
        if args.only:
            trials = {k: v for k, v in trials.items() if k in args.only}
        done = {r["name"] for r in results if "score" in r}
        log(f"study {args.study}: {len(trials)} trials, {len(done & set(trials))} already done")
        for name, overrides in trials.items():
            if name in done:
                log(f"[{name}] already done, skipping")
                continue
            result = run_trial(args, study_dir, name, overrides)
            results = [r for r in results if r["name"] != name] + [result]
            with open(results_path, "w") as f:
                f.writelines(json.dumps(r) + "\n" for r in results)
            log(f"[{name}] {json.dumps({k: v for k, v in result.items() if k not in ('overrides', 'checkpoint')})}")
            with open(os.path.join(study_dir, "leaderboard.md"), "w") as f:
                f.write(leaderboard(results) + "\n")

    table = leaderboard(results)
    with open(os.path.join(study_dir, "leaderboard.md"), "w") as f:
        f.write(table + "\n")
    print("\n" + table, flush=True)


if __name__ == "__main__":
    sys.exit(main())
