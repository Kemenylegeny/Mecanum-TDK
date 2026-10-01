#!/bin/bash
# Penalty calibration for OUR OWN robot (Mecanum-Navigation-Flat-Own-v0), queued after the running jobs:
#   stage 0: no penalties (final_position 50 + exploration bias, goals <= 3 m, 9 s), gate 80 % success
#   stage 1-3: wheel torque -> action rate -> wheel slip, each from scratch, w = -f * 50 / (m_ref * 9 s) with m_ref
#              measured on OUR robot's previous accepted policy; f tuned (too strong: weaker, no effect: stronger)
# Every run -> EXPERIMENTS.md + git push. Log: logs/queue/own_flat3m_penalties.log
cd "$(dirname "$0")/../.."
PY=/home/bence.farkas@egroup.hu/miniconda3/envs/env_isaaclab/bin/python
mkdir -p logs/queue
WANDB_USERNAME=${WANDB_USERNAME:-varadipeter05-budapesti-m-szaki-s-gazdas-gtudom-nyi-egyetem} setsid nohup \
  $PY -u scripts/tools/run_when_idle.py -- $PY -u scripts/tools/penalty_study.py \
  --study own_flat3m --task Mecanum-Navigation-Flat-Own-v0 --experiment mecanum_navigation_flat_own \
  --episode_s 9 --task_max 50 --gate_success 0.8 \
  --base env.commands.goal_pose.max_distance=3.0 env.rewards.final_position.weight=50.0 \
         env.rewards.wheel_torque_l2.weight=0.0 env.rewards.action_rate_l2.weight=0.0 env.rewards.wheel_slip_l2.weight=0.0 \
  --terms wheel_torque_l2 action_rate_l2 wheel_slip_l2 \
  --floor torque2=1e-4 action_rate2=1e-4 slip2=1e-4 \
  --wandb --video --git > logs/queue/own_flat3m_penalties.log 2>&1 < /dev/null &
echo "queued (log: logs/queue/own_flat3m_penalties.log)"
