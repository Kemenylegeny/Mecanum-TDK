# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg


@configclass
class PPORunnerCfg(RslRlOnPolicyRunnerCfg):
    # as in Rudin et al. 2022: 48 steps per env and iteration (larger batch against the sparse task reward), 2000 its
    num_steps_per_env = 48
    max_iterations = 2000
    save_interval = 100
    experiment_name = "mecanum_navigation_pillars"
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=0.5,
        actor_obs_normalization=True,
        critic_obs_normalization=True,
        actor_hidden_dims=[256, 128, 64],
        critic_hidden_dims=[256, 128, 64],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        # no entropy bonus (see PPORunnerFlatCfg); all pillar runs since 2026-09-29 used 0 via Hydra
        entropy_coef=0.0,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )


@configclass
class PPORunnerIkCfg(PPORunnerCfg):
    experiment_name = "mecanum_navigation_pillars_ik"


@configclass
class PPORunnerFlatCfg(PPORunnerCfg):
    experiment_name = "mecanum_navigation_flat"
    # no entropy bonus: with only the task reward and the exploration bias (no action / motion penalties) nothing
    # counteracts it, and since the wheel commands are clipped, a wider action distribution costs almost no reward.
    # With entropy_coef = 0.005 the action std grew from 0.5 to 4.3 (Loss/entropy 2.9 -> 11.5) in 500 iterations and
    # the sampled wheel commands became random full-speed bang-bang (robots jittering in place during training).
    algorithm = PPORunnerCfg().algorithm.replace(entropy_coef=0.0)


@configclass
class PPORunnerFlatOwnCfg(PPORunnerFlatCfg):
    experiment_name = "mecanum_navigation_flat_own"
