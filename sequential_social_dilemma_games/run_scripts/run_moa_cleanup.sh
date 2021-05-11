#!/usr/bin/env bash

python train.py \
--env cleanup \
--model baseline \
--algorithm A3C \
--num_agents 2 \
--num_workers 4 \
--rollout_fragment_length 1000 \
--num_envs_per_worker 4 \
--stop_at_timesteps_total $((500 * 10 ** 6)) \
--memory $((160 * 10 ** 9)) \
--cpus_per_worker 1 \
--gpus_per_worker 0 \
--gpus_for_driver 0 \
--cpus_for_driver 1 \
--num_samples 4 \
--entropy_coeff 0.00176 \
--moa_loss_weight 0.06663557 \
--lr_schedule_steps 0 20000000 \
--lr_schedule_weights 0.00126 0.000012 \
--contribute_reward_weight 1.0 \
--contribute_reward_schedule_steps 0 10000000 100000000 300000000 \
--contribute_reward_schedule_weights 0.0 0.0 1.0 0.5