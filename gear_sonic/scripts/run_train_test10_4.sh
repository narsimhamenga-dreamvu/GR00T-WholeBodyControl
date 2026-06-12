#!/bin/bash
accelerate launch \
  --num_processes=1 \
  --mixed_precision=no \
  --num_machines=1 \
  --dynamo_backend=no \
  gear_sonic/train_agent_trl.py \
  +exp=manager/universal_token/all_modes/sonic_release \
  +checkpoint=sonic_release/last.pt \
  num_envs=4096 \
  headless=True \
  "++algo.trl.bf16=false" \
  "++manager_env.commands.motion.motion_lib_cfg.motion_file=gear_sonic/data/gmr_test10_4_motion_lib.pkl" \
  "++manager_env.commands.motion.motion_lib_cfg.smpl_motion_file=dummy" \
  "++manager_env.commands.motion.encoder_sample_probs.g1=1.0" \
  "++manager_env.commands.motion.encoder_sample_probs.teleop=0.0" \
  "++manager_env.commands.motion.encoder_sample_probs.smpl=0.0"
