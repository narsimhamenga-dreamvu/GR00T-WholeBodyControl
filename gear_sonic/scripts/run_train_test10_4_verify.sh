#!/bin/bash
# Quick config-verify run: 1 process, 64 envs, stops after 10 iterations.
# Use this to confirm overrides are accepted before launching the full 8-GPU job.
export VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json

accelerate launch \
  --num_processes=1 \
  --mixed_precision=no \
  --num_machines=1 \
  --dynamo_backend=no \
  gear_sonic/train_agent_trl.py \
  +exp=manager/universal_token/all_modes/sonic_release \
  +checkpoint=sonic_release/last.pt \
  num_envs=64 \
  headless=True \
  "++algo.config.save_interval=100" \
  "++algo.config.num_learning_iterations=10" \
  "++algo.trl.bf16=false" \
  "++manager_env.commands.motion.motion_lib_cfg.motion_file=gear_sonic/data/gmr_test10_4_motion_lib.pkl" \
  "++manager_env.commands.motion.motion_lib_cfg.smpl_motion_file=dummy" \
  "++manager_env.commands.motion.encoder_sample_probs.g1=1.0" \
  "++manager_env.commands.motion.encoder_sample_probs.teleop=0.0" \
  "++manager_env.commands.motion.encoder_sample_probs.smpl=0.0"
