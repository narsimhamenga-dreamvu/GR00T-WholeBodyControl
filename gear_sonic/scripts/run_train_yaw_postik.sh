#!/bin/bash
ulimit -n 65536
export VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json

NUM_GPUS=${1:-1}

accelerate launch \
  --num_processes=${NUM_GPUS} \
  --mixed_precision=no \
  --num_machines=1 \
  --dynamo_backend=no \
  gear_sonic/train_agent_trl.py \
  +exp=manager/universal_token/all_modes/sonic_release \
  +checkpoint=sonic_release/last.pt \
  num_envs=512 \
  headless=True \
  "++algo.config.save_interval=100" \
  "++algo.trl.bf16=false" \
  "++manager_env.commands.motion.motion_lib_cfg.motion_file=gear_sonic/data/gmr_yaw_postik_motion_lib.pkl" \
  "++manager_env.commands.motion.motion_lib_cfg.smpl_motion_file=dummy" \
  "++manager_env.commands.motion.encoder_sample_probs.g1=1.0" \
  "++manager_env.commands.motion.encoder_sample_probs.teleop=0.0" \
  "++manager_env.commands.motion.encoder_sample_probs.smpl=0.0"
