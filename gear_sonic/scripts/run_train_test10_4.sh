#!/bin/bash
# GPUs are in Exclusive_Process compute mode with MPS server (PID 3853) holding
# the exclusive CUDA context on all 8 GPUs.  Do NOT bypass MPS — each accelerate
# rank connects through MPS which multiplexes them onto their assigned GPU.
# (Bypassing MPS causes "busy or unavailable" because MPS owns the exclusive slot.)
#
# VK_ICD_FILENAMES — required for IsaacSim Vulkan device creation on headless servers.
export VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json

NUM_GPUS=${1:-8}

accelerate launch \
  --num_processes=${NUM_GPUS} \
  --mixed_precision=no \
  --num_machines=1 \
  --dynamo_backend=no \
  gear_sonic/train_agent_trl.py \
  +exp=manager/universal_token/all_modes/sonic_release \
  +checkpoint=sonic_release/last.pt \
  num_envs=4096 \
  headless=True \
  "++algo.trl.save_interval=100" \
  "++algo.trl.bf16=false" \
  "++manager_env.commands.motion.motion_lib_cfg.motion_file=gear_sonic/data/gmr_test10_4_motion_lib.pkl" \
  "++manager_env.commands.motion.motion_lib_cfg.smpl_motion_file=dummy" \
  "++manager_env.commands.motion.encoder_sample_probs.g1=1.0" \
  "++manager_env.commands.motion.encoder_sample_probs.teleop=0.0" \
  "++manager_env.commands.motion.encoder_sample_probs.smpl=0.0"
