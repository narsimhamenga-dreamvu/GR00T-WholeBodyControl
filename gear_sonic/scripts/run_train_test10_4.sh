#!/bin/bash
# Bypass NVIDIA MPS so IsaacSim's CUDA init can enumerate GPUs directly.
# MPS is active on this machine but its server is not ready (Error 807),
# which blocks carb.cudainterop from calling cudaGetDeviceCount().
# Pointing CUDA_MPS_PIPE_DIRECTORY to a missing path forces CUDA to skip MPS.
export CUDA_MPS_PIPE_DIRECTORY=/tmp/no_mps_$$
# Vulkan ICD path — required for IsaacSim GPU device creation on headless servers
export VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json

accelerate launch \
  --num_processes=8 \
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
