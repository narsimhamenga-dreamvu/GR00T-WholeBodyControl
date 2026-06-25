#!/bin/bash
ulimit -n 65536

python gear_sonic/eval_agent_trl.py \
  +checkpoint=sonic_release/last.pt \
  +headless=True \
  ++run_eval_loop=True \
  ++run_once=True \
  ++num_envs=1338 \
  ++log_robot_states=/hdd/gmr_sonic_out/yaw_postik_eval/robot_states.npz \
  "+manager_env/terminations=tracking/eval" \
  "++manager_env.commands.motion.encoder_sample_probs.g1=1.0" \
  "++manager_env.commands.motion.encoder_sample_probs.teleop=0.0" \
  "++manager_env.commands.motion.encoder_sample_probs.smpl=0.0" \
  "++manager_env.commands.motion.motion_lib_cfg.motion_file=gear_sonic/data/gmr_yaw_postik_motion_lib.pkl" \
  "++manager_env.commands.motion.motion_lib_cfg.smpl_motion_file=dummy"
