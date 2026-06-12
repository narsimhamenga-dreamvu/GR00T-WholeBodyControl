#!/bin/bash
python gear_sonic/eval_agent_trl.py \
  +checkpoint=sonic_release/last.pt \
  +headless=True \
  ++run_eval_loop=True \
  ++run_once=True \
  ++num_envs=10 \
  ++log_robot_states=/data/fast/workspaces/user3/sonic_inference_out/test10_4/robot_states.npz \
  "+manager_env/terminations=tracking/eval" \
  "++manager_env.commands.motion.encoder_sample_probs.g1=1.0" \
  "++manager_env.commands.motion.encoder_sample_probs.teleop=0.0" \
  "++manager_env.commands.motion.encoder_sample_probs.smpl=0.0" \
  "++manager_env.commands.motion.motion_lib_cfg.motion_file=gear_sonic/data/gmr_test10_4_motion_lib.pkl" \
  "++manager_env.commands.motion.motion_lib_cfg.smpl_motion_file=dummy"
