# SONIC Inference on GMR Outputs — Build Plan

## Goal

Run GMR-retargeted G1 robot motions through the SONIC RL policy to produce
physically-consistent whole-body motions. The two steps are:

1. **Convert** GMR PKL files → SONIC `motion_lib` PKL format
2. **Run** `eval_agent_trl.py` with the converted file + SONIC checkpoint

---

## Input Data

**Location**: `/home/ubuntu/menga/GMR/outputs/test_10/pkls/`
10 files: `000000.pkl` … `000009.pkl`

**GMR PKL schema** (each file, Python pickle):
```python
{
  "fps":       int          # 25–30 (typically 30 after GMR resampling)
  "root_pos":  (T, 3)       # root pelvis XYZ, world frame
  "root_rot":  (T, 4)       # root quaternion, **xyzw** (scipy convention)
                            # already converted from MuJoCo wxyz in run_gmr_pipeline.py:
                            #   root_rot = q[3:7][[1,2,3,0]]  # wxyz -> xyzw
  "dof_pos":   (T, 29)      # joint angles, **MuJoCo order** (from qpos[7:])
                            # MuJoCo order = MJCF actuator order in g1_mocap_29dof.xml
  "local_body_pos": None    # unused
  "link_body_list": None    # unused
}
```

---

## Target Format (SONIC `motion_lib`)

The file loaded by `gear_sonic/utils/motion_lib/motion_lib_base.py` via `joblib.load()`:
```python
# Top-level dict (file mode):
{
  "000000": { <motion_data> },
  "000001": { <motion_data> },
  ...
}

# Per-motion dict:
{
  "root_trans_offset": (T, 3)    float32  # root XYZ
  "pose_aa":           (T, 30, 3) float32  # axis-angle per body (pelvis + 29 joints)
                                           # body 0 = pelvis (root rotation as rotvec)
                                           # bodies 1–29 = DOF_AXIS * dof_value
  "dof":               (T, 29)   float32  # joint angles, MuJoCo order
  "root_rot":          (T, 4)    float32  # quaternion xyzw (scipy)
  "smpl_joints":       (T, 24, 3) float32 # placeholder zeros (not used in robot-only mode)
  "fps":               float              # frame rate
}
```

---

## Conversion Logic

From `gear_sonic/data_process/convert_soma_csv_to_motion_lib.py` (DOF_AXIS constants):

```python
NUM_DOF    = 29
NUM_BODIES = 30  # pelvis + 29 joints

DOF_AXIS = np.array([   # one row per MuJoCo DOF, axis of rotation
    [0,1,0],[1,0,0],[0,0,1],        # left hip pitch/roll/yaw
    [0,1,0],[0,1,0],[1,0,0],        # left knee, ankle pitch/roll
    [0,1,0],[1,0,0],[0,0,1],        # right hip pitch/roll/yaw
    [0,1,0],[0,1,0],[1,0,0],        # right knee, ankle pitch/roll
    [0,0,1],[1,0,0],[0,1,0],        # waist yaw/roll/pitch
    [0,1,0],[1,0,0],[0,0,1],[0,1,0],[1,0,0],[0,1,0],[0,0,1],  # left arm
    [0,1,0],[1,0,0],[0,0,1],[0,1,0],[1,0,0],[0,1,0],[0,0,1],  # right arm
], dtype=np.float32)

def convert_gmr_to_sonic(motion: dict) -> dict:
    root_trans_offset = motion["root_pos"].astype(np.float32)   # (T,3) — no change
    root_quat_xyzw    = motion["root_rot"].astype(np.float32)   # (T,4) — already xyzw
    dof               = motion["dof_pos"].astype(np.float32)    # (T,29) — already MuJoCo order

    T = root_trans_offset.shape[0]
    pose_aa = np.zeros((T, NUM_BODIES, 3), dtype=np.float32)
    pose_aa[:, 0, :] = Rotation.from_quat(root_quat_xyzw).as_rotvec()  # root
    pose_aa[:, 1:, :] = DOF_AXIS[None, :, :] * dof[:, :, None]          # joints

    return {
        "root_trans_offset": root_trans_offset,
        "pose_aa":           pose_aa,
        "dof":               dof,
        "root_rot":          root_quat_xyzw,
        "smpl_joints":       np.zeros((T, 24, 3), dtype=np.float32),
        "fps":               float(motion["fps"]),
    }
```

**Key note**: GMR `dof_pos` is already in MuJoCo order — **no** `MJ_TO_IL` reordering needed.
The SOMA converter needed reordering because its input was in IsaacLab order; GMR uses `qpos[7:]`
directly from the MuJoCo simulation.

---

## Step 1 — Write the Converter Script

**File to create**: `gear_sonic/data_process/convert_gmr_to_motion_lib.py`

```
Usage:
    conda run -n h2r python gear_sonic/data_process/convert_gmr_to_motion_lib.py \
        --input  /home/ubuntu/menga/GMR/outputs/test_10/pkls \
        --output gear_sonic/data/gmr_test10_motion_lib.pkl

Script should:
  1. glob all *.pkl files in --input directory
  2. For each file:
     a. pickle.load() it (NOT joblib — GMR saves with pickle)
     b. Call convert_gmr_to_sonic(motion)
     c. Use the stem (e.g. "000000") as the dict key
  3. joblib.dump(all_motions, output_path)
```

**Environment**: `conda run -n h2r` (has `scipy`, `numpy`, `joblib`, `gear_sonic`)

**Output**: `gear_sonic/data/gmr_test10_motion_lib.pkl`

---

## Step 2 — Download SONIC Checkpoint

The released checkpoint lives on HuggingFace at `nvidia/GEAR-SONIC`:

```bash
conda run -n h2r python download_from_hf.py --training --no-smpl
# Downloads to: sonic_release/last.pt  and  sonic_release/config.yaml
```

(Use `--no-smpl` to skip the 30 GB SMPL data; we only need the checkpoint.)

If already downloaded, checkpoint is at `sonic_release/last.pt`.

---

## Step 3 — Run SONIC Inference

Requires **Isaac Lab** (not currently installed on this machine — install separately per
https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html).

Once Isaac Lab is available, activate its environment and run from repo root:

```bash
# --- Metrics (headless) ---
python gear_sonic/eval_agent_trl.py \
    +checkpoint=sonic_release/last.pt \
    +headless=True \
    ++eval_callbacks=im_eval \
    ++run_eval_loop=False \
    ++num_envs=10 \
    "+manager_env/terminations=tracking/eval" \
    "++manager_env.commands.motion.motion_lib_cfg.motion_file=gear_sonic/data/gmr_test10_motion_lib.pkl" \
    "++manager_env.commands.motion.motion_lib_cfg.smpl_motion_file=dummy"

# --- Render videos ---
python gear_sonic/eval_agent_trl.py \
    +checkpoint=sonic_release/last.pt \
    +headless=True \
    ++eval_callbacks=im_eval \
    ++run_eval_loop=False \
    ++num_envs=10 \
    ++manager_env.config.render_results=True \
    "++manager_env.config.save_rendering_dir=/tmp/sonic_renders" \
    ++manager_env.config.env_spacing=10.0 \
    "~manager_env/recorders=empty" "+manager_env/recorders=render" \
    "++manager_env.commands.motion.motion_lib_cfg.motion_file=gear_sonic/data/gmr_test10_motion_lib.pkl" \
    "++manager_env.commands.motion.motion_lib_cfg.smpl_motion_file=dummy"
```

The `smpl_motion_file=dummy` tells motion_lib to skip SMPL data (robot-only tracking mode).
The released checkpoint config embeds internal training paths, so `motion_file` must always
be overridden explicitly.

---

## Environment Summary

| Task                    | Environment              | Notes                              |
|-------------------------|--------------------------|------------------------------------|
| Conversion script       | `conda run -n h2r`       | Has gear_sonic, scipy, joblib      |
| Checkpoint download     | `conda run -n h2r`       | Needs huggingface_hub              |
| SONIC eval (Isaac Lab)  | Isaac Lab env (missing)  | Must install separately            |

---

## Relevant Files

| File | Purpose |
|------|---------|
| `gear_sonic/data_process/convert_soma_csv_to_motion_lib.py` | Reference converter (SOMA→motion_lib); contains DOF_AXIS constants |
| `gear_sonic/utils/motion_lib/motion_lib_base.py` | motion_lib loader; shows expected PKL schema |
| `gear_sonic/eval_agent_trl.py` | SONIC eval entry point |
| `gear_sonic/config/base_eval.yaml` | Base eval config |
| `data/run_gmr_pipeline.py` | GMR pipeline that produced the input PKLs (shows root_rot is xyzw, dof_pos is MuJoCo order) |
| `download_from_hf.py` | Download SONIC checkpoint from HuggingFace |
