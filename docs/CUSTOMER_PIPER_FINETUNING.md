# Fine-tune UniWAM on a Piper/AgileX task

This path adapts the six-source 20D-state/26D-action checkpoint to a **single
Piper/AgileX manipulation dataset**. The customer supplies their own task text,
videos, robot calibration, and LeRobot episodes. It does not require any of the
original six datasets. It is not a generic adapter for other robot kinematics,
camera layouts, or action units.

## Supported input contract

The action-current LeRobot input must have `observation.state` as 23 floats:
`[base x, base y, base yaw, left EEF xyz/row-major Rot6D/gripper,
right EEF xyz/row-major Rot6D/gripper]`. `action.manip` is the corresponding
20-float dual-arm target in the two arm-base frames. XYZ and gripper widths are
in meters; rotations are the first two **rows** of a rotation matrix. The
manipulation camera keys are `cam_manip_high`, `cam_left_wrist`, and
`cam_right_wrist` at 424x240. Each LeRobot episode needs at least 33 frames,
contiguous `frame_index`, videos, `meta/episodes.jsonl`, and `meta/tasks.jsonl`.
The `task` field in `tasks.jsonl` is the customer's task sentence. Keep the
source dataset and its video directory available after conversion; the
camera-frame dataset links to those videos.

The Piper/AgileX q01/q99 stats shipped in `training/data_indices` are reused
for both fine-tuning and deployment. Do not substitute the unified or Franka
stats. The six auxiliary point-tracking outputs are masked out of the loss;
they are not needed in the customer dataset. This keeps the parent 26D model
shape without inventing point-tracking labels.

## Environment and assets

Use Python 3.10 and a CUDA-matched PyTorch 2.7.1/torchvision 0.22.1 build.
The non-CUDA training dependencies from the tested environment are listed in
`training/requirements-training.txt`; cloud IK also needs
`deployment/requirements-cloud.txt`. Install the CUDA build first, then these
files. The edge device additionally needs ROS 2 Humble, Piper/AgileX ROS
packages, and `deployment/scripts/bootstrap_robot_python_deps.sh`.
An FFmpeg executable is required for episode video decoding.

```bash
export REPO=/absolute/path/to/UniWAM
python -m pip install -r "$REPO/training/requirements-training.txt"
python -m pip install -r "$REPO/deployment/requirements-cloud.txt"
```

The Wan/ActionDiT weights, a complete 12 GB+ six-source **split-stats**
checkpoint with its adjacent `config.yaml`, and a Piper URDF are external
assets. They are not licensed or distributed by this source repository.

```bash
export REPO=/absolute/path/to/UniWAM
export UNIWAM_TRAINING_ROOT="$REPO/training"
export DIFFSYNTH_MODEL_BASE_PATH=/absolute/path/to/Wan-and-ActionDiT-weights
export UNIWAM_PARENT_CHECKPOINT=/absolute/path/to/parent-run/checkpoints/weights/step_200000.pt
export UNIWAM_TEXT_CACHE_DIR=/absolute/path/to/customer-text-cache
export UNIWAM_CUSTOM_SOURCE_NAME=customer_piper_task
export UNIWAM_CUSTOM_DATA=/absolute/path/to/customer_piper_task
export UNIWAM_CUSTOM_WINDOW_INDEX=/absolute/path/to/customer_piper_task_h32.parquet
export UNIWAM_CUSTOM_LATENT_ROOT=/absolute/path/to/customer-vae-cache
export PYTHONPATH="$UNIWAM_TRAINING_ROOT/src:$UNIWAM_TRAINING_ROOT"
```

For newly generated latent caches, `UNIWAM_CUSTOM_SOURCE_NAME` must equal the
final camera-frame dataset directory name. For an existing cache, set it to
the directory name inside `UNIWAM_CUSTOM_LATENT_ROOT`. All paths must be
readable on every training rank.

## Prepare customer data

Measure `camera_from_left_base` and `camera_from_right_base` for **this robot
and camera rig**. Store the two rigid 4x4 matrices in a JSON file. Never reuse
the repository's example/historical extrinsics without measuring your rig.

```bash
python "$REPO/training/data_pipeline/prepare_customer_piper_views.py" \
  --source-root /absolute/path/to/action_current_lerobot \
  --output-root "$UNIWAM_CUSTOM_DATA" \
  --calibration /absolute/path/to/customer_camera_extrinsics.json

python "$REPO/training/data_pipeline/prepare_customer_piper_h32.py" \
  --dataset-root "$UNIWAM_CUSTOM_DATA" \
  --calibration /absolute/path/to/customer_camera_extrinsics.json \
  --output "$UNIWAM_CUSTOM_WINDOW_INDEX"
```

The second command checks every episode's raw-to-camera pose transform,
Rot6D geometry, gripper units, and H32 boundaries. It refuses to overwrite
an existing index. The historical `prepare_agilex_camera_views.py` remains
only for byte-equivalent reconstruction of the older six-source mobile data;
do not use it for a new customer dataset.

Encode the customer's exact task sentences and the 33-frame visual windows:

```bash
cd "$UNIWAM_TRAINING_ROOT"
python scripts/precompute_text_embeds.py task=uniwam_customer_piper_manip_sft
TASK=uniwam_customer_piper_manip_sft \
OUTPUT_ROOT="$UNIWAM_CUSTOM_LATENT_ROOT" \
NPROC_PER_NODE=1 \
bash data_pipeline/launch_precompute_vae.sh
python scripts/preflight_customer_piper.py
```

Raise `NPROC_PER_NODE` only to the number of visible GPUs. The preflight
loads a real sample and verifies the parent identity, stats hash, latent/text
caches, 26D action shape, and zero mask on all six auxiliary features. It must
print `CUSTOMER_SFT_INPUTS_OK` before training.

## Fine-tune

```bash
cd "$UNIWAM_TRAINING_ROOT"
bash scripts/train_zero1.sh 1 task=uniwam_customer_piper_manip_sft
```

The template loads **weights only** from `UNIWAM_PARENT_CHECKPOINT`, starts a
fresh optimizer at step 0, uses a fresh learning-rate schedule, and defaults
to 20,000 steps with one checkpoint every 2,000 steps. Its per-GPU batch size
is 16. Increase the first launcher argument for more GPUs on one host. Check
the first loss and later loss/throughput in the run log before relying on the
result; a successful Hydra compose or data preflight is not a training run.

## Cloud and Piper edge

Use the fine-tuned checkpoint, **not** the parent checkpoint, with the
customer task/source contract. Keep the training root and its customer data
paths accessible to the cloud process because the saved checkpoint config
contains those paths. The cloud service validates the checkpoint config,
Piper stats, camera mosaic, action representation, and prompt prefix before
listening.

```bash
export UNIWAM_CLOUD_TASK=uniwam_customer_piper_manip_sft
export UNIWAM_CLOUD_SOURCE_INDEX=0
export UNIWAM_CLOUD_AUX_WEIGHT=0.0
export UNIWAM_PIPER_URDF=/absolute/path/to/piper_description.urdf
export CHECKPOINT=/absolute/path/to/sft-run/checkpoints/weights/step_020000.pt
export TASK_PROMPT='Your exact task sentence from meta/tasks.jsonl'

CONFIG="$REPO/deployment/configs/uniwam_cloud_parent_async.yaml" \
bash "$REPO/deployment/scripts/precompute_inference_prompt_async_prefix12_rot6d.sh"

PYTHONPATH="$REPO/deployment/cloud:$REPO/deployment/common:$UNIWAM_TRAINING_ROOT/src" \
python "$REPO/deployment/scripts/preflight_cloud_contract.py" \
  --config "$REPO/deployment/configs/uniwam_cloud_parent_async.yaml" \
  --checkpoint "$CHECKPOINT"

CONFIG="$REPO/deployment/configs/uniwam_cloud_parent_async.yaml" \
bash "$REPO/deployment/scripts/launch_uniwam_cloud_server.sh"
```

Run prompt precomputation separately for each task sentence used at inference.
On the Piper machine, generate a config from the **same** calibration JSON
used for training; the generator verifies its matrices against the dataset:

```bash
python "$REPO/deployment/scripts/make_customer_piper_robot_config.py" \
  --dataset-root "$UNIWAM_CUSTOM_DATA" \
  --calibration /absolute/path/to/customer_camera_extrinsics.json \
  --output /absolute/path/to/customer_robot.yaml

CONFIG=/absolute/path/to/customer_robot.yaml \
bash "$REPO/deployment/scripts/launch_uniwam_piper_client.sh" \
  server_host=127.0.0.1 server_port=8031 inference_mode=manip_only \
  task_prompt="$TASK_PROMPT" control.dry_run=true base.dry_run=true
```

If cloud and edge are on different machines, establish an authenticated SSH
forward for the cloud listen port and set `server_port` to the chosen local
forward. Verify camera topics, observation pose, camera transforms, IK, action
signs, and gripper units in dry-run recordings before any physical motion.
Physical execution is specific to the customer's robot safety procedure and
is not validated by this repository's automated tests.
