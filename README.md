# UniWAM

UniWAM is a camera-frame world-action model for dual-arm mobile manipulation.
This repository provides the training, data preparation, cloud inference, and
robot-client code needed to adapt the model to a customer's own task data.

The repository does not contain customer data, videos, model weights, text
embedding caches, VAE latent caches, camera calibration, or robot description
files. These assets stay on the customer's machines.

## Repository layout

- `training/src/uniwam`: model, dataset, normalization, and training code.
- `training/data_pipeline`: conversion from LeRobot episodes to camera-frame
  data, H32 windows, auxiliary labels, statistics, and latent-cache indexes.
- `training/configs`: Hydra training and data contracts.
- `deployment`: cloud server, prompt preparation, and robot-client code.
- `docs`: data, action-slot, fine-tuning, and deployment instructions.

## Quick start

Use Python 3.10 or newer, a CUDA-compatible PyTorch installation, Hydra,
PyArrow, NumPy, and the packages listed in the training and deployment
requirements files.

After replacing the placeholders in the environment template, activate it in
the shell that will run training:

```bash
source scripts/activate_customer_env.sh
```

The template is an ordinary shell file. It does not guess paths, download
private assets, or modify the repository. Edit the path block once for the
customer's machine, then source it before running the commands below. The
script checks the inputs needed at this stage. Prepared data, window indexes,
and caches can be generated afterward.

## Data contract

The default input is a dual-arm, fixed-camera dataset in LeRobot format. Each
episode contains at least 33 contiguous frames, the three manipulation camera
streams, `meta/episodes.jsonl`, and `meta/tasks.jsonl`. The task sentence in
`tasks.jsonl` is the prompt used by training and inference.

The model state contains 20 values:

`left XYZ + left row-major Rot6D + left gripper + right XYZ + right row-major Rot6D + right gripper`.

The manipulation action head contains 26 values: 20 robot-control values plus
six optional image-plane tracking values. The navigation head in the supplied
configuration is 3 values `(dx, dy, dtheta)`. An optional navigation extension
uses 9 values, `3 + 4 + 2`, by appending target-object bbox `(xmin, ymin,
xmax, ymax)` and operating-point `(x, y)`. See
[Action Slots and Feature Masks](docs/ACTION_SLOTS.md) before changing a
dimension. The supplied checkpoint remains 3D; enabling 9D requires coordinated
changes to the sidecar, dataset, model head, normalization, decoder, and tests.

### Image sizes

The source camera frames are `W=424, H=240`. The loader builds the camera
mosaic and resize/pads it to the model canvas `W=384, H=320` (`video_size:
[320, 384]`). Thus `424x240` describes the source frame, while `384x320`
describes the model input canvas. Do not reinterpret these as competing model
resolutions. Auxiliary image coordinates must declare whether they use the
source frame or the final `384x320` mosaic; the supplied EEF sidecar contract
uses the final mosaic coordinates.

### Missing auxiliary labels

Customers without point-tracking labels should append finite placeholder values
and set their six feature-mask entries to `false`. For the included customer
fine-tuning recipe, use `append_missing_eef_xy_slots=true` together with
`disable_manip_aux_training=true` (or an auxiliary loss weight of zero). Masked
features are removed from the loss numerator and denominator. Never mark a
missing value valid just to satisfy a tensor shape.

For the 9D navigation extension, return a value tensor and boolean mask for
each of the six appended bbox/operating-point slots. Missing bbox coordinates
and operating points are masked independently.

## Prepare a customer's data

Measure the rigid transforms from each arm base to the fixed camera and save
them as `camera_from_left_base` and `camera_from_right_base` in a JSON file.
Do not reuse calibration from another robot or camera rig.

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

The converter validates rigid transforms, row-major Rot6D geometry, gripper
units, frame continuity, and the H32 boundary. For a dataset already expressed
in the fixed camera frame, use the documented camera-frame option. For a legacy
23-value observation with a three-value base prefix, use the documented
`base3_arm20` option; the model still receives only the 20 arm values.

Customers may train with one dataset or combine multiple datasets. For a
multi-dataset experiment, add one dataset entry per prepared dataset in a copy
of the customer Hydra data config, assign explicit sampling weights, and give
each entry its own window index, latent source, prompt set, and statistics
contract. Do not copy a dataset name or ratio from this repository merely
because it appears in an example.

## Build caches and train

Prepare the exact task prompts, then generate text embeddings and VAE latents:

```bash
cd "$UNIWAM_TRAINING_ROOT"
python scripts/precompute_text_embeds.py task=uniwam_customer_piper_manip_sft

TASK=uniwam_customer_piper_manip_sft \
OUTPUT_ROOT="$UNIWAM_CUSTOM_LATENT_ROOT" \
NPROC_PER_NODE=1 \
bash data_pipeline/launch_precompute_vae.sh

python scripts/preflight_customer_piper.py
```

The preflight must report `CUSTOMER_SFT_INPUTS_OK` before training. Start a
single-host run with:

```bash
bash scripts/train_zero1.sh 1 task=uniwam_customer_piper_manip_sft
```

The recipe loads the supplied initialization checkpoint as weights, creates a
new optimizer, and writes periodic checkpoints. Increase the first argument to
the number of GPUs on one host. For multiple hosts, use the standard
`NODE_RANK`, `MASTER_ADDR`, `MASTER_PORT`, and `NNODES` environment variables
with the same customer task configuration on every host.

## Cloud inference and robot execution

Use the exact fine-tuned checkpoint and the exact task sentence from the
customer's `tasks.jsonl`:

```bash
export CHECKPOINT=/absolute/path/to/run/checkpoints/weights/step_xxxxxx.pt
export TASK_PROMPT='The exact task sentence used for this run'
export UNIWAM_PIPER_URDF=/absolute/path/to/robot_description.urdf

export UNIWAM_CLOUD_TASK=uniwam_customer_piper_manip_sft
export UNIWAM_CLOUD_SOURCE_INDEX=0
export UNIWAM_CLOUD_AUX_WEIGHT=0.0

CONFIG="$REPO/deployment/configs/uniwam_cloud_customer_async.yaml" \
bash "$REPO/deployment/scripts/precompute_inference_prompt_async_prefix12_rot6d.sh"

CONFIG="$REPO/deployment/configs/uniwam_cloud_customer_async.yaml" \
bash "$REPO/deployment/scripts/launch_uniwam_cloud_server.sh"
```

Generate the robot-client configuration from the same calibration JSON used by
data preparation, then begin with a dry run:

```bash
python "$REPO/deployment/scripts/make_customer_piper_robot_config.py" \
  --dataset-root "$UNIWAM_CUSTOM_DATA" \
  --calibration /absolute/path/to/customer_camera_extrinsics.json \
  --output /absolute/path/to/customer_robot.yaml

CONFIG=/absolute/path/to/customer_robot.yaml \
bash "$REPO/deployment/scripts/launch_uniwam_piper_client.sh" \
  server_host=127.0.0.1 server_port=8031 \
  inference_mode=manip_only task_prompt="$TASK_PROMPT" \
  control.dry_run=true base.dry_run=true
```

Confirm camera topics, frame transforms, action signs, units, and IK outputs in
the dry-run recording before enabling physical motion. Network forwarding and
robot safety procedures are deployment-specific.

## License

UniWAM contributions are released under the [MIT License](LICENSE). Bundled
third-party code retains its original terms; see
[Third-Party Notices](THIRD_PARTY_NOTICES.md).
