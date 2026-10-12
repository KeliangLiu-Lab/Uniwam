# UniWAM

UniWAM is a unified camera-frame world-action model for dual-arm mobile manipulation.
This repository contains the renamed `uniwam` training and deployment implementation,
a thin `fastwam` compatibility namespace, and the prepared-LeRobot to six-source
training-data contract pipeline.

This is a source-only release. Large datasets, videos, calibrated tracks, VAE latent
shards, text caches, checkpoints, and robot URDFs are external inputs. No credentials
are required by the code.

## Layout

`training/src/uniwam` is the canonical implementation. `training/src/fastwam` is the
legacy compatibility layer. `training/configs` contains Hydra contracts,
`training/data_pipeline` contains prepared-data processing, `training/reference`
contains parent indexes and latent row remaps, and `deployment` contains cloud and
Piper edge code.

For a customer-owned Piper/AgileX dataset and task prompt, follow
[Customer Piper Fine-Tuning](docs/CUSTOMER_PIPER_FINETUNING.md). This is the
supported adaptation path; the six-source recipes below document the original
training contract and are not prerequisites for customer data.

## Environment

Use Python 3.10+, PyTorch, Hydra, PyArrow, NumPy, and the dependencies required by
the training and deployment environments. Before invoking a command, set all paths
to local assets explicitly in the current shell:

```bash
export UNIWAM_TRAINING_ROOT=/path/to/UniWAM/training
export UNIWAM_EEF_SIDECAR_ROOT=/path/to/eef_xy_sidecars
export UNIWAM_FRANKA_DATA=/path/to/franka_camera_frame
export UNIWAM_CUP_TRAY_DATA=/path/to/cup_tray_camera_frame
export UNIWAM_MOVE_WHITE_DATA=/path/to/move_white_box_camera_frame
export UNIWAM_COLOR_DATA=/path/to/color_blocks_camera_frame
export UNIWAM_ORDERED_COLOR_DATA=/path/to/ordered_color_camera_frame
export UNIWAM_7000_DATA=/path/to/agilex7000_camera_frame
export UNIWAM_FRANKA_LATENT_ROOT=/path/to/franka_vae_latents
export UNIWAM_MOBILE_LATENT_ROOT=/path/to/mobile_vae_latents
export UNIWAM_ORDERED_LATENT_ROOT=/path/to/ordered_color_vae_latents
export UNIWAM_7000_LATENT_ROOT=/path/to/agilex7000_vae_latents
export UNIWAM_TEXT_CACHE_DIR=/path/to/text_embedding_cache
export DIFFSYNTH_MODEL_BASE_PATH=/path/to/Wan-and-ActionDiT-checkpoints
export UNIWAM_PIPER_URDF=/path/to/piper_description.urdf
```

These are shell environment inputs, not a script to run unchanged. Replace every
placeholder with a path on the user's machine. The repository does not create a
machine-specific environment file, silently discover datasets, or embed the paths
above in a launcher. The launchers only consume variables that are already set;
`UNIWAM_TRAINING_ROOT` may be derived from the checked-out `training` directory by
the launcher, but data, latent, text-cache, model-weight, sidecar, and URDF paths
must remain explicit.

The default model contract is `proprio_dim=20`, `manip_action_dim=26`, and
`nav_action_dim=3`: 20 robot manipulation values plus six optional tracking
values, and three navigation values `(dx, dy, dtheta)`. The planned navigation
extension is `3 + 4 + 2 = 9` values; see [Action Slots and Feature Masks](docs/ACTION_SLOTS.md).
Actions use H32 horizons and 33 observation frames, with Rot6D camera-frame poses,
q01/q99 normalization, and separate Franka and Piper/AgileX statistics.

The historical six-source recipe is an example of a multi-dataset experiment. Its
source weights are `[3, 1, 1, 1, 1, 7]`, but customer fine-tuning may use a different
number of datasets, a single dataset, and different sampling weights. The names
`agilex7000_camera_frame` and the other historical source paths refer to our
example data and are not required or distributed for customer use.

For a faithful reproduction of the pre-rename six-source 200k run, use
`uniwam_camera_frame_six_source_manip26_embodiment_stats_200k`. It keeps the
parent's source membership, filtered mobile windows, latent remaps, and
Franka/Piper split statistics. The older mobile LeRobot datasets are external;
the newer `action_current_v1_camera_frame` datasets are **not** substitutes.
The launcher checks the parent dataset names, frame counts, index hashes, and
stats hashes before training. It also requires `UNIWAM_PARENT_RECOMPUTED_STATS`
to point to Piper stats rebuilt from the selected episodes and old indexes;
their state/action/contract values must match the parent exactly. This catches
prepared datasets with the right names and frame counts but different camera
poses. Episode contents still cannot be proven byte-identical without the
original episode files.

`uniwam_six_source_retrain_200k` is retained only to inspect the completed
2026-09-28 checkpoint. That run used different mobile windows, reused the
parent statistics, and its three newly converted mobile sources have a
row/column Rot6D serialization error. It is not a name-only reproduction and
is blocked by the two-node launcher.

## Data Pipeline

The pipeline is parameterized by the customer's own LeRobot sources. The six-source
paths and counts in the reference recipe are provenance examples, not requirements.
For a new task, replace them with the customer's camera-frame dataset, indexes,
sidecars, latent cache, and exact prompt cache, then validate the resulting contract.

Start with camera-frame LeRobot episode tables and external EEF sidecars. Build or
restore H32 and phase indexes, latent row remaps, split stats, and prompt caches.
Fill every path in `training/data_pipeline/pipeline_manifest.example.json`.
The stats and prompt builders require a manifest and fail closed when inputs are
absent.
Run `bash training/scripts/bootstrap_reference_artifacts.sh` once after cloning;
it restores the exact parent and retrain index/remap files and checks all 34
file hashes before training.

The mobile conversion script now writes correct row-major Rot6D by default.
Its `--legacy-retrain-rot6d` flag reproduces the malformed 2026-09-28 tables
for auditing only; it must not be used to prepare new training data. See
`training/data_pipeline/README.md` for the distinction.

```bash
python training/data_pipeline/validate_six_source_prepared_data.py \
  --manifest /path/to/pipeline_manifest.json \
  --parent-reference training/data_pipeline/parent_200k_artifact_reference.json \
  --parent-index-root training/reference/indices \
  --output validation_report.json
```

The validator separates structural validity from exact parent-artifact identity.
Use `compare_prepared_episode_tables.py` for numeric episode-value comparison.
VAE tensors and robot calibration data remain external and must be checked in the
environment where they are supplied.

## Training

```bash
cd training
./scripts/bootstrap_reference_artifacts.sh
export PYTHONPATH="$PWD/src:$PWD"
export UNIWAM_TRAINING_ROOT="$PWD"
python scripts/train.py --cfg job --config-name train \
  task=uniwam_camera_frame_six_source_manip26_embodiment_stats_200k
```

For two hosts with eight GPUs per host, set `NODE_RANK`, `MASTER_ADDR`,
`MASTER_PORT`, and `DIFFSYNTH_MODEL_BASE_PATH`, then run
`training/scripts/launch_uniwam_embodiment_stats_2n16.sh`. It defaults to the
parent reproduction recipe and refuses the historical malformed retrain task.
Run the reference-artifact bootstrap once and reproduce the Piper stats before
launching both hosts. The launcher uses
ZeRO-1, bf16, per-GPU batch size 32, compile-enabled denoising/action/VAE
inference, and disables W&B unless explicitly enabled.

## Deployment

Set `UNIWAM_TRAINING_ROOT`, `DIFFSYNTH_MODEL_BASE_PATH`, and `CHECKPOINT`, then
start `deployment/scripts/launch_uniwam_cloud_server.sh` with a source contract.
Set `UNIWAM_PIPER_URDF` to a locally obtained Piper URDF before launching the
cloud server; the URDF is not redistributed in this repository.
The server validates the checkpoint contract and stats content hash before
listening. Use `deployment/scripts/precompute_inference_prompt_async_prefix12_rot6d.sh`
with the exact task sentence before an edge client requests inference.
For the renamed retrain checkpoint, set
`UNIWAM_CLOUD_TASK=uniwam_six_source_retrain_200k`; otherwise the cloud
configs use the earlier parent task by default.

## Compatibility

```python
import uniwam
import fastwam  # legacy compatibility path
```

## Verification

The release is checked with Python compilation, shell syntax checks, Hydra
composition, six-source dataset construction, prompt-prefix consistency, stats
reproduction, H32 index fixtures, archive checksums, and extracted namespace
imports. The customer path also has a raw-to-camera-to-H32 fixture, masked
single-source action sample check, and train/cloud contract checks. Physical
robot execution is not covered by automated tests. A model checkpoint is
intentionally not included.

## License

Original UniWAM contributions are released under the [MIT License](LICENSE).
Bundled third-party code retains its original terms; see
[Third-Party Notices](THIRD_PARTY_NOTICES.md). External model weights, robot
datasets, and the Piper URDF retain their own terms.
