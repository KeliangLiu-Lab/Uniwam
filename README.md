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

## Environment

Use Python 3.10+, PyTorch, Hydra, PyArrow, NumPy, and the dependencies required by
the training and deployment environments. Set all paths to local assets explicitly:

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

The model contract is `proprio_dim=20`, `manip_action_dim=26`, `nav_action_dim=3`,
H32 actions, 33 observation frames, Rot6D camera-frame poses, q01/q99
normalization, and separate Franka and Piper/AgileX statistics. Source sampling
weights are `[3, 1, 1, 1, 1, 7]`.

Two data recipes are intentionally separate. `uniwam_six_source_retrain_200k`
matches the renamed training run completed in September 2026 and uses the
full newer mobile sources. `uniwam_camera_frame_six_source_manip26_embodiment_stats_200k`
keeps the earlier split-stats parent checkpoint's filtered mobile indexes.
Their checkpoints and mobile data indexes must not be interchanged.

## Data Pipeline

Start with camera-frame LeRobot episode tables and external EEF sidecars. Build or
restore H32 and phase indexes, latent row remaps, split stats, and prompt caches.
Fill every path in `training/data_pipeline/pipeline_manifest.example.json`.
The stats and prompt builders require a manifest and fail closed when inputs are
absent.

For the completed `retrain` recipe, the three mobile camera-frame datasets can
be rebuilt from the action-current LeRobot sources with
`training/data_pipeline/prepare_agilex_camera_views.py` and the shipped
`retrain_camera_extrinsics.json`. All 1,285 mobile episode state/action tables
were independently rebuilt and compared to the completed training run with
strict Arrow equality. See `training/data_pipeline/README.md` for commands.

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
export PYTHONPATH="$PWD/src:$PWD"
export UNIWAM_TRAINING_ROOT="$PWD"
python scripts/train.py --cfg job --config-name train \
  task=uniwam_six_source_retrain_200k
```

For two hosts with eight GPUs per host, set `NODE_RANK`, `MASTER_ADDR`,
`MASTER_PORT`, and `DIFFSYNTH_MODEL_BASE_PATH`, then run
`training/scripts/launch_uniwam_embodiment_stats_2n16.sh`. It defaults to the
renamed `retrain` recipe; set `TASK=uniwam_camera_frame_six_source_manip26_embodiment_stats_200k`
only with that earlier parent's prepared data. The launcher uses
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
imports. A model checkpoint is intentionally not included.

## License

Original UniWAM contributions are released under the [MIT License](LICENSE).
Bundled third-party code retains its original terms; see
[Third-Party Notices](THIRD_PARTY_NOTICES.md). External model weights, robot
datasets, and the Piper URDF retain their own terms.
