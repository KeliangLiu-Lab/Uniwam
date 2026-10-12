# Prepared-Data Pipeline

This directory contains the reproducible processing stages from prepared
LeRobot episode tables to UniWAM model inputs. The six-source recipe is one
historical example; customer data may contain one or many datasets. It does not embed raw
datasets, videos, VAE tensors, or text encoder caches.

For a new Piper/AgileX customer task, use `prepare_customer_piper_views.py`
and `prepare_customer_piper_h32.py`, then follow
`docs/CUSTOMER_PIPER_FINETUNING.md`. `prepare_agilex_camera_views.py` now uses
the same row-major rotation convention by default. Only the explicit
`--legacy-retrain-rot6d` option reproduces the malformed 2026-09-28 tables.

## Stages

1. Convert legacy state/action representations with the Rot6D and SE(2)
   converters when required by the source contract.
2. Materialize camera-frame state/action columns and build H32 window tables.
   Do not substitute the newer mobile sources for the pre-rename parent run.
3. Route navigation/manipulation phase rows and preserve stable
   `(source_episode_index, start_frame)` keys.
4. Build EEF-XY/visibility sidecars and validate their six-feature masks.
5. Build per-embodiment q01/q99 statistics and a prompt manifest for the
   customer's task set. The historical six-source example has 818 prompt rows;
   this is not a required dataset or prompt count.
6. Generate or remap VAE latent rows and run the fail-closed validator before
   training.

The exact six-source parent indexes and latent row remaps are included under
`training/reference_archives/` as hashed archive parts. Run
`training/scripts/bootstrap_reference_artifacts.sh` to restore them under
`training/reference/`. The episode tables and latent tensors remain external.

## Prepare a new mobile data experiment

Run each source into a new output directory. The converter refuses to replace
an existing output and publishes the directory only after all episodes finish.

```bash
python3 training/data_pipeline/prepare_agilex_camera_views.py \
  --source-root /path/to/cup_tray_action_current_v1 \
  --output-root /path/to/cup_tray_camera_frame \
  --calibration training/data_pipeline/retrain_camera_extrinsics.json
```

Repeat for moving white box and ordinary color blocks. Corrected output is
validated against row-major Rot6D geometry. The completed 2026-09-28 run used
the older, malformed conversion; adding `--legacy-retrain-rot6d` to the command
above reproduces that run's three source tables, but must only be used for
historical comparison. A new experiment needs its own matching H32/phase
indexes, VAE latent cache, EEF sidecars, and recomputed embodiment statistics;
do not reuse the parent's stats or call it a name-only reproduction.

## Manifest

Copy `pipeline_manifest.example.json`, fill every source, sidecar, cache, and
prompt-input path, then use the manifest with the stats and prompt builders.
Those builders require the manifest and never fall back to machine-specific
defaults. `prepare_camera_frame_cross_embodiment_h32.py` similarly requires a
manifest and an empty output directory.

The parent gate accepts logical Parquet equality even when Parquet writer
compression metadata differs. Use `compare_prepared_episode_tables.py --atol
1e-6` for numeric camera-frame comparisons; use its default mode when exact
Arrow equality is required.

## Checks

```bash
python3 -m unittest discover -s training/tests -p 'test_data_pipeline.py'
python3 training/data_pipeline/validate_six_source_prepared_data.py \
  --manifest /path/to/pipeline_manifest.json \
  --parent-reference training/data_pipeline/parent_200k_artifact_reference.json \
  --parent-index-root training/reference/indices \
  --output validation_report.json
```

The validator reports structural validity separately from parent identity. Do
not replace a source with a newer dataset that happens to have the same task
name; source episode membership, phase rows, normalization statistics, and
latent remaps are part of the training contract.
