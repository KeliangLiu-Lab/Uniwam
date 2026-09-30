# Prepared-Data Pipeline

This directory contains the reproducible processing stages from prepared
LeRobot episode tables to the six-source model inputs. It does not embed raw
datasets, videos, VAE tensors, or text encoder caches.

## Stages

1. Convert legacy state/action representations with the Rot6D and SE(2)
   converters when required by the source contract.
2. Materialize camera-frame state/action columns and build H32 window tables.
   For the completed renamed retrain, `prepare_agilex_camera_views.py` uses
   `retrain_camera_extrinsics.json` for the three mobile sources.
3. Route navigation/manipulation phase rows and preserve stable
   `(source_episode_index, start_frame)` keys.
4. Build EEF-XY/visibility sidecars and validate their six-feature masks.
5. Build per-embodiment q01/q99 statistics and the 818-entry prompt manifest.
6. Generate or remap VAE latent rows and run the fail-closed validator before
   training.

The exact six-source parent indexes and latent row remaps are included under
`training/reference/`. They are small reference artifacts; the episode tables
and latent tensors remain external.

## Rebuild the renamed retrain sources

Run each source into a new output directory. The converter refuses to replace
an existing output and publishes the directory only after all episodes finish.

```bash
python3 training/data_pipeline/prepare_agilex_camera_views.py \
  --source-root /path/to/cup_tray_action_current_v1 \
  --output-root /path/to/cup_tray_camera_frame \
  --calibration training/data_pipeline/retrain_camera_extrinsics.json
```

Repeat for moving white box and ordinary color blocks. This exact converter
was checked against the training inputs: 534/534 cup-tray, 477/477 white-box,
and 274/274 color-block episodes had equal camera-frame state/action columns.
Use `compare_prepared_episode_tables.py` to check any independently rebuilt
copy. The six-source `retrain` task uses the matching newer H32/phase indexes
under `training/reference/retrain_indices/` and the moving-white-box remap
under `training/reference/retrain_latent_remap/`.

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
