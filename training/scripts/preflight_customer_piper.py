#!/usr/bin/env python3
"""Check customer Piper SFT inputs and decode one real training sample."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from omegaconf import OmegaConf

from uniwam.datasets.mixed_stream import MixedStreamCollator


ROOT = Path(__file__).resolve().parents[1]
TASK = "uniwam_customer_piper_manip_sft"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def check_parent(checkpoint: Path) -> None:
    if checkpoint.parent.name != "weights" or checkpoint.parent.parent.name != "checkpoints":
        raise ValueError("Parent checkpoint must be under RUN/checkpoints/weights/")
    if not checkpoint.is_file() or checkpoint.stat().st_size < 12_000_000_000:
        raise ValueError("A complete 12 GB+ six-source parent checkpoint is required")
    train_config = checkpoint.parents[2] / "config.yaml"
    parent = OmegaConf.load(train_config)
    model = parent.model
    if int(model.proprio_dim) != 20 or int(model.action_dit_config.manip_action_dim) != 26:
        raise ValueError("Parent checkpoint is not the 20D/26D UniWAM model")
    datasets = parent.data.train.datasets
    if len(datasets) != 6:
        raise ValueError("Expected the six-source split-stats parent checkpoint")
    names = [Path(str(item.dataset.pretrained_norm_stats)).name for item in datasets]
    if names[0] != "camera_frame_franka_only_q01q99.json" or names[1:] != [
        "camera_frame_piper_agx_only_q01q99.json"
    ] * 5:
        raise ValueError(f"Parent is not embodiment-split-stats: {names}")
    reference = json.loads((ROOT / "data_pipeline/parent_200k_artifact_reference.json").read_text())
    stats = ROOT / "data_indices/camera_frame_piper_agx_only_q01q99.json"
    if sha256(stats) != reference["stats_sha256"]["piper_agx"]:
        raise ValueError("Bundled Piper stats differ from the six-source parent reference")


def main() -> None:
    if Path(os.environ["UNIWAM_TRAINING_ROOT"]).resolve() != ROOT:
        raise ValueError("UNIWAM_TRAINING_ROOT must point to this release's training directory")
    with initialize_config_dir(version_base="1.3", config_dir=str(ROOT / "configs")):
        cfg = compose(config_name="train", overrides=[f"task={TASK}"])
    OmegaConf.resolve(cfg)
    check_parent(Path(str(cfg.resume)))
    data = cfg.data.train.datasets[0].dataset
    dataset_root = Path(str(data.dataset_dirs[0]))
    cache_source = Path(str(data.precomputed_latent_root)) / str(data.precomputed_latent_source_name)
    if not cache_source.is_dir():
        raise FileNotFoundError(f"Missing VAE latent source directory: {cache_source}")
    if data.get("manip_eef_xy_index_root") is not None or not data.append_missing_eef_xy_slots:
        raise ValueError("Customer SFT must mask the six auxiliary point-tracking slots")
    if float(cfg.model.loss.lambda_manip_aux_action) != 0:
        raise ValueError("Customer SFT must not supervise the point-tracking auxiliary loss")
    if not bool(cfg.model.disable_manip_aux_training):
        raise ValueError("Customer SFT must disable auxiliary training inputs and noise")
    dataset = instantiate(cfg.data.train)
    sample = dataset[0]
    collate = MixedStreamCollator(
        manip_action_dim=26, nav_action_dim=3, manip_horizon=32, nav_horizon=32,
    )
    batch = collate([sample])
    action = batch["manip_action"]
    mask = batch["manip_action_feature_mask"]
    if tuple(action.shape) != (1, 32, 26) or not bool(mask[..., :20].all()) or bool(mask[..., 20:].any()):
        raise ValueError("Customer action shape or point-tracking feature mask is wrong")
    action_abs_max = float(action[..., :20].abs().max())
    state_abs_max = float(batch["proprio"].abs().max())
    if action_abs_max >= 4.99 or state_abs_max >= 4.99:
        raise ValueError("First sample saturates the [-5,5] normalizer; check units and stats")
    print(json.dumps({
        "status": "CUSTOMER_SFT_INPUTS_OK", "windows": len(dataset),
        "source": str(dataset_root), "checkpoint": str(cfg.resume),
        "stats_sha256": sha256(Path(str(data.pretrained_norm_stats))),
        "first_action_shape": list(action.shape), "aux_supervised": False,
        "first_control_norm_abs_max": action_abs_max,
        "first_state_norm_abs_max": state_abs_max,
    }, sort_keys=True))


if __name__ == "__main__":
    main()
