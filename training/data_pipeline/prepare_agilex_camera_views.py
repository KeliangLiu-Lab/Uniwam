#!/usr/bin/env python3
"""Materialize the three mobile camera-frame sources used by the retrain recipe."""

from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


def load_transform(value: object, name: str) -> np.ndarray:
    transform = np.asarray(value, dtype=np.float64)
    if transform.shape != (4, 4) or not np.isfinite(transform).all():
        raise ValueError(f"{name} must be a finite 4x4 transform")
    if not np.allclose(transform[3], (0.0, 0.0, 0.0, 1.0), atol=1e-9):
        raise ValueError(f"{name} has an invalid homogeneous row")
    rotation = transform[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-4) or not np.isclose(np.linalg.det(rotation), 1.0, atol=1e-4):
        raise ValueError(f"{name} is not a proper rigid rotation")
    return transform


def rot6d_to_matrix(values: np.ndarray) -> np.ndarray:
    first = values[..., :3]
    second = values[..., 3:6]
    first = first / np.linalg.norm(first, axis=-1, keepdims=True)
    second = second - np.sum(first * second, axis=-1, keepdims=True) * first
    second = second / np.linalg.norm(second, axis=-1, keepdims=True)
    return np.stack((first, second, np.cross(first, second)), axis=-1)


def matrix_to_rot6d(matrix: np.ndarray) -> np.ndarray:
    return np.asarray(matrix[..., :, :2], dtype=np.float32).reshape(-1, 6)


def camera_state(base_arm: np.ndarray, left: np.ndarray, right: np.ndarray) -> np.ndarray:
    output = np.empty((base_arm.shape[0], 20), dtype=np.float32)
    for start, transform in ((0, left), (10, right)):
        position = base_arm[:, start : start + 3]
        rotation = rot6d_to_matrix(base_arm[:, start + 3 : start + 9])
        output[:, start : start + 3] = position @ transform[:3, :3].T + transform[:3, 3]
        output[:, start + 3 : start + 9] = matrix_to_rot6d(transform[:3, :3] @ rotation)
        output[:, start + 9] = base_arm[:, start + 9]
    return output


def fixed_list(values: np.ndarray, width: int) -> pa.Array:
    flat = pa.array(np.asarray(values, dtype=np.float32).reshape(-1), type=pa.float32())
    return pa.FixedSizeListArray.from_arrays(flat, width)


def build_source(source: Path, output: Path, left: np.ndarray, right: np.ndarray) -> None:
    staging = output.with_name(output.name + ".building")
    if output.exists() or staging.exists():
        raise FileExistsError(f"Refusing to overwrite {output} or {staging}")
    paths = sorted((source / "data").glob("chunk-*/*.parquet"))
    if not paths:
        raise FileNotFoundError(f"No LeRobot episode Parquet files under {source / 'data'}")
    (staging / "data").mkdir(parents=True)
    (staging / "meta").mkdir(parents=True)
    (staging / "videos").symlink_to((source / "videos").resolve(), target_is_directory=True)
    for item in (source / "meta").iterdir():
        if item.name not in {"info.json", "stats.json"}:
            (staging / "meta" / item.name).symlink_to(item.resolve(), target_is_directory=item.is_dir())

    info = json.loads((source / "meta/info.json").read_text())
    arm_feature = copy.deepcopy(info["features"]["action.manip"])
    info["features"]["observation.state.camera_dual_arm"] = arm_feature
    info["features"]["action.manip.camera_dual_arm"] = arm_feature
    info["fastwam_camera_frame_conversion"] = {
        "source": str(source),
        "camera_from_left_base": left.tolist(),
        "camera_from_right_base": right.tolist(),
    }
    (staging / "meta/info.json").write_text(json.dumps(info, indent=2) + "\n")

    state_values = []
    action_values = []
    for source_path in paths:
        relative = source_path.relative_to(source / "data")
        target = staging / "data" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        table = pq.read_table(source_path)
        raw_state = np.asarray(table["observation.state"].to_pylist(), dtype=np.float32)
        raw_action = np.asarray(table["action.manip"].to_pylist(), dtype=np.float32)
        if raw_state.shape != (table.num_rows, 23) or raw_action.shape != (table.num_rows, 20):
            raise ValueError(f"Expected state [T,23] and action [T,20] in {source_path}")
        state_camera = camera_state(raw_state[:, 3:], left, right)
        action_camera = camera_state(raw_action, left, right)
        state_values.append(state_camera)
        action_values.append(action_camera)
        table = table.append_column("observation.state.camera_dual_arm", fixed_list(state_camera, 20))
        table = table.append_column("action.manip.camera_dual_arm", fixed_list(action_camera, 20))
        pq.write_table(table, target, compression="zstd")

    stats_path = source / "meta/stats.json"
    stats = json.loads(stats_path.read_text()) if stats_path.is_file() else {}
    for key, values in (
        ("observation.state.camera_dual_arm", np.concatenate(state_values)),
        ("action.manip.camera_dual_arm", np.concatenate(action_values)),
    ):
        stats[key] = {
            "mean": values.mean(axis=0).tolist(),
            "std": values.std(axis=0).tolist(),
            "min": values.min(axis=0).tolist(),
            "max": values.max(axis=0).tolist(),
            "q01": np.quantile(values, 0.01, axis=0).tolist(),
            "q99": np.quantile(values, 0.99, axis=0).tolist(),
        }
    (staging / "meta/stats.json").write_text(json.dumps(stats, indent=2) + "\n")
    os.replace(staging, output)
    print(f"CAMERA_FRAME_SOURCE_READY episodes={len(paths)} output={output}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    args = parser.parse_args()
    calibration = json.loads(args.calibration.read_text())
    left = load_transform(calibration["camera_from_left_base"], "camera_from_left_base")
    right = load_transform(calibration["camera_from_right_base"], "camera_from_right_base")
    build_source(args.source_root, args.output_root, left, right)


if __name__ == "__main__":
    main()
