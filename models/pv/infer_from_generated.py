#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from utils.config_loader import load_dataset_module
from models.pv.unet_regressor import ModifiedUNetRegressor
from utils.pv_utils import (
    denormalize_targets,
    extract_last_frame_batch,
    load_label_stats,
    maybe_uint8_to_float,
    save_predictions_csv,
)


def _pick_path(C, name: str, default: Path) -> Path:
    return Path(getattr(C, name, default))


def _infer_source_indices_path(C, gen_path: Path):
    p = getattr(C, "GENERATED_SOURCE_INDICES_NPY", None)
    if p is not None:
        p = Path(p)
        if p.exists():
            return p

    name = gen_path.name
    parent = gen_path.parent
    candidates = []
    if name.endswith("_sequences.npy"):
        candidates.append(parent / name.replace("_sequences.npy", "_source_indices.npy"))
    if name.endswith(".npy"):
        candidates.append(parent / name.replace(".npy", "_source_indices.npy"))

    for cand in candidates:
        if cand.exists():
            return cand
    return None


def _build_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _amp_dtype(C):
    name = str(getattr(C, "TORCH_AMP_DTYPE", "bf16")).lower()
    return torch.bfloat16 if name == "bf16" else torch.float16


def _load_model(C, image_shape):
    best_path = _pick_path(C, "TORCH_BEST_MODEL_PATH", C.PV_MODEL_DIR / "best_pv_mapper_torch.pt")
    device = _build_device()
    ckpt = torch.load(best_path, map_location=device, weights_only=False)
    cfg = ckpt.get("config", {})

    model = ModifiedUNetRegressor(
        in_channels=int(image_shape[-1]),
        base_channels=int(cfg.get("UNET_BASE_CHANNELS", getattr(C, "UNET_BASE_CHANNELS", 32))),
        bottleneck_res_blocks=int(cfg.get("UNET_BOTTLENECK_RES_BLOCKS", getattr(C, "UNET_BOTTLENECK_RES_BLOCKS", 2))),
        dropout=float(cfg.get("UNET_DROPOUT", getattr(C, "UNET_DROPOUT", 0.0))),
    ).to(device)

    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, device


def _batched_predict(model, device, frames: np.ndarray, batch_size: int, amp_enabled: bool, amp_dtype) -> np.ndarray:
    preds = []
    for start in tqdm(range(0, len(frames), batch_size), desc="PV infer", leave=False):
        batch = maybe_uint8_to_float(frames[start:start + batch_size])
        batch = np.transpose(batch, (0, 3, 1, 2))
        x = torch.from_numpy(batch).float().to(device, non_blocking=True)

        with torch.no_grad():
            with torch.autocast(
                device_type="cuda",
                dtype=amp_dtype,
                enabled=amp_enabled and device.type == "cuda",
            ):
                pred = model(x)

        preds.append(pred.detach().float().cpu().numpy())

    return np.concatenate(preds).astype(np.float32)


def _filter_invalid_source_indices(source_indices: np.ndarray, last_frames: np.ndarray, n_windows: int):
    valid_mask = (source_indices >= 0) & (source_indices < n_windows)
    invalid_count = int((~valid_mask).sum())

    if invalid_count > 0:
        invalid_vals = source_indices[~valid_mask]
        preview = invalid_vals[:10].tolist()
        print(
            f"[warn] Found {invalid_count} invalid source_indices "
            f"(valid range: 0 ~ {n_windows - 1}). First invalid values: {preview}",
            flush=True,
        )

    if invalid_count == 0:
        return source_indices, last_frames

    if len(source_indices) != last_frames.shape[0]:
        raise ValueError(
            f"source_indices length {len(source_indices)} != generated samples {last_frames.shape[0]}"
        )

    filtered_indices = source_indices[valid_mask]
    if last_frames.ndim == 4:
        filtered_frames = last_frames[valid_mask]
    elif last_frames.ndim == 5:
        filtered_frames = last_frames[valid_mask, ...]
    else:
        raise ValueError(f"Unexpected last_frames ndim: {last_frames.ndim}")

    print(
        f"[warn] After filtering invalid source_indices: kept {len(filtered_indices)} / {len(source_indices)} samples",
        flush=True,
    )
    return filtered_indices, filtered_frames


def _resolve_output_paths(C, split: str, input_mode: str):
    if split == "val" and input_mode == "real":
        scenario_path = Path(getattr(C, "VAL_REAL_SCENARIO_PREDICTIONS_CSV"))
        agg_path = Path(getattr(C, "VAL_REAL_AGGREGATED_PREDICTIONS_CSV"))
    elif split == "test" and input_mode == "generated":
        scenario_path = Path(getattr(C, "TEST_GENERATED_SCENARIO_PREDICTIONS_CSV"))
        agg_path = Path(getattr(C, "TEST_GENERATED_AGGREGATED_PREDICTIONS_CSV"))
    else:
        raise ValueError(f"Unsupported split/input_mode combination: {split}/{input_mode}")
    scenario_path.parent.mkdir(parents=True, exist_ok=True)
    agg_path.parent.mkdir(parents=True, exist_ok=True)
    return scenario_path, agg_path


def _load_val_real_inputs(C):
    with h5py.File(C.PV_MAPPING_H5, "r") as f:
        grp = f["val"]
        images = grp["images"][:]
        y_true_t15 = grp["target_ghi_15min"][:].astype(np.float32)
        y_true_t = grp["irr_ghi"][:].astype(np.float32)
        ts_t = np.array([x.decode("utf-8") if isinstance(x, bytes) else str(x) for x in grp["timestamp_str"][:]], dtype=object)
        image_shape = tuple(f["train"]["images"].shape[1:])

    n = len(y_true_t15)
    source_window_index = np.arange(n, dtype=np.int64)
    ts_t15 = ts_t.copy()
    return images, source_window_index, ts_t, ts_t15, y_true_t15, y_true_t, image_shape


def _load_test_generated_inputs(C):
    gen_path = Path(C.GENERATED_SEQUENCES_NPY)
    gen = np.load(gen_path, mmap_mode="r")
    last_frames = extract_last_frame_batch(gen, int(getattr(C, "GENERATED_LAST_FRAME_INDEX", -1)))

    src_idx_path = _infer_source_indices_path(C, gen_path)
    source_indices = np.load(src_idx_path).astype(np.int64) if src_idx_path is not None else None

    with h5py.File(C.PV_MAPPING_H5, "r") as f:
        grp = f["test"]["window_eval"]
        source_window_index = grp["source_window_index"][:].astype(np.int64)
        ts_t = np.array([x.decode("utf-8") if isinstance(x, bytes) else str(x) for x in grp["timestamp_t_str"][:]], dtype=object)
        ts_t15 = np.array([x.decode("utf-8") if isinstance(x, bytes) else str(x) for x in grp["timestamp_tplus15_str"][:]], dtype=object)
        y_true_t15 = grp["target_ghi_tplus15"][:].astype(np.float32)
        y_true_t = grp["irr_ghi_t"][:].astype(np.float32)
        image_shape = tuple(f["train"]["images"].shape[1:])

    n_windows = len(source_window_index)

    print(f"[info] Generated frames shape: {last_frames.shape}", flush=True)
    print(f"[info] window_eval count: {n_windows}", flush=True)
    print(f"[info] source_indices path: {src_idx_path}", flush=True)

    if source_indices is not None:
        print(
            f"[info] source_indices length: {len(source_indices)}, min={source_indices.min()}, max={source_indices.max()}",
            flush=True,
        )
        source_indices, last_frames = _filter_invalid_source_indices(source_indices, last_frames, n_windows)
        align_idx = source_indices
    else:
        if last_frames.shape[0] != n_windows:
            raise ValueError(
                f"Generated sample count {last_frames.shape[0]} != test window count {n_windows}, and no source_indices file found."
            )
        align_idx = np.arange(n_windows, dtype=np.int64)

    ts_t = ts_t[align_idx]
    ts_t15 = ts_t15[align_idx]
    y_true_t15 = y_true_t15[align_idx]
    y_true_t = y_true_t[align_idx]
    aligned_source_window_index = source_window_index[align_idx]

    return last_frames, aligned_source_window_index, ts_t, ts_t15, y_true_t15, y_true_t, image_shape


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=["skippd", "folmos"])
    parser.add_argument("--split", required=True, choices=["val", "test"])
    parser.add_argument("--input_mode", required=True, choices=["real", "generated"])
    args = parser.parse_args()

    C = load_dataset_module(args.dataset, "pv")
    stats = load_label_stats(C.NORMALIZATION_STATS_JSON)

    if args.split == "val" and args.input_mode == "real":
        frames, aligned_source_window_index, ts_t, ts_t15, y_true_t15, y_true_t, image_shape = _load_val_real_inputs(C)
    elif args.split == "test" and args.input_mode == "generated":
        frames, aligned_source_window_index, ts_t, ts_t15, y_true_t15, y_true_t, image_shape = _load_test_generated_inputs(C)
    else:
        raise ValueError(
            f"Unsupported combination: split={args.split}, input_mode={args.input_mode}. Use val+real or test+generated."
        )

    scenario_csv, agg_csv = _resolve_output_paths(C, args.split, args.input_mode)

    model, device = _load_model(C, image_shape)
    amp_enabled = bool(getattr(C, "TORCH_USE_AMP", True))
    amp_dtype = _amp_dtype(C)
    batch_size = int(getattr(C, "TEST_BATCH_SIZE", 64))

    if frames.ndim == 4:
        n = frames.shape[0]
        preds_norm = _batched_predict(model, device, frames, batch_size, amp_enabled, amp_dtype).reshape(-1)
        preds = denormalize_targets(preds_norm, stats)

        scenario_df = pd.DataFrame({
            "sample_idx": np.arange(n, dtype=np.int64),
            "source_window_index": aligned_source_window_index,
            "scenario_idx": np.zeros(n, dtype=np.int64),
            "timestamp_t": ts_t,
            "timestamp_tplus15": ts_t15,
            "pred_ghi_tplus15": preds.astype(np.float32),
            "true_ghi_tplus15": y_true_t15,
            "true_ghi_t": y_true_t,
        })

        agg_df = scenario_df.copy()
        agg_df["pred_mean"] = agg_df["pred_ghi_tplus15"]
        agg_df["pred_p10"] = agg_df["pred_ghi_tplus15"]
        agg_df["pred_p50"] = agg_df["pred_ghi_tplus15"]
        agg_df["pred_p90"] = agg_df["pred_ghi_tplus15"]

    elif frames.ndim == 5:
        n, k = frames.shape[:2]
        preds_all = np.zeros((n, k), dtype=np.float32)
        rows = []

        for s in range(k):
            preds_norm = _batched_predict(
                model, device, frames[:, s, :, :, :], batch_size, amp_enabled, amp_dtype
            ).reshape(-1)
            preds = denormalize_targets(preds_norm, stats)
            preds_all[:, s] = preds

            rows.append(pd.DataFrame({
                "sample_idx": np.arange(n, dtype=np.int64),
                "source_window_index": aligned_source_window_index,
                "scenario_idx": np.full(n, s, dtype=np.int64),
                "timestamp_t": ts_t,
                "timestamp_tplus15": ts_t15,
                "pred_ghi_tplus15": preds.astype(np.float32),
                "true_ghi_tplus15": y_true_t15,
                "true_ghi_t": y_true_t,
            }))

        scenario_df = pd.concat(rows, axis=0, ignore_index=True)

        agg_df = pd.DataFrame({
            "sample_idx": np.arange(n, dtype=np.int64),
            "source_window_index": aligned_source_window_index,
            "timestamp_t": ts_t,
            "timestamp_tplus15": ts_t15,
            "true_ghi_tplus15": y_true_t15,
            "true_ghi_t": y_true_t,
            "pred_mean": np.mean(preds_all, axis=1),
            "pred_p10": np.percentile(preds_all, 10, axis=1),
            "pred_p50": np.percentile(preds_all, 50, axis=1),
            "pred_p90": np.percentile(preds_all, 90, axis=1),
        })

    else:
        raise ValueError(f"Unexpected frames ndim: {frames.ndim}")

    save_predictions_csv(scenario_csv, scenario_df)
    save_predictions_csv(agg_csv, agg_df)

    print("Saved scenario predictions :", scenario_csv, flush=True)
    print("Saved aggregated predictions:", agg_csv, flush=True)


if __name__ == "__main__":
    main()
