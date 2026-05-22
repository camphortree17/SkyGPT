#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import json
import random
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_json(path: Path, obj) -> None:
    ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def save_predictions_csv(path: Path, df: pd.DataFrame) -> None:
    ensure_dir(path.parent)
    df.to_csv(path, index=False, encoding="utf-8")


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def mae(y_true, y_pred) -> float:
    y_true = np.asarray(y_true, dtype=np.float32)
    y_pred = np.asarray(y_pred, dtype=np.float32)
    return float(np.mean(np.abs(y_true - y_pred)))


def mse(y_true, y_pred) -> float:
    y_true = np.asarray(y_true, dtype=np.float32)
    y_pred = np.asarray(y_pred, dtype=np.float32)
    return float(np.mean((y_true - y_pred) ** 2))


def rmse(y_true, y_pred) -> float:
    return float(np.sqrt(mse(y_true, y_pred)))


def nmae(y_true, y_pred) -> float:
    y_true = np.asarray(y_true, dtype=np.float32)
    denom = float(np.max(np.abs(y_true)) - np.min(np.abs(y_true)))
    if denom <= 1e-8:
        denom = float(np.max(np.abs(y_true)))
    return float(mae(y_true, y_pred) / max(denom, 1e-8))


def nrmse(y_true, y_pred) -> float:
    y_true = np.asarray(y_true, dtype=np.float32)
    denom = float(np.max(np.abs(y_true)) - np.min(np.abs(y_true)))
    if denom <= 1e-8:
        denom = float(np.max(np.abs(y_true)))
    return float(rmse(y_true, y_pred) / max(denom, 1e-8))


def interval_coverage(y_true, lower, upper) -> float:
    y_true = np.asarray(y_true, dtype=np.float32)
    lower = np.asarray(lower, dtype=np.float32)
    upper = np.asarray(upper, dtype=np.float32)
    return float(np.mean((y_true >= lower) & (y_true <= upper)))


def pinaw(y_true, lower, upper) -> float:
    y_true = np.asarray(y_true, dtype=np.float32)
    width = np.asarray(upper, dtype=np.float32) - np.asarray(lower, dtype=np.float32)
    denom = max(float(np.max(y_true) - np.min(y_true)), 1e-8)
    return float(np.mean(width) / denom)


def empirical_crps(y_true, ensemble) -> float:
    y_true = np.asarray(y_true, dtype=np.float32).reshape(-1, 1)
    ensemble = np.asarray(ensemble, dtype=np.float32)
    term1 = np.mean(np.abs(ensemble - y_true), axis=1)
    pairwise = np.abs(ensemble[:, :, None] - ensemble[:, None, :])
    term2 = 0.5 * np.mean(pairwise, axis=(1, 2))
    return float(np.mean(term1 - term2))


def normalize_targets(y, stats: dict) -> np.ndarray:
    y = np.asarray(y, dtype=np.float32)
    if stats["mode"] == "zscore":
        return (y - float(stats["mean"])) / max(float(stats["std"]), 1e-8)
    return y


def denormalize_targets(y, stats: dict, mode=None) -> np.ndarray:
    y = np.asarray(y, dtype=np.float32)
    mode = stats["mode"] if mode is None else mode
    if mode == "zscore":
        return y * float(stats["std"]) + float(stats["mean"])
    return y


def load_label_stats(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def maybe_uint8_to_float(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x)
    if x.dtype == np.uint8:
        return x.astype(np.float32) / 255.0
    x = x.astype(np.float32)
    if np.max(x) > 1.5:
        x = x / 255.0
    return x


def extract_last_frame_batch(arr: np.ndarray, last_frame_index: int = -1) -> np.ndarray:
    arr = np.asarray(arr)
    if arr.ndim == 4:
        if arr.shape[-1] in (1, 3):
            return arr
        if arr.shape[1] in (1, 3):
            return np.transpose(arr, (0, 2, 3, 1))
        raise ValueError(f"Unsupported 4D shape: {arr.shape}")
    if arr.ndim == 5:
        if arr.shape[-1] in (1, 3):
            if arr.shape[1] == 15:
                return arr[:, last_frame_index, :, :, :]
            return arr
        if arr.shape[2] in (1, 3):
            if arr.shape[1] == 15:
                return np.transpose(arr[:, last_frame_index, :, :, :], (0, 2, 3, 1))
            return np.transpose(arr, (0, 1, 3, 4, 2))
        raise ValueError(f"Unsupported 5D shape: {arr.shape}")
    if arr.ndim == 6:
        if arr.shape[-1] in (1, 3):
            return arr[:, :, last_frame_index, :, :, :]
        if arr.shape[3] in (1, 3):
            return np.transpose(arr[:, :, last_frame_index, :, :, :], (0, 1, 3, 4, 2))
        raise ValueError(f"Unsupported 6D shape: {arr.shape}")
    raise ValueError(f"Unsupported ndim: {arr.ndim}, shape={arr.shape}")


class H5ImageRegressionDataset(Dataset):
    def __init__(
        self,
        h5_path: Path,
        split: str,
        label_stats: dict,
        image_key: str = "images",
        label_key: str = "irr_ghi",
        preload_to_ram: bool = True,
    ):
        self.h5_path = str(h5_path)
        self.split = split
        self.label_stats = label_stats
        self.image_key = image_key
        self.label_key = label_key

        with h5py.File(self.h5_path, "r") as f:
            grp = f[split]
            if preload_to_ram:
                images = grp[image_key][:]
                labels = grp[label_key][:]
                ts = grp["timestamp_str"][:] if "timestamp_str" in grp else None
            else:
                raise RuntimeError("This dataset version is designed for preload_to_ram=True.")

        images = maybe_uint8_to_float(images)
        images = np.transpose(images, (0, 3, 1, 2)).astype(np.float32, copy=False)
        labels = normalize_targets(labels, label_stats).astype(np.float32, copy=False)

        if ts is not None:
            timestamps = np.array([x.decode("utf-8") if isinstance(x, bytes) else str(x) for x in ts], dtype=object)
        else:
            timestamps = None

        self.images = images
        self.labels = labels
        self.timestamps = timestamps
        self.length = int(len(self.labels))

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        ts = None if self.timestamps is None else self.timestamps[idx]
        return {
            "image": torch.from_numpy(self.images[idx]),
            "label": torch.tensor(self.labels[idx], dtype=torch.float32),
            "index": torch.tensor(idx, dtype=torch.long),
            "timestamp": ts,
        }


from PIL import Image
import pandas as pd

def read_resize_rgb(image_path: Path, image_size: int) -> np.ndarray:
    img = Image.open(image_path).convert("RGB")
    img = img.resize((image_size, image_size), Image.BILINEAR)
    return np.asarray(img, dtype=np.uint8)

def decode_bytes_array(arr):
    return np.array([x.decode("utf-8") if isinstance(x, bytes) else str(x) for x in arr], dtype=object)
