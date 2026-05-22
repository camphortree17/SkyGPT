#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import json
import random
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, Iterable, List, Optional, Sequence, Union

import h5py
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


def add_repo_to_path(repo_root):
    repo_root_path = Path(repo_root).expanduser().resolve()
    package_root = repo_root_path / "codes" / "video_prediction" / "SkyGPT"
    if not package_root.exists():
        raise FileNotFoundError(
            f"Could not find SkyGPT package root: {package_root}\n"
            f"Please set SKYGPT_REPO_ROOT correctly in project_config.py"
        )
    package_root_str = str(package_root)
    if package_root_str not in sys.path:
        sys.path.insert(0, package_root_str)
    return package_root


def ensure_dir(path):
    p = Path(path).expanduser().resolve()
    p.mkdir(parents=True, exist_ok=True)
    return p


def seed_everything(seed: int = 1234) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    try:
        import pytorch_lightning as pl
        pl.seed_everything(seed)
    except Exception:
        pass


def save_json(data, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def config_to_argv(config: Dict[str, object]) -> List[str]:
    argv: List[str] = []
    for key, value in config.items():
        if value is None:
            continue
        flag = f"--{key}"
        if isinstance(value, bool):
            if value:
                argv.append(flag)
        elif isinstance(value, (list, tuple)):
            argv.append(flag)
            argv.extend([str(v) for v in value])
        else:
            argv.extend([flag, str(value)])
    return argv


def resolve_device_gpus(configured_gpus):
    if configured_gpus is None:
        return 1 if torch.cuda.is_available() else 0
    if configured_gpus > 0 and not torch.cuda.is_available():
        return 0
    return configured_gpus


def resolve_best_checkpoint(ckpt_dir_or_file):
    p = Path(ckpt_dir_or_file).expanduser().resolve()
    if p.is_file():
        if p.suffix != ".ckpt":
            raise ValueError(f"Expected a .ckpt file, got: {p}")
        return p
    if not p.exists():
        raise FileNotFoundError(f"Checkpoint path not found: {p}")

    best_named = p / "best.ckpt"
    if best_named.exists():
        return best_named

    ckpts = sorted(p.rglob("*.ckpt"), key=lambda x: x.stat().st_mtime, reverse=True)
    if not ckpts:
        raise FileNotFoundError(f"No .ckpt files found under: {p}")
    return ckpts[0]


def video_uint8_to_tensor_cthw(video_uint8: np.ndarray) -> torch.Tensor:
    if video_uint8.ndim != 4 or video_uint8.shape[-1] != 3:
        raise ValueError(f"Expected THWC uint8 video, got shape={video_uint8.shape}")
    tensor = torch.from_numpy(video_uint8).float() / 255.0
    tensor = tensor.permute(3, 0, 1, 2).contiguous()
    tensor = tensor - 0.5
    return tensor


def tensor_cthw_to_uint8(video_tensor: torch.Tensor) -> np.ndarray:
    video = video_tensor.detach().cpu().float()
    if video.min().item() < 0.0:
        video = video + 0.5
    video = torch.clamp(video, 0.0, 1.0)
    video = (video * 255.0).round().byte().permute(1, 2, 3, 0).contiguous()
    return video.numpy()


def make_contact_sheet(video_thwc: np.ndarray, pad: int = 2) -> Image.Image:
    if video_thwc.ndim != 4 or video_thwc.shape[-1] != 3:
        raise ValueError(f"Expected THWC uint8 video, got shape={video_thwc.shape}")
    frames = [Image.fromarray(frame) for frame in video_thwc]
    width, height = frames[0].size
    canvas = Image.new("RGB", (len(frames) * width + (len(frames) - 1) * pad, height), color=(255, 255, 255))
    x = 0
    for frame in frames:
        canvas.paste(frame, (x, 0))
        x += width + pad
    return canvas


class DeterministicEvalDataset(Dataset):
    def __init__(self, h5_path, sequence_length, cond_frames):
        self.h5_path = str(Path(h5_path).expanduser().resolve())
        self.sequence_length = int(sequence_length)
        self.cond_frames = int(cond_frames)
        self._h5: Optional[h5py.File] = None
        with h5py.File(self.h5_path, "r") as f:
            self.window_start = f["window_start"][:].astype(np.int64)
            self.current_ts = f["current_ts"][:].astype(np.int64)
            self.target_ts = f["target_ts"][:].astype(np.int64)
            self.length = len(self.window_start)
        if self.length == 0:
            raise ValueError(f"No evaluation windows found in {self.h5_path}")

    def _lazy_open(self) -> None:
        if self._h5 is None:
            self._h5 = h5py.File(self.h5_path, "r")

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        self._lazy_open()
        assert self._h5 is not None
        start = int(self.window_start[idx])
        end = start + self.sequence_length
        video_uint8 = self._h5["videos"][start:end]
        if len(video_uint8) != self.sequence_length:
            raise RuntimeError(
                f"Bad window slice at idx={idx}: expected {self.sequence_length} frames, got {len(video_uint8)}."
            )
        return {
            "video": video_uint8_to_tensor_cthw(video_uint8),
            "current_ts": torch.tensor(int(self.current_ts[idx]), dtype=torch.long),
            "target_ts": torch.tensor(int(self.target_ts[idx]), dtype=torch.long),
            "index": torch.tensor(idx, dtype=torch.long),
        }
