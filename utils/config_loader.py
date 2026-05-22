from __future__ import annotations

import importlib
from pathlib import Path

DATASETS = {"skippd", "folmos", "sirta"}

def package_root() -> Path:
    return Path(__file__).resolve().parents[1]

def load_dataset_module(dataset: str, stage: str):
    dataset = dataset.lower()
    if dataset not in DATASETS:
        raise ValueError(f"Unsupported dataset: {dataset}. Expected one of {sorted(DATASETS)}")
    if stage not in {"video", "pv"}:
        raise ValueError(f"Unsupported stage: {stage}")
    return importlib.import_module(f"configs.{dataset}.{stage}_config")
