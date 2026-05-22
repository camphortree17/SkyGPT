#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Compatibility helpers for older pytorch_lightning / torchmetrics combos."""

from typing import Any


def patch_torchmetrics_for_legacy_lightning() -> None:
    # Fix environments where torchmetrics/lightning expect tqdm.auto to exist
    # as an attribute on the top-level tqdm package during import.
    try:
        import importlib
        import tqdm

        tqdm_auto = importlib.import_module("tqdm.auto")
        if not hasattr(tqdm, "auto"):
            tqdm.auto = tqdm_auto
    except Exception:
        pass

    # Fix older Lightning codepaths that still import get_num_classes from
    # torchmetrics.utilities.data when newer torchmetrics may not expose it.
    try:
        import importlib
        tm_data = importlib.import_module("torchmetrics.utilities.data")
        if hasattr(tm_data, "get_num_classes"):
            return

        def get_num_classes(preds: Any, target: Any, num_classes: Any = None) -> int:
            if num_classes is not None:
                try:
                    return int(num_classes)
                except Exception:
                    pass
            try:
                import torch
                if preds is not None:
                    if hasattr(preds, "ndim") and preds.ndim > 1 and hasattr(preds, "shape"):
                        return int(preds.shape[1])
                    return int(torch.max(preds).item()) + 1
            except Exception:
                pass
            try:
                import torch
                if target is not None:
                    return int(torch.max(target).item()) + 1
            except Exception:
                pass
            return 1

        tm_data.get_num_classes = get_num_classes
    except Exception:
        pass
