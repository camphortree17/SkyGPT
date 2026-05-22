#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from utils.config_loader import load_dataset_module


def calibrate_abs_residual_quantile(y_true: np.ndarray, y_pred: np.ndarray, alpha: float = 0.1) -> float:
    residual = np.abs(np.asarray(y_true, dtype=np.float64) - np.asarray(y_pred, dtype=np.float64))
    return float(np.quantile(residual, 1.0 - alpha))


def build_prediction_interval(y_pred: np.ndarray, q_radius: float, lower_bound: float = 0.0):
    y_pred = np.asarray(y_pred, dtype=np.float64)
    lower = np.maximum(lower_bound, y_pred - q_radius)
    upper = y_pred + q_radius
    return lower, upper


def compute_winkler_score(y_true: np.ndarray, lower: np.ndarray, upper: np.ndarray, alpha: float = 0.1):
    y_true = np.asarray(y_true, dtype=np.float64)
    lower = np.asarray(lower, dtype=np.float64)
    upper = np.asarray(upper, dtype=np.float64)

    width = upper - lower
    below = y_true < lower
    above = y_true > upper
    inside = ~(below | above)

    score = width.copy()
    score[below] = width[below] + (2.0 / alpha) * (lower[below] - y_true[below])
    score[above] = width[above] + (2.0 / alpha) * (y_true[above] - upper[above])

    coverage = np.mean(inside.astype(np.float64))
    sharpness = np.mean(width)
    overall_ws = np.mean(score)

    return {
        "overall_ws": float(overall_ws),
        "overall_coverage": float(coverage),
        "overall_sharpness": float(sharpness),
    }


def _pick_point_pred_column(df: pd.DataFrame) -> str:
    if "pred_mean" in df.columns:
        return "pred_mean"
    if "pred_ghi_tplus15" in df.columns:
        return "pred_ghi_tplus15"
    raise KeyError("No usable point prediction column found. Expected 'pred_mean' or 'pred_ghi_tplus15'.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=["skippd", "folmos"])
    args = parser.parse_args()

    C = load_dataset_module(args.dataset, "pv")

    val_csv = Path(C.VAL_REAL_AGGREGATED_PREDICTIONS_CSV)
    test_csv = Path(C.TEST_GENERATED_AGGREGATED_PREDICTIONS_CSV)
    metrics_json = Path(C.PV_WS_METRICS_JSON)
    metrics_csv = Path(C.PV_WS_METRICS_CSV)
    alpha = float(getattr(C, "PV_WS_ALPHA", 0.1))

    if not val_csv.exists():
        raise FileNotFoundError(f"Val prediction csv not found: {val_csv}")
    if not test_csv.exists():
        raise FileNotFoundError(f"Test prediction csv not found: {test_csv}")

    val_df = pd.read_csv(val_csv)
    test_df = pd.read_csv(test_csv)

    pred_col_val = _pick_point_pred_column(val_df)
    pred_col_test = _pick_point_pred_column(test_df)

    y_val = val_df["true_ghi_tplus15"].to_numpy(dtype=np.float64)
    yhat_val = val_df[pred_col_val].to_numpy(dtype=np.float64)

    y_test = test_df["true_ghi_tplus15"].to_numpy(dtype=np.float64)
    yhat_test = test_df[pred_col_test].to_numpy(dtype=np.float64)

    q_radius = calibrate_abs_residual_quantile(y_val, yhat_val, alpha=alpha)
    lower, upper = build_prediction_interval(yhat_test, q_radius=q_radius, lower_bound=0.0)
    ws = compute_winkler_score(y_test, lower, upper, alpha=alpha)

    rmse = float(np.sqrt(np.mean((y_test - yhat_test) ** 2)))
    mae = float(np.mean(np.abs(y_test - yhat_test)))
    y_max = float(np.max(y_test)) if np.max(y_test) > 0 else 1.0
    nrmse = float(rmse / y_max)
    nmae = float(mae / y_max)

    out = {
        "dataset": args.dataset,
        "alpha": alpha,
        "val_point_prediction_column": pred_col_val,
        "test_point_prediction_column": pred_col_test,
        "q_radius": q_radius,
        "rmse": rmse,
        "mae": mae,
        "nrmse": nrmse,
        "nmae": nmae,
        "overall_ws": float(ws["overall_ws"]),
        "overall_coverage": float(ws["overall_coverage"]),
        "overall_sharpness": float(ws["overall_sharpness"]),
        "val_count": int(len(val_df)),
        "test_count": int(len(test_df)),
        "val_csv": str(val_csv),
        "test_csv": str(test_csv),
    }

    metrics_json.parent.mkdir(parents=True, exist_ok=True)
    metrics_csv.parent.mkdir(parents=True, exist_ok=True)

    with open(metrics_json, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)

    pd.DataFrame([out]).to_csv(metrics_csv, index=False)

    print("=" * 80, flush=True)
    print(json.dumps(out, indent=2, ensure_ascii=False), flush=True)
    print(f"Saved WS metrics json: {metrics_json}", flush=True)
    print(f"Saved WS metrics csv : {metrics_csv}", flush=True)


if __name__ == "__main__":
    main()
