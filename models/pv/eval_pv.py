#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

from utils.config_loader import load_dataset_module
from utils.pv_utils import empirical_crps, interval_coverage, mae, mse, nmae, nrmse, pinaw, rmse, save_predictions_csv, write_json

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=["skippd", "folmos", "sirta"])
    args = parser.parse_args()
    C = load_dataset_module(args.dataset, "pv")

    agg_df = pd.read_csv(C.AGGREGATED_PREDICTIONS_CSV)
    y_true = agg_df["true_ghi_tplus15"].to_numpy(dtype=np.float32)
    y_mean = agg_df["pred_mean"].to_numpy(dtype=np.float32)
    metrics = {"mae_mean_forecast": mae(y_true, y_mean), "mse_mean_forecast": mse(y_true, y_mean), "rmse_mean_forecast": rmse(y_true, y_mean), "nmae_mean_forecast": nmae(y_true, y_mean), "nrmse_mean_forecast": nrmse(y_true, y_mean)}
    if {"pred_p10", "pred_p50", "pred_p90"}.issubset(set(agg_df.columns)):
        p10 = agg_df["pred_p10"].to_numpy(dtype=np.float32); p50 = agg_df["pred_p50"].to_numpy(dtype=np.float32); p90 = agg_df["pred_p90"].to_numpy(dtype=np.float32)
        metrics.update({"mae_p50": mae(y_true, p50), "coverage_p10_p90": interval_coverage(y_true, p10, p90), "pinaw_p10_p90": pinaw(y_true, p10, p90)})
    scen_df = pd.read_csv(C.SCENARIO_PREDICTIONS_CSV)
    if "scenario_idx" in scen_df.columns and scen_df["scenario_idx"].nunique() > 1:
        pivot = scen_df.pivot(index="sample_idx", columns="scenario_idx", values="pred_ghi_tplus15").sort_index()
        ens = pivot.to_numpy(dtype=np.float32)
        truth = agg_df.sort_values("sample_idx")["true_ghi_tplus15"].to_numpy(dtype=np.float32)
        metrics["crps_ensemble"] = empirical_crps(truth, ens)

    write_json(C.PV_METRICS_JSON, metrics)
    metrics_df = pd.DataFrame({"metric": list(metrics.keys()), "value": list(metrics.values())})
    save_predictions_csv(C.PV_METRICS_CSV, metrics_df)
    print(json.dumps(metrics, indent=2))

if __name__ == "__main__":
    main()
