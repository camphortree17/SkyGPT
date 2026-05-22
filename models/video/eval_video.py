#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import csv
import datetime as dt
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch

from utils.config_loader import load_dataset_module
from utils.video_utils import DeterministicEvalDataset, add_repo_to_path, ensure_dir, resolve_best_checkpoint, save_json, seed_everything, tensor_cthw_to_uint8

def mae_mse(pred: np.ndarray, target: np.ndarray) -> Dict[str, float]:
    pred_f = pred.astype(np.float32) / 255.0
    tgt_f = target.astype(np.float32) / 255.0
    diff = pred_f - tgt_f
    return {"mae": float(np.mean(np.abs(diff))), "mse": float(np.mean(diff ** 2))}

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=["skippd", "folmos", "sirta"])
    args = parser.parse_args()
    cfg = load_dataset_module(args.dataset, "video")

    add_repo_to_path(cfg.SKYGPT_REPO_ROOT)
    from models.video.patched_skygpt import ExternalSkyGPT

    out_dir = ensure_dir(cfg.EVAL_DIR / "video")
    seed_everything(int(cfg.COMMON["seed"]))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt_path = resolve_best_checkpoint(cfg.TRANSFORMER_CKPT_DIR)
    model = ExternalSkyGPT.load_from_checkpoint(str(ckpt_path), map_location=device)
    model = model.to(device)
    model.eval()

    dataset = DeterministicEvalDataset(cfg.PREPARED_TEST_EVAL_H5, int(cfg.DATA["sequence_length"]), int(cfg.DATA["cond_frames"]))
    requested_cases = int(cfg.INFERENCE["eval_num_cases"])
    total_cases = len(dataset) if requested_cases == 0 else min(requested_cases, len(dataset))
    num_scenarios = int(cfg.INFERENCE["num_scenarios"])
    future_len = int(cfg.DATA["future_frames"])

    per_case_rows: List[Dict[str, object]] = []
    scenario_mae: List[float] = []
    scenario_mse: List[float] = []
    best_mae: List[float] = []
    best_mse: List[float] = []

    for idx in range(total_cases):
        item = dataset[idx]
        gt_full = tensor_cthw_to_uint8(item["video"])
        gt_future = gt_full[-future_len:]

        batch_video = item["video"].unsqueeze(0).repeat(num_scenarios, 1, 1, 1, 1).to(device)
        batch = {"video": batch_video}
        with torch.no_grad():
            generated_full, _, _ = model.sample(num_scenarios, batch=batch)

        metrics_this_case = []
        for s_idx in range(num_scenarios):
            pred_full = tensor_cthw_to_uint8(generated_full[s_idx])
            pred_future = pred_full[-future_len:]
            metrics_this_case.append(mae_mse(pred_future, gt_future))

        current_dt = dt.datetime.utcfromtimestamp(int(item["current_ts"].item()))
        target_dt = dt.datetime.utcfromtimestamp(int(item["target_ts"].item()))
        mae_values = [m["mae"] for m in metrics_this_case]
        mse_values = [m["mse"] for m in metrics_this_case]
        scenario_mae.extend(mae_values)
        scenario_mse.extend(mse_values)
        best_mae.append(min(mae_values))
        best_mse.append(min(mse_values))
        per_case_rows.append({
            "case_idx": idx,
            "current_ts": current_dt.strftime("%Y-%m-%d %H:%M:%S"),
            "target_ts": target_dt.strftime("%Y-%m-%d %H:%M:%S"),
            "scenario_mae_mean": float(np.mean(mae_values)),
            "scenario_mse_mean": float(np.mean(mse_values)),
            "best_mae": float(min(mae_values)),
            "best_mse": float(min(mse_values)),
        })

    summary = {
        "dataset": args.dataset,
        "num_cases": int(total_cases),
        "num_scenarios": int(num_scenarios),
        "scenario_mae_mean": float(np.mean(scenario_mae)),
        "scenario_mse_mean": float(np.mean(scenario_mse)),
        "best_mae_mean": float(np.mean(best_mae)),
        "best_mse_mean": float(np.mean(best_mse)),
    }
    save_json(summary, out_dir / "video_metrics_summary.json")
    with open(out_dir / "video_metrics_per_case.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(per_case_rows[0].keys()) if per_case_rows else ["case_idx"])
        writer.writeheader()
        if per_case_rows:
            writer.writerows(per_case_rows)
    print(summary)

if __name__ == "__main__":
    main()
