#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import datetime as dt
from pathlib import Path

import numpy as np
import torch

from utils.config_loader import load_dataset_module
from utils.video_utils import DeterministicEvalDataset, add_repo_to_path, ensure_dir, make_contact_sheet, resolve_best_checkpoint, seed_everything, tensor_cthw_to_uint8

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=["skippd", "folmos", "sirta"])
    args = parser.parse_args()
    cfg = load_dataset_module(args.dataset, "video")

    add_repo_to_path(cfg.SKYGPT_REPO_ROOT)
    from models.video.patched_skygpt import ExternalSkyGPT

    out_dir = ensure_dir(cfg.IMAGE_OUTPUT_DIR / "qualitative_samples")
    vis_dir = ensure_dir(out_dir / "visuals")
    samples_dir = ensure_dir(out_dir / "raw_samples")
    seed_everything(int(cfg.COMMON["seed"]))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_path = resolve_best_checkpoint(cfg.TRANSFORMER_CKPT_DIR)
    model = ExternalSkyGPT.load_from_checkpoint(str(ckpt_path), map_location=device)
    model = model.to(device)
    model.eval()

    dataset = DeterministicEvalDataset(cfg.PREPARED_TEST_EVAL_H5, int(cfg.DATA["sequence_length"]), int(cfg.DATA["cond_frames"]))
    num_cases = min(int(cfg.INFERENCE["sample_num_cases"]), len(dataset))
    num_scenarios = int(cfg.INFERENCE["num_scenarios"])
    cond_frames = int(cfg.DATA["cond_frames"])

    for case_idx in range(num_cases):
        item = dataset[case_idx]
        batch_video = item["video"].unsqueeze(0).repeat(num_scenarios, 1, 1, 1, 1).to(device)
        batch = {"video": batch_video}
        with torch.no_grad():
            generated_full, _, _ = model.sample(num_scenarios, batch=batch)

        gt_uint8 = tensor_cthw_to_uint8(item["video"])
        gt_history = gt_uint8[:cond_frames]
        gt_future = gt_uint8[cond_frames:]

        current_dt = dt.datetime.utcfromtimestamp(int(item["current_ts"].item()))
        target_dt = dt.datetime.utcfromtimestamp(int(item["target_ts"].item()))
        stem = f"case_{case_idx:05d}_t{current_dt.strftime('%Y%m%d_%H%M')}_target_{target_dt.strftime('%Y%m%d_%H%M')}"

        make_contact_sheet(gt_history).save(vis_dir / f"{stem}_history.png")
        make_contact_sheet(gt_future).save(vis_dir / f"{stem}_future_gt.png")

        raw_dir = ensure_dir(samples_dir / stem)
        for s_idx in range(num_scenarios):
            pred_full = tensor_cthw_to_uint8(generated_full[s_idx])
            pred_future = pred_full[-int(cfg.DATA["future_frames"]):]
            make_contact_sheet(pred_future).save(vis_dir / f"{stem}_future_pred_s{s_idx:02d}.png")
            np.save(Path(raw_dir) / f"scenario_{s_idx:02d}.npy", pred_full)

    print(f"Saved qualitative samples to: {out_dir}")

if __name__ == "__main__":
    main()
