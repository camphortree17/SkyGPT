#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from utils.config_loader import load_dataset_module
from models.pv.unet_regressor import ModifiedUNetRegressor
from utils.pv_utils import (
    H5ImageRegressionDataset,
    denormalize_targets,
    ensure_dir,
    mae,
    mse,
    nmae,
    rmse,
    save_predictions_csv,
    set_global_seed,
    write_json,
)

def _pick_path(C, name: str, default: Path) -> Path:
    return Path(getattr(C, name, default))

def _build_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def _amp_dtype(C):
    name = str(getattr(C, "TORCH_AMP_DTYPE", "bf16")).lower()
    return torch.bfloat16 if name == "bf16" else torch.float16

def _derive_paths(C):
    return (
        _pick_path(C, "TORCH_BEST_MODEL_PATH", C.PV_MODEL_DIR / "best_pv_mapper_torch.pt"),
        _pick_path(C, "TORCH_LAST_MODEL_PATH", C.PV_MODEL_DIR / "last_pv_mapper_torch.pt"),
        _pick_path(C, "TORCH_TRAIN_HISTORY_CSV", C.PV_LOG_DIR / "train_history_torch.csv"),
        _pick_path(C, "TORCH_CURRENT_METRICS_JSON", C.PV_EVAL_DIR / "concurrent_pv_mapping_metrics_torch.json"),
        _pick_path(C, "TORCH_CURRENT_PREDICTIONS_CSV", C.PV_PRED_DIR / "current_time_mapping_predictions_torch.csv"),
    )

def log(msg: str):
    print(msg, flush=True)

def evaluate_regression(model, loader, device, stats, amp_enabled: bool, amp_dtype):
    model.eval()
    preds, truths, timestamps = [], [], []
    with torch.no_grad():
        for batch in loader:
            x = batch["image"].to(device, non_blocking=True)
            y = batch["label"].to(device, non_blocking=True)
            with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=amp_enabled and device.type == "cuda"):
                pred = model(x)
            preds.append(pred.detach().float().cpu().numpy())
            truths.append(y.detach().float().cpu().numpy())
            timestamps.extend(batch["timestamp"])
    preds = np.concatenate(preds).astype(np.float32)
    truths = np.concatenate(truths).astype(np.float32)
    return denormalize_targets(preds, stats), denormalize_targets(truths, stats), timestamps

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=["skippd", "folmos"])
    args = parser.parse_args()
    C = load_dataset_module(args.dataset, "pv")

    set_global_seed(int(C.SEED))
    ensure_dir(C.PV_MODEL_DIR); ensure_dir(C.PV_LOG_DIR); ensure_dir(C.PV_PRED_DIR); ensure_dir(C.PV_EVAL_DIR)

    best_path, last_path, history_csv, metrics_json, pred_csv = _derive_paths(C)
    device = _build_device()
    amp_enabled = bool(getattr(C, "TORCH_USE_AMP", True))
    amp_dtype = _amp_dtype(C)
    log_interval = int(getattr(C, "TORCH_LOG_INTERVAL", 200))

    with h5py.File(C.PV_MAPPING_H5, "r") as f:
        train_y = f["train"]["irr_ghi"][:].astype(np.float32)
        val_y = f["val"]["irr_ghi"][:].astype(np.float32)
        test_y = f["test"]["irr_ghi"][:].astype(np.float32)
        image_shape = tuple(f["train"]["images"].shape[1:])
        test_target_t15 = f["test"]["target_ghi_15min"][:].astype(np.float32)

    if C.LABEL_NORMALIZATION == "zscore":
        mean = float(np.mean(train_y))
        std = max(float(np.std(train_y)), 1e-8)
    else:
        mean, std = 0.0, 1.0
    stats = {"mode": C.LABEL_NORMALIZATION, "mean": mean, "std": std, "train_count": int(len(train_y)), "val_count": int(len(val_y)), "test_count": int(len(test_y))}
    write_json(C.NORMALIZATION_STATS_JSON, stats)

    log("[fast] loading train/val/test splits from H5 into RAM ...")
    t0 = time.time()
    train_ds = H5ImageRegressionDataset(C.PV_MAPPING_H5, "train", stats, preload_to_ram=True)
    val_ds = H5ImageRegressionDataset(C.PV_MAPPING_H5, "val", stats, preload_to_ram=True)
    test_ds = H5ImageRegressionDataset(C.PV_MAPPING_H5, "test", stats, preload_to_ram=True)
    log(f"[fast] RAM preload done in {time.time() - t0:.1f}s")

    num_workers = int(getattr(C, "TORCH_NUM_WORKERS", 0))
    pin = device.type == "cuda"
    persistent = num_workers > 0

    train_loader = DataLoader(train_ds, batch_size=int(C.TRAIN_BATCH_SIZE), shuffle=bool(C.SHUFFLE_TRAIN), num_workers=num_workers, pin_memory=pin, persistent_workers=persistent, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=int(C.VAL_BATCH_SIZE), shuffle=False, num_workers=num_workers, pin_memory=pin, persistent_workers=persistent, drop_last=False)
    test_loader = DataLoader(test_ds, batch_size=int(C.TEST_BATCH_SIZE), shuffle=False, num_workers=num_workers, pin_memory=pin, persistent_workers=persistent, drop_last=False)

    model = ModifiedUNetRegressor(in_channels=int(image_shape[2]), base_channels=int(C.UNET_BASE_CHANNELS), bottleneck_res_blocks=int(C.UNET_BOTTLENECK_RES_BLOCKS), dropout=float(C.UNET_DROPOUT)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(C.LEARNING_RATE), weight_decay=float(getattr(C, "WEIGHT_DECAY", 0.0)))
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=float(C.REDUCE_LR_FACTOR), patience=int(C.REDUCE_LR_PATIENCE), min_lr=1e-7)
    scaler = torch.amp.GradScaler(enabled=(amp_enabled and device.type == "cuda" and amp_dtype == torch.float16))
    criterion = torch.nn.MSELoss()

    best_val, best_epoch, bad_epochs = float("inf"), -1, 0
    min_delta = float(C.MIN_DELTA)

    with open(history_csv, "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(["epoch", "train_loss", "val_loss", "train_mae_denorm", "val_mae_denorm", "lr", "epoch_sec"])

    for epoch in range(1, int(C.NUM_EPOCHS) + 1):
        epoch_t0 = time.time()
        model.train()
        train_losses, train_pred_chunks, train_true_chunks = [], [], []
        for step, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch}/{int(C.NUM_EPOCHS)} [train]", leave=False)):
            x = batch["image"].to(device, non_blocking=True)
            y = batch["label"].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=amp_enabled and device.type == "cuda"):
                pred = model(x)
                loss = criterion(pred, y)
            if scaler.is_enabled():
                scaler.scale(loss).backward(); scaler.step(optimizer); scaler.update()
            else:
                loss.backward(); optimizer.step()
            train_losses.append(loss.detach().item())
            train_pred_chunks.append(pred.detach().float().cpu()); train_true_chunks.append(y.detach().float().cpu())
            if (step + 1) % log_interval == 0:
                log(f"[epoch {epoch:03d}] train step {step + 1}/{len(train_loader)} loss={float(np.mean(train_losses[-log_interval:])):.6f}")

        train_loss = float(np.mean(train_losses))
        train_pred_denorm = denormalize_targets(torch.cat(train_pred_chunks).numpy(), stats)
        train_true_denorm = denormalize_targets(torch.cat(train_true_chunks).numpy(), stats)
        train_mae_denorm = mae(train_true_denorm, train_pred_denorm)

        model.eval()
        val_losses, val_pred_chunks, val_true_chunks = [], [], []
        with torch.no_grad():
            for step, batch in enumerate(tqdm(val_loader, desc=f"Epoch {epoch}/{int(C.NUM_EPOCHS)} [val]", leave=False)):
                x = batch["image"].to(device, non_blocking=True); y = batch["label"].to(device, non_blocking=True)
                with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=amp_enabled and device.type == "cuda"):
                    pred = model(x); loss = criterion(pred, y)
                val_losses.append(loss.detach().item()); val_pred_chunks.append(pred.detach().float().cpu()); val_true_chunks.append(y.detach().float().cpu())

        val_loss = float(np.mean(val_losses))
        val_pred_denorm = denormalize_targets(torch.cat(val_pred_chunks).numpy(), stats)
        val_true_denorm = denormalize_targets(torch.cat(val_true_chunks).numpy(), stats)
        val_mae_denorm = mae(val_true_denorm, val_pred_denorm)
        scheduler.step(val_loss)
        lr = float(optimizer.param_groups[0]["lr"])
        epoch_sec = float(time.time() - epoch_t0)

        with open(history_csv, "a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow([epoch, train_loss, val_loss, train_mae_denorm, val_mae_denorm, lr, epoch_sec])

        ckpt = {"epoch": epoch, "model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict(), "stats": stats, "image_shape": image_shape, "config": {"UNET_BASE_CHANNELS": int(C.UNET_BASE_CHANNELS), "UNET_BOTTLENECK_RES_BLOCKS": int(C.UNET_BOTTLENECK_RES_BLOCKS), "UNET_DROPOUT": float(C.UNET_DROPOUT)}, "val_loss": val_loss}
        torch.save(ckpt, last_path)
        if val_loss < best_val - min_delta:
            best_val, best_epoch, bad_epochs = val_loss, epoch, 0
            torch.save(ckpt, best_path)
            log(f"[checkpoint] new best saved to: {best_path}")
        else:
            bad_epochs += 1
            if bad_epochs >= int(C.EARLY_STOPPING_PATIENCE):
                log(f"[early-stop] patience reached at epoch {epoch}, best_epoch={best_epoch}, best_val={best_val:.6f}")
                break

    best_ckpt = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(best_ckpt["model_state_dict"])
    preds, truths, timestamps = evaluate_regression(model, test_loader, device, stats, amp_enabled, amp_dtype)
    test_metrics = {"test_mae_current": mae(truths, preds), "test_mse_current": mse(truths, preds), "test_rmse_current": rmse(truths, preds), "test_nmae_current": nmae(truths, preds)}
    write_json(metrics_json, test_metrics)

    import pandas as pd
    out_df = pd.DataFrame({"timestamp": timestamps, "pred_irr_ghi_t": preds.astype(np.float32), "true_irr_ghi_t": truths.astype(np.float32), "target_ghi_15min": test_target_t15.astype(np.float32)})
    save_predictions_csv(pred_csv, out_df)
    log("Concurrent test metrics: " + json.dumps(test_metrics, indent=2))

if __name__ == "__main__":
    main()
