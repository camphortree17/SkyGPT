#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from utils.config_loader import load_dataset_module, package_root
from utils.lightning_compat import patch_torchmetrics_for_legacy_lightning
from utils.video_utils import add_repo_to_path, config_to_argv, ensure_dir, resolve_device_gpus, seed_everything

patch_torchmetrics_for_legacy_lightning()

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint

def main() -> None:
    parser = argparse.ArgumentParser(description="Train VQ-VAE")
    parser.add_argument("--dataset", required=True, choices=["skippd", "folmos"])
    args0 = parser.parse_args()

    cfg = load_dataset_module(args0.dataset, "video")
    add_repo_to_path(cfg.SKYGPT_REPO_ROOT)
    from skygpt.data import VideoData
    from skygpt.vqvae import VQVAE as RepoVQVAE

    ensure_dir(cfg.VQVAE_DIR)
    ensure_dir(cfg.VQVAE_CKPT_DIR)
    seed_everything(int(cfg.COMMON["seed"]))

    parser2 = argparse.ArgumentParser(description="Dataset-specific VQ-VAE training")
    parser2 = pl.Trainer.add_argparse_args(parser2)
    parser2 = RepoVQVAE.add_model_specific_args(parser2)
    parser2.add_argument("--data_path", type=str, default=str(cfg.PREPARED_TRAIN_VAL_H5))
    parser2.add_argument("--sequence_length", type=int, default=int(cfg.VQVAE["sequence_length"]))
    parser2.add_argument("--resolution", type=int, default=int(cfg.VQVAE["resolution"]))
    parser2.add_argument("--batch_size", type=int, default=int(cfg.VQVAE["batch_size"]))
    parser2.add_argument("--num_workers", type=int, default=int(cfg.COMMON["num_workers"]))

    merged = {
        "data_path": str(cfg.PREPARED_TRAIN_VAL_H5),
        "sequence_length": int(cfg.VQVAE["sequence_length"]),
        "resolution": int(cfg.VQVAE["resolution"]),
        "batch_size": int(cfg.VQVAE["batch_size"]),
        "num_workers": int(cfg.COMMON["num_workers"]),
        "downsample": list(cfg.VQVAE["downsample"]),
        "gpus": resolve_device_gpus(int(cfg.COMMON["gpus"])),
        "max_epochs": int(cfg.VQVAE["max_epochs"]),
        "check_val_every_n_epoch": int(cfg.COMMON["check_val_every_n_epoch"]),
        "gradient_clip_val": float(cfg.VQVAE["gradient_clip_val"]),
    }
    args = parser2.parse_args(config_to_argv(merged))
    print("args.sequence_length =", args.sequence_length, flush=True)
    print("args.resolution =", args.resolution, flush=True)
    print("args.downsample =", args.downsample, flush=True)
    print(f"[wrapper] dataset={args0.dataset}", flush=True)
    print(f"[wrapper] train data path: {cfg.PREPARED_TRAIN_VAL_H5}", flush=True)
    print(f"[wrapper] output dir: {cfg.VQVAE_DIR}", flush=True)
    print(f"[wrapper] checkpoint dir: {cfg.VQVAE_CKPT_DIR}", flush=True)

    data = VideoData(args)
    model = RepoVQVAE(args)

    checkpoint_callback = ModelCheckpoint(
        dirpath=str(cfg.VQVAE_CKPT_DIR),
        filename="best",
        monitor="val/recon_loss",
        mode="min",
        save_top_k=1,
        save_last=True,
    )

    trainer = pl.Trainer.from_argparse_args(
        args,
        default_root_dir=str(cfg.VQVAE_DIR),
        callbacks=[checkpoint_callback],
    )
    trainer.fit(model, datamodule=data)

if __name__ == "__main__":
    main()
