#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
from pathlib import Path

from utils.config_loader import load_dataset_module
from utils.lightning_compat import patch_torchmetrics_for_legacy_lightning
from utils.video_utils import add_repo_to_path, config_to_argv, ensure_dir, resolve_device_gpus, seed_everything, resolve_best_checkpoint

patch_torchmetrics_for_legacy_lightning()

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping


def _estimate_steps_per_epoch(data_module):
    try:
        if hasattr(data_module, "prepare_data"):
            data_module.prepare_data()
    except Exception:
        pass
    try:
        if hasattr(data_module, "setup"):
            data_module.setup("fit")
    except TypeError:
        data_module.setup()
    train_loader = data_module.train_dataloader()
    return int(len(train_loader))


def resolve_vqvae_checkpoint(cfg):
    ckpt_dir = Path(cfg.VQVAE_CKPT_DIR)
    best_ckpt = ckpt_dir / "best.ckpt"
    if best_ckpt.exists():
        return best_ckpt
    return resolve_best_checkpoint(ckpt_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Transformer")
    parser.add_argument("--dataset", required=True, choices=["skippd", "folmos", "sirta"])
    args0 = parser.parse_args()

    cfg = load_dataset_module(args0.dataset, "video")
    add_repo_to_path(cfg.SKYGPT_REPO_ROOT)

    from skygpt.data import VideoData
    from models.video.patched_skygpt import ExternalSkyGPT

    ensure_dir(cfg.TRANSFORMER_DIR)
    ensure_dir(cfg.TRANSFORMER_CKPT_DIR)
    seed_everything(int(cfg.COMMON["seed"]))

    vqvae_ckpt = resolve_vqvae_checkpoint(cfg)

    parser2 = argparse.ArgumentParser(description="Dataset-specific Transformer training")
    parser2 = pl.Trainer.add_argparse_args(parser2)
    parser2 = ExternalSkyGPT.add_model_specific_args(parser2)

    parser2.add_argument("--data_path", type=str, default=str(cfg.PREPARED_TRAIN_VAL_H5))
    parser2.add_argument("--resolution", type=int, default=int(cfg.TRANSFORMER["resolution"]))
    parser2.add_argument("--sequence_length", type=int, default=int(cfg.TRANSFORMER["sequence_length"]))
    parser2.add_argument("--batch_size", type=int, default=int(cfg.TRANSFORMER["batch_size"]))
    parser2.add_argument("--num_workers", type=int, default=int(cfg.COMMON["num_workers"]))

    merged = {
        "data_path": str(cfg.PREPARED_TRAIN_VAL_H5),
        "vqvae": str(vqvae_ckpt),
        "resolution": int(cfg.TRANSFORMER["resolution"]),
        "sequence_length": int(cfg.TRANSFORMER["sequence_length"]),
        "batch_size": int(cfg.TRANSFORMER["batch_size"]),
        "num_workers": int(cfg.COMMON["num_workers"]),
        "n_cond_frames": int(cfg.TRANSFORMER["n_cond_frames"]),
        "gpus": resolve_device_gpus(int(cfg.COMMON["gpus"])),
        "default_root_dir": str(cfg.TRANSFORMER_DIR),
        "gradient_clip_val": float(cfg.TRANSFORMER["gradient_clip_val"]),
        "hidden_dim": int(cfg.TRANSFORMER["hidden_dim"]),
        "heads": int(cfg.TRANSFORMER["heads"]),
        "layers": int(cfg.TRANSFORMER["layers"]),
        "dropout": float(cfg.TRANSFORMER["dropout"]),
        "attn_type": str(cfg.TRANSFORMER.get("attn_type", "full")),
        "attn_dropout": float(cfg.TRANSFORMER["attn_dropout"]),
        "max_epochs": int(cfg.TRANSFORMER["max_epochs"]),
        "check_val_every_n_epoch": int(cfg.COMMON["check_val_every_n_epoch"]),
    }

    args = parser2.parse_args(config_to_argv(merged))

    print(f"[wrapper] dataset={args0.dataset}", flush=True)
    print(f"[wrapper] Resolved VQ-VAE checkpoint: {vqvae_ckpt}", flush=True)
    print(f"[wrapper] Train data path: {cfg.PREPARED_TRAIN_VAL_H5}", flush=True)
    print(f"[wrapper] Transformer output dir: {cfg.TRANSFORMER_DIR}", flush=True)
    print(f"[wrapper] Transformer checkpoint dir: {cfg.TRANSFORMER_CKPT_DIR}", flush=True)

    data = VideoData(args)
    steps_per_epoch = _estimate_steps_per_epoch(data)
    max_epochs = int(cfg.TRANSFORMER["max_epochs"])
    max_steps = int(steps_per_epoch * max_epochs)

    args.max_epochs = max_epochs
    args.max_steps = max_steps
    args.class_cond_dim = data.n_classes if getattr(args, "class_cond", False) else None

    model = ExternalSkyGPT(args)

    callbacks = [
        ModelCheckpoint(
            dirpath=str(cfg.TRANSFORMER_CKPT_DIR),
            filename="best",
            monitor="val/loss",
            mode="min",
            save_top_k=1,
            save_last=True,
        )
    ]

    trainer = pl.Trainer.from_argparse_args(
        args,
        callbacks=callbacks,
        max_epochs=max_epochs,
        max_steps=max_steps,
        default_root_dir=str(cfg.TRANSFORMER_DIR),
        gradient_clip_val=float(cfg.TRANSFORMER["gradient_clip_val"]),
    )
    trainer.fit(model, data)


if __name__ == "__main__":
    main()
