#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import itertools
import json
import math
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from numpy.lib.format import open_memmap
from torch.utils.data import DataLoader, Subset

from utils.config_loader import load_dataset_module
from utils.lightning_compat import patch_torchmetrics_for_legacy_lightning
from utils.video_utils import (
    DeterministicEvalDataset,
    add_repo_to_path,
    ensure_dir,
    seed_everything,
    tensor_cthw_to_uint8,
)

patch_torchmetrics_for_legacy_lightning()

USE_AMP = True
AMP_DTYPE = "bf16"
ENABLE_TF32 = True


def _amp_context(device: torch.device):
    if not USE_AMP or device.type != "cuda":
        return nullcontext()
    dtype = torch.bfloat16 if AMP_DTYPE.lower() == "bf16" else torch.float16
    return torch.autocast(device_type="cuda", dtype=dtype)


def _patch_legacy_lightning_checkpoint_loading():
    try:
        import pytorch_lightning.core.saving as pl_saving
    except Exception:
        pl_saving = None

    try:
        import pytorch_lightning.utilities.cloud_io as pl_cloud_io
    except Exception:
        pl_cloud_io = None

    def _compat_load(f, map_location=None):
        return torch.load(f, map_location=map_location, weights_only=False)

    if pl_saving is not None:
        pl_saving.pl_load = _compat_load
    if pl_cloud_io is not None:
        pl_cloud_io.load = _compat_load


def _resolve_transformer_ckpt(cfg) -> Path:
    ckpt_dir = Path(cfg.TRANSFORMER_CKPT_DIR).expanduser().resolve()
    for name in ("best.ckpt", "last.ckpt"):
        p = ckpt_dir / name
        if p.exists():
            return p
    ckpts = sorted(ckpt_dir.glob("*.ckpt"))
    if not ckpts:
        raise FileNotFoundError(f"No transformer checkpoint found under {ckpt_dir}")
    return ckpts[0]


def _resolve_vqvae_ckpt(cfg) -> Path:
    ckpt_dir = Path(cfg.VQVAE_CKPT_DIR).expanduser().resolve()
    for name in ("best.ckpt", "last.ckpt"):
        p = ckpt_dir / name
        if p.exists():
            return p
    ckpts = sorted(ckpt_dir.glob("*.ckpt"))
    if not ckpts:
        raise FileNotFoundError(f"No VQ-VAE checkpoint found under {ckpt_dir}")
    return ckpts[0]


def _set_if_missing(args, name, value):
    if not hasattr(args, name):
        setattr(args, name, value)


def _build_model_args(cfg, vqvae_ckpt: Path, ExternalSkyGPT):
    parser = argparse.ArgumentParser()
    parser = ExternalSkyGPT.add_model_specific_args(parser)
    args = parser.parse_args([])

    args.vqvae = str(vqvae_ckpt)
    args.data_path = str(cfg.PREPARED_TRAIN_VAL_H5)
    args.resolution = int(cfg.TRANSFORMER["resolution"])
    args.sequence_length = int(cfg.TRANSFORMER["sequence_length"])
    args.batch_size = int(cfg.TRANSFORMER["batch_size"])
    args.num_workers = int(cfg.COMMON["num_workers"])
    args.n_cond_frames = int(cfg.TRANSFORMER["n_cond_frames"])
    args.gradient_clip_val = float(cfg.TRANSFORMER["gradient_clip_val"])

    _set_if_missing(args, "hidden_dim", 576)
    _set_if_missing(args, "heads", 4)
    _set_if_missing(args, "layers", 8)
    _set_if_missing(args, "dropout", 0.1)
    _set_if_missing(args, "attn_type", "full")
    _set_if_missing(args, "attn_dropout", 0.1)
    _set_if_missing(args, "class_cond", False)
    _set_if_missing(args, "class_cond_dim", None)

    args.hidden_dim = int(cfg.TRANSFORMER.get("hidden_dim", getattr(args, "hidden_dim", 576)))
    args.heads = int(cfg.TRANSFORMER.get("heads", getattr(args, "heads", 4)))
    args.layers = int(cfg.TRANSFORMER.get("layers", getattr(args, "layers", 8)))
    args.dropout = float(cfg.TRANSFORMER.get("dropout", getattr(args, "dropout", 0.1)))
    args.attn_type = str(cfg.TRANSFORMER.get("attn_type", getattr(args, "attn_type", "full")))
    args.attn_dropout = float(cfg.TRANSFORMER.get("attn_dropout", getattr(args, "attn_dropout", 0.1)))
    args.class_cond = bool(cfg.TRANSFORMER.get("class_cond", getattr(args, "class_cond", False)))
    args.class_cond_dim = None
    return args


def _manual_load_transformer(model, ckpt_path: Path, device: torch.device):
    ckpt = torch.load(str(ckpt_path), map_location=device, weights_only=False)
    state_dict = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    print(f"[export] Missing keys: {len(missing)}", flush=True)
    print(f"[export] Unexpected keys: {len(unexpected)}", flush=True)


def _decode_future_uint8(decoded_bcthw: torch.Tensor, future_frames: int) -> np.ndarray:
    batch_np = []
    bsz = decoded_bcthw.shape[0]
    for i in range(bsz):
        arr = tensor_cthw_to_uint8(decoded_bcthw[i])
        batch_np.append(arr[-future_frames:])
    return np.stack(batch_np, axis=0)


def _patch_sample_method(model):
    from skygpt.utils import shift_dim
    import types

    def fixed_sample(self, n, batch=None):
        device = self.fc_in.weight.device
        cond = {}

        if self.use_frame_cond or self.args.class_cond:
            assert batch is not None
            video = batch["video"]
            if self.args.class_cond:
                label = batch["label"]
                cond["class_cond"] = F.one_hot(label, self.args.class_cond_dim).type_as(video)
            if self.use_frame_cond:
                cond["frame_cond"] = video[:, :, :self.args.n_cond_frames]

        samples = torch.zeros((n,) + self.shape, dtype=torch.long, device=device)
        samples_phycell = torch.zeros((n,) + self.shape, dtype=torch.long, device=device)
        samples_transformer = torch.zeros((n,) + self.shape, dtype=torch.long, device=device)
        idxs = list(itertools.product(*[range(s) for s in self.shape]))

        with torch.no_grad():
            prev_idx = None
            _, base_embeddings = self.vqvae.encode(cond["frame_cond"], include_embeddings=True)
            base_embeddings = shift_dim(base_embeddings, 1, -1)
            reuse_phycell = False

            for i, idx in enumerate(idxs):
                batch_idx_slice = (slice(None, None), *[slice(v, v + 1) for v in idx])
                batch_idx = (slice(None, None), *idx)

                if idx[1] == 0 and idx[2] == 0:
                    if idx[0] == 0:
                        _, base_embeddings = self.vqvae.encode(cond["frame_cond"], include_embeddings=True)
                        base_embeddings = shift_dim(base_embeddings, 1, -1)
                    else:
                        base_embeddings = torch.cat(
                            [base_embeddings, self.vqvae.codebook.dictionary_lookup(samples)[:, idx[0] - 1][:, None]],
                            axis=1,
                        )
                    reuse_phycell = False

                embeddings = self.vqvae.codebook.dictionary_lookup(samples)

                if prev_idx is None:
                    embeddings_slice = embeddings[batch_idx_slice]
                    samples_slice = samples[batch_idx_slice]
                else:
                    embeddings_slice = embeddings[prev_idx]
                    samples_slice = samples[prev_idx]

                with _amp_context(device):
                    logits, logits_phy, logits_trans = self(
                        embeddings_slice,
                        samples_slice,
                        cond,
                        decode_step=i,
                        decode_idx=idx,
                        x_full=base_embeddings,
                        include_loss=False,
                        reuse_phycell=reuse_phycell,
                    )

                def _sample_from_logits(logits_tensor):
                    lt = logits_tensor.squeeze().unsqueeze(0) if logits_tensor.shape[0] == 1 else logits_tensor.squeeze()
                    probs = F.softmax(lt.float(), dim=-1)
                    return torch.multinomial(probs, 1).squeeze(-1)

                samples[batch_idx] = _sample_from_logits(logits)
                samples_phycell[batch_idx] = _sample_from_logits(logits_phy)
                samples_transformer[batch_idx] = _sample_from_logits(logits_trans)

                prev_idx = batch_idx_slice
                reuse_phycell = True

        return (
            self.vqvae.decode(samples),
            self.vqvae.decode(samples_phycell),
            self.vqvae.decode(samples_transformer),
        )

    model.sample = types.MethodType(fixed_sample, model)
    return model


def _flush_memmap(mm):
    if mm is None:
        return
    mm.flush()
    base = getattr(mm, "_mmap", None)
    if base is not None:
        base.flush()


def _existing_resume_count(cur_path: Path, tgt_path: Path, src_path: Path, expected_export_len: int):
    if not (cur_path.exists() and tgt_path.exists() and src_path.exists()):
        return 0

    try:
        cur = np.load(cur_path, mmap_mode="r")
        tgt = np.load(tgt_path, mmap_mode="r")
        src = np.load(src_path, mmap_mode="r")
    except Exception:
        return 0

    if len(cur) != expected_export_len or len(tgt) != expected_export_len or len(src) != expected_export_len:
        return 0

    valid = (src >= 0) & (cur > 0) & (tgt > 0)
    if not np.any(valid):
        return 0

    first_invalid = np.where(~valid)[0]
    if len(first_invalid) == 0:
        return expected_export_len
    return int(first_invalid[0])


def _open_or_create_1d_memmap(path: Path, length: int, dtype, resume: bool):
    mode = "r+" if (resume and path.exists()) else "w+"
    return open_memmap(path, mode=mode, dtype=dtype, shape=(length,))


def _open_or_create_seq_memmap(path: Path, shape: tuple[int, ...], resume: bool):
    mode = "r+" if (resume and path.exists()) else "w+"
    return open_memmap(path, mode=mode, dtype=np.uint8, shape=shape)


def main() -> None:
    parser = argparse.ArgumentParser(description="Export generated future sequences incrementally with resume support")
    parser.add_argument("--dataset", required=True, choices=["skippd", "folmos"])
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--start_index", type=int, default=0)
    parser.add_argument("--stop_index", type=int, default=0, help="0 means export to the end")
    parser.add_argument("--flush_every", type=int, default=20, help="flush npy memmap every N batches")
    parser.add_argument("--log_every", type=int, default=20, help="print progress every N batches")
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--resume", action="store_true", help="resume from existing partial npy outputs")
    parser.add_argument("--reset", action="store_true", help="ignore existing outputs and restart from scratch")
    parser.add_argument("--resume_from", type=int, default=-1, help="manually force local resume offset within [start_index, stop_index)")
    args = parser.parse_args()

    if args.resume and args.reset:
        raise ValueError("--resume and --reset cannot be used together")

    cfg = load_dataset_module(args.dataset, "video")
    add_repo_to_path(cfg.SKYGPT_REPO_ROOT)
    _patch_legacy_lightning_checkpoint_loading()

    from models.video.patched_skygpt import ExternalSkyGPT

    seed_everything(int(cfg.COMMON["seed"]))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if device.type == "cuda" and ENABLE_TF32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True

    transformer_ckpt = _resolve_transformer_ckpt(cfg)
    vqvae_ckpt = _resolve_vqvae_ckpt(cfg)
    out_dir = ensure_dir(Path(cfg.IMAGE_OUTPUT_DIR))

    print(f"[export] dataset={args.dataset}", flush=True)
    print(f"[export] transformer ckpt={transformer_ckpt}", flush=True)
    print(f"[export] vqvae ckpt={vqvae_ckpt}", flush=True)
    print(f"[export] test eval h5={cfg.PREPARED_TEST_EVAL_H5}", flush=True)
    print(f"[export] output dir={out_dir}", flush=True)

    model_args = _build_model_args(cfg, vqvae_ckpt, ExternalSkyGPT)
    model = ExternalSkyGPT(model_args)
    _manual_load_transformer(model, transformer_ckpt, device)
    model = model.to(device)
    model.eval()
    model = _patch_sample_method(model)

    dataset = DeterministicEvalDataset(
        cfg.PREPARED_TEST_EVAL_H5,
        int(cfg.DATA["sequence_length"]),
        int(cfg.DATA["cond_frames"]),
    )

    total_dataset_len = len(dataset)
    start_index = max(0, int(args.start_index))
    stop_index = total_dataset_len if int(args.stop_index) <= 0 else min(int(args.stop_index), total_dataset_len)
    if start_index >= stop_index:
        raise ValueError(
            f"Invalid range: start_index={start_index}, stop_index={stop_index}, dataset_len={total_dataset_len}"
        )

    export_len = stop_index - start_index
    future_frames = int(cfg.DATA["future_frames"])
    num_scenarios = int(cfg.INFERENCE.get("num_scenarios", 1))

    seq_path = out_dir / "generated_test_images_sequences.npy"
    cur_path = out_dir / "generated_test_images_current_ts.npy"
    tgt_path = out_dir / "generated_test_images_target_ts.npy"
    src_path = out_dir / "generated_test_images_source_indices.npy"
    meta_path = out_dir / "generated_test_images_meta.json"

    resume_count = 0
    if not args.reset:
        if args.resume_from >= 0:
            if not (0 <= int(args.resume_from) <= export_len):
                raise ValueError(f"--resume_from must be in [0, {export_len}]")
            resume_count = int(args.resume_from)
        elif args.resume:
            resume_count = _existing_resume_count(cur_path, tgt_path, src_path, export_len)

    resume_global = start_index + resume_count
    if resume_count > 0:
        print(
            f"[export] resume enabled: local_offset={resume_count}, "
            f"global_index={resume_global}, remaining={export_len - resume_count}",
            flush=True,
        )
    else:
        print(f"[export] start from scratch: local_offset=0, global_index={start_index}", flush=True)

    if resume_count >= export_len:
        print(f"[export] nothing to do: export already completed for range [{start_index}, {stop_index})", flush=True)
        meta = {
            "dataset": args.dataset,
            "transformer_checkpoint": str(transformer_ckpt),
            "vqvae_checkpoint": str(vqvae_ckpt),
            "test_eval_h5": str(cfg.PREPARED_TEST_EVAL_H5),
            "total_dataset_len": int(total_dataset_len),
            "start_index": int(start_index),
            "stop_index": int(stop_index),
            "export_len": int(export_len),
            "resume_count": int(resume_count),
            "num_scenarios": int(num_scenarios),
            "future_frames": int(future_frames),
            "batch_size": int(args.batch_size),
            "num_workers": int(args.num_workers),
            "completed": True,
            "files": {
                "generated_sequences_npy": str(seq_path),
                "current_ts_npy": str(cur_path),
                "target_ts_npy": str(tgt_path),
                "source_indices_npy": str(src_path),
            },
        }
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)
        return

    seq_shape = (
        export_len,
        num_scenarios,
        future_frames,
        int(cfg.DATA["image_size"]),
        int(cfg.DATA["image_size"]),
        3,
    )
    seq_mm = _open_or_create_seq_memmap(seq_path, seq_shape, resume=(resume_count > 0 and not args.reset))
    cur_mm = _open_or_create_1d_memmap(cur_path, export_len, np.int64, resume=(resume_count > 0 and not args.reset))
    tgt_mm = _open_or_create_1d_memmap(tgt_path, export_len, np.int64, resume=(resume_count > 0 and not args.reset))
    src_mm = _open_or_create_1d_memmap(src_path, export_len, np.int64, resume=(resume_count > 0 and not args.reset))

    if resume_count == 0 or args.reset:
        src_mm[:] = -1
        cur_mm[:] = 0
        tgt_mm[:] = 0
        _flush_memmap(cur_mm)
        _flush_memmap(tgt_mm)
        _flush_memmap(src_mm)

    remaining_indices = list(range(start_index + resume_count, stop_index))

    loader_kwargs = dict(
        dataset=Subset(dataset, remaining_indices),
        batch_size=int(args.batch_size),
        shuffle=False,
        num_workers=int(args.num_workers),
        pin_memory=(device.type == "cuda"),
        drop_last=False,
    )
    if int(args.num_workers) > 0:
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"] = 2

    loader = DataLoader(**loader_kwargs)

    write_ptr = resume_count
    batch_count = 0
    remaining_batches = math.ceil(len(remaining_indices) / int(args.batch_size)) if len(remaining_indices) > 0 else 0

    for batch in loader:
        videos = batch["video"]
        current_ts = batch["current_ts"].numpy()
        target_ts = batch["target_ts"].numpy()
        bsz = videos.shape[0]

        if num_scenarios > 1:
            batch_video = videos.repeat_interleave(num_scenarios, dim=0)
        else:
            batch_video = videos

        if device.type == "cuda":
            batch_video = batch_video.cuda(non_blocking=True)

        sample_batch = {"video": batch_video}

        with torch.inference_mode():
            generated_full, _, _ = model.sample(batch_video.shape[0], batch=sample_batch)

        pred_future = _decode_future_uint8(generated_full, future_frames)

        if num_scenarios > 1:
            pred_future = pred_future.reshape(bsz, num_scenarios, future_frames, *pred_future.shape[-3:])
        else:
            pred_future = pred_future[:, None, :, :, :, :]

        seq_mm[write_ptr:write_ptr + bsz] = pred_future
        cur_mm[write_ptr:write_ptr + bsz] = current_ts
        tgt_mm[write_ptr:write_ptr + bsz] = target_ts
        src_mm[write_ptr:write_ptr + bsz] = np.arange(
            start_index + write_ptr,
            start_index + write_ptr + bsz,
            dtype=np.int64,
        )

        write_ptr += bsz
        batch_count += 1

        if batch_count % int(args.flush_every) == 0:
            _flush_memmap(seq_mm)
            _flush_memmap(cur_mm)
            _flush_memmap(tgt_mm)
            _flush_memmap(src_mm)

        if batch_count % int(args.log_every) == 0 or write_ptr == export_len:
            print(
                f"[export] processed {write_ptr}/{export_len} "
                f"(global {start_index + write_ptr}/{total_dataset_len}) "
                f"[resume_from local {resume_count}, global {resume_global}] "
                f"[batch {batch_count}/{remaining_batches}]",
                flush=True,
            )

    _flush_memmap(seq_mm)
    _flush_memmap(cur_mm)
    _flush_memmap(tgt_mm)
    _flush_memmap(src_mm)

    meta = {
        "dataset": args.dataset,
        "transformer_checkpoint": str(transformer_ckpt),
        "vqvae_checkpoint": str(vqvae_ckpt),
        "test_eval_h5": str(cfg.PREPARED_TEST_EVAL_H5),
        "total_dataset_len": int(total_dataset_len),
        "start_index": int(start_index),
        "stop_index": int(stop_index),
        "export_len": int(export_len),
        "resume_count": int(resume_count),
        "resume_global_index": int(resume_global),
        "num_scenarios": int(num_scenarios),
        "future_frames": int(future_frames),
        "batch_size": int(args.batch_size),
        "num_workers": int(args.num_workers),
        "files": {
            "generated_sequences_npy": str(seq_path),
            "current_ts_npy": str(cur_path),
            "target_ts_npy": str(tgt_path),
            "source_indices_npy": str(src_path),
        },
    }

    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    print(f"[export] saved: {seq_path}", flush=True)
    print(f"[export] saved: {cur_path}", flush=True)
    print(f"[export] saved: {tgt_path}", flush=True)
    print(f"[export] saved: {src_path}", flush=True)
    print(f"[export] saved: {meta_path}", flush=True)


if __name__ == "__main__":
    main()
