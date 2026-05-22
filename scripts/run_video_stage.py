#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

STAGES = {
    "prepare_video": "data/scripts/prepare_video_data.py",
    "train_vqvae": "models/video/train_vqvae.py",
    "train_transformer": "models/video/train_transformer.py",
    "sample": "models/video/sample.py",
    "eval_video": "models/video/eval_video.py",
    "export_generated": "models/video/export_generated_sequences.py",
}


def run(script: str, dataset: str, extra_args: list[str] | None = None):
    root = Path(__file__).resolve().parents[1]
    path = root / script
    cmd = [sys.executable, str(path), "--dataset", dataset]
    if extra_args:
        cmd.extend(extra_args)

    print("=" * 80, flush=True)
    print("Running:", path, flush=True)
    print("Command:", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=["skippd", "folmos", "sirta"])
    parser.add_argument("--only", choices=list(STAGES.keys()))

    # 这些参数主要给 export_generated 用
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--start_index", type=int, default=None)
    parser.add_argument("--stop_index", type=int, default=None)
    parser.add_argument("--flush_every", type=int, default=None)
    parser.add_argument("--log_every", type=int, default=None)
    parser.add_argument("--num_workers", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--reset", action="store_true")
    parser.add_argument("--resume_from", type=int, default=None)

    args = parser.parse_args()

    extra_args: list[str] = []

    if args.only == "export_generated":
        if args.batch_size is not None:
            extra_args += ["--batch_size", str(args.batch_size)]
        if args.start_index is not None:
            extra_args += ["--start_index", str(args.start_index)]
        if args.stop_index is not None:
            extra_args += ["--stop_index", str(args.stop_index)]
        if args.flush_every is not None:
            extra_args += ["--flush_every", str(args.flush_every)]
        if args.log_every is not None:
            extra_args += ["--log_every", str(args.log_every)]
        if args.num_workers is not None:
            extra_args += ["--num_workers", str(args.num_workers)]
        if args.resume:
            extra_args += ["--resume"]
        if args.reset:
            extra_args += ["--reset"]
        if args.resume_from is not None:
            extra_args += ["--resume_from", str(args.resume_from)]

    if args.only:
        run(STAGES[args.only], args.dataset, extra_args=extra_args)
        return

    for key in ["prepare_video", "train_vqvae", "train_transformer", "sample", "eval_video", "export_generated"]:
        run(STAGES[key], args.dataset, extra_args=(extra_args if key == "export_generated" else None))


if __name__ == "__main__":
    main()
