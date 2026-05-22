#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

STAGES = {
    "train_pv": "models/pv/train_pv_mapper.py",
    "infer_pv_val_real": "models/pv/infer_from_generated.py",
    "infer_pv_test_generated": "models/pv/infer_from_generated.py",
    "eval_pv_ws": "models/pv/eval_pv_ws.py",
}

def run(script: str, dataset: str, extra_args: list[str] | None = None):
    root = Path(__file__).resolve().parents[1]
    path = root / script

    env = os.environ.copy()
    env["PYTHONPATH"] = f"{root}:{env.get('PYTHONPATH', '')}" if env.get("PYTHONPATH") else str(root)

    cmd = [sys.executable, str(path), "--dataset", dataset]
    if extra_args:
        cmd.extend(extra_args)

    print("=" * 80, flush=True)
    print("Running:", path, flush=True)
    print("PYTHONPATH:", env["PYTHONPATH"], flush=True)
    print("Command:", " ".join(cmd), flush=True)

    subprocess.run(cmd, check=True, env=env)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=["skippd", "folmos"])
    parser.add_argument("--only", choices=list(STAGES.keys()))
    args = parser.parse_args()

    if args.only:
        if args.only == "infer_pv_val_real":
            run(STAGES[args.only], args.dataset, ["--split", "val", "--input_mode", "real"])
        elif args.only == "infer_pv_test_generated":
            run(STAGES[args.only], args.dataset, ["--split", "test", "--input_mode", "generated"])
        else:
            run(STAGES[args.only], args.dataset)
        return

    run(STAGES["train_pv"], args.dataset)
    run(STAGES["infer_pv_val_real"], args.dataset, ["--split", "val", "--input_mode", "real"])
    run(STAGES["infer_pv_test_generated"], args.dataset, ["--split", "test", "--input_mode", "generated"])
    run(STAGES["eval_pv_ws"], args.dataset)

if __name__ == "__main__":
    main()
