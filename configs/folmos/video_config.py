from pathlib import Path
from utils.config_loader import package_root

PROJECT_ROOT = package_root()
SKYGPT_REPO_ROOT = PROJECT_ROOT / "SkyGPT"
DATASET_NAME = "folmos"

DATA_ROOT = PROJECT_ROOT / "data" / DATASET_NAME
RAW_ROOT = DATA_ROOT / "raw"
PREPARED_VIDEO_DIR = DATA_ROOT / "prepared" / "video" / "prepared_video_data"

OUTPUT_ROOT = PROJECT_ROOT / "outputs" / DATASET_NAME
VQVAE_DIR = OUTPUT_ROOT / "vqvae"
TRANSFORMER_DIR = OUTPUT_ROOT / "transformer"
IMAGE_OUTPUT_DIR = OUTPUT_ROOT / "image_output_val"
EVAL_DIR = OUTPUT_ROOT / "eval"

TRAIN_CSV = RAW_ROOT / "train.csv"
VAL_CSV = RAW_ROOT / "val.csv"
TEST_CSV = RAW_ROOT / "val.csv"

DATA = {
    "timestamp_col": "timestamp",
    "image_col": "image_path",
    "irr_col": "irr_ghi",
    "target_col": "target_ghi_15min",
    "image_size": 64,
    "sequence_length": 27,
    "cond_frames": 24,
    "future_frames": 3,
    "freq_minutes": 5,
    "drop_missing_images": True,
    "limit_rows_per_split": 0,
}

COMMON = {
    "seed": 1234,
    "gpus": 1,
    "num_workers": 4,
    "check_val_every_n_epoch": 1,
}

VQVAE = {
    "batch_size": 4,
    "max_epochs": 60,
    "resolution": DATA["image_size"],
    "sequence_length": DATA["sequence_length"],
    "downsample": [1, 4, 4],
    "gradient_clip_val": 1.0,
}

TRANSFORMER = {
    "batch_size": 2,
    "max_epochs": 120,
    "resolution": DATA["image_size"],
    "sequence_length": DATA["sequence_length"],
    "n_cond_frames": DATA["cond_frames"],
    "gradient_clip_val": 1.0,
    "hidden_dim": 576,
    "heads": 4,
    "layers": 6,
    "dropout": 0.1,
    "attn_dropout": 0.1,

}

INFERENCE = {
    "sample_num_cases": 20,
    "num_scenarios": 1,
    "eval_num_cases": 0,
}

PREPARED_TRAIN_VAL_H5 = PREPARED_VIDEO_DIR / "train_val.h5"
PREPARED_VAL_EVAL_H5 = PREPARED_VIDEO_DIR / "val_eval.h5"
PREPARED_TEST_EVAL_H5 = PREPARED_VIDEO_DIR / "val_eval.h5"

TIMES_CURR_TRAIN = PREPARED_VIDEO_DIR / "times_curr_train.npy"
TIMES_CURR_VAL = PREPARED_VIDEO_DIR / "times_curr_val.npy"
TIMES_CURR_TEST = PREPARED_VIDEO_DIR / "times_curr_val.npy"
TIMES_TARGET_TRAIN = PREPARED_VIDEO_DIR / "times_target_train.npy"
TIMES_TARGET_VAL = PREPARED_VIDEO_DIR / "times_target_val.npy"
TIMES_TARGET_TEST = PREPARED_VIDEO_DIR / "times_target_val.npy"

VQVAE_CKPT_DIR = VQVAE_DIR / "checkpoints"
TRANSFORMER_CKPT_DIR = TRANSFORMER_DIR / "checkpoints"
