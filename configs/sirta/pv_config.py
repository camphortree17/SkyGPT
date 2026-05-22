from pathlib import Path
from utils.config_loader import package_root

PROJECT_ROOT = package_root()
DATASET_NAME = "sirta"

DATA_ROOT = PROJECT_ROOT / "data" / DATASET_NAME
RAW_ROOT = DATA_ROOT / "raw"
PREPARED_PV_DIR = DATA_ROOT / "prepared" / "pv"

OUTPUT_ROOT = PROJECT_ROOT / "outputs" / DATASET_NAME
PV_MODEL_DIR = OUTPUT_ROOT / "pv_model"
PV_PRED_DIR = OUTPUT_ROOT / "pv_output"
PV_EVAL_DIR = OUTPUT_ROOT / "eval"
PV_LOG_DIR = PV_MODEL_DIR / "logs"

TRAIN_CSV = RAW_ROOT / "train.csv"
VAL_CSV = RAW_ROOT / "val.csv"
TEST_CSV = RAW_ROOT / "test.csv"

PV_MAPPING_H5 = PREPARED_PV_DIR / "pv_mapping_dataset.h5"
PV_MANIFEST_JSON = PREPARED_PV_DIR / "pv_mapping_manifest.json"

IMAGE_SIZE = 64
IMAGE_CHANNELS = 3
IMAGE_RESAMPLE = "bilinear"
STORE_IMAGES_AS_UINT8 = True
H5_COMPRESSION = "gzip"
H5_COMPRESSION_OPTS = 4

CSV_COLUMNS = {
    "timestamp": "timestamp",
    "irr_ghi": "Global_Solar_Flux",
    "target_ghi_15min": "target_Global_Solar_Flux_15min",
    "image_path": "image_path",
}

WINDOW_COND_FRAMES = 15
WINDOW_FUTURE_FRAMES = 15
WINDOW_FREQ_MINUTES = 1
DROP_BAD_IMAGES = True

TRAIN_BATCH_SIZE = 128
VAL_BATCH_SIZE = 128
TEST_BATCH_SIZE = 128
NUM_EPOCHS = 50
TORCH_LOG_INTERVAL = 200
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 0.0
EARLY_STOPPING_PATIENCE = 8
REDUCE_LR_PATIENCE = 4
REDUCE_LR_FACTOR = 0.5
MIN_DELTA = 1e-5
LABEL_NORMALIZATION = "zscore"
SHUFFLE_TRAIN = True
SEED = 1234
VERBOSE = 1

UNET_BASE_CHANNELS = 32
UNET_DROPOUT = 0.0
UNET_BOTTLENECK_RES_BLOCKS = 2

TORCH_NUM_WORKERS = 1
TORCH_USE_AMP = True
TORCH_AMP_DTYPE = "bf16"

TORCH_BEST_MODEL_PATH = PV_MODEL_DIR / "best_pv_mapper_torch.pt"
TORCH_LAST_MODEL_PATH = PV_MODEL_DIR / "last_pv_mapper_torch.pt"
TORCH_TRAIN_HISTORY_CSV = PV_LOG_DIR / "train_history_torch.csv"
TORCH_CURRENT_METRICS_JSON = PV_EVAL_DIR / "concurrent_pv_mapping_metrics_torch.json"
TORCH_CURRENT_PREDICTIONS_CSV = PV_PRED_DIR / "current_time_mapping_predictions_torch.csv"

NORMALIZATION_STATS_JSON = PV_MODEL_DIR / "label_stats.json"

GENERATED_SEQUENCES_NPY = OUTPUT_ROOT / "image_output" / "generated_test_images_sequences.npy"
GENERATED_SOURCE_INDICES_NPY = OUTPUT_ROOT / "image_output" / "generated_test_images_source_indices.npy"
GENERATED_LAST_FRAME_INDEX = -1

VAL_REAL_SCENARIO_PREDICTIONS_CSV = PV_PRED_DIR / "val_real_scenario_predictions.csv"
VAL_REAL_AGGREGATED_PREDICTIONS_CSV = PV_PRED_DIR / "val_real_aggregated_predictions.csv"
TEST_GENERATED_SCENARIO_PREDICTIONS_CSV = PV_PRED_DIR / "scenario_predictions.csv"
TEST_GENERATED_AGGREGATED_PREDICTIONS_CSV = PV_PRED_DIR / "aggregated_predictions.csv"

PV_WS_METRICS_JSON = PV_EVAL_DIR / "pv_ws_metrics.json"
PV_WS_METRICS_CSV = PV_EVAL_DIR / "pv_ws_metrics.csv"
PV_WS_ALPHA = 0.1

SCENARIO_PREDICTIONS_CSV = PV_PRED_DIR / "scenario_predictions.csv"
AGGREGATED_PREDICTIONS_CSV = PV_PRED_DIR / "aggregated_predictions.csv"
PV_METRICS_JSON = PV_EVAL_DIR / "pv_metrics.json"
PV_METRICS_CSV = PV_EVAL_DIR / "pv_metrics.csv"
