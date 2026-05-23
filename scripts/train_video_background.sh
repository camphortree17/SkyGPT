#!/usr/bin/env bash
set -euo pipefail

DATASET="${1:-skippd}"
ACTION="${2:-start}"
STAGE="${3:-export_generated}"
MODE="${4:-resume}"   # resume | reset | fresh, only used by export_generated
ENV_NAME="${ENV_NAME:-skygpt_4090}"

if [[ "${STAGE}" == "resume" || "${STAGE}" == "reset" || "${STAGE}" == "fresh" ]]; then
  MODE="${STAGE}"
  STAGE="export_generated"
fi

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUTPUT_ROOT="${ROOT_DIR}/outputs/${DATASET}"

# 可通过 export 覆盖
EXPORT_BATCH_SIZE="${EXPORT_BATCH_SIZE:-16}"
EXPORT_START_INDEX="${EXPORT_START_INDEX:-0}"
EXPORT_STOP_INDEX="${EXPORT_STOP_INDEX:-0}"
EXPORT_FLUSH_EVERY="${EXPORT_FLUSH_EVERY:-20}"
EXPORT_LOG_EVERY="${EXPORT_LOG_EVERY:-20}"
EXPORT_NUM_WORKERS="${EXPORT_NUM_WORKERS:-2}"
EXPORT_RESUME_FROM="${EXPORT_RESUME_FROM:--1}"

ts() { date '+%Y-%m-%d %H:%M:%S'; }

validate_stage() {
  case "${STAGE}" in
    all|prepare_video|train_vqvae|train_transformer|sample|eval_video|export_generated)
      ;;
    *)
      echo "[$(ts)] invalid STAGE=${STAGE}, expected: all | prepare_video | train_vqvae | train_transformer | sample | eval_video | export_generated"
      exit 1
      ;;
  esac
}

validate_mode() {
  case "${MODE}" in
    resume|reset|fresh)
      ;;
    *)
      echo "[$(ts)] invalid MODE=${MODE}, expected: resume | reset | fresh"
      exit 1
      ;;
  esac
}

resolve_log_dir() {
  case "${STAGE}" in
    train_vqvae)
      echo "${OUTPUT_ROOT}/vqvae/logs"
      ;;
    train_transformer)
      echo "${OUTPUT_ROOT}/transformer/logs"
      ;;
    sample|export_generated)
      echo "${OUTPUT_ROOT}/image_output/logs"
      ;;
    eval_video)
      echo "${OUTPUT_ROOT}/eval/logs"
      ;;
    prepare_video)
      echo "${OUTPUT_ROOT}/prepared/logs"
      ;;
    all)
      echo "${OUTPUT_ROOT}/logs"
      ;;
  esac
}

build_run_args() {
  local args=(
    "--dataset" "${DATASET}"
  )

  if [[ "${STAGE}" != "all" ]]; then
    args+=("--only" "${STAGE}")
  fi

  if [[ "${STAGE}" == "export_generated" ]]; then
    args+=(
      "--batch_size" "${EXPORT_BATCH_SIZE}"
      "--start_index" "${EXPORT_START_INDEX}"
      "--stop_index" "${EXPORT_STOP_INDEX}"
      "--flush_every" "${EXPORT_FLUSH_EVERY}"
      "--log_every" "${EXPORT_LOG_EVERY}"
      "--num_workers" "${EXPORT_NUM_WORKERS}"
    )

    case "${MODE}" in
      resume)
        args+=("--resume")
        ;;
      reset)
        args+=("--reset")
        ;;
      fresh)
        ;;
    esac

    if [[ "${EXPORT_RESUME_FROM}" != "-1" ]]; then
      args+=("--resume_from" "${EXPORT_RESUME_FROM}")
    fi
  fi

  printf '%s\n' "${args[@]}"
}

start() {
  validate_stage
  validate_mode
  LOG_DIR="$(resolve_log_dir)"
  PID_FILE="${LOG_DIR}/video_${STAGE}.pid"
  mkdir -p "${LOG_DIR}"

  if [[ -f "${PID_FILE}" ]] && kill -0 "$(cat "${PID_FILE}")" >/dev/null 2>&1; then
    echo "[$(ts)] video stage already running, PID=$(cat "${PID_FILE}")"
    exit 0
  fi

  local log_file="${LOG_DIR}/video_pipeline_$(date '+%Y%m%d_%H%M%S').log"
  mapfile -t CMD_ARGS < <(build_run_args)

  (
    cd "${ROOT_DIR}"

    if [[ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]]; then
      source "$HOME/miniconda3/etc/profile.d/conda.sh"
    elif [[ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]]; then
      source "$HOME/anaconda3/etc/profile.d/conda.sh"
    else
      echo "[$(ts)] conda.sh not found"
      exit 1
    fi

    conda activate "${ENV_NAME}"
    export PYTHONPATH="${ROOT_DIR}:${PYTHONPATH:-}"

    echo "[$(ts)] dataset=${DATASET}"
    echo "[$(ts)] stage=${STAGE}"
    if [[ "${STAGE}" == "export_generated" ]]; then
      echo "[$(ts)] mode=${MODE}"
      echo "[$(ts)] batch_size=${EXPORT_BATCH_SIZE}"
      echo "[$(ts)] start_index=${EXPORT_START_INDEX}"
      echo "[$(ts)] stop_index=${EXPORT_STOP_INDEX}"
      echo "[$(ts)] flush_every=${EXPORT_FLUSH_EVERY}"
      echo "[$(ts)] log_every=${EXPORT_LOG_EVERY}"
      echo "[$(ts)] num_workers=${EXPORT_NUM_WORKERS}"
      echo "[$(ts)] resume_from=${EXPORT_RESUME_FROM}"
    fi
    echo "[$(ts)] command=python scripts/run_video_stage.py ${CMD_ARGS[*]}"

    python scripts/run_video_stage.py "${CMD_ARGS[@]}"
  ) >"${log_file}" 2>&1 &

  echo $! > "${PID_FILE}"
  echo "[$(ts)] started video stage, PID=$(cat "${PID_FILE}")"
  echo "[$(ts)] stage=${STAGE}"
  if [[ "${STAGE}" == "export_generated" ]]; then
    echo "[$(ts)] mode=${MODE}"
  fi
  echo "[$(ts)] log=${log_file}"
}

stop() {
  validate_stage
  LOG_DIR="$(resolve_log_dir)"
  PID_FILE="${LOG_DIR}/video_${STAGE}.pid"

  if [[ -f "${PID_FILE}" ]]; then
    kill "$(cat "${PID_FILE}")" || true
    rm -f "${PID_FILE}"
    echo "[$(ts)] stopped"
  else
    echo "[$(ts)] not running"
  fi
}

status() {
  validate_stage
  LOG_DIR="$(resolve_log_dir)"
  PID_FILE="${LOG_DIR}/video_${STAGE}.pid"

  if [[ -f "${PID_FILE}" ]] && kill -0 "$(cat "${PID_FILE}")" >/dev/null 2>&1; then
    echo "[$(ts)] running, PID=$(cat "${PID_FILE}")"
  else
    echo "[$(ts)] not running"
  fi
}

case "${ACTION}" in
  start) start ;;
  stop) stop ;;
  status) status ;;
  *)
    echo "Usage: bash scripts/train_video_background.sh [skippd|folmos|sirta] [start|stop|status] [all|prepare_video|train_vqvae|train_transformer|sample|eval_video|export_generated] [resume|reset|fresh]"
    exit 1
    ;;
esac
