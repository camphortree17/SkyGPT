#!/usr/bin/env bash
set -euo pipefail

DATASET="${1:-skippd}"
ACTION="${2:-start}"
MODE="${3:-resume}"   # resume | reset | fresh
ENV_NAME="${ENV_NAME:-skygpt_4090}"

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_DIR="${ROOT_DIR}/outputs/${DATASET}/image_output_val/logs"
PID_FILE="${LOG_DIR}/video_train.pid"
mkdir -p "${LOG_DIR}"

# 可通过 export 覆盖
EXPORT_BATCH_SIZE="${EXPORT_BATCH_SIZE:-16}"
EXPORT_START_INDEX="${EXPORT_START_INDEX:-0}"
EXPORT_STOP_INDEX="${EXPORT_STOP_INDEX:-0}"
EXPORT_FLUSH_EVERY="${EXPORT_FLUSH_EVERY:-20}"
EXPORT_LOG_EVERY="${EXPORT_LOG_EVERY:-20}"
EXPORT_NUM_WORKERS="${EXPORT_NUM_WORKERS:-2}"
EXPORT_RESUME_FROM="${EXPORT_RESUME_FROM:--1}"

ts() { date '+%Y-%m-%d %H:%M:%S'; }

build_export_args() {
  local args=(
    "--dataset" "${DATASET}"
    "--only" "export_generated"
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
    *)
      echo "[$(ts)] invalid MODE=${MODE}, expected: resume | reset | fresh"
      exit 1
      ;;
  esac

  if [[ "${EXPORT_RESUME_FROM}" != "-1" ]]; then
    args+=("--resume_from" "${EXPORT_RESUME_FROM}")
  fi

  printf '%s\n' "${args[@]}"
}

start() {
  if [[ -f "${PID_FILE}" ]] && kill -0 "$(cat "${PID_FILE}")" >/dev/null 2>&1; then
    echo "[$(ts)] video stage already running, PID=$(cat "${PID_FILE}")"
    exit 0
  fi

  local log_file="${LOG_DIR}/video_pipeline_$(date '+%Y%m%d_%H%M%S').log"
  mapfile -t CMD_ARGS < <(build_export_args)

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
    echo "[$(ts)] mode=${MODE}"
    echo "[$(ts)] batch_size=${EXPORT_BATCH_SIZE}"
    echo "[$(ts)] start_index=${EXPORT_START_INDEX}"
    echo "[$(ts)] stop_index=${EXPORT_STOP_INDEX}"
    echo "[$(ts)] flush_every=${EXPORT_FLUSH_EVERY}"
    echo "[$(ts)] log_every=${EXPORT_LOG_EVERY}"
    echo "[$(ts)] num_workers=${EXPORT_NUM_WORKERS}"
    echo "[$(ts)] resume_from=${EXPORT_RESUME_FROM}"
    echo "[$(ts)] command=python scripts/run_video_stage.py ${CMD_ARGS[*]}"

    python scripts/run_video_stage.py "${CMD_ARGS[@]}"
  ) >"${log_file}" 2>&1 &

  echo $! > "${PID_FILE}"
  echo "[$(ts)] started video stage, PID=$(cat "${PID_FILE}")"
  echo "[$(ts)] mode=${MODE}"
  echo "[$(ts)] log=${log_file}"
}

stop() {
  if [[ -f "${PID_FILE}" ]]; then
    kill "$(cat "${PID_FILE}")" || true
    rm -f "${PID_FILE}"
    echo "[$(ts)] stopped"
  else
    echo "[$(ts)] not running"
  fi
}

status() {
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
    echo "Usage: bash scripts/train_video_background.sh [skippd|folmos] [start|stop|status] [resume|reset|fresh]"
    exit 1
    ;;
esac