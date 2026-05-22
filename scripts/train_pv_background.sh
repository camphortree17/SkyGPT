#!/usr/bin/env bash
set -euo pipefail

DATASET="${1:-skippd}"
ENV_NAME="${ENV_NAME:-skygpt_4090}"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_DIR="${ROOT_DIR}/outputs/${DATASET}/eval/logs"
PID_FILE="${LOG_DIR}/pv_train.pid"
mkdir -p "${LOG_DIR}"

ts() { date '+%Y-%m-%d %H:%M:%S'; }

start() {
  if [[ -f "${PID_FILE}" ]] && kill -0 "$(cat "${PID_FILE}")" >/dev/null 2>&1; then
    echo "[$(ts)] pv stage already running, PID=$(cat "${PID_FILE}")"
    exit 0
  fi
  local log_file="${LOG_DIR}/pv_pipeline_$(date '+%Y%m%d_%H%M%S').log"
  (
    cd "${ROOT_DIR}"
    if [[ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]]; then
      source "$HOME/miniconda3/etc/profile.d/conda.sh"
    elif [[ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]]; then
      source "$HOME/anaconda3/etc/profile.d/conda.sh"
    fi
    conda activate "${ENV_NAME}"
    python scripts/run_pv_stage.py --dataset "${DATASET}" --only eval_pv_ws
  ) >"${log_file}" 2>&1 &
  echo $! > "${PID_FILE}"
  echo "[$(ts)] started pv stage, PID=$(cat "${PID_FILE}")"
  echo "[$(ts)] log=${log_file}"
}

stop() {
  if [[ -f "${PID_FILE}" ]]; then
    kill "$(cat "${PID_FILE}")" || true
    rm -f "${PID_FILE}"
    echo "[$(ts)] stopped"
  fi
}

status() {
  if [[ -f "${PID_FILE}" ]] && kill -0 "$(cat "${PID_FILE}")" >/dev/null 2>&1; then
    echo "[$(ts)] running, PID=$(cat "${PID_FILE}")"
  else
    echo "[$(ts)] not running"
  fi
}

case "${2:-start}" in
  start) start ;;
  stop) stop ;;
  status) status ;;
  *) echo "Usage: bash scripts/train_pv_background.sh [skippd|folmos] [start|stop|status]" ;;
esac
