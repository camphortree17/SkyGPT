#!/usr/bin/env bash
set -euo pipefail

DATASET="${1:-skippd}"
ACTION="${2:-start}"
STAGE="${3:-eval_pv_ws}"
ENV_NAME="${ENV_NAME:-skygpt_4090}"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_DIR="${ROOT_DIR}/outputs/${DATASET}/eval/logs"
PID_FILE="${LOG_DIR}/pv_train.pid"
mkdir -p "${LOG_DIR}"

ts() { date '+%Y-%m-%d %H:%M:%S'; }

validate_stage() {
  case "${STAGE}" in
    all|train_pv|infer_pv_val_real|infer_pv_test_generated|eval_pv_ws)
      ;;
    *)
      echo "[$(ts)] invalid STAGE=${STAGE}, expected: all | train_pv | infer_pv_val_real | infer_pv_test_generated | eval_pv_ws"
      exit 1
      ;;
  esac
}

build_run_args() {
  local args=("--dataset" "${DATASET}")
  if [[ "${STAGE}" != "all" ]]; then
    args+=("--only" "${STAGE}")
  fi
  printf '%s\n' "${args[@]}"
}

start() {
  if [[ -f "${PID_FILE}" ]] && kill -0 "$(cat "${PID_FILE}")" >/dev/null 2>&1; then
    echo "[$(ts)] pv stage already running, PID=$(cat "${PID_FILE}")"
    exit 0
  fi
  local log_file="${LOG_DIR}/pv_pipeline_$(date '+%Y%m%d_%H%M%S').log"
  validate_stage
  mapfile -t CMD_ARGS < <(build_run_args)
  (
    cd "${ROOT_DIR}"
    if [[ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]]; then
      source "$HOME/miniconda3/etc/profile.d/conda.sh"
    elif [[ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]]; then
      source "$HOME/anaconda3/etc/profile.d/conda.sh"
    fi
    conda activate "${ENV_NAME}"
    export PYTHONPATH="${ROOT_DIR}:${PYTHONPATH:-}"

    echo "[$(ts)] dataset=${DATASET}"
    echo "[$(ts)] stage=${STAGE}"
    echo "[$(ts)] command=python scripts/run_pv_stage.py ${CMD_ARGS[*]}"

    python scripts/run_pv_stage.py "${CMD_ARGS[@]}"
  ) >"${log_file}" 2>&1 &
  echo $! > "${PID_FILE}"
  echo "[$(ts)] started pv stage, PID=$(cat "${PID_FILE}")"
  echo "[$(ts)] stage=${STAGE}"
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

case "${ACTION}" in
  start) start ;;
  stop) stop ;;
  status) status ;;
  *)
    echo "Usage: bash scripts/train_pv_background.sh [skippd|folmos|sirta] [start|stop|status] [all|train_pv|infer_pv_val_real|infer_pv_test_generated|eval_pv_ws]"
    exit 1
    ;;
esac
