#!/usr/bin/env bash
set -euo pipefail

CONFIG_PATH="${CONFIG_PATH:-configs/gemma4_e2b_aws_g6e.yaml}"
RUN_ID="${RUN_ID:-local-$(date -u +%Y%m%dT%H%M%SZ)}"
S3_BUCKET="${S3_BUCKET:-}"
MAX_WALL_MINUTES="${MAX_WALL_MINUTES:-120}"

export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.92}"
export JAX_COMPILATION_CACHE_DIR="${JAX_COMPILATION_CACHE_DIR:-/tmp/jax_cache}"

OUT_DIR="runs/${RUN_ID}"
mkdir -p "${OUT_DIR}"
LOG_FILE="${OUT_DIR}/train.log"

sync_to_s3() {
  if [ -n "${S3_BUCKET}" ]; then
    if [ -f /var/log/qqn-llm-user-data.log ]; then
      cp /var/log/qqn-llm-user-data.log "${OUT_DIR}/user-data.log" 2>/dev/null || true
    fi
    aws s3 sync "${OUT_DIR}" "s3://${S3_BUCKET}/qqn-llm/runs/${RUN_ID}/" --only-show-errors || true
  fi
}

# Periodic background sync every 30s so `npm run aws:tail` shows live progress
SYNC_PID=""
if [ -n "${S3_BUCKET}" ]; then
  (
    while true; do
      sleep 30
      sync_to_s3
    done
  ) &
  SYNC_PID=$!
fi

cleanup() {
  if [ -n "${SYNC_PID}" ]; then
    kill "${SYNC_PID}" 2>/dev/null || true
  fi
  sync_to_s3
}
trap cleanup EXIT

if [ ! -d .venv ]; then
  echo "[qqn-llm] Creating venv and installing dependencies..." | tee -a "${LOG_FILE}"
  python3 -m venv .venv
  .venv/bin/pip install --upgrade pip >> "${LOG_FILE}" 2>&1
  .venv/bin/pip install -e ./qqn-jax -e ".[cuda]" >> "${LOG_FILE}" 2>&1
fi

echo "[qqn-llm] Starting run ${RUN_ID} with config ${CONFIG_PATH} (max ${MAX_WALL_MINUTES}m)" | tee -a "${LOG_FILE}"
timeout "${MAX_WALL_MINUTES}m" \
  .venv/bin/python -m qqn_llm.train \
  --config "${CONFIG_PATH}" \
  --output-dir "${OUT_DIR}" 2>&1 | tee -a "${LOG_FILE}"

if command -v node >/dev/null 2>&1; then
  node scripts/report.mjs "${OUT_DIR}" 2>&1 | tee -a "${LOG_FILE}" || true
fi

echo "[qqn-llm] Run ${RUN_ID} finished." | tee -a "${LOG_FILE}"
