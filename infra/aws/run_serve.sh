#!/usr/bin/env bash
set -euo pipefail

CHECKPOINT_URI="${CHECKPOINT_URI:?CHECKPOINT_URI (s3://... or local path) is required}"
SERVE_PORT="${SERVE_PORT:-8000}"
RUN_ID="${RUN_ID:-serve-$(date -u +%Y%m%dT%H%M%SZ)}"
S3_BUCKET="${S3_BUCKET:-}"
MAX_WALL_MINUTES="${MAX_WALL_MINUTES:-120}"
EVAL_POKEMON_PIRATE="${EVAL_POKEMON_PIRATE:-1}"

export XLA_PYTHON_CLIENT_ALLOCATOR="${XLA_PYTHON_CLIENT_ALLOCATOR:-cuda_async}"
export XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}"
export JAX_COMPILATION_CACHE_DIR="${JAX_COMPILATION_CACHE_DIR:-/tmp/jax_cache}"

OUT_DIR="runs/${RUN_ID}"
mkdir -p "${OUT_DIR}"
LOG_FILE="${OUT_DIR}/serve.log"

sync_logs() {
  if [ -n "${S3_BUCKET}" ]; then
    aws s3 sync "${OUT_DIR}" "s3://${S3_BUCKET}/qqn-llm/serve/${RUN_ID}/" --only-show-errors || true
  fi
}

SYNC_PID=""
if [ -n "${S3_BUCKET}" ]; then
  (
    while true; do
      sleep 30
      sync_logs
    done
  ) &
  SYNC_PID=$!
fi

cleanup() {
  if [ -n "${SYNC_PID}" ]; then
    kill "${SYNC_PID}" 2>/dev/null || true
  fi
  sync_logs
}
trap cleanup EXIT

if [ ! -d .venv ]; then
  echo "[qqn-llm-serve] Creating venv and installing dependencies..." | tee -a "${LOG_FILE}"
  python3 -m venv .venv
  .venv/bin/pip install --upgrade pip >> "${LOG_FILE}" 2>&1
  .venv/bin/pip install -e ./qqn-jax -e ".[cuda]" >> "${LOG_FILE}" 2>&1
fi

EXTRA_FLAGS=()
if [ "${EVAL_POKEMON_PIRATE}" = "1" ]; then
  EXTRA_FLAGS+=(--eval-pokemon-pirate --eval-out "${OUT_DIR}/pokemon_pirate_eval.json")
fi

echo "[qqn-llm-serve] Starting inference server on port ${SERVE_PORT} with ${CHECKPOINT_URI} (max ${MAX_WALL_MINUTES}m)" | tee -a "${LOG_FILE}"
timeout "${MAX_WALL_MINUTES}m" \
  .venv/bin/python -m qqn_llm.serve \
  --checkpoint "${CHECKPOINT_URI}" \
  --port "${SERVE_PORT}" \
  "${EXTRA_FLAGS[@]}" 2>&1 | tee -a "${LOG_FILE}"
