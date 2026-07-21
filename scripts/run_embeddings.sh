#!/bin/bash
# Run all yaml configs in parallel across GPUs.
#
# Configs are assigned round-robin to the available GPUs; each GPU then works
# through its queue sequentially (one job per GPU at a time), so total
# parallelism == number of GPUs. Each config runs in its own container per the
# project convention (data locations come from .env via the compose mounts):
#
#   docker compose run --rm --remove-orphans dev \
#     CUDA_VISIBLE_DEVICES=<gpu> python -m gelos.generation -y configs/<name>
#
# Already-complete runs (.embeddings_complete) are skipped by gelos.generation
# itself unless OVERWRITE is set.
#
# Usage:
#   scripts/run_embeddings.sh [CONFIG_DIR_OR_GLOB]   # default: configs/
#
# Environment overrides:
#   GPUS       comma-separated GPU ids (default: all, via nvidia-smi)
#   LOG_DIR    per-config logs (default: ./generation_logs)
#   OVERWRITE  non-empty -> pass --overwrite
set -uo pipefail

cd "$(dirname "$0")/.."  # repo root, where compose.yml and .env live

configs_arg=${1:-configs}
log_dir=${LOG_DIR:-./generation_logs}
overwrite=${OVERWRITE:-}

if [ -d "$configs_arg" ]; then
  mapfile -t configs < <(ls "$configs_arg"/*.yaml 2>/dev/null | sort)
else
  # shellcheck disable=SC2086 # intentional glob expansion of the argument
  mapfile -t configs < <(ls $configs_arg 2>/dev/null | sort)
fi
if [ ${#configs[@]} -eq 0 ]; then
  echo "no yaml configs found for: $configs_arg" >&2
  exit 1
fi

if [ -n "${GPUS:-}" ]; then
  IFS=',' read -ra gpus <<<"$GPUS"
else
  mapfile -t gpus < <(nvidia-smi --query-gpu=index --format=csv,noheader)
fi
if [ ${#gpus[@]} -eq 0 ]; then
  echo "no GPUs detected (set GPUS=0,1,... to specify manually)" >&2
  exit 1
fi

mkdir -p "$log_dir"
echo "running ${#configs[@]} configs across ${#gpus[@]} GPUs (${gpus[*]})"

run_queue() {
  local gpu=$1
  shift
  local rc=0 cfg name
  for cfg in "$@"; do
    name=$(basename "$cfg")
    echo "[gpu $gpu] $name starting"
    if docker compose run --rm --remove-orphans dev \
      CUDA_VISIBLE_DEVICES="$gpu" python -m gelos.generation \
      -y "configs/$name" \
      ${overwrite:+--overwrite} \
      >"$log_dir/${name%.yaml}.log" 2>&1; then
      echo "[gpu $gpu] $name done"
    else
      echo "[gpu $gpu] $name FAILED (see $log_dir/${name%.yaml}.log)" >&2
      rc=1
    fi
  done
  return $rc
}

pids=()
for gi in "${!gpus[@]}"; do
  queue=()
  for ci in "${!configs[@]}"; do
    if [ $((ci % ${#gpus[@]})) -eq "$gi" ]; then
      queue+=("${configs[$ci]}")
    fi
  done
  [ ${#queue[@]} -eq 0 ] && continue
  run_queue "${gpus[$gi]}" "${queue[@]}" &
  pids+=($!)
done

fail=0
for pid in "${pids[@]}"; do
  wait "$pid" || fail=1
done
if [ $fail -ne 0 ]; then
  echo "one or more configs failed — check $log_dir" >&2
  exit 1
fi
echo "all configs complete"
