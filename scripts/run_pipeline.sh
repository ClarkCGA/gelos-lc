#!/bin/bash
# Run the full pipeline for all yaml configs: embeddings + analysis in
# parallel across GPUs, then comparisons once every config has completed.
#
# Configs are assigned round-robin to the available worker slots (one per
# GPU by default, PER_GPU to run several configs concurrently on each GPU);
# each worker runs through its queue sequentially, running generation then
# analysis for each config before moving to the next. When all queues have
# drained without failures, all comparison configs (configs/comparisons/)
# run in one container.
# Each step runs per the project convention (data locations come from .env
# via the compose mounts):
#
#   docker compose run --rm --remove-orphans dev \
#     CUDA_VISIBLE_DEVICES=<gpu> python -m gelos.generation -y configs/<name>
#   docker compose run --rm --remove-orphans dev \
#     CUDA_VISIBLE_DEVICES=<gpu> python -m gelos.analysis -y configs/<name>
#   docker compose run --rm --remove-orphans dev python -m gelos.comparison
#
# Already-complete runs are skipped unless OVERWRITE is set: generation via
# its .embeddings_complete marker, analysis via its .analysis_complete marker
# (with OVERWRITE, analysis re-enters but per-step caches still skip, so only
# missing artifacts are recomputed). Comparison always (re)runs.
#
# Usage:
#   scripts/run_pipeline.sh [CONFIG_DIR_OR_GLOB]   # default: configs/
#
# Environment overrides:
#   SERVICE    compose service to run steps in (default: dev). SERVICE=prod
#              uses the baked-in gelos instead of reinstalling the local
#              checkout each run — rebuild the image (docker compose build
#              prod) after bumping GELOS_VERSION, and run with
#              COMPOSE_FILE=compose.yml to skip the compose.override.yml
#              that force-installs the local gelos at container start
#   GPUS       comma-separated GPU ids (default: all, via nvidia-smi)
#   PER_GPU    concurrent configs per GPU (default: 1) — make sure GPU
#              memory fits PER_GPU models before raising this
#   LOG_DIR    per-step logs (default: ./generation_logs)
#   OVERWRITE  non-empty -> pass --overwrite to gelos.generation and gelos.analysis
set -uo pipefail

cd "$(dirname "$0")/.."  # repo root, where compose.yml and .env live

configs_arg=${1:-configs}
log_dir=${LOG_DIR:-./generation_logs}
overwrite=${OVERWRITE:-}
per_gpu=${PER_GPU:-1}
service=${SERVICE:-prod}

if ! [[ "$per_gpu" =~ ^[1-9][0-9]*$ ]]; then
  echo "PER_GPU must be a positive integer, got: $per_gpu" >&2
  exit 1
fi

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

# one worker slot per (gpu, worker) pair; configs are dealt round-robin
# across slots, so consecutive configs land on different GPUs first
slots=()
for ((w = 0; w < per_gpu; w++)); do
  slots+=("${gpus[@]}")
done

mkdir -p "$log_dir"
echo "running ${#configs[@]} configs across ${#gpus[@]} GPUs (${gpus[*]}), $per_gpu per GPU: generation + analysis, then comparisons"

# label every container this run starts, so an interrupt can find and stop
# them — containers run under the docker daemon and outlive our compose
# clients, and backgrounded workers ignore SIGINT in non-interactive bash
run_label="gelos.pipeline=$$"

pids=()
cleanup() {
  trap - INT TERM
  echo "interrupted — stopping workers and containers" >&2
  [ ${#pids[@]} -gt 0 ] && kill "${pids[@]}" 2>/dev/null
  docker ps -q --filter "label=$run_label" | xargs -r docker stop -t 5 >/dev/null
  wait 2>/dev/null
  exit 130
}
trap cleanup INT TERM

run_step() {
  local gpu=$1 step=$2 name=$3
  shift 3
  echo "[gpu $gpu] $name $step starting"
  # CUDA_VISIBLE_DEVICES goes via -e: as a command word it only works on
  # images whose entrypoint routes the command through env (dev does,
  # prod does not)
  if docker compose run --rm --remove-orphans --label "$run_label" \
    -e CUDA_VISIBLE_DEVICES="$gpu" "$service" \
    python -m "gelos.$step" \
    -y "configs/$name" "$@" \
    >"$log_dir/${name%.yaml}.$step.log" 2>&1; then
    echo "[gpu $gpu] $name $step done"
  else
    echo "[gpu $gpu] $name $step FAILED (see $log_dir/${name%.yaml}.$step.log)" >&2
    return 1
  fi
}

run_queue() {
  local gpu=$1
  shift
  local rc=0 cfg name
  for cfg in "$@"; do
    name=$(basename "$cfg")
    # a failed generation makes analysis pointless for that config, but the
    # rest of the queue still runs
    if run_step "$gpu" generation "$name" ${overwrite:+--overwrite}; then
      run_step "$gpu" analysis "$name" ${overwrite:+--overwrite} || rc=1
    else
      rc=1
    fi
  done
  return $rc
}

for si in "${!slots[@]}"; do
  queue=()
  for ci in "${!configs[@]}"; do
    if [ $((ci % ${#slots[@]})) -eq "$si" ]; then
      queue+=("${configs[$ci]}")
    fi
  done
  [ ${#queue[@]} -eq 0 ] && continue
  run_queue "${slots[$si]}" "${queue[@]}" &
  pids+=($!)
done

fail=0
for pid in "${pids[@]}"; do
  wait "$pid" || fail=1
done

if [ $fail -ne 0 ]; then
  echo "one or more configs failed — skipping comparisons, check $log_dir" >&2
  exit 1
fi

echo "all configs complete — running comparisons"
if docker compose run --rm --remove-orphans --label "$run_label" "$service" python -m gelos.comparison \
  >"$log_dir/comparisons.log" 2>&1; then
  echo "comparisons done"
else
  echo "comparisons FAILED (see $log_dir/comparisons.log)" >&2
  exit 1
fi
echo "pipeline complete"
