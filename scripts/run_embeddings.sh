#!/bin/bash
# this script runs all yamls in parallel rather than sequentially
set -euo pipefail

gpus=(0 1 2 3 4 5 6 7)
i=0
for cfg in $1; do
  CUDA_VISIBLE_DEVICES=${gpus[$((i % ${#gpus[@]}))]} \
    python -m gelos.embedding_generation \
      --yaml-path "$cfg" \
      --raw-data-dir /app/data/raw \
      --processed-data-dir /app/data/processed &
  ((i++)) || true
done
wait
