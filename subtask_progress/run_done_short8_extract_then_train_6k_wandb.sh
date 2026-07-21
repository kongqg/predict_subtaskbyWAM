#!/usr/bin/env bash
set -euo pipefail

export CLIPS=8
export GPU_LIST=${GPU_LIST:-"0 1 2 3 4 5 6 7"}
export TRAIN_AFTER_EXTRACT=1
export TRAIN_WANDB_MODE=online
: "${WANDB_API_KEY:?set WANDB_API_KEY before running online wandb sync}"
export WANDB_INIT_TIMEOUT=${WANDB_INIT_TIMEOUT:-300}

cd /datashare/kqg
exec /datashare/kqg/run_done_short_parallel_8gpu.sh
