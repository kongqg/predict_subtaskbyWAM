#!/usr/bin/env bash
set -euo pipefail

PY=${PY:-/datashare/hengguo/miniconda3/envs/meeting_room_training/bin/python}
CFG=${CFG:-/datashare/kqg/subtask_progress/configs/two_task_0_1_completed_6000_done_short8_hardneg30x3_unitree.yaml}
GPU=${GPU:-0}

export PYTHONPATH=${PYTHONPATH:-/datashare/kqg}
export WANDB_MODE=${WANDB_MODE:-offline}
export WANDB_INIT_TIMEOUT=${WANDB_INIT_TIMEOUT:-300}

cd /datashare/kqg
CUDA_VISIBLE_DEVICES="$GPU" "$PY" -m subtask_progress.train --config "$CFG" --device cuda
