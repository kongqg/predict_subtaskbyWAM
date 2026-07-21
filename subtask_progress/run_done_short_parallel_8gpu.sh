#!/usr/bin/env bash
set -euo pipefail

BASE=/datashare/kqg
PY=/datashare/hengguo/miniconda3/envs/meeting_room_training/bin/python
DATA_ROOT=/datashare/kqg/meeting_room_progress_outputs/two_task_0_1_full
MODEL_REPO=/datashare/kqg/models/vjepa2
CKPT=/datashare/kqg/models/vjepa2_1_vit_base_384/vjepa2_1_vitb_dist_vitG_384.pt
GPU_COUNT=${GPU_COUNT:-8}
GPU_LIST=${GPU_LIST:-}
BATCH_SIZE=${BATCH_SIZE:-64}
TRAIN_AFTER_EXTRACT=${TRAIN_AFTER_EXTRACT:-1}
TRAIN_WANDB_MODE=${TRAIN_WANDB_MODE:-offline}
LOG_DIR=/datashare/kqg/done_short_parallel_logs
JOB_FILE=/datashare/kqg/done_short_extract_jobs.tsv

export PYTHONPATH=/datashare/kqg
mkdir -p "$LOG_DIR"
cd "$BASE"

cameras=(
  observation.images.cam_left_high
  observation.images.cam_right_high
  observation.images.cam_left_wrist
  observation.images.cam_right_wrist
)
view_names=(cam_left_high cam_right_high cam_left_wrist cam_right_wrist)

raw_count() {
  find "$DATA_ROOT/$1/raw/data" -name 'episode_*.parquet' | wc -l
}

feature_count() {
  local clip=$1 split=$2 view=$3
  local dir="$DATA_ROOT/$split/features_done_short${clip}_${view}/data/chunk-000"
  [ -d "$dir" ] && find "$dir" -name 'episode_*.parquet' | wc -l || echo 0
}

run_extract() {
  local gpu=$1 clip=$2 split=$3 camera=$4 view=$5 shards=$6 shard=$7
  local out="$DATA_ROOT/$split/features_done_short${clip}_${view}"
  local log="$LOG_DIR/extract_short${clip}_${split}_${view}_shard${shard}.log"
  echo "[$(date '+%F %T')] gpu=$gpu short=$clip split=$split view=$view shard=$shard/$shards" | tee -a "$log"
  CUDA_VISIBLE_DEVICES="$gpu" "$PY" -m subtask_progress.extract_vjepa_features \
    --root "$DATA_ROOT/$split/raw" \
    --output-root "$out" \
    --repo-dir "$MODEL_REPO" \
    --checkpoint "$CKPT" \
    --camera "$camera" \
    --clip-length "$clip" \
    --batch-size "$BATCH_SIZE" \
    --num-shards "$shards" \
    --shard-index "$shard" \
    --device cuda >> "$log" 2>&1
}

: > "$JOB_FILE"
CLIPS=(${CLIPS:-8 16})
if [ -n "$GPU_LIST" ]; then
  GPUS=($GPU_LIST)
else
  GPUS=($(seq 0 $((GPU_COUNT - 1))))
fi
WORKER_COUNT=${#GPUS[@]}

for clip in "${CLIPS[@]}"; do
  for split in train val; do
    expected=$(raw_count "$split")
    for i in "${!cameras[@]}"; do
      view=${view_names[$i]}
      have=$(feature_count "$clip" "$split" "$view")
      if [ "$have" -lt "$expected" ]; then
        for shard in $(seq 0 $((WORKER_COUNT - 1))); do
          printf '%s\t%s\t%s\t%s\t%s\t%s\n' "$clip" "$split" "${cameras[$i]}" "$view" "$WORKER_COUNT" "$shard" >> "$JOB_FILE"
        done
      else
        echo "skip complete short=$clip split=$split view=$view count=$have"
      fi
    done
  done
done

job_count=$(wc -l < "$JOB_FILE")
echo "extract jobs: $job_count"
if [ "$job_count" -gt 0 ]; then
  for worker in "${!GPUS[@]}"; do
    gpu=${GPUS[$worker]}
    (
      idx=0
      while IFS=$'\t' read -r clip split camera view shards shard; do
        if [ $((idx % WORKER_COUNT)) -eq "$worker" ]; then
          run_extract "$gpu" "$clip" "$split" "$camera" "$view" "$shards" "$shard"
        fi
        idx=$((idx + 1))
      done < "$JOB_FILE"
    ) &
  done
  wait
fi

for clip in "${CLIPS[@]}"; do
  for split in train val; do
    expected=$(raw_count "$split")
    roots=()
    for view in "${view_names[@]}"; do
      have=$(feature_count "$clip" "$split" "$view")
      if [ "$have" -ne "$expected" ]; then
        echo "feature count mismatch: short=$clip split=$split view=$view have=$have expected=$expected" >&2
        exit 1
      fi
      roots+=(--feature-root "$DATA_ROOT/$split/features_done_short${clip}_${view}")
    done
    combine_out="$DATA_ROOT/$split/features_done_short${clip}_4view_structured_npy"
    "$PY" -m subtask_progress.combine_feature_views \
      --output-root "$combine_out" \
      "${roots[@]}" \
      --mode stack \
      --output-format npy \
      --overwrite > "$LOG_DIR/combine_short${clip}_${split}.log" 2>&1
  done
done

if [ "$TRAIN_AFTER_EXTRACT" != "1" ]; then
  echo "TRAIN_AFTER_EXTRACT=$TRAIN_AFTER_EXTRACT, skip training"
  exit 0
fi

make_train_config() {
  local src=$1 dst=$2 mode=$3
  "$PY" - "$src" "$dst" "$mode" <<'PY'
import sys
from pathlib import Path
import yaml

src, dst, mode = sys.argv[1:4]
cfg = yaml.safe_load(Path(src).read_text())
cfg.setdefault("wandb", {})["mode"] = mode
Path(dst).parent.mkdir(parents=True, exist_ok=True)
Path(dst).write_text(yaml.safe_dump(cfg, sort_keys=False))
PY
}

train_one() {
  local clip=$1 gpu=$2
  local cfg=/datashare/kqg/subtask_progress/configs/two_task_0_1_completed_6000_done_short${clip}_unitree.yaml
  local run_cfg="$cfg"
  local out=/datashare/kqg/meeting_room_progress_outputs/two_task_0_1_completed_6000_done_short${clip}_run
  if [ "$TRAIN_WANDB_MODE" != "offline" ]; then
    run_cfg=/datashare/kqg/tmp_configs/two_task_0_1_completed_6000_done_short${clip}_${TRAIN_WANDB_MODE}.yaml
    make_train_config "$cfg" "$run_cfg" "$TRAIN_WANDB_MODE"
  fi
  rm -rf "$out"
  mkdir -p "$out"
  WANDB_MODE="$TRAIN_WANDB_MODE" CUDA_VISIBLE_DEVICES="$gpu" "$PY" -m subtask_progress.train \
    --config "$run_cfg" \
    --device cuda 2>&1 | tee "$out/train_stdout.log"
}

for clip in "${CLIPS[@]}"; do
  gpu=0
  [ "$clip" = "16" ] && gpu=1
  train_one "$clip" "$gpu" &
done
wait
