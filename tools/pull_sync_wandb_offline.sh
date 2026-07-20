#!/usr/bin/env bash
set -euo pipefail

REMOTE_WANDB_DIR=${1:-root@10.3.1.228:/datashare/kqg/meeting_room_progress_outputs/wandb_offline/wandb}
LOCAL_WANDB_DIR=${2:-$HOME/wandb_offline/meeting_room}
PROJECT=${WANDB_PROJECT:-meeting_room_subtask_progress}

if [[ -z "${WANDB_API_KEY:-}" ]]; then
  echo "WANDB_API_KEY is required" >&2
  exit 2
fi

mkdir -p "$LOCAL_WANDB_DIR"
rsync -av "$REMOTE_WANDB_DIR/" "$LOCAL_WANDB_DIR/"

if ! python3 - <<'PY'
try:
    import wandb  # noqa: F401
except Exception:
    raise SystemExit(1)
PY
then
  python3 -m pip install --user -i https://pypi.tuna.tsinghua.edu.cn/simple wandb
fi

python3 -m wandb login --relogin "$WANDB_API_KEY"
shopt -s nullglob
runs=("$LOCAL_WANDB_DIR"/offline-run-*)
if (( ${#runs[@]} == 0 )); then
  echo "no offline wandb runs found in $LOCAL_WANDB_DIR" >&2
  exit 1
fi
python3 -m wandb sync --project "$PROJECT" "${runs[@]}"
