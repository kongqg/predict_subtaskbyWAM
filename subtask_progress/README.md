# Subtask Progress Transformer

This folder is intentionally separate from the GR00T VLA policy. It trains a
small monitor that predicts:

- local progress of the current canonical subtask, in `[0, 1]`
- done logit for the current canonical subtask

It does not predict actions, roll out candidates, train a world model, or run RL.

## Current Data Assumption

The current Unitree meeting-room split is LeRobot v2.1. Each segmented subtask is
already one episode under:

```text
/datashare/kqg/meeting_room_code/meeting_room_404_0511_order_success_with_subtask_segmented_split/{train,val}
```

`episodes.jsonl` provides `episode_index`, `task_index`, `length`,
`source_episode_index`, and `sub_tasks[0].start/end`. The parquet files currently
contain:

```text
observation.state, action, timestamp, frame_index, episode_index, index, task_index, sub_task
```

There is no precomputed visual feature column in the raw split. The state config
uses `observation.state` only as a pipeline sanity check. The V-JEPA path writes
separate feature parquet files and trains from those:

```yaml
dataset:
  feature_root:
    train: /datashare/kqg/meeting_room_progress_outputs/vjepa_features/train
    val: /datashare/kqg/meeting_room_progress_outputs/vjepa_features/val
  visual_column: visual_features
```

No visual encoder runs inside the Dataset.

## V-JEPA Feature Extraction

The extractor uses a local V-JEPA repo plus a local checkpoint, so the training
server does not need to download model weights during extraction:

```bash
python -m subtask_progress.extract_vjepa_features \
  --root /datashare/kqg/meeting_room_code/meeting_room_404_0511_order_success_with_subtask_segmented_split/train \
  --output-root /datashare/kqg/meeting_room_progress_outputs/vjepa_features/train \
  --repo-dir /datashare/kqg/meeting_room_code/models/vjepa2 \
  --checkpoint /datashare/kqg/meeting_room_code/models/vjepa2_1_vit_base_384/vjepa2_1_vitb_dist_vitG_384.pt \
  --camera observation.images.cam_left_high \
  --overwrite
```

Run the same command for the `val` split with the matching paths.

## Done Labels

The current data has no explicit completion column. The default weak label is:

```yaml
done_label_strategy: last_window
done_window: 3
```

That marks the final 3 frames of each segment as done. Use
`done_label_strategy: column` and set `done_column` when explicit completion
labels exist.

## Commands

Tiny overfit sanity check on the remote training server:

```bash
cd /datashare/kqg/meeting_room_code
source /datashare/hengguo/miniconda3/etc/profile.d/conda.sh
conda activate meeting_room_training
python -m subtask_progress.train \
  --config subtask_progress/configs/tiny_overfit.yaml \
  --tiny-overfit
```

Train:

```bash
python -m subtask_progress.train \
  --config subtask_progress/configs/meeting_room_state_progress.yaml
```

Train from V-JEPA features:

```bash
python -m subtask_progress.train \
  --config subtask_progress/configs/meeting_room_vjepa_progress.yaml
```

Evaluate:

```bash
python -m subtask_progress.eval \
  --config subtask_progress/configs/meeting_room_state_progress.yaml \
  --checkpoint /datashare/kqg/meeting_room_progress_outputs/state_progress_20k/best.pt
```

## Minimal Online Use

```python
import torch
from subtask_progress.model import SubtaskProgressTransformer, SubtaskProgressTransformerConfig
from subtask_progress.monitor import ProgressMonitor

model = SubtaskProgressTransformer(SubtaskProgressTransformerConfig(
    visual_dim=43,
    proprio_dim=0,
    num_tasks=13,
    history_length=32,
))
model.load_state_dict(torch.load("best.pt", map_location="cpu")["model"])

monitor = ProgressMonitor(model, history_length=32, done_threshold=0.9, done_patience=3)
monitor.reset(task_id=0, start_visual=start_feature)
result = monitor.update(current_feature)
```

Call `reset()` explicitly when switching subtasks.
