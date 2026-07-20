# Structured 4-View Progress/Done Report

Date: 2026-07-20

Remote: `root@10.3.1.228`

Remote code: `/datashare/kqg/subtask_progress`

Remote run:
`/datashare/kqg/meeting_room_progress_outputs/two_task_0_1_completed_2000_structured_4view_run`

## Goal

Replace old 4-view concat input:

```text
[B, T, 3072] -> Linear(3072, 128)
```

with structured 4-view fusion:

```text
[B, T, V=4, D=768]
per-view shared projection 768 -> d_model
+ learnable view embedding
+ task-conditioned attention over views
-> [B, T, d_model]
-> existing temporal transformer
```

Single-view configs remain compatible with the old `[B, T, D]` input.

## Changed Files

```text
subtask_progress/model.py
subtask_progress/dataset.py
subtask_progress/train.py
subtask_progress/eval.py
subtask_progress/metrics.py
subtask_progress/combine_feature_views.py
tools/episode_done_summary.py
subtask_progress/configs/two_task_0_1_completed_2000_structured_4view_unitree.yaml
subtask_progress/tests/test_model.py
subtask_progress/tests/test_dataset.py
subtask_progress/tests/test_episode_done_summary.py
```

## Data Layout

View order:

```text
0 cam_left_high
1 cam_right_high
2 cam_left_wrist
3 cam_right_wrist
```

The first implementation read four independent feature roots during training.
That was correct but slow because each sample path had to resolve and stack data
from four parquet-backed episode caches.

The current implementation pre-merges features per episode into a structured
`.npy` file:

```text
features_4view_structured_npy/chunk-xxx/episode_yyyyyy.npy
shape = [frames, 4, 768]
```

This does not change labels, features, split, frame order, or model inputs. It
only changes storage layout. Source feature roots are checked for matching
episodes, matching `frame_index`, and matching length before writing the merged
cache.

Train data:

```text
segments: 572
samples: 226340
visual_features: [32, 4, 768]
start_visual: [4, 768]
```

Validation data:

```text
segments: 64
samples: 25951
visual_features: [32, 4, 768]
start_visual: [4, 768]
```

## Model I/O

Batch input:

```text
visual_features: [B, T=32, V=4, D=768]
start_visual:     [B, V=4, D=768]
task_ids:         [B]
padding_mask:     [B, T]
view_mask:        [B, V]
proprio:          unused, proprio_dim=0
```

Output:

```text
progress:             [B], sigmoid in [0, 1]
done_logit:           [B]
view_attention:       [B, T, 4]
start_view_attention: [B, 4]
```

Training uses view dropout with `view_dropout_prob: 0.25`. Validation disables
view dropout.

## Training Config

```text
max_steps: 2000
seed: 41
history_length: 32
d_model: 128
num_layers: 2
num_heads: 4
lambda_progress: 0.2
lambda_done: 1.0
lambda_rank: 0.0
batch_size: 256
learning_rate: 3e-4
warmup_steps: 500
scheduler: warmup cosine, min_lr=1e-5
done trigger eval: threshold=0.9, window=10, votes=5
```

Final train log:

```json
{"step": 2000, "lr": 0.00001, "loss": 0.04739793762564659, "progress_loss": 0.004531506448984146, "done_loss": 0.04649163782596588, "rank_loss": 0.0}
```

## Speed

Observed during remote run:

```text
direct 4 parquet roots: step 100 about 2m05s, step 300 about 8m
structured npy cache:  step 100 about 1m18s, step 1000 about 9m50s
```

The `.npy` cache avoids four parquet reads plus per-sample view stacking in the
hot training loop. It should not change the mathematical result except normal
training nondeterminism; the first logged losses matched the direct-root run for
the same seed before the storage change.

## 2000-Step Validation Metrics

Frame-level progress:

| metric | value |
|---|---:|
| MAE | 0.0551 |
| RMSE | 0.0735 |
| bin accuracy 10 | 0.5436 |
| pairwise monotonic accuracy | 0.9760 |

Frame-level done:

| metric | value |
|---|---:|
| accuracy | 0.9600 |
| precision | 0.7933 |
| recall | 0.8343 |
| F1 | 0.8133 |
| AUROC | 0.9872 |
| confusion matrix | TN 22652, FP 589, FN 449, TP 2261 |

Per task:

| task | progress MAE | done precision | done recall | done F1 | AUROC |
|---:|---:|---:|---:|---:|---:|
| 0 | 0.0514 | 0.8789 | 0.8447 | 0.8615 | 0.9899 |
| 1 | 0.0598 | 0.6287 | 0.8076 | 0.7070 | 0.9825 |

## Episode Metrics

Trigger rule:

```text
done_prob >= 0.9
10-frame window
at least 5 votes
```

| metric | overall | task0 | task1 | early | miss |
|---|---:|---:|---:|---:|---:|
| strict | 45/64 = 70.31% | 27/32 = 84.38% | 18/32 = 56.25% | 16 | 3 |
| tolerance@5 | 49/64 = 76.56% | 29/32 = 90.62% | 20/32 = 62.50% | 12 | 3 |
| tolerance@10 | 53/64 = 82.81% | 30/32 = 93.75% | 23/32 = 71.88% | 8 | 3 |

## View Attention

Order:

```text
cam_left_high, cam_right_high, cam_left_wrist, cam_right_wrist
```

| scope | cam_left_high | cam_right_high | cam_left_wrist | cam_right_wrist |
|---|---:|---:|---:|---:|
| overall | 0.2758 | 0.2672 | 0.2975 | 0.1594 |
| task0 | 0.2893 | 0.2436 | 0.3023 | 0.1648 |
| task1 | 0.2582 | 0.2980 | 0.2913 | 0.1525 |

## 2000-Step Comparison

All rows use the same validation split and `threshold=0.9, window=10, votes=5`
for episode triggering.

| run | fusion | strict episode acc | task0 | task1 | early | miss | progress MAE | done precision | done recall | done F1 | AUROC |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| single-view baseline | cam_left_high only | 48/64 = 75.00% | 28/32 = 87.50% | 20/32 = 62.50% | 9 | 7 | 0.0609 | 0.7665 | 0.7498 | 0.7581 | 0.9835 |
| old 4-view concat | `[B,T,3072] -> Linear` | 48/64 = 75.00% | 29/32 = 90.62% | 19/32 = 59.38% | 5 | 11 | 0.0577 | 0.8944 | 0.7155 | 0.7950 | 0.9851 |
| structured 4-view | task-conditioned view attention | 45/64 = 70.31% | 27/32 = 84.38% | 18/32 = 56.25% | 16 | 3 | 0.0551 | 0.7933 | 0.8343 | 0.8133 | 0.9872 |

Structured 4-view is better on frame-level done F1 than old concat and
single-view at 2000 steps, and better on progress MAE than single-view. However,
strict episode accuracy is worse because it creates more early triggers. With a
10-frame tolerance, structured 4-view reaches `53/64 = 82.81%`, which means many
of the early errors are close to the annotated boundary rather than large misses.

## Checks

Passed:

```text
15 pytest tests passed
no NaN in train/eval logs
validation view dropout disabled
validation view_mask is all true
feature cache generated after strict episode/frame_index/length alignment checks
model supports old single-view [B,T,D] path
```

Current risk:

```text
strict episode accuracy did not improve at 2000 steps.
structured fusion is more recall-heavy and still needs threshold/tolerance or
training-step selection before replacing the current best single-view baseline.
```

## 3000-Step Continuation

Continuation config:
`subtask_progress/configs/two_task_0_1_completed_3000_structured_4view_unitree.yaml`

Checkpoint:
`/datashare/kqg/meeting_room_progress_outputs/two_task_0_1_completed_2000_structured_4view_run/checkpoint_3000.pt`

This run resumed from `checkpoint_2000.pt` and kept LR at `1e-5`.

| step | train loss | progress loss | done loss | progress MAE | done precision | done recall | done F1 | AUROC |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2000 | 0.0474 | 0.0045 | 0.0465 | 0.0551 | 0.7933 | 0.8343 | 0.8133 | 0.9872 |
| 3000 | 0.0499 | 0.0040 | 0.0491 | 0.0552 | 0.7671 | 0.8531 | 0.8078 | 0.9869 |

Episode trigger, `threshold=0.9, window=10, votes=5`:

| step | strict | task0 strict | task1 strict | early | miss | tolerance@5 | tolerance@10 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 2000 | 45/64 = 70.31% | 27/32 = 84.38% | 18/32 = 56.25% | 16 | 3 | 49/64 = 76.56% | 53/64 = 82.81% |
| 3000 | 44/64 = 68.75% | 27/32 = 84.38% | 17/32 = 53.12% | 18 | 2 | 49/64 = 76.56% | 53/64 = 82.81% |

Per-task 3000:

| task | progress MAE | done precision | done recall | done F1 | strict episode |
|---:|---:|---:|---:|---:|---:|
| 0 | 0.0513 | 0.8555 | 0.8590 | 0.8573 | 27/32 = 84.38% |
| 1 | 0.0602 | 0.6028 | 0.8379 | 0.7012 | 17/32 = 53.12% |

Conclusion: continuing to 3000 did not improve this checkpoint. Recall went up,
precision went down, and strict episode accuracy dropped by one episode. For the
structured 4-view branch, `checkpoint_2000.pt` is the better choice so far.

## Hard Negative 30x3

Config:
`subtask_progress/configs/two_task_0_1_completed_2000_structured_4view_hardneg30x3_unitree.yaml`

Run:
`/datashare/kqg/meeting_room_progress_outputs/two_task_0_1_completed_2000_structured_4view_hardneg30x3_run`

Change:

```text
done_start - 30 <= frame < done_start:
  target_done stays 0
  done BCE weight = 3.0
all other frames:
  unchanged
```

Train weighting check:

```text
train samples: 226340
normal weight 1.0: 209180
hard negative weight 3.0: 17160
masked weight 0.0: 0
```

2000-step comparison:

| run | train loss | strict | task0 | task1 | early | miss | progress MAE | done precision | done recall | done F1 | AUROC |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| single-view baseline | 0.0378 | 48/64 = 75.00% | 28/32 = 87.50% | 20/32 = 62.50% | 9 | 7 | 0.0609 | 0.7665 | 0.7498 | 0.7581 | 0.9835 |
| structured 4-view | 0.0474 | 45/64 = 70.31% | 27/32 = 84.38% | 18/32 = 56.25% | 16 | 3 | 0.0551 | 0.7933 | 0.8343 | 0.8133 | 0.9872 |
| structured 4-view hardneg30x3 | 0.0623 | 47/64 = 73.44% | 28/32 = 87.50% | 19/32 = 59.38% | 12 | 5 | 0.0579 | 0.8507 | 0.7613 | 0.8035 | 0.9874 |

Tolerance comparison for structured variants:

| run | tolerance@5 | tolerance@10 |
|---|---:|---:|
| structured 4-view | 49/64 = 76.56% | 53/64 = 82.81% |
| structured 4-view hardneg30x3 | 51/64 = 79.69% | 56/64 = 87.50% |

Hard negative weighting moved the model in the intended direction: fewer early
triggers and higher precision. The cost is more misses and slightly worse
progress MAE. It improves structured 4-view but still does not beat the
single-view baseline on strict episode accuracy.

### Hard Negative 3000-Step Continuation

The `hardneg30x3` checkpoint was continued from 2000 to 3000 steps with LR fixed
at `1e-5`.

| step | train loss | strict | task0 | task1 | early | miss | progress MAE | done precision | done recall | done F1 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2000 | 0.0623 | 47/64 = 73.44% | 28/32 = 87.50% | 19/32 = 59.38% | 12 | 5 | 0.0579 | 0.8507 | 0.7613 | 0.8035 |
| 3000 | 0.0544 | 47/64 = 73.44% | 28/32 = 87.50% | 19/32 = 59.38% | 13 | 4 | 0.0581 | 0.8347 | 0.7768 | 0.8047 |

Near-done negative frames, `done_start-30 <= frame < done_start`:

| step | false done | false done rate | mean done_prob | p90 done_prob |
|---:|---:|---:|---:|---:|
| 2000 | 179/1920 | 9.32% | 0.1986 | 0.8778 |
| 3000 | 194/1920 | 10.10% | 0.2116 | 0.9029 |

The 3000-step continuation lowered train loss but did not improve the actual
trigger behavior. It became slightly less conservative near the done boundary,
so the better hard-negative checkpoint is still `checkpoint_2000.pt`.

### Hard Negative 4000-Step Continuation

The same `hardneg30x3` run was continued from `checkpoint_3000.pt` to
`checkpoint_4000.pt`, still at LR `1e-5`.

| step | train loss | strict | task0 | task1 | early | miss | progress MAE | done precision | done recall | done F1 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2000 | 0.0623 | 47/64 = 73.44% | 28/32 = 87.50% | 19/32 = 59.38% | 12 | 5 | 0.0579 | 0.8507 | 0.7613 | 0.8035 |
| 3000 | 0.0544 | 47/64 = 73.44% | 28/32 = 87.50% | 19/32 = 59.38% | 13 | 4 | 0.0581 | 0.8347 | 0.7768 | 0.8047 |
| 4000 | 0.0527 | 47/64 = 73.44% | 27/32 = 84.38% | 20/32 = 62.50% | 14 | 3 | 0.0578 | 0.8312 | 0.7812 | 0.8054 |

Near-done negative frames, `done_start-30 <= frame < done_start`:

| step | false done | false done rate | mean done_prob | p90 done_prob |
|---:|---:|---:|---:|---:|
| 2000 | 179/1920 | 9.32% | 0.1986 | 0.8778 |
| 3000 | 194/1920 | 10.10% | 0.2116 | 0.9029 |
| 4000 | 205/1920 | 10.68% | 0.2137 | 0.9205 |

4000 keeps the same strict accuracy as 2000/3000, but it is less conservative:
early increases from 12 to 14 while miss drops from 5 to 3. For the current
goal, `checkpoint_2000.pt` remains the best hard-negative checkpoint.

## Trigger Window 3/2 Check

This reuses existing predictions; no retraining.

Trigger rule:

```text
done_prob >= 0.9
3-frame window
at least 2 votes
```

| run | strict | task0 | task1 | early | miss | tolerance@5 | tolerance@10 |
|---|---:|---:|---:|---:|---:|---:|---:|
| single-view baseline | 44/64 = 68.75% | 28/32 = 87.50% | 16/32 = 50.00% | 14 | 6 | 51/64 = 79.69% | 53/64 = 82.81% |
| structured 4-view | 44/64 = 68.75% | 27/32 = 84.38% | 17/32 = 53.12% | 18 | 2 | 47/64 = 73.44% | 52/64 = 81.25% |
| structured 4-view hardneg30x3 | 45/64 = 70.31% | 28/32 = 87.50% | 17/32 = 53.12% | 15 | 4 | 49/64 = 76.56% | 52/64 = 81.25% |

Compared with `10/5`, `3/2` is less conservative. It reduces misses slightly in
some runs, but increases early triggers and lowers strict episode accuracy.
Keep `10/5` for the current goal of avoiding early subtask switches.
