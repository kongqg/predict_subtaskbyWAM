# Done Short8 + Hardneg30x3 Results

Date: 2026-07-21

## Setup

- Data: `two_task_0_1_full`
- Train split: task0 and task1 manually annotated training episodes
- Validation split: 64 episodes, task0 = 32, task1 = 32
- Progress input: original structured 4-view V-JEPA feature, `[B, 32, 4, 768]`
- Done input: short V-JEPA feature, `[B, 8, 4, 768]`
- Done model: independent 1-layer Done Verifier, not sharing `task_hidden` with progress
- Hard negative: frames in `[done_start - 30, done_start)` get `done_loss_mask = 3.0`
- Loss: `0.2 * progress_loss + 1.0 * done_loss`
- LR: warmup cosine to step 6000, then 8000 continuation used fixed `1e-5`

## Train Loss

| step | lr | total loss | progress loss | done loss |
| --- | ---: | ---: | ---: | ---: | ---: |
| 2000 | 2.4995e-4 | 0.13899 | 0.000645 | 0.13886 |
| 4000 | 9.4765e-5 | 0.10487 | 0.000495 | 0.10477 |
| 6000 | 1.0000e-5 | 0.13622 | 0.000489 | 0.13612 |
| 8000 | 1.0000e-5 | 0.06407 | 0.000341 | 0.06400 |

## Episode Done Results

Default trigger: `done_prob >= 0.9`, `window = 10`, `votes = 5`.

| step | rule | overall | task0 | task1 | early | miss |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 2000 | 10/5 | 51/64 = 79.69% | 31/32 | 20/32 | 4 | 9 |
| 4000 | 10/5 | 46/64 = 71.88% | 29/32 | 17/32 | 8 | 10 |
| 6000 | 10/5 | 48/64 = 75.00% | 28/32 | 20/32 | 10 | 6 |
| 8000 | 10/5 | 47/64 = 73.44% | 27/32 | 20/32 | 11 | 6 |
| 2000 | 3/2 | 51/64 = 79.69% | 31/32 | 20/32 | 6 | 7 |
| 2000 | 1/1 | 51/64 = 79.69% | 30/32 | 21/32 | 8 | 5 |
| 4000 | 3/2 | 52/64 = 81.25% | 29/32 | 23/32 | 9 | 3 |
| 4000 | 1/1 | 52/64 = 81.25% | 29/32 | 23/32 | 9 | 3 |
| 6000 | 3/2 | 49/64 = 76.56% | 27/32 | 22/32 | 12 | 3 |
| 8000 | 3/2 | 48/64 = 75.00% | 26/32 | 22/32 | 13 | 3 |

## Frame-Level Metrics

| step | progress MAE | done precision | done recall | done F1 | done AUROC |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 2000 | 0.04212 | 0.8714 | 0.6974 | 0.7747 | 0.9763 |
| 4000 | 0.03870 | 0.8927 | 0.7125 | 0.7925 | 0.9825 |
| 6000 | 0.03926 | 0.8749 | 0.7432 | 0.8037 | 0.9817 |
| 8000 | 0.03925 | 0.8745 | 0.7428 | 0.8033 | 0.9821 |

## Readout

Best current checkpoint for the business trigger is:

`short8 hardneg30x3 checkpoint_4000.pt` with `done_prob >= 0.9`, `window = 3`, `votes = 2`.

It gets `52/64 = 81.25%` episode success, task0 `29/32`, task1 `23/32`.

Continuing to 6000/8000 lowers the final train loss and slightly improves frame-level done F1, but the episode trigger gets worse. For this task, checkpoint selection should use episode trigger accuracy first, not frame-level BCE/F1 alone.
