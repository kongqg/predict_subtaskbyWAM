"""Evaluation metrics for subtask progress prediction."""

from __future__ import annotations

import math
from typing import Any

import numpy as np


def regression_metrics(pred: np.ndarray, target: np.ndarray) -> dict[str, float]:
    err = pred - target
    return {
        "mae": float(np.mean(np.abs(err))),
        "rmse": float(math.sqrt(np.mean(err * err))),
    }


def progress_bin_accuracy(pred: np.ndarray, target: np.ndarray, bins: int = 10) -> float:
    pred_bin = np.clip((pred * bins).astype(int), 0, bins - 1)
    target_bin = np.clip((target * bins).astype(int), 0, bins - 1)
    return float(np.mean(pred_bin == target_bin))


def pairwise_monotonic_accuracy(
    pred: np.ndarray,
    target: np.ndarray,
    segment_ids: np.ndarray,
    target_margin: float = 0.05,
) -> float:
    total = 0
    correct = 0
    for segment_id in np.unique(segment_ids):
        idx = np.where(segment_ids == segment_id)[0]
        if idx.size < 2:
            continue
        t = target[idx]
        p = pred[idx]
        diff = t[:, None] - t[None, :]
        pairs = np.argwhere(diff > target_margin)
        if pairs.size == 0:
            continue
        total += pairs.shape[0]
        correct += int(np.sum(p[pairs[:, 0]] > p[pairs[:, 1]]))
    return float(correct / total) if total else float("nan")


def done_metrics(done_prob: np.ndarray, target: np.ndarray, threshold: float = 0.5) -> dict[str, Any]:
    pred = done_prob >= threshold
    target_bool = target.astype(bool)
    tp = int(np.sum(pred & target_bool))
    tn = int(np.sum(~pred & ~target_bool))
    fp = int(np.sum(pred & ~target_bool))
    fn = int(np.sum(~pred & target_bool))
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    return {
        "accuracy": float((tp + tn) / max(len(target), 1)),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "auroc": auroc(done_prob, target_bool),
        "confusion_matrix": {"tn": tn, "fp": fp, "fn": fn, "tp": tp},
    }


def auroc(scores: np.ndarray, target_bool: np.ndarray) -> float:
    pos = target_bool.astype(bool)
    n_pos = int(pos.sum())
    n_neg = int((~pos).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(scores)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(scores) + 1)
    rank_sum_pos = float(ranks[pos].sum())
    return float((rank_sum_pos - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def summarize_predictions(rows: list[dict[str, Any]], done_threshold: float = 0.5) -> dict[str, Any]:
    pred = np.asarray([r["pred_progress"] for r in rows], dtype=np.float64)
    target = np.asarray([r["target_progress"] for r in rows], dtype=np.float64)
    done_prob = np.asarray([r["done_probability"] for r in rows], dtype=np.float64)
    done_target = np.asarray([r["target_done"] for r in rows], dtype=np.float64)
    done_mask = np.asarray([r.get("done_loss_mask", 1.0) for r in rows], dtype=np.float64) > 0
    task_ids = np.asarray([r["task_id"] for r in rows], dtype=np.int64)
    segment_ids = np.asarray([r["segment_id"] for r in rows], dtype=np.int64)

    out: dict[str, Any] = {
        "progress": {
            **regression_metrics(pred, target),
            "bin_accuracy_10": progress_bin_accuracy(pred, target, bins=10),
            "pairwise_monotonic_accuracy": pairwise_monotonic_accuracy(pred, target, segment_ids),
        },
        "done": done_metrics(done_prob[done_mask], done_target[done_mask], done_threshold),
        "per_task": {},
    }
    for task_id in sorted(np.unique(task_ids).tolist()):
        mask = task_ids == task_id
        task_done_mask = mask & done_mask
        out["per_task"][str(task_id)] = {
            "progress": regression_metrics(pred[mask], target[mask]),
            "done": done_metrics(done_prob[task_done_mask], done_target[task_done_mask], done_threshold),
        }
    return out
