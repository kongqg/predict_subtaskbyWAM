"""Episode-level diagnostics for done-trigger behavior."""

from __future__ import annotations

from collections import Counter, defaultdict
import math
from pathlib import Path
from typing import Any

import numpy as np

VIEW_ORDER = ["cam_left_high", "cam_right_high", "cam_left_wrist", "cam_right_wrist"]


def summarize_diagnostics(
    rows: list[dict[str, Any]],
    threshold: float = 0.9,
    window: int = 10,
    votes: int = 5,
    view_order: list[str] | None = None,
) -> dict[str, Any]:
    episodes = episode_diagnostics(rows, threshold, window, votes)
    return {
        "trigger_config": {"threshold": threshold, "window": window, "votes": votes},
        "episodes": episodes,
        "strict": summarize_status(episodes, tolerance=0),
        "tolerance@5": summarize_status(episodes, tolerance=5),
        "tolerance@10": summarize_status(episodes, tolerance=10),
        "lead_lag": lead_lag_summary(episodes),
        "max_pre_done_votes": vote_summary(episodes),
        "max_pre_done_window_score": score_summary(episodes, "max_pre_done_window_score"),
        "pre_done_trigger_windows": score_summary(episodes, "pre_done_trigger_windows"),
        "distance_bins": distance_bin_summary(rows),
        "miss_analysis": miss_analysis(episodes),
        "attention": attention_summary(rows, episodes, window, view_order or VIEW_ORDER),
        "progress_done_coupling": progress_done_coupling(rows),
    }


def episode_diagnostics(rows: list[dict[str, Any]], threshold: float, window: int, votes: int) -> list[dict[str, Any]]:
    out = []
    for (task_id, episode), ep_rows in grouped_rows(rows).items():
        ep_rows = sorted(ep_rows, key=lambda r: int(r["frame_index"]))
        done_frames = [int(r["frame_index"]) for r in ep_rows if float(r["target_done"]) >= 0.5]
        if not done_frames:
            continue
        done_start = min(done_frames)
        trigger = first_vote_event(ep_rows, threshold, window, votes)
        pre_rows = [r for r in ep_rows if int(r["frame_index"]) < done_start]
        post_rows = [r for r in ep_rows if int(r["frame_index"]) >= done_start]
        pre_votes, pre_score, pre_windows = max_window_stats(pre_rows, threshold, window, votes)
        post_votes, post_score, _ = max_window_stats(post_rows, threshold, window, votes)
        row = {
            "task_id": task_id,
            "episode": episode,
            "done_start": done_start,
            "trigger_frame": trigger,
            "lead_lag": None if trigger is None else trigger - done_start,
            "early_by_frames": None if trigger is None else max(done_start - trigger, 0),
            "max_pre_done_votes": pre_votes,
            "max_pre_done_window_score": pre_score,
            "pre_done_trigger_windows": pre_windows,
        }
        if trigger is None:
            probs = [float(r["done_probability"]) for r in post_rows]
            row["miss"] = {
                "post_done_frames": len(post_rows),
                "max_post_done_probability": max(probs) if probs else None,
                "max_post_done_votes": post_votes,
                "max_post_done_window_score": post_score,
                "reached_votes_minus_one": post_votes == votes - 1,
                "missing_votes": max(votes - post_votes, 0),
            }
        out.append(row)
    return sorted(out, key=lambda r: (r["task_id"], r["episode"]))


def grouped_rows(rows: list[dict[str, Any]]) -> dict[tuple[int, int], list[dict[str, Any]]]:
    grouped: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(int(row["task_id"]), int(row["segment_id"]))].append(row)
    return grouped


def first_vote_event(rows: list[dict[str, Any]], threshold: float, window: int, votes: int) -> int | None:
    flags = [float(row["done_probability"]) >= threshold for row in rows]
    for i, row in enumerate(rows):
        if sum(flags[max(0, i - window + 1) : i + 1]) >= votes:
            return int(row["frame_index"])
    return None


def max_window_stats(rows: list[dict[str, Any]], threshold: float, window: int, votes: int) -> tuple[int, float, int]:
    probs = [float(r["done_probability"]) for r in rows]
    best_votes = 0
    best_score = 0.0
    trigger_windows = 0
    for i in range(len(probs)):
        chunk = probs[max(0, i - window + 1) : i + 1]
        vote_count = sum(p >= threshold for p in chunk)
        best_votes = max(best_votes, vote_count)
        if len(chunk) >= votes:
            score = sorted(chunk, reverse=True)[votes - 1]
            best_score = max(best_score, score)
            trigger_windows += int(score >= threshold)
    return int(best_votes), float(best_score), int(trigger_windows)


def summarize_status(episodes: list[dict[str, Any]], tolerance: int = 0) -> dict[str, Any]:
    out = {"ok": 0, "early": 0, "miss": 0, "total": 0, "accuracy": 0.0, "per_task": {}}
    for ep in episodes:
        trigger = ep["trigger_frame"]
        done_start = ep["done_start"]
        status = "miss" if trigger is None else "early" if trigger < done_start - tolerance else "ok"
        task = out["per_task"].setdefault(str(ep["task_id"]), {"ok": 0, "early": 0, "miss": 0, "total": 0})
        out[status] += 1
        out["total"] += 1
        task[status] += 1
        task["total"] += 1
    out["accuracy"] = out["ok"] / max(out["total"], 1)
    for task in out["per_task"].values():
        task["accuracy"] = task["ok"] / max(task["total"], 1)
    return out


def lead_lag_summary(episodes: list[dict[str, Any]]) -> dict[str, Any]:
    return by_scope(episodes, lambda eps: lead_lag_stats(eps))


def lead_lag_stats(episodes: list[dict[str, Any]]) -> dict[str, Any]:
    offsets = [ep["lead_lag"] for ep in episodes if ep["lead_lag"] is not None]
    early = [-x for x in offsets if x < 0]
    buckets = Counter()
    for frames in early:
        if frames <= 5:
            buckets["early_1_5"] += 1
        elif frames <= 10:
            buckets["early_6_10"] += 1
        elif frames <= 30:
            buckets["early_11_30"] += 1
        else:
            buckets["early_gt30"] += 1
    return {
        "triggered": len(offsets),
        "miss": len(episodes) - len(offsets),
        "mean": mean(offsets),
        "median": median(offsets),
        "max_early_frames": max(early) if early else 0,
        "early_buckets": dict(buckets),
    }


def vote_summary(episodes: list[dict[str, Any]]) -> dict[str, Any]:
    def stats(eps: list[dict[str, Any]]) -> dict[str, Any]:
        values = [int(ep["max_pre_done_votes"]) for ep in eps]
        hist = {str(i): values.count(i) for i in range(11)}
        return {"mean": mean(values), "median": median(values), "max": max(values) if values else 0, "histogram": hist}

    return by_scope(episodes, stats)


def score_summary(episodes: list[dict[str, Any]], key: str) -> dict[str, Any]:
    return by_scope(episodes, lambda eps: numeric_stats([float(ep[key]) for ep in eps]))


def miss_analysis(episodes: list[dict[str, Any]]) -> dict[str, Any]:
    misses = [ep for ep in episodes if ep["trigger_frame"] is None]
    return by_scope(misses, lambda eps: {
        "count": len(eps),
        "max_post_done_probability": numeric_stats([ep["miss"]["max_post_done_probability"] for ep in eps]),
        "max_post_done_votes": numeric_stats([ep["miss"]["max_post_done_votes"] for ep in eps]),
        "reached_votes_minus_one": sum(bool(ep["miss"]["reached_votes_minus_one"]) for ep in eps),
        "missing_votes": numeric_stats([ep["miss"]["missing_votes"] for ep in eps]),
    })


def distance_bin_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    done_start = done_start_by_episode(rows)
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    task_groups: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        offset = int(row["frame_index"]) - done_start[int(row["segment_id"])]
        name = distance_bin(offset)
        if not name:
            continue
        groups[name].append(row)
        task_groups[str(row["task_id"])][name].append(row)
    return {
        "overall": {name: probability_stats(rs) for name, rs in sorted(groups.items())},
        "per_task": {
            task: {name: probability_stats(rs) for name, rs in sorted(items.items())}
            for task, items in sorted(task_groups.items())
        },
    }


def distance_bin(offset: int) -> str | None:
    if offset <= -61:
        return "pre_gt60"
    if -60 <= offset <= -31:
        return "pre_31_60"
    if -30 <= offset <= -11:
        return "pre_11_30"
    if -10 <= offset <= -6:
        return "pre_6_10"
    if -5 <= offset <= -1:
        return "pre_1_5"
    if 0 <= offset <= 5:
        return "post_0_5"
    if 6 <= offset <= 10:
        return "post_6_10"
    if 11 <= offset <= 30:
        return "post_11_30"
    return None


def probability_stats(rows: list[dict[str, Any]]) -> dict[str, Any]:
    probs = [float(r["done_probability"]) for r in rows]
    out = numeric_stats(probs)
    out["prob_ge_0.5"] = ratio([p >= 0.5 for p in probs])
    out["prob_ge_0.9"] = ratio([p >= 0.9 for p in probs])
    return out


def attention_summary(
    rows: list[dict[str, Any]], episodes: list[dict[str, Any]], window: int, view_order: list[str]
) -> dict[str, Any]:
    if not rows or "view_attention_current" not in rows[0]:
        return {"available": False}
    done_start = done_start_by_episode(rows)
    phases: dict[str, list[dict[str, Any]]] = defaultdict(list)
    task_phases: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        offset = int(row["frame_index"]) - done_start[int(row["segment_id"])]
        phase = attention_phase(offset)
        if phase:
            phases[phase].append(row)
            task_phases[str(row["task_id"])][phase].append(row)

    grouped = grouped_rows(rows)
    trigger_groups = {"early_trigger_window": [], "normal_trigger_window": []}
    for ep in episodes:
        trigger = ep["trigger_frame"]
        if trigger is None:
            continue
        ep_rows = sorted(grouped[(int(ep["task_id"]), int(ep["episode"]))], key=lambda r: int(r["frame_index"]))
        trigger_idx = next(i for i, r in enumerate(ep_rows) if int(r["frame_index"]) == trigger)
        key = "early_trigger_window" if trigger < ep["done_start"] else "normal_trigger_window"
        trigger_groups[key].extend(ep_rows[max(0, trigger_idx - window + 1) : trigger_idx + 1])

    return {
        "available": True,
        "view_order": view_order[: len(rows[0]["view_attention_current"])],
        "overall_by_phase": {k: attention_stats(v) for k, v in sorted(phases.items())},
        "per_task_by_phase": {
            task: {phase: attention_stats(rs) for phase, rs in sorted(items.items())}
            for task, items in sorted(task_phases.items())
        },
        "trigger_windows": {k: attention_stats(v) for k, v in trigger_groups.items()},
    }


def attention_phase(offset: int) -> str | None:
    if -60 <= offset <= -31:
        return "pre_31_60"
    if -30 <= offset <= -11:
        return "pre_11_30"
    if -10 <= offset <= -1:
        return "pre_1_10"
    if 0 <= offset <= 10:
        return "post_0_10"
    return None


def attention_stats(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"n": 0}
    current = np.asarray([r["view_attention_current"] for r in rows], dtype=np.float64)
    history = np.asarray([r.get("view_attention_history_mean", r.get("view_attention")) for r in rows], dtype=np.float64)
    entropy = [float(r["view_attention_entropy_current"]) for r in rows]
    return {
        "n": int(len(rows)),
        "current_mean": current.mean(axis=0).tolist(),
        "history_mean": history.mean(axis=0).tolist(),
        "entropy_current_mean": mean(entropy),
    }


def progress_done_coupling(rows: list[dict[str, Any]]) -> dict[str, Any]:
    neg = [r for r in rows if float(r["target_done"]) < 0.5]
    pred = np.asarray([float(r["pred_progress"]) for r in neg], dtype=np.float64)
    prob = np.asarray([float(r["done_probability"]) for r in neg], dtype=np.float64)
    corr = float(np.corrcoef(pred, prob)[0, 1]) if len(pred) > 1 and pred.std() > 0 and prob.std() > 0 else float("nan")
    bins = {}
    for lo in np.linspace(0.0, 0.8, 5):
        hi = lo + 0.2
        mask = (pred >= lo) & (pred < hi if hi < 1.0 else pred <= hi)
        bins[f"{lo:.1f}-{hi:.1f}"] = probability_array_stats(prob[mask])
    return {"target_done_0": {"n": len(neg), "pred_progress_done_prob_corr": corr, "pred_progress_bins": bins}}


def done_start_by_episode(rows: list[dict[str, Any]]) -> dict[int, int]:
    out = {}
    for row in rows:
        if float(row["target_done"]) >= 0.5:
            ep = int(row["segment_id"])
            out[ep] = min(out.get(ep, int(row["frame_index"])), int(row["frame_index"]))
    return out


def by_scope(items: list[dict[str, Any]], fn: Any) -> dict[str, Any]:
    out = {"overall": fn(items), "per_task": {}}
    by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in items:
        by_task[str(item["task_id"])].append(item)
    out["per_task"] = {task: fn(values) for task, values in sorted(by_task.items())}
    return out


def numeric_stats(values: list[Any]) -> dict[str, Any]:
    clean = [float(v) for v in values if v is not None and not math.isnan(float(v))]
    return {
        "n": len(clean),
        "mean": mean(clean),
        "median": median(clean),
        "p90": quantile(clean, 0.9),
        "max": max(clean) if clean else None,
    }


def probability_array_stats(values: np.ndarray) -> dict[str, Any]:
    vals = values.astype(np.float64)
    if vals.size == 0:
        return {"n": 0, "mean": None, "median": None, "p90": None, "prob_ge_0.9": None}
    return {
        "n": int(vals.size),
        "mean": float(vals.mean()),
        "median": float(np.quantile(vals, 0.5)),
        "p90": float(np.quantile(vals, 0.9)),
        "prob_ge_0.9": float((vals >= 0.9).mean()),
    }


def mean(values: list[Any]) -> float | None:
    return float(np.mean(values)) if values else None


def median(values: list[Any]) -> float | None:
    return float(np.median(values)) if values else None


def quantile(values: list[Any], q: float) -> float | None:
    return float(np.quantile(values, q)) if values else None


def ratio(flags: list[bool]) -> float | None:
    return float(np.mean(flags)) if flags else None


def write_diagnostic_plots(
    rows: list[dict[str, Any]],
    diagnostics: dict[str, Any],
    output_dir: str | Path,
    pre: int = 90,
    post: int = 60,
) -> None:
    import matplotlib.pyplot as plt

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    plot_done_aligned_curve(rows, output / "done_aligned_probability_curve.png", pre, post, plt)
    plot_probability_heatmap(rows, diagnostics["episodes"], output / "per_episode_probability_heatmap.png", pre, post, plt)
    plot_lead_lag_histogram(diagnostics["episodes"], output / "trigger_lead_lag_histogram.png", plt)
    plot_max_pre_done_votes(diagnostics["episodes"], output / "max_pre_done_votes_histogram.png", plt)
    if diagnostics["attention"].get("available"):
        plot_attention_by_phase(diagnostics["attention"], output / "view_attention_by_phase.png", plt)


def aligned_probs(rows: list[dict[str, Any]], pre: int, post: int) -> dict[int, list[float]]:
    starts = done_start_by_episode(rows)
    out: dict[int, list[float]] = {i: [] for i in range(-pre, post + 1)}
    for row in rows:
        offset = int(row["frame_index"]) - starts[int(row["segment_id"])]
        if -pre <= offset <= post:
            out[offset].append(float(row["done_probability"]))
    return out


def plot_done_aligned_curve(rows: list[dict[str, Any]], path: Path, pre: int, post: int, plt: Any) -> None:
    data = aligned_probs(rows, pre, post)
    xs = np.asarray(sorted(data))
    med = np.asarray([np.quantile(data[x], 0.5) if data[x] else np.nan for x in xs])
    q10 = np.asarray([np.quantile(data[x], 0.1) if data[x] else np.nan for x in xs])
    q90 = np.asarray([np.quantile(data[x], 0.9) if data[x] else np.nan for x in xs])
    plt.figure(figsize=(9, 4))
    plt.fill_between(xs, q10, q90, alpha=0.25, label="10%-90%")
    plt.plot(xs, med, label="median")
    plt.axvline(0, color="black", linewidth=1)
    plt.axhline(0.9, color="red", linestyle="--", linewidth=1)
    plt.xlabel("frame offset from done_start")
    plt.ylabel("done probability")
    plt.ylim(-0.02, 1.02)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def plot_probability_heatmap(
    rows: list[dict[str, Any]], episodes: list[dict[str, Any]], path: Path, pre: int, post: int, plt: Any
) -> None:
    starts = done_start_by_episode(rows)
    by_ep = grouped_rows(rows)
    episodes = sorted(episodes, key=lambda e: (e["task_id"], 999999 if e["lead_lag"] is None else e["lead_lag"]))
    xs = list(range(-pre, post + 1))
    matrix = np.full((len(episodes), len(xs)), np.nan)
    for row_i, ep in enumerate(episodes):
        for row in by_ep[(int(ep["task_id"]), int(ep["episode"]))]:
            offset = int(row["frame_index"]) - starts[int(row["segment_id"])]
            if -pre <= offset <= post:
                matrix[row_i, offset + pre] = float(row["done_probability"])
    plt.figure(figsize=(10, max(4, len(episodes) * 0.08)))
    plt.imshow(matrix, aspect="auto", vmin=0, vmax=1, cmap="viridis", extent=[-pre, post, len(episodes), 0])
    plt.axvline(0, color="white", linewidth=1)
    plt.xlabel("frame offset from done_start")
    plt.ylabel("episode")
    plt.colorbar(label="done probability")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def plot_lead_lag_histogram(episodes: list[dict[str, Any]], path: Path, plt: Any) -> None:
    offsets = [ep["lead_lag"] for ep in episodes if ep["lead_lag"] is not None]
    misses = sum(ep["lead_lag"] is None for ep in episodes)
    plt.figure(figsize=(7, 4))
    plt.hist(offsets, bins=30)
    plt.axvline(0, color="black", linewidth=1)
    plt.title(f"trigger lead/lag, miss={misses}")
    plt.xlabel("trigger_frame - done_start")
    plt.ylabel("episodes")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def plot_max_pre_done_votes(episodes: list[dict[str, Any]], path: Path, plt: Any) -> None:
    values = [int(ep["max_pre_done_votes"]) for ep in episodes]
    counts = [values.count(i) for i in range(11)]
    plt.figure(figsize=(7, 4))
    plt.bar(range(11), counts)
    plt.axvline(5, color="red", linestyle="--", linewidth=1)
    plt.xlabel("max pre-done votes in a 10-frame window")
    plt.ylabel("episodes")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def plot_attention_by_phase(attention: dict[str, Any], path: Path, plt: Any) -> None:
    phases = ["pre_31_60", "pre_11_30", "pre_1_10", "post_0_10"]
    views = attention["view_order"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    for ax, task in zip(axes, ["0", "1"]):
        data = attention["per_task_by_phase"].get(task, {})
        x = np.arange(len(phases))
        width = 0.18
        for view_idx, view in enumerate(views):
            vals = [data.get(phase, {}).get("current_mean", [np.nan] * len(views))[view_idx] for phase in phases]
            ax.bar(x + (view_idx - 1.5) * width, vals, width, label=view)
        ax.set_title(f"task{task}")
        ax.set_xticks(x)
        ax.set_xticklabels(phases, rotation=25, ha="right")
        ax.set_ylim(0, 1)
    axes[0].set_ylabel("current-frame attention")
    axes[1].legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()
