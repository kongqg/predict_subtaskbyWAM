from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--threshold", type=float, default=0.9)
    parser.add_argument("--window", type=int, default=10)
    parser.add_argument("--votes", type=int, default=5)
    parser.add_argument("--output", default="")
    args = parser.parse_args()

    segments: dict[tuple[int, int], list[dict]] = defaultdict(list)
    with open(args.predictions) as f:
        for line in f:
            if line.strip():
                row = json.loads(line)
                segments[(int(row["task_id"]), int(row["segment_id"]))].append(row)

    summary = {
        "threshold": args.threshold,
        "window": args.window,
        "votes": args.votes,
        "total": 0,
        "episodes": [],
    }
    for (task_id, segment_id), rows in sorted(segments.items()):
        rows.sort(key=lambda row: int(row["frame_index"]))
        done_frames = [int(row["frame_index"]) for row in rows if float(row["target_done"]) >= 0.5]
        if not done_frames:
            continue
        done_start = min(done_frames)
        event = first_vote_event(rows, args.threshold, args.window, args.votes)
        summary["total"] += 1
        summary["episodes"].append(
            {
                "task_id": task_id,
                "episode": segment_id,
                "trigger_frame": event,
                "done_start": done_start,
                "early_by_frames": None if event is None else max(done_start - event, 0),
                "trigger_offset_frames": None if event is None else event - done_start,
            }
        )

    strict = summarize_with_tolerance(summary["episodes"], tolerance=0)
    summary.update({k: strict[k] for k in ("ok", "early", "miss", "per_task", "accuracy")})
    summary["strict"] = strict
    summary["tolerance@5"] = summarize_with_tolerance(summary["episodes"], tolerance=5)
    summary["tolerance@10"] = summarize_with_tolerance(summary["episodes"], tolerance=10)

    text = json.dumps(summary, ensure_ascii=False, indent=2)
    if args.output:
        Path(args.output).write_text(text + "\n")
    print(text)


def first_vote_event(rows: list[dict], threshold: float, window: int, votes: int) -> int | None:
    flags = [float(row["done_probability"]) >= threshold for row in rows]
    for i, row in enumerate(rows):
        lo = max(0, i - window + 1)
        if sum(flags[lo : i + 1]) >= votes:
            return int(row["frame_index"])
    return None


def summarize_with_tolerance(episodes: list[dict], tolerance: int) -> dict:
    out = {"ok": 0, "early": 0, "miss": 0, "total": 0, "per_task": {}, "episodes": []}
    for episode in episodes:
        event = episode["trigger_frame"]
        done_start = episode["done_start"]
        kind = "miss" if event is None else "early" if event < done_start - tolerance else "ok"
        task = out["per_task"].setdefault(str(episode["task_id"]), {"ok": 0, "early": 0, "miss": 0, "total": 0})
        out[kind] += 1
        out["total"] += 1
        task[kind] += 1
        task["total"] += 1
        row = dict(episode)
        row["status"] = kind
        row["tolerance"] = tolerance
        out["episodes"].append(row)
    out["accuracy"] = out["ok"] / max(out["total"], 1)
    for task in out["per_task"].values():
        task["accuracy"] = task["ok"] / max(task["total"], 1)
    return out


if __name__ == "__main__":
    main()
