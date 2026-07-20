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
        "ok": 0,
        "early": 0,
        "miss": 0,
        "total": 0,
        "per_task": {},
    }
    for (task_id, segment_id), rows in sorted(segments.items()):
        rows.sort(key=lambda row: int(row["frame_index"]))
        done_frames = [int(row["frame_index"]) for row in rows if float(row["target_done"]) >= 0.5]
        if not done_frames:
            continue
        done_start = min(done_frames)
        event = first_vote_event(rows, args.threshold, args.window, args.votes)
        kind = "miss" if event is None else "early" if event < done_start else "ok"
        task = summary["per_task"].setdefault(str(task_id), {"ok": 0, "early": 0, "miss": 0, "total": 0})
        summary[kind] += 1
        summary["total"] += 1
        task[kind] += 1
        task["total"] += 1

    summary["accuracy"] = summary["ok"] / max(summary["total"], 1)
    for task in summary["per_task"].values():
        task["accuracy"] = task["ok"] / max(task["total"], 1)

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


if __name__ == "__main__":
    main()
