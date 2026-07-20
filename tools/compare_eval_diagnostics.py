#!/usr/bin/env python3
"""Compare strict episode outcomes across eval diagnostics files."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("runs", nargs="+", help="name=/path/to/diagnostics.json")
    parser.add_argument("--output", default="")
    args = parser.parse_args()

    runs = {}
    for spec in args.runs:
        name, path = spec.split("=", 1)
        runs[name] = load_statuses(Path(path))

    keys = sorted(set.intersection(*(set(statuses) for statuses in runs.values())))
    names = list(runs)
    rows = []
    combo = Counter()
    for key in keys:
        statuses = {name: runs[name][key] for name in names}
        combo[tuple(statuses[name] for name in names)] += 1
        rows.append({"task_id": key[0], "episode": key[1], **statuses})

    pairwise = {}
    for i, left in enumerate(names):
        for right in names[i + 1 :]:
            table = Counter((runs[left][key], runs[right][key]) for key in keys)
            pairwise[f"{left}_vs_{right}"] = {f"{a}->{b}": n for (a, b), n in sorted(table.items())}

    out = {
        "runs": names,
        "episodes": len(keys),
        "pairwise": pairwise,
        "combination_counts": {"|".join(k): v for k, v in sorted(combo.items())},
        "per_episode": rows,
    }
    text = json.dumps(out, ensure_ascii=False, indent=2)
    if args.output:
        Path(args.output).write_text(text + "\n")
    print(text)


def load_statuses(path: Path) -> dict[tuple[int, int], str]:
    data = json.loads(path.read_text())
    out = {}
    for episode in data["episodes"]:
        trigger = episode["trigger_frame"]
        done_start = episode["done_start"]
        status = "miss" if trigger is None else "early" if trigger < done_start else "ok"
        out[(int(episode["task_id"]), int(episode["episode"]))] = status
    return out


if __name__ == "__main__":
    main()
