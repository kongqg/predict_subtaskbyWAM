"""Plotting helpers kept separate from model code."""

from __future__ import annotations

from pathlib import Path
from typing import Any


def plot_segment_curves(rows: list[dict[str, Any]], output_dir: str | Path, max_segments: int = 8) -> None:
    import matplotlib.pyplot as plt

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    by_segment: dict[int, list[dict[str, Any]]] = {}
    for row in rows:
        by_segment.setdefault(int(row["segment_id"]), []).append(row)

    for segment_id, segment_rows in list(by_segment.items())[:max_segments]:
        segment_rows = sorted(segment_rows, key=lambda r: r["frame_index"])
        x = [r["frame_index"] for r in segment_rows]
        plt.figure(figsize=(8, 4))
        plt.plot(x, [r["target_progress"] for r in segment_rows], label="target_progress")
        plt.plot(x, [r["pred_progress"] for r in segment_rows], label="pred_progress")
        plt.plot(x, [r["done_probability"] for r in segment_rows], label="done_probability")
        plt.ylim(-0.05, 1.05)
        plt.xlabel("frame_index")
        plt.legend()
        plt.tight_layout()
        plt.savefig(output / f"segment_{segment_id:06d}.png")
        plt.close()
