"""Combine per-view feature parquets into one feature stream."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--feature-root", action="append", required=True)
    parser.add_argument("--feature-column", default="visual_features")
    parser.add_argument("--mode", choices=["concat", "stack"], default="concat")
    parser.add_argument("--output-format", choices=["parquet", "npy"], default="parquet")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    roots = [Path(p) for p in args.feature_root]
    output_root = Path(args.output_root)
    out_data = output_root / "data" / "chunk-000"
    out_meta = output_root / "meta"
    out_data.mkdir(parents=True, exist_ok=True)
    out_meta.mkdir(parents=True, exist_ok=True)

    first_files = sorted((roots[0] / "data").rglob("episode_*.parquet"))
    if not first_files:
        raise FileNotFoundError(f"no feature parquets under {roots[0] / 'data'}")

    written = []
    for first in first_files:
        out_path = out_data / (first.with_suffix(".npy").name if args.output_format == "npy" else first.name)
        if out_path.exists() and not args.overwrite:
            written.append(int(first.stem.split("_")[-1]))
            continue

        dfs = [pd.read_parquet(root / "data" / "chunk-000" / first.name) for root in roots]
        frame_index = dfs[0]["frame_index"].to_numpy()
        for root, df in zip(roots[1:], dfs[1:]):
            if len(df) != len(dfs[0]) or not np.array_equal(df["frame_index"].to_numpy(), frame_index):
                raise ValueError(f"frame mismatch in {root / 'data' / 'chunk-000' / first.name}")

        view_arrays = [
            np.stack([np.asarray(x, dtype=np.float32) for x in df[args.feature_column].to_list()], axis=0)
            for df in dfs
        ]
        array = np.concatenate(view_arrays, axis=1) if args.mode == "concat" else np.stack(view_arrays, axis=1)

        if args.output_format == "npy":
            tmp_path = out_path.with_suffix(".tmp.npy")
            with open(tmp_path, "wb") as f:
                np.save(f, array)
            tmp_path.replace(out_path)
            frames = int(array.shape[0])
        else:
            features = [x for x in array] if args.mode == "concat" else [x.tolist() for x in array]
            tmp_path = out_path.with_suffix(".tmp.parquet")
            pd.DataFrame({"frame_index": frame_index, args.feature_column: features}).to_parquet(tmp_path)
            tmp_path.replace(out_path)
            frames = len(features)
        episode_index = int(first.stem.split("_")[-1])
        written.append(episode_index)
        print(json.dumps({"episode_index": episode_index, "frames": frames}, ensure_ascii=False), flush=True)

    info: dict[str, Any] = {
        "source_feature_roots": [str(root) for root in roots],
        "feature_column": args.feature_column,
        "mode": args.mode,
        "output_format": args.output_format,
        "num_views": len(roots),
        "episodes": sorted(written),
    }
    with open(out_meta / "feature_info.json", "w") as f:
        json.dump(info, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
