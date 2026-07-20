"""Evaluate a trained subtask progress checkpoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from .config import load_config
from .metrics import summarize_predictions
from .model import SubtaskProgressTransformer, SubtaskProgressTransformerConfig
from .plotting import plot_segment_curves
from .train import build_dataset, build_model_config, predict_rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", default="val", choices=["train", "val"])
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--device", default="")
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = torch.device(args.device or cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
    dataset = build_dataset(cfg, args.split)
    visual_dim, proprio_dim = dataset.infer_dims()

    ckpt = torch.load(args.checkpoint, map_location=device)
    model_cfg_dict = ckpt.get("model_config") or vars(build_model_config(cfg, visual_dim, proprio_dim))
    model = SubtaskProgressTransformer(SubtaskProgressTransformerConfig(**model_cfg_dict)).to(device)
    model.load_state_dict(ckpt["model"])

    loader = DataLoader(
        dataset,
        batch_size=int(cfg["training"].get("eval_batch_size", cfg["training"].get("batch_size", 64))),
        shuffle=False,
        num_workers=int(cfg["training"].get("num_workers", 0)),
    )
    rows = predict_rows(model, loader, device)
    metrics = summarize_predictions(rows, cfg["evaluation"].get("done_threshold", 0.5))

    output_dir = Path(args.output_dir or Path(args.checkpoint).parent / f"eval_{args.split}")
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    with open(output_dir / "predictions.jsonl", "w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    plot_segment_curves(rows, output_dir / "curves", cfg["evaluation"].get("num_curve_plots", 8))
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
