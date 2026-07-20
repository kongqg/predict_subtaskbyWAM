"""Train the subtask progress Transformer."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import random
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from .config import deep_update, load_config
from .dataset import SubtaskProgressDataset
from .losses import ProgressLoss
from .metrics import summarize_predictions
from .model import SubtaskProgressTransformer, SubtaskProgressTransformerConfig


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--resume", default="")
    parser.add_argument("--tiny-overfit", action="store_true")
    parser.add_argument("--device", default="")
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(int(cfg.get("seed", 0)))
    device = torch.device(args.device or cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
    output_dir = Path(cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    train_ds = build_dataset(cfg, "train", tiny_overfit=args.tiny_overfit)
    val_ds = build_dataset(cfg, "val", tiny_overfit=args.tiny_overfit)
    visual_dim, proprio_dim = train_ds.infer_dims()
    model_cfg = build_model_config(cfg, visual_dim, proprio_dim)
    model = SubtaskProgressTransformer(model_cfg).to(device)
    loss_fn = ProgressLoss(**cfg.get("loss", {})).to(device)
    optim = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg["training"].get("learning_rate", 1e-4)),
        weight_decay=float(cfg["training"].get("weight_decay", 1e-5)),
    )
    scheduler = build_lr_scheduler(optim, cfg)

    start_step = 0
    best_mae = float("inf")
    if args.resume:
        start_step, best_mae = load_checkpoint(args.resume, model, optim, device, scheduler)

    train_loader = DataLoader(
        train_ds,
        batch_size=int(cfg["training"].get("batch_size", 64)),
        shuffle=True,
        num_workers=int(cfg["training"].get("num_workers", 0)),
        pin_memory=device.type == "cuda",
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=int(cfg["training"].get("eval_batch_size", cfg["training"].get("batch_size", 64))),
        shuffle=False,
        num_workers=int(cfg["training"].get("num_workers", 0)),
        pin_memory=device.type == "cuda",
    )

    max_steps = int(cfg["training"].get("max_steps", 10000))
    log_every = int(cfg["training"].get("log_every", 50))
    eval_every = int(cfg["training"].get("eval_every", 1000))
    checkpoint_every = int(cfg["training"].get("checkpoint_every", eval_every))
    grad_clip = float(cfg["training"].get("grad_clip", 1.0))
    amp = bool(cfg["training"].get("amp", True)) and device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=amp)
    log_path = output_dir / "train_log.jsonl"

    loader_iter = iter(train_loader)
    for step in range(start_step + 1, max_steps + 1):
        try:
            batch = next(loader_iter)
        except StopIteration:
            loader_iter = iter(train_loader)
            batch = next(loader_iter)

        model.train()
        batch = to_device(batch, device)
        optim.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, enabled=amp):
            output = model(
                batch["visual_features"],
                batch["start_visual"],
                batch["task_ids"],
                batch["proprio"],
                batch["padding_mask"],
            )
            losses = loss_fn(
                output["progress"],
                output["done_logit"],
                batch["target_progress"],
                batch["target_done"],
                batch["segment_ids"],
                batch.get("done_loss_mask"),
            )
        scaler.scale(losses["loss"]).backward()
        if grad_clip > 0:
            scaler.unscale_(optim)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        scaler.step(optim)
        scaler.update()
        if scheduler is not None and not is_plateau_scheduler(scheduler):
            scheduler.step()

        if step % log_every == 0 or step == 1:
            row = {"step": step, "lr": current_lr(optim), **{k: float(v) for k, v in losses.items()}}
            append_jsonl(log_path, row)
            print(json.dumps(row, ensure_ascii=False))

        if step % eval_every == 0 or step == max_steps:
            rows = predict_rows(model, val_loader, device)
            metrics = summarize_predictions(rows, cfg["evaluation"].get("done_threshold", 0.5))
            mae = metrics["progress"]["mae"]
            write_json(output_dir / "last_eval.json", metrics)
            if scheduler is not None and is_plateau_scheduler(scheduler):
                scheduler.step(mae)
            if mae < best_mae:
                best_mae = mae
                save_checkpoint(output_dir / "best.pt", model, optim, scheduler, step, best_mae, cfg, model_cfg)
            print(
                json.dumps(
                    {"step": step, "val_mae": mae, "best_mae": best_mae, "lr": current_lr(optim)},
                    ensure_ascii=False,
                )
            )

        if step % checkpoint_every == 0 or step == max_steps:
            save_checkpoint(
                output_dir / f"checkpoint_{step}.pt", model, optim, scheduler, step, best_mae, cfg, model_cfg
            )


def build_lr_scheduler(
    optim: torch.optim.Optimizer, cfg: dict[str, Any]
) -> Any | None:
    scheduler_cfg = cfg["training"].get("lr_scheduler") or {}
    if scheduler_cfg.get("type", "none") in {None, "none"}:
        return None
    if scheduler_cfg["type"] == "warmup_cosine":
        max_steps = int(scheduler_cfg.get("max_steps", cfg["training"].get("max_steps", 10000)))
        warmup_steps = int(scheduler_cfg.get("warmup_steps", 0))
        min_lr = float(scheduler_cfg.get("min_lr", 0.0))
        base_lr = float(cfg["training"].get("learning_rate", 1e-4))
        min_ratio = min_lr / base_lr if base_lr > 0 else 0.0

        def lr_lambda(step: int) -> float:
            if warmup_steps > 0 and step < warmup_steps:
                return min_ratio + (1.0 - min_ratio) * step / warmup_steps
            denom = max(max_steps - warmup_steps, 1)
            progress = min(max((step - warmup_steps) / denom, 0.0), 1.0)
            return min_ratio + 0.5 * (1.0 - min_ratio) * (1.0 + math.cos(math.pi * progress))

        return torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda=lr_lambda)
    if scheduler_cfg["type"] != "reduce_on_plateau":
        raise ValueError("training.lr_scheduler.type only supports warmup_cosine, reduce_on_plateau, or none")
    return torch.optim.lr_scheduler.ReduceLROnPlateau(
        optim,
        mode="min",
        factor=float(scheduler_cfg.get("factor", 0.5)),
        patience=int(scheduler_cfg.get("patience", 0)),
        threshold=float(scheduler_cfg.get("threshold", 1e-4)),
        threshold_mode=str(scheduler_cfg.get("threshold_mode", "rel")),
        cooldown=int(scheduler_cfg.get("cooldown", 0)),
        min_lr=float(scheduler_cfg.get("min_lr", 1e-5)),
    )


def current_lr(optim: torch.optim.Optimizer) -> float:
    return float(optim.param_groups[0]["lr"])


def is_plateau_scheduler(scheduler: Any) -> bool:
    return isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau)


def build_dataset(cfg: dict[str, Any], split: str, tiny_overfit: bool = False) -> SubtaskProgressDataset:
    ds_cfg = deep_update(dict(cfg["dataset"]), cfg.get(f"{split}_dataset", {}))
    root = ds_cfg.pop(f"{split}_root")
    ds_cfg.pop("train_root", None)
    ds_cfg.pop("val_root", None)
    feature_root = ds_cfg.get("feature_root")
    if isinstance(feature_root, dict):
        ds_cfg["feature_root"] = feature_root.get(split)
    ds_cfg.setdefault("history_length", int(cfg["model"]["history_length"]))
    if tiny_overfit:
        ds_cfg["max_samples"] = int(cfg.get("tiny_overfit", {}).get("max_samples", 32))
        if split == "val":
            root = cfg["dataset"]["train_root"]
    return SubtaskProgressDataset(root=root, **ds_cfg)


def build_model_config(
    cfg: dict[str, Any], inferred_visual_dim: int, inferred_proprio_dim: int
) -> SubtaskProgressTransformerConfig:
    model_cfg = dict(cfg["model"])
    model_cfg["visual_dim"] = int(model_cfg.get("visual_dim") or inferred_visual_dim)
    model_cfg["proprio_dim"] = int(
        inferred_proprio_dim if model_cfg.get("proprio_dim") is None else model_cfg["proprio_dim"]
    )
    return SubtaskProgressTransformerConfig(**model_cfg)


def predict_rows(model: SubtaskProgressTransformer, loader: DataLoader, device: torch.device) -> list[dict[str, Any]]:
    model.eval()
    rows: list[dict[str, Any]] = []
    with torch.no_grad():
        for batch in loader:
            batch = to_device(batch, device)
            output = model(
                batch["visual_features"],
                batch["start_visual"],
                batch["task_ids"],
                batch["proprio"],
                batch["padding_mask"],
            )
            done_prob = torch.sigmoid(output["done_logit"])
            done_loss_mask = batch.get("done_loss_mask", torch.ones_like(batch["target_done"]))
            for i in range(output["progress"].shape[0]):
                rows.append(
                    {
                        "pred_progress": float(output["progress"][i].cpu()),
                        "target_progress": float(batch["target_progress"][i].cpu()),
                        "done_probability": float(done_prob[i].cpu()),
                        "target_done": float(batch["target_done"][i].cpu()),
                        "done_loss_mask": float(done_loss_mask[i].cpu()),
                        "task_id": int(batch["task_ids"][i].cpu()),
                        "segment_id": int(batch["segment_ids"][i].cpu()),
                        "frame_index": int(batch["frame_index"][i].cpu()),
                    }
                )
    return rows


def to_device(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {k: v.to(device, non_blocking=True) if torch.is_tensor(v) else v for k, v in batch.items()}


def save_checkpoint(
    path: Path,
    model: SubtaskProgressTransformer,
    optim: torch.optim.Optimizer,
    scheduler: Any | None,
    step: int,
    best_mae: float,
    cfg: dict[str, Any],
    model_cfg: SubtaskProgressTransformerConfig,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optim.state_dict(),
            "scheduler": scheduler.state_dict() if scheduler is not None else None,
            "step": step,
            "best_mae": best_mae,
            "config": cfg,
            "model_config": vars(model_cfg),
        },
        path,
    )


def load_checkpoint(
    path: str | Path,
    model: SubtaskProgressTransformer,
    optim: torch.optim.Optimizer,
    device: torch.device,
    scheduler: Any | None = None,
) -> tuple[int, float]:
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model"])
    optim.load_state_dict(ckpt["optimizer"])
    if scheduler is not None and ckpt.get("scheduler"):
        scheduler.load_state_dict(ckpt["scheduler"])
    return int(ckpt.get("step", 0)), float(ckpt.get("best_mae", float("inf")))


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    with open(path, "a") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_json(path: Path, row: dict[str, Any]) -> None:
    with open(path, "w") as f:
        json.dump(row, f, ensure_ascii=False, indent=2)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


if __name__ == "__main__":
    main()
