"""Offline V-JEPA feature extraction for LeRobot video episodes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time
from typing import Any

import numpy as np
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, help="LeRobot split root")
    parser.add_argument("--output-root", required=True, help="Feature dataset root")
    parser.add_argument("--repo-dir", default="models/vjepa2")
    parser.add_argument(
        "--checkpoint",
        default="models/vjepa2_1_vit_base_384/vjepa2_1_vitb_dist_vitG_384.pt",
    )
    parser.add_argument("--model-name", default="vjepa2_1_vit_base_384")
    parser.add_argument("--checkpoint-key", default="ema_encoder")
    parser.add_argument("--camera", default="observation.images.cam_left_high")
    parser.add_argument("--feature-column", default="visual_features")
    parser.add_argument("--image-size", type=int, default=384)
    parser.add_argument("--clip-length", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default="")
    parser.add_argument("--max-episodes", type=int, default=0)
    parser.add_argument("--episode-index", type=int, action="append", default=[])
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--torch-hub-dir", default="")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    import torch

    root = Path(args.root)
    output_root = Path(args.output_root)
    out_data = output_root / "data" / "chunk-000"
    out_data.mkdir(parents=True, exist_ok=True)
    (output_root / "meta").mkdir(parents=True, exist_ok=True)
    if args.torch_hub_dir:
        torch.hub.set_dir(args.torch_hub_dir)

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    encoder = load_encoder(args, device)
    rows = select_episodes(
        load_episode_rows(root),
        args.episode_index,
        args.max_episodes,
        args.num_shards,
        args.shard_index,
    )
    for row in rows:
        episode_index = int(row["episode_index"])
        out_path = out_data / f"episode_{episode_index:06d}.parquet"
        if out_path.exists() and not args.overwrite:
            print(f"skip existing {out_path}", flush=True)
            continue

        video_path = find_video(root, args.camera, episode_index)
        length = int(row["length"])
        frames = read_video_rgb(video_path, length)
        features, timing = encode_frames(
            encoder=encoder,
            frames=frames,
            image_size=args.image_size,
            clip_length=args.clip_length,
            batch_size=args.batch_size,
            device=device,
        )
        tmp_path = out_path.with_suffix(".tmp.parquet")
        pd.DataFrame(
            {
                "frame_index": np.arange(features.shape[0], dtype=np.int64),
                args.feature_column: [x.astype(np.float32) for x in features],
            }
        ).to_parquet(tmp_path)
        tmp_path.replace(out_path)
        print(
            json.dumps(
                {"episode_index": episode_index, "frames": len(features), "path": str(out_path), **timing},
                ensure_ascii=False,
            ),
            flush=True,
        )

    write_feature_info(output_root, args, rows)


def load_encoder(args: argparse.Namespace, device: Any):
    import sys
    import torch

    repo_dir = Path(args.repo_dir)
    if str(repo_dir.resolve()) not in sys.path:
        sys.path.insert(0, str(repo_dir.resolve()))
    loaded = torch.hub.load(
        str(repo_dir),
        args.model_name,
        source="local",
        pretrained=False,
        num_frames=args.clip_length,
    )
    encoder = loaded[0] if isinstance(loaded, tuple) else loaded
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    state = ckpt.get(args.checkpoint_key)
    if state is None:
        raise KeyError(f"{args.checkpoint_key!r} not found in checkpoint keys: {list(ckpt.keys())}")
    state = {k.replace("module.", "").replace("backbone.", ""): v for k, v in state.items()}
    missing, unexpected = encoder.load_state_dict(state, strict=False)
    if unexpected:
        print(f"unexpected encoder keys: {len(unexpected)}")
    if missing:
        print(f"missing encoder keys: {len(missing)}")
    return encoder.to(device).eval()


def load_episode_rows(root: Path) -> list[dict[str, Any]]:
    path = root / "meta" / "episodes.jsonl"
    with open(path, "r") as f:
        return [json.loads(line) for line in f if line.strip()]


def select_episodes(
    rows: list[dict[str, Any]],
    episode_indices: list[int],
    max_episodes: int,
    num_shards: int = 1,
    shard_index: int = 0,
) -> list[dict[str, Any]]:
    if num_shards < 1:
        raise ValueError("--num-shards must be >= 1")
    if shard_index < 0 or shard_index >= num_shards:
        raise ValueError("--shard-index must be in [0, num_shards)")
    if episode_indices:
        wanted = set(episode_indices)
        rows = [row for row in rows if int(row["episode_index"]) in wanted]
    rows = [row for i, row in enumerate(rows) if i % num_shards == shard_index]
    if max_episodes > 0:
        rows = rows[:max_episodes]
    if not rows:
        raise ValueError("no episodes selected")
    return rows


def find_video(root: Path, camera: str, episode_index: int) -> Path:
    matches = sorted((root / "videos").glob(f"**/{camera}/episode_{episode_index:06d}.mp4"))
    if not matches:
        matches = sorted((root / "videos").glob(f"**/episode_{episode_index:06d}.mp4"))
    if not matches:
        raise FileNotFoundError(f"episode_{episode_index:06d}.mp4 under {root / 'videos'}")
    return matches[0]


def read_video_rgb(path: Path, expected_len: int) -> np.ndarray:
    import cv2

    cap = cv2.VideoCapture(str(path))
    frames: list[np.ndarray] = []
    while len(frames) < expected_len:
        ok, bgr = cap.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    cap.release()
    if not frames:
        raise ValueError(f"could not read frames from {path}")
    if len(frames) < expected_len:
        frames.extend([frames[-1]] * (expected_len - len(frames)))
    return np.stack(frames[:expected_len])


def encode_frames(
    *,
    encoder: Any,
    frames: np.ndarray,
    image_size: int,
    clip_length: int,
    batch_size: int,
    device: Any,
) -> tuple[np.ndarray, dict[str, float]]:
    import torch
    import torch.nn.functional as F

    total_start = time.perf_counter()
    x = torch.from_numpy(frames).permute(0, 3, 1, 2).float().div_(255.0)
    x = F.interpolate(x, size=(image_size, image_size), mode="bilinear", align_corners=False)
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    x = (x - mean) / std

    outputs: list[np.ndarray] = []
    forward_sec = 0.0
    amp = device.type == "cuda"
    with torch.inference_mode():
        for start in range(0, len(x), batch_size):
            clips = []
            for t in range(start, min(start + batch_size, len(x))):
                first = max(0, t - clip_length + 1)
                idx = [first] * (clip_length - (t - first + 1)) + list(range(first, t + 1))
                clips.append(x[idx].permute(1, 0, 2, 3))
            batch = torch.stack(clips).to(device, non_blocking=True)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            forward_start = time.perf_counter()
            with torch.autocast(device_type=device.type, enabled=amp):
                out = encoder(batch)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            forward_sec += time.perf_counter() - forward_start
            if isinstance(out, (tuple, list)):
                out = out[-1]
            while out.ndim > 2:
                out = out.mean(dim=1)
            outputs.append(out.float().cpu().numpy())
    total_sec = time.perf_counter() - total_start
    frame_count = max(len(x), 1)
    return np.concatenate(outputs, axis=0), {
        "encode_total_sec": total_sec,
        "encode_sec_per_frame": total_sec / frame_count,
        "model_forward_sec": forward_sec,
        "model_forward_sec_per_frame": forward_sec / frame_count,
    }


def write_feature_info(output_root: Path, args: argparse.Namespace, rows: list[dict[str, Any]]) -> None:
    info = {
        "source": str(args.root),
        "camera": args.camera,
        "model_name": args.model_name,
        "checkpoint": str(args.checkpoint),
        "feature_column": args.feature_column,
        "episodes": [int(row["episode_index"]) for row in rows],
    }
    with open(output_root / "meta" / "feature_info.json", "w") as f:
        json.dump(info, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
