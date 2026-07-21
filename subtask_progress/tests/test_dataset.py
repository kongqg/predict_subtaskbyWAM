import json
from pathlib import Path

import numpy as np
import pandas as pd

from subtask_progress.dataset import SubtaskProgressDataset


def _write_episode(root: Path, episode_index: int, task_id: int, length: int):
    chunk = root / "data" / "chunk-000"
    chunk.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(
        {
            "visual": [np.asarray([t, episode_index], dtype=np.float32) for t in range(length)],
            "proprio": [np.asarray([episode_index], dtype=np.float32) for _ in range(length)],
            "task_index": [task_id] * length,
            "frame_index": list(range(length)),
            "episode_index": [episode_index] * length,
        }
    )
    df.to_parquet(chunk / f"episode_{episode_index:06d}.parquet")


def _make_root(tmp_path: Path) -> Path:
    root = tmp_path / "ds"
    (root / "meta").mkdir(parents=True)
    (root / "meta" / "info.json").write_text(json.dumps({"fps": 30}))
    rows = [
        {
            "episode_index": 0,
            "task_index": 0,
            "length": 5,
            "sub_tasks": [{"start": 0, "end": 5}],
            "source_episode_index": 0,
        },
        {
            "episode_index": 1,
            "task_index": 1,
            "length": 4,
            "sub_tasks": [{"start": 0, "end": 4}],
            "source_episode_index": 0,
        },
    ]
    with open(root / "meta" / "episodes.jsonl", "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    _write_episode(root, 0, 0, 5)
    _write_episode(root, 1, 1, 4)
    return root


def _make_feature_root(tmp_path: Path) -> Path:
    root = tmp_path / "features"
    chunk = root / "data" / "chunk-000"
    chunk.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(
        {
            "frame_index": list(range(5)),
            "visual_features": [np.asarray([100 + t], dtype=np.float32) for t in range(5)],
        }
    )
    df.to_parquet(chunk / "episode_000000.parquet")
    return root


def _make_done_feature_root(tmp_path: Path) -> Path:
    root = tmp_path / "done_features"
    chunk = root / "data" / "chunk-000"
    chunk.mkdir(parents=True, exist_ok=True)
    arr = np.asarray([[[1000 + t, 2000 + v] for v in range(4)] for t in range(5)], dtype=np.float32)
    with open(chunk / "episode_000000.npy", "wb") as f:
        np.save(f, arr)
    return root


def _make_view_feature_root(tmp_path: Path, view_idx: int, length: int = 5, frame_offset: int = 0) -> Path:
    root = tmp_path / f"features_view_{view_idx}"
    chunk = root / "data" / "chunk-000"
    chunk.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(
        {
            "frame_index": [t + frame_offset for t in range(length)],
            "visual_features": [np.asarray([view_idx, t], dtype=np.float32) for t in range(length)],
        }
    )
    df.to_parquet(chunk / "episode_000000.parquet")
    return root


def test_dataset_does_not_read_future_frames(tmp_path):
    ds = SubtaskProgressDataset(_make_root(tmp_path), "visual", history_length=3)
    idx = ds.samples.index((0, 2))
    sample = ds[idx]
    valid = ~sample["padding_mask"]
    frames = sample["visual_features"][valid, 0].numpy()
    assert frames.max() <= sample["frame_index"].item()
    assert frames.tolist() == [0.0, 1.0, 2.0]


def test_progress_resets_after_segment_switch(tmp_path):
    ds = SubtaskProgressDataset(_make_root(tmp_path), "visual", history_length=3)
    first_ep0 = ds[ds.samples.index((0, 0))]
    first_ep1 = ds[ds.samples.index((1, 0))]
    last_ep0 = ds[ds.samples.index((0, 4))]
    assert first_ep0["target_progress"].item() == 0.0
    assert first_ep1["target_progress"].item() == 0.0
    assert last_ep0["target_progress"].item() == 1.0


def test_dataset_reads_external_feature_root(tmp_path):
    ds = SubtaskProgressDataset(
        _make_root(tmp_path),
        "visual_features",
        history_length=3,
        feature_root=_make_feature_root(tmp_path),
    )
    assert len(ds.segments) == 1
    sample = ds[ds.samples.index((0, 2))]
    valid = ~sample["padding_mask"]
    assert sample["visual_features"][valid, 0].tolist() == [100.0, 101.0, 102.0]
    assert sample["start_visual"].tolist() == [100.0]


def test_dataset_reads_short_done_feature_root(tmp_path):
    ds = SubtaskProgressDataset(
        _make_root(tmp_path),
        "visual_features",
        history_length=3,
        feature_root=_make_feature_root(tmp_path),
        done_feature_root=_make_done_feature_root(tmp_path),
        done_history_length=2,
    )
    sample = ds[ds.samples.index((0, 2))]
    assert sample["visual_features"].shape == (3, 1)
    assert sample["done_visual_features"].shape == (2, 4, 2)
    assert sample["done_padding_mask"].tolist() == [False, False]
    assert sample["done_visual_features"][:, 0, 0].tolist() == [1001.0, 1002.0]


def test_done_ignore_window_masks_ambiguous_frames(tmp_path):
    root = _make_root(tmp_path)
    ann = tmp_path / "done.jsonl"
    ann.write_text(json.dumps({"episode_index": 0, "done_start_frame": 3}) + "\n")
    ds = SubtaskProgressDataset(
        root,
        "visual",
        history_length=3,
        done_label_strategy="annotation",
        done_annotation_path=ann,
        done_ignore_before=1,
        done_ignore_after=1,
    )
    assert ds[ds.samples.index((0, 1))]["done_loss_mask"].item() == 1.0
    assert ds[ds.samples.index((0, 2))]["done_loss_mask"].item() == 0.0
    assert ds[ds.samples.index((0, 3))]["done_loss_mask"].item() == 0.0
    assert ds[ds.samples.index((0, 4))]["done_loss_mask"].item() == 0.0


def test_done_hard_negative_window_weights_pre_done_frames(tmp_path):
    root = _make_root(tmp_path)
    ann = tmp_path / "done.jsonl"
    ann.write_text(json.dumps({"episode_index": 0, "done_start_frame": 3}) + "\n")
    ds = SubtaskProgressDataset(
        root,
        "visual",
        history_length=3,
        done_label_strategy="annotation",
        done_annotation_path=ann,
        done_hard_negative_window=2,
        done_hard_negative_weight=3.0,
    )
    assert ds[ds.samples.index((0, 0))]["done_loss_mask"].item() == 1.0
    assert ds[ds.samples.index((0, 1))]["done_loss_mask"].item() == 3.0
    assert ds[ds.samples.index((0, 2))]["done_loss_mask"].item() == 3.0
    assert ds[ds.samples.index((0, 3))]["done_loss_mask"].item() == 1.0


def test_done_positive_delay_shifts_annotation_label(tmp_path):
    root = _make_root(tmp_path)
    ann = tmp_path / "done.jsonl"
    ann.write_text(json.dumps({"episode_index": 0, "done_start_frame": 3}) + "\n")
    ds = SubtaskProgressDataset(
        root,
        "visual",
        history_length=3,
        done_label_strategy="annotation",
        done_annotation_path=ann,
        done_positive_delay=1,
    )
    assert ds[ds.samples.index((0, 3))]["target_done"].item() == 0.0
    assert ds[ds.samples.index((0, 4))]["target_done"].item() == 1.0


def test_dataset_reads_aligned_feature_roots_and_view_dropout(tmp_path):
    roots = [_make_view_feature_root(tmp_path, i) for i in range(4)]
    ds = SubtaskProgressDataset(
        _make_root(tmp_path),
        "visual_features",
        history_length=3,
        feature_roots=roots,
        view_dropout_prob=0.99,
        view_dropout_enabled=True,
        seed=0,
    )
    sample = ds[ds.samples.index((0, 2))]
    assert sample["visual_features"].shape == (3, 4, 2)
    assert sample["start_visual"].shape == (4, 2)
    assert sample["view_mask"][:2].any()


def test_dataset_rejects_misaligned_feature_roots(tmp_path):
    roots = [_make_view_feature_root(tmp_path, i) for i in range(4)]
    roots[2] = _make_view_feature_root(tmp_path, 20, frame_offset=1)
    ds = SubtaskProgressDataset(
        _make_root(tmp_path),
        "visual_features",
        history_length=3,
        feature_roots=roots,
    )
    try:
        ds[ds.samples.index((0, 2))]
    except ValueError as exc:
        assert "frame_index mismatch" in str(exc)
    else:
        raise AssertionError("expected feature alignment failure")
