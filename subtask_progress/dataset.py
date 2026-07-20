"""LeRobot segment dataset for subtask progress prediction."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
import json
from pathlib import Path
import random
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


@dataclass(frozen=True)
class SegmentInfo:
    episode_index: int
    task_id: int
    start: int
    end: int
    length: int
    source_episode_index: int


class SubtaskProgressDataset(Dataset):
    """Samples past-only windows from already segmented LeRobot episodes."""

    def __init__(
        self,
        root: str | Path,
        visual_column: str,
        history_length: int,
        feature_root: str | Path | None = None,
        proprio_column: str | None = None,
        task_column: str = "task_index",
        done_column: str | None = None,
        done_annotation_path: str | Path | None = None,
        done_label_strategy: str = "last_window",
        done_window: int = 3,
        done_positive_delay: int = 0,
        done_positive_delay_ratio: float = 0.0,
        done_ignore_before: int = 0,
        done_ignore_after: int = 0,
        segment_end_mode: str = "exclusive",
        frame_stride: int = 1,
        sample_tail_frames: int | None = None,
        exclude_tail_frames: int = 0,
        sample_modulus: int | None = None,
        sample_remainder: int = 0,
        sample_mod_mode: str = "include",
        max_samples: int | None = None,
        max_cached_episodes: int = 8,
        seed: int = 0,
        augment: dict[str, Any] | None = None,
    ):
        self.root = Path(root)
        self.feature_root = Path(feature_root) if feature_root else None
        self.visual_column = visual_column
        self.proprio_column = proprio_column or None
        self.task_column = task_column
        self.done_column = done_column or None
        self.done_annotation_path = Path(done_annotation_path) if done_annotation_path else None
        self.done_label_strategy = done_label_strategy
        self.done_window = int(done_window)
        self.done_positive_delay = int(done_positive_delay)
        self.done_positive_delay_ratio = float(done_positive_delay_ratio)
        self.done_ignore_before = int(done_ignore_before)
        self.done_ignore_after = int(done_ignore_after)
        self.segment_end_mode = segment_end_mode
        self.history_length = int(history_length)
        self.frame_stride = int(frame_stride)
        self.sample_tail_frames = None if sample_tail_frames is None else int(sample_tail_frames)
        self.exclude_tail_frames = int(exclude_tail_frames)
        self.sample_modulus = None if sample_modulus is None else int(sample_modulus)
        self.sample_remainder = int(sample_remainder)
        self.sample_mod_mode = sample_mod_mode
        self.max_cached_episodes = int(max_cached_episodes)
        self.rng = random.Random(seed)
        self.augment = augment or {}
        self._cache: OrderedDict[int, pd.DataFrame] = OrderedDict()
        self._feature_cache: OrderedDict[int, pd.DataFrame] = OrderedDict()

        if self.history_length <= 0:
            raise ValueError("history_length must be > 0")
        if self.done_label_strategy not in {"last_frame", "last_window", "column", "annotation"}:
            raise ValueError("done_label_strategy must be last_frame, last_window, column, or annotation")
        if self.segment_end_mode not in {"exclusive", "inclusive"}:
            raise ValueError("segment_end_mode must be exclusive or inclusive")
        if self.sample_tail_frames is not None and self.sample_tail_frames <= 0:
            raise ValueError("sample_tail_frames must be > 0")
        if self.exclude_tail_frames < 0:
            raise ValueError("exclude_tail_frames must be >= 0")
        if self.done_positive_delay < 0:
            raise ValueError("done_positive_delay must be >= 0")
        if self.done_positive_delay_ratio < 0:
            raise ValueError("done_positive_delay_ratio must be >= 0")
        if self.done_ignore_before < 0 or self.done_ignore_after < 0:
            raise ValueError("done ignore windows must be >= 0")
        if self.sample_tail_frames is not None and self.exclude_tail_frames:
            raise ValueError("sample_tail_frames and exclude_tail_frames are mutually exclusive")
        if self.sample_modulus is not None and self.sample_modulus <= 0:
            raise ValueError("sample_modulus must be > 0")
        if self.sample_mod_mode not in {"include", "exclude"}:
            raise ValueError("sample_mod_mode must be include or exclude")

        self.info = self._load_info()
        self.done_start_by_episode = self._load_done_annotations()
        self.parquet_by_episode = self._index_parquets()
        self.feature_by_episode = self._index_features() if self.feature_root else {}
        self.segments = self._load_segments()
        self.samples = self._build_samples(max_samples)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        segment_idx, t = self.samples[index]
        segment = self.segments[segment_idx]
        df = self._episode_df(segment.episode_index)
        visual_df = self._feature_df(segment.episode_index) if self.feature_root else df

        raw_indices = self._history_indices(segment, t)
        indices = self._augment_indices(raw_indices)
        visual = self._pad_feature(self._stack_feature(visual_df, self.visual_column, indices))
        start_visual = self._stack_feature(visual_df, self.visual_column, [segment.start])[0]
        padding_mask = self._padding_mask(len(indices))

        if self.proprio_column:
            proprio_np = self._pad_feature(self._stack_feature(df, self.proprio_column, indices))
        else:
            proprio_np = np.zeros((self.history_length, 0), dtype=np.float32)

        progress = self._progress(segment, t)
        done = self._done_label(df, segment, t)
        done_loss_mask = self._done_loss_mask(segment, t)

        return {
            "visual_features": torch.from_numpy(visual),
            "start_visual": torch.from_numpy(start_visual.astype(np.float32)),
            "task_ids": torch.tensor(segment.task_id, dtype=torch.long),
            "proprio": torch.from_numpy(proprio_np),
            "padding_mask": torch.from_numpy(padding_mask),
            "target_progress": torch.tensor(progress, dtype=torch.float32),
            "target_done": torch.tensor(done, dtype=torch.float32),
            "done_loss_mask": torch.tensor(done_loss_mask, dtype=torch.float32),
            "segment_ids": torch.tensor(segment.episode_index, dtype=torch.long),
            "frame_index": torch.tensor(t, dtype=torch.long),
            "source_episode_index": torch.tensor(segment.source_episode_index, dtype=torch.long),
        }

    def infer_dims(self) -> tuple[int, int]:
        sample = self[0]
        return sample["visual_features"].shape[-1], sample["proprio"].shape[-1]

    def _load_info(self) -> dict[str, Any]:
        path = self.root / "meta" / "info.json"
        if not path.exists():
            return {}
        with open(path, "r") as f:
            return json.load(f)

    def _index_parquets(self) -> dict[int, Path]:
        out: dict[int, Path] = {}
        for path in sorted((self.root / "data").rglob("*.parquet")):
            stem = path.stem
            if stem.startswith("episode_"):
                out[int(stem.split("_")[-1])] = path
        if not out:
            raise FileNotFoundError(f"no parquet files under {self.root / 'data'}")
        return out

    def _load_done_annotations(self) -> dict[int, int]:
        if self.done_label_strategy != "annotation":
            return {}
        if self.done_annotation_path is None:
            raise ValueError("done_label_strategy=annotation requires done_annotation_path")
        out: dict[int, int] = {}
        with open(self.done_annotation_path, "r") as f:
            for line in f:
                if not line.strip():
                    continue
                row = json.loads(line)
                out[int(row["episode_index"])] = int(row["done_start_frame"])
        return out

    def _index_features(self) -> dict[int, Path]:
        assert self.feature_root is not None
        out: dict[int, Path] = {}
        for path in sorted((self.feature_root / "data").rglob("*.parquet")):
            stem = path.stem
            if stem.startswith("episode_"):
                out[int(stem.split("_")[-1])] = path
        if not out:
            raise FileNotFoundError(f"no feature parquet files under {self.feature_root / 'data'}")
        return out

    def _load_segments(self) -> list[SegmentInfo]:
        episodes_path = self.root / "meta" / "episodes.jsonl"
        if not episodes_path.exists():
            raise FileNotFoundError(f"missing {episodes_path}")

        segments: list[SegmentInfo] = []
        with open(episodes_path, "r") as f:
            for line in f:
                if not line.strip():
                    continue
                row = json.loads(line)
                episode_index = int(row["episode_index"])
                if episode_index not in self.parquet_by_episode:
                    continue
                if self.feature_root and episode_index not in self.feature_by_episode:
                    continue
                sub = (row.get("sub_tasks") or [{}])[0]
                start = int(sub.get("start", 0))
                length = int(row.get("length", 0))
                end = int(sub.get("end", length))
                if self.segment_end_mode == "inclusive":
                    end += 1
                if end <= start:
                    continue
                segments.append(
                    SegmentInfo(
                        episode_index=episode_index,
                        task_id=int(row.get("task_index", 0)),
                        start=start,
                        end=end,
                        length=length,
                        source_episode_index=int(row.get("source_episode_index", episode_index)),
                    )
                )
        if not segments:
            raise ValueError(f"no usable segments in {episodes_path}")
        return segments

    def _build_samples(self, max_samples: int | None) -> list[tuple[int, int]]:
        samples = []
        for i, seg in enumerate(self.segments):
            sample_start = seg.start
            sample_end = seg.end
            if self.exclude_tail_frames:
                sample_end = max(seg.start, seg.end - self.exclude_tail_frames)
            if self.sample_tail_frames is not None:
                sample_start = max(seg.start, seg.end - self.sample_tail_frames)
            if sample_end <= sample_start:
                continue
            ts = list(range(sample_start, sample_end, self.frame_stride))
            if ts[-1] != sample_end - 1:
                ts.append(sample_end - 1)
            if self.sample_modulus is not None:
                ts = [
                    t
                    for t in ts
                    if ((t - seg.start) % self.sample_modulus == self.sample_remainder)
                    == (self.sample_mod_mode == "include")
                ]
            samples.extend((i, t) for t in ts)
        if max_samples is not None and len(samples) > max_samples:
            samples = self.rng.sample(samples, max_samples)
        return samples

    def _episode_df(self, episode_index: int) -> pd.DataFrame:
        if episode_index in self._cache:
            self._cache.move_to_end(episode_index)
            return self._cache[episode_index]
        df = pd.read_parquet(self.parquet_by_episode[episode_index])
        self._cache[episode_index] = df
        if len(self._cache) > self.max_cached_episodes:
            self._cache.popitem(last=False)
        return df

    def _feature_df(self, episode_index: int) -> pd.DataFrame:
        if episode_index in self._feature_cache:
            self._feature_cache.move_to_end(episode_index)
            return self._feature_cache[episode_index]
        df = pd.read_parquet(self.feature_by_episode[episode_index])
        self._feature_cache[episode_index] = df
        if len(self._feature_cache) > self.max_cached_episodes:
            self._feature_cache.popitem(last=False)
        return df

    def _history_indices(self, segment: SegmentInfo, t: int) -> list[int]:
        interval = int(self.augment.get("max_sample_interval", 1))
        if interval > 1 and self.augment.get("random_sample_interval", False):
            step = self.rng.randint(1, interval)
        else:
            step = 1
        max_len = self.history_length
        if self.augment.get("random_history_crop", False):
            min_len = int(self.augment.get("min_history_length", 1))
            max_len = self.rng.randint(max(1, min_len), self.history_length)

        start = max(segment.start, t - (max_len - 1) * step)
        indices = list(range(start, t + 1, step))
        if indices[-1] != t:
            indices.append(t)
        return indices[-self.history_length :]

    def _augment_indices(self, indices: list[int]) -> list[int]:
        if not indices:
            return indices
        out = list(indices)
        drop_prob = float(self.augment.get("random_skip_prob", 0.0))
        if drop_prob > 0 and len(out) > 1:
            kept = [i for i in out[:-1] if self.rng.random() >= drop_prob]
            out = kept + [out[-1]]

        repeat_prob = float(self.augment.get("random_repeat_prob", 0.0))
        pause_prob = float(self.augment.get("random_pause_prob", 0.0))
        if repeat_prob > 0 or pause_prob > 0:
            expanded: list[int] = []
            for idx in out:
                expanded.append(idx)
                if self.rng.random() < repeat_prob:
                    expanded.append(idx)
                if self.rng.random() < pause_prob:
                    expanded.extend([idx, idx])
            out = expanded
        return out[-self.history_length :]

    def _stack_feature(self, df: pd.DataFrame, column: str, indices: list[int]) -> np.ndarray:
        if column not in df.columns:
            raise KeyError(f"{column!r} not found in parquet columns: {list(df.columns)}")
        values = df.iloc[indices][column].to_list()
        arr = np.asarray(values, dtype=np.float32)
        if arr.ndim == 1:
            arr = arr[:, None]
        return arr

    def _pad_feature(self, arr: np.ndarray) -> np.ndarray:
        arr = arr.astype(np.float32)
        if arr.shape[0] > self.history_length:
            arr = arr[-self.history_length :]
        pad = self.history_length - arr.shape[0]
        if pad <= 0:
            return arr
        zeros = np.zeros((pad, arr.shape[1]), dtype=np.float32)
        return np.concatenate([zeros, arr], axis=0)

    def _padding_mask(self, valid_len: int) -> np.ndarray:
        valid_len = min(valid_len, self.history_length)
        mask = np.ones((self.history_length,), dtype=bool)
        mask[-valid_len:] = False
        return mask

    def _progress(self, segment: SegmentInfo, t: int) -> float:
        end_for_progress = segment.end - 1
        denom = max(end_for_progress - segment.start, 1)
        return float(np.clip((t - segment.start) / denom, 0.0, 1.0))

    def _done_label(self, df: pd.DataFrame, segment: SegmentInfo, t: int) -> float:
        if self.done_label_strategy == "column":
            if not self.done_column:
                raise ValueError("done_label_strategy=column requires done_column")
            return float(df.iloc[t][self.done_column])
        if self.done_label_strategy == "last_frame":
            return float(t >= segment.end - 1)
        if self.done_label_strategy == "annotation":
            if segment.episode_index not in self.done_start_by_episode:
                raise KeyError(f"missing done annotation for episode {segment.episode_index}")
            return float(t >= self._effective_done_start(segment))
        return float(t >= segment.end - max(self.done_window, 1))

    def _done_loss_mask(self, segment: SegmentInfo, t: int) -> float:
        if self.done_label_strategy != "annotation":
            return 1.0
        if self.done_ignore_before == 0 and self.done_ignore_after == 0:
            return 1.0
        done_start = self._effective_done_start(segment)
        lo = done_start - self.done_ignore_before
        hi = done_start + self.done_ignore_after
        return float(not (lo <= t <= hi))

    def _effective_done_start(self, segment: SegmentInfo) -> int:
        raw_done_start = self.done_start_by_episode[segment.episode_index]
        done_after = max(segment.end - raw_done_start, 0)
        ratio_delay = int(round(self.done_positive_delay_ratio * done_after))
        done_start = raw_done_start + self.done_positive_delay + ratio_delay
        return min(done_start, segment.end - 1)
