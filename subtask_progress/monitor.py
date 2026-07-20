"""Online progress monitor."""

from __future__ import annotations

from collections import deque

import numpy as np
import torch

from .model import SubtaskProgressTransformer


class ProgressMonitor:
    """Stateful online wrapper with an internal history buffer."""

    def __init__(
        self,
        model: SubtaskProgressTransformer,
        history_length: int,
        device: str | torch.device = "cpu",
        done_threshold: float = 0.9,
        done_patience: int = 3,
        ema_alpha: float | None = None,
    ):
        self.model = model.to(device).eval()
        self.history_length = int(history_length)
        self.device = torch.device(device)
        self.done_threshold = float(done_threshold)
        self.done_patience = int(done_patience)
        self.ema_alpha = ema_alpha
        self.visual_buffer: deque[np.ndarray] = deque(maxlen=self.history_length)
        self.proprio_buffer: deque[np.ndarray] = deque(maxlen=self.history_length)
        self.task_id: int | None = None
        self.start_visual: np.ndarray | None = None
        self.done_count = 0
        self.progress_ema: float | None = None

    def reset(self, task_id: int, start_visual: np.ndarray | torch.Tensor) -> None:
        """Start monitoring a new subtask segment."""
        self.task_id = int(task_id)
        self.start_visual = self._to_numpy(start_visual)
        self.visual_buffer.clear()
        self.proprio_buffer.clear()
        self.done_count = 0
        self.progress_ema = None

    @torch.no_grad()
    def update(
        self,
        visual_feature: np.ndarray | torch.Tensor,
        proprio: np.ndarray | torch.Tensor | None = None,
    ) -> dict[str, float | bool]:
        if self.task_id is None or self.start_visual is None:
            raise RuntimeError("call reset(task_id, start_visual) before update")

        self.visual_buffer.append(self._to_numpy(visual_feature))
        if self.model.config.proprio_dim > 0:
            if proprio is None:
                raise ValueError("proprio is required because model.proprio_dim > 0")
            self.proprio_buffer.append(self._to_numpy(proprio))

        visual, padding_mask = self._padded(np.asarray(list(self.visual_buffer), dtype=np.float32))
        if self.model.config.proprio_dim > 0:
            proprio_arr, _ = self._padded(np.asarray(list(self.proprio_buffer), dtype=np.float32))
        else:
            proprio_arr = np.zeros((self.history_length, 0), dtype=np.float32)

        output = self.model(
            visual_features=torch.from_numpy(visual[None]).to(self.device),
            start_visual=torch.from_numpy(self.start_visual[None].astype(np.float32)).to(self.device),
            task_ids=torch.tensor([self.task_id], device=self.device),
            proprio=torch.from_numpy(proprio_arr[None]).to(self.device),
            padding_mask=torch.from_numpy(padding_mask[None]).to(self.device),
        )
        progress = float(output["progress"].item())
        done_probability = float(torch.sigmoid(output["done_logit"]).item())
        if done_probability >= self.done_threshold:
            self.done_count += 1
        else:
            self.done_count = 0

        if self.ema_alpha is not None:
            if self.progress_ema is None:
                self.progress_ema = progress
            else:
                self.progress_ema = self.ema_alpha * progress + (1 - self.ema_alpha) * self.progress_ema

        return {
            "progress": progress,
            "display_progress": self.progress_ema if self.progress_ema is not None else progress,
            "done_probability": done_probability,
            "should_switch": self.done_count >= self.done_patience,
        }

    def _padded(self, arr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if arr.ndim == 1:
            arr = arr[:, None]
        arr = arr[-self.history_length :].astype(np.float32)
        pad = self.history_length - arr.shape[0]
        mask = np.ones((self.history_length,), dtype=bool)
        if pad > 0:
            arr = np.concatenate([np.zeros((pad, arr.shape[1]), dtype=np.float32), arr], axis=0)
        mask[-min(len(self.visual_buffer), self.history_length) :] = False
        return arr, mask

    @staticmethod
    def _to_numpy(x: np.ndarray | torch.Tensor) -> np.ndarray:
        if isinstance(x, torch.Tensor):
            x = x.detach().cpu().numpy()
        return np.asarray(x, dtype=np.float32)
