"""Feature adapter protocol for future visual backbones."""

from __future__ import annotations

from typing import Protocol

import numpy as np


class VisualFeatureEncoder(Protocol):
    """Minimal protocol for frozen visual backbones.

    The core progress model consumes precomputed feature vectors and does not
    depend on any concrete image encoder.
    """

    @property
    def output_dim(self) -> int:
        """Feature dimension returned by encode."""

    def encode(self, frames: object) -> np.ndarray:
        """Return visual features with shape [T, output_dim]."""


class PrecomputedFeatureAdapter:
    """Identity adapter for already-computed features."""

    def __init__(self, output_dim: int):
        self._output_dim = int(output_dim)

    @property
    def output_dim(self) -> int:
        return self._output_dim

    def encode(self, frames: object) -> np.ndarray:
        arr = np.asarray(frames, dtype=np.float32)
        if arr.ndim != 2 or arr.shape[-1] != self.output_dim:
            raise ValueError(f"expected [T, {self.output_dim}] features, got {arr.shape}")
        return arr
