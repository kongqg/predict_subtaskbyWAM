"""Subtask-conditioned temporal progress Transformer."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass
class SubtaskProgressTransformerConfig:
    visual_dim: int
    proprio_dim: int
    num_tasks: int
    history_length: int = 32
    d_model: int = 256
    num_layers: int = 3
    num_heads: int = 8
    dim_feedforward: int = 1024
    dropout: float = 0.1
    activation: str = "gelu"


class _Head(nn.Module):
    def __init__(self, d_model: int, dropout: float, sigmoid: bool):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1),
        )
        self.sigmoid = sigmoid

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.net(x).squeeze(-1)
        return torch.sigmoid(y) if self.sigmoid else y


class SubtaskProgressTransformer(nn.Module):
    """Predict local subtask progress and completion from past-only history."""

    TASK_TYPE = 0
    START_TYPE = 1
    FRAME_TYPE = 2

    def __init__(self, config: SubtaskProgressTransformerConfig):
        super().__init__()
        if config.visual_dim <= 0:
            raise ValueError("visual_dim must be > 0")
        if config.proprio_dim < 0:
            raise ValueError("proprio_dim must be >= 0")
        if config.num_tasks <= 0:
            raise ValueError("num_tasks must be > 0")

        self.config = config
        self.task_embedding = nn.Embedding(config.num_tasks, config.d_model)
        self.type_embedding = nn.Embedding(3, config.d_model)
        self.temporal_position_embedding = nn.Embedding(config.history_length, config.d_model)

        self.visual_norm = nn.LayerNorm(config.visual_dim)
        self.visual_projection = nn.Linear(config.visual_dim, config.d_model)
        self.start_norm = nn.LayerNorm(config.visual_dim)
        self.start_projection = nn.Linear(config.visual_dim, config.d_model)

        if config.proprio_dim > 0:
            self.proprio_norm = nn.LayerNorm(config.proprio_dim)
            self.proprio_projection = nn.Linear(config.proprio_dim, config.d_model)
        else:
            self.proprio_norm = None
            self.proprio_projection = None

        layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.num_heads,
            dim_feedforward=config.dim_feedforward,
            dropout=config.dropout,
            activation=config.activation,
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=config.num_layers)
        self.progress_head = _Head(config.d_model, config.dropout, sigmoid=True)
        self.done_head = _Head(config.d_model, config.dropout, sigmoid=False)

    def forward(
        self,
        visual_features: torch.Tensor,
        start_visual: torch.Tensor,
        task_ids: torch.Tensor,
        proprio: torch.Tensor | None = None,
        padding_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Run the model.

        Args:
            visual_features: [B, T, visual_dim].
            start_visual: [B, visual_dim].
            task_ids: [B].
            proprio: optional [B, T, proprio_dim].
            padding_mask: optional [B, T], True for padded frames.
        """
        self._check_inputs(visual_features, start_visual, task_ids, proprio, padding_mask)
        batch, steps, _ = visual_features.shape
        device = visual_features.device

        task_token = self.task_embedding(task_ids.long())
        task_token = task_token + self.type_embedding.weight[self.TASK_TYPE]

        start_token = self.start_projection(self.start_norm(start_visual))
        start_token = start_token + self.type_embedding.weight[self.START_TYPE]

        frame_tokens = self.visual_projection(self.visual_norm(visual_features))
        if self.config.proprio_dim > 0:
            assert proprio is not None
            frame_tokens = frame_tokens + self.proprio_projection(self.proprio_norm(proprio))

        positions = torch.arange(steps, device=device)
        frame_tokens = frame_tokens + self.temporal_position_embedding(positions).unsqueeze(0)
        frame_tokens = frame_tokens + self.type_embedding.weight[self.FRAME_TYPE]

        tokens = torch.cat([task_token[:, None], start_token[:, None], frame_tokens], dim=1)
        token_padding_mask = None
        if padding_mask is not None:
            prefix = torch.zeros(batch, 2, dtype=torch.bool, device=device)
            token_padding_mask = torch.cat([prefix, padding_mask.bool()], dim=1)

        encoded = self.encoder(tokens, src_key_padding_mask=token_padding_mask)
        task_hidden = encoded[:, 0]
        return {
            "progress": self.progress_head(task_hidden),
            "done_logit": self.done_head(task_hidden),
            "task_hidden": task_hidden,
        }

    def _check_inputs(
        self,
        visual_features: torch.Tensor,
        start_visual: torch.Tensor,
        task_ids: torch.Tensor,
        proprio: torch.Tensor | None,
        padding_mask: torch.Tensor | None,
    ) -> None:
        cfg = self.config
        if visual_features.ndim != 3 or visual_features.shape[-1] != cfg.visual_dim:
            raise ValueError(f"visual_features must be [B,T,{cfg.visual_dim}]")
        batch, steps, _ = visual_features.shape
        if steps > cfg.history_length:
            raise ValueError(f"T={steps} exceeds history_length={cfg.history_length}")
        if start_visual.shape != (batch, cfg.visual_dim):
            raise ValueError(f"start_visual must be [B,{cfg.visual_dim}]")
        if task_ids.shape != (batch,):
            raise ValueError("task_ids must be [B]")
        if cfg.proprio_dim > 0:
            if proprio is None or proprio.shape != (batch, steps, cfg.proprio_dim):
                raise ValueError(f"proprio must be [B,T,{cfg.proprio_dim}]")
        elif proprio is not None and proprio.numel() > 0:
            raise ValueError("proprio was provided but proprio_dim=0")
        if padding_mask is not None and padding_mask.shape != (batch, steps):
            raise ValueError("padding_mask must be [B,T]")
