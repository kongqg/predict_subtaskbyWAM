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
    num_views: int = 1
    history_length: int = 32
    d_model: int = 256
    num_layers: int = 3
    num_heads: int = 8
    dim_feedforward: int = 1024
    dropout: float = 0.1
    activation: str = "gelu"
    done_verifier_enabled: bool = False
    done_history_length: int = 8
    done_num_layers: int = 1
    done_num_heads: int = 0
    done_dim_feedforward: int = 0


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


class _DoneVerifier(nn.Module):
    """Independent done classifier over the latest multi-view visual tokens."""

    def __init__(self, config: SubtaskProgressTransformerConfig):
        super().__init__()
        if config.num_views <= 1:
            raise ValueError("done verifier requires structured multi-view input")
        if config.done_history_length <= 0:
            raise ValueError("done_history_length must be > 0")
        heads = config.done_num_heads or config.num_heads
        ff = config.done_dim_feedforward or config.dim_feedforward
        self.history_length = config.done_history_length
        self.visual_norm = nn.LayerNorm(config.visual_dim)
        self.visual_projection = nn.Linear(config.visual_dim, config.d_model)
        self.view_embedding = nn.Embedding(config.num_views, config.d_model)
        self.temporal_position_embedding = nn.Embedding(config.done_history_length, config.d_model)
        self.task_embedding = nn.Embedding(config.num_tasks, config.d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=heads,
            dim_feedforward=ff,
            dropout=config.dropout,
            activation=config.activation,
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=config.done_num_layers)
        self.head = _Head(config.d_model, config.dropout, sigmoid=False)

    def forward(
        self,
        visual_features: torch.Tensor,
        task_ids: torch.Tensor,
        padding_mask: torch.Tensor | None,
        view_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        recent = visual_features[:, -self.history_length :]
        batch, steps, views = recent.shape[:3]
        tokens = self.visual_projection(self.visual_norm(recent))
        view_embed = self.view_embedding.weight[:views].view(1, 1, views, -1)
        pos = self.temporal_position_embedding.weight[-steps:].view(1, steps, 1, -1)
        tokens = tokens + view_embed + pos
        tokens = tokens.reshape(batch, steps * views, -1)

        token_mask = None
        if padding_mask is not None or view_mask is not None:
            time_mask = (
                padding_mask[:, -steps:].bool()
                if padding_mask is not None
                else torch.zeros(batch, steps, dtype=torch.bool, device=visual_features.device)
            )
            view_pad = (
                ~view_mask.bool()
                if view_mask is not None
                else torch.zeros(batch, views, dtype=torch.bool, device=visual_features.device)
            )
            token_mask = (time_mask[:, :, None] | view_pad[:, None, :]).reshape(batch, steps * views)

        task_token = self.task_embedding(task_ids.long())[:, None]
        tokens = torch.cat([task_token, tokens], dim=1)
        if token_mask is not None:
            token_mask = torch.cat(
                [torch.zeros(batch, 1, dtype=torch.bool, device=visual_features.device), token_mask],
                dim=1,
            )
        return self.head(self.encoder(tokens, src_key_padding_mask=token_mask)[:, 0])


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
        if config.num_views <= 0:
            raise ValueError("num_views must be > 0")

        self.config = config
        self.task_embedding = nn.Embedding(config.num_tasks, config.d_model)
        self.type_embedding = nn.Embedding(3, config.d_model)
        self.temporal_position_embedding = nn.Embedding(config.history_length, config.d_model)
        self.view_embedding = nn.Embedding(config.num_views, config.d_model) if config.num_views > 1 else None

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
        self.done_head = (
            None if config.done_verifier_enabled else _Head(config.d_model, config.dropout, sigmoid=False)
        )
        self.done_verifier = _DoneVerifier(config) if config.done_verifier_enabled else None

    def forward(
        self,
        visual_features: torch.Tensor,
        start_visual: torch.Tensor,
        task_ids: torch.Tensor,
        proprio: torch.Tensor | None = None,
        padding_mask: torch.Tensor | None = None,
        view_mask: torch.Tensor | None = None,
        done_visual_features: torch.Tensor | None = None,
        done_padding_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Run the model.

        Args:
            visual_features: [B, T, visual_dim] or [B, T, V, visual_dim].
            start_visual: [B, visual_dim] or [B, V, visual_dim].
            task_ids: [B].
            proprio: optional [B, T, proprio_dim].
            padding_mask: optional [B, T], True for padded frames.
            view_mask: optional [B, V], True for usable views.
            done_visual_features: optional [B, done_T, V, visual_dim] for the done verifier.
            done_padding_mask: optional [B, done_T], True for padded done frames.
        """
        self._check_inputs(
            visual_features,
            start_visual,
            task_ids,
            proprio,
            padding_mask,
            view_mask,
            done_visual_features,
            done_padding_mask,
        )
        batch, steps = visual_features.shape[:2]
        device = visual_features.device

        task_base = self.task_embedding(task_ids.long())
        task_token = task_base + self.type_embedding.weight[self.TASK_TYPE]

        view_attention = None
        start_view_attention = None
        if visual_features.ndim == 4:
            frame_tokens, view_attention = self._fuse_views(
                visual_features, task_base, view_mask, self.visual_norm, self.visual_projection
            )
            start_tokens, start_view_attention = self._fuse_views(
                start_visual[:, None], task_base, view_mask, self.start_norm, self.start_projection
            )
            start_token = start_tokens[:, 0]
        else:
            frame_tokens = self.visual_projection(self.visual_norm(visual_features))
            start_token = self.start_projection(self.start_norm(start_visual))
        start_token = start_token + self.type_embedding.weight[self.START_TYPE]

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
        if self.done_verifier is not None:
            done_logit = self.done_verifier(
                done_visual_features if done_visual_features is not None else visual_features,
                task_ids,
                done_padding_mask if done_padding_mask is not None else padding_mask,
                view_mask,
            )
        else:
            assert self.done_head is not None
            done_logit = self.done_head(task_hidden)
        out = {
            "progress": self.progress_head(task_hidden),
            "done_logit": done_logit,
            "task_hidden": task_hidden,
        }
        if view_attention is not None:
            out["view_attention"] = view_attention
            out["start_view_attention"] = start_view_attention[:, 0]
        return out

    def _fuse_views(
        self,
        features: torch.Tensor,
        task_query: torch.Tensor,
        view_mask: torch.Tensor | None,
        norm: nn.LayerNorm,
        projection: nn.Linear,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Task-conditioned attention over views.

        Args:
            features: [B, T, V, visual_dim].
            task_query: [B, d_model].
            view_mask: optional [B, V], True for usable views.
        """
        assert self.view_embedding is not None
        tokens = projection(norm(features))
        views = tokens.shape[2]
        tokens = tokens + self.view_embedding.weight[:views].view(1, 1, views, -1)
        logits = (tokens * task_query[:, None, None]).sum(dim=-1) / (self.config.d_model**0.5)
        if view_mask is not None:
            logits = logits.masked_fill(~view_mask.bool()[:, None], -1e4)
        weights = torch.softmax(logits, dim=-1)
        return (weights[..., None] * tokens).sum(dim=2), weights

    def _check_inputs(
        self,
        visual_features: torch.Tensor,
        start_visual: torch.Tensor,
        task_ids: torch.Tensor,
        proprio: torch.Tensor | None,
        padding_mask: torch.Tensor | None,
        view_mask: torch.Tensor | None,
        done_visual_features: torch.Tensor | None,
        done_padding_mask: torch.Tensor | None,
    ) -> None:
        cfg = self.config
        if visual_features.ndim not in {3, 4} or visual_features.shape[-1] != cfg.visual_dim:
            raise ValueError(f"visual_features must be [B,T,{cfg.visual_dim}] or [B,T,V,{cfg.visual_dim}]")
        batch, steps = visual_features.shape[:2]
        if visual_features.ndim == 4 and visual_features.shape[2] != cfg.num_views:
            raise ValueError(f"visual_features view dim must be {cfg.num_views}")
        if visual_features.ndim == 3 and cfg.num_views != 1:
            raise ValueError("visual_features is single-view but model num_views != 1")
        if steps > cfg.history_length:
            raise ValueError(f"T={steps} exceeds history_length={cfg.history_length}")
        expected_start = (batch, cfg.num_views, cfg.visual_dim) if cfg.num_views > 1 else (batch, cfg.visual_dim)
        if start_visual.shape != expected_start:
            raise ValueError(f"start_visual must be {expected_start}")
        if cfg.num_views > 1 and self.view_embedding is None:
            raise ValueError("num_views > 1 requires view_embedding")
        if cfg.num_views > 1 and view_mask is not None and view_mask.shape != (batch, cfg.num_views):
            raise ValueError(f"view_mask must be [B,{cfg.num_views}]")
        if cfg.num_views > 1 and view_mask is not None and not bool(view_mask.bool().any(dim=1).all()):
            raise ValueError("each sample must keep at least one view")
        if cfg.done_verifier_enabled and visual_features.ndim != 4:
            raise ValueError("done verifier requires visual_features [B,T,V,D]")
        if done_visual_features is not None:
            if not cfg.done_verifier_enabled:
                raise ValueError("done_visual_features requires done_verifier_enabled")
            if done_visual_features.ndim != 4 or done_visual_features.shape[0] != batch:
                raise ValueError("done_visual_features must be [B,T,V,D]")
            if done_visual_features.shape[2:] != (cfg.num_views, cfg.visual_dim):
                raise ValueError(f"done_visual_features must end with [{cfg.num_views},{cfg.visual_dim}]")
            if done_visual_features.shape[1] > cfg.done_history_length:
                raise ValueError(f"done T exceeds done_history_length={cfg.done_history_length}")
        if done_padding_mask is not None:
            if done_visual_features is None:
                raise ValueError("done_padding_mask requires done_visual_features")
            if done_padding_mask.shape != done_visual_features.shape[:2]:
                raise ValueError("done_padding_mask must be [B,done_T]")
        if cfg.num_views == 1 and view_mask is not None and view_mask.numel() > 0:
            raise ValueError("view_mask was provided but num_views=1")
        if task_ids.shape != (batch,):
            raise ValueError("task_ids must be [B]")
        if cfg.proprio_dim > 0:
            if proprio is None or proprio.shape != (batch, steps, cfg.proprio_dim):
                raise ValueError(f"proprio must be [B,T,{cfg.proprio_dim}]")
        elif proprio is not None and proprio.numel() > 0:
            raise ValueError("proprio was provided but proprio_dim=0")
        if padding_mask is not None and padding_mask.shape != (batch, steps):
            raise ValueError("padding_mask must be [B,T]")
