"""Losses for progress monitoring."""

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


class ProgressLoss(nn.Module):
    """Huber progress loss + BCE done loss + optional same-segment ranking."""

    def __init__(
        self,
        lambda_progress: float = 1.0,
        lambda_done: float = 1.0,
        lambda_rank: float = 0.1,
        target_margin: float = 0.05,
        ranking_margin: float = 0.02,
        done_pos_weight: float | None = None,
        max_pairs_per_segment: int = 2048,
    ):
        super().__init__()
        self.lambda_progress = lambda_progress
        self.lambda_done = lambda_done
        self.lambda_rank = lambda_rank
        self.target_margin = target_margin
        self.ranking_margin = ranking_margin
        self.max_pairs_per_segment = max_pairs_per_segment
        self.progress_loss = nn.SmoothL1Loss()
        if done_pos_weight is None:
            self.register_buffer("done_pos_weight", None)
        else:
            self.register_buffer("done_pos_weight", torch.tensor(float(done_pos_weight)))

    def forward(
        self,
        pred_progress: torch.Tensor,
        done_logit: torch.Tensor,
        target_progress: torch.Tensor,
        target_done: torch.Tensor,
        segment_ids: torch.Tensor | None = None,
        done_loss_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        progress_loss = self.progress_loss(pred_progress, target_progress)
        done_loss_raw = F.binary_cross_entropy_with_logits(
            done_logit,
            target_done,
            pos_weight=self.done_pos_weight,
            reduction="none",
        )
        if done_loss_mask is None:
            done_loss = done_loss_raw.mean()
        else:
            mask = done_loss_mask.to(done_loss_raw.dtype)
            done_loss = (done_loss_raw * mask).sum() / mask.sum().clamp_min(1.0)
        rank_loss = pred_progress.new_tensor(0.0)
        if self.lambda_rank > 0 and segment_ids is not None:
            rank_loss = pairwise_ranking_loss(
                pred_progress,
                target_progress,
                segment_ids,
                self.target_margin,
                self.ranking_margin,
                self.max_pairs_per_segment,
            )
        total = (
            self.lambda_progress * progress_loss
            + self.lambda_done * done_loss
            + self.lambda_rank * rank_loss
        )
        return {
            "loss": total,
            "progress_loss": progress_loss.detach(),
            "done_loss": done_loss.detach(),
            "rank_loss": rank_loss.detach(),
        }


def pairwise_ranking_loss(
    pred_progress: torch.Tensor,
    target_progress: torch.Tensor,
    segment_ids: torch.Tensor,
    target_margin: float,
    ranking_margin: float,
    max_pairs_per_segment: int = 2048,
) -> torch.Tensor:
    """Same-segment pairwise monotonic ranking loss."""
    losses: list[torch.Tensor] = []
    for segment_id in torch.unique(segment_ids):
        idx = torch.nonzero(segment_ids == segment_id, as_tuple=False).flatten()
        if idx.numel() < 2:
            continue
        target = target_progress[idx]
        diff = target[:, None] - target[None, :]
        pairs = torch.nonzero(diff > target_margin, as_tuple=False)
        if pairs.numel() == 0:
            continue
        if pairs.shape[0] > max_pairs_per_segment:
            keep = torch.randperm(pairs.shape[0], device=pairs.device)[:max_pairs_per_segment]
            pairs = pairs[keep]
        j = idx[pairs[:, 0]]
        i = idx[pairs[:, 1]]
        losses.append(F.relu(ranking_margin - pred_progress[j] + pred_progress[i]))
    if not losses:
        return pred_progress.new_tensor(0.0)
    return torch.cat(losses).mean()
