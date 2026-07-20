import numpy as np
import torch

from subtask_progress.losses import ProgressLoss
from subtask_progress.model import SubtaskProgressTransformer, SubtaskProgressTransformerConfig
from subtask_progress.monitor import ProgressMonitor


def test_progress_monitor_reset_clears_history():
    model = SubtaskProgressTransformer(
        SubtaskProgressTransformerConfig(
            visual_dim=3,
            proprio_dim=0,
            num_tasks=2,
            history_length=4,
            d_model=16,
            num_layers=1,
            num_heads=4,
            dim_feedforward=32,
        )
    )
    monitor = ProgressMonitor(model, history_length=4)
    monitor.reset(0, np.zeros(3, dtype=np.float32))
    monitor.update(np.ones(3, dtype=np.float32))
    assert len(monitor.visual_buffer) == 1
    monitor.reset(1, np.zeros(3, dtype=np.float32))
    assert len(monitor.visual_buffer) == 0
    assert monitor.done_count == 0


def test_tiny_batch_can_overfit():
    torch.manual_seed(0)
    model = SubtaskProgressTransformer(
        SubtaskProgressTransformerConfig(
            visual_dim=2,
            proprio_dim=0,
            num_tasks=1,
            history_length=4,
            d_model=32,
            num_layers=1,
            num_heads=4,
            dim_feedforward=64,
            dropout=0.0,
        )
    )
    loss_fn = ProgressLoss(lambda_rank=0.0)
    optim = torch.optim.AdamW(model.parameters(), lr=1e-2)
    target = torch.linspace(0, 1, 8)
    visual = torch.zeros(8, 4, 2)
    visual[:, -1, 0] = target
    batch = {
        "visual_features": visual,
        "start_visual": torch.zeros(8, 2),
        "task_ids": torch.zeros(8, dtype=torch.long),
        "proprio": torch.zeros(8, 4, 0),
        "padding_mask": torch.zeros(8, 4, dtype=torch.bool),
        "target_progress": target,
        "target_done": (target > 0.9).float(),
        "segment_ids": torch.zeros(8, dtype=torch.long),
    }

    def step():
        out = model(
            batch["visual_features"],
            batch["start_visual"],
            batch["task_ids"],
            batch["proprio"],
            batch["padding_mask"],
        )
        return loss_fn(
            out["progress"],
            out["done_logit"],
            batch["target_progress"],
            batch["target_done"],
            batch["segment_ids"],
        )["loss"]

    initial = float(step())
    for _ in range(120):
        optim.zero_grad()
        loss = step()
        loss.backward()
        optim.step()
    final = float(step())
    assert final < initial * 0.5


def test_done_loss_mask_ignores_ambiguous_samples():
    loss_fn = ProgressLoss(lambda_progress=0.0, lambda_done=1.0, lambda_rank=0.0)
    losses = loss_fn(
        torch.zeros(2),
        torch.tensor([0.0, 100.0]),
        torch.zeros(2),
        torch.tensor([0.0, 0.0]),
        done_loss_mask=torch.tensor([1.0, 0.0]),
    )
    assert float(losses["loss"]) < 1.0
