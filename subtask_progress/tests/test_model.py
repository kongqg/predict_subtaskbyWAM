import torch

from subtask_progress.model import SubtaskProgressTransformer, SubtaskProgressTransformerConfig


def _model(proprio_dim: int = 3):
    return SubtaskProgressTransformer(
        SubtaskProgressTransformerConfig(
            visual_dim=5,
            proprio_dim=proprio_dim,
            num_tasks=4,
            history_length=6,
            d_model=32,
            num_layers=1,
            num_heads=4,
            dim_feedforward=64,
            dropout=0.0,
        )
    )


def test_model_shapes_and_progress_range():
    model = _model()
    out = model(
        torch.randn(2, 6, 5),
        torch.randn(2, 5),
        torch.tensor([0, 1]),
        torch.randn(2, 6, 3),
    )
    assert out["progress"].shape == (2,)
    assert out["done_logit"].shape == (2,)
    assert torch.all((out["progress"] >= 0) & (out["progress"] <= 1))


def test_model_runs_without_proprio_and_with_padding_mask():
    model = _model(proprio_dim=0)
    out = model(
        torch.randn(2, 6, 5),
        torch.randn(2, 5),
        torch.tensor([0, 1]),
        padding_mask=torch.tensor([[True, True, False, False, False, False], [False] * 6]),
    )
    assert out["progress"].shape == (2,)


def test_task_id_changes_task_hidden_for_same_inputs():
    torch.manual_seed(0)
    model = _model(proprio_dim=0).eval()
    visual = torch.randn(1, 6, 5).repeat(2, 1, 1)
    start = torch.randn(1, 5).repeat(2, 1)
    out = model(visual, start, torch.tensor([0, 1]))
    assert not torch.allclose(out["task_hidden"][0], out["task_hidden"][1])


def test_structured_4view_fusion_outputs_attention():
    model = SubtaskProgressTransformer(
        SubtaskProgressTransformerConfig(
            visual_dim=5,
            proprio_dim=0,
            num_tasks=2,
            num_views=4,
            history_length=6,
            d_model=32,
            num_layers=1,
            num_heads=4,
            dim_feedforward=64,
            dropout=0.0,
        )
    )
    out = model(
        torch.randn(2, 6, 4, 5),
        torch.randn(2, 4, 5),
        torch.tensor([0, 1]),
        padding_mask=torch.tensor([[True, False, False, False, False, False], [False] * 6]),
        view_mask=torch.tensor([[True, False, True, True], [False, True, True, False]]),
    )
    assert out["progress"].shape == (2,)
    assert out["view_attention"].shape == (2, 6, 4)
    assert torch.allclose(out["view_attention"][0, :, 1], torch.zeros(6), atol=1e-6)


def test_independent_done_verifier_outputs_done_logit():
    model = SubtaskProgressTransformer(
        SubtaskProgressTransformerConfig(
            visual_dim=5,
            proprio_dim=0,
            num_tasks=2,
            num_views=4,
            history_length=12,
            d_model=32,
            num_layers=1,
            num_heads=4,
            dim_feedforward=64,
            dropout=0.0,
            done_verifier_enabled=True,
            done_history_length=8,
            done_num_layers=1,
        )
    )
    out = model(
        torch.randn(2, 12, 4, 5),
        torch.randn(2, 4, 5),
        torch.tensor([0, 1]),
        padding_mask=torch.tensor(
            [[True, True, False, False, False, False, False, False, False, False, False, False], [False] * 12]
        ),
        view_mask=torch.tensor([[True, False, True, True], [False, True, True, False]]),
    )
    assert out["progress"].shape == (2,)
    assert out["done_logit"].shape == (2,)
    assert model.done_head is None
