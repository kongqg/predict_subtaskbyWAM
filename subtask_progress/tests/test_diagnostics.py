from subtask_progress.diagnostics import summarize_diagnostics


def test_diagnostics_reports_early_and_pre_done_votes():
    rows = []
    for t in range(8):
        rows.append(
            {
                "task_id": 0,
                "segment_id": 1,
                "frame_index": t,
                "target_done": float(t >= 5),
                "done_probability": 0.95 if t in {2, 3, 4} else 0.1,
                "pred_progress": t / 7,
                "target_progress": t / 7,
                "view_attention_current": [0.7, 0.1, 0.1, 0.1],
                "view_attention_history_mean": [0.4, 0.2, 0.2, 0.2],
                "view_attention_entropy_current": 0.5,
            }
        )

    out = summarize_diagnostics(rows, threshold=0.9, window=3, votes=2)
    episode = out["episodes"][0]
    assert episode["trigger_frame"] == 3
    assert episode["lead_lag"] == -2
    assert episode["max_pre_done_votes"] == 3
    assert episode["max_pre_done_window_score"] >= 0.9
    assert out["strict"]["early"] == 1
    assert out["attention"]["available"] is True
    assert out["progress_done_coupling"]["target_done_0"]["n"] == 5
