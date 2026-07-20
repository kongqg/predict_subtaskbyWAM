from tools.episode_done_summary import summarize_with_tolerance


def test_episode_tolerance_reclassifies_near_early():
    episodes = [
        {"task_id": 0, "episode": 1, "trigger_frame": 95, "done_start": 100, "early_by_frames": 5},
        {"task_id": 1, "episode": 2, "trigger_frame": 80, "done_start": 100, "early_by_frames": 20},
        {"task_id": 1, "episode": 3, "trigger_frame": None, "done_start": 100, "early_by_frames": None},
    ]
    strict = summarize_with_tolerance(episodes, tolerance=0)
    tol5 = summarize_with_tolerance(episodes, tolerance=5)
    assert strict["ok"] == 0
    assert strict["early"] == 2
    assert tol5["ok"] == 1
    assert tol5["early"] == 1
    assert tol5["miss"] == 1
