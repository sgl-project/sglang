import json

import pytest
from sglang_simulator.time_predictor import ScheduleBatch, ScheduleRequest
from sglang_simulator.time_predictor.replay import ReplayTimePredictor


def make_predictor(tmp_path, **kwargs):
    table_path = tmp_path / "replay.json"
    table_path.write_text(json.dumps({json.dumps([[4, 0]]): 0.125}))
    return ReplayTimePredictor(
        model=None,
        hw=None,
        config=None,
        database_path=str(table_path),
        **kwargs,
    )


def test_replay_reports_exact_and_knn_fallback_metrics(tmp_path):
    predictor = make_predictor(
        tmp_path,
        miss_strategy="knn",
    )

    assert predictor.predict_infer_time(
        ScheduleBatch([ScheduleRequest(4, 0)])
    ) == pytest.approx(0.125)
    assert predictor.predict_infer_time(
        ScheduleBatch([ScheduleRequest(8, 0)])
    ) == pytest.approx(0.125)

    assert predictor.get_metrics() == {
        "replay_exact_match_steps": 1,
        "replay_miss_steps": 1,
        "replay_zero_fallback_steps": 0,
        "replay_knn_fallback_steps": 1,
        "replay_fallback_rate": 0.5,
    }

    predictor.reset_metrics()
    assert predictor.get_metrics()["replay_fallback_rate"] == 0.0
    assert predictor.get_metrics()["replay_exact_match_steps"] == 0
    assert predictor.get_metrics()["replay_miss_steps"] == 0
