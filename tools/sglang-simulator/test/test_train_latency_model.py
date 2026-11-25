"""Minimal example: train on collected steps, then load the models for prediction."""

import json
import math
from pathlib import Path

from sglang_simulator.time_predictor.base import ScheduleBatch, ScheduleRequest
from sglang_simulator.time_predictor.ml import MLTimePredictor
from sglang_simulator.tools.train_ml_time_predictor import train as trainer


def test_train_and_predict(tmp_path):
    example = Path(trainer.__file__).with_name("example_steps.jsonl")
    output = tmp_path / "models"
    trainer.main(["--data-root", str(example), "--out-dir", str(output)])

    summary = json.loads((output / "summary.json").read_text())
    assert summary["train_rows"] > 0 and summary["test_rows"] > 0
    assert summary["train_rows"] + summary["test_rows"] == summary["final_fit_rows"]
    assert set(summary["metrics"]) == {"l2", "p50"}

    batch = ScheduleBatch(
        [
            ScheduleRequest(extend_length=4, past_kv_length=12),
            ScheduleRequest(extend_length=8, past_kv_length=0),
        ]
    )
    for loss in ("l2", "p50"):
        model = output / f"latency_model_hgbmono_{loss}.pkl"
        predictor = MLTimePredictor(None, None, None, str(model))
        latency = predictor.predict_infer_time(batch)
        assert math.isfinite(latency) and latency > 0
