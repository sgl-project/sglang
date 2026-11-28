import json
import math
from pathlib import Path
from unittest.mock import patch

import msgspec
import pytest
from sglang_simulator.simulation.manager.config import ConfigManager
from sglang_simulator.time_predictor import (
    InferTimePredictor,
    ReplayTimePredictor,
    ScheduleBatch,
    ScheduleRequest,
)
from sglang_simulator.time_predictor.energy import (
    EnergyCalibration,
    EnergyPredictor,
    calc_energy_per_token_metrics,
)
from sglang_simulator.time_predictor.ml_features import (
    ML_FEATURE_NAMES,
    extract_ml_features,
)

CALIBRATION = {
    "format_version": 1,
    "idle_power_w": 50.0,
    "power_limit_w": 400.0,
    "prefill": {"intercept_w": 300.0, "coefficients_w": {"log1p_sum_extend": 10.0}},
    "decode": {"intercept_w": -100.0, "coefficients_w": {"log1p_batch_size": 200.0}},
    "provenance": {"device": "test fixture"},
}


class FixedLatencyPredictor(InferTimePredictor):
    def __init__(self, latencies: dict[int, float]):
        super().__init__(None, None, None)
        self.latencies = latencies
        self.resets = 0

    def predict_infer_time(self, batch: ScheduleBatch) -> float:
        return self.latencies[batch.batch_size]

    def get_metrics(self) -> dict:
        return {"inner_resets": self.resets}

    def reset_metrics(self) -> None:
        self.resets += 1


class ManualClock:
    def __init__(self):
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


def _batch(batch_size: int, *, extend: int = 1, past: int = 16) -> ScheduleBatch:
    return ScheduleBatch(
        reqs=[
            ScheduleRequest(extend_length=extend, past_kv_length=past)
            for _ in range(batch_size)
        ]
    )


def _write(tmp_path: Path, name: str, data: dict) -> Path:
    path = tmp_path / name
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


def _make_predictor(tmp_path, latencies, calibration=CALIBRATION):
    clock = ManualClock()
    inner = FixedLatencyPredictor(latencies)
    predictor = EnergyPredictor(
        inner=inner,
        calibration=EnergyCalibration.from_file(
            str(_write(tmp_path, "calibration.json", calibration))
        ),
        clock=clock,
    )
    return predictor, inner, clock


def test_steps_are_charged_at_clipped_power_and_gaps_at_idle(tmp_path):
    predictor, _, clock = _make_predictor(tmp_path, {2: 0.5, 1: 0.1, 64: 0.1, 3: -1.0})

    assert predictor.predict_infer_time(_batch(2, extend=1024, past=0)) == 0.5
    clock.now = 1.0
    predictor.predict_infer_time(_batch(1))  # 38.6 W before clipping to idle
    clock.now = 1.1
    predictor.predict_infer_time(_batch(64))  # 734.9 W before clipping to the limit
    assert predictor.predict_infer_time(_batch(3)) == -1.0
    clock.now = 1.2  # back to back with the batch-64 step: no gap
    predictor.predict_infer_time(_batch(1))

    prefill_w = 300.0 + 10.0 * math.log1p(2 * 1024)
    metrics = predictor.get_metrics()
    assert metrics["inner_resets"] == 0
    assert metrics["energy_prefill_j"] == pytest.approx(prefill_w * 0.5)
    assert metrics["energy_decode_j"] == pytest.approx(50.0 * 0.2 + 400.0 * 0.1)
    assert metrics["energy_busy_s"] == pytest.approx(0.8)
    assert metrics["energy_idle_s"] == pytest.approx(0.5)
    assert metrics["energy_total_j"] == pytest.approx(prefill_w * 0.5 + 50.0 + 25.0)
    assert metrics["energy_avg_power_w"] == pytest.approx(
        metrics["energy_total_j"] / 1.3
    )


def test_active_gap_charges_the_start_of_each_gap(tmp_path):
    calibration = {**CALIBRATION, "active_gap": {"power_w": 100.0, "max_s": 0.2}}
    predictor, _, clock = _make_predictor(tmp_path, {1: 0.1}, calibration)

    predictor.predict_infer_time(_batch(1))
    clock.now = 0.2  # 0.1 s gap, all of it active
    predictor.predict_infer_time(_batch(1))
    clock.now = 1.0  # 0.7 s gap: 0.2 s active, 0.5 s idle
    predictor.predict_infer_time(_batch(1))

    metrics = predictor.get_metrics()
    assert metrics["energy_active_gap_s"] == pytest.approx(0.3)
    assert metrics["energy_idle_s"] == pytest.approx(0.5)
    assert metrics["energy_active_gap_j"] == pytest.approx(30.0)
    assert metrics["energy_idle_j"] == pytest.approx(25.0)
    assert metrics["energy_total_j"] == pytest.approx(3 * 0.1 * 50.0 + 30.0 + 25.0)


def test_reset_starts_a_new_interval(tmp_path):
    """Time before a profile restart must not be charged as idle to the next interval."""
    predictor, inner, clock = _make_predictor(tmp_path, {1: 0.1})
    predictor.predict_infer_time(_batch(1))

    predictor.reset_metrics()
    clock.now = 100.0
    predictor.predict_infer_time(_batch(1))

    metrics = predictor.get_metrics()
    assert metrics["energy_idle_s"] == 0.0
    assert metrics["energy_busy_s"] == pytest.approx(0.1)
    assert inner.resets == 1


@pytest.mark.parametrize(
    "name",
    [*ML_FEATURE_NAMES, *(f"log1p_{n}" for n in ML_FEATURE_NAMES)],
)
@pytest.mark.parametrize("extend", [1, 300])
def test_every_feature_matches_the_ml_extractor(tmp_path, name, extend):
    """Some features are read from ScheduleBatch sums instead of the extractor."""
    calibration = {
        "format_version": 1,
        "idle_power_w": 0.0,
        "prefill": {"intercept_w": 1.0, "coefficients_w": {name: 1.0}},
        "decode": {"intercept_w": 1.0, "coefficients_w": {name: 1.0}},
    }
    batch = ScheduleBatch(
        reqs=[
            ScheduleRequest(extend_length=extend, past_kv_length=past)
            for past in (0, 7, 4096)
        ]
    )
    predictor, _, _ = _make_predictor(tmp_path, {3: 2.0}, calibration)
    predictor.predict_infer_time(batch)

    features = extract_ml_features([extend] * 3, [0, 7, 4096])
    base = name.removeprefix("log1p_")
    value = features[ML_FEATURE_NAMES.index(base)]
    if base != name:
        value = math.log1p(value)
    assert predictor.get_metrics()["energy_busy_j"] == pytest.approx(2.0 * (1 + value))


def test_energy_is_opt_in(tmp_path):
    table = _write(tmp_path, "replay.json", {"[[1, 0]]": 0.125})
    replay = {"name": "replay", "database_path": str(table)}
    with patch.object(ConfigManager, "_raw_config", {"predictor": replay}):
        predictor = ConfigManager.get_inference_time_predictor(None, None, None)
    assert type(predictor) is ReplayTimePredictor
    assert calc_energy_per_token_metrics({"total_input": 10, "total_output": 4}) == {}

    calibration = _write(tmp_path, "calibration.json", CALIBRATION)
    config = {"predictor": {**replay, "energy": {"calibration_path": str(calibration)}}}
    with patch.object(ConfigManager, "_raw_config", config):
        wrapped = ConfigManager.get_inference_time_predictor(None, None, None)
    assert isinstance(wrapped, EnergyPredictor)
    assert wrapped.predict_infer_time(_batch(1, past=0)) == 0.125


def test_calibration_path_expands_environment_variables(tmp_path, monkeypatch):
    _write(tmp_path, "calibration.json", CALIBRATION)
    monkeypatch.setenv("ENERGY_CALIBRATION_DIR", str(tmp_path))
    calibration = EnergyCalibration.from_file(
        "${ENERGY_CALIBRATION_DIR}/calibration.json"
    )
    assert calibration.idle_power_w == CALIBRATION["idle_power_w"]


def test_calibration_path_is_relative_to_the_config(tmp_path, monkeypatch):
    table = _write(tmp_path, "replay.json", {"[[1, 0]]": 0.125})
    _write(tmp_path, "calibration.json", CALIBRATION)
    config = {
        "predictor": {
            "name": "replay",
            "database_path": str(table),
            "energy": {"calibration_path": "calibration.json"},
        }
    }
    monkeypatch.setenv(
        "SGLANG_SIMULATOR_CONFIG_PATH", str(_write(tmp_path, "sim_config.json", config))
    )
    with patch.object(ConfigManager, "_raw_config", config):
        wrapped = ConfigManager.get_inference_time_predictor(None, None, None)
    assert isinstance(wrapped, EnergyPredictor)


@pytest.mark.parametrize(
    "energy", [{}, {"calibration_path": "x.json", "idle_power_w": 1.0}, "x.json"]
)
def test_energy_block_takes_only_calibration_path(tmp_path, energy):
    table = _write(tmp_path, "replay.json", {"[[1, 0]]": 0.125})
    config = {
        "predictor": {"name": "replay", "database_path": str(table), "energy": energy}
    }
    with patch.object(ConfigManager, "_raw_config", config):
        with pytest.raises(ValueError, match="calibration_path"):
            ConfigManager.get_inference_time_predictor(None, None, None)


@pytest.mark.parametrize(
    "change",
    [
        {"format_version": 2},
        {"idle_power": 50.0},
        {"power_limit_w": 10.0},
        {"decode": {"intercept_w": 1.0, "coefficients_w": {"log1p_batchsize": 1.0}}},
        {"active_gap": {"power_w": 500.0, "max_s": 0.1}},
        {"active_gap": {"power_w": 10.0, "max_s": 0.1}},
        {"active_gap": {"power_w": 100.0, "max_s": -1.0}},
    ],
)
def test_calibration_rejects_invalid_files(tmp_path, change):
    path = _write(tmp_path, "bad.json", {**CALIBRATION, **change})
    with pytest.raises(msgspec.ValidationError):
        EnergyCalibration.from_file(str(path))
