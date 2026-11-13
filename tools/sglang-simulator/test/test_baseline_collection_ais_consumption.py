"""The collector's exported observations file trains the real AIS SDK."""

import hashlib
import json

import pytest
from sglang_simulator.time_predictor.aisimulate import batch_to_fpm_metrics
from sglang_simulator.time_predictor.base import ScheduleBatch, ScheduleRequest
from test_aisimulate_native import _predictor
from test_baseline_collection_export import (
    _decode_batch,
    _mixed_batch,
    _prefill_batch,
    install_collector,
)

QUERIES = {
    "prefill": ScheduleBatch([ScheduleRequest(512, 0, "extend")], "extend"),
    "decode": ScheduleBatch([ScheduleRequest(1, 512, "decode")] * 2, "decode"),
    "mixed": ScheduleBatch(
        [
            ScheduleRequest(256, 128, "extend"),
            ScheduleRequest(1, 512, "decode"),
            ScheduleRequest(1, 512, "decode"),
        ],
        "mixed",
    ),
}

_FIELDS = {
    "num_prefill_requests",
    "sum_prefill_tokens",
    "sum_prefill_kv_tokens",
    "var_prefill_length",
    "num_decode_requests",
    "sum_decode_kv_tokens",
    "var_decode_kv_tokens",
    "extend_lengths",
    "past_kv_lengths",
}


@pytest.fixture
def exported_observations(tmp_path, monkeypatch):
    """A real collector export: six steps per mode, varied work per step and
    deterministic distinct wall times from the shared FakeTime clock."""
    scheduler, state = install_collector(tmp_path, monkeypatch)
    for extend in (64, 128, 256, 384, 512, 1024):
        scheduler.run_batch(_prefill_batch(extend))
    for seqs in (
        [17, 33],
        [65],
        [17, 33, 129, 257],
        [513, 1025],
        [2049],
        [33, 65, 129],
    ):
        scheduler.run_batch(_decode_batch(seqs))
    for extend, prefix, decode_seqs in (
        (128, 0, [129]),
        (192, 64, [257, 513]),
        (256, 128, [641]),
        (384, 0, [129, 257]),
        (512, 256, [1025, 1025]),
        (768, 512, [769]),
    ):
        scheduler.run_batch(_mixed_batch(extend, prefix, decode_seqs))
    paths = state.stop_and_export()
    path = tmp_path / "TP0.ais_observations.json"
    observations = json.loads(path.read_text())
    assert path in paths and len(observations) == 18
    # Every row carries exactly the SDK's native observation fields.
    assert all(
        set(obs[0]) == {"scheduled_requests", "wall_time"} for obs in observations
    )
    assert all(set(obs[0]["scheduled_requests"]) == _FIELDS for obs in observations)
    return path


@pytest.mark.parametrize(
    "estimation_mode,estimator_config",
    [
        ("fpm_regression", {}),
        ("op_level", {"correction": {"enabled": True}}),
    ],
)
def test_collector_export_tunes_real_sdk(
    exported_observations, estimation_mode, estimator_config
):
    """The exported file is consumed as-is through tune_observations: the
    regression stores become ready for every local workload, the corrected
    op_level factor leaves its default, and the file is never modified."""
    path = exported_observations
    before = hashlib.sha256(path.read_bytes()).hexdigest()
    predictor = _predictor(
        database_mode="SOL",
        estimation_mode=estimation_mode,
        estimator_config=estimator_config,
        tune_observations=str(path),
    )
    try:
        for query in QUERIES.values():
            assert predictor.predict_infer_time(query) > 0
        if estimation_mode == "fpm_regression":
            readiness = {
                diag["workload_kind"]: diag
                for diag in predictor._fpm.regression_store_diagnostics()
            }
            assert all(
                readiness[kind]["ready"]
                for kind in ("pure_prefill", "pure_decode", "contains_locally_mixed")
            )
        else:
            metrics = batch_to_fpm_metrics(QUERIES["prefill"])
            detailed = predictor._fpm.estimate_forward_pass_detailed(metrics)
            assert detailed["correction_factor"] != pytest.approx(1.0)
    finally:
        predictor._fpm.close()
    assert hashlib.sha256(path.read_bytes()).hexdigest() == before
