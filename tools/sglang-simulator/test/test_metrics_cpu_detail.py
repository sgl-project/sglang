"""Opt-in numeric CPU path; synthetic leaf costs, never timing admission."""

import copy
import io
import json
from types import SimpleNamespace as NS

import pytest
from msgspec.structs import replace
from sglang_simulator.simulation.iteration_stats import (
    IterationStats,
    append_iteration_metrics_or_detail,
)
from sglang_simulator.simulation.sglang.dspark import (
    CommittedHistory,
    DSparkSession,
    DSparkTopology,
)
from sglang_simulator.time_predictor.base import PredictorError
from sglang_simulator.time_predictor.simopus_native import SimOpusNativeCostAdapter
from sglang_simulator.utils.evidence import metrics_cpu_detail_enabled
from test_native_public_sdk import leaves as leaves
from test_native_public_sdk import public_adapter as public_adapter
from test_native_public_sdk import step_for


def enable(monkeypatch):
    monkeypatch.setenv("SGLANG_SIMULATOR_EVIDENCE_LEVEL", "metrics")
    monkeypatch.setenv("SGLANG_SIMULATOR_METRICS_CPU_DETAIL", "off")


def numeric_adapter(public_adapter, monkeypatch):
    _, database, options = public_adapter
    enable(monkeypatch)
    return SimOpusNativeCostAdapter.from_sdk(
        database, "deepseek-ai/DeepSeek-V4.1-Flash", **options
    )


def test_explicit_flag_requires_metrics_retention(monkeypatch):
    monkeypatch.delenv("SGLANG_SIMULATOR_METRICS_CPU_DETAIL", raising=False)
    assert metrics_cpu_detail_enabled()
    monkeypatch.setenv("SGLANG_SIMULATOR_METRICS_CPU_DETAIL", "off")
    monkeypatch.setenv("SGLANG_SIMULATOR_EVIDENCE_LEVEL", "full")
    with pytest.raises(ValueError, match="requires metrics"):
        metrics_cpu_detail_enabled()
    enable(monkeypatch)
    assert not metrics_cpu_detail_enabled()
    monkeypatch.setenv("SGLANG_SIMULATOR_METRICS_CPU_DETAIL", "unknown")
    with pytest.raises(ValueError, match="on or off"):
        metrics_cpu_detail_enabled()


@pytest.mark.parametrize("mode", ["EXTEND", "DECODE"])
def test_public_numeric_preserves_cost_bits_and_leaf_queries(
    public_adapter, leaves, monkeypatch, mode
):
    full, _, _ = public_adapter
    step = step_for(full, mode)
    expected = full.predict_step(step)
    queries = copy.deepcopy(leaves)
    leaves.clear()
    small = numeric_adapter(public_adapter, monkeypatch)
    actual = small.predict_step(step)
    assert [p.seconds.hex() for p in actual.phases] == [
        p.seconds.hex() for p in expected.phases
    ]
    assert actual.seconds.hex() == expected.seconds.hex()
    assert leaves == queries
    assert all(p.evidence_ref.startswith("simopus-metrics:") for p in actual.phases)
    assert small.last_evaluations == []
    assert small.evaluated_evidence_for(step) == ()
    evidence = small.get_cost_metrics()["evaluated_evidence"]
    assert evidence["sha256"] is None and evidence["bytes"] is None
    assert evidence["full_native_cost_ledger_available"] is False
    assert evidence["full_cost_digest_available"] is False
    assert evidence["query_source_classification_available"] is False
    assert evidence["persisted"] is False


def test_numeric_requires_factory_not_arbitrary_model(public_adapter, monkeypatch):
    full, _, _ = public_adapter
    enable(monkeypatch)
    with pytest.raises(PredictorError, match="public SDK factory"):
        SimOpusNativeCostAdapter(full.model, full.assets)


def test_prepared_numeric_pair_rejects_unrelated_assets(public_adapter, monkeypatch):
    full, _, _ = public_adapter
    enable(monkeypatch)
    with pytest.raises(PredictorError, match="bound to these assets"):
        SimOpusNativeCostAdapter.from_prepared_sdk(full.model, copy.copy(full.assets))
    with pytest.raises(PredictorError, match="bound to these assets"):
        SimOpusNativeCostAdapter.from_prepared_sdk(
            NS(estimate_prepared_forward_ms=lambda **kwargs: 1.0), full.assets
        )


@pytest.mark.parametrize("detail", ["on", "off"])
def test_source_and_policy_guards_survive(public_adapter, monkeypatch, detail):
    _, database, options = public_adapter
    monkeypatch.setenv("SGLANG_SIMULATOR_EVIDENCE_LEVEL", "metrics")
    monkeypatch.setenv("SGLANG_SIMULATOR_METRICS_CPU_DETAIL", detail)
    bad = copy.deepcopy(options)
    bad["source_binding"]["profile"]["sha256"] = "0" * 64
    with pytest.raises(PredictorError):
        SimOpusNativeCostAdapter.from_sdk(
            database, "deepseek-ai/DeepSeek-V4.1-Flash", **bad
        )
    from simopus.sdk.native_segment_module import SOURCE_WORK_POLICY

    with pytest.raises(PredictorError):
        SimOpusNativeCostAdapter.from_sdk(
            database,
            "deepseek-ai/DeepSeek-V4.1-Flash",
            **options,
            source_work_policy=SOURCE_WORK_POLICY,
        )


@pytest.mark.parametrize("detail", ["on", "off"])
@pytest.mark.parametrize("mutation", ["topology", "runtime", "buffer_capacity"])
def test_native_domain_guards_survive(
    public_adapter, leaves, monkeypatch, detail, mutation
):
    full, database, options = public_adapter
    step = step_for(full)
    monkeypatch.setenv("SGLANG_SIMULATOR_EVIDENCE_LEVEL", "metrics")
    monkeypatch.setenv("SGLANG_SIMULATOR_METRICS_CPU_DETAIL", detail)
    adapter = SimOpusNativeCostAdapter.from_sdk(
        database, "deepseek-ai/DeepSeek-V4.1-Flash", **options
    )
    if mutation == "topology":
        step = replace(
            step, topology=DSparkTopology(4 if step.topology.tp_size == 2 else 2, 1)
        )
    elif mutation == "runtime":
        step = replace(step, runtime_profile_json='{"invalid":true}')
    else:
        group = step.groups[0]
        inputs = json.loads(group.execution_json)
        inputs["native_region_stage_inputs"]["buffer_capacity"] = -1
        step = replace(
            step,
            groups=(
                replace(
                    group,
                    execution_json=json.dumps(
                        inputs, sort_keys=True, separators=(",", ":")
                    ),
                ),
            ),
        )
    with pytest.raises(PredictorError):
        adapter.predict_step(step)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), 0.0, -1.0, True])
def test_numeric_invalid_cost_refused(public_adapter, monkeypatch, value):
    full, _, _ = public_adapter
    step = step_for(full)
    adapter = numeric_adapter(public_adapter, monkeypatch)
    monkeypatch.setattr(adapter, "_numeric_forward", lambda **kwargs: value)
    with pytest.raises(PredictorError, match="prepared stage refused"):
        adapter.predict_step(step)
    with pytest.raises(PredictorError, match="coverage is incomplete"):
        adapter.evaluated_evidence_for(step)


def test_numeric_late_failure_does_not_draw_progress_or_commit(
    public_adapter, leaves, monkeypatch
):
    from simopus.sdk.perf_database.database import PerfDataNotAvailableError

    full, _, _ = public_adapter
    step = step_for(full)
    adapter = numeric_adapter(public_adapter, monkeypatch)
    original = adapter._numeric_forward

    def fail_late(**kwargs):
        if kwargs["runtime"]["native_region_stage"]["phase"] == "target_verify":
            raise PerfDataNotAvailableError("synthetic missing numeric cost")
        return original(**kwargs)

    monkeypatch.setattr(adapter, "_numeric_forward", fail_late)
    draws = []
    session = DSparkSession(
        adapter, NS(predict_progress=lambda before: draws.append(before))
    )
    for request in step.requests:
        session.record_history(
            request.rid,
            CommittedHistory(
                request.target_prefix_len,
                request.draft_valid_from,
                request.draft_valid_to,
            ),
        )
    history = copy.deepcopy(session._history)
    with pytest.raises(PredictorError):
        session.execute(step)
    assert not draws and session._history == history and session._last_forward_id == -1
    with pytest.raises(PredictorError, match="coverage is incomplete"):
        adapter.evaluated_evidence_for(step)


@pytest.mark.parametrize("mode", ["full", "metrics", "off"])
def test_lazy_detail_keeps_default_bytes_and_metrics(tmp_path, monkeypatch, mode):
    monkeypatch.setenv(
        "SGLANG_SIMULATOR_EVIDENCE_LEVEL", "full" if mode == "full" else "metrics"
    )
    monkeypatch.setenv(
        "SGLANG_SIMULATOR_METRICS_CPU_DETAIL", "off" if mode == "off" else "on"
    )
    store = IterationStats(lambda: tmp_path, metrics_only=mode != "full")
    row = {
        "forward_mode": "EXTEND",
        "requests": [{"rid": "one"}],
        "forward_latency": 0.125,
        "l2_load_latency": 0.25,
        "cpu_overhead": 0.5,
        "native_worker_event": {"scope": "synthetic"},
    }
    metrics = {
        key: row[key] for key in ("forward_latency", "l2_load_latency", "cpu_overhead")
    }
    calls = []

    def detail():
        calls.append(True)
        return row

    append_iteration_metrics_or_detail(store, metrics, detail)
    assert store.metric_rows() == [metrics]
    assert bool(calls) == (mode != "off")
    if mode == "full":
        output = io.StringIO()
        store.write_jsonl(output)
        assert output.getvalue() == json.dumps(row) + "\n"
    else:
        assert store.path is None
    store.clear()
