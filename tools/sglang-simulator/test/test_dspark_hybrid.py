"""Hybrid provenance and output failures; no silicon data or latencies supplied."""

import hashlib
import io
import json
from contextlib import nullcontext
from dataclasses import replace
from types import SimpleNamespace

import pytest
from sglang_simulator.time_predictor.base import PredictorError
from sglang_simulator.time_predictor.dspark_umd import (
    UMDDSparkCostModel,
    _native_cost_query_evidence,
)
from sglang_simulator.time_predictor.infercast_dspark import (
    InferCastDSparkTimePredictor,
)
from test_dspark_source_work import protocol


def analytic_protocol():
    recorder, args, query = protocol()
    document = {
        "policy": "dsv41_native_region_roofline_v1",
        "execution_profile_id": args["source_policy"].execution_profile_id,
    }
    query.update(source="sol", resolution="sol", axes=[], resolved_key=None)
    recorder.nodes[0]["source"] = "sol"
    query["row_metadata"] = {
        "analytic_policy": document["policy"],
        "analytic_model_sha256": hashlib.sha256(
            json.dumps(document, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
        "execution_profile_id": document["execution_profile_id"],
    }
    return (
        recorder,
        {
            "point": args["point"],
            "phase": args["phase"],
            "source": "sol",
            "model": SimpleNamespace(to_dict=lambda: document),
        },
        query,
    )


def test_analytic_leaf_retains_typed_query_and_model_identity():
    recorder, args, query = analytic_protocol()
    assert _native_cost_query_evidence(recorder, **args) is query
    assert "latency_ms" not in query


@pytest.mark.parametrize(
    "change",
    [
        "dropped",
        "duplicate",
        "source",
        "requested",
        "policy",
        "model",
        "profile",
        "resolution",
    ],
)
def test_analytic_cost_cannot_hide_incomplete_or_different_evidence(change):
    recorder, args, query = analytic_protocol()
    if change == "dropped":
        recorder.dropped_queries = 1
    elif change == "duplicate":
        recorder.nodes.append(recorder.nodes[0])
    elif change == "source":
        query["source"] = "silicon"
    elif change == "requested":
        query["requested_key"] = {}
    elif change == "resolution":
        query["resolution"] = "exact"
    else:
        key = {
            "policy": "analytic_policy",
            "model": "analytic_model_sha256",
            "profile": "execution_profile_id",
        }[change]
        query["row_metadata"][key] = "different"
    with pytest.raises(PredictorError, match="matching SDK query evidence"):
        _native_cost_query_evidence(recorder, **args)


@pytest.mark.parametrize(
    "options",
    [
        {"cost_mode": "SOL"},
        {"cost_mode": True},
        {"cost_mode": "HYBRID"},
        {"native_analytic_model": {}},
        {"native_cost_evidence_path": "costs.jsonl"},
        {"cost_mode": "HYBRID", "native_analytic_model": {}},
    ],
)
def test_hybrid_requires_explicit_model_and_native_golden_route(options):
    with pytest.raises(PredictorError):
        InferCastDSparkTimePredictor(None, None, None, **options)


def test_cost_ledger_hashes_the_exact_written_jsonl_bytes():
    model = UMDDSparkCostModel(None)
    sink = io.BytesIO()
    model._cost_sink = sink
    line = b'{"event":"structural_provenance_test"}\n'
    model._write_cost_line(line)
    assert sink.getvalue() == line
    assert model._cost_sink_bytes == len(line)
    assert model._cost_sink_digest.hexdigest() == hashlib.sha256(line).hexdigest()
    assert model._cost_counts == {}


@pytest.mark.parametrize(
    "request_ids", [(("second", "first"),), (("active",), ()), ((), ("active",))]
)
def test_phase_record_retains_ordered_request_ids_outside_query_identity(
    monkeypatch, request_ids
):
    """Exercise the real writer with symbolic costs, not a measured cost fixture.

    The evaluator and cost value are protocol stand-ins. This test grants no
    source attestation, timing qualification or ADP2 cost support.
    """
    recorder, args, query = analytic_protocol()
    from infercast.sdk.diagnostics import prediction_evidence
    from infercast.sdk.models.umd import description, executor
    from sglang_simulator.time_predictor import dspark_umd
    from test_dspark_step import request, step

    before = step(dp=len(request_ids), fid=7)
    before = replace(
        before,
        groups=tuple(
            replace(
                group,
                forward_mode="TARGET_VERIFY" if ids else "IDLE",
                requests=tuple(request(rid) for rid in ids),
            )
            for group, ids in zip(before.groups, request_ids)
        ),
    )
    graph = SimpleNamespace(
        generation=[SimpleNamespace(op="protocol_only")],
        generation_composition=[],
        layer_repeat={},
    )
    model = UMDDSparkCostModel(
        SimpleNamespace(slice_ref=SimpleNamespace(slice_dir="protocol-only")),
        phase_preparers={
            "proposal": lambda **_: (
                SimpleNamespace(op_graph=graph, extra={}),
                args["point"],
                "generation",
            )
        },
    )
    model.native_analytic_model = args["model"]
    model._cost_sink = io.BytesIO()
    monkeypatch.setattr(description, "bind_runtime_point", lambda *args: {})
    monkeypatch.setattr(
        executor,
        "evaluate_phase_composition",
        lambda *args, **kwargs: SimpleNamespace(source="sol"),
    )
    monkeypatch.setattr(
        prediction_evidence,
        "capture_prediction_evidence",
        lambda: nullcontext(recorder),
    )

    class SymbolicSeconds(str):
        def __truediv__(self, divisor):
            assert divisor == 1000
            return self

        def __radd__(self, other):
            return self

    monkeypatch.setattr(
        dspark_umd, "float", lambda _: SymbolicSeconds("unmeasured"), raising=False
    )
    monkeypatch.setattr(
        dspark_umd,
        "DSparkPhaseCost",
        lambda phase, seconds, evidence_ref: SimpleNamespace(
            phase=phase, seconds=seconds, evidence_ref=evidence_ref
        ),
    )
    cost = model.query_phase("proposal", before)
    line = model._cost_sink.getvalue()
    record = json.loads(line)
    assert record["forward_id"] == before.forward_id
    assert record["request_ids_by_attention_dp"] == [list(ids) for ids in request_ids]
    assert record["identity"] == {
        "phase": "proposal",
        "runtime": args["point"],
        "slice": "protocol-only",
        "native_cost_query": query,
    }
    assert "request_ids_by_attention_dp" not in query["requested_key"]
    assert cost.evidence_ref == "umd:" + hashlib.sha256(line).hexdigest()
    assert model._cost_sink_digest.hexdigest() == hashlib.sha256(line).hexdigest()
    assert model._cost_sink_bytes == len(line)
    renamed = replace(
        before,
        groups=tuple(
            replace(
                group,
                requests=tuple(replace(r, rid=r.rid + "-new") for r in group.requests),
            )
            for group in before.groups
        ),
    )
    next_cost = model.query_phase("proposal", renamed)
    next_line = model._cost_sink.getvalue()[len(line) :]
    next_record = json.loads(next_line)
    assert next_record["identity"] == record["identity"]
    assert next_record["request_ids_by_attention_dp"] == [
        [rid + "-new" for rid in ids] for ids in request_ids
    ]
    assert next_cost.evidence_ref != cost.evidence_ref
    assert next_cost.evidence_ref == "umd:" + hashlib.sha256(next_line).hexdigest()


@pytest.mark.parametrize("partial", [False, True])
def test_sink_failure_cannot_publish_cost_counters_or_successful_digest(partial):
    class FailedSink:
        def write(self, raw):
            if partial:
                return len(raw) - 1
            raise OSError("owned test sink cannot write")

    model = UMDDSparkCostModel(None)
    model._cost_sink = FailedSink()
    with pytest.raises(PredictorError, match="evidence write failed"):
        model._write_cost_line(b'{"event":"structural_provenance_test"}\n')
    assert model._cost_sink_failed is True
    assert model._cost_sink_bytes == 0
    assert model._cost_sink_digest.hexdigest() == hashlib.sha256().hexdigest()
    assert model._cost_counts == {}
    with pytest.raises(PredictorError, match="evidence write failed"):
        model._write_cost_line(b'{"event":"retry_after_failure"}\n')
    with pytest.raises(PredictorError, match="evidence write failed"):
        model.get_cost_metrics()


def test_buffered_raw_failure_is_reported_before_returning_cost():
    class FailedRaw(io.RawIOBase):
        def writable(self):
            return True

        def write(self, raw):
            raise OSError("actual raw write failure below BufferedWriter")

    model = UMDDSparkCostModel(None)
    model._cost_sink = io.BufferedWriter(FailedRaw(), buffer_size=8192)
    with pytest.raises(PredictorError, match="evidence write failed"):
        model._write_cost_line(b'{"event":"buffered_write_failure"}\n')
    assert model._cost_sink_bytes == 0
    assert model._cost_sink_digest.hexdigest() == hashlib.sha256().hexdigest()
    with pytest.raises(PredictorError, match="evidence write failed"):
        model.get_cost_metrics()
