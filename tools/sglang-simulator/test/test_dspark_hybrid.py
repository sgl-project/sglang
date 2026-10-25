"""Hybrid provenance and output failures; no silicon data or latencies supplied."""

import hashlib
import io
import json
from types import SimpleNamespace

import pytest
from test_dspark_source_work import protocol

from sglang_simulator.time_predictor.base import PredictorError
from sglang_simulator.time_predictor.dspark_umd import (
    UMDDSparkCostModel,
    _native_cost_query_evidence,
)
from sglang_simulator.time_predictor.infercast_dspark import (
    InferCastDSparkTimePredictor,
)


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
