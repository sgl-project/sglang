"""Structural recorder protocol only: no source qualification or timing rows."""

import copy
import hashlib
import json
from types import SimpleNamespace

import pytest
from sglang_simulator.time_predictor.base import PredictorError
from sglang_simulator.time_predictor.dspark_umd import (
    UMDDSparkCostModel,
    _source_work_query_evidence,
)
from test_dspark_native_regions import CASES, prepared


def protocol(
    *, region="decode_target", phase="target_verify", same_prefix=False, tp_size=2
):
    sdk = pytest.importorskip("infercast.sdk.native_segment_module")
    if not hasattr(sdk, "SOURCE_WORK_POLICY"):
        pytest.skip("SDK source-work API unavailable")
    from infercast.sdk.diagnostics.prediction_evidence import linear_axis

    binding = prepared("TARGET_VERIFY", phase, region, capture_size=2, tp_size=tp_size)
    original = sdk.NativeSegmentQuery.from_shape(
        json.loads(binding["point"]["native_segment_queries"])[0]
    )

    def query(prefix, budget, bound):
        value = json.loads(original.module_variant)
        native = value["native_forward_input"]
        native["groups"][0]["requests"][0]["prefix_len"] = prefix
        native["groups"][0]["target_seq_len_upper_bound"] = bound
        recipe = native["sampling_recipe"]["per_attention_dp"][0][0]
        recipe.update(ignore_eos=True, max_new_tokens=budget)
        return sdk.NativeSegmentQuery(
            original.execution_profile_id,
            region,
            json.dumps(value, sort_keys=True, separators=(",", ":")),
        )

    requested = query(512, 20, 700)
    lower, upper = (512, 512) if same_prefix else (500, 524)
    anchors = [query(lower, 14, 701).to_shape(), query(upper, 14, 702).to_shape()]
    if same_prefix:
        anchors[1] = copy.deepcopy(anchors[0])
    coordinate, prefix = requested.source_work_coordinate(policy=sdk.SOURCE_WORK_POLICY)
    weight = 0.0 if same_prefix else 0.5
    record = {
        "component": sdk.OP,
        "role": "leaf",
        "source": "empirical",
        "kernel_source": "sglang",
        "kernel_impl": requested.to_shape()["kernel_impl"],
        "resolution": "fallback" if same_prefix else "interpolated",
        "requested_key": requested.to_shape(),
        "resolved_key": anchors[0],
        "axes": [linear_axis("prefix_len", prefix, lower, upper, weight)],
        "row_metadata": {
            "execution_profile_id": requested.execution_profile_id,
            "estimated_source_work_policy": sdk.SOURCE_WORK_POLICY,
            "interpolation_method": sdk.SOURCE_WORK_POLICY,
            "source_binding_scope": "conditional_helper_file_inventory",
            "source_work_coordinate_sha256": hashlib.sha256(
                coordinate.encode()
            ).hexdigest(),
            "anchor_identities": anchors,
            "anchor_weights": [1 - weight, weight],
        },
    }
    recorder = SimpleNamespace(
        dropped_queries=0,
        nodes=[
            {
                "op": sdk.OP,
                "node_id": region,
                "phase": "generation",
                "status": "resolved",
                "source": "empirical",
                "count": 1,
                "repeat": 1,
                "shape": requested.to_shape(),
                "queries": [record],
            }
        ],
    )
    args = {
        "point": {"native_segment_queries": json.dumps([requested.to_shape()])},
        "phase": "generation",
        "policy": sdk.SOURCE_WORK_POLICY,
        "source_policy": SimpleNamespace(
            execution_profile_id=requested.execution_profile_id
        ),
    }
    return recorder, args, record


@pytest.mark.parametrize("same_prefix", [False, True])
@pytest.mark.parametrize("tp_size", [2, 4])
@pytest.mark.parametrize("mode,phase,region", CASES[4:])
def test_bounded_original_query_proof_for_each_decode_region(
    mode, phase, region, same_prefix, tp_size
):
    recorder, args, record = protocol(
        region=region, phase=phase, same_prefix=same_prefix, tp_size=tp_size
    )
    assert _source_work_query_evidence(recorder, **args) is record
    # No numeric latency, row or qualified source-policy fixture was supplied.
    assert "latency_ms" not in record


def test_tp4_query_proof_cannot_borrow_tp2_anchor_identities():
    recorder, args, record = protocol(tp_size=4)
    _, _, tp2 = protocol(tp_size=2)
    record["row_metadata"]["anchor_identities"] = tp2["row_metadata"][
        "anchor_identities"
    ]
    with pytest.raises(PredictorError, match="matching SDK source-work evidence"):
        _source_work_query_evidence(recorder, **args)


@pytest.mark.parametrize(
    "change",
    [
        "dropped",
        "no_node",
        "two_nodes",
        "wrong_op",
        "wrong_phase",
        "count_bool",
        "repeat",
        "skipped",
        "node_source",
        "node_shape",
        "no_query",
        "two_queries",
        "component",
        "role",
        "source",
        "kernel_source",
        "kernel_impl",
        "resolution",
        "requested",
        "resolved",
        "profile",
        "policy",
        "scope",
        "coordinate",
        "anchors",
        "different_capture",
        "out_of_domain",
        "weights",
        "axes",
        "missing_metadata",
    ],
)
def test_empirical_string_or_incomplete_mismatched_proof_is_rejected(change):
    recorder, args, record = protocol()
    node, metadata = recorder.nodes[0], record["row_metadata"]
    if change == "dropped":
        recorder.dropped_queries = 1
    elif change == "no_node":
        recorder.nodes = []
    elif change == "two_nodes":
        recorder.nodes.append(copy.deepcopy(node))
    elif change == "wrong_op":
        node["op"] = "gemm"
    elif change == "wrong_phase":
        node["phase"] = "context"
    elif change == "count_bool":
        node["count"] = True
    elif change == "repeat":
        node["repeat"] = 2
    elif change == "skipped":
        node["status"] = "skipped"
    elif change == "node_source":
        node["source"] = "silicon"
    elif change == "node_shape":
        node["shape"] = {}
    elif change == "no_query":
        node["queries"] = []
    elif change == "two_queries":
        node["queries"].append(copy.deepcopy(record))
    elif change in {
        "component",
        "role",
        "source",
        "kernel_source",
        "kernel_impl",
        "resolution",
    }:
        record[change] = "other"
    elif change == "requested":
        record["requested_key"] = {}
    elif change == "resolved":
        record["resolved_key"] = metadata["anchor_identities"][1]
    elif change == "profile":
        metadata["execution_profile_id"] = "b" * 64
    elif change == "policy":
        metadata["estimated_source_work_policy"] = "other"
    elif change == "scope":
        metadata["source_binding_scope"] = "caller_approval"
    elif change == "coordinate":
        metadata["source_work_coordinate_sha256"] = "b" * 64
    elif change == "anchors":
        metadata["anchor_identities"] = []
    elif change in {"different_capture", "out_of_domain"}:
        from infercast.sdk.native_segment_module import NativeSegmentQuery

        anchor = NativeSegmentQuery.from_shape(metadata["anchor_identities"][0])
        value = json.loads(anchor.module_variant)
        if change == "different_capture":
            value["target_capture_key"]["size"] = 4
            value["draft_capture_key"]["size"] = 4
        else:
            value["native_forward_input"]["groups"][0]["requests"][0]["prefix_len"] = (
                520
            )
        metadata["anchor_identities"][0] = NativeSegmentQuery(
            anchor.execution_profile_id,
            anchor.segment_kind,
            json.dumps(value, sort_keys=True, separators=(",", ":")),
        ).to_shape()
    elif change == "weights":
        metadata["anchor_weights"] = [0.1, 0.9]
    elif change == "axes":
        record["axes"] = []
    elif change == "missing_metadata":
        record["row_metadata"] = {}
    with pytest.raises(PredictorError, match="matching SDK source-work evidence"):
        _source_work_query_evidence(recorder, **args)


@pytest.mark.parametrize(
    "policy", [True, "unknown", "native_region_golden351_decode_work_v1"]
)
def test_runtime_option_never_enables_a_missing_typed_native_policy(policy):
    pytest.importorskip("infercast.sdk.native_golden_material")
    with pytest.raises(PredictorError, match="verified golden source"):
        UMDDSparkCostModel(None, native_source_work_policy=policy)


def test_default_cost_model_has_no_selected_work_policy():
    assert UMDDSparkCostModel(None).native_source_work_policy is None


def test_same_prefix_fallback_cannot_hide_two_different_original_anchors():
    recorder, args, record = protocol(same_prefix=True)
    from infercast.sdk.native_segment_module import NativeSegmentQuery

    anchors = record["row_metadata"]["anchor_identities"]
    anchor = NativeSegmentQuery.from_shape(anchors[1])
    value = json.loads(anchor.module_variant)
    value["native_forward_input"]["sampling_recipe"]["per_attention_dp"][0][0][
        "max_new_tokens"
    ] = 15
    anchors[1] = NativeSegmentQuery(
        anchor.execution_profile_id,
        anchor.segment_kind,
        json.dumps(value, sort_keys=True, separators=(",", ":")),
    ).to_shape()
    with pytest.raises(PredictorError, match="matching SDK source-work evidence"):
        _source_work_query_evidence(recorder, **args)


def test_missing_query_proof_precedes_progress_and_history_mutation():
    from test_dspark_step import session, step

    recorder, args, _ = protocol()
    recorder.nodes[0]["queries"] = []

    class MissingProofCost:
        def predict_step(self, before):
            return _source_work_query_evidence(recorder, **args)

    class NoProgress:
        def predict_progress(self, before):
            pytest.fail("missing SDK proof reached progress")

    before = step()
    run = session(before, cost=MissingProofCost(), progress=NoProgress())
    original = tuple(run.history_for(r.rid) for r in before.requests)
    with pytest.raises(PredictorError, match="matching SDK source-work evidence"):
        run.execute(before)
    assert tuple(run.history_for(r.rid) for r in before.requests) == original
