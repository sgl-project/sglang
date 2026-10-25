"""Validate the SDK-to-simulator native phase binding without projecting work.

The SDK owns causal projection, source qualification and UMD preparation. These
checks only bind its one-region graph to the existing simulator phase and step.
They neither construct source attestations nor enable an incomplete factory.
"""

import hashlib
import json

from sglang_simulator.time_predictor.base import PredictorError

_REGIONS = {
    "EXTEND": {
        "worker_preamble": "prefill_preamble",
        "target_prefill": "prefill_target",
        "draft_kv_inject": "prefill_draft_kv",
        "worker_tail": "prefill_tail",
    },
    "TARGET_VERIFY": {
        "worker_preamble": "decode_preamble",
        "proposal": "decode_proposal",
        "target_verify": "decode_target",
        "accept_commit_publication": "decode_accept_commit",
        "worker_tail": "decode_tail",
    },
}


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def validate_native_region_phase(*, description, point, graph_phase, phase, step):
    """Return whether this is a native region; reject an incomplete binding.

    Runtime/source attestations must originate in the SDK's prospective
    preparer. Current SDK region preparation alone does not issue them. The
    existing canonical module_variant is transported unchanged, not rebuilt
    from a simulator mirror or a current-forward outcome.
    """
    graph = description.op_graph
    nodes = getattr(graph, graph_phase)
    if description.extra.get(
        "module_operation"
    ) != "dsv41_native_segment_module" and not any(
        node.op == "dsv41_native_segment_module" for node in nodes
    ):
        return False

    from infercast.sdk.native_segment_module import OP, NativeSegmentQuery

    def require(condition):
        if not condition:
            raise ValueError("native region does not bind the complete causal phase")

    try:
        require(
            (
                step.topology.tp_size,
                step.topology.attention_dp_size,
                step.topology.pp_size,
            )
            == (2, 1, 1)
            and len(step.groups) == 1
        )
        group = step.groups[0]
        region = _REGIONS[group.forward_mode][phase]
        expected_phase = "context" if group.forward_mode == "EXTEND" else "generation"
        require(graph_phase == expected_phase)
        require(
            len(nodes) == 1
            and not getattr(
                graph, "generation" if graph_phase == "context" else "context"
            )
        )
        node = nodes[0]
        require(
            node.id == region
            and node.op == OP
            and type(node.count) is int
            and node.count == 1
            and node.cond is None
            and node.layer_group == "all"
            and graph.layer_repeat == {"all": 1}
            and type(graph.layer_repeat["all"]) is int
            and getattr(graph, graph_phase + "_composition")
            == ["sum", ["node", region]]
        )
        shapes = json.loads(point["native_segment_queries"])
        require(isinstance(shapes, list) and len(shapes) == 1)
        query = NativeSegmentQuery.from_shape(shapes[0])
        native = query.native_input
        require(
            query.segment_kind == region
            and _canonical(node.shape)
            == _canonical(
                {key: ["lit", value] for key, value in query.to_shape().items()}
            )
            and description.extra.get("module_operation") == OP
            and description.extra.get("native_worker_phase") == native.forward_mode
            and description.extra.get("complete_dspark_worker_phase") == phase
            and description.extra.get("native_segment_profile_id")
            == query.execution_profile_id
            and description.extra.get("native_segment_runtime_profile_sha256")
            == hashlib.sha256(step.runtime_profile_json.encode()).hexdigest()
        )
        execution = json.loads(group.execution_json)
        require(
            execution["native_segment_profile_id"] == query.execution_profile_id
            and execution["native_segment_module_variant"] == query.module_variant
            and tuple((r.prefix_len, r.extend_len) for r in native.requests)
            == tuple((r.target_prefix_len, r.extend_len) for r in group.requests)
            and _canonical(native.to_dict()["sampling_recipe"]["per_attention_dp"])
            == _canonical(
                [[json.loads(r.sampling_params_json) for r in group.requests]]
            )
        )
        if group.forward_mode == "TARGET_VERIFY":
            variant = json.loads(query.module_variant)
            require(
                (
                    variant["schema_version"] in (2, 3)
                    or variant["draft_history_tokens"]
                    == [r.bounded_draft_history_tokens(128) for r in group.requests]
                )
                and type(execution["draft_buffer_capacity"]) is int
                and variant["buffer_capacity"] == execution["draft_buffer_capacity"]
                and variant["buffer_device_matches"]
                is execution["draft_buffer_device_matches"]
            )
    except (KeyError, IndexError, ValueError, TypeError) as error:
        raise PredictorError(
            "data_unavailable", "SDK native region phase/source binding is unavailable"
        ) from error
    return True
