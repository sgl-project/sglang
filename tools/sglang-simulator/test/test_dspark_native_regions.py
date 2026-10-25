"""CPU phase-binding guards, without timing rows or source qualification.

Attestation strings below are protocol fixtures, not a constructed/admitted
execution profile. No SDK prospective projector or factory activation is faked.
"""

import json
from dataclasses import replace

import pytest
from sglang_simulator.simulation.sglang.dspark_runtime import canonical
from sglang_simulator.time_predictor.base import PredictorError
from sglang_simulator.time_predictor.dspark_native_regions import (
    validate_native_region_phase,
)
from sglang_simulator.time_predictor.dspark_umd import UMDDSparkCostModel
from test_dspark_step import session, step

CASES = (
    ("EXTEND", "worker_preamble", "prefill_preamble"),
    ("EXTEND", "target_prefill", "prefill_target"),
    ("EXTEND", "draft_kv_inject", "prefill_draft_kv"),
    ("EXTEND", "worker_tail", "prefill_tail"),
    ("TARGET_VERIFY", "worker_preamble", "decode_preamble"),
    ("TARGET_VERIFY", "proposal", "decode_proposal"),
    ("TARGET_VERIFY", "target_verify", "decode_target"),
    ("TARGET_VERIFY", "accept_commit_publication", "decode_accept_commit"),
    ("TARGET_VERIFY", "worker_tail", "decode_tail"),
)


def prepared(mode, phase, region, *, batch_size=1, capture_size=None):
    sdk = pytest.importorskip("infercast.sdk.native_segment_module")
    if capture_size is not None and not hasattr(
        sdk.NativeSegmentQuery, "from_graph_work"
    ):
        pytest.skip("SDK does not expose graph-work queries")
    from infercast.sdk.models.umd.templates.deepseek_v41_segments import (
        prepare_native_dspark_phase,
    )
    from infercast.sdk.native_forward_input import NativeForwardInput
    from infercast.sdk.native_sampling import SAMPLING_FIELDS
    from infercast.sdk.native_sources import NATIVE_SOURCE_0522

    before = step()
    recipe = dict.fromkeys(SAMPLING_FIELDS)
    recipe.update(is_normalized=True, max_new_tokens=20, temperature=1.0)
    requests = tuple(
        replace(
            before.requests[0],
            rid=str(index),
            target_prefix_len=512 + index,
            draft_valid_to=512 + index,
            sampling_params_json=canonical(recipe),
        )
        for index in range(batch_size)
    )
    native = NativeForwardInput.from_dict(
        {
            "schema_version": 2,
            "source_pin": NATIVE_SOURCE_0522.name,
            "tp_size": 2,
            "attention_dp_size": 1,
            "sampling_recipe": {
                "source_sha256": NATIVE_SOURCE_0522.sampling_params,
                "per_attention_dp": [[recipe] * batch_size],
            },
            "groups": [
                {
                    "attention_dp_rank": 0,
                    "phase": "EXTEND" if mode == "EXTEND" else "DECODE",
                    "requests": [
                        {"prefix_len": r.target_prefix_len, "extend_len": r.extend_len}
                        for r in requests
                    ],
                    # Graph and planner bounds deliberately differ.
                    "target_seq_len_upper_bound": None if mode == "EXTEND" else 700,
                }
            ],
        }
    )
    capture = {
        "size": batch_size if capture_size is None else capture_size,
        "attention_variant": None,
        "stream_idx": None,
        "variant_label": None,
    }
    decode = mode == "TARGET_VERIFY"
    project = (
        sdk.NativeSegmentQuery.from_input
        if capture_size is None
        else sdk.NativeSegmentQuery.from_graph_work
    )
    query = project(
        execution_profile_id="a" * 64,
        segment_kind=region,
        native_forward_input=native,
        **(
            {"draft_history_tokens": [128] * batch_size if decode else None}
            if capture_size is None
            else {}
        ),
        buffer_capacity=0 if decode else None,
        buffer_device_matches=False if decode else None,
        draft_capture_key=capture if decode else None,
        target_capture_key={**capture, "attention_variant": "candidate_unfiltered"}
        if decode
        else None,
        verify_plan_cpu_max_seq_len=900 if decode else None,
    )
    execution = {
        "native_segment_profile_id": query.execution_profile_id,
        "native_segment_module_variant": query.module_variant,
        "draft_buffer_capacity": 0,
        "draft_buffer_device_matches": False,
    }
    before = replace(
        before,
        groups=(
            replace(
                before.groups[0],
                forward_mode=mode,
                requests=requests,
                execution_json=canonical(execution),
            ),
        ),
    )
    description, point, graph_phase = prepare_native_dspark_phase(
        query=query, phase=phase, runtime_profile_json=before.runtime_profile_json
    )
    return {
        "description": description,
        "point": point,
        "graph_phase": graph_phase,
        "phase": phase,
        "step": before,
    }


@pytest.mark.parametrize("mode,phase,region", CASES)
def test_all_nine_real_sdk_graphs_bind_only_the_matching_phase(mode, phase, region):
    binding = prepared(mode, phase, region)
    assert validate_native_region_phase(**binding) is True
    native = json.loads(binding["point"]["native_segment_queries"])[0]
    variant = json.loads(native["module_variant"])
    if mode == "TARGET_VERIFY":
        assert variant["verify_plan_cpu_max_seq_len"] == 900
        assert (
            variant["native_forward_input"]["groups"][0]["target_seq_len_upper_bound"]
            == 700
        )


@pytest.mark.parametrize("batch_size,capture_size", [(1, 2), (9, 10)])
@pytest.mark.parametrize("mode,phase,region", CASES[4:])
def test_sdk_graph_work_schema3_preserves_native_phase_binding(
    batch_size, capture_size, mode, phase, region
):
    binding = prepared(
        mode, phase, region, batch_size=batch_size, capture_size=capture_size
    )
    native = json.loads(binding["point"]["native_segment_queries"])[0]
    variant = json.loads(native["module_variant"])
    assert variant["schema_version"] == 3
    assert "draft_history_tokens" not in variant
    assert variant["draft_capture_key"]["size"] == capture_size
    assert len(binding["step"].requests) == batch_size
    assert validate_native_region_phase(**binding) is True


def test_graph_work_schema3_still_requires_the_before_step_buffer():
    binding = prepared(
        "TARGET_VERIFY", "target_verify", "decode_target", capture_size=2
    )
    before = binding["step"]
    group = before.groups[0]
    execution = json.loads(group.execution_json)
    execution["draft_buffer_capacity"] = 2
    binding["step"] = replace(
        before, groups=(replace(group, execution_json=canonical(execution)),)
    )
    with pytest.raises(PredictorError, match="phase/source binding"):
        validate_native_region_phase(**binding)


@pytest.mark.parametrize(
    "field",
    [
        "complete_dspark_worker_phase",
        "native_segment_profile_id",
        "native_segment_runtime_profile_sha256",
    ],
)
def test_native_proposal_requires_every_sdk_attestation(field):
    binding = prepared("TARGET_VERIFY", "proposal", "decode_proposal")
    description = binding["description"]
    binding["description"] = replace(
        description, extra={k: v for k, v in description.extra.items() if k != field}
    )
    model = UMDDSparkCostModel(
        None,
        phase_preparers={
            "proposal": lambda **_: (
                binding["description"],
                binding["point"],
                binding["graph_phase"],
            )
        },
    )
    with pytest.raises(PredictorError, match="phase/source binding"):
        model.query_phase("proposal", binding["step"])


@pytest.mark.parametrize(
    "change",
    ["runtime", "request_order", "sampling", "history", "buffer", "variant", "profile"],
)
def test_changed_current_step_cannot_reuse_another_prepared_cost(change):
    binding = prepared("TARGET_VERIFY", "target_verify", "decode_target", batch_size=2)
    before = binding["step"]
    group = before.groups[0]
    execution = json.loads(group.execution_json)
    if change == "runtime":
        before = replace(
            before, runtime_profile_json='{"source":"different CPU fixture"}'
        )
    elif change == "request_order":
        group = replace(group, requests=tuple(reversed(group.requests)))
    elif change in {"sampling", "history"}:
        request = group.requests[0]
        if change == "sampling":
            recipe = json.loads(request.sampling_params_json)
            recipe["temperature"] = 0.5
            request = replace(request, sampling_params_json=canonical(recipe))
        else:
            request = replace(request, draft_valid_from=request.target_prefix_len - 127)
        group = replace(group, requests=(request, *group.requests[1:]))
    else:
        execution[
            {
                "buffer": "draft_buffer_capacity",
                "variant": "native_segment_module_variant",
                "profile": "native_segment_profile_id",
            }[change]
        ] = 2 if change == "buffer" else "b" * 64
        group = replace(group, execution_json=canonical(execution))
    binding["step"] = replace(before, groups=(group,))
    with pytest.raises(PredictorError, match="phase/source binding"):
        validate_native_region_phase(**binding)


@pytest.mark.parametrize(
    "change", ["region", "multiplier", "composition", "graph_phase", "shape_type"]
)
def test_native_phase_rejects_wrong_region_or_additional_work(change):
    binding = prepared("TARGET_VERIFY", "target_verify", "decode_target")
    description = binding["description"]
    graph = description.op_graph
    node = graph.generation[0]
    if change == "region":
        binding["phase"] = "proposal"
    elif change == "multiplier":
        graph = replace(graph, generation=(replace(node, count=2),))
    elif change == "composition":
        graph = replace(
            graph, generation_composition=["sum", ["node", node.id], ["node", node.id]]
        )
    elif change == "shape_type":
        graph = replace(
            graph,
            generation=(
                replace(node, shape={**node.shape, "batch_size": ["lit", True]}),
            ),
        )
    else:
        binding["graph_phase"] = "context"
    binding["description"] = replace(description, op_graph=graph)
    with pytest.raises(PredictorError, match="phase/source binding"):
        validate_native_region_phase(**binding)


@pytest.mark.parametrize("value", [True, 1])
def test_sampling_identity_preserves_json_scalar_types(value):
    binding = prepared("TARGET_VERIFY", "target_verify", "decode_target")
    before = binding["step"]
    request = before.requests[0]
    recipe = json.loads(request.sampling_params_json)
    assert recipe["temperature"] == value  # Python equality collapses these types.
    recipe["temperature"] = value
    request = replace(request, sampling_params_json=canonical(recipe))
    binding["step"] = replace(
        before, groups=(replace(before.groups[0], requests=(request,)),)
    )
    with pytest.raises(PredictorError, match="phase/source binding"):
        validate_native_region_phase(**binding)


@pytest.mark.parametrize("mode,phase,region", CASES)
def test_real_umd_executor_preserves_missing_native_cost_gap(
    monkeypatch, mode, phase, region
):
    binding = prepared(mode, phase, region)
    from infercast.sdk.operations.compute.dsv41_native_segment import DSV41NativeSegment
    from infercast.sdk.perf_database.database import PerfDataNotAvailableError

    calls = []

    def unavailable(cls, database):
        calls.append(database)
        raise PerfDataNotAvailableError("CPU test: no qualified timing rows supplied")

    monkeypatch.setattr(DSV41NativeSegment, "load_data", classmethod(unavailable))
    model = UMDDSparkCostModel(
        None,
        phase_preparers={
            phase: lambda **_: (
                binding["description"],
                binding["point"],
                binding["graph_phase"],
            )
        },
    )
    with pytest.raises(PredictorError, match="missing qualified DSPark cost"):
        model.query_phase(phase, binding["step"])
    assert calls == [None]


def test_missing_source_binding_fails_before_progress_or_history_commit():
    binding = prepared("TARGET_VERIFY", "worker_preamble", "decode_preamble")
    description = binding["description"]
    description = replace(
        description,
        extra={**description.extra, "native_segment_runtime_profile_sha256": "b" * 64},
    )

    class Progress:
        def predict_progress(self, before):
            pytest.fail("source gap reached progress")

    model = UMDDSparkCostModel(
        None,
        phase_preparers={
            name: lambda **_: (description, binding["point"], "generation")
            for name in binding["step"].required_phases
        },
    )
    run = session(binding["step"], cost=model, progress=Progress())
    previous = tuple(run.history_for(r.rid) for r in binding["step"].requests)
    with pytest.raises(PredictorError, match="phase/source binding"):
        run.execute(binding["step"])
    assert tuple(run.history_for(r.rid) for r in binding["step"].requests) == previous


def test_old_proposal_cannot_silently_overwrite_another_cost_grain():
    with pytest.raises(PredictorError, match="conflicting cost grains"):
        UMDDSparkCostModel(
            None,
            proposal_profile_id="a" * 64,
            phase_preparers={"proposal": lambda **_: pytest.fail("overwritten")},
        )
