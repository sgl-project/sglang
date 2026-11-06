"""Frozen CPU source contracts and synthetic leaves, never timing admission."""

import copy
import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace as NS

import pytest
from sglang_simulator.simulation.sglang.dspark import (
    CommittedHistory,
    DSparkGroup,
    DSparkRequest,
    DSparkSession,
    DSparkStep,
    DSparkTopology,
)
from sglang_simulator.simulation.sglang.dspark_reference import ReferenceCPUView
from sglang_simulator.time_predictor.base import PredictorError
from sglang_simulator.time_predictor.simopus_native import SimOpusNativeCostAdapter


@pytest.fixture(params=[2, 4])
def public_adapter(request, monkeypatch):
    from simopus.sdk import DecodeExecutionProfile
    from simopus.sdk.models.umd import umd_static
    from simopus.sdk.models.umd.prediction_result import DatabaseProvenance
    from simopus.sdk.perf_database.common import DatabaseMode

    root = (
        Path(__file__).parent / "fixtures/simopus-native-source" / f"tp{request.param}"
    )
    fixture = json.loads((root / "fixture.json").read_text())
    database = NS(
        system="mi355x",
        system_spec=NS(gpu_sku="mi355x"),
        default_mode=DatabaseMode.SILICON,
        stack_meta=fixture["stack_meta"],
        cache_key=lambda: ("CPU-source-fixture", request.param),
    )
    monkeypatch.setattr(
        umd_static.DatabaseProvenance,
        "from_database",
        lambda db: DatabaseProvenance(
            "mi355x",
            "framework",
            "sglang",
            "0.5.22",
            "SILICON",
            None,
            "CPU-only",
            None,
            "0" * 64,
        ),
    )
    options = dict(
        execution_profile=DecodeExecutionProfile.from_dict(fixture["capture_profile"]),
        native_segment_profile_id=fixture["profile"]["sha256"],
        source_binding=dict(
            root=str(root.resolve()),
            **{k: fixture[k] for k in ("profile", "cohort", "stack")},
        ),
    )
    adapter = SimOpusNativeCostAdapter.from_sdk(
        database, "deepseek-ai/DeepSeek-V4.1-Flash", **options
    )
    return adapter, database, options


def step_for(adapter, mode="DECODE", batch=2):
    from simopus.sdk.native_sampling import SAMPLING_FIELDS

    recipe = dict.fromkeys(SAMPLING_FIELDS)
    recipe.update(
        is_normalized=True,
        max_new_tokens=20,
        ignore_eos=True,
        temperature=1.0,
        top_k=1073741824,
    )
    requests = tuple(
        DSparkRequest(
            str(i),
            8192 + i if mode == "DECODE" else 7680 + i,
            6 if mode == "DECODE" else 512,
            8064 if mode == "DECODE" else 7680 + i,
            8192 + i if mode == "DECODE" else 7680 + i,
            1 if mode == "DECODE" else 0,
            20,
            json.dumps(recipe, sort_keys=True, separators=(",", ":")),
        )
        for i in range(batch)
    )
    group = DSparkGroup(
        0,
        "TARGET_VERIFY" if mode == "DECODE" else "EXTEND",
        requests,
        '{"pending":true}',
    )
    native = NS(
        device="cpu",
        forward_mode=NS(name=mode),
        reqs=[
            NS(
                rid=r.rid,
                origin_input_ids=[0] * (8192 + i),
                sampling_params=NS(**recipe),
                to_finish=None,
                input_embeds=None,
                multimodal_inputs=None,
            )
            for i, r in enumerate(requests)
        ],
        seq_lens_cpu=None if mode == "DECODE" else [8192 + i for i in range(batch)],
        spec_info=NS(nxt_kv_lens_cpu=[8448] * batch) if mode == "DECODE" else None,
    )
    view = (
        ReferenceCPUView(None, tuple([8448] * batch))
        if mode == "DECODE"
        else ReferenceCPUView(tuple(8192 + i for i in range(batch)), None)
    )
    execution = adapter.execution_for(
        group=group,
        runtime_profile_json=adapter.assets.runtime_profile_json,
        native_batch=native,
        reference_view=view,
        buffer_capacity=8,
        buffer_device_matches=True,
    )
    group = replace(
        group,
        execution_json=json.dumps(execution, sort_keys=True, separators=(",", ":")),
    )
    return DSparkStep(
        0,
        DSparkTopology(adapter.native_source_policy.tp_size, 1),
        (group,),
        adapter.assets.runtime_profile_json,
    )


@pytest.fixture
def leaves(monkeypatch):
    from simopus.sdk.operations.compute.dsv41_native_segment import DSV41NativeSegment
    from simopus.sdk.perf_database.performance_result import PerformanceResult

    calls = []

    def query(op, database):
        calls.append(dict(op.identity))
        return op._record_query(
            PerformanceResult(1.25, source="silicon"),
            component="dsv41_native_segment_module",
            requested_key=op.identity,
            resolved_key=op.identity,
            kernel_source="sglang",
            kernel_impl=op.identity["kernel_impl"],
            resolution="exact",
        )

    monkeypatch.setattr(DSV41NativeSegment, "_query_measured", query)
    return calls


@pytest.mark.parametrize("mode", ["EXTEND", "DECODE"])
def test_public_nine_stages_keep_ordered_work_and_full_evidence(
    public_adapter, leaves, mode
):
    adapter, _, _ = public_adapter
    step = step_for(adapter, mode)
    result = adapter.predict_step(step)
    assert result.seconds == pytest.approx(len(step.required_phases) * 0.00125)
    assert len(leaves) == len(step.required_phases)
    for phase, record in zip(
        step.required_phases, adapter.last_evaluations, strict=True
    ):
        prediction = record["prediction"]
        assert (
            prediction["runtime_resolved"]["work_scope"]["complete_dspark_worker_phase"]
            == phase
        )
        assert (
            prediction["runtime_resolved"]["work_scope"]["token_progress_modeled"]
            is False
        )
        assert prediction["runtime_input"]["requests"] == [
            dict(prefix_len=r.target_prefix_len, extend_len=r.extend_len)
            for r in step.requests
        ]
        assert (
            prediction["nodes"][0]["queries"][0]["requested_key"]
            == prediction["nodes"][0]["queries"][0]["resolved_key"]
        )
        assert record["committed_event"] is False


def test_late_missing_cost_retains_evaluation_without_progress_or_commit(
    public_adapter, leaves, monkeypatch
):
    from simopus.sdk.perf_database.database import PerfDataNotAvailableError

    adapter, _, _ = public_adapter
    step = step_for(adapter)
    original = adapter.model.estimate_prepared_forward_result

    def estimate(**kwargs):
        if kwargs["runtime"]["native_region_stage"]["phase"] == "target_verify":
            raise PerfDataNotAvailableError("CPU missing-cost control")
        return original(**kwargs)

    monkeypatch.setattr(adapter.model, "estimate_prepared_forward_result", estimate)
    draws = []
    session = DSparkSession(
        adapter, NS(predict_progress=lambda before: draws.append(before))
    )
    for r in step.requests:
        session.record_history(
            r.rid,
            CommittedHistory(r.target_prefix_len, r.draft_valid_from, r.draft_valid_to),
        )
    history = copy.deepcopy(session._history)
    with pytest.raises(PredictorError) as error:
        session.execute(step)
    assert error.value.code == "data_unavailable"
    assert (
        draws == [] and session._history == history and session._last_forward_id == -1
    )
    assert len(adapter.last_evaluations) == 2


def test_public_source_policy_cannot_be_enabled_without_golden_originals(
    public_adapter,
):
    adapter, database, options = public_adapter
    from simopus.sdk.native_segment_module import SOURCE_WORK_POLICY

    with pytest.raises(PredictorError):
        SimOpusNativeCostAdapter.from_sdk(
            database,
            "deepseek-ai/DeepSeek-V4.1-Flash",
            **options,
            source_work_policy=SOURCE_WORK_POLICY,
        )
