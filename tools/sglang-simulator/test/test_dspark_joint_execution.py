"""Prospective CPU protocol controls; no observed timings or native execution."""

import copy
import hashlib
import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace as NS

import pytest
from sglang_simulator.simulation.sglang.dspark import (
    DSparkGroup,
    DSparkRequest,
    DSparkStep,
    DSparkTopology,
)
from sglang_simulator.simulation.sglang.dspark_joint_execution import (
    NativeJointExecutionResolver,
)
from sglang_simulator.simulation.sglang.dspark_reference import ReferenceCPUView
from sglang_simulator.simulation.sglang.dspark_runtime import canonical
from sglang_simulator.time_predictor.base import PredictorError


@pytest.fixture
def case():
    pytest.importorskip("infercast.sdk.native_joint_forward")
    from infercast.sdk.native_dp_progress import NativeDPProgressPolicy
    from infercast.sdk.native_graph_work import PADDING_SOURCES_0522
    from infercast.sdk.perf_database.dsv41_native_forward import parse_forward_profile

    # Reuse the existing declared source contract, not fixture latency values.
    document = json.loads(
        Path(__file__).with_name("dspark_joint_declared.json").read_bytes()
    )
    physical = document["runtime_profile"]
    for role in physical["roles"]:
        role["graph"]["padding_policy"]["sources"] = {
            name: {
                "path": "/declared/" + name.rsplit(".", 1)[-1] + ".py",
                "sha256": digest,
            }
            for name, digest in PADDING_SOURCES_0522.items()
        }
    runtime = canonical(physical)
    digest = hashlib.sha256(runtime.encode()).hexdigest()
    profiles = {}
    for mode, value in document["profiles"].items():
        old = parse_forward_profile(value)
        value = old.identity_dict()
        value["runtime_compatibility_id"] = value[
            "runtime_compatibility_manifest_sha256"
        ] = digest
        profiles[mode] = type(old)(**value)
    policy = NativeDPProgressPolicy(
        **{k: tuple(v) for k, v in document["source_policy_inputs"].items()}
    )
    resolver = NativeJointExecutionResolver(
        source_policy=policy, runtime_profile_json=runtime, profiles=profiles
    )
    return NS(
        resolver=resolver,
        physical=physical,
        policy=policy,
        runtime=runtime,
        profiles=profiles,
    )


def prepare(case, counts=(1, 0), mode="DECODE", *, mirror=False):
    from infercast.sdk.native_sampling import SAMPLING_FIELDS

    recipe = dict.fromkeys(SAMPLING_FIELDS)
    recipe.update(is_normalized=True, max_new_tokens=14, ignore_eos=True)
    groups, views, batches, executions = [], [], [], []
    for rank, count in enumerate(counts):
        requests = tuple(
            DSparkRequest(
                f"dp{rank}-{i}",
                478 + rank * 20000 + i * 13,
                6 if mode == "DECODE" else 17 + i,
                0,
                478 + rank * 20000 + i * 13,
                1,
                14,
                canonical(recipe),
            )
            for i in range(count)
        )
        capacity = rank + 1
        before = canonical(
            {
                "selection": "native_peer_before_work_unresolved",
                "draft_buffer_capacity": capacity,
                "draft_buffer_device_matches": bool(rank),
            }
        )
        group = DSparkGroup(
            rank,
            "IDLE" if not count else "TARGET_VERIFY" if mode == "DECODE" else "EXTEND",
            requests,
            before,
        )
        seq = tuple(
            r.target_prefix_len + (r.extend_len if mode == "EXTEND" else 0)
            for r in requests
        )
        view = (
            ReferenceCPUView((), None, rank, True)
            if not count
            else ReferenceCPUView(
                seq if mode == "EXTEND" or mirror else None,
                None
                if mode == "EXTEND"
                else tuple(32768 + i * 256 for i in range(count)),
                rank,
            )
        )
        native = NS(
            device="cpu",
            forward_mode=NS(name="IDLE" if not count else mode),
            reqs=[
                NS(
                    rid=r.rid,
                    origin_input_ids=tuple(range(r.target_prefix_len)),
                    sampling_params=NS(**recipe),
                    to_finish=None,
                    input_embeds=None,
                    multimodal_inputs=None,
                )
                for r in requests
            ],
            seq_lens_cpu=list(seq),
            spec_info=None,
        )
        execution = case.resolver.execution_for(
            group=group,
            runtime_profile_json=case.runtime,
            native_batch=native,
            reference_view=view,
            buffer_capacity=capacity,
            buffer_device_matches=bool(rank),
        )
        groups.append(group)
        views.append(view)
        batches.append(native)
        executions.append(execution)
    return NS(
        step=DSparkStep(13, DSparkTopology(2, 2), tuple(groups), case.runtime),
        reference_views=tuple(views),
        local_executions=tuple(executions),
        batches=batches,
    )


def resolve(case, data):
    return case.resolver(
        step=data.step,
        reference_views=data.reference_views,
        local_executions=data.local_executions,
    )


def test_profile_cannot_omit_native_intermediate_capture_bucket(case):
    # B2 would otherwise round to the still-materialized B4 graph. Checking
    # membership of only the selected key cannot prove the native bucket choice.
    profile = replace(
        case.profiles["DECODE"],
        graph_capture_batches=tuple(
            size for size in case.profiles["DECODE"].graph_capture_batches if size != 2
        ),
    )
    with pytest.raises(PredictorError, match="complete capture inventory"):
        NativeJointExecutionResolver(
            source_policy=case.policy,
            runtime_profile_json=case.runtime,
            profiles={**case.profiles, "DECODE": profile},
        )


def query(case, resolved):
    from infercast.sdk.native_joint_forward import NativeJointForwardQuery

    rows = [json.loads(g.execution_json) for g in resolved.groups]
    assert rows[0] == rows[1]
    mode = (
        "EXTEND"
        if any(g.forward_mode == "EXTEND" for g in resolved.groups)
        else "DECODE"
    )
    return NativeJointForwardQuery(
        case.profiles[mode].profile_id, rows[0]["native_joint_forward_query"]
    )


@pytest.mark.parametrize("mode", ["EXTEND", "DECODE"])
@pytest.mark.parametrize("counts", [(1, 0), (0, 1), (1, 2), (3, 6), (64, 64)])
def test_prospective_ordered_idle_ragged_and_full_batch(case, mode, counts):
    data = prepare(case, counts, mode)
    resolved = resolve(case, data)
    assert resolved.forward_id == data.step.forward_id
    assert all(
        replace(after, execution_json=before.execution_json) == before
        for before, after in zip(data.step.groups, resolved.groups, strict=True)
    )
    value = query(case, resolved).to_dict()
    groups, work = value["native_forward_input"]["groups"], value["group_execution"]
    assert [len(g["requests"]) for g in groups] == list(counts)
    assert [g["phase"] for g in groups] == ["IDLE" if not n else mode for n in counts]
    if mode == "EXTEND":
        assert all(
            all(v is None for k, v in w.items() if k != "attention_dp_rank")
            for w in work
        )
    else:
        assert [w["buffer_capacity"] for w in work] == [1, 2]
        assert [w["buffer_device_matches"] for w in work] == [False, True]
        for rank, count in enumerate(counts):
            assert work[rank]["verify_plan_cpu_max_seq_len"] == (
                32768 + (count - 1) * 256 if count else None
            )
            assert len(work[rank]["draft_read_extent_tokens"]) == count
            assert work[rank]["target_capture_key"]["attention_variant"] == (
                "candidate_unfiltered" if rank == 0 and count else "candidate_filtered"
            )
        assert (
            work[0]["draft_capture_key"]["size"] == work[1]["draft_capture_key"]["size"]
        )


def test_mirror_graph_bound_is_distinct_from_reserved_planner_input(case):
    data = prepare(case, (1, 1), mirror=True)
    value = query(case, resolve(case, data)).to_dict()
    assert [
        g["target_seq_len_upper_bound"] for g in value["native_forward_input"]["groups"]
    ] == [478, 20478]
    assert [w["verify_plan_cpu_max_seq_len"] for w in value["group_execution"]] == [
        478,
        20478,
    ]
    data.reference_views = tuple(
        replace(v, seq_lens_cpu=None) for v in data.reference_views
    )
    value = query(case, resolve(case, data)).to_dict()
    assert [
        g["target_seq_len_upper_bound"] for g in value["native_forward_input"]["groups"]
    ] == [498, 20498]
    assert [w["verify_plan_cpu_max_seq_len"] for w in value["group_execution"]] == [
        32768,
        32768,
    ]


def test_prospective_does_not_use_completed_raw_frames_or_create_timestamps(
    case, monkeypatch
):
    import infercast.sdk.native_joint_forward as owner

    def forbidden(*args, **kwargs):
        pytest.fail("prospective projection used a completed raw-frame converter")

    monkeypatch.setattr(owner, "native_joint_group_execution", forbidden)
    resolved = resolve(case, prepare(case))
    value = query(case, resolved).to_dict()
    assert "native_joint_input_completed_ns" not in canonical(value)


@pytest.mark.parametrize(
    "fault",
    [
        "rank",
        "source",
        "buffer",
        "bool_buffer",
        "request",
        "unknown",
        "view_rank",
        "runtime",
        "mixed",
    ],
)
def test_mismatch_rejects_without_mutating_frozen_step(case, fault):
    data = prepare(case, (1, 1))
    original = data.step
    local = copy.deepcopy(data.local_executions)
    if fault == "rank":
        local[0]["local_attention_dp_rank"] = 1
    elif fault == "source":
        local[0]["native_joint_source_identity_sha256"] = "0" * 64
    elif fault == "buffer":
        local[0]["draft_buffer_capacity"] = 99
    elif fault == "bool_buffer":
        local[0]["draft_buffer_capacity"] = True
    elif fault == "request":
        local[0]["native_joint_request_facts"][0]["rid"] = "other"
    elif fault == "unknown":
        local[0]["observed_ns"] = 1
    elif fault == "view_rank":
        data.reference_views = data.reference_views[::-1]
    elif fault == "runtime":
        data.step = replace(data.step, runtime_profile_json='{"other":true}')
    else:
        data.step = replace(
            data.step,
            groups=(
                replace(data.step.groups[0], forward_mode="EXTEND"),
                data.step.groups[1],
            ),
        )
    data.local_executions = local
    with pytest.raises(PredictorError):
        resolve(case, data)
    assert original.groups[0].requests[0].rid == "dp0-0"
    assert (
        json.loads(original.groups[0].execution_json)["selection"]
        == "native_peer_before_work_unresolved"
    )


@pytest.mark.parametrize("fault", ["sampling", "request_order", "idle_requests"])
def test_local_adapter_checks_actual_native_batch(case, fault):
    data = prepare(case, (2, 0))
    rank = 1 if fault == "idle_requests" else 0
    batch = data.batches[rank]
    if fault == "sampling":
        batch.reqs[0].sampling_params.max_new_tokens = 15
    elif fault == "request_order":
        batch.reqs.reverse()
    else:
        batch.reqs = data.batches[0].reqs
    with pytest.raises(PredictorError):
        case.resolver.execution_for(
            group=data.step.groups[rank],
            runtime_profile_json=case.runtime,
            native_batch=batch,
            reference_view=data.reference_views[rank],
            buffer_capacity=rank + 1,
            buffer_device_matches=bool(rank),
        )


def test_missing_selected_capture_never_skips_to_another_bucket(case):
    data = prepare(case, (3, 1))
    graph = case.resolver._graphs[0, "target"]
    graph["graph_keys"] = [k for k in graph["graph_keys"] if k["size"] != 4]
    with pytest.raises(PredictorError, match="selected graph"):
        resolve(case, data)
