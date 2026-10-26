"""Typed joint assembly and real entry calls over a bounded CPU group double.

Declared source receipts, transport and native resource objects are structural
controls. They do not attest native communication, GPU costs or public support.
"""

import copy
import json
import sys
import threading
from pathlib import Path
from types import ModuleType
from types import SimpleNamespace as NS

import pytest
from sglang_simulator.simulation.sglang import dspark_runtime as adapter
from sglang_simulator.simulation.sglang.dspark import STEP_ATTRIBUTE
from sglang_simulator.time_predictor.base import PredictorError
from sglang_simulator.time_predictor.infercast_dspark import _native_joint_runtime
from test_dspark_hybrid import joint_case as joint_case
from test_dspark_joint_execution import case as _joint_geometry
from test_dspark_native_peer_transactions import (
    both,
    native_batch,
    passed,
    process,
)
from test_dspark_native_peer_transactions import owners as owners
from test_dspark_reference import native_page_sizing

_sampling_snapshot = adapter.sampling_snapshot
joint_geometry = _joint_geometry
pytestmark = pytest.mark.usefixtures(native_page_sizing.__name__)


def declared_mirror_policy():
    from infercast.sdk.native_dp_progress import NativeDPProgressPolicy
    from infercast.sdk.native_reference_policy import (
        MIRROR_ENV,
        MIRROR_FIELD,
        REFERENCE_MIRROR_SOURCES_0522,
    )
    from infercast.sdk.native_sources import NATIVE_SOURCE_0522

    facts = json.loads(
        Path(__file__).with_name("dspark_joint_declared.json").read_bytes()
    )["source_policy_inputs"]
    material = dict(
        schema_version=1,
        native_source_pin=NATIVE_SOURCE_0522.name,
        implementation_sha256={
            key: value[1] for key, value in REFERENCE_MIRROR_SOURCES_0522.items()
        },
        backend_selection="draft_property",
        hosts_draft=True,
        speculative_algorithm="DSPARK",
        enable_overlap=True,
        enable_two_batch_overlap=False,
        module_is_hip=True,
        confidence_budget_prepare_present=False,
        backends=[
            dict(
                role=role,
                speculative_algorithm="DSPARK",
                present=True,
                class_name=REFERENCE_MIRROR_SOURCES_0522["hip_backend"][0]
                + ".DeepseekV4HipRadixBackend",
                source_sha256=REFERENCE_MIRROR_SOURCES_0522["hip_backend"][1],
                attribute_present=True,
                value=False,
                effective=False,
            )
            for role in ("target", "draft")
        ],
        environment={MIRROR_ENV: dict(present=True, raw="0", effective=False)},
        needs_cpu_seq_lens=False,
    )
    for row in facts["startup"]:
        row["server_args"]["disable_overlap_schedule"] = False
    windows = []
    for path in facts["proposal"]:
        rank, pid, now = path["topology"]["tp_rank"], path["pid"], path["time_ns"]
        rids = [f"declared-{rank}"]
        windows.append(
            dict(
                pid=pid,
                metadata=[
                    dict(
                        topology=copy.deepcopy(path["topology"]),
                        pid=pid,
                        time_ns=now + 2,
                        forward_ct=7,
                        mode="DECODE",
                        request_ids=rids,
                        cpu_batch={
                            "request_ids": rids,
                            "scheduler_seq_lens_cpu": None,
                            MIRROR_FIELD: dict(
                                schema_version=1,
                                pid=pid,
                                tp_rank=rank,
                                forward_ct=7,
                                request_ids=rids,
                                observed_ns=now + 1,
                                future_indices_present=True,
                                material=copy.deepcopy(material),
                            ),
                        },
                    )
                ],
            )
        )
    return NativeDPProgressPolicy(
        **{key: tuple(value) for key, value in facts.items()}, windows=tuple(windows)
    )


@pytest.fixture
def context(joint_case, joint_geometry):
    from infercast.sdk.native_segment_analytic import NativeSegmentAnalyticModel

    geometry = joint_geometry
    policy = declared_mirror_policy()
    models = {
        mode: NativeSegmentAnalyticModel.from_dict(
            {
                **model.to_dict(),
                "execution_profile_id": geometry.profiles[mode].profile_id,
            }
        )
        for mode, model in joint_case.models.items()
    }
    return dict(
        database=joint_case.database,
        source_policy=policy,
        runtime_profile_json=geometry.runtime,
        profiles=geometry.profiles,
        analytic_models=models,
        model_config_sha256=models["DECODE"].model_config_sha256,
        model_seed=42,
    )


@pytest.fixture
def routed(owners, context, monkeypatch):
    # Only the transport/native containers are doubles. Factory, resolver,
    # Session, UMD, PerfDatabase and the typed numeric-miss path are real.
    local = threading.local()
    module = ModuleType("sglang.srt.runtime_context")
    module.get_parallel = lambda: local.parallel
    monkeypatch.setitem(sys.modules, module.__name__, module)
    monkeypatch.setattr(adapter, "sampling_snapshot", _sampling_snapshot)
    for run in owners:
        results = run.runtime.results
        run.runtime = _native_joint_runtime(**context)
        run.runtime.results = results
        run.driver = run.runtime.reference_driver
        run.scheduler.forward_ct = 0
        run.scheduler.waiting_queue = []
        run.parallel = NS(
            tp_group=run.group,
            tp_rank=run.rank,
            attn_dp_rank=run.rank,
            tp_size=2,
            attn_dp_size=2,
            attn_tp_size=1,
            attn_tp_rank=0,
            attn_cp_size=1,
            pp_size=1,
            nnodes=1,
            attn_dp_enabled=True,
        )

    def call(fn):
        def invoke(run):
            local.parallel = run.parallel
            return fn(run)

        return both(owners, invoke)

    return owners, call


def batches_for(runs, counts, mode="EXTEND", prior=None):
    from infercast.sdk.native_sampling import SAMPLING_FIELDS

    recipe = dict.fromkeys(SAMPLING_FIELDS)
    recipe.update(is_normalized=True, max_new_tokens=100, ignore_eos=True)
    batches = []
    for run, count in zip(runs, counts, strict=True):
        current = native_batch(
            run,
            "IDLE" if not count else mode,
            size=count or 1,
            requests=None if prior is None else prior[run.rank].reqs,
        )
        for req in current.reqs:
            req.sampling_params = NS(**recipe)
            req.origin_input_ids = [1] * (512 + current.reqs.index(req) * 7)
            req.to_finish = req.input_embeds = req.multimodal_inputs = None
        batches.append(current)
    return batches


@pytest.mark.parametrize("counts", [(1, 2), (0, 2), (2, 0)])
def test_selected_then_worker_entry_uses_one_joint_umd_and_local_results(
    routed, counts
):
    runs, call = routed
    # The real overlap loop binds its scheduler before the first selection.
    # Attention TP rank is singleton zero on BOTH schedulers, but DP1 owns its
    # actual local group1 rather than the new ledger's unbound default group0.
    for run in runs:
        run.driver.bind(run.scheduler)
        assert run.parallel.attn_tp_rank == 0
    passed(call(lambda run: run.driver.selected(run.scheduler, None)))
    assert [r.driver.adapter.ledger.local_attention_dp_rank for r in runs] == [0, 1]
    first = batches_for(runs, counts)
    for mode, batches in (
        ("EXTEND", first),
        ("DECODE", None),
    ):
        if batches is None:
            batches = batches_for(runs, counts, mode, first)
        passed(call(lambda run: run.driver.selected(run.scheduler, batches[run.rank])))
        for run in runs:
            run.driver.overlap_decision(batches[run.rank], False)
            run.scheduler.forward_ct += 1
        results = passed(
            call(
                lambda run: run.runtime.forward_batch_generation(
                    worker=NS(query_token_num=5, verify_num_draft_tokens=6),
                    batch=batches[run.rank],
                )
            )
        )
        assert [len(result.new_seq_lens) for result in results] == list(counts)
        assert [
            getattr(result, STEP_ATTRIBUTE).local_attention_dp_rank
            for result in results
        ] == [0, 1]
        assert runs[0].runtime.session._history == runs[1].runtime.session._history
        metrics = [run.runtime.session.cost_model.get_cost_metrics() for run in runs]
        assert metrics[0]["by_source"]["sol"]["queries"] == (
            1 if mode == "EXTEND" else 2
        )
        assert metrics[1]["by_source"] == {}
        for run in runs:
            process(run, batches[run.rank], results[run.rank])
    assert [run.runtime._draft_buffer_capacity for run in runs] == list(counts)


def test_paired_reset_before_first_selection_never_invents_a_native_fid(routed):
    runs, call = routed
    passed(call(lambda run: run.runtime.clear_cache_pool()))
    assert all(
        run.runtime._forward_id == 0 and run.runtime._native_dp_scheduler is None
        for run in runs
    )


@pytest.mark.parametrize(
    "bad",
    [
        "counter",
        "bool-counter",
        "rank",
        "topology",
        "bool-topology",
        "source",
        "progress",
    ],
)
def test_one_bad_local_route_rendezvous_before_cost_or_progress(routed, bad):
    runs, call = routed
    if bad == "counter":
        runs[1].scheduler.forward_ct = -1
    elif bad == "bool-counter":
        runs[1].scheduler.forward_ct = False
    elif bad == "rank":
        runs[1].parallel.attn_dp_rank = 0
    elif bad == "topology":
        runs[1].parallel.attn_tp_size = 2
    elif bad == "bool-topology":
        runs[1].parallel.pp_size = True
    elif bad == "progress":
        runs[1].runtime.session.progress_model.model_ref = "changed"
    else:
        runs[1].runtime._native_dp_identity = ("changed",)
    errors = call(lambda run: run.driver.selected(run.scheduler, None))
    assert all(isinstance(error, PredictorError) for error in errors)
    assert all(
        run.runtime._forward_id == 0 and run.runtime.session._history == {}
        for run in runs
    )


def test_factory_missing_mirror_and_wrong_model_fail_before_sink_creation(
    context, tmp_path
):
    from infercast.sdk.native_dp_progress import NativeDPProgressPolicy

    facts = json.loads(
        Path(__file__).with_name("dspark_joint_declared.json").read_bytes()
    )["source_policy_inputs"]
    missing = NativeDPProgressPolicy(
        **{key: tuple(value) for key, value in facts.items()}
    )
    sink = tmp_path / "must-not-exist.jsonl"
    for changes, match in (
        ({"source_policy": missing}, "mirror observations"),
        ({"model_config_sha256": "0" * 64}, "canonical model"),
        ({"output_token_id": True}, "output_token_id"),
    ):
        with pytest.raises(PredictorError, match=match):
            _native_joint_runtime(
                **{**context, **changes}, cost_evidence_path=str(sink)
            )
        assert not sink.exists()


def test_joint_cost_never_invokes_the_adp1_projector(context):
    runtime = _native_joint_runtime(**context)
    with pytest.raises(PredictorError, match="complete peer execution resolver"):
        runtime.session.cost_model.execution_for(
            group=None, runtime_profile_json=context["runtime_profile_json"]
        )
