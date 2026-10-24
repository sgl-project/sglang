"""Real SDK projection/UMD integration with CPU metadata and no timing rows.

These fixtures describe public source-policy schemas, not collected hardware
evidence. InferCast owns projection and cost lookup; the simulator owns history,
reference bookkeeping and phase dispatch. No native engine or GPU is imported.
"""

import copy
import hashlib
import importlib.util
import json
import sys
from dataclasses import replace
from pathlib import Path
from types import ModuleType
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
from sglang_simulator.simulation.sglang.dspark_reference import (
    DSparkReferenceLedger,
    ReferenceCPUView,
    ReferenceRequestKV,
)
from sglang_simulator.simulation.sglang.dspark_reference_adapter import (
    prepare_reference_step,
)
from sglang_simulator.simulation.sglang.dspark_runtime import canonical
from sglang_simulator.time_predictor.base import PredictorError
from sglang_simulator.time_predictor.dspark_native_regions import (
    validate_native_region_phase,
)
from sglang_simulator.time_predictor.dspark_umd import UMDDSparkCostModel
from test_dspark_reference import native_page_sizing

pytestmark = pytest.mark.usefixtures(native_page_sizing.__name__)

pytest.importorskip("infercast.sdk.native_segment_projection")

from infercast._serialization import canonical_json_bytes
from infercast.sdk.decode_forward import DecodeExecutionProfile
from infercast.sdk.native_sampling import SAMPLING_FIELDS
from infercast.sdk.native_segment_module import OP, NativeSegmentQuery
from infercast.sdk.native_segment_projection import NativeSegmentSourcePolicy
from infercast.sdk.native_segments import BOUNDARY, SOURCES
from infercast.sdk.native_sources import NATIVE_SOURCE_0522
from infercast.sdk.native_verify_plan import PLANNER_SOURCE
from infercast.sdk.perf_database import PerfDatabase
from infercast.sdk.perf_database.database import PerfDataNotAvailableError
from infercast.sdk.perf_database.dsv41_dspark import (
    MODEL_CONFIG_SHA256,
    NATIVE_RUNTIME_ENVIRONMENT,
)
from infercast.sdk.perf_database.dsv41_native_forward import (
    COMPONENT_REALIZATION,
    FULL_FORWARD_REALIZATION,
)
from infercast.sdk.perf_database.dsv41_native_runtime import (
    forward_environment,
    native_forward_runtime,
)
from infercast.sdk.perf_database.dsv41_native_segment import (
    DEFINITION,
    MODEL_REVISION,
    POOL_SOURCE,
    PROPOSAL_SOURCES,
    REDUCTION,
    RUNTIME,
    RUNTIME_IMAGE,
    TIMING_METHOD,
    build_native_segment_profile,
)

CASES = (
    ("EXTEND", "worker_preamble", "prefill_preamble"),
    ("EXTEND", "target_prefill", "prefill_target"),
    ("EXTEND", "draft_kv_inject", "prefill_draft_kv"),
    ("EXTEND", "worker_tail", "prefill_tail"),
    ("DECODE", "worker_preamble", "decode_preamble"),
    ("DECODE", "proposal", "decode_proposal"),
    ("DECODE", "target_verify", "decode_target"),
    ("DECODE", "accept_commit_publication", "decode_accept_commit"),
    ("DECODE", "worker_tail", "decode_tail"),
)


def recipe():
    result = dict.fromkeys(SAMPLING_FIELDS)
    result.update(
        is_normalized=True,
        max_new_tokens=2,
        ignore_eos=True,
        temperature=1.0,
        top_p=1.0,
        top_k=1073741824,
        stop_strs=[],
        stop_regex_strs=[],
    )
    return result


def retain(assets, kind, value):
    raw = canonical_json_bytes(value)
    digest = hashlib.sha256(raw).hexdigest()
    path = f"_meta/dsv41-native-segment-{kind}-{digest}.json"
    assets[path] = raw
    return {"path": path, "sha256": digest}


@pytest.fixture
def source_case(tmp_path):
    """Bounded TP2 public metadata schemas, with explicit CPU-only provenance."""
    runtime = native_forward_runtime(NATIVE_SOURCE_0522.name)
    environment = forward_environment(NATIVE_RUNTIME_ENVIRONMENT, runtime)
    environment.update(AITER_BF16_FP8_MOE_BOUND="0", GPU_MAX_HW_QUEUES="2")
    sections = {
        "kernel": {"monokernel_backend": "amd", "enable_nvidia_megakernel": False},
        "graph": {
            "cuda_graph_backend_prefill": "disabled",
            "disable_cuda_graph": False,
            "disable_decode_cuda_graph": False,
        },
        "features": {"enable_encoder_swa_bounded_replay": False},
    }
    settings = {
        "page_size": 256,
        "speculative_dspark_block_size": 5,
        "speculative_num_draft_tokens": 6,
        "speculative_num_steps": 1,
        "kv_cache_dtype": "fp8_e4m3",
        "disable_overlap_schedule": False,
    }
    modules = {
        "model_runner": {
            "sha256": runtime.model_runner,
            "git_head": runtime.git_head,
            "tracked_diff_sha256": runtime.tracked_diff,
        }
    }
    physical = {
        "image": RUNTIME_IMAGE,
        "checkpoint": {"model": RUNTIME["model_path"], "revision": MODEL_REVISION},
        "software": {
            "sglang": RUNTIME["runtime_version"],
            "torch": "CPU schema fixture",
            "torch_git": "CPU schema fixture",
            "HIP": "CPU schema fixture",
            "modules": modules,
        },
        "execution": {
            "decode_graph": {"backend": "full"},
            "sections": sections,
            "settings": settings,
            "environment": environment,
        },
        "roles": [],
    }
    captures, startup, proposals, windows = [], [], [], []
    for rank in range(2):
        part = {
            "tp_rank": rank,
            "tp_size": 2,
            "attn_tp_rank": rank,
            "attn_tp_size": 2,
            "attn_dp_rank": 0,
            "attn_dp_size": 1,
            "pp_rank": 0,
            "pp_size": 1,
        }
        parallel = dict(
            **part,
            moe_ep_size=1,
            dp_size=1,
            attn_cp_size=1,
            attn_cp_rank=0,
            moe_ep_rank=0,
            moe_tp_size=2,
            moe_tp_rank=rank,
        )
        pid = 100 + rank
        for draft in (False, True):
            width = 5 if draft else 6
            graph = {"ragged_verify_mode": False, "captured_req_width": width}
            physical["roles"].append(
                {
                    "role": "draft" if draft else "target",
                    "parallel": parallel,
                    "monokernel_installed": True,
                    "graph": graph,
                    "kv": {
                        "model_dtype": "torch.bfloat16",
                        "configured_KV_dtype": "torch.float8_e4m3fn",
                        "configured_KV_dtype_name": "fp8_e4m3",
                        "pool_dtype": "torch.float8_e4m3fn",
                        "pool_store_dtype": "torch.uint8",
                        "pool_class": "sglang.srt.mem_cache.deepseek_v4_memory_pool.DeepSeekV4TokenToKVPool",
                    },
                    "pool_source": {"module_is_hip": True, "sha256": POOL_SOURCE},
                }
            )
            captures.append(
                dict(
                    schema_version=1,
                    time_ns=2,
                    pid=pid,
                    is_draft_worker=draft,
                    topology=part,
                    runner_class="sglang.srt.model_executor.runner.decode_cuda_graph_runner.DecodeCudaGraphRunner",
                    runner_source={
                        "path": "/CPU-schema-only/decode_cuda_graph_runner.py",
                        "sha256": runtime.graph_runner,
                    },
                    capture_batches=list(range(1, 9)),
                    request_pool_size=8,
                    **graph,
                    graph_keys=[
                        {
                            "size": n,
                            "attention_variant": v,
                            "stream_idx": None,
                            "variant_label": None,
                        }
                        for n in range(1, 9)
                        for v in (
                            (None,)
                            if draft
                            else ("candidate_filtered", "candidate_unfiltered")
                        )
                    ],
                    scope="CPU schema fixture only; no measurements or qualification",
                )
            )
            args = dict(
                tp_size=2,
                ep_size=1,
                dp_size=1,
                attn_dp_size=1,
                pp_size=1,
                nnodes=1,
                attn_cp_size=1,
                speculative_algorithm="DSPARK",
                enable_decoder_swa_bounded_replay=True,
                disable_cuda_graph=False,
                cuda_graph_backend_prefill="disabled",
                enable_two_batch_overlap=False,
                enable_single_batch_overlap=False,
                model_path=f"/CPU-schema-only/{MODEL_REVISION}",
                cuda_graph_config={"decode": physical["execution"]["decode_graph"]},
                **settings,
            )
            startup.append(
                {
                    "rank": rank,
                    "pid": pid,
                    "time_ns": 1,
                    "is_draft_worker": draft,
                    "recorded_after_successful_init": True,
                    "sglang_version": RUNTIME["runtime_version"],
                    "device_name": "AMD Instinct MI355X",
                    "server_args": args,
                    "runtime_environment": environment,
                    "parallel": parallel,
                    "exec_runtime": sections,
                    "model_metadata_files": {
                        "config.json": {"sha256": MODEL_CONFIG_SHA256}
                    },
                    "model_runner_module": modules["model_runner"],
                    "loaded_realization_modules": modules,
                    "torch_version": "CPU schema fixture",
                    "torch_git_version": "CPU schema fixture",
                    "torch_hip": "CPU schema fixture",
                }
            )
        proposals.append(
            {
                "schema_version": 1,
                "pid": pid,
                "time_ns": 3,
                "topology": part,
                "native_source_pin": NATIVE_SOURCE_0522.name,
                "cost_import_supported": False,
                "allocated_runtime_artifact": f"allocated-runtime-pid-{pid}.json",
                "draft_capture_status": {
                    "is_draft_worker": True,
                    "kind": "full_inventory_recorded",
                },
                "facts": {
                    "resolved": {
                        "gamma": 5,
                        "query_token_num": 5,
                        "sample_from_anchor": True,
                        "greedy_draft": False,
                        "worker_greedy_draft": False,
                        "simulate_acc_len": 0.0,
                        "simulate_acc_greedy": False,
                    },
                    "effective_environment": {
                        "SGLANG_DSPARK_FOLDED_PROPOSAL": True,
                        "SGLANG_DSPARK_USE_GREEDY_DRAFT": False,
                        "SGLANG_SIMULATE_ACC_LEN": 0.0,
                        "SGLANG_SIMULATE_ACC_GREEDY": False,
                        "SGLANG_DSPARK_SIMULATE_FOLD": False,
                    },
                    "sampler_state": "present",
                    "sampler": {
                        "gamma": 5,
                        "query_token_num": 5,
                        "sample_from_anchor": True,
                        "greedy_draft": False,
                        "max_bs": 8,
                    },
                    "sources": {k: {"sha256": v} for k, v in PROPOSAL_SOURCES.items()},
                    "markov_head_state": "present",
                    "confidence_head_state": "none",
                    "draft_monokernel_registration_state": "present",
                    "draft_pool": {
                        "swa_valid_from_state": "none",
                        "swa_req_ring_size_state": "none",
                        "swa_req_ring_size": None,
                        "source": {"module_is_hip": True, "sha256": POOL_SOURCE},
                    },
                    "draft_decode_runner": {
                        "state": "present",
                        "class_name": captures[-1]["runner_class"],
                        "model_back_reference_matches": True,
                        "identity": {"source": {"sha256": runtime.graph_runner}},
                    },
                },
            }
        )
        windows.append(
            {
                "pid": pid,
                "metadata": [
                    {
                        "topology": part,
                        "pid": pid,
                        "time_ns": 4,
                        "mode": "DECODE",
                        "source_sha256": NATIVE_SOURCE_0522.worker,
                        "request_ids": ["a"],
                        "case": {"observer_control": "B_buffered", "stage": "profile"},
                        "cpu_batch": {
                            "source_sampling_before_worker": {
                                "schema_version": 1,
                                "source_sha256": NATIVE_SOURCE_0522.sampling_params,
                                "observed_ns": 4,
                                "scope": "CPU schema fixture only",
                                "requests": [
                                    {"rid": "a", "effective_sampling_params": recipe()}
                                ],
                            }
                        },
                        "actual_full_graph_replays": [
                            {
                                "is_draft_worker": False,
                                "shape_key": {
                                    "attention_variant": "candidate_unfiltered"
                                },
                            }
                        ],
                        "source_graph_selection_inputs": [
                            {
                                "observed_before_select_ns": 5,
                                "cpu_max_seq_len": None,
                                "candidate_max_seq_len_upper_bound": 8200,
                                "verify_extra_tokens": 6,
                                "graph_limits": [["candidate_unfiltered", 16384]],
                                "graph_source_sha256": NATIVE_SOURCE_0522.graph_variants,
                                "selected_variant": "candidate_unfiltered",
                            }
                        ],
                    }
                ],
            }
        )
    assets = {}
    desc = {
        name: tuple(retain(assets, name, row) for row in rows)
        for name, rows in {
            "captures": captures,
            "startup": startup,
            "windows": windows,
            "proposal": proposals,
        }.items()
    }
    stack = {
        "gpu_sku": "mi355x",
        "inference": {
            "backend": "sglang",
            "runtime_version": RUNTIME["runtime_version"],
        },
        "operator_libs": {"scope": "CPU metadata only; no performance rows"},
    }
    physical_ref, stack_ref = (
        retain(assets, "physical", physical),
        retain(assets, "stack", stack),
    )
    source_ref = retain(
        assets,
        "source",
        {
            "schema_version": 1,
            "physical_source": physical_ref,
            "implementation_sha256": {**SOURCES, "alloc_verify_window": PLANNER_SOURCE},
            "evidence": [*desc["captures"], *desc["startup"]],
        },
    )
    profile_ref = retain(
        assets,
        "profile",
        build_native_segment_profile(source=source_ref, stack=stack_ref),
    )
    capture_profile = DecodeExecutionProfile(
        decode_graph_backend="full",
        kv_page_size=256,
        runtime_compatibility_id="CPU-structure-only",
        runtime_compatibility_manifest_sha256=profile_ref["sha256"],
        full_forward_realization=FULL_FORWARD_REALIZATION,
        component_realization=COMPONENT_REALIZATION,
        graph_capture_batches=tuple(range(1, 9)),
        graph_batch_policy="pad_to_capture",
        fallback_policy="reject",
    )
    policy = NativeSegmentSourcePolicy(
        profile_ref,
        lambda path, digest: assets[path],
        stack_meta=stack,
        capture_profile=capture_profile,
        capture_descriptors=desc["captures"],
        startup_descriptors=desc["startup"],
        window_descriptors=desc["windows"],
        proposal_descriptors=desc["proposal"],
    )
    systems = tmp_path / "systems"
    systems.mkdir()
    (systems / "mi355x.json").write_text(
        json.dumps({"data_dir": "data/mi355x"}), encoding="utf-8"
    )
    slice_dir = systems / "data/mi355x/framework/sglang/0.5.22"
    (slice_dir / "_meta").mkdir(parents=True)
    (slice_dir / "_meta/stack.json").write_bytes(canonical_json_bytes(stack))
    database = PerfDatabase.open_fidb(
        "mi355x", "sglang", "0.5.22", systems_root=str(systems), database_mode="SILICON"
    )
    return NS(
        policy=policy,
        assets=assets,
        profile_ref=profile_ref,
        stack=stack,
        capture_profile=capture_profile,
        descriptors=desc,
        systems=systems,
        slice_dir=slice_dir,
        database=database,
    )


def before(mode, source_case, *, batch_size=1):
    requests = tuple(
        DSparkRequest(
            rid=str(i),
            target_prefix_len=7680 if mode == "EXTEND" else 8192,
            extend_len=512 if mode == "EXTEND" else 6,
            draft_valid_from=7680 if mode == "EXTEND" else 8064,
            draft_valid_to=7680 if mode == "EXTEND" else 8192,
            output_tokens_before=0 if mode == "EXTEND" else 1,
            max_new_tokens=2,
            sampling_params_json=canonical(recipe()),
        )
        for i in range(batch_size)
    )
    group = DSparkGroup(
        0,
        "EXTEND" if mode == "EXTEND" else "TARGET_VERIFY",
        requests,
        '{"selection":"pending"}',
    )
    step = DSparkStep(
        0, DSparkTopology(2, 1), (group,), source_case.policy.runtime_profile_json
    )
    native = NS(
        device="cpu",
        forward_mode=NS(name=mode),
        reqs=[
            NS(
                rid=r.rid,
                origin_input_ids=[0] * 8192,
                sampling_params=NS(**recipe()),
                to_finish=None,
                input_embeds=None,
                multimodal_inputs=None,
            )
            for i, r in enumerate(requests)
        ],
        seq_lens_cpu=[8192] * batch_size,
        spec_info=NS(nxt_kv_lens_cpu=[8448] * batch_size),
    )
    view = (
        ReferenceCPUView((8192,) * batch_size, None)
        if mode == "EXTEND"
        else ReferenceCPUView(None, (8448,) * batch_size)
    )
    return step, native, view


def resolve(model, source_case, mode, *, batch_size=1):
    step, native, view = before(mode, source_case, batch_size=batch_size)
    ledger = (
        DSparkReferenceLedger(
            requests=tuple(
                ReferenceRequestKV(
                    r.rid,
                    r.target_prefix_len,
                    8448,
                    r.target_prefix_len,
                    r.output_tokens_before,
                    r.max_new_tokens,
                    prefill_complete=True,
                )
                for r in step.requests
            )
        )
        if mode == "DECODE"
        else DSparkReferenceLedger()
    )
    transaction = prepare_reference_step(
        ledger,
        step,
        page_size=256,
        future_indices_ready=mode == "DECODE",
        needs_cpu_seq_lens=False if mode == "DECODE" else None,
        process_previous_before=False,
        execution_resolver=model.execution_for,
        native_batch=native,
        buffer_capacity=4,
        buffer_device_matches=True,
    )
    assert transaction.view == view
    assert transaction.base_ledger is ledger and ledger.prepared is None
    return transaction


@pytest.mark.parametrize("mode,phase,region", CASES)
def test_real_sdk_projector_prepares_each_region_and_ordinary_perfdb_has_no_cost(
    source_case, mode, phase, region
):
    model = UMDDSparkCostModel(
        source_case.database, native_source_policy=source_case.policy
    )
    step = resolve(model, source_case, mode).step
    description, point, graph_phase = model.preparers[phase](phase=phase, step=step)
    assert validate_native_region_phase(
        description=description,
        point=point,
        graph_phase=graph_phase,
        phase=phase,
        step=step,
    )
    query = NativeSegmentQuery.from_shape(
        json.loads(point["native_segment_queries"])[0]
    )
    assert query.segment_kind == region
    variant = json.loads(query.module_variant)
    if mode == "DECODE":
        assert (
            query.native_input.to_dict()["groups"][0]["target_seq_len_upper_bound"]
            == 8200
        )
        assert variant["verify_plan_cpu_max_seq_len"] == 8448
        assert variant["draft_history_tokens"] == [128]
        assert query.to_shape()["physical_rows"] == (
            5 if region == "decode_proposal" else 6
        )
    else:
        assert query.to_shape()["physical_rows"] == 512
        assert variant["verify_plan_cpu_max_seq_len"] is None
    with pytest.raises(
        PredictorError, match="missing qualified DSPark cost"
    ) as failure:
        model.query_phase(phase, step)
    assert isinstance(failure.value.__cause__, PerfDataNotAvailableError)
    assert "no finalized native region measurements" in str(failure.value.__cause__)
    assert not list(source_case.slice_dir.glob("*_perf.*"))


@pytest.mark.parametrize("mode", ["EXTEND", "DECODE"])
def test_full_phase_coverage_reaches_missing_perfdb_before_progress_or_history(
    source_case, mode
):
    model = UMDDSparkCostModel(
        source_case.database, native_source_policy=source_case.policy
    )
    transaction = resolve(model, source_case, mode, batch_size=2)
    step = transaction.step
    assert set(step.required_phases) <= set(model.preparers)

    class NoProgress:
        calls = 0

        def predict_progress(self, _step):
            self.calls += 1
            pytest.fail("missing PerfDB cost reached progress")

    progress = NoProgress()
    session = DSparkSession(model, progress)
    if mode == "DECODE":
        for r in step.requests:
            session.record_history(
                r.rid,
                CommittedHistory(
                    r.target_prefix_len, r.draft_valid_from, r.draft_valid_to
                ),
            )
    history = dict(session._history)
    last_forward_id = session._last_forward_id
    with pytest.raises(
        PredictorError, match="missing qualified DSPark cost for phase: worker_preamble"
    ):
        session.predict(step)
    assert progress.calls == 0 and session._history == history
    assert session._last_forward_id == last_forward_id
    assert (
        transaction.base_ledger.prepared is None and not transaction.base_ledger.pending
    )
    assert transaction.base_ledger.last_forward_id == -1


@pytest.mark.parametrize(
    "change", ["history", "original_input", "reference_view", "runtime"]
)
def test_missing_causal_inputs_do_not_become_default_native_facts(source_case, change):
    model = UMDDSparkCostModel(
        source_case.database, native_source_policy=source_case.policy
    )
    step, native, view = before("DECODE", source_case)
    if change == "history":
        r = step.requests[0]
        step = replace(
            step,
            groups=(
                replace(
                    step.groups[0],
                    requests=(replace(r, draft_valid_to=r.target_prefix_len - 1),),
                ),
            ),
        )
    elif change == "original_input":
        del native.reqs[0].origin_input_ids
    elif change == "reference_view":
        view = None
    else:
        step = replace(step, runtime_profile_json='{"different":"CPU fixture"}')
    with pytest.raises(PredictorError):
        model.execution_for(
            group=step.groups[0],
            runtime_profile_json=step.runtime_profile_json,
            native_batch=native,
            reference_view=view,
            buffer_capacity=4,
            buffer_device_matches=True,
        )


def stage_policy_metadata(case, *, failure=None):
    """Retain existing metadata formats, deliberately without a cost table."""
    assets = case.assets
    native = list(case.descriptors["windows"])
    if failure == "epoch":
        row = json.loads(assets[native[0]["path"]])
        row["pid"] += 1000
        for item in row["metadata"]:
            item["pid"] = row["pid"]
        native.append(retain(assets, "window-other-epoch", row))
    elif failure == "duplicate_native":
        native.append(native[0])
    qualification = retain(
        assets,
        "qualification",
        {
            "schema_version": 1,
            "execution_profile_id": case.policy.execution_profile_id,
            "timing_method": TIMING_METHOD,
            "status": "not_admitted",
            "decision": None,
        },
    )
    causal = retain(
        assets,
        "causal",
        {"schema_version": 1, "definition": DEFINITION, "inputs": [], "evidence": []},
    )
    measurement = retain(
        assets,
        "measurement",
        {
            "schema_version": True if failure == "schema" else 1,
            "execution_profile_id": case.policy.execution_profile_id,
            "timing_method": TIMING_METHOD,
            "reduction": REDUCTION,
            "qualification": qualification,
            "native": native,
            "proposal_paths": list(case.descriptors["proposal"]),
            "causal_inputs": causal,
            "cells": [],
        },
    )
    model = UMDDSparkCostModel(case.database, native_source_policy=case.policy)
    execution = json.loads(resolve(model, case, "EXTEND").step.groups[0].execution_json)
    query = NativeSegmentQuery(
        case.policy.execution_profile_id,
        "prefill_preamble",
        execution["native_segment_module_variant"],
    )
    identities = (
        []
        if failure == "profile"
        else [
            {
                "op": OP,
                "config_token": case.policy.execution_profile_id,
                "shape_key": canonical_json_bytes(query.to_shape()).decode(),
                "kernel_source": "sglang",
                "exec_mode": query.to_shape()["kernel_impl"],
                "kernel_name": BOUNDARY,
                "extra": {
                    "execution_profile": case.profile_ref,
                    "measurement_manifest": measurement,
                },
            }
        ]
    )
    assets["_meta/kernel_identity.json"] = canonical_json_bytes(
        {"schema_version": 1, "identities": identities}
    )
    stack_raw = canonical_json_bytes(case.stack)
    from infercast.schema import SLICE_MANIFEST_SCHEMA_VERSION

    manifest = {
        "schema_version": SLICE_MANIFEST_SCHEMA_VERSION,
        "release_id": "CPU-contract",
        "finalized_at": "2026-10-25T00:00:00Z",
        "collector_version": "CPU-contract",
        "stack": {
            "sha256": hashlib.sha256(stack_raw).hexdigest(),
            "oplib_fingerprint": "CPU-contract",
        },
        "plan": None,
        "coverage": {
            "status": "not_applicable",
            "expected_count": None,
            "actual_count": None,
            "blocking_findings": None,
        },
        "collection_run": {},
        "tables": [],
        "artifacts": [
            {
                "path": path,
                "sha256": hashlib.sha256(raw).hexdigest(),
                "size_bytes": len(raw),
            }
            for path, raw in assets.items()
        ],
    }
    for path, raw in assets.items():
        (case.slice_dir / path).write_bytes(raw)
    (case.slice_dir / "_meta/slice_manifest.json").write_bytes(
        canonical_json_bytes(manifest)
    )


@pytest.fixture
def real_factory(source_case, tmp_path, monkeypatch):
    """Keep InferCast real; mock only local model/config and progress boundaries."""
    import infercast
    from infercast.sdk.models import model_config, umd
    from sglang_simulator.simulation.types import SchedulerConfig
    from sglang_simulator.time_predictor import dspark_progress
    from sglang_simulator.time_predictor.infercast_dspark import (
        InferCastDSparkTimePredictor,
    )

    # A unit-test installed-build metadata seam, never a claim about dirty HEAD.
    monkeypatch.setattr(infercast, "__revision__", "b" * 40)
    authoritative = {
        "model_type": "deepseek_v41",
        "architectures": ["DeepseekV41ForCausalLM"],
        "dtype": "bfloat16",
        "bos_token_id": 0,
        "eos_token_id": 1,
        "pad_token_id": 2,
        "text_config": {
            "model_type": "deepseek_v41_text",
            "hidden_size": 16,
            "num_hidden_layers": 2,
            "compress_ratios": [0, 4, 0],
            "n_routed_experts": 4,
            "num_experts_per_tok": 2,
            "moe_intermediate_size": 8,
            "candidate_topk_blocks": 2,
            "candidate_block_size": 4,
            "rope_theta": 10000,
            "rope_scaling": {"rope_type": "yarn", "factor": 16},
        },
        "vision_config": {"num_hidden_layers": 2, "hidden_size": 12},
    }

    class CPUConfig:
        def __init__(self, **kwargs):
            vars(self).update(kwargs)

        def to_dict(self):
            return copy.deepcopy(vars(self))

    transformers = ModuleType("transformers")
    transformers.DeepseekV3Config = CPUConfig
    transformers.PretrainedConfig = CPUConfig
    monkeypatch.setitem(sys.modules, "transformers", transformers)

    def source_module(name, relative):
        path = Path(__file__).resolve().parents[3] / relative
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        monkeypatch.setitem(sys.modules, name, module)
        spec.loader.exec_module(module)
        return module

    adapter = source_module(
        "sglang.srt.configs.deepseek_v41", "python/sglang/srt/configs/deepseek_v41.py"
    )
    patches = source_module(
        "sglang.srt.utils.hf_transformers_patches",
        "python/sglang/srt/utils/hf_transformers_patches.py",
    )
    native_config = adapter.DeepseekV41Config(**copy.deepcopy(authoritative))
    patches.normalize_rope_scaling_compat(native_config)
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "config.json").write_bytes(canonical_json_bytes(authoritative))
    model = NS(
        model_path=str(checkpoint),
        hf_config=native_config.to_dict(),
        hidden_size=16,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=1,
        head_dim=8,
        vocab_size=32,
        qk_rope_head_dim=2,
        indexer_head_dim=4,
        window_size=8,
        compression_ratios=[0, 4, 0],
    )
    geometry = {
        "hidden_size": 16,
        "num_layers": 2,
        "num_attention_heads": 4,
        "num_key_value_heads": 1,
        "head_dim": 8,
        "vocab_size": 32,
        "qk_rope_head_dim": 2,
        "index_head_dim": 4,
        "sliding_window": 8,
        "compression_ratios_source": (0, 4, 0),
    }
    monkeypatch.setattr(
        umd,
        "build_model_description",
        lambda *args, **kwargs: NS(
            model=RUNTIME["model_path"], backend="sglang", geometry=geometry
        ),
    )
    monkeypatch.setattr(
        model_config,
        "load_model_config",
        lambda *args, **kwargs: copy.deepcopy(authoritative),
    )
    progress_calls, progress_reads = [], []

    class NoProgress:
        topology = (2, 1, 1)

        def __init__(self, path, **kwargs):
            progress_reads.append(path)
            self.model_ref = "sha256:" + kwargs["expected_sha256"]

        def predict_progress(self, step):
            progress_calls.append(step)
            pytest.fail("missing native PerfDB rows reached progress")

    monkeypatch.setattr(dspark_progress, "EmpiricalDSparkProgress", NoProgress)

    def asset(name, raw):
        path = tmp_path / name
        path.write_bytes(raw)
        return {"path": str(path), "sha256": hashlib.sha256(raw).hexdigest()}

    kwargs = {
        "model_id": RUNTIME["model_path"],
        "model_revision": MODEL_REVISION,
        "provider_revision": infercast.__revision__,
        "system": "mi355x",
        "systems_root": str(source_case.systems),
        "runtime_profile": asset(
            "runtime.json", source_case.policy.runtime_profile_json.encode()
        ),
        "progress_artifact": asset(
            "progress.json", b'{"scope":"CPU loader boundary only"}'
        ),
        "native_segment_profile_id": source_case.policy.execution_profile_id,
        "native_capture_profile": asset(
            "capture.json", canonical_json_bytes(source_case.capture_profile.to_dict())
        ),
    }
    config = SchedulerConfig(tp_size=2, backend_version="0.5.22")

    def build(**changes):
        return InferCastDSparkTimePredictor(
            model, NS(name="mi355x"), config, **{**kwargs, **changes}
        )

    return NS(
        build=build,
        asset=asset,
        progress_calls=progress_calls,
        progress_reads=progress_reads,
    )


@pytest.mark.parametrize("mode", ["EXTEND", "DECODE"])
def test_factory_loads_verified_existing_metadata_then_reaches_ordinary_missing_cost(
    source_case, real_factory, mode
):
    stage_policy_metadata(source_case)
    predictor = real_factory.build()
    runtime = predictor.dspark_runtime
    assert len(real_factory.progress_reads) == 1
    model = runtime.session.cost_model
    assert (
        model.native_source_policy.execution_profile_id
        == source_case.policy.execution_profile_id
    )
    transaction = resolve(model, source_case, mode)
    step = transaction.step
    for phase in step.required_phases:
        description, point, graph_phase = model.preparers[phase](phase=phase, step=step)
        assert validate_native_region_phase(
            description=description,
            point=point,
            graph_phase=graph_phase,
            phase=phase,
            step=step,
        )
    if mode == "DECODE":
        for r in step.requests:
            runtime.session.record_history(
                r.rid,
                CommittedHistory(
                    r.target_prefix_len, r.draft_valid_from, r.draft_valid_to
                ),
            )
    previous = dict(runtime.session._history)
    with pytest.raises(
        PredictorError, match="missing qualified DSPark cost"
    ) as failure:
        runtime.session.predict(step)
    assert isinstance(failure.value.__cause__, PerfDataNotAvailableError)
    assert not real_factory.progress_calls and runtime.session._history == previous
    assert runtime._forward_id == 0 and runtime._draft_buffer_capacity == 0


@pytest.mark.parametrize(
    "failure", ["runtime", "profile", "epoch", "duplicate_native", "schema"]
)
def test_factory_native_policy_rejects_unbound_runtime_profile_or_epoch(
    source_case, real_factory, failure
):
    stage_policy_metadata(
        source_case, failure=None if failure == "runtime" else failure
    )
    changes = {}
    if failure == "runtime":
        changes["runtime_profile"] = real_factory.asset(
            "changed-runtime.json", b'{"different":"CPU fixture"}'
        )
    expected = {
        "runtime": "configured runtime differs",
        "profile": "missing or ambiguous",
        "epoch": "measurement process epoch differs",
        "duplicate_native": "repeats original native",
        "schema": "measurement belongs to another profile",
    }
    with pytest.raises(PredictorError, match=expected[failure]):
        real_factory.build(**changes)
    assert not real_factory.progress_calls and not real_factory.progress_reads
