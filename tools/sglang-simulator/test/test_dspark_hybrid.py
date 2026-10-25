"""Hybrid provenance and output failures; no silicon data or latencies supplied."""

import hashlib
import io
import json
from contextlib import contextmanager, nullcontext
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


@pytest.fixture
def joint_case(tmp_path):
    """Real SDK/PerfDatabase, with declared source facts and zero timing rows."""
    pytest.importorskip("infercast.sdk.native_joint_forward")
    from pathlib import Path

    from infercast.sdk.decode_forward import DecodeExecutionProfile
    from infercast.sdk.extend_forward import ExtendExecutionProfile
    from infercast.sdk.native_dp_progress import NativeDPProgressPolicy
    from infercast.sdk.native_segment_analytic import NativeSegmentAnalyticModel
    from infercast.sdk.perf_database import PerfDatabase

    if not hasattr(NativeSegmentAnalyticModel, "estimate_joint"):
        pytest.skip("SDK has no joint analytic operation")
    raw = json.loads(
        Path(__file__).with_name("dspark_joint_declared.json").read_bytes()
    )
    facts = raw["source_policy_inputs"]
    policy = NativeDPProgressPolicy(**{k: tuple(v) for k, v in facts.items()})
    profiles = {
        "EXTEND": ExtendExecutionProfile.from_dict(raw["profiles"]["EXTEND"]),
        "DECODE": DecodeExecutionProfile.from_dict(raw["profiles"]["DECODE"]),
    }
    models = {
        mode: NativeSegmentAnalyticModel.from_dict(value)
        for mode, value in raw["analytic_models"].items()
    }

    def canonical(value):
        return json.dumps(value, sort_keys=True, separators=(",", ":"))

    systems = tmp_path / "systems"
    systems.mkdir()
    (systems / "mi355x.json").write_text(
        canonical({"data_dir": "data/mi355x", "gpu": {"name": "mi355x"}})
    )
    meta = systems / "data/mi355x/framework/sglang/0.5.22/_meta"
    meta.mkdir(parents=True)
    stack = canonical(
        {
            "schema_version": 1,
            "gpu_sku": "mi355x",
            "operator_libs": {},
            "inference": {
                "backend": "sglang",
                "version": "0.5.22",
                "runtime_version": "0.5.22",
            },
            "collection": {"profile": "framework"},
        }
    ).encode()
    (meta / "stack.json").write_bytes(stack)
    identities = canonical({"schema_version": 1, "identities": []}).encode()
    (meta / "kernel_identity.json").write_bytes(identities)
    (meta / "slice_manifest.json").write_text(
        canonical(
            {
                "schema_version": 1,
                "release_id": "structural-no-timings",
                "finalized_at": "not-a-measurement",
                "collector_version": "structural-fixture",
                "stack": {
                    "sha256": hashlib.sha256(stack).hexdigest(),
                    "oplib_fingerprint": "structural-only",
                },
                "plan": None,
                "collection_run": {},
                "tables": [],
                "coverage": {
                    "status": "not_checked",
                    "expected_count": None,
                    "actual_count": None,
                    "blocking_findings": None,
                },
                "artifacts": [
                    {
                        "path": "_meta/kernel_identity.json",
                        "sha256": hashlib.sha256(identities).hexdigest(),
                        "size_bytes": len(identities),
                    }
                ],
            }
        )
    )
    database = PerfDatabase.open_fidb(
        "mi355x", "sglang", "0.5.22", systems_root=str(systems), database_mode="HYBRID"
    )
    return SimpleNamespace(
        policy=policy,
        runtime=canonical(raw["runtime_profile"]),
        profiles=profiles,
        models=models,
        database=database,
        canonical=canonical,
    )


def joint_step(case, mode="DECODE", counts=(1, 2)):
    from infercast.sdk.native_forward_input import NativeForwardInput
    from infercast.sdk.native_graph_work import native_capture_key
    from infercast.sdk.native_joint_forward import NativeJointForwardQuery
    from infercast.sdk.native_sampling import SAMPLING_FIELDS
    from infercast.sdk.native_sources import NATIVE_SOURCE_0522
    from sglang_simulator.simulation.sglang.dspark import (
        DSparkGroup,
        DSparkRequest,
        DSparkStep,
        DSparkTopology,
    )

    recipe = dict.fromkeys(SAMPLING_FIELDS)
    recipe.update(
        is_normalized=True, max_new_tokens=20, ignore_eos=True, temperature=1.0
    )
    groups, native_groups, execution, sampling = [], [], [], []
    for rank, count in enumerate(counts):
        requests = tuple(
            DSparkRequest(
                f"dp{rank}-{i}",
                512 + rank * 4096 + i * 17,
                6 if mode == "DECODE" else 17 + i,
                0,
                512 + rank * 4096 + i * 17,
                1,
                20,
                case.canonical(recipe),
            )
            for i in range(count)
        )
        groups.append(
            DSparkGroup(
                rank,
                "IDLE"
                if not count
                else "TARGET_VERIFY"
                if mode == "DECODE"
                else "EXTEND",
                requests,
                '{"pending":true}',
            )
        )
        native_groups.append(
            {
                "attention_dp_rank": rank,
                "phase": "IDLE" if not count else mode,
                "requests": [
                    {"prefix_len": r.target_prefix_len, "extend_len": r.extend_len}
                    for r in requests
                ],
                "target_seq_len_upper_bound": None
                if mode == "EXTEND" or not count
                else 9000,
            }
        )
        sampling.append([recipe] * count)
        execution.append(
            {
                "attention_dp_rank": rank,
                "buffer_capacity": rank * 4 if mode == "DECODE" else None,
                "buffer_device_matches": bool(rank) if mode == "DECODE" else None,
                "draft_read_extent_tokens": [
                    min(128, r.target_prefix_len) for r in requests
                ]
                if mode == "DECODE"
                else None,
                "draft_capture_key": native_capture_key(
                    max(counts),
                    case.profiles["DECODE"].graph_capture_batches,
                    None,
                    disable_padding=False,
                )
                if mode == "DECODE"
                else None,
                "target_capture_key": native_capture_key(
                    max(counts),
                    case.profiles["DECODE"].graph_capture_batches,
                    "candidate_unfiltered" if count else "candidate_filtered",
                    disable_padding=False,
                )
                if mode == "DECODE"
                else None,
                "verify_plan_cpu_max_seq_len": None
                if mode == "EXTEND" or rank
                else 9216,
            }
        )
    query = NativeJointForwardQuery.from_input(
        NativeForwardInput.from_dict(
            {
                "schema_version": 1,
                "source_pin": NATIVE_SOURCE_0522.name,
                "sampling_recipe": {
                    "source_sha256": NATIVE_SOURCE_0522.sampling_params,
                    "per_attention_dp": sampling,
                },
                "groups": native_groups,
            }
        ),
        profile=case.profiles[mode],
        source_policy=case.policy,
        runtime_profile_json=case.runtime,
        group_execution=execution,
    )
    return DSparkStep(
        7,
        DSparkTopology(2, 2),
        tuple(
            replace(g, execution_json=case.canonical(query.to_runtime(case.policy)))
            for g in groups
        ),
        case.runtime,
    ), query


@pytest.mark.parametrize("mode", ["EXTEND", "DECODE"])
@pytest.mark.parametrize("counts", [(1, 2), (1, 0), (0, 1), (3, 4), (64, 64)])
def test_real_joint_umd_empty_database_returns_one_whole_worker(
    joint_case, tmp_path, mode, counts
):
    case = joint_case
    before, query = joint_step(case, mode, counts)
    path = tmp_path / "costs.jsonl"
    cost = UMDDSparkCostModel.for_native_joint(
        case.database,
        source_policy=case.policy,
        runtime_profile_json=case.runtime,
        profiles=case.profiles,
        analytic_models=case.models,
        cost_evidence_path=str(path),
    )
    timing = cost.predict_step(before)
    assert len(timing.phases) == 1 and timing.phases[0].phase == "whole_worker"
    assert (
        timing.seconds.hex()
        == (case.models[mode].estimate_joint(query)[0] / 1000).hex()
    )
    record = json.loads(path.read_bytes())
    assert record["phase"] == "whole_worker" and record["source"] == "sol"
    proof = record["identity"]["native_cost_query"]
    assert proof["requested_key"] == proof["resolved_key"] == query.to_shape()
    assert proof["component"] == "dsv41_native_forward_module"
    assert proof["row_metadata"]["calibration_anchor"] is None
    assert record["request_ids_by_attention_dp"] == [
        [r.rid for r in g.requests] for g in before.groups
    ]
    assert record["identity"]["runtime"] == query.to_runtime(case.policy)
    assert (
        timing.phases[0].evidence_ref
        == "umd:" + hashlib.sha256(path.read_bytes()).hexdigest()
    )
    assert cost.get_cost_metrics()["by_source"]["sol"]["queries"] == 1
    assert not list(path.parent.rglob("*_perf.*"))
    with pytest.raises(PredictorError, match="missing complete DSPark worker phase"):
        cost.query_phase("proposal", before)


@pytest.mark.parametrize("mode", ["EXTEND", "DECODE"])
@pytest.mark.parametrize("database_mode", ["SILICON", "HYBRID"])
def test_joint_without_an_analytic_model_remains_a_typed_data_gap(
    joint_case, mode, database_mode
):
    from infercast.sdk.perf_database.common import DatabaseMode
    from infercast.sdk.perf_database.database import PerfDataNotAvailableError

    case = joint_case
    before, _ = joint_step(case, mode)
    case.database.set_default_database_mode(DatabaseMode[database_mode])
    cost = UMDDSparkCostModel.for_native_joint(
        case.database,
        source_policy=case.policy,
        runtime_profile_json=case.runtime,
        profiles=case.profiles,
    )
    with pytest.raises(
        PredictorError, match="missing qualified DSPark joint worker"
    ) as error:
        cost.predict_step(before)
    assert isinstance(error.value.__cause__, PerfDataNotAvailableError)
    assert cost.get_cost_metrics()["by_source"] == {}


@pytest.mark.parametrize("invalid", ["dropped", "repeat", "shape", "source", "model"])
def test_joint_requires_the_complete_matching_real_sdk_recorder(
    joint_case, monkeypatch, invalid
):
    from infercast.sdk.diagnostics import prediction_evidence

    case = joint_case
    before, _ = joint_step(case)
    original = prediction_evidence.capture_prediction_evidence

    @contextmanager
    def damaged_recorder():
        with original() as recorder:
            yield recorder
        node = recorder.nodes[0]
        if invalid == "dropped":
            recorder.dropped_queries += 1
        elif invalid == "repeat":
            node["repeat"] = True
        elif invalid == "shape":
            node["shape"] = {}
        elif invalid == "source":
            node["queries"][0]["source"] = "silicon"
        else:
            node["queries"][0]["row_metadata"]["analytic_model_sha256"] = "0" * 64

    monkeypatch.setattr(
        prediction_evidence, "capture_prediction_evidence", damaged_recorder
    )
    cost = UMDDSparkCostModel.for_native_joint(
        case.database,
        source_policy=case.policy,
        runtime_profile_json=case.runtime,
        profiles=case.profiles,
        analytic_models=case.models,
    )
    with pytest.raises(PredictorError, match="one matching SDK leaf"):
        cost.predict_step(before)
    assert cost.get_cost_metrics()["by_source"] == {}


@pytest.mark.parametrize(
    "invalid", ["policy", "runtime", "profiles", "models", "mode", "ledger"]
)
def test_joint_constructor_keeps_explicit_typed_cost_context(joint_case, invalid):
    from infercast.sdk.perf_database.common import DatabaseMode

    case = joint_case
    args = {
        "source_policy": case.policy,
        "runtime_profile_json": case.runtime,
        "profiles": case.profiles,
        "analytic_models": case.models,
    }
    if invalid == "policy":
        args["source_policy"] = (
            SimpleNamespace(**case.policy.__dict__)
            if hasattr(case.policy, "__dict__")
            else object()
        )
    elif invalid == "runtime":
        args["runtime_profile_json"] = case.runtime + " "
    elif invalid == "profiles":
        args["profiles"] = dict(reversed(list(case.profiles.items())))
        args["profiles"]["EXTEND"] = case.profiles["DECODE"]
    elif invalid == "models":
        args["analytic_models"] = {
            "EXTEND": case.models["DECODE"],
            "DECODE": case.models["EXTEND"],
        }
    elif invalid == "mode":
        case.database.set_default_database_mode(DatabaseMode.SILICON)
    else:
        args.update(analytic_models=None, cost_evidence_path="unused.jsonl")
    with pytest.raises(PredictorError):
        UMDDSparkCostModel.for_native_joint(case.database, **args)


@pytest.mark.parametrize(
    "invalid",
    [
        "geometry",
        "order",
        "sampling",
        "peer",
        "source",
        "runtime",
        "topology",
        "mixed",
        "capture",
    ],
)
def test_joint_step_mismatch_fails_before_numeric_lookup(
    joint_case, monkeypatch, invalid
):
    from infercast.sdk.operations.compute.dsv41_native_forward import DSV41NativeForward
    from sglang_simulator.simulation.sglang.dspark import DSparkTopology

    case = joint_case
    before, _ = joint_step(case)
    groups = list(before.groups)
    if invalid in {"geometry", "order", "sampling"}:
        group = groups[1]
        requests = list(group.requests)
        if invalid == "order":
            requests.reverse()
        elif invalid == "geometry":
            requests[0] = replace(
                requests[0], target_prefix_len=requests[0].target_prefix_len + 1
            )
        else:
            recipe = json.loads(requests[0].sampling_params_json)
            recipe["temperature"] = (
                True  # Python equality must not collapse JSON types.
            )
            requests[0] = replace(
                requests[0], sampling_params_json=case.canonical(recipe)
            )
        groups[1] = replace(group, requests=tuple(requests))
    elif invalid in {"peer", "source", "capture"}:
        value = json.loads(groups[1].execution_json)
        if invalid == "peer":
            value["unexpected"] = True
        elif invalid == "source":
            value["native_joint_source_identity_sha256"] = "0" * 64
        else:
            query = json.loads(value["native_joint_forward_query"])
            query["group_execution"][1]["target_capture_key"]["size"] = 64
            value["native_joint_forward_query"] = case.canonical(query)
            groups[0] = replace(groups[0], execution_json=case.canonical(value))
        groups[1] = replace(groups[1], execution_json=case.canonical(value))
    elif invalid == "runtime":
        before = replace(before, runtime_profile_json='{"different":"runtime"}')
    elif invalid == "topology":
        before = replace(before, topology=DSparkTopology(2, 1), groups=(groups[0],))
        groups = list(before.groups)
    else:
        groups[1] = replace(groups[1], forward_mode="EXTEND")
    before = replace(before, groups=tuple(groups))
    cost = UMDDSparkCostModel.for_native_joint(
        case.database,
        source_policy=case.policy,
        runtime_profile_json=case.runtime,
        profiles=case.profiles,
        analytic_models=case.models,
    )
    monkeypatch.setattr(
        DSV41NativeForward,
        "query",
        lambda *args: pytest.fail("mismatched step reached numeric lookup"),
    )
    with pytest.raises(PredictorError, match="joint"):
        cost.predict_step(before)
    assert cost.get_cost_metrics()["by_source"] == {}


@pytest.mark.parametrize("invalid", ["source", "physical", "row", "proof"])
def test_joint_real_database_errors_do_not_become_numeric_misses(joint_case, invalid):
    from pathlib import Path

    case = joint_case
    if invalid in {"source", "physical"}:
        runtime = json.loads(case.runtime)
        if invalid == "source":
            next(iter(runtime["software"]["modules"].values()))["sha256"] = "0" * 64
        else:
            runtime["roles"][-1]["parallel"]["attn_tp_size"] = 2
        case.runtime = case.canonical(runtime)
    elif invalid == "row":
        Path(
            case.database.slice_ref.perf_file_path(
                "dsv41_native_forward_module_perf.csv"
            )
        ).write_text("op_name,latency_ms\nwrong-op,not-a-time\n")
    else:
        (
            Path(case.database.slice_ref.slice_dir) / "_meta/kernel_identity.json"
        ).write_bytes(b"tampered")
    before, _ = joint_step(case)
    cost = UMDDSparkCostModel.for_native_joint(
        case.database,
        source_policy=case.policy,
        runtime_profile_json=case.runtime,
        profiles=case.profiles,
        analytic_models=case.models,
    )
    with pytest.raises(ValueError):
        cost.predict_step(before)
    assert cost.get_cost_metrics()["by_source"] == {}


def test_joint_failed_cost_sink_precedes_session_progress_and_history(joint_case):
    from sglang_simulator.simulation.sglang.dspark import (
        CommittedHistory,
        DSparkSession,
    )

    case = joint_case
    before, _ = joint_step(case)
    cost = UMDDSparkCostModel.for_native_joint(
        case.database,
        source_policy=case.policy,
        runtime_profile_json=case.runtime,
        profiles=case.profiles,
        analytic_models=case.models,
    )

    class FailedSink:
        def write(self, raw):
            raise OSError("real joint record sink failure")

    class NoProgress:
        def predict_progress(self, step):
            pytest.fail("failed joint cost reached progress")

    cost._cost_sink = FailedSink()
    run = DSparkSession(cost, NoProgress())
    for request in before.requests:
        run.record_history(
            request.rid,
            CommittedHistory(
                request.target_prefix_len,
                request.draft_valid_from,
                request.draft_valid_to,
            ),
        )
    history = tuple(run.history_for(r.rid) for r in before.requests)
    with pytest.raises(PredictorError, match="evidence write failed"):
        run.predict(before)
    assert tuple(run.history_for(r.rid) for r in before.requests) == history
    assert run._last_forward_id == -1 and cost._cost_counts == {}
    with pytest.raises(PredictorError, match="evidence write failed"):
        cost.get_cost_metrics()
