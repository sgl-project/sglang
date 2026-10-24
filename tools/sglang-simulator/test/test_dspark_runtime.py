"""Native-adapter protocol checks; fixture costs never qualify performance."""

from types import SimpleNamespace as NS

import pytest
from sglang_simulator.simulation.sglang import dspark_runtime as adapter
from sglang_simulator.simulation.sglang.dspark import (
    STEP_ATTRIBUTE,
    CommittedHistory,
    DSparkSession,
    DSparkTopology,
)
from sglang_simulator.time_predictor.base import PredictorError
from test_dspark_step import Cost, Progress


def batch(mode="EXTEND", *, prefix=0, extend=512, chunks=0):
    req = NS(
        rid="a",
        output_ids=[] if mode == "EXTEND" else [1],
        kv=NS(kv_committed_len=prefix),
        sampling_params=object(),
        grammar=None,
        inflight_middle_chunks=chunks,
    )
    return NS(
        device="cpu",
        forward_mode=NS(name=mode),
        reqs=[req],
        prefix_lens=[prefix],
        extend_lens=[extend],
        seq_lens_cpu=[prefix + extend if mode == "EXTEND" else prefix],
        seq_lens=[prefix + extend if mode == "EXTEND" else prefix],
    )


class Results:
    def build(self, *, batch, prediction):
        counts = dict(prediction.progress.committed_tokens)
        result = NS(
            new_seq_lens=[
                r.target_prefix_len
                + (
                    r.extend_len
                    if prediction.local_group.forward_mode == "EXTEND"
                    else counts[r.rid]
                )
                for r in prediction.local_group.requests
            ]
        )
        setattr(result, STEP_ATTRIBUTE, prediction)
        return result


@pytest.fixture
def runtime(monkeypatch):
    # Only recipe capture is replaced in pure protocol tests. The production
    # path reads all fields from the native Struct via the SDK-owned inventory.
    monkeypatch.setattr(
        adapter,
        "sampling_snapshot",
        lambda params: '{"is_normalized":true,"max_new_tokens":100}',
    )
    return adapter.DSparkNativeRuntime(
        session=DSparkSession(Cost(), Progress()),
        topology=DSparkTopology(2, 1),
        runtime_profile_json='{"source":"CPU fixture"}',
        execution_resolver=lambda **kwargs: {"path": "CPU fixture"},
        output_token_id=1,
        result_factory=Results(),
    )


def forward(runtime, native_batch, publish=None):
    return runtime.forward_batch_generation(
        worker=NS(query_token_num=5, verify_num_draft_tokens=6),
        batch=native_batch,
        on_publish=publish,
    )


def test_prefill_then_native_committed_decode_updates_next_causal_work(runtime):
    first = forward(runtime, batch())
    assert first.new_seq_lens == [512]
    assert runtime.session.history_for("a") == CommittedHistory(512, 0, 512)
    second = forward(runtime, batch("DECODE", prefix=512))
    assert second.new_seq_lens == [514]
    third_before = runtime.capture(batch("DECODE", prefix=514))
    assert third_before.requests[0].target_prefix_len == 514
    assert third_before.requests[0].draft_valid_to == 514


def test_cached_target_prefix_does_not_invent_draft_prefix(runtime):
    prediction = getattr(
        forward(runtime, batch(prefix=1024, extend=64)), STEP_ATTRIBUTE
    )
    assert prediction.step.requests[0].draft_history_tokens == 0
    assert runtime.session.history_for("a") == CommittedHistory(1088, 1024, 1088)


def test_intermediate_chunk_injects_kv_without_emitting_or_sampling_acceptance(runtime):
    result = forward(runtime, batch(chunks=1))
    assert getattr(result, STEP_ATTRIBUTE).progress.committed_tokens == (("a", 0),)
    assert runtime.session.progress_model.calls == 0
    assert runtime.session.history_for("a").target_prefix_len == 512


def test_decode_needs_actual_prefill_and_matching_native_host_mirror(runtime):
    with pytest.raises(PredictorError, match="prior injection"):
        forward(runtime, batch("DECODE", prefix=512))
    forward(runtime, batch())
    wrong = batch("DECODE", prefix=512)
    wrong.reqs[0].kv.kv_committed_len = 900
    with pytest.raises(PredictorError, match="host mirrors"):
        forward(runtime, wrong)


def test_publication_failure_leaves_causal_history_and_forward_id_unchanged(runtime):
    def fail(value):
        raise RuntimeError("publication failed")

    with pytest.raises(RuntimeError, match="publication failed"):
        forward(runtime, batch(), fail)
    with pytest.raises(PredictorError, match="prior injection"):
        runtime.session.history_for("a")
    result = forward(runtime, batch())
    assert getattr(result, STEP_ATTRIBUTE).step.forward_id == 0


def test_native_result_failure_never_commits_history(runtime):
    def fail(**kwargs):
        raise RuntimeError("result failed")

    runtime.results.build = fail
    with pytest.raises(RuntimeError, match="result failed"):
        forward(runtime, batch())
    with pytest.raises(PredictorError):
        runtime.session.history_for("a")


def test_missing_cost_never_samples_progress_or_publishes(runtime):
    def fail(step):
        raise PredictorError("data_unavailable", "missing real cost")

    runtime.session.cost_model.predict_step = fail
    calls = []
    with pytest.raises(PredictorError, match="missing real cost"):
        forward(runtime, batch(), calls.append)
    assert calls == [] and runtime.session.progress_model.calls == 0


def test_completion_and_native_cache_reset_forget_the_same_session(runtime):
    forward(runtime, batch())
    runtime.note_request_finished(rid="a", natural_stop=False)
    with pytest.raises(PredictorError):
        runtime.session.history_for("a")
    forward(runtime, batch())
    runtime.clear_cache_pool()
    with pytest.raises(PredictorError):
        runtime.session.history_for("a")


def test_adp2_needs_actual_peer_scheduling_not_a_partition_of_global_batch(runtime):
    runtime.topology = DSparkTopology(2, 2)
    with pytest.raises(PredictorError, match="actual peer scheduling"):
        forward(runtime, batch())


def test_umd_cost_missing_phases_fail_before_preparation():
    from sglang_simulator.time_predictor.dspark_umd import UMDDSparkCostModel
    from test_dspark_step import step

    model = UMDDSparkCostModel(None, proposal_profile_id="a" * 64)
    with pytest.raises(PredictorError) as error:
        model.predict_step(step())
    assert error.value.code == "data_unavailable"
    assert error.value.details["missing_worker_phases"] == [
        "worker_preamble",
        "target_verify",
        "accept_commit_publication",
        "worker_tail",
    ]


def test_two_decode_commits_keep_source_window128_through_real_sdk_projection():
    from dataclasses import replace

    sdk = pytest.importorskip("infercast.sdk.native_sampling")
    from sglang_simulator.time_predictor.dspark_umd import UMDDSparkCostModel
    from test_dspark_step import step

    recipe = dict.fromkeys(sdk.SAMPLING_FIELDS)
    recipe.update(
        is_normalized=True, temperature=1.0, top_k=1073741824, max_new_tokens=20
    )
    before = step()
    request = replace(
        before.requests[0],
        draft_valid_from=384,
        sampling_params_json=adapter.canonical(recipe),
    )
    execution = {
        "draft_capture_key": {
            "size": 1,
            "attention_variant": None,
            "stream_idx": None,
            "variant_label": None,
        },
        "draft_buffer_capacity": 1,
        "draft_buffer_device_matches": True,
        "draft_graph_eligible": True,
    }
    before = replace(
        before,
        groups=(
            replace(
                before.groups[0],
                requests=(request,),
                execution_json=adapter.canonical(execution),
            ),
        ),
    )
    session = DSparkSession(Cost(), Progress())
    session.record_history("a", CommittedHistory(512, 384, 512))
    bridge = UMDDSparkCostModel(None, proposal_profile_id="a" * 64)
    for index in range(2):
        description, _, _ = bridge._prepare_proposal(step=before, phase="proposal")
        assert description.extra["before_proposal_work"]["draft_history_tokens"] == [
            128
        ]
        session.execute(before)
        history = session.history_for("a")
        assert history.draft_valid_to - history.draft_valid_from == 130 + index * 2
        request = replace(
            request,
            target_prefix_len=history.target_prefix_len,
            draft_valid_to=history.draft_valid_to,
            output_tokens_before=3 + index * 2,
        )
        before = replace(
            before,
            forward_id=index + 1,
            groups=(replace(before.groups[0], requests=(request,)),),
        )
    short = replace(request, draft_valid_from=request.draft_valid_to - 127)
    short_step = replace(before, groups=(replace(before.groups[0], requests=(short,)),))
    with pytest.raises(PredictorError, match="causal cost class"):
        bridge._prepare_proposal(step=short_step, phase="proposal")
    with pytest.raises(PredictorError, match="trailing KV gap"):
        replace(
            request, draft_valid_to=request.target_prefix_len - 1
        ).bounded_draft_history_tokens(128)


def test_sampling_capture_checks_all_sdk_fields_at_adapter_entry():
    sdk = pytest.importorskip("infercast.sdk.native_sampling")
    params = NS(**dict.fromkeys(sdk.SAMPLING_FIELDS))
    params.is_normalized = True
    params.max_new_tokens = 20
    recipe = adapter.sampling_snapshot(params)
    import json

    assert set(json.loads(recipe)) == sdk.SAMPLING_FIELDS
    delattr(params, "top_p")
    with pytest.raises(PredictorError, match="incomplete"):
        adapter.sampling_snapshot(params)


def test_dflash_hook_preserves_unmarked_native_inputs(monkeypatch):
    from sglang_simulator.simulation.sglang.dspark_worker import C_DSparkDraftInputHook

    class DraftInput:
        def prepare_for_decode(self, batch):
            return "original"

    C_DSparkDraftInputHook.hook(DraftInput)
    obj = DraftInput()
    assert obj.prepare_for_decode(object()) == "original"
    setattr(obj, adapter.INPUT_ATTRIBUTE, 6)
    monkeypatch.setattr(adapter, "prepare_cpu_decode", lambda obj, batch: "CPU DSPark")
    assert obj.prepare_for_decode(object()) == "CPU DSPark"


def test_proposal_buffer_capacity_is_causal_run_state_not_resolver_output(runtime):
    runtime.execution_resolver = lambda **kwargs: {"draft_buffer_capacity": 9000}
    forward(runtime, batch())
    before = runtime.capture(batch("DECODE", prefix=512))
    import json

    assert json.loads(before.groups[0].execution_json)["draft_buffer_capacity"] == 0
    forward(runtime, batch("DECODE", prefix=512))
    before = runtime.capture(batch("DECODE", prefix=514))
    assert json.loads(before.groups[0].execution_json)["draft_buffer_capacity"] == 1
    runtime.clear_cache_pool()
    assert runtime._draft_buffer_capacity == 1


def test_umd_draft_capture_comes_from_verified_profile_metadata(monkeypatch):
    import json
    from dataclasses import replace

    pytest.importorskip("infercast.sdk")
    from infercast.sdk.perf_database import slice_manifest
    from sglang_simulator.time_predictor.dspark_umd import UMDDSparkCostModel
    from test_dspark_step import step

    key = {
        "size": 1,
        "attention_variant": None,
        "stream_idx": None,
        "variant_label": None,
    }
    ref = {"path": "_meta/profile.json", "sha256": "a" * 64}
    identities = {
        "identities": [
            {
                "op": "dsv41_dspark_proposal_module",
                "config_token": "a" * 64,
                "extra": {"execution_profile": ref},
            }
        ]
    }
    profile = {
        "graph_definition": {
            "tokens_per_request": 5,
            "ragged_verify_mode": False,
            "capture_keys": [key],
        }
    }
    reads = []

    def read(root, path, **kwargs):
        reads.append((path, kwargs))
        return json.dumps(
            identities if path.endswith("kernel_identity.json") else profile
        ).encode()

    monkeypatch.setattr(slice_manifest, "read_verified_slice_artifact", read)
    model = UMDDSparkCostModel(
        NS(slice_ref=NS(slice_dir="CPU fixture")), proposal_profile_id="a" * 64
    )
    group = step().groups[0]
    assert model.execution_for(group=group, runtime_profile_json="unused") == {
        "draft_capture_key": key,
        "draft_graph_eligible": True,
    }
    assert reads[1][1] == {"expected_sha256": "a" * 64}
    larger = replace(
        group, requests=(group.requests[0], replace(group.requests[0], rid="b"))
    )
    with pytest.raises(PredictorError, match="no matching source profile"):
        model.execution_for(group=larger, runtime_profile_json="unused")
