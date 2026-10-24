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
        lambda params: '{"ignore_eos":true,"is_normalized":true,"max_new_tokens":100}',
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


@pytest.mark.parametrize(
    "override",
    [{"ignore_eos": value} for value in (False, None, 1)]
    + [
        {name: ["active stop"]}
        for name in ("stop", "stop_regex", "stop_strs", "stop_regex_strs")
    ]
    + [{"stop_token_ids": [1]}],
)
def test_synthetic_output_rejects_content_stops_before_cost_or_state(
    runtime, monkeypatch, override
):
    recipe = {"ignore_eos": True, "is_normalized": True, "max_new_tokens": 100}
    recipe.update(override)
    monkeypatch.setattr(
        adapter, "sampling_snapshot", lambda _: adapter.canonical(recipe)
    )
    published = []
    with pytest.raises(PredictorError, match="ignore_eos=True and no stop rules"):
        forward(runtime, batch(), published.append)
    assert runtime.session.cost_model.calls == 0
    assert runtime.session.progress_model.calls == 0
    assert runtime._forward_id == 0 and runtime._draft_buffer_capacity == 0
    assert published == []
    with pytest.raises(PredictorError, match="no prior injection/commit"):
        runtime.session.history_for("a")


def test_synthetic_output_allows_explicit_empty_normalized_stop_fields(
    runtime, monkeypatch
):
    recipe = {
        "ignore_eos": True,
        "is_normalized": True,
        "max_new_tokens": 100,
        "stop": None,
        "stop_regex": None,
        "stop_strs": [],
        "stop_regex_strs": [],
        "stop_token_ids": None,
    }
    monkeypatch.setattr(
        adapter, "sampling_snapshot", lambda _: adapter.canonical(recipe)
    )
    assert forward(runtime, batch()).new_seq_lens == [512]


def test_prefill_then_native_committed_decode_updates_next_causal_work(runtime):
    first = forward(runtime, batch())
    assert first.new_seq_lens == [512]
    assert runtime.session.history_for("a") == CommittedHistory(512, 0, 512)
    second = forward(runtime, batch("DECODE", prefix=512))
    assert second.new_seq_lens == [514]
    third_before = runtime.capture(batch("DECODE", prefix=514))
    assert third_before.requests[0].target_prefix_len == 514
    assert third_before.requests[0].draft_valid_to == 514


def test_native_runtime_excludes_all_replacement_worker_wall_from_virtual_cpu_time(
    runtime, monkeypatch
):
    clock = [0.0]
    monkeypatch.setattr(adapter.time, "perf_counter", lambda: clock[0])

    def timed(function, seconds):
        def call(*args, **kwargs):
            result = function(*args, **kwargs)
            clock[0] += seconds
            return result

        return call

    runtime.capture = timed(runtime.capture, 2.0)
    runtime.session.cost_model.predict_step = timed(
        runtime.session.cost_model.predict_step, 3.0
    )
    runtime.results.build = timed(runtime.results.build, 5.0)
    runtime.session.commit = timed(runtime.session.commit, 11.0)

    def publish(_value):
        clock[0] += 7.0

    result = forward(runtime, batch(), publish)
    prediction = getattr(result, STEP_ATTRIBUTE)
    assert prediction.prediction_wall_seconds == 28.0
    assert prediction.timing.seconds == pytest.approx(0.04)
    assert runtime.session.history_for("a").target_prefix_len == 512


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


def test_resolver_receives_original_batch_and_before_state_before_cost(runtime):
    import json

    native_batch = batch()

    def resolve(
        *,
        group,
        runtime_profile_json,
        native_batch,
        buffer_capacity,
        buffer_device_matches,
    ):
        assert native_batch is original
        assert runtime_profile_json == runtime.runtime_profile_json
        assert group.requests[0].target_prefix_len == 0
        assert group.requests[0].draft_history_tokens == 0
        assert buffer_capacity == 0 and buffer_device_matches is False
        assert (
            runtime.session.cost_model.calls
            == runtime.session.progress_model.calls
            == 0
        )
        assert runtime._forward_id == 0
        return {"draft_buffer_capacity": 9000, "draft_buffer_device_matches": True}

    original = native_batch
    runtime.execution_resolver = resolve
    # This legacy batch deliberately lacks all optional source-fact attributes.
    result = forward(runtime, native_batch)
    execution = json.loads(
        getattr(result, STEP_ATTRIBUTE).step.groups[0].execution_json
    )
    assert execution["draft_buffer_capacity"] == 0
    assert execution["draft_buffer_device_matches"] is False


def test_resolver_buffer_growth_and_failed_cost_keep_transactional_state(runtime):
    import json

    seen = []

    def resolve(**kwargs):
        seen.append(kwargs)
        return {"draft_buffer_capacity": 9000, "draft_buffer_device_matches": False}

    runtime.execution_resolver = resolve
    forward(runtime, batch())
    second_prefill = batch()
    second_prefill.reqs[0].rid = "b"
    forward(runtime, second_prefill)
    forward(runtime, batch("DECODE", prefix=512))
    assert seen[-1]["buffer_capacity"] == 0
    assert seen[-1]["buffer_device_matches"] is False
    assert runtime._draft_buffer_capacity == 1

    larger = batch("DECODE", prefix=514)
    second = batch("DECODE", prefix=512).reqs[0]
    second.rid = "b"
    larger.reqs.append(second)
    larger.seq_lens_cpu = larger.seq_lens = [514, 512]
    before_history = tuple(runtime.session.history_for(rid) for rid in ("a", "b"))
    before_id = runtime._forward_id
    progress_calls = runtime.session.progress_model.calls
    predict = runtime.session.cost_model.predict_step

    def fail(_step):
        raise PredictorError("data_unavailable", "missing larger-batch cost")

    runtime.session.cost_model.predict_step = fail
    published = []
    with pytest.raises(PredictorError, match="missing larger-batch cost"):
        forward(runtime, larger, published.append)
    assert seen[-1]["native_batch"] is larger
    assert seen[-1]["buffer_capacity"] == 1
    assert seen[-1]["buffer_device_matches"] is True
    assert [req.target_prefix_len for req in seen[-1]["group"].requests] == [514, 512]
    assert (
        tuple(runtime.session.history_for(rid) for rid in ("a", "b")) == before_history
    )
    assert runtime._forward_id == before_id and runtime._draft_buffer_capacity == 1
    assert runtime.session.progress_model.calls == progress_calls and published == []

    runtime.session.cost_model.predict_step = predict
    result = forward(runtime, larger)
    execution = json.loads(
        getattr(result, STEP_ATTRIBUTE).step.groups[0].execution_json
    )
    assert execution["draft_buffer_capacity"] == 1
    assert execution["draft_buffer_device_matches"] is True
    assert runtime._draft_buffer_capacity == 2
    runtime.capture(batch("DECODE", prefix=516))
    assert seen[-1]["buffer_capacity"] == 2
    assert seen[-1]["buffer_device_matches"] is True


def source_fact_batch():
    from array import array

    native_batch = batch("DECODE", prefix=512)
    req = native_batch.reqs[0]
    req.origin_input_ids = array("q", [100, 101])
    req.sampling_params = NS(max_new_tokens=100)
    req.to_finish = req.input_embeds = req.multimodal_inputs = None
    native_batch.spec_info = NS(nxt_kv_lens_cpu=[768])
    return native_batch


def test_optional_simulator_source_facts_freeze_order_without_mutating_native_state():
    from dataclasses import FrozenInstanceError

    native_batch = source_fact_batch()
    second = source_fact_batch().reqs[0]
    second.rid = "b"
    second.origin_input_ids = [10, 11, 12]
    second.sampling_params.max_new_tokens = 200
    second.to_finish = object()
    second.input_embeds = object()
    second.multimodal_inputs = object()
    native_batch.reqs.append(second)
    native_batch.seq_lens_cpu = [512, 128]
    native_batch.spec_info.nxt_kv_lens_cpu = [768, 256]

    facts = adapter.capture_simulator_source_facts(native_batch)
    assert facts.observation_scope == "simulator_cpu_before_worker_v1"
    assert facts.forward_mode == "DECODE" and facts.draft_input_present is True
    assert [(r.rid, r.origin_input_len, r.max_new_tokens) for r in facts.requests] == [
        ("a", 2, 100),
        ("b", 3, 200),
    ]
    assert facts.requests[0].to_finish_present is False
    assert facts.requests[1].to_finish_present is True
    assert facts.requests[1].input_embeds_present is True
    assert facts.requests[1].multimodal_inputs_present is True
    assert facts.seq_lens_cpu == (512, 128) and facts.nxt_kv_lens_cpu == (768, 256)
    assert native_batch.reqs[0].origin_input_ids.tolist() == [100, 101]
    assert native_batch.seq_lens_cpu == [512, 128]
    assert native_batch.spec_info.nxt_kv_lens_cpu == [768, 256]

    native_batch.reqs[0].origin_input_ids.append(102)
    native_batch.reqs[1].sampling_params.max_new_tokens = 1
    native_batch.reqs.reverse()
    native_batch.seq_lens_cpu[0] = 900
    native_batch.spec_info.nxt_kv_lens_cpu[0] = 1024
    assert facts.requests[0].rid == "a" and facts.requests[0].origin_input_len == 2
    assert facts.requests[1].max_new_tokens == 200
    assert facts.seq_lens_cpu == (512, 128) and facts.nxt_kv_lens_cpu == (768, 256)
    with pytest.raises(FrozenInstanceError):
        facts.requests[0].origin_input_len = 9
    with pytest.raises(FrozenInstanceError):
        facts.seq_lens_cpu = (900, 128)


def test_optional_source_facts_preserve_known_absent_mirrors_and_draft_input():
    native_batch = source_fact_batch()
    native_batch.seq_lens_cpu = None
    native_batch.spec_info.nxt_kv_lens_cpu = None
    facts = adapter.capture_simulator_source_facts(native_batch)
    assert facts.seq_lens_cpu is None and facts.nxt_kv_lens_cpu is None
    assert facts.draft_input_present is True
    native_batch.spec_info = None
    facts = adapter.capture_simulator_source_facts(native_batch)
    assert facts.draft_input_present is False and facts.nxt_kv_lens_cpu is None


@pytest.mark.parametrize(
    "owner,name",
    [
        ("batch", "seq_lens_cpu"),
        ("batch", "spec_info"),
        ("draft_input", "nxt_kv_lens_cpu"),
        ("request", "origin_input_ids"),
        ("request", "to_finish"),
        ("request", "input_embeds"),
        ("request", "multimodal_inputs"),
        ("sampling", "max_new_tokens"),
    ],
)
def test_optional_source_facts_reject_unknown_fields(owner, name):
    native_batch = source_fact_batch()
    owners = {
        "batch": native_batch,
        "draft_input": native_batch.spec_info,
        "request": native_batch.reqs[0],
        "sampling": native_batch.reqs[0].sampling_params,
    }
    delattr(owners[owner], name)
    with pytest.raises(PredictorError, match="missing a native field"):
        adapter.capture_simulator_source_facts(native_batch)


@pytest.mark.parametrize("name", ["seq_lens_cpu", "nxt_kv_lens_cpu"])
def test_optional_source_facts_reject_non_cpu_mirrors_without_reading_them(name):
    class DeviceVector:
        device = NS(type="cuda")
        ndim = 1

        def __len__(self):
            raise AssertionError("non-CPU mirror was inspected")

        def __iter__(self):
            raise AssertionError("non-CPU mirror was read")

    native_batch = source_fact_batch()
    owner = native_batch if name == "seq_lens_cpu" else native_batch.spec_info
    setattr(owner, name, DeviceVector())
    with pytest.raises(PredictorError, match="CPU vector"):
        adapter.capture_simulator_source_facts(native_batch)


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
