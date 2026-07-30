from array import array
from types import SimpleNamespace

from sglang.srt.disaggregation.rust_pd import RustPdFatalError
from sglang.srt.disaggregation.utils import DisaggregationMode
from sglang.srt.environ import envs
from sglang.srt.managers.io_struct import TokenizedGenerateReqInput
from sglang.srt.managers.rust_server import (
    install_parent_shutdown_handlers,
    install_scheduler_shutdown_handlers,
)
from sglang.srt.managers.scheduler import Scheduler
from sglang.srt.managers.scheduler_components.load_inquirer import (
    SchedulerLoadInquirer,
)
from sglang.srt.managers.scheduler_components.metrics_reporter import (
    SchedulerMetricsReporter,
)
from sglang.srt.managers.tokenizer_manager import (
    scheduler_shutdown_timeout_seconds,
)
from sglang.srt.sampling.sampling_params import SamplingParams


def test_rust_pd_branch_returns_before_all_legacy_owner_construction(monkeypatch):
    import sglang.srt.disaggregation.rust_pd as rust_pd
    import sglang.srt.managers.scheduler as scheduler_module

    calls = []
    sentinel = object()

    def fail_legacy(*_args, **_kwargs):
        raise AssertionError("legacy Python PD owner was constructed")

    for name in (
        "MetadataBuffers",
        "PrefillBootstrapQueue",
        "DecodePreallocQueue",
        "DecodeTransferQueue",
        "ReqToMetadataIdxAllocator",
    ):
        monkeypatch.setattr(scheduler_module, name, fail_legacy)

    monkeypatch.setattr(
        rust_pd.RustPdSchedulerAdapter,
        "from_scheduler",
        classmethod(lambda cls, scheduler: calls.append(scheduler) or sentinel),
    )
    scheduler = SimpleNamespace(
        server_args=SimpleNamespace(
            disaggregation_mode="prefill",
            disaggregation_transfer_backend="mooncake",
        )
    )

    with envs.SGLANG_RUST_SERVER.override(True):
        Scheduler.init_disaggregation(scheduler)

    assert calls == [scheduler]
    assert scheduler.rust_pd_adapter is sentinel
    assert scheduler.disagg_prefill_bootstrap_queue is None
    assert scheduler.disagg_decode_prealloc_queue is None
    assert scheduler.disagg_decode_transfer_queue is None
    assert scheduler.enable_staging is False


def test_rust_pd_prefill_does_not_commit_dummy_token_for_zero_limit(monkeypatch):
    import sglang.srt.disaggregation.prefill as prefill

    calls = []
    adapter = SimpleNamespace(
        sender_send_chunks=lambda requests, sizes: calls.append(("send", list(sizes))),
        add_prefill_inflight=lambda requests: calls.append(("inflight", len(requests))),
    )
    request = SimpleNamespace(
        origin_input_ids=[1, 2],
        output_ids=[],
        sampling_params=SimpleNamespace(max_new_tokens=0),
        inflight_middle_chunks=0,
        time_stats=SimpleNamespace(
            set_prefill_finished_time=lambda: None,
            set_prefill_transfer_queue_entry_time=lambda: None,
        ),
    )
    scheduler = SimpleNamespace(
        rust_pd_adapter=adapter,
        tree_cache=object(),
        metrics_reporter=SimpleNamespace(report_prefill_stats=lambda **_: None),
    )
    result = SimpleNamespace(
        copy_done=None,
        next_token_ids=SimpleNamespace(tolist=lambda: [123]),
        can_run_cuda_graph=False,
        dp_cooperation_info=None,
    )
    batch = SimpleNamespace(reqs=[request], prefill_stats=None, dp_cooperation_info=None)
    monkeypatch.setattr(prefill, "maybe_cache_unfinished_req", lambda *_: None)

    prefill.SchedulerDisaggregationPrefillMixin.process_batch_result_rust_pd_prefill(scheduler, batch, result)

    assert request.output_ids == []
    assert calls == [("send", [2 * 56 * 2_048]), ("inflight", 1)]


def test_rust_pd_prefill_clears_terminal_before_streaming_completion(monkeypatch):
    import sglang.srt.disaggregation.prefill as prefill

    events = []
    request = SimpleNamespace(
        return_logprob=False,
        finished_reason=None,
        time_stats=SimpleNamespace(
            set_prefill_kv_transfer_finish_time=lambda: None,
            set_completion_time=lambda: None,
        ),
    )
    scheduler = SimpleNamespace(
        rust_pd_adapter=SimpleNamespace(
            poll_inflight=lambda: ([request], []),
            clear_terminal=lambda requests: events.append(("clear", list(requests))),
        ),
        tree_cache=object(),
        output_streamer=SimpleNamespace(
            stream_output=lambda requests, return_logprob, _: events.append(("stream", list(requests), return_logprob))
        ),
    )
    monkeypatch.setattr(
        prefill,
        "release_kv_cache",
        lambda request, tree_cache: events.append(("release", request, tree_cache)),
    )

    assert prefill.SchedulerDisaggregationPrefillMixin.process_rust_pd_prefill_inflight(scheduler) == [request]
    assert events == [
        ("release", request, scheduler.tree_cache),
        ("clear", [request]),
        ("stream", [request], False),
    ]


def test_rust_pd_decode_zero_limit_prebuilt_needs_no_dummy_last_token(monkeypatch):
    import sglang.srt.disaggregation.decode_schedule_batch_mixin as decode_batch

    cached = []
    request = SimpleNamespace(
        is_prefill_only=True,
        output_ids=[],
        grammar=None,
    )
    batch = SimpleNamespace(
        reqs=[request],
        tree_cache=object(),
        input_ids=object(),
    )
    future_map = SimpleNamespace(
        stash=lambda *_: (_ for _ in ()).throw(AssertionError("zero-output request must not stash a dummy token"))
    )
    monkeypatch.setattr(
        decode_batch,
        "maybe_cache_unfinished_req",
        lambda req, cache: cached.append((req, cache)),
    )

    decode_batch.ScheduleBatchDisaggregationDecodeMixin.process_prebuilt(
        batch,
        SimpleNamespace(),
        future_map,
    )

    assert batch.input_ids is None
    assert request.output_ids == []
    assert cached == [(request, batch.tree_cache)]


def test_scheduler_shutdown_orders_transport_pool_and_distributed_release(monkeypatch):
    import sglang.srt.managers.scheduler as scheduler_module

    events = []
    monkeypatch.setattr(
        scheduler_module,
        "destroy_model_parallel",
        lambda: events.append("model_parallel"),
    )
    monkeypatch.setattr(
        scheduler_module,
        "destroy_distributed_environment",
        lambda: events.append("distributed"),
    )
    scheduler = SimpleNamespace(
        rust_pd_adapter=SimpleNamespace(
            shutdown=lambda: events.append("transport") or "SafeTerminal",
            safe_to_release_pools=True,
        ),
        release_host_resources=lambda: events.append("pool"),
    )

    Scheduler.shutdown_rust_pd_before_host_resources(scheduler)

    assert events == ["transport", "pool", "model_parallel", "distributed"]


def test_scheduler_unsafe_transport_shutdown_never_releases_host_pool():
    events = []

    def unsafe_shutdown():
        events.append("transport")
        raise RustPdFatalError("PD_LOCAL_FATAL shutdown=FatalUnsafe")

    scheduler = SimpleNamespace(
        rust_pd_adapter=SimpleNamespace(
            shutdown=unsafe_shutdown,
            safe_to_release_pools=False,
        ),
        release_host_resources=lambda: events.append("pool"),
    )

    try:
        Scheduler.shutdown_rust_pd_before_host_resources(scheduler)
    except RustPdFatalError:
        pass
    else:
        raise AssertionError("unsafe Rust PD shutdown was swallowed")

    assert events == ["transport"]


def test_rust_pd_scheduler_shutdown_uses_full_host_watchdog_budget():
    server_args = SimpleNamespace(
        disaggregation_mode="prefill",
        disaggregation_transfer_backend="mooncake",
    )

    with envs.SGLANG_RUST_SERVER.override(True):
        assert scheduler_shutdown_timeout_seconds(server_args) == 620


def test_non_rust_pd_scheduler_shutdown_keeps_existing_watchdog_budget():
    server_args = SimpleNamespace(
        disaggregation_mode="prefill",
        disaggregation_transfer_backend="mooncake",
    )

    with envs.SGLANG_RUST_SERVER.override(False):
        assert scheduler_shutdown_timeout_seconds(server_args) == 15


def test_rust_server_parent_sigterm_stops_watchdog_then_forwards_once(monkeypatch):
    handlers = {}
    events = []
    watchdog = SimpleNamespace(stop=lambda: events.append(("watchdog", None)))

    monkeypatch.setattr(
        "sglang.srt.managers.rust_server.signal.signal",
        lambda signum, handler: handlers.__setitem__(signum, handler),
    )
    monkeypatch.setattr(
        "sglang.srt.managers.rust_server.os.kill",
        lambda pid, signum: events.append((pid, signum)),
    )

    install_parent_shutdown_handlers([101, 102, 101], watchdog)
    handlers[15](15, None)

    assert events == [("watchdog", None), (101, 15), (102, 15)]


def test_rust_server_scheduler_sigterm_requests_graceful_exit(monkeypatch):
    handlers = {}
    scheduler = SimpleNamespace(gracefully_exit=False)

    monkeypatch.setattr(
        "sglang.srt.managers.rust_server.signal.signal",
        lambda signum, handler: handlers.__setitem__(signum, handler),
    )

    install_scheduler_shutdown_handlers(scheduler)
    handlers[15](15, None)

    assert scheduler.gracefully_exit is True


def test_peer_unready_race_returns_typed_terminal_without_killing_scheduler():
    streamed = []
    request = SimpleNamespace(return_logprob=False, finished_reason=None)
    scheduler = SimpleNamespace(
        _set_or_validate_priority=lambda _request: True,
        rust_pd_adapter=SimpleNamespace(
            enqueue=lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("PD_PEER_UNAVAILABLE"))
        ),
        output_streamer=SimpleNamespace(
            stream_output=lambda requests, return_logprob: streamed.append((list(requests), return_logprob))
        ),
    )

    Scheduler._add_request_to_queue(scheduler, request)

    assert request.finished_reason.pd_reason == "PD_PEER_UNAVAILABLE"
    assert request.finished_reason.status_code == 503
    assert streamed == [([request], False)]


def test_rust_pd_idle_check_never_reads_unconstructed_legacy_queues():
    scheduler = SimpleNamespace(
        running_batch=SimpleNamespace(is_empty=lambda: True),
        chunked_req=None,
        dllm_manager=SimpleNamespace(any_staging_reqs=lambda: False),
        last_batch=None,
        enable_overlap=False,
        _pp_microbatches_drained=lambda: True,
        waiting_queue=[],
        grammar_manager=SimpleNamespace(grammar_queue=[]),
        disaggregation_mode=DisaggregationMode.PREFILL,
        rust_pd_adapter=SimpleNamespace(
            active_count=0,
            pending_count=1,
            inflight_count=0,
        ),
        disagg_prefill_inflight_queue=None,
        disagg_prefill_bootstrap_queue=None,
        disagg_decode_prealloc_queue=None,
        disagg_decode_transfer_queue=None,
        enable_hisparse=False,
        enable_hierarchical_cache=False,
    )

    assert not Scheduler.is_fully_idle(scheduler)
    scheduler.rust_pd_adapter.pending_count = 0
    assert Scheduler.is_fully_idle(scheduler)

    scheduler.disaggregation_mode = DisaggregationMode.DECODE
    scheduler.rust_pd_adapter.inflight_count = 1
    assert not Scheduler.is_fully_idle(scheduler)
    scheduler.rust_pd_adapter.inflight_count = 0
    assert Scheduler.is_fully_idle(scheduler)


def test_rust_pd_observers_use_adapter_snapshots_without_legacy_queues():
    pending = SimpleNamespace(rid="pending", seqlen=8)
    inflight = SimpleNamespace(rid="inflight", seqlen=16)
    adapter = SimpleNamespace(
        pending_requests=(pending,),
        inflight_requests=(inflight,),
    )
    inquirer = SimpleNamespace(
        get_rust_pd_adapter=lambda: adapter,
        get_disagg_prefill_bootstrap_queue=lambda: (_ for _ in ()).throw(AssertionError("legacy prefill queue read")),
        get_disagg_prefill_inflight_queue=lambda: (_ for _ in ()).throw(AssertionError("legacy prefill inflight read")),
        get_disagg_decode_prealloc_queue=lambda: (_ for _ in ()).throw(AssertionError("legacy decode queue read")),
        get_disagg_decode_transfer_queue=lambda: (_ for _ in ()).throw(AssertionError("legacy decode transfer read")),
    )
    reporter = SimpleNamespace(
        scheduler=SimpleNamespace(
            rust_pd_adapter=adapter,
            disaggregation_mode=DisaggregationMode.PREFILL,
            disagg_prefill_bootstrap_queue=None,
            disagg_prefill_inflight_queue=None,
            disagg_decode_prealloc_queue=None,
            disagg_decode_transfer_queue=None,
        )
    )

    assert SchedulerLoadInquirer._get_disaggregation_queues(inquirer) == (
        (pending,),
        (inflight,),
        (),
    )
    assert SchedulerMetricsReporter._get_disaggregation_queues(reporter) == (
        (pending,),
        (inflight,),
        (),
    )


def test_invalid_disaggregated_request_without_time_stats_is_rejected():
    streamed = []
    scheduler = SimpleNamespace(
        server_args=SimpleNamespace(
            enable_session_radix_cache=False,
            disaggregation_bootstrap_port=8998,
        ),
        model_config=SimpleNamespace(
            hf_eos_token_id=[0],
            vocab_size=151_936,
        ),
        disaggregation_mode=DisaggregationMode.PREFILL,
        transfer_backend="mooncake",
        tokenizer=None,
        metrics_reporter=SimpleNamespace(enable_metrics=False),
        metrics_collector=None,
        dllm_config=None,
        output_streamer=SimpleNamespace(
            stream_output=lambda requests, return_logprob: streamed.append((list(requests), return_logprob))
        ),
    )
    request = TokenizedGenerateReqInput(
        rid="health-probe",
        input_text="health",
        input_ids=array("q", [1]),
        input_embeds=None,
        mm_inputs=None,
        token_type_ids=None,
        sampling_params=SamplingParams(max_new_tokens=1),
        return_logprob=False,
        logprob_start_len=0,
        top_logprobs_num=0,
        token_ids_logprob=None,
        stream=False,
        time_stats=None,
    )

    Scheduler.handle_generate_request(scheduler, request)

    assert len(streamed) == 1
    assert streamed[0][0][0].finished_reason.status_code == 400
