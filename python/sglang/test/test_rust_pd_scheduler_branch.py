from types import SimpleNamespace

from sglang.srt.disaggregation.rust_pd import RustPdFatalError
from sglang.srt.environ import envs
from sglang.srt.managers.scheduler import Scheduler
from sglang.srt.managers.tokenizer_manager import (
    scheduler_shutdown_timeout_seconds,
)


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
    batch = SimpleNamespace(
        reqs=[request], prefill_stats=None, dp_cooperation_info=None
    )
    monkeypatch.setattr(prefill, "maybe_cache_unfinished_req", lambda *_: None)

    prefill.SchedulerDisaggregationPrefillMixin.process_batch_result_rust_pd_prefill(
        scheduler, batch, result
    )

    assert request.output_ids == []
    assert calls == [("send", [2 * 56 * 2_048]), ("inflight", 1)]


def test_scheduler_shutdown_orders_transport_before_host_pool_release():
    events = []
    scheduler = SimpleNamespace(
        rust_pd_adapter=SimpleNamespace(
            shutdown=lambda: events.append("transport") or "SafeTerminal",
            safe_to_release_pools=True,
        ),
        release_host_resources=lambda: events.append("pool"),
    )

    Scheduler.shutdown_rust_pd_before_host_resources(scheduler)

    assert events == ["transport", "pool"]


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


def test_peer_unready_race_returns_typed_terminal_without_killing_scheduler():
    streamed = []
    request = SimpleNamespace(return_logprob=False, finished_reason=None)
    scheduler = SimpleNamespace(
        _set_or_validate_priority=lambda _request: True,
        rust_pd_adapter=SimpleNamespace(
            enqueue=lambda *_args, **_kwargs: (_ for _ in ()).throw(
                RuntimeError("PD_PEER_UNAVAILABLE")
            )
        ),
        output_streamer=SimpleNamespace(
            stream_output=lambda requests, return_logprob: streamed.append(
                (list(requests), return_logprob)
            )
        ),
    )

    Scheduler._add_request_to_queue(scheduler, request)

    assert request.finished_reason.pd_reason == "PD_PEER_UNAVAILABLE"
    assert request.finished_reason.status_code == 503
    assert streamed == [([request], False)]
