from types import SimpleNamespace

from sglang.srt.environ import envs
from sglang.srt.managers.scheduler import Scheduler


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
