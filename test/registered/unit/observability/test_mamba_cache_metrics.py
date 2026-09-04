from types import SimpleNamespace

import torch

from sglang.srt.managers.schedule_policy import (
    get_mamba_cache_miss_tokens,
    match_prefix_for_req,
)
from sglang.srt.managers.scheduler_components.metrics_reporter import PrefillStats
from sglang.srt.mem_cache.base_prefix_cache import MatchResult
from sglang.srt.observability.metrics_collector import SchedulerMetricsCollector
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class _RecordingMetric:
    def __init__(self):
        self.values = []

    def labels(self, **_labels):
        return self

    def inc(self, value):
        self.values.append(value)


def _match_result(*, branching_seqlen=None, full_kv_hit_length=0, device_len=0):
    return MatchResult(
        device_indices=torch.empty((device_len,), dtype=torch.int64),
        last_device_node=None,
        last_host_node=None,
        best_match_node=None,
        mamba_branching_seqlen=branching_seqlen,
        full_kv_hit_length=full_kv_hit_length,
    )


def test_mamba_cache_miss_tokens_uses_reusable_boundary():
    result = _match_result(
        branching_seqlen=8192,
        full_kv_hit_length=10000,
        device_len=2048,
    )

    assert get_mamba_cache_miss_tokens(result) == 6144


def test_mamba_cache_miss_tokens_ignores_non_mamba_misses():
    assert get_mamba_cache_miss_tokens(_match_result(full_kv_hit_length=10000)) == 0
    assert (
        get_mamba_cache_miss_tokens(
            _match_result(branching_seqlen=8192, full_kv_hit_length=0)
        )
        == 0
    )


def test_match_prefix_records_mamba_cache_miss_on_request():
    result = _match_result(
        branching_seqlen=8192,
        full_kv_hit_length=10000,
        device_len=2048,
    )

    class _TreeCache:
        def swa_reprefill_tail_tokens(self):
            return 0

        def match_prefix(self, _params):
            return result

    req = SimpleNamespace(
        origin_input_ids=[1, 2, 3],
        output_ids=[],
        extra_key=None,
        cache_salt=None,
        kv=SimpleNamespace(cache_protected_len=0),
        _compute_max_prefix_len=lambda input_len: input_len,
    )

    match_prefix_for_req(_TreeCache(), req)

    assert req.mamba_cache_miss_tokens == 6144


def test_prefill_stats_reports_each_request_once():
    miss_req = SimpleNamespace(
        mamba_cache_miss_tokens=6144,
        _mamba_cache_miss_reported=False,
    )
    clean_req = SimpleNamespace(
        mamba_cache_miss_tokens=0,
        _mamba_cache_miss_reported=False,
    )
    adder = SimpleNamespace(
        can_run_list=[miss_req, clean_req],
        log_input_tokens=1,
        log_hit_tokens=2,
        reprocessed_log_input_tokens=0,
        reprocessed_log_hit_tokens=0,
        log_device_hit_tokens=2,
        log_host_hit_tokens=0,
        log_storage_hit_tokens=0,
        new_token_ratio=1.0,
    )

    first = PrefillStats.from_adder(adder, [])
    second = PrefillStats.from_adder(adder, [])

    assert (first.mamba_cache_miss_requests, first.mamba_cache_miss_tokens) == (1, 6144)
    assert (second.mamba_cache_miss_requests, second.mamba_cache_miss_tokens) == (0, 0)


def test_scheduler_metrics_collector_increments_mamba_cache_miss_counters():
    collector = object.__new__(SchedulerMetricsCollector)
    collector.labels = {"model_name": "test"}
    collector.mamba_cache_miss_requests_total = _RecordingMetric()
    collector.mamba_cache_miss_tokens_total = _RecordingMetric()

    collector.increment_mamba_cache_miss(num_requests=2, num_tokens=12288)

    assert collector.mamba_cache_miss_requests_total.values == [2]
    assert collector.mamba_cache_miss_tokens_total.values == [12288]
