"""CPU regression tests executing real metrics methods with Prometheus collectors."""

import sys
from functools import partial
from types import SimpleNamespace as NS

import pytest
from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram

from sglang.srt.disaggregation.utils import DisaggregationMode
from sglang.srt.managers.tokenizer_manager import TokenizerManager
from sglang.srt.observability.metrics_collector import TokenizerMetricsCollector
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

COLLECT = TokenizerManager.collect_metrics


@pytest.fixture(autouse=True)
def observability_config(monkeypatch):
    monkeypatch.setattr(
        "sglang.srt.observability.metrics_collector.get_observability",
        lambda: NS(prompt_tokens_buckets=None, generation_tokens_buckets=None),
    )


def collector(role):
    registry = CollectorRegistry()

    class Collector(TokenizerMetricsCollector):
        _counter_cls = staticmethod(partial(Counter, registry=registry))
        _gauge_cls = staticmethod(partial(Gauge, registry=registry))
        _histogram_cls = staticmethod(partial(Histogram, registry=registry))

    return registry, Collector(labels={"engine_type": role})


def request(role, *, stream=False, tokens=101, reason="length", finished=True):
    registry, metrics = collector(role)
    state = NS(
        obj=NS(stream=stream),
        ttft_observed=False,
        last_completion_tokens=1,
        finished=finished,
        time_stats=NS(
            get_first_token_latency=lambda: 0.1,
            get_interval=lambda: 0.1,
            get_e2e_latency=lambda: 1.0,
            set_last_time=lambda: None,
        ),
    )
    manager = NS(
        metrics_collector=metrics,
        disaggregation_mode=DisaggregationMode(role),
        enable_priority_scheduling=False,
        _request_has_grammar=lambda obj: False,
    )
    recv = NS(
        completion_tokens=[tokens],
        finished_reasons=[{"type": reason}],
        prompt_tokens=[100],
        cached_tokens=[60],
    )
    return registry, manager, state, recv


@pytest.mark.parametrize("role", ["null", "decode", "prefill"])
def test_abort_without_output_does_not_observe_first_token(role):
    registry, manager, state, recv = request(role, tokens=0, reason="abort")
    COLLECT(manager, state, recv, 0)
    assert not state.ttft_observed
    assert (
        registry.get_sample_value(
            "sglang:time_to_first_token_seconds_count",
            {"engine_type": role, "is_streaming": "false"},
        )
        is None
    )
    assert (
        registry.get_sample_value(
            "sglang:finished_requests_by_outcome_total",
            {"engine_type": role, "outcome": "abort"},
        )
        == 1
    )


def test_prefill_never_observes_decode_intervals():
    registry, manager, state, recv = request("prefill", tokens=0, finished=False)
    for count in (0, 1, 8, 0):
        recv.completion_tokens = [count]
        COLLECT(manager, state, recv, 0)
    assert (
        registry.get_sample_value(
            "sglang:inter_token_latency_seconds_count", {"engine_type": "prefill"}
        )
        is None
    )


def test_token_count_reset_changes_baseline_without_negative_histograms():
    registry, manager, state, recv = request("decode", tokens=8, finished=False)
    COLLECT(manager, state, recv, 0)
    for count in (0, 0, 2):
        recv.completion_tokens = [count]
        COLLECT(manager, state, recv, 0)
    labels = {"engine_type": "decode"}
    assert state.last_completion_tokens == 2
    assert (
        registry.get_sample_value("sglang:inter_token_latency_seconds_count", labels)
        == 2
    )
    assert registry.get_sample_value(
        "sglang:inter_token_latency_seconds_sum", labels
    ) == pytest.approx(0.1)


@pytest.mark.parametrize("delta", [-10, -1, 0])
def test_collector_rejects_nonpositive_token_weights(delta):
    registry, metrics = collector("decode")
    metrics.observe_inter_token_latency({"engine_type": "decode"}, 0.1, delta)
    assert (
        registry.get_sample_value(
            "sglang:inter_token_latency_seconds_count", {"engine_type": "decode"}
        )
        is None
    )


@pytest.mark.parametrize(
    "reason,outcome",
    [
        ("stop", "success"),
        ("length", "success"),
        ("abort", "abort"),
        ("other", "other"),
    ],
)
def test_terminal_outcomes_and_tokens_are_separated(reason, outcome):
    registry, manager, state, recv = request("decode", reason=reason)
    COLLECT(manager, state, recv, 0)
    labels = {"engine_type": "decode", "outcome": outcome}
    for metric, expected in [
        ("finished_requests_by_outcome", 1),
        ("finished_prompt_tokens_by_outcome", 100),
        ("finished_cached_tokens_by_outcome", 60),
    ]:
        assert (
            registry.get_sample_value("sglang:" + metric + "_total", labels) == expected
        )


@pytest.mark.parametrize("injected_backend", [False, True])
def test_positive_intervals_preserve_weight_for_both_collector_backends(
    injected_backend,
):
    registry, metrics = collector("decode")
    if not injected_backend:
        metrics._histogram_cls = None
    metrics.observe_inter_token_latency({"engine_type": "decode"}, 0.12, 3)
    labels = {"engine_type": "decode"}
    assert (
        registry.get_sample_value("sglang:inter_token_latency_seconds_count", labels)
        == 3
    )
    assert registry.get_sample_value(
        "sglang:inter_token_latency_seconds_sum", labels
    ) == pytest.approx(0.12)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
