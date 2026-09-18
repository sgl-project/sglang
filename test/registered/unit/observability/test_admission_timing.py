import sys
from functools import partial
from types import SimpleNamespace as NS

import pytest
from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram

from sglang.srt.disaggregation.utils import DisaggregationMode
from sglang.srt.managers.tokenizer_manager import TokenizerManager
from sglang.srt.observability.admission_timing import parse_admission_wait
from sglang.srt.observability.metrics_collector import TokenizerMetricsCollector
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


@pytest.mark.parametrize("value", [None, "bad", "nan", "inf", "-1"])
def test_invalid_timing_is_zero(value):
    assert parse_admission_wait(value) == 0


@pytest.mark.parametrize(
    "role",
    [DisaggregationMode.NULL, DisaggregationMode.PREFILL, DisaggregationMode.DECODE],
)
@pytest.mark.parametrize("trusted_wait", [None, 0.0, 3.0])
@pytest.mark.parametrize("stream", [False, True])
def test_admission_metrics_are_separate_and_observed_once(
    role, trusted_wait, stream, monkeypatch
):
    monkeypatch.setattr(
        "sglang.srt.observability.metrics_collector.get_observability",
        lambda: NS(prompt_tokens_buckets=None, generation_tokens_buckets=None),
    )
    registry = CollectorRegistry()

    class Collector(TokenizerMetricsCollector):
        _counter_cls = staticmethod(partial(Counter, registry=registry))
        _gauge_cls = staticmethod(partial(Gauge, registry=registry))
        _histogram_cls = staticmethod(partial(Histogram, registry=registry))

    collector = Collector(labels={"engine_type": role.value})
    manager = NS(
        metrics_collector=collector,
        enable_priority_scheduling=False,
        disaggregation_mode=role,
        _request_has_grammar=lambda _: False,
    )
    time_stats = NS(
        get_first_token_latency=lambda: 2.0,
        get_e2e_latency=lambda: 8.0,
        get_interval=lambda: 1.0,
        set_last_time=lambda: None,
    )
    state = NS(
        obj=NS(stream=stream),
        ttft_observed=False,
        last_completion_tokens=0,
        admission_wait_seconds=trusted_wait,
        admission_ttft_observed=False,
        time_stats=time_stats,
        finished=False,
    )
    recv = NS(completion_tokens=[1], prompt_tokens=[20], cached_tokens=[10])
    TokenizerManager.collect_metrics(manager, state, recv, 0)
    state.finished = True
    recv.completion_tokens = [2]
    TokenizerManager.collect_metrics(manager, state, recv, 0)
    labels = {"engine_type": role.value, "is_streaming": str(stream).lower()}
    observed = role != DisaggregationMode.PREFILL and trusted_wait is not None
    for metric, engine_seconds in (
        ("admission_inclusive_time_to_first_token_seconds", 2.0),
        ("admission_inclusive_e2e_request_latency_seconds", 8.0),
    ):
        count = registry.get_sample_value(f"sglang:{metric}_count", labels)
        total = registry.get_sample_value(f"sglang:{metric}_sum", labels)
        assert count == (1 if observed else None)
        assert total == (trusted_wait + engine_seconds if observed else None)
    if role != DisaggregationMode.PREFILL:
        # Adding admission timing must not change the existing engine TTFT.
        assert (
            registry.get_sample_value("sglang:time_to_first_token_seconds_sum", labels)
            == 2.0
        )
        assert (
            registry.get_sample_value(
                "sglang:time_to_first_token_seconds_count", labels
            )
            == 1
        )


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
