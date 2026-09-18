"""CPU regression tests executing real metrics methods with Prometheus collectors."""

import unittest
from functools import partial
from types import SimpleNamespace as NS
from unittest.mock import patch

from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram

from sglang.srt.disaggregation.utils import DisaggregationMode
from sglang.srt.managers.tokenizer_manager import TokenizerManager
from sglang.srt.observability.metrics_collector import TokenizerMetricsCollector
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

COLLECT = TokenizerManager.collect_metrics


def collector(role):
    registry = CollectorRegistry()

    class Collector(TokenizerMetricsCollector):
        _counter_cls = staticmethod(partial(Counter, registry=registry))
        _gauge_cls = staticmethod(partial(Gauge, registry=registry))
        _histogram_cls = staticmethod(partial(Histogram, registry=registry))

    return registry, Collector(labels={"engine_type": role})


def request(role, *, tokens, finished=True):
    registry, metrics = collector(role)
    state = NS(
        obj=NS(stream=False),
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
        prompt_tokens=[100],
        cached_tokens=[60],
    )
    return registry, manager, state, recv


class TestPDRequestMetrics(CustomTestCase):
    def setUp(self):
        super().setUp()
        config = patch(
            "sglang.srt.observability.metrics_collector.get_observability",
            return_value=NS(prompt_tokens_buckets=None, generation_tokens_buckets=None),
        )
        config.start()
        self.addCleanup(config.stop)

    def test_abort_without_output_does_not_observe_first_token(self):
        """An output-free terminal request must not contribute a TTFT sample."""
        for role in ("null", "decode", "prefill"):
            with self.subTest(role=role):
                registry, manager, state, recv = request(role, tokens=0)
                COLLECT(manager, state, recv, 0)
                self.assertFalse(state.ttft_observed)
                self.assertIsNone(
                    registry.get_sample_value(
                        "sglang:time_to_first_token_seconds_count",
                        {"engine_type": role, "is_streaming": "false"},
                    )
                )

    def test_prefill_never_observes_decode_intervals(self):
        """Prefill token counts must not enter the decode ITL histogram."""
        registry, manager, state, recv = request("prefill", tokens=0, finished=False)
        for count in (0, 1, 8, 0):
            recv.completion_tokens = [count]
            COLLECT(manager, state, recv, 0)
        self.assertIsNone(
            registry.get_sample_value(
                "sglang:inter_token_latency_seconds_count", {"engine_type": "prefill"}
            )
        )

    def test_token_count_reset_changes_baseline_without_negative_histograms(self):
        registry, manager, state, recv = request("decode", tokens=8, finished=False)
        COLLECT(manager, state, recv, 0)
        for count in (0, 0, 2):
            recv.completion_tokens = [count]
            COLLECT(manager, state, recv, 0)
        labels = {"engine_type": "decode"}
        self.assertEqual(state.last_completion_tokens, 2)
        self.assertEqual(
            registry.get_sample_value(
                "sglang:inter_token_latency_seconds_count", labels
            ),
            2,
        )
        self.assertAlmostEqual(
            registry.get_sample_value("sglang:inter_token_latency_seconds_sum", labels),
            0.1,
        )

    def test_collector_rejects_nonpositive_token_weights(self):
        for delta in (-10, -1, 0):
            with self.subTest(delta=delta):
                registry, metrics = collector("decode")
                metrics.observe_inter_token_latency(
                    {"engine_type": "decode"}, 0.1, delta
                )
                self.assertIsNone(
                    registry.get_sample_value(
                        "sglang:inter_token_latency_seconds_count",
                        {"engine_type": "decode"},
                    )
                )


if __name__ == "__main__":
    unittest.main()
