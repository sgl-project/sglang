"""Terminal outcome counters preserve reported tokens and aggregate metrics."""

import unittest
from functools import partial
from types import SimpleNamespace
from unittest.mock import patch

from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram

from sglang.srt.disaggregation.utils import DisaggregationMode
from sglang.srt.managers.tokenizer_manager import TokenizerManager
from sglang.srt.observability.metrics_collector import TokenizerMetricsCollector
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


class TestTerminalOutcomeMetrics(CustomTestCase):
    def setUp(self):
        super().setUp()
        config = patch(
            "sglang.srt.observability.metrics_collector.get_observability",
            return_value=SimpleNamespace(
                prompt_tokens_buckets=None, generation_tokens_buckets=None
            ),
        )
        config.start()
        self.addCleanup(config.stop)

    def test_terminal_outcomes_preserve_tokens_and_aggregate_counters(self):
        for reason, outcome, tokens in (
            ("stop", "success", 8),
            ("length", "success", 8),
            ("abort", "abort", 0),
            ("abort", "abort", 8),
            ("unknown", "other", 0),
            (None, "other", 0),
        ):
            with self.subTest(reason=reason, tokens=tokens):
                registry = CollectorRegistry()

                class Collector(TokenizerMetricsCollector):
                    _counter_cls = staticmethod(partial(Counter, registry=registry))
                    _gauge_cls = staticmethod(partial(Gauge, registry=registry))
                    _histogram_cls = staticmethod(partial(Histogram, registry=registry))

                labels = {"model_name": "test"}
                manager = SimpleNamespace(
                    metrics_collector=Collector(labels=labels),
                    disaggregation_mode=DisaggregationMode.NULL,
                    enable_priority_scheduling=False,
                    _request_has_grammar=lambda obj: False,
                )
                state = SimpleNamespace(
                    obj=SimpleNamespace(stream=False),
                    ttft_observed=False,
                    last_completion_tokens=1,
                    finished=False,
                    time_stats=SimpleNamespace(
                        get_first_token_latency=lambda: 0.1,
                        get_interval=lambda: 0.1,
                        get_e2e_latency=lambda: 1.0,
                        set_last_time=lambda: None,
                    ),
                )
                recv = SimpleNamespace(
                    completion_tokens=[tokens],
                    finished_reasons=[{"type": reason} if reason else None],
                    prompt_tokens=[100],
                    cached_tokens=[60],
                )
                outcome_labels = {**labels, "outcome": outcome}
                TokenizerManager.collect_metrics(manager, state, recv, 0)
                self.assertIsNone(
                    registry.get_sample_value(
                        "sglang:finished_requests_by_outcome_total", outcome_labels
                    )
                )
                state.finished = True
                TokenizerManager.collect_metrics(manager, state, recv, 0)
                for metric, expected in (
                    ("finished_requests_by_outcome", 1),
                    ("finished_prompt_tokens_by_outcome", 100),
                    ("finished_cached_tokens_by_outcome", 60),
                ):
                    self.assertEqual(
                        registry.get_sample_value(
                            f"sglang:{metric}_total", outcome_labels
                        ),
                        expected,
                    )
                self.assertEqual(
                    registry.get_sample_value(
                        "sglang:prompt_tokens_total",
                        {**labels, "is_streaming": "false"},
                    ),
                    100,
                )


if __name__ == "__main__":
    unittest.main()
