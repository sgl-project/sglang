# Copyright 2023-2024 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Utilities for Prometheus Metrics Collection."""

from dataclasses import dataclass
from typing import Dict, Union


@dataclass
class SchedulerStats:
    num_running_reqs: int = 0
    num_used_tokens: int = 0
    token_usage: float = 0.0
    gen_throughput: float = 0.0
    num_queue_reqs: int = 0
    cache_hit_rate: float = 0.0


class SchedulerMetricsCollector:

    def __init__(self, labels: Dict[str, str]) -> None:
        # We need to import prometheus_client after setting the env variable `PROMETHEUS_MULTIPROC_DIR`
        from prometheus_client import Gauge

        self.labels = labels

        self.num_running_reqs = Gauge(
            name="sglang:num_running_reqs",
            documentation="The number of running requests",
            labelnames=labels.keys(),
            multiprocess_mode="sum",
        )

        self.num_used_tokens = Gauge(
            name="sglang:num_used_tokens",
            documentation="The number of used tokens",
            labelnames=labels.keys(),
            multiprocess_mode="sum",
        )

        self.token_usage = Gauge(
            name="sglang:token_usage",
            documentation="The token usage",
            labelnames=labels.keys(),
            multiprocess_mode="mostrecent",
        )

        self.gen_throughput = Gauge(
            name="sglang:gen_throughput",
            documentation="The generate throughput (token/s)",
            labelnames=labels.keys(),
            multiprocess_mode="sum",
        )

        self.num_queue_reqs = Gauge(
            name="sglang:num_queue_reqs",
            documentation="The number of requests in the waiting queue",
            labelnames=labels.keys(),
            multiprocess_mode="sum",
        )

        self.cache_hit_rate = Gauge(
            name="sglang:cache_hit_rate",
            documentation="The cache hit rate",
            labelnames=labels.keys(),
            multiprocess_mode="mostrecent",
        )

    def _log_gauge(self, gauge, data: Union[int, float]) -> None:
        # Convenience function for logging to gauge.
        gauge.labels(**self.labels).set(data)

    def log_stats(self, stats: SchedulerStats) -> None:
        self._log_gauge(self.num_running_reqs, stats.num_running_reqs)
        self._log_gauge(self.num_used_tokens, stats.num_used_tokens)
        self._log_gauge(self.token_usage, stats.token_usage)
        self._log_gauge(self.gen_throughput, stats.gen_throughput)
        self._log_gauge(self.num_queue_reqs, stats.num_queue_reqs)
        self._log_gauge(self.cache_hit_rate, stats.cache_hit_rate)


class TokenizerMetricsCollector:
    def __init__(self, labels: Dict[str, str]) -> None:
        # We need to import prometheus_client after setting the env variable `PROMETHEUS_MULTIPROC_DIR`
        from prometheus_client import Counter, Histogram

        self.labels = labels

        self.prompt_tokens_total = Counter(
            name="sglang:prompt_tokens_total",
            documentation="Number of prefill tokens processed.",
            labelnames=labels.keys(),
        )

        self.generation_tokens_total = Counter(
            name="sglang:generation_tokens_total",
            documentation="Number of generation tokens processed.",
            labelnames=labels.keys(),
        )

        self.histogram_time_to_first_token = Histogram(
            name="sglang:time_to_first_token_seconds",
            documentation="Histogram of time to first token in seconds.",
            labelnames=labels.keys(),
            buckets=[
                0.001,
                0.005,
                0.01,
                0.02,
                0.04,
                0.06,
                0.08,
                0.1,
                0.25,
                0.5,
                0.75,
                1.0,
                2.5,
                5.0,
                7.5,
                10.0,
                15.0,
                20.0,
                25.0,
                30.0,
            ],
        )

        self.histogram_time_per_output_token = Histogram(
            name="sglang:time_per_output_token_seconds",
            documentation="Histogram of time per output token in seconds.",
            labelnames=labels.keys(),
            buckets=[
                0.005,
                0.01,
                0.015,
                0.02,
                0.025,
                0.03,
                0.04,
                0.05,
                0.075,
                0.1,
                0.15,
                0.2,
                0.3,
                0.4,
                0.5,
                0.75,
                1.0,
                2.5,
            ],
        )

        self.histogram_e2e_request_latency = Histogram(
            name="sglang:e2e_request_latency_seconds",
            documentation="Histogram of End-to-end request latency in seconds",
            labelnames=labels.keys(),
            buckets=[
                0.3,
                0.5,
                0.8,
                1.0,
                1.5,
                2.0,
                2.5,
                5.0,
                10.0,
                15.0,
                20.0,
                30.0,
                40.0,
                50.0,
                60.0,
            ],
        )

    def _log_histogram(self, histogram, data: Union[int, float]) -> None:
        histogram.labels(**self.labels).observe(data)

    def _log_counter(self, counter, data: Union[int, float]) -> None:
        # Convenience function for logging to counter.
        counter.labels(**self.labels).inc(data)

    def inc_prompt_tokens(self, value: int):
        self._log_counter(self.prompt_tokens_total, value)

    def inc_generation_tokens(self, value: int):
        self._log_counter(self.generation_tokens_total, value)

    def observe_time_to_first_token(self, value: Union[float, int]):
        self._log_histogram(self.histogram_time_to_first_token, value)

    def observe_time_per_output_token(self, value: Union[float, int]):
        self._log_histogram(self.histogram_time_per_output_token, value)

    def observe_e2e_request_latency(self, value: Union[float, int]):
        self._log_histogram(self.histogram_e2e_request_latency, value)
