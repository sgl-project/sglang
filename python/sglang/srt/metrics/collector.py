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

import time
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
    spec_accept_length: float = 0.0
    avg_request_queue_latency: float = 0.0


class SchedulerMetricsCollector:

    def __init__(self, labels: Dict[str, str]) -> None:
        # We need to import prometheus_client after setting the env variable `PROMETHEUS_MULTIPROC_DIR`
        from prometheus_client import Gauge, Histogram

        self.labels = labels
        self.last_log_time = time.time()

        self.num_running_reqs = Gauge(
            name="sglang:num_running_reqs",
            documentation="The number of running requests.",
            labelnames=labels.keys(),
            multiprocess_mode="mostrecent",
        )

        self.num_used_tokens = Gauge(
            name="sglang:num_used_tokens",
            documentation="The number of used tokens.",
            labelnames=labels.keys(),
            multiprocess_mode="mostrecent",
        )

        self.token_usage = Gauge(
            name="sglang:token_usage",
            documentation="The token usage.",
            labelnames=labels.keys(),
            multiprocess_mode="mostrecent",
        )

        self.gen_throughput = Gauge(
            name="sglang:gen_throughput",
            documentation="The generation throughput (token/s).",
            labelnames=labels.keys(),
            multiprocess_mode="mostrecent",
        )

        self.num_queue_reqs = Gauge(
            name="sglang:num_queue_reqs",
            documentation="The number of requests in the waiting queue.",
            labelnames=labels.keys(),
            multiprocess_mode="mostrecent",
        )

        self.cache_hit_rate = Gauge(
            name="sglang:cache_hit_rate",
            documentation="The prefix cache hit rate.",
            labelnames=labels.keys(),
            multiprocess_mode="mostrecent",
        )

        self.spec_accept_length = Gauge(
            name="sglang:spec_accept_length",
            documentation="The average acceptance length of speculative decoding.",
            labelnames=labels.keys(),
            multiprocess_mode="mostrecent",
        )

        self.avg_request_queue_latency = Gauge(
            name="sglang:avg_request_queue_latency",
            documentation="The average request queue latency for the last batch of requests in seconds.",
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
        self._log_gauge(self.spec_accept_length, stats.spec_accept_length)
        self._log_gauge(self.avg_request_queue_latency, stats.avg_request_queue_latency)
        self.last_log_time = time.time()


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

        self.cached_tokens_total = Counter(
            name="sglang:cached_tokens_total",
            documentation="Number of cached prompt tokens.",
            labelnames=labels.keys(),
        )

        self.num_requests_total = Counter(
            name="sglang:num_requests_total",
            documentation="Number of requests processed.",
            labelnames=labels.keys(),
        )

        self.histogram_time_to_first_token = Histogram(
            name="sglang:time_to_first_token_seconds",
            documentation="Histogram of time to first token in seconds.",
            labelnames=labels.keys(),
            buckets=[
                0.1,
                0.2,
                0.4,
                0.6,
                0.8,
                1,
                2,
                4,
                6,
                8,
                10,
                20,
                40,
                60,
                80,
                100,
                200,
                400,
            ],
        )

        self.histogram_inter_token_latency_seconds = Histogram(
            name="sglang:inter_token_latency_seconds",
            documentation="Histogram of inter-token latency in seconds.",
            labelnames=labels.keys(),
            buckets=[
                0.002,
                0.004,
                0.006,
                0.008,
                0.010,
                0.015,
                0.020,
                0.025,
                0.030,
                0.035,
                0.040,
                0.060,
                0.080,
                0.100,
                0.200,
                0.400,
                0.600,
                0.800,
                1.000,
                2.000,
                4.000,
                6.000,
                8.000,
            ],
        )

        self.histogram_e2e_request_latency = Histogram(
            name="sglang:e2e_request_latency_seconds",
            documentation="Histogram of End-to-end request latency in seconds",
            labelnames=labels.keys(),
            buckets=[
                0.1,
                0.2,
                0.4,
                0.6,
                0.8,
                1,
                2,
                4,
                6,
                8,
                10,
                20,
                40,
                60,
                80,
                100,
                200,
                400,
                800,
            ],
        )

    def _log_histogram(self, histogram, data: Union[int, float]) -> None:
        histogram.labels(**self.labels).observe(data)

    def observe_one_finished_request(
        self,
        prompt_tokens: int,
        generation_tokens: int,
        cached_tokens: int,
        e2e_latency: float,
    ):
        self.prompt_tokens_total.labels(**self.labels).inc(prompt_tokens)
        self.generation_tokens_total.labels(**self.labels).inc(generation_tokens)
        if cached_tokens > 0:
            self.cached_tokens_total.labels(**self.labels).inc(cached_tokens)
        self.num_requests_total.labels(**self.labels).inc(1)
        self._log_histogram(self.histogram_e2e_request_latency, e2e_latency)

    def observe_time_to_first_token(self, value: float):
        self.histogram_time_to_first_token.labels(**self.labels).observe(value)

    def observe_inter_token_latency(self, internval: float, num_new_tokens: int):
        adjusted_interval = internval / num_new_tokens

        # A faster version of the Histogram::observe which observes multiple values at the same time.
        # reference: https://github.com/prometheus/client_python/blob/v0.21.1/prometheus_client/metrics.py#L639
        his = self.histogram_inter_token_latency_seconds.labels(**self.labels)
        his._sum.inc(internval)

        for i, bound in enumerate(his._upper_bounds):
            if adjusted_interval <= bound:
                his._buckets[i].inc(num_new_tokens)
                break
