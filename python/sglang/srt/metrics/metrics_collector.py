"""
Copyright 2023-2024 SGLang Team
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

"""Utilities for Prometheus Metrics Collection."""

import logging
from abc import ABC, abstractmethod
from typing import Counter as CollectionsCounter
from typing import Dict, List, Union

import numpy as np
from prometheus_client import Counter, Gauge, Histogram

from sglang.srt.metrics.metrics_types import Stats


class Metrics:
    """
    SGLang Metrics
    """

    def __init__(self, labelnames: List[str], max_model_len):

        # Configuration Stats
        self.max_total_num_tokens = Gauge(
            name="sglang:max_total_num_tokens",
            documentation="Maximum total number of tokens",
            labelnames=labelnames,
            multiprocess_mode="min",
        )  # static across processes

        self.max_prefill_tokens = Gauge(
            name="sglang:max_prefill_tokens",
            documentation="Maximum prefill tokens",
            labelnames=labelnames,
            multiprocess_mode="min",
        )  # static across processes

        self.max_running_requests = Gauge(
            name="sglang:max_running_requests",
            documentation="Maximum running requests",
            labelnames=labelnames,
            multiprocess_mode="min",
        )  # static across processes

        self.context_len = Gauge(
            name="sglang:context_len",
            documentation="Context length",
            labelnames=labelnames,
            multiprocess_mode="min",
        )  # static across processes
        # Decode Stats
        self.num_running_sys = Gauge(
            name="sglang:num_requests_running",
            documentation="Number of requests currently running on GPU",
            labelnames=labelnames,
            multiprocess_mode="sum",
        )
        self.num_waiting_sys = Gauge(
            name="sglang:num_requests_waiting",
            documentation="Number of requests waiting to be processed.",
            labelnames=labelnames,
            multiprocess_mode="sum",
        )
        self.gen_throughput = Gauge(
            name="sglang:gen_throughput",
            documentation="Gen token throughput (token/s)",
            labelnames=labelnames,
            multiprocess_mode="sum",
        )
        self.token_usage = Gauge(
            name="sglang:token_usage",
            documentation="Total token usage",
            labelnames=labelnames,
            multiprocess_mode="sum",
        )
        # System Stats
        #   KV Cache Usage in %
        # self.gpu_cache_usage_sys = Gauge(
        #     "gpu_cache_usage_perc",
        #     "GPU KV-cache usage. 1 means 100 percent usage.",
        #     labelnames=labelnames,
        #     multiprocess_mode="sum")

        self.new_seq = Gauge(
            name="sglang:new_seq",
            documentation="Number of new sequences",
            labelnames=labelnames,
            multiprocess_mode="sum",
        )
        self.new_token = Gauge(
            name="sglang:new_token",
            documentation="Number of new token",
            labelnames=labelnames,
            multiprocess_mode="sum",
        )
        #   Prefix caching block hit rate
        self.cached_token = Gauge(
            name="sglang:cached_token",
            documentation="Number of cached token",
            labelnames=labelnames,
            multiprocess_mode="sum",
        )
        self.cache_hit_rate = Gauge(
            name="sglang:cache_hit_rate",
            documentation="Cache hit rate",
            labelnames=labelnames,
            multiprocess_mode="sum",
        )
        self.queue_req = Gauge(
            name="sglang:queue_req",
            documentation="Number of queued requests",
            labelnames=labelnames,
            multiprocess_mode="sum",
        )

        # Iteration stats
        self.counter_prompt_tokens = Counter(
            name="sglang:prompt_tokens_total",
            documentation="Number of prefill tokens processed.",
            labelnames=labelnames,
        )
        self.counter_generation_tokens = Counter(
            name="sglang:generation_tokens_total",
            documentation="Number of generation tokens processed.",
            labelnames=labelnames,
        )
        self.histogram_time_to_first_token = Histogram(
            name="sglang:time_to_first_token_seconds",
            documentation="Histogram of time to first token in seconds.",
            labelnames=labelnames,
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
            labelnames=labelnames,
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

        # Request Stats
        #   Metadata
        self.num_prompt_tokens_requests = Histogram(
            name="sglang:request_prompt_tokens",
            documentation="Number of prefill tokens processed",
            labelnames=labelnames,
            buckets=build_1_2_5_buckets(max_model_len),
        )
        self.num_generation_tokens_requests = Histogram(
            name="sglang:request_generation_tokens",
            documentation="Number of generation tokens processed.",
            labelnames=labelnames,
            buckets=build_1_2_5_buckets(max_model_len),
        )
        self.finished_reason_requests = Counter(
            name="sglang:request_success_total",
            documentation="Count of successfully processed requests.",
            labelnames=labelnames + ["finished_reason"],
        )
        self.histogram_time_e2e_requests = Histogram(
            name="sglang:e2e_request_latency_seconds",
            documentation="Histogram of End-to-end request latency in seconds",
            labelnames=labelnames,
            buckets=build_1_2_5_buckets(max_model_len),
        )
        self.histogram_time_waiting_requests = Histogram(
            name="sglang:waiting_request_latency_seconds",
            documentation="Histogram of request waiting time in seconds",
            labelnames=labelnames,
            buckets=build_1_2_5_buckets(max_model_len),
        )
        self.histogram_time_decode_requests = Histogram(
            name="sglang:decode_request_latency_seconds",
            documentation="Histogram of request decoding time in seconds",
            labelnames=labelnames,
            buckets=build_1_2_5_buckets(max_model_len),
        )


class MetricsCollector(ABC):
    """
    SGLang Metrics Collector
    """

    @abstractmethod
    def log_stats(self, stats: Stats) -> None:
        pass


class PrometheusMetricsCollector(MetricsCollector):
    """
    SGLang Metrics Collector
    """

    def __init__(self, labels: Dict[str, str], max_model_len: int) -> None:
        self.labels = labels
        self.metrics = Metrics(
            labelnames=list(labels.keys()), max_model_len=max_model_len
        )

    def _log_gauge(self, gauge, data: Union[int, float]) -> None:
        # Convenience function for logging to gauge.
        gauge.labels(**self.labels).set(data)

    def _log_counter(self, counter, data: Union[int, float]) -> None:
        # Convenience function for logging to counter.
        counter.labels(**self.labels).inc(data)

    def _log_counter_labels(
        self, counter, data: CollectionsCounter, label_key: str
    ) -> None:
        # Convenience function for collection counter of labels.
        for label, count in data.items():
            counter.labels(**{**self.labels, label_key: label}).inc(count)

    def _log_histogram(self, histogram, data: Union[List[int], List[float]]) -> None:
        # Convenience function for logging list to histogram.
        for datum in data:
            histogram.labels(**self.labels).observe(datum)

    def log_stats(self, stats: Stats) -> None:
        self._log_gauge(self.metrics.max_total_num_tokens, stats.max_total_num_tokens)
        self._log_gauge(self.metrics.max_prefill_tokens, stats.max_prefill_tokens)
        self._log_gauge(self.metrics.max_running_requests, stats.max_running_requests)
        self._log_gauge(self.metrics.context_len, stats.context_len)
        self._log_histogram(
            self.metrics.num_prompt_tokens_requests, stats.num_prompt_tokens_requests
        )
        self._log_histogram(
            self.metrics.num_generation_tokens_requests,
            stats.num_generation_tokens_requests,
        )

        self._log_counter(
            self.metrics.counter_prompt_tokens, stats.num_prompt_tokens_iter
        )
        self._log_counter(
            self.metrics.counter_generation_tokens, stats.num_generation_tokens_iter
        )
        self._log_histogram(
            self.metrics.histogram_time_to_first_token, stats.time_to_first_tokens_iter
        )
        self._log_histogram(
            self.metrics.histogram_time_per_output_token,
            stats.time_per_output_tokens_iter,
        )

        # self._log_gauge(self.metrics.gpu_cache_usage_sys, stats.gpu_cache_usage_sys)
        self._log_gauge(self.metrics.num_running_sys, stats.num_running_req)
        self._log_gauge(self.metrics.num_waiting_sys, stats.num_waiting_req)
        self._log_gauge(self.metrics.gen_throughput, stats.gen_throughput)
        self._log_gauge(self.metrics.token_usage, stats.token_usage)
        self._log_histogram(
            self.metrics.histogram_time_e2e_requests, stats.time_e2e_requests
        )
        self._log_histogram(
            self.metrics.histogram_time_waiting_requests, stats.time_waiting_requests
        )
        self._log_histogram(
            self.metrics.histogram_time_decode_requests, stats.time_decode_requests
        )
        self._log_gauge(self.metrics.new_seq, stats.new_seq)
        self._log_gauge(self.metrics.new_token, stats.new_token)
        self._log_gauge(self.metrics.cached_token, stats.cached_token)
        self._log_gauge(self.metrics.cache_hit_rate, stats.cache_hit_rate)
        self._log_gauge(self.metrics.queue_req, stats.queue_req)


def build_1_2_5_buckets(max_value: int) -> List[int]:
    """
    Builds a list of buckets with increasing powers of 10 multiplied by
    mantissa values (1, 2, 5) until the value exceeds the specified maximum.

    Example:
    >>> build_1_2_5_buckets(100)
    [1, 2, 5, 10, 20, 50, 100]
    """
    mantissa_lst = [1, 2, 5]
    exponent = 0
    buckets: List[int] = []
    while True:
        for m in mantissa_lst:
            value = m * 10**exponent
            if value <= max_value:
                buckets.append(value)
            else:
                return buckets
        exponent += 1
