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

from typing import Dict

from sglang.srt.metrics.metrics_types import Stats


class PrometheusMetricsCollector:

    def __init__(self, labels: Dict[str, str], context_len: int) -> None:
        # We need to import this one after the environment variable `PROMETHEUS_MULTIPROC_DIR` is set
        from prometheus_client import Counter, Gauge, Histogram

        self.labels = labels

        self.num_running_reqs = Gauge(
            name="sglang:num_running_reqs",
            documentation="The number of running requests",
            labelnames=list(labels.keys()),
            multiprocess_mode="sum",
        )

        self.num_used_tokens = Gauge(
            name="sglang:num_used_tokens",
            documentation="The number of used tokens",
            labelnames=list(labels.keys()),
            multiprocess_mode="sum",
        )

        self.token_usage = Gauge(
            name="sglang:token_usage",
            documentation="The token usage",
            labelnames=list(labels.keys()),
            multiprocess_mode="mostrecent",
        )

        self.gen_throughput = Gauge(
            name="sglang:gen_throughput",
            documentation="The generate throughput (token/s)",
            labelnames=list(labels.keys()),
            multiprocess_mode="sum",
        )

        self.num_queue_reqs = Gauge(
            name="sglang:num_queue_reqs",
            documentation="The number of requests in the waiting queue",
            labelnames=list(labels.keys()),
            multiprocess_mode="sum",
        )

        self.cache_hit_rate = Gauge(
            name="sglang:cache_hit_rate",
            documentation="The cache hit rate",
            labelnames=list(labels.keys()),
            multiprocess_mode="mostrecent",
        )

    def log_stats(self, stats: Stats) -> None:
        self.num_running_reqs.labels(**self.labels).set(stats.num_running_reqs)
        self.num_used_tokens.labels(**self.labels).set(stats.num_used_tokens)
        self.token_usage.labels(**self.labels).set(stats.token_usage)
        self.gen_throughput.labels(**self.labels).set(stats.gen_throughput)
        self.num_queue_reqs.labels(**self.labels).set(stats.num_queue_reqs)
        self.cache_hit_rate.labels(**self.labels).set(stats.cache_hit_rate)
