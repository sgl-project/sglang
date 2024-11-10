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

from typing import Counter as CollectionsCounter
from typing import Dict, List, Union

from prometheus_client import Counter, Gauge, Histogram

from sglang.srt.metrics.metrics_types import Stats


class PrometheusMetricsCollector:

    def __init__(self, labels: Dict[str, str], context_len: int) -> None:
        self.labels = labels

        self.num_running_reqs = Gauge(
            name="sglang:num_running_reqs",
            documentation="The number of running requests",
            labelnames=labels,
            multiprocess_mode="sum",
        )

        self.num_used_tokens = Gauge(
            name="sglang:num_used_tokens",
            documentation="The number of used tokens",
            labelnames=labels,
            multiprocess_mode="sum",
        )

        self.token_usage = Gauge(
            name="sglang:token_usage",
            documentation="The token usage",
            labelnames=labels,
            multiprocess_mode="mostrecent",
        )

        self.gen_throughput = Gauge(
            name="sglang:gen_throughput",
            documentation="The generate throughput (token/s)",
            labelnames=labels,
            multiprocess_mode="sum",
        )

        self.num_queue_reqs = Gauge(
            name="sglang:num_queue_reqs",
            documentation="The number of requests in the waiting queue",
            labelnames=labels,
            multiprocess_mode="sum",
        )

        self.cache_hit_rate = Gauge(
            name="sglang:cache_hit_rate",
            documentation="The cache hit rate",
            labelnames=labels,
            multiprocess_mode="mostrecent",
        )

    def log_stats(self, stats: Stats) -> None:
        self.num_running_reqs.set(stats.num_running_reqs)
