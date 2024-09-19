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

from typing import Dict, List, Optional, Union, Counter as CollectionsCounter

import numpy as np
from prometheus_client import Gauge

from sglang.srt.metrics.metrics_types import ConfigStats


class Metrics:
    """
    SGLang Metrics
    """

    def __init__(self, labelnames: List[str]):

        # Configuration Stats
        self.max_total_num_tokens = Gauge(
            "max_total_num_tokens", 
            "Maximum total number of tokens", 
            labelnames=labelnames, 
            multiprocess_mode="min") # static across processes

        self.max_prefill_tokens = Gauge(
            "max_prefill_tokens", 
            "Maximum prefill tokens", 
            labelnames=labelnames, 
            multiprocess_mode="min") # static across processes

        self.max_running_requests = Gauge(
            "max_running_requests", 
            "Maximum running requests", 
            labelnames=labelnames,
            multiprocess_mode="min") # static across processes

        self.context_len = Gauge(
            "context_len", 
            "Context length", 
            labelnames=labelnames,
            multiprocess_mode="min") # static across processes

        # Decode Stats in tp_worker.py
        # logger.info(
        #     f"Decode batch. "
        #     f"#running-req: {len(self.running_batch.reqs)}, "
        #     f"#token: {num_used}, "
        #     f"token usage: {num_used / self.max_total_num_tokens:.2f}, "
        #     f"gen throughput (token/s): {throughput:.2f}, "
        #     f"#queue-req: {len(self.waiting_queue)}"
        # )

        # prefill batch in tp_worker.py
        # logger.info(
        #             f"Prefill batch"
        #             f"(mixed #running-req: {num_mixed_running}). "
        #             f"#new-seq: {len(can_run_list)}, "
        #             f"#new-token: {adder.log_input_tokens}, "
        #             f"#cached-token: {adder.log_hit_tokens}, "
        #             f"cache hit rate: {100.0 * tree_cache_hit_rate:.2f}%, "
        #             f"#queue-req: {len(self.waiting_queue) - len(can_run_list) + has_inflight}"
        #         )


class SGLangMetricsCollector:
    """
    SGLang Metrics Collector
    """

    def __init__(self, labels: Dict[str, str]) -> None:
        self.labels = labels
        self.metrics = Metrics(labelnames=list(labels.keys()))

    def _log_gauge(self, gauge, data: Union[int, float]) -> None:
        # Convenience function for logging to gauge.
        gauge.labels(**self.labels).set(data)

    def _log_counter(self, counter, data: Union[int, float]) -> None:
        # Convenience function for logging to counter.
        counter.labels(**self.labels).inc(data)

    def _log_counter_labels(self, counter, data: CollectionsCounter,
                            label_key: str) -> None:
        # Convenience function for collection counter of labels.
        for label, count in data.items():
            counter.labels(**{**self.labels, label_key: label}).inc(count)

    def _log_histogram(self, histogram, data: Union[List[int],
                                                    List[float]]) -> None:
        # Convenience function for logging list to histogram.
        for datum in data:
            histogram.labels(**self.labels).observe(datum)

    def log_config_stats(self, stats: ConfigStats) -> None:
        self._log_gauge(self.metrics.max_total_num_tokens, stats.max_total_num_tokens)
        self._log_gauge(self.metrics.max_prefill_tokens, stats.max_prefill_tokens)
        self._log_gauge(self.metrics.max_running_requests, stats.max_running_requests)
        self._log_gauge(self.metrics.context_len, stats.context_len)
