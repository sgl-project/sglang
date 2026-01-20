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
import dataclasses
import logging
import os
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union

from sglang.srt.disaggregation.utils import DisaggregationMode
from sglang.srt.environ import envs
from sglang.srt.metrics.utils import exponential_buckets, generate_buckets
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.server_args import ServerArgs
from sglang.srt.utils import get_bool_env_var
from sglang.srt.utils.gauge_histogram import GaugeHistogram

SGLANG_TEST_REQUEST_TIME_STATS = get_bool_env_var("SGLANG_TEST_REQUEST_TIME_STATS")


logger = logging.getLogger(__name__)


def get_histogram_conf_from_env(env_var_name: str) -> Optional[List[float]]:
    """
    Get the histogram configuration from the environment variable.
    env value should be like "0.1,0.2,0.5,1,2"
    """
    if env_var_name not in os.environ:
        return None
    # if the env var is not set or empty, return None
    env_var_value = os.environ[env_var_name]
    if not env_var_value:
        return None
    return [float(x) for x in env_var_value.split(",")]


@dataclass
class TimeStats:
    """
    Store the timestamps for each stage of a request.

    Unified: wait_queue -> forward -> completion
    Prefill: bootstrap_queue -> wait_queue -> forward -> transfer_queue -> completion
    Decode: prealloc_queue -> transfer_queue -> wait_queue -> forward -> completion
    """

    disagg_mode: DisaggregationMode = DisaggregationMode.NULL
    lb_entry_time: float = 0.0
    wait_queue_entry_time: float = 0.0
    forward_entry_time: float = 0.0
    completion_time: float = 0.0
    prefill_bootstrap_queue_entry_time: float = 0.0
    prefill_transfer_queue_entry_time: float = 0.0
    decode_prealloc_queue_entry_time: float = 0.0
    decode_transfer_queue_entry_time: float = 0.0
    # TODO: correct set them
    bootstrap_duration: float = 0.0
    alloc_waiting_duration: float = 0.0
    prefill_start_time_host: float = 0.0
    prefill_end_time_host: float = 0.0
    transfer_speed_gb_s: float = 0.0
    transfer_total_mb: float = 0.0
    # Number of prefill retries for this request
    prefill_retry_count: int = 0

    # Timestamp when prefill phase finishes, obtained from `time.time()`.
    # Note that this differs from the other `_time` fields tracked by the
    # `TimeStats` class, which are obtained from `time.perf_counter()`.
    # We use `time.time()` instead of `time.perf_counter()` here in order to
    # maintain unit consistency with other timestamp fields tracked by the `ReqState` class.
    prefill_finished_ts: float = 0.0

    def get_queueing_time(self) -> float:
        return self.forward_entry_time - self.wait_queue_entry_time

    def get_prefill_launch_delay(self) -> Optional[float]:
        if self.prefill_start_time_host > 0.0:
            return self.prefill_start_time_host - self.forward_entry_time
        return None

    def get_prefill_launch_latency(self) -> Optional[float]:
        if self.prefill_start_time_host > 0.0 and self.prefill_end_time_host > 0.0:
            return self.prefill_end_time_host - self.prefill_start_time_host
        return None

    def get_prefill_finished_ts(self) -> Optional[float]:
        if self.prefill_finished_ts > 0.0:
            return self.prefill_finished_ts
        return None

    def convert_to_duration(self) -> str:
        if self.disagg_mode == DisaggregationMode.NULL:
            queue_duration = self.forward_entry_time - self.wait_queue_entry_time
            forward_duration = self.completion_time - self.forward_entry_time

            if SGLANG_TEST_REQUEST_TIME_STATS:
                assert (
                    queue_duration >= 0 and forward_duration >= 0
                ), f"queue_duration={queue_duration} < 0 or forward_duration={forward_duration} < 0"

            return f"queue_duration={self.format_duration(queue_duration)}, forward_duration={self.format_duration(forward_duration)}, start_time={self.wait_queue_entry_time:.3f}"
        elif self.disagg_mode == DisaggregationMode.PREFILL:
            bootstrap_duration = (
                self.wait_queue_entry_time - self.prefill_bootstrap_queue_entry_time
            )
            queue_duration = self.forward_entry_time - self.wait_queue_entry_time
            forward_duration = self.completion_time - self.forward_entry_time

            if SGLANG_TEST_REQUEST_TIME_STATS:
                if self.wait_queue_entry_time > 0:
                    assert (
                        bootstrap_duration >= 0
                        and queue_duration >= 0
                        and forward_duration >= 0
                    ), f"bootstrap_duration={bootstrap_duration} < 0 or queue_duration={queue_duration} < 0 or forward_duration={forward_duration} < 0"

            other = max(
                0.0,
                bootstrap_duration
                - (self.alloc_waiting_duration + self.bootstrap_duration),
            )
            return (
                f"bootstrap_queue_duration({self.format_duration(bootstrap_duration)}) "
                f"= alloc_wait({self.format_duration(self.alloc_waiting_duration)}) "
                f"+ bootstrap({self.format_duration(self.bootstrap_duration)}) "
                f"+ other({self.format_duration(other)}); "
                f"queue_duration={self.format_duration(queue_duration)}, "
                f"forward_duration={self.format_duration(forward_duration)}, "
                f"start={self.prefill_bootstrap_queue_entry_time:.3f}, "
                f"transfer_speed={self.transfer_speed_gb_s:.2f}GB/s, "
                f"transfer_total={self.transfer_total_mb:.2f}MB, "
                f"#retries={self.prefill_retry_count}"
            )
        elif self.disagg_mode == DisaggregationMode.DECODE:
            prealloc_duration = (
                self.decode_transfer_queue_entry_time
                - self.decode_prealloc_queue_entry_time
            )
            transfer_duration = (
                self.wait_queue_entry_time - self.decode_transfer_queue_entry_time
            )
            queue_duration = self.forward_entry_time - self.wait_queue_entry_time
            forward_duration = self.completion_time - self.forward_entry_time

            if SGLANG_TEST_REQUEST_TIME_STATS:
                if self.wait_queue_entry_time > 0:
                    assert (
                        prealloc_duration >= 0
                        and transfer_duration >= 0
                        and queue_duration >= 0
                        and forward_duration >= 0
                    ), f"prealloc_duration={prealloc_duration} < 0 or transfer_duration={transfer_duration} < 0 or queue_duration={queue_duration} < 0 or forward_duration={forward_duration} < 0. {self=}"

            other = max(
                0.0,
                prealloc_duration
                - (self.alloc_waiting_duration + self.bootstrap_duration),
            )
            return (
                f"prealloc_queue_duration({self.format_duration(prealloc_duration)}) "
                f"= alloc_wait({self.format_duration(self.alloc_waiting_duration)}) "
                f"+ bootstrap({self.format_duration(self.bootstrap_duration)}) "
                f"+ other({self.format_duration(other)}); "
                f"transfer_duration={self.format_duration(transfer_duration)}; "
                f"queue_duration={self.format_duration(queue_duration)}, "
                f"forward_duration={self.format_duration(forward_duration)}, "
                f"start={self.decode_prealloc_queue_entry_time:.3f}"
            )
        else:
            return "Unknown Time Stats"

    def format_duration(self, duration: float) -> str:
        return f"{duration * 1e3:.2f}ms"

    def disagg_mode_str(self) -> str:
        if self.disagg_mode == DisaggregationMode.NULL:
            return "unified"
        elif self.disagg_mode == DisaggregationMode.DECODE:
            return "decode"
        elif self.disagg_mode == DisaggregationMode.PREFILL:
            return "prefill"
        else:
            return "unknown"


@dataclass
class SchedulerStats:
    # Basics
    num_running_reqs: int = 0
    num_used_tokens: int = 0
    token_usage: float = 0.0
    pending_prealloc_token_usage: float = 0.0
    swa_token_usage: float = 0.0
    mamba_usage: float = 0.0
    decode_sum_seq_lens: int = 0
    gen_throughput: float = 0.0
    num_queue_reqs: int = 0
    num_grammar_queue_reqs: int = 0
    num_running_reqs_offline_batch: int = 0
    cache_hit_rate: float = 0.0

    max_total_num_tokens: int = 0

    # Speculative decoding
    spec_accept_length: float = 0.0
    spec_accept_rate: float = 0.0

    # Retract
    num_retracted_reqs: int = 0
    num_paused_reqs: int = 0

    # PD disaggregation
    num_prefill_prealloc_queue_reqs: int = 0
    num_prefill_inflight_queue_reqs: int = 0
    num_decode_prealloc_queue_reqs: int = 0
    num_decode_transfer_queue_reqs: int = 0
    kv_transfer_speed_gb_s: float = 0.0
    kv_transfer_latency_ms: float = 0.0
    kv_transfer_bootstrap_ms: float = 0.0
    kv_transfer_alloc_ms: float = 0.0
    kv_transfer_total_mb: float = 0.0

    # Utilization
    utilization: float = 0.0
    max_running_requests_under_SLO: Optional[int] = None

    # Engine startup
    engine_startup_time: float = 0.0
    engine_load_weights_time: float = 0.0
    new_token_ratio: float = 0.0

    # CUDA graph
    is_cuda_graph: float = 0.0

    # LoRA pool metrics
    lora_pool_slots_used: int = 0
    lora_pool_slots_total: int = 0
    lora_pool_utilization: float = 0.0

    # Routing key metrics
    num_unique_running_routing_keys: int = 0
    routing_key_running_req_counts: List[int] = field(default_factory=list)
    routing_key_all_req_counts: List[int] = field(default_factory=list)


ROUTING_KEY_REQ_COUNT_BUCKET_BOUNDS = [1, 2, 3, 5, 7, 10, 20, 50, 100, 200]


def compute_routing_key_stats(routing_keys: List[Optional[str]]) -> tuple:
    """Returns (num_unique_keys, per_key_counts)."""
    from collections import Counter

    key_counts = Counter(k for k in routing_keys if k is not None)
    return len(key_counts), list(key_counts.values())


@dataclass
class DPCooperationInfo:
    # Users can derive that, except for cases with idle, num_decode_ranks=world_size-num_prefill_ranks
    # We do not provide `num_decode_ranks` to avoid cardinality explosion.
    num_prefill_ranks: int

    @staticmethod
    def create(forward_modes: List[int]):
        return DPCooperationInfo(
            num_prefill_ranks=sum(
                1 for mode in forward_modes if mode == ForwardMode.EXTEND.value
            ),
        )

    def to_labels(self):
        return dataclasses.asdict(self)


class SchedulerMetricsCollector:

    def __init__(
        self,
        labels: Dict[str, str],
        enable_lora: bool = False,
        server_args: Optional["ServerArgs"] = None,
    ) -> None:
        # We need to import prometheus_client after setting the env variable `PROMETHEUS_MULTIPROC_DIR`
        from prometheus_client import Counter, Gauge, Histogram, Summary

        self.labels = labels
        self.enable_lora = enable_lora
        self.last_log_time = time.perf_counter()

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
        self.pending_prealloc_token_usage = Gauge(
            name="sglang:pending_prealloc_token_usage",
            documentation="The token usage for pending preallocated tokens (not preallocated yet).",
            labelnames=labels.keys(),
            multiprocess_mode="mostrecent",
        )
        self.swa_token_usage = Gauge(
            name="sglang:swa_token_usage",
            documentation="The token usage for SWA layers.",
            labelnames=labels.keys(),
            multiprocess_mode="mostrecent",
        )
        self.mamba_usage = Gauge(
            name="sglang:mamba_usage",
            documentation="The token usage for Mamba layers.",
            labelnames=labels.keys(),
            multiprocess_mode="mostrecent",
        )
        self.decode_sum_seq_lens = Gauge(
            name="sglang:decode_sum_seq_lens",
            documentation="The sum of all sequence lengths in decode.",
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
        self.num_grammar_queue_reqs = Gauge(
            name="sglang:num_grammar_queue_reqs",
            documentation="The number of requests in the grammar waiting queue.",
            labelnames=labels.keys(),
            multiprocess_mode="mostrecent",
        )
        self.num_running_reqs_offline_batch = Gauge(
            name="sglang:num_running_reqs_offline_batch",
            documentation="The number of running low-priority offline batch requests(label is 'batch').",
            labelnames=labels.keys(),
            multiprocess_mode="mostrecent",
        )
        self.cache_hit_rate = Gauge(
            name="sglang:cache_hit_rate",
            documentation="The prefix cache hit rate.",
            labelnames=labels.keys(),
            multiprocess_mode="mostrecent",
        )

        self.max_total_num_tokens = Gauge(
            name="sglang:max_total_num_tokens",
            documentation="Maximum total number of tokens in the KV cache pool.",
            labelnames=labels.keys(),
            multiprocess_mode="mostrecent",
        )

        # Speculative decoding
        self.spec_accept_length = Gauge(
            name="sglang:spec_accept_length",
            documentation="The average acceptance length of speculative decoding.",
            labelnames=labels.keys(),
            multiprocess_mode="mostrecent",
        )
        self.spec_accept_rate = Gauge(
            name="sglang:spec_accept_rate",
            documentation="The average acceptance rate of speculative decoding (`accepted tokens / total draft tokens` in batch).",
            labelnames=labels.keys(),
            multiprocess_mode="mostrecent",
        )

        # Retract
        # TODO maybe remove this old gauge in favor of the new counter
        self.num_retracted_reqs = Gauge(
            name="sglang:num_retracted_reqs",
            documentation="The number of retracted requests.",
            labelnames=labels.keys(),
        )
        self.num_retracted_reqs_total = Counter(
            # The name is `requests` instead of `reqs` to avoid dup name error
            name="sglang:num_retracted_requests_total",
            documentation="Total number of retracted requests.",
            labelnames=labels.keys(),
        )
        self.num_retracted_input_tokens_total = Counter(
            name="sglang:num_retracted_input_tokens_total",
            documentation="Total number of retracted input tokens.",
            labelnames=labels.keys(),
        )
        self.num_retracted_output_tokens_total = Counter(
            name="sglang:num_retracted_output_tokens_total",
            documentation="Total number of retracted output tokens.",
            labelnames=labels.keys(),
        )
        self.num_paused_reqs = Gauge(
            name="sglang:num_paused_reqs",
            documentation="The number of paused requests by async weight sync.",
            labelnames=labels.keys(),
        )

        # PD disaggregation
        self.num_prefill_prealloc_queue_reqs = Gauge(
            name="sglang:num_prefill_prealloc_queue_reqs",
            documentation="The number of requests in the prefill prealloc queue.",
            labelnames=labels.keys(),
            multiprocess_mode="mostrecent",
        )
        self.num_prefill_inflight_queue_reqs = Gauge(
            name="sglang:num_prefill_inflight_queue_reqs",
            documentation="The number of requests in the prefill inflight queue.",
            labelnames=labels.keys(),
            multiprocess_mode="mostrecent",
        )
        self.num_decode_prealloc_queue_reqs = Gauge(
            name="sglang:num_decode_prealloc_queue_reqs",
            documentation="The number of requests in the decode prealloc queue.",
            labelnames=labels.keys(),
            multiprocess_mode="mostrecent",
        )
        self.num_decode_transfer_queue_reqs = Gauge(
            name="sglang:num_decode_transfer_queue_reqs",
            documentation="The number of requests in the decode transfer queue.",
            labelnames=labels.keys(),
            multiprocess_mode="mostrecent",
        )
        self.num_bootstrap_failed_reqs = Counter(
            name="sglang:num_bootstrap_failed_reqs_total",
            documentation="The number of bootstrap failed requests.",
            labelnames=labels.keys(),
        )
        self.num_transfer_failed_reqs = Counter(
            name="sglang:num_transfer_failed_reqs_total",
            documentation="The number of transfer failed requests.",
            labelnames=labels.keys(),
        )
        self.num_prefill_retries_total = Counter(
            name="sglang:num_prefill_retries_total",
            documentation="Total number of prefill retries.",
            labelnames=labels.keys(),
        )
        self.kv_transfer_speed_gb_s = Gauge(
            name="sglang:kv_transfer_speed_gb_s",
            documentation="The transfer speed of the KV cache in GB/s.",
            labelnames=labels.keys(),
            multiprocess_mode="mostrecent",
        )
        self.kv_transfer_latency_ms = Gauge(
            name="sglang:kv_transfer_latency_ms",
            documentation="The transfer latency of the KV cache in ms.",
            labelnames=labels.keys(),
            multiprocess_mode="mostrecent",
        )
        self.kv_transfer_bootstrap_ms = Gauge(
            name="sglang:kv_transfer_bootstrap_ms",
            documentation="The bootstrap time of the KV transfer in ms.",
            labelnames=labels.keys(),
            multiprocess_mode="mostrecent",
        )
        self.kv_transfer_alloc_ms = Gauge(
            name="sglang:kv_transfer_alloc_ms",
            documentation="The allocation waiting time of the KV transfer in ms.",
            labelnames=labels.keys(),
            multiprocess_mode="mostrecent",
        )
        self.kv_transfer_total_mb = Gauge(
            name="sglang:kv_transfer_total_mb",
            documentation="The total number of tokens transferred in the KV cache.",
            labelnames=labels.keys(),
            multiprocess_mode="mostrecent",
        )

        # Utilization
        self.utilization = Gauge(
            name="sglang:utilization",
            documentation="The utilization.",
            labelnames=labels.keys(),
            multiprocess_mode="mostrecent",
        )
        self.max_running_requests_under_SLO = Gauge(
            name="sglang:max_running_requests_under_SLO",
            documentation="The maximum number of running requests under SLO.",
            labelnames=labels.keys(),
            multiprocess_mode="mostrecent",
        )

        # Engine startup
        self.engine_startup_time = Gauge(
            name="sglang:engine_startup_time",
            documentation="The time taken for the engine to start up.",
            labelnames=labels.keys(),
            multiprocess_mode="mostrecent",
        )
        self.engine_load_weights_time = Gauge(
            name="sglang:engine_load_weights_time",
            documentation="The time taken for the engine to load weights.",
            labelnames=labels.keys(),
            multiprocess_mode="mostrecent",
        )

        # Additional queueing time histogram
        self.queue_time = Histogram(
            name="sglang:queue_time_seconds",
            documentation="Histogram of queueing time in seconds.",
            labelnames=labels.keys(),
            buckets=[
                0.0,
                0.1,
                0.2,
                0.5,
                1,
                2,
                3,
                4,
                5,
                10,
                15,
                20,
                30,
                40,
                50,
                60,
                70,
                80,
                90,
                100,
                200,
                300,
                400,
                500,
                600,
                700,
                800,
                900,
                1000,
                1200,
                1400,
                1600,
                1800,
                2000,
                2500,
                3000,
            ],
        )

        # Grammar metrics
        self.grammar_compilation_time = Histogram(
            name="sglang:grammar_compilation_time_seconds",
            documentation="Histogram of grammar compilation time in seconds.",
            labelnames=labels.keys(),
            buckets=[
                0.0,
                0.01,
                0.02,
                0.05,
                0.1,
                0.2,
                0.5,
                1,
                2,
                5,
                10,
                20,
                30,
                60,
                90,
                120,
                240,
            ],
        )
        self.num_grammar_cache_hit = Counter(
            name="sglang:num_grammar_cache_hit_total",
            documentation="Number of grammar cache hits.",
            labelnames=labels.keys(),
        )
        self.num_grammar_aborted = Counter(
            name="sglang:num_grammar_aborted_total",
            documentation="Number of grammar aborted requests.",
            labelnames=labels.keys(),
        )
        self.num_grammar_timeout = Counter(
            name="sglang:num_grammar_timeout_total",
            documentation="Number of grammar timeouts.",
            labelnames=labels.keys(),
        )
        self.num_grammar_total = Counter(
            name="sglang:num_grammar_total",
            documentation="Number of the total grammar requests.",
            labelnames=labels.keys(),
        )
        self.grammar_schema_count = Histogram(
            name="sglang:grammar_schema_count",
            documentation="Histogram of grammar schema count.",
            labelnames=labels.keys(),
            buckets=[
                0,
                1,
                2,
                5,
                10,
                20,
                30,
                40,
                60,
                80,
                100,
                120,
                140,
                160,
                180,
                200,
                300,
                400,
                500,
                700,
                1000,
            ],
        )
        self.grammar_ebnf_size = Histogram(
            name="sglang:grammar_ebnf_size",
            documentation="Histogram of grammar EBNF size.",
            labelnames=labels.keys(),
            buckets=[
                0,
                50,
                100,
                200,
                300,
                500,
                1000,
                2000,
                3000,
                5000,
                10000,
                20000,
                30000,
                50000,
                100000,
            ],
        )

        tree_traversal_time_buckets = [
            0.0,
            0.01,
            0.02,
            0.05,
            0.1,
            0.2,
            0.5,
            1,
            2,
            5,
            10,
            15,
            30,
            60,
            90,
            120,
            240,
        ]
        self.grammar_tree_traversal_time_avg = Histogram(
            name="sglang:grammar_tree_traversal_time_avg",
            documentation="Histogram of average grammar tree traversal time in seconds.",
            labelnames=labels.keys(),
            buckets=tree_traversal_time_buckets,
        )
        self.grammar_tree_traversal_time_max = Histogram(
            name="sglang:grammar_tree_traversal_time_max",
            documentation="Histogram of max grammar tree traversal time in seconds.",
            labelnames=labels.keys(),
            buckets=tree_traversal_time_buckets,
        )

        self.per_stage_req_latency_seconds = Histogram(
            name="sglang:per_stage_req_latency_seconds",
            documentation="The latency of each stage of requests.",
            # captures latency in range [1ms - ~1191s]
            buckets=exponential_buckets(start=0.001, width=1.62, length=30),
            labelnames=list(labels.keys()) + ["stage"],
        )

        # TODO maybe remove this old gauge in favor of the new counter
        self.is_cuda_graph = Gauge(
            name="sglang:is_cuda_graph",
            documentation="Whether the batch is using CUDA graph.",
            labelnames=labels.keys(),
            multiprocess_mode="mostrecent",
        )
        self.cuda_graph_passes_total = Counter(
            name="sglang:cuda_graph_passes_total",
            documentation="Total number of forward passes categorized by CUDA graph.",
            labelnames=list(labels.keys()) + ["mode"],
        )

        if (
            labels["moe_ep_rank"] == 0
        ) and envs.SGLANG_ENABLE_EPLB_BALANCEDNESS_METRIC.get():
            self.eplb_balancedness = Summary(
                name="sglang:eplb_balancedness",
                documentation="Balancedness of MoE in expert parallelism.",
                labelnames=list(labels.keys()) + ["forward_mode"],
            )

        # LoRA pool metrics (only created when LoRA is enabled)
        if self.enable_lora:
            self.lora_pool_slots_used = Gauge(
                name="sglang:lora_pool_slots_used",
                documentation="Number of LoRA adapter slots currently occupied in GPU memory.",
                labelnames=labels.keys(),
                multiprocess_mode="mostrecent",
            )
            self.lora_pool_slots_total = Gauge(
                name="sglang:lora_pool_slots_total",
                documentation="Total number of LoRA adapter slots available (max_loras_per_batch).",
                labelnames=labels.keys(),
                multiprocess_mode="mostrecent",
            )
            self.lora_pool_utilization = Gauge(
                name="sglang:lora_pool_utilization",
                documentation="LoRA pool utilization ratio (used/total). 1.0 means pool is full.",
                labelnames=labels.keys(),
                multiprocess_mode="mostrecent",
            )

        self.num_unique_running_routing_keys = Gauge(
            name="sglang:num_unique_running_routing_keys",
            documentation="Number of unique routing keys in running batch.",
            labelnames=labels.keys(),
            multiprocess_mode="mostrecent",
        )
        self.routing_key_running_req_count = GaugeHistogram(
            name="sglang:routing_key_running_req_count",
            documentation="Distribution of routing keys by running request count (gt < count <= le).",
            labelnames=list(labels.keys()),
            bucket_bounds=ROUTING_KEY_REQ_COUNT_BUCKET_BOUNDS,
        )
        self.routing_key_all_req_count = GaugeHistogram(
            name="sglang:routing_key_all_req_count",
            documentation="Distribution of routing keys by running+waiting request count (gt < count <= le).",
            labelnames=list(labels.keys()),
            bucket_bounds=ROUTING_KEY_REQ_COUNT_BUCKET_BOUNDS,
        )

        self.new_token_ratio = Gauge(
            name="sglang:new_token_ratio",
            documentation="The new token ratio.",
            labelnames=labels.keys(),
            multiprocess_mode="mostrecent",
        )

        self.realtime_tokens_total = Counter(
            name="sglang:realtime_tokens_total",
            documentation=(
                "Total number of tokens processed (updated on each log interval). "
                "mode: prefill_compute, prefill_cache, decode."
            ),
            labelnames=list(labels.keys()) + ["mode"],
        )
        self.gpu_execution_seconds_total = Counter(
            name="sglang:gpu_execution_seconds_total",
            documentation=(
                "Total time that GPU is busy executing a workload. "
                "Refer to ForwardMode for category labels."
            ),
            labelnames=list(labels.keys()) + ["category"],
        )

        self.dp_cooperation_realtime_tokens_total = Counter(
            name="sglang:dp_cooperation_realtime_tokens_total",
            documentation=(
                "Total number of tokens processed with labels about DP cooperation. "
                "mode: prefill_compute, prefill_cache, decode."
            ),
            labelnames=list(labels.keys()) + ["mode", "num_prefill_ranks"],
        )
        self.dp_cooperation_gpu_execution_seconds_total = Counter(
            name="sglang:dp_cooperation_gpu_execution_seconds_total",
            documentation=(
                "Total time that GPU is busy executing a workload with labels about DP cooperation. "
                "Refer to ForwardMode for category labels."
            ),
            labelnames=list(labels.keys()) + ["category", "num_prefill_ranks"],
        )

        max_delay = server_args.prefill_delayer_max_delay_passes
        self.prefill_delayer_wait_forward_passes = Histogram(
            name="sglang:prefill_delayer_wait_forward_passes",
            documentation="Histogram of forward passes waited by prefill delayer.",
            labelnames=labels.keys(),
            buckets=sorted(
                set(
                    x
                    for x in (
                        server_args.prefill_delayer_forward_passes_buckets
                        or [5, 20, 50, 100, 200]
                    )
                    if x < max_delay
                )
                # Need bucket "<=0" for zero-delay cases, and "max_delay-1" to distinguish "max_delay" timeout passes
                | {0, max_delay - 1}
            ),
        )
        self.prefill_delayer_wait_seconds = Histogram(
            name="sglang:prefill_delayer_wait_seconds",
            documentation="Histogram of wait time in seconds by prefill delayer.",
            labelnames=labels.keys(),
            buckets=sorted(
                set(
                    server_args.prefill_delayer_wait_seconds_buckets
                    or [1, 2, 5, 10, 20, 50, 100, 200, 500]
                )
                # Need bucket "<=0" for zero-delay cases
                | {0}
            ),
        )
        self.prefill_delayer_outcomes_total = Counter(
            name="sglang:prefill_delayer_outcomes_total",
            documentation="Prefill delayer outcome counts.",
            labelnames=[
                *labels.keys(),
                "input_estimation",
                "output_allow",
                "output_reason",
                "actual_execution",
            ],
        )

    def _log_gauge(self, gauge, data: Union[int, float]) -> None:
        # Convenience function for logging to gauge.
        gauge.labels(**self.labels).set(data)

    def _log_histogram(self, histogram, data: Union[int, float]) -> None:
        histogram.labels(**self.labels).observe(data)

    def increment_bootstrap_failed_reqs(self) -> None:
        self.num_bootstrap_failed_reqs.labels(**self.labels).inc(1)

    def increment_transfer_failed_reqs(self) -> None:
        self.num_transfer_failed_reqs.labels(**self.labels).inc(1)

    def increment_prefill_retries(self, count: int) -> None:
        if count > 0:
            self.num_prefill_retries_total.labels(**self.labels).inc(count)

    def observe_per_stage_req_latency(self, stage: str, latency: float) -> None:
        labels_with_stage = {**self.labels, "stage": stage}
        self.per_stage_req_latency_seconds.labels(**labels_with_stage).observe(latency)

    def observe_queue_time(self, latency: float) -> None:
        self._log_histogram(self.queue_time, latency)

    def observe_prefill_delayer_outcome(
        self,
        forward_passes: int,
        wait_seconds: float,
        input_estimation: str,
        output_allow: bool,
        output_reason: str,
        actual_execution: bool,
    ) -> None:
        if output_allow and actual_execution:
            self._log_histogram(
                self.prefill_delayer_wait_forward_passes, forward_passes
            )
            self._log_histogram(self.prefill_delayer_wait_seconds, wait_seconds)

        self.prefill_delayer_outcomes_total.labels(
            **self.labels,
            input_estimation=input_estimation,
            output_allow=str(output_allow).lower(),
            output_reason=output_reason,
            actual_execution=str(actual_execution).lower(),
        ).inc(1)

    def increment_retracted_reqs(
        self,
        num_retracted_reqs: int,
        num_retracted_input_tokens: int,
        num_retracted_output_tokens: int,
    ) -> None:
        self.num_retracted_reqs_total.labels(**self.labels).inc(num_retracted_reqs)
        self.num_retracted_input_tokens_total.labels(**self.labels).inc(
            num_retracted_input_tokens
        )
        self.num_retracted_output_tokens_total.labels(**self.labels).inc(
            num_retracted_output_tokens
        )

    def increment_cuda_graph_pass(self, value: bool) -> None:
        # leave room for piecewise cuda graph, etc
        mode = "decode_cuda_graph" if value else "decode_none"
        self.cuda_graph_passes_total.labels(**self.labels, mode=mode).inc(1)

    def increment_eplb_balancedness(
        self, forward_mode: str, balancedness: float
    ) -> None:
        self.eplb_balancedness.labels(**self.labels, forward_mode=forward_mode).observe(
            balancedness
        )

    def increment_realtime_tokens(
        self,
        dp_cooperation_info: Optional[DPCooperationInfo],
        prefill_compute_tokens=0,
        prefill_cache_tokens=0,
        decode_tokens=0,
    ):
        for mode, delta in [
            ("prefill_compute", prefill_compute_tokens),
            ("prefill_cache", prefill_cache_tokens),
            ("decode", decode_tokens),
        ]:
            self.realtime_tokens_total.labels(**self.labels, mode=mode).inc(delta)
            if dp_cooperation_info is not None:
                self.dp_cooperation_realtime_tokens_total.labels(
                    **self.labels,
                    mode=mode,
                    **dp_cooperation_info.to_labels(),
                ).inc(delta)

    def increment_gpu_execution_seconds(
        self,
        category: str,
        t: float,
        dp_cooperation_info: Optional[DPCooperationInfo],
    ):
        logger.debug(f"GPU execution seconds: {category=} {t=:.3f}")
        self.gpu_execution_seconds_total.labels(**self.labels, category=category).inc(t)
        if dp_cooperation_info is not None:
            self.dp_cooperation_gpu_execution_seconds_total.labels(
                **self.labels,
                category=category,
                **dp_cooperation_info.to_labels(),
            ).inc(t)

    def log_stats(self, stats: SchedulerStats) -> None:
        self._log_gauge(self.num_running_reqs, stats.num_running_reqs)
        self._log_gauge(self.num_used_tokens, stats.num_used_tokens)
        self._log_gauge(self.token_usage, stats.token_usage)
        self._log_gauge(
            self.pending_prealloc_token_usage, stats.pending_prealloc_token_usage
        )
        self._log_gauge(self.swa_token_usage, stats.swa_token_usage)
        self._log_gauge(self.mamba_usage, stats.mamba_usage)
        self._log_gauge(self.decode_sum_seq_lens, stats.decode_sum_seq_lens)
        self._log_gauge(self.gen_throughput, stats.gen_throughput)
        self._log_gauge(self.num_queue_reqs, stats.num_queue_reqs)
        self._log_gauge(self.num_grammar_queue_reqs, stats.num_grammar_queue_reqs)
        self._log_gauge(
            self.num_running_reqs_offline_batch, stats.num_running_reqs_offline_batch
        )
        self._log_gauge(self.cache_hit_rate, stats.cache_hit_rate)

        self._log_gauge(self.max_total_num_tokens, stats.max_total_num_tokens)

        # Speculative decoding
        self._log_gauge(self.spec_accept_length, stats.spec_accept_length)
        self._log_gauge(self.spec_accept_rate, stats.spec_accept_rate)

        # PD disaggregation
        self._log_gauge(
            self.num_prefill_prealloc_queue_reqs, stats.num_prefill_prealloc_queue_reqs
        )
        self._log_gauge(
            self.num_prefill_inflight_queue_reqs, stats.num_prefill_inflight_queue_reqs
        )
        self._log_gauge(
            self.num_decode_prealloc_queue_reqs, stats.num_decode_prealloc_queue_reqs
        )
        self._log_gauge(
            self.num_decode_transfer_queue_reqs, stats.num_decode_transfer_queue_reqs
        )
        self._log_gauge(self.kv_transfer_speed_gb_s, stats.kv_transfer_speed_gb_s)
        self._log_gauge(self.kv_transfer_latency_ms, stats.kv_transfer_latency_ms)
        self._log_gauge(self.kv_transfer_bootstrap_ms, stats.kv_transfer_bootstrap_ms)
        self._log_gauge(self.kv_transfer_alloc_ms, stats.kv_transfer_alloc_ms)
        self._log_gauge(self.kv_transfer_total_mb, stats.kv_transfer_total_mb)

        # Retract
        self._log_gauge(self.num_retracted_reqs, stats.num_retracted_reqs)
        self._log_gauge(self.num_paused_reqs, stats.num_paused_reqs)

        # Utilization
        self._log_gauge(self.utilization, stats.utilization)
        if stats.max_running_requests_under_SLO is not None:
            self._log_gauge(
                self.max_running_requests_under_SLO,
                stats.max_running_requests_under_SLO,
            )

        # Engine startup time
        self._log_gauge(self.engine_startup_time, stats.engine_startup_time)
        if stats.engine_load_weights_time is not None:
            self._log_gauge(
                self.engine_load_weights_time, stats.engine_load_weights_time
            )
        self._log_gauge(self.new_token_ratio, stats.new_token_ratio)

        # CUDA graph
        self._log_gauge(self.is_cuda_graph, stats.is_cuda_graph)

        # LoRA pool metrics (only logged if LoRA is enabled)
        if self.enable_lora:
            self._log_gauge(self.lora_pool_slots_used, stats.lora_pool_slots_used)
            self._log_gauge(self.lora_pool_slots_total, stats.lora_pool_slots_total)
            self._log_gauge(self.lora_pool_utilization, stats.lora_pool_utilization)

        self._log_gauge(
            self.num_unique_running_routing_keys, stats.num_unique_running_routing_keys
        )
        self.routing_key_running_req_count.set_by_current_observations(
            self.labels, stats.routing_key_running_req_counts
        )
        self.routing_key_all_req_count.set_by_current_observations(
            self.labels, stats.routing_key_all_req_counts
        )

        self.last_log_time = time.perf_counter()

    def log_grammar_stats(self, grammar_stats) -> None:
        if grammar_stats.compilation_time is not None:
            self._log_histogram(
                self.grammar_compilation_time, grammar_stats.compilation_time
            )
        if grammar_stats.schema_count is not None:
            self._log_histogram(self.grammar_schema_count, grammar_stats.schema_count)
        if grammar_stats.ebnf_size is not None:
            self._log_histogram(self.grammar_ebnf_size, grammar_stats.ebnf_size)
        tree_times = grammar_stats.tree_traversal_time
        if tree_times:
            max_time = max(tree_times)
            avg_time = sum(tree_times) / len(tree_times)
            self._log_histogram(self.grammar_tree_traversal_time_max, max_time)
            self._log_histogram(self.grammar_tree_traversal_time_avg, avg_time)
        if grammar_stats.is_cache_hit:
            self.num_grammar_cache_hit.labels(**self.labels).inc(1)
        if grammar_stats.is_grammar_aborted:
            self.num_grammar_aborted.labels(**self.labels).inc(1)
        if grammar_stats.num_timeout > 0:
            self.num_grammar_timeout.labels(**self.labels).inc(
                grammar_stats.num_timeout
            )
        self.num_grammar_total.labels(**self.labels).inc(1)


class TokenizerMetricsCollector:
    def __init__(
        self,
        server_args: Optional[ServerArgs] = None,
        labels: Dict[str, str] = None,
        bucket_time_to_first_token: Optional[List[float]] = None,
        bucket_inter_token_latency: Optional[List[float]] = None,
        bucket_e2e_request_latency: Optional[List[float]] = None,
        collect_tokens_histogram: bool = False,
    ) -> None:
        # We need to import prometheus_client after setting the env variable `PROMETHEUS_MULTIPROC_DIR`
        from prometheus_client import Counter, Histogram

        self.labels = labels or {}
        self.collect_tokens_histogram = collect_tokens_histogram

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

        if collect_tokens_histogram:
            default_bucket_prompt_tokens = [
                100,
                300,
                500,
                700,
                1000,
                1500,
                2000,
                3000,
                4000,
                5000,
                6000,
                7000,
                8000,
                9000,
                10000,
                12000,
                15000,
                20000,
                22000,
                25000,
                30000,
                35000,
                40000,
                66000,
                99000,
                132000,
                300000,
                600000,
                900000,
                1100000,
            ]
            self.prompt_tokens_histogram = Histogram(
                name="sglang:prompt_tokens_histogram",
                documentation="Histogram of prompt token length.",
                labelnames=labels.keys(),
                buckets=generate_buckets(
                    server_args.prompt_tokens_buckets, default_bucket_prompt_tokens
                ),
            )
            self.generation_tokens_histogram = Histogram(
                name="sglang:generation_tokens_histogram",
                documentation="Histogram of generation token length.",
                labelnames=labels.keys(),
                buckets=generate_buckets(
                    server_args.generation_tokens_buckets,
                    default_bucket_prompt_tokens,
                ),
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

        self.num_so_requests_total = Counter(
            name="sglang:num_so_requests_total",
            documentation="Number of structured output requests processed.",
            labelnames=labels.keys(),
        )

        self.num_aborted_requests_total = Counter(
            name="sglang:num_aborted_requests_total",
            documentation="Number of requests aborted.",
            labelnames=labels.keys(),
        )

        if bucket_time_to_first_token is None:
            bucket_time_to_first_token = [
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
            ]

        if bucket_e2e_request_latency is None:
            bucket_e2e_request_latency = [
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
                600,
                1200,
                1800,
                2400,
            ]

        if bucket_inter_token_latency is None:
            bucket_inter_token_latency = [
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
            ]

        self.histogram_time_to_first_token = Histogram(
            name="sglang:time_to_first_token_seconds",
            documentation="Histogram of time to first token in seconds.",
            labelnames=labels.keys(),
            buckets=bucket_time_to_first_token,
        )

        self.histogram_inter_token_latency = Histogram(
            name="sglang:inter_token_latency_seconds",
            documentation="Histogram of inter-token latency in seconds.",
            labelnames=labels.keys(),
            buckets=bucket_inter_token_latency,
        )

        self.histogram_e2e_request_latency = Histogram(
            name="sglang:e2e_request_latency_seconds",
            documentation="Histogram of End-to-end request latency in seconds",
            labelnames=labels.keys(),
            buckets=bucket_e2e_request_latency,
        )

        # Retraction count histogram
        self.num_retractions = Histogram(
            name="sglang:num_retractions",
            documentation="Histogram of retraction counts per request.",
            labelnames=labels.keys(),
            buckets=[
                0,
                1,
                2,
                3,
                4,
                5,
                6,
                7,
                8,
                9,
                10,
                15,
                20,
                25,
                30,
                40,
                50,
                75,
                100,
            ],
        )

    def observe_one_finished_request(
        self,
        labels: Dict[str, str],
        prompt_tokens: int,
        generation_tokens: int,
        cached_tokens: int,
        e2e_latency: float,
        has_grammar: bool,
        retraction_count: int,
    ):
        self.prompt_tokens_total.labels(**labels).inc(prompt_tokens)
        self.generation_tokens_total.labels(**labels).inc(generation_tokens)
        if cached_tokens > 0:
            self.cached_tokens_total.labels(**labels).inc(cached_tokens)
        self.num_requests_total.labels(**labels).inc(1)
        if has_grammar:
            self.num_so_requests_total.labels(**labels).inc(1)
        self.histogram_e2e_request_latency.labels(**labels).observe(float(e2e_latency))
        if self.collect_tokens_histogram:
            self.prompt_tokens_histogram.labels(**labels).observe(float(prompt_tokens))
            self.generation_tokens_histogram.labels(**labels).observe(
                float(generation_tokens)
            )
        self.num_retractions.labels(**labels).observe(retraction_count)

    def observe_time_to_first_token(self, labels: Dict[str, str], value: float):
        self.histogram_time_to_first_token.labels(**labels).observe(value)

    def check_time_to_first_token_straggler(self, value: float) -> bool:
        his = self.histogram_time_to_first_token.labels(**self.labels)
        total_observations = sum(bucket._value for bucket in his._buckets)
        if total_observations < 100:
            return False
        p99_threshold = total_observations * 0.99
        cumulative_count = 0
        for i, bucket in enumerate(his._buckets):
            cumulative_count += bucket._value
            if cumulative_count > p99_threshold:
                return value >= his._upper_bounds[i]
        return False

    def observe_inter_token_latency(
        self, labels: Dict[str, str], internval: float, num_new_tokens: int
    ):
        adjusted_interval = internval / num_new_tokens

        # A faster version of the Histogram::observe which observes multiple values at the same time.
        # reference: https://github.com/prometheus/client_python/blob/v0.21.1/prometheus_client/metrics.py#L639
        his = self.histogram_inter_token_latency.labels(**labels)
        his._sum.inc(internval)

        for i, bound in enumerate(his._upper_bounds):
            if adjusted_interval <= bound:
                his._buckets[i].inc(num_new_tokens)
                break

    def observe_one_aborted_request(self, labels: Dict[str, str]):
        self.num_aborted_requests_total.labels(**labels).inc(1)


@dataclass
class StorageMetrics:
    prefetch_pgs: List[int] = field(default_factory=list)
    backup_pgs: List[int] = field(default_factory=list)
    prefetch_bandwidth: List[float] = field(default_factory=list)
    backup_bandwidth: List[float] = field(default_factory=list)


class StorageMetricsCollector:
    def __init__(
        self,
        labels: Dict[str, str],
    ):
        from prometheus_client import Counter, Histogram

        self.labels = labels

        self.prefetched_tokens_total = Counter(
            name="sglang:prefetched_tokens_total",
            documentation="Number of prefetched prompt tokens.",
            labelnames=labels.keys(),
        )

        self.backuped_tokens_total = Counter(
            name="sglang:backuped_tokens_total",
            documentation="Number of backuped tokens.",
            labelnames=labels.keys(),
        )

        bucket_io = [
            1,
            5,
            10,
            50,
            100,
        ]

        bucket_bandwidth = [
            0.1,
            0.5,
            1,
            5,
            10,
            50,
            100,
        ]

        self.histogram_prefetch_pgs = Histogram(
            name="sglang:prefetch_pgs",
            documentation="Histogram of prefetch pages of batches.",
            labelnames=labels.keys(),
            buckets=bucket_io,
        )

        self.histogram_backup_pgs = Histogram(
            name="sglang:backup_pgs",
            documentation="Histogram of backup pages of batches.",
            labelnames=labels.keys(),
            buckets=bucket_io,
        )

        self.histogram_prefetch_bandwidth = Histogram(
            name="sglang:prefetch_bandwidth",
            documentation="Histogram of prefetch bandwidth in GB/s.",
            labelnames=labels.keys(),
            buckets=bucket_bandwidth,
        )

        self.histogram_backup_bandwidth = Histogram(
            name="sglang:backup_bandwidth",
            documentation="Histogram of backup bandwidth in GB/s.",
            labelnames=labels.keys(),
            buckets=bucket_bandwidth,
        )

    def log_prefetched_tokens(self, prefetched_tokens: int):
        if prefetched_tokens > 0:
            self.prefetched_tokens_total.labels(**self.labels).inc(prefetched_tokens)

    def log_backuped_tokens(self, backuped_tokens: int):
        if backuped_tokens > 0:
            self.backuped_tokens_total.labels(**self.labels).inc(backuped_tokens)

    def _log_histogram(self, histogram, data: Union[int, float]):
        histogram.labels(**self.labels).observe(data)

    def log_storage_metrics(self, storage_metrics: Optional[StorageMetrics] = None):
        if storage_metrics is None:
            return

        assert isinstance(storage_metrics, StorageMetrics)

        for v in storage_metrics.prefetch_pgs:
            self._log_histogram(self.histogram_prefetch_pgs, v)
        for v in storage_metrics.backup_pgs:
            self._log_histogram(self.histogram_backup_pgs, v)
        for v in storage_metrics.prefetch_bandwidth:
            self._log_histogram(self.histogram_prefetch_bandwidth, v)
        for v in storage_metrics.backup_bandwidth:
            self._log_histogram(self.histogram_backup_bandwidth, v)


class ExpertDispatchCollector:
    def __init__(self, ep_size: int) -> None:
        from prometheus_client import Histogram

        ep_size_buckets = [i for i in range(ep_size)]
        self.eplb_gpu_physical_count = Histogram(
            name="sglang:eplb_gpu_physical_count",
            documentation="The selected count of physical experts on each layer and GPU rank.",
            labelnames={"layer"},
            buckets=ep_size_buckets,
        )


class RadixCacheMetricsCollector:
    def __init__(
        self,
        labels: Dict[str, str],
    ) -> None:
        # We need to import prometheus_client after setting the env variable `PROMETHEUS_MULTIPROC_DIR`
        from prometheus_client import Counter, Histogram

        self.labels = labels

        bucket_eviction_duration = get_histogram_conf_from_env(
            "SGLANG_BUCKET_EVICTION_DURATION"
        )
        if bucket_eviction_duration is None:
            bucket_eviction_duration = [
                0.001,
                0.002,
                0.003,
                0.004,
                0.005,
                0.006,
                0.007,
                0.008,
                0.009,
                0.01,
                0.02,
                0.03,
                0.04,
                0.05,
                0.1,
                0.2,
                0.5,
                1.0,
            ]
        bucket_load_back_duration = get_histogram_conf_from_env(
            "SGLANG_BUCKET_LOAD_BACK_DURATION"
        )
        if bucket_load_back_duration is None:
            bucket_load_back_duration = [
                0.001,
                0.002,
                0.003,
                0.004,
                0.005,
                0.006,
                0.007,
                0.008,
                0.009,
                0.01,
                0.02,
                0.03,
                0.04,
                0.05,
                0.1,
                0.2,
                0.5,
                1.0,
            ]
        self.eviction_duration_seconds = Histogram(
            name="sglang:eviction_duration_seconds",
            documentation="Time taken to evict memory from GPU to CPU in seconds.",
            labelnames=labels.keys(),
            buckets=bucket_eviction_duration,
        )

        self.eviction_num_tokens = Counter(
            name="sglang:evicted_tokens_total",
            documentation="The number of tokens evicted from GPU to CPU.",
            labelnames=labels.keys(),
        )

        self.load_back_duration_seconds = Histogram(
            name="sglang:load_back_duration_seconds",
            documentation="Time taken to load memory from CPU to GPU in seconds.",
            labelnames=labels.keys(),
            buckets=bucket_load_back_duration,
        )

        self.load_back_num_tokens = Counter(
            name="sglang:load_back_tokens_total",
            documentation="The number of tokens loaded from CPU to GPU.",
            labelnames=labels.keys(),
        )

    def increment_eviction_num_tokens(self, num_tokens: int) -> None:
        self.eviction_num_tokens.labels(**self.labels).inc(num_tokens)

    def increment_load_back_num_tokens(self, num_tokens: int) -> None:
        self.load_back_num_tokens.labels(**self.labels).inc(num_tokens)

    def observe_eviction_duration(self, duration_seconds: float) -> None:
        self.eviction_duration_seconds.labels(**self.labels).observe(duration_seconds)

    def observe_load_back_duration(self, duration_seconds: float) -> None:
        self.load_back_duration_seconds.labels(**self.labels).observe(duration_seconds)
