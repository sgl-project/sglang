from __future__ import annotations

import dataclasses
import logging
import time
from collections import defaultdict
from contextlib import contextmanager
from typing import TYPE_CHECKING, List, Optional, Tuple, Union

from sglang.srt.disaggregation.kv_events import EventPublisherFactory, KVEventBatch
from sglang.srt.disaggregation.utils import DisaggregationMode
from sglang.srt.environ import envs
from sglang.srt.managers.io_struct import (
    DisaggregationMetrics,
    GetLoadsReqInput,
    GetLoadsReqOutput,
    LoRAMetrics,
    MemoryMetrics,
    QueueMetrics,
    SpeculativeMetrics,
)
from sglang.srt.managers.scheduler import ScheduleBatch
from sglang.srt.managers.utils import GenerationBatchResult
from sglang.srt.observability.metrics_collector import (
    DPCooperationInfo,
    QueueCount,
    SchedulerMetricsCollector,
    SchedulerStats,
    compute_routing_key_stats,
)
from sglang.srt.utils.device_timer import DeviceTimer, GapTimer
from sglang.srt.utils.scheduler_status_logger import SchedulerStatusLogger

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import Req
    from sglang.srt.managers.schedule_policy import PrefillAdder
    from sglang.srt.managers.scheduler import EmbeddingBatchResult, Scheduler

logger = logging.getLogger(__name__)

RECORD_STEP_TIME = envs.SGLANG_RECORD_STEP_TIME.get()
LOG_FORWARD_ITERS = envs.SGLANG_LOG_FORWARD_ITERS.get()
ENABLE_METRICS_DEVICE_TIMER = envs.SGLANG_ENABLE_METRICS_DEVICE_TIMER.get()


@dataclasses.dataclass
class PrefillStats:
    """Stats for logging prefill batch metrics."""

    log_input_tokens: int
    log_hit_tokens: int
    new_token_ratio: float
    num_running_reqs: QueueCount
    num_new_seqs: int  # len(can_run_list)
    num_pending_tokens: int = 0

    @classmethod
    def from_adder(
        cls,
        adder: PrefillAdder,
        running_reqs: List[Req],
        enable_priority_scheduling: bool = False,
        num_pending_tokens: int = 0,
    ):
        return cls(
            log_input_tokens=adder.log_input_tokens,
            log_hit_tokens=adder.log_hit_tokens,
            new_token_ratio=adder.new_token_ratio,
            num_running_reqs=QueueCount.from_reqs(
                running_reqs, enable_priority_scheduling
            ),
            num_new_seqs=len(adder.can_run_list),
            num_pending_tokens=num_pending_tokens,
        )


class KvMetrics:
    def __init__(self):
        self.request_active_slots = None
        self.request_total_slots = None
        self.kv_active_blocks = None
        self.kv_total_blocks = None
        self.num_requests_waiting = None
        self.gpu_cache_usage_perc = None
        self.gpu_prefix_cache_hit_rate = None
        self.data_parallel_rank = None


class SchedulerMetricsMixin:
    def init_metrics(
        self: Scheduler, tp_rank: int, pp_rank: int, dp_rank: Optional[int]
    ):
        # Basic stats
        self.forward_ct_decode = 0
        self.num_generated_tokens = 0
        self.last_decode_stats_tic = time.perf_counter()
        self.last_prefill_stats_tic = time.perf_counter()
        self.last_gen_throughput: float = 0.0
        self.last_input_throughput: float = 0.0
        self.step_time_dict = defaultdict(list)  # Dict[batch size -> step time]

        # The number of accepted tokens and forward ct for the recent `decode_log_interval` batches (for logging)
        self.spec_num_accepted_tokens = 0
        self.spec_num_forward_ct = 0
        # The total number of accepted tokens and forward ct for the whole server lifetime
        self.spec_total_num_accepted_tokens = 0
        self.spec_total_num_forward_ct = 0

        # For PD disaggregation
        self.kv_transfer_speed_gb_s: float = 0.0
        self.kv_transfer_latency_ms: float = 0.0

        self.stats = SchedulerStats()

        # Metrics
        self.enable_mfu_metrics = False
        self.enable_metrics = self.server_args.enable_metrics
        self.is_stats_logging_rank = self.attn_tp_rank == 0
        self.current_scheduler_metrics_enabled = self.enable_metrics and (
            self.attn_tp_rank == 0 or self.server_args.enable_metrics_for_all_schedulers
        )
        if self.enable_metrics:
            if self.server_args.disaggregation_mode == DisaggregationMode.PREFILL.value:
                engine_type = "prefill"
            elif (
                self.server_args.disaggregation_mode == DisaggregationMode.DECODE.value
            ):
                engine_type = "decode"
            else:
                engine_type = "unified"

            labels = {
                "model_name": self.server_args.served_model_name,
                "engine_type": engine_type,
                "tp_rank": tp_rank,
                "pp_rank": pp_rank,
                "moe_ep_rank": self.moe_ep_rank,
            }
            if self.enable_priority_scheduling:
                labels["priority"] = ""
            if dp_rank is not None:
                labels["dp_rank"] = dp_rank
            if self.server_args.extra_metric_labels:
                labels.update(self.server_args.extra_metric_labels)
            self.metrics_collector = SchedulerMetricsCollector(
                labels=labels,
                enable_lora=self.enable_lora,
                enable_hierarchical_cache=self.enable_hierarchical_cache,
                enable_streaming_session=self.server_args.enable_streaming_session,
                server_args=self.server_args,
            )
            self.enable_mfu_metrics = bool(self.server_args.enable_mfu_metrics)
            if self.enable_mfu_metrics:
                self._init_estimated_perf_constants()
                self._mfu_log_flops = 0.0
                self._mfu_log_read_bytes = 0.0
                self._mfu_log_write_bytes = 0.0

            if ENABLE_METRICS_DEVICE_TIMER:
                self.forward_pass_device_timer = DeviceTimer(
                    reporter=self.metrics_collector.increment_gpu_execution_seconds,
                )
                self.bubble_timer = GapTimer(
                    reporter=self.metrics_collector.increment_gpu_overlap_wait_seconds,
                )

        self.init_kv_events(self.server_args.kv_events_config)

        self.scheduler_status_logger = SchedulerStatusLogger.maybe_create(
            enable_metrics=self.enable_metrics
        )

    def init_kv_events(self: Scheduler, kv_events_config: Optional[str]):
        self.enable_kv_cache_events = bool(
            kv_events_config and self.attn_tp_rank == 0 and self.attn_cp_rank == 0
        )

        if self.enable_kv_cache_events:
            self.kv_event_publisher = EventPublisherFactory.create(
                kv_events_config, self.attn_dp_rank
            )

    def update_spec_metrics(self: Scheduler, bs: int, num_accepted_tokens: int):
        self.spec_num_accepted_tokens += num_accepted_tokens + bs
        self.spec_num_forward_ct += bs
        self.num_generated_tokens += num_accepted_tokens

    def _init_estimated_perf_constants(self: Scheduler) -> None:
        model_config = self.model_config
        hf_text_config = model_config.hf_text_config

        hidden_size = float(model_config.hidden_size)
        num_layers = float(getattr(model_config, "num_attention_layers", 0))
        head_dim = float(getattr(model_config, "head_dim", 0))
        num_attn_heads = float(model_config.get_num_attention_heads(self.tp_size))
        num_kv_heads = float(model_config.get_num_kv_heads(self.tp_size))
        intermediate_size = getattr(hf_text_config, "intermediate_size", None)
        if intermediate_size is None:
            intermediate_size = getattr(hf_text_config, "ffn_hidden_size", 0)
        intermediate_size = float(intermediate_size)

        dtype_num_bytes = getattr(model_config.dtype, "itemsize", None)
        if dtype_num_bytes is None:
            dtype_num_bytes = 2
        # Keep this estimator lightweight and consistent with current server dtype.
        # KV cache quantization-aware bytes can be added in a follow-up.
        act_bytes = float(dtype_num_bytes)
        w_bytes = float(dtype_num_bytes)
        cache_bytes = float(dtype_num_bytes)

        # Linear-layer FLOPs per token on one GPU.
        attn_linear_flops = (
            2.0 * hidden_size * head_dim * (num_attn_heads + 2.0 * num_kv_heads)
            + 2.0 * hidden_size * head_dim * num_attn_heads
        )
        mlp_flops = (
            6.0 * hidden_size * intermediate_size if intermediate_size > 0 else 0.0
        )
        self._linear_flops_per_token = max(
            0.0, (attn_linear_flops + mlp_flops) * num_layers
        )

        # Attention dot-product FLOPs coefficient to multiply token-context product.
        # attn_qk + attn_av = 4 * q * TC * d * L
        self._attn_dot_flops_coeff = 4.0 * num_attn_heads * head_dim * num_layers

        # KV cache bytes (write one K and one V vector per generated token).
        self._kv_cache_bytes_per_token = (
            2.0 * num_layers * num_kv_heads * head_dim * cache_bytes
        )

        # Weight read bytes per token.
        self._weight_read_bytes_per_token = (
            hidden_size
            * head_dim
            * (num_attn_heads + 2.0 * num_kv_heads)
            * w_bytes
            * num_layers
            + hidden_size * head_dim * num_attn_heads * w_bytes * num_layers
            + (
                3.0 * hidden_size * intermediate_size * w_bytes * num_layers
                if intermediate_size > 0
                else 0.0
            )
        )

        # Activation movement bytes per token (coarse approximation).
        self._qkv_act_bytes_per_token = (
            hidden_size * act_bytes * num_layers
            + (num_attn_heads + 2.0 * num_kv_heads) * head_dim * act_bytes * num_layers
            + head_dim * num_attn_heads * act_bytes * num_layers
            + hidden_size * act_bytes * num_layers
        )
        self._ffn_act_bytes_per_token = (
            3.0 * intermediate_size * act_bytes * num_layers
            if intermediate_size > 0
            else 0.0
        )

        # Prefill reads Q/K/V activations from on-device memory.
        self._prefill_attn_act_read_per_token = (
            (num_attn_heads + 2.0 * num_kv_heads) * head_dim * act_bytes * num_layers
        )

        # Decode reads Q from activation memory; K/V reads are from KV cache.
        self._decode_q_read_bytes_per_token = (
            num_attn_heads * head_dim * act_bytes * num_layers
        )

    def _estimate_prefill_perf(
        self: Scheduler, num_tokens: int
    ) -> Tuple[float, float, float]:
        tokens = max(0, int(num_tokens))
        if tokens == 0:
            return 0.0, 0.0, 0.0

        # Causal prefill token-context product.
        context_product = tokens * (tokens + 1) / 2.0
        flops = (
            tokens * self._linear_flops_per_token
            + self._attn_dot_flops_coeff * context_product
        )

        read_bytes = (
            tokens * self._weight_read_bytes_per_token
            + tokens * self._qkv_act_bytes_per_token
            + tokens * self._prefill_attn_act_read_per_token
        )
        write_bytes = (
            tokens * self._kv_cache_bytes_per_token
            + tokens * self._qkv_act_bytes_per_token
            + tokens * self._ffn_act_bytes_per_token
        )
        return flops, read_bytes, write_bytes

    def _estimate_decode_perf(
        self: Scheduler, batch: ScheduleBatch, num_tokens: int
    ) -> Tuple[float, float, float]:
        tokens = max(0, int(num_tokens))
        if tokens == 0:
            return 0.0, 0.0, 0.0

        total_context = float(batch.seq_lens_cpu.sum().item())
        flops = (
            tokens * self._linear_flops_per_token
            + self._attn_dot_flops_coeff * total_context
        )
        read_bytes = (
            tokens * self._weight_read_bytes_per_token
            + tokens * self._qkv_act_bytes_per_token
            + tokens * self._decode_q_read_bytes_per_token
            + total_context * self._kv_cache_bytes_per_token
        )
        write_bytes = (
            tokens * self._kv_cache_bytes_per_token
            + tokens * self._qkv_act_bytes_per_token
            + tokens * self._ffn_act_bytes_per_token
        )
        return flops, read_bytes, write_bytes

    def reset_metrics(self: Scheduler):
        self.forward_ct_decode = 0
        self.num_generated_tokens = 0
        self.spec_num_accepted_tokens = 0
        self.spec_num_forward_ct = 0
        self.spec_total_num_accepted_tokens = 0
        self.spec_total_num_forward_ct = 0

    def report_prefill_stats(
        self: Scheduler,
        prefill_stats: PrefillStats,
        can_run_cuda_graph: bool,
        dp_cooperation_info: Optional[DPCooperationInfo] = None,
    ):
        if (
            not self.is_stats_logging_rank
            and not self.current_scheduler_metrics_enabled
        ):
            return

        now = time.perf_counter()
        gap_latency = now - self.last_prefill_stats_tic
        self.last_prefill_stats_tic = now
        self.last_input_throughput = (
            prefill_stats.log_input_tokens / gap_latency if gap_latency > 0 else 0.0
        )

        pool_stats = self.get_pool_stats()
        token_usage_msg = ", ".join(pool_stats.get_prefill_usage_msg_parts()) + ", "

        self.stats.new_token_ratio = prefill_stats.new_token_ratio
        iter_msg = f" [{self.forward_ct + 1}]" if LOG_FORWARD_ITERS else ""

        msg = (
            f"Prefill batch{iter_msg}, "
            f"#new-seq: {prefill_stats.num_new_seqs}, "
            f"#new-token: {prefill_stats.log_input_tokens}, "
            f"#cached-token: {prefill_stats.log_hit_tokens}, "
            f"{token_usage_msg}"
            f"#running-req: {prefill_stats.num_running_reqs.total}, "
            f"#queue-req: {len(self.waiting_queue)}, "
            f"#pending-token: {prefill_stats.num_pending_tokens}, "
        )

        if self.disaggregation_mode == DisaggregationMode.PREFILL:
            msg += f"#prealloc-req: {len(self.disagg_prefill_bootstrap_queue.queue)}, "
            msg += f"#inflight-req: {len(self.disagg_prefill_inflight_queue)}, "

        if (
            self.server_args.language_only
            and self.server_args.encoder_transfer_backend == "zmq_to_scheduler"
        ):
            msg += f"waiting-image-req: {len(self.mm_receiver.waiting_list)}, "
        graph_backend = defaultdict(
            lambda: "cuda graph",
            {
                "cpu": "cpu graph",
                "npu": "npu graph",
            },
        )

        msg += f"{graph_backend[self.device]}: {can_run_cuda_graph}, "
        msg += f"input throughput (token/s): {self.last_input_throughput:.2f}"

        if self.enable_mfu_metrics and gap_latency > 0:
            flops, _, _ = self._estimate_prefill_perf(prefill_stats.log_input_tokens)
            tflops_per_s = flops / gap_latency / 1e12
            msg += f", est. prefill TFLOPS/s (per GPU): {tflops_per_s:.2f}"

        if self.is_stats_logging_rank:
            logger.info(msg)

        if self.current_scheduler_metrics_enabled:
            self.metrics_collector.increment_prefill_cuda_graph_pass(
                value=can_run_cuda_graph
            )
            self.metrics_collector.increment_realtime_tokens(
                prefill_compute_tokens=prefill_stats.log_input_tokens,
                prefill_cache_tokens=prefill_stats.log_hit_tokens,
                dp_cooperation_info=dp_cooperation_info,
            )
            if self.enable_mfu_metrics:
                flops, read_bytes, write_bytes = self._estimate_prefill_perf(
                    prefill_stats.log_input_tokens
                )
                self.metrics_collector.increment_estimated_perf(
                    num_flops_per_gpu=flops,
                    num_read_bytes_per_gpu=read_bytes,
                    num_write_bytes_per_gpu=write_bytes,
                )

            # Basics
            total_tokens = prefill_stats.log_input_tokens + prefill_stats.log_hit_tokens
            cache_hit_rate = (
                prefill_stats.log_hit_tokens / total_tokens if total_tokens > 0 else 0.0
            )

            self.stats.num_running_reqs = prefill_stats.num_running_reqs
            self.stats.num_running_reqs_offline_batch = 0
            pool_stats.update_scheduler_stats(self.stats)

            priority_enabled = self.enable_priority_scheduling
            self.stats.num_queue_reqs = QueueCount.from_reqs(
                self.waiting_queue, priority_enabled
            )
            self.stats.num_grammar_queue_reqs = len(self.grammar_manager)
            self.stats.cache_hit_rate = cache_hit_rate

            self.stats.max_total_num_tokens = self.max_total_num_tokens

            # Retract
            self.stats.num_retracted_reqs = self.num_retracted_reqs
            self.stats.num_paused_reqs = self.num_paused_reqs
            self.num_retracted_reqs = self.num_paused_reqs = 0

            # PD disaggregation
            if self.disaggregation_mode == DisaggregationMode.PREFILL:
                self.stats.num_prefill_prealloc_queue_reqs = QueueCount.from_reqs(
                    self.disagg_prefill_bootstrap_queue.queue, priority_enabled
                )
                self.stats.num_prefill_inflight_queue_reqs = QueueCount.from_reqs(
                    self.disagg_prefill_inflight_queue, priority_enabled
                )
                self.stats.kv_transfer_speed_gb_s = self.kv_transfer_speed_gb_s
                self.stats.kv_transfer_latency_ms = self.kv_transfer_latency_ms
            elif self.disaggregation_mode == DisaggregationMode.DECODE:
                self.stats.num_decode_prealloc_queue_reqs = QueueCount.from_reqs(
                    self.disagg_decode_prealloc_queue.queue, priority_enabled
                )
                self.stats.num_decode_transfer_queue_reqs = QueueCount.from_reqs(
                    self.disagg_decode_transfer_queue.queue, priority_enabled
                )

            # Others
            self.calculate_utilization()
            self.update_lora_metrics()
            self._log_hicache_stats()
            self.metrics_collector.log_stats(self.stats)
            self._emit_kv_metrics()
        self._publish_kv_events()

    def report_decode_stats(
        self: Scheduler,
        can_run_cuda_graph: bool,
        running_batch: ScheduleBatch = None,
        num_accepted_tokens: int = 0,
    ):
        batch = running_batch or self.running_batch

        # Every-iteration work: realtime token counting + status logger
        if self.current_scheduler_metrics_enabled:
            decode_tokens = batch.batch_size() + num_accepted_tokens
            self.metrics_collector.increment_realtime_tokens(
                # TODO unify this w/ the bumping logic in `Scheduler.num_generated_tokens` accumulator
                decode_tokens=decode_tokens,
                dp_cooperation_info=batch.dp_cooperation_info,
            )
            if self.enable_mfu_metrics:
                flops, read_bytes, write_bytes = self._estimate_decode_perf(
                    batch, decode_tokens
                )
                self.metrics_collector.increment_estimated_perf(
                    num_flops_per_gpu=flops,
                    num_read_bytes_per_gpu=read_bytes,
                    num_write_bytes_per_gpu=write_bytes,
                )
                self._mfu_log_flops += flops
                self._mfu_log_read_bytes += read_bytes
                self._mfu_log_write_bytes += write_bytes

            if x := self.scheduler_status_logger:
                x.maybe_dump(batch, self.waiting_queue)

        # Periodic work: log + heavy metrics at decode_log_interval
        if self.forward_ct_decode % self.server_args.decode_log_interval != 0:
            return
        if (
            not self.is_stats_logging_rank
            and not self.current_scheduler_metrics_enabled
        ):
            return

        gap_latency = time.perf_counter() - self.last_decode_stats_tic
        self.last_decode_stats_tic = time.perf_counter()
        self.last_gen_throughput = self.num_generated_tokens / gap_latency

        self.num_generated_tokens = 0
        num_running_reqs = len(batch.reqs)
        num_running_reqs_offline_batch = 0

        pool_stats = self.get_pool_stats()
        token_usage_msg = ", ".join(pool_stats.get_decode_usage_msg_parts()) + ", "

        if RECORD_STEP_TIME:
            self.step_time_dict[num_running_reqs].append(
                gap_latency / self.server_args.decode_log_interval
            )

        iter_msg = f" [{self.forward_ct}]" if LOG_FORWARD_ITERS else ""
        msg = f"Decode batch{iter_msg}, #running-req: {num_running_reqs}, {token_usage_msg}"

        if self.spec_algorithm.is_none():
            spec_accept_length = 0
            spec_accept_rate = 0
        else:
            spec_accept_length = (
                self.spec_num_accepted_tokens / self.spec_num_forward_ct
            )
            # Calculate acceptance rate: accepted tokens / total draft tokens
            draft_tokens_fallback = (self.server_args.speculative_num_steps or 0) + 1
            num_draft_tokens = (
                self.server_args.speculative_num_draft_tokens or draft_tokens_fallback
            )
            total_draft_tokens = self.spec_num_forward_ct * num_draft_tokens

            spec_accept_rate = (
                self.spec_num_accepted_tokens / total_draft_tokens
                if total_draft_tokens > 0
                else 0
            )
            self.spec_total_num_accepted_tokens += self.spec_num_accepted_tokens
            self.spec_total_num_forward_ct += self.spec_num_forward_ct
            self.spec_num_accepted_tokens = self.spec_num_forward_ct = 0
            msg += f"accept len: {spec_accept_length:.2f}, accept rate: {spec_accept_rate:.2f}, "
        cache_hit_rate = 0.0

        if self.disaggregation_mode == DisaggregationMode.DECODE:
            msg += f"pre-allocated usage: {self.disagg_decode_prealloc_queue.num_tokens_pre_allocated / self.max_total_num_tokens:.2f}, "
            msg += f"#prealloc-req: {len(self.disagg_decode_prealloc_queue.queue)}, "
            msg += f"#transfer-req: {len(self.disagg_decode_transfer_queue.queue)}, "
            msg += f"#retracted-req: {len(self.disagg_decode_prealloc_queue.retracted_queue)}, "

        if (
            self.server_args.language_only
            and self.server_args.encoder_transfer_backend == "zmq_to_scheduler"
        ):
            msg += f"waiting-image-req: {len(self.mm_receiver.waiting_list)}, "

        graph_backend = defaultdict(
            lambda: "cuda graph",
            {
                "cpu": "cpu graph",
                "npu": "npu graph",
            },
        )
        msg += (
            f"{graph_backend[self.device]}: {can_run_cuda_graph}, "
            f"gen throughput (token/s): {self.last_gen_throughput:.2f}, "
            f"#queue-req: {len(self.waiting_queue)}"
        )

        if self.enable_mfu_metrics and gap_latency > 0:
            flops_per_s = self._mfu_log_flops / gap_latency
            read_bytes_per_s = self._mfu_log_read_bytes / gap_latency
            write_bytes_per_s = self._mfu_log_write_bytes / gap_latency
            tflops_per_s = flops_per_s / 1e12
            read_gb_per_s = read_bytes_per_s / 1e9
            write_gb_per_s = write_bytes_per_s / 1e9
            msg += (
                f", est. decode TFLOPS/s (per GPU): {tflops_per_s:.2f}, "
                f"est. read BW (GB/s per GPU): {read_gb_per_s:.2f}, "
                f"est. write BW (GB/s per GPU): {write_gb_per_s:.2f}"
            )
            self._mfu_log_flops = 0.0
            self._mfu_log_read_bytes = 0.0
            self._mfu_log_write_bytes = 0.0

        if self.is_stats_logging_rank:
            logger.info(msg)
        if self.current_scheduler_metrics_enabled:
            priority_enabled = self.enable_priority_scheduling
            # Basics
            self.stats.num_running_reqs = QueueCount.from_reqs(
                batch.reqs, priority_enabled
            )
            self.stats.num_running_reqs_offline_batch = num_running_reqs_offline_batch
            pool_stats.update_scheduler_stats(self.stats)
            self.stats.decode_sum_seq_lens = batch.seq_lens_cpu.sum().item()
            self.stats.gen_throughput = self.last_gen_throughput
            self.stats.num_queue_reqs = QueueCount.from_reqs(
                self.waiting_queue, priority_enabled
            )
            self.stats.num_grammar_queue_reqs = len(self.grammar_manager)
            self.stats.cache_hit_rate = cache_hit_rate

            self.stats.max_total_num_tokens = self.max_total_num_tokens
            self.stats.num_streaming_sessions = self._streaming_session_count()
            self.stats.streaming_session_held_tokens = self._session_held_tokens()

            # Speculative decoding
            self.stats.spec_accept_rate = spec_accept_rate
            self.stats.spec_accept_length = spec_accept_length

            # Retract
            self.stats.num_retracted_reqs = self.num_retracted_reqs
            self.stats.num_paused_reqs = self.num_paused_reqs
            self.num_retracted_reqs = self.num_paused_reqs = 0

            # PD disaggregation
            if self.disaggregation_mode == DisaggregationMode.PREFILL:
                self.stats.num_prefill_prealloc_queue_reqs = QueueCount.from_reqs(
                    self.disagg_prefill_bootstrap_queue.queue, priority_enabled
                )
                self.stats.num_prefill_inflight_queue_reqs = QueueCount.from_reqs(
                    self.disagg_prefill_inflight_queue, priority_enabled
                )
            elif self.disaggregation_mode == DisaggregationMode.DECODE:
                self.stats.num_decode_prealloc_queue_reqs = QueueCount.from_reqs(
                    self.disagg_decode_prealloc_queue.queue, priority_enabled
                )
                self.stats.num_decode_transfer_queue_reqs = QueueCount.from_reqs(
                    self.disagg_decode_transfer_queue.queue, priority_enabled
                )
            running_routing_keys = [r.routing_key for r in batch.reqs]
            waiting_routing_keys = [r.routing_key for r in self.waiting_queue]
            (
                self.stats.num_unique_running_routing_keys,
                self.stats.routing_key_running_req_counts,
            ) = compute_routing_key_stats(running_routing_keys)
            _, self.stats.routing_key_all_req_counts = compute_routing_key_stats(
                running_routing_keys + waiting_routing_keys
            )

            # Others
            self.calculate_utilization()
            self.update_lora_metrics()
            self._log_hicache_stats()
            self.metrics_collector.log_stats(self.stats)
            self._emit_kv_metrics()
        self._publish_kv_events()

    def log_batch_result_stats(
        self: Scheduler,
        batch: ScheduleBatch,
        result: Union[GenerationBatchResult, EmbeddingBatchResult],
    ):
        if not self.enable_metrics:
            return
        if not isinstance(result, GenerationBatchResult):
            return

        if (m := result.expert_distribution_metrics) is not None:
            self.metrics_collector.increment_eplb_balancedness(
                forward_mode=batch.forward_mode.name.lower(),
                balancedness=m.eplb_balancedness.item(),
            )

    def _emit_kv_metrics(self: Scheduler):
        if not self.enable_kv_cache_events:
            return

        kv_metrics = KvMetrics()
        kv_metrics.request_active_slots = self.stats.num_running_reqs.total
        kv_metrics.request_total_slots = self.max_running_requests
        kv_metrics.kv_active_blocks = int(
            self.stats.token_usage * self.max_total_num_tokens
        )
        kv_metrics.kv_total_blocks = self.max_total_num_tokens
        kv_metrics.num_requests_waiting = self.stats.num_queue_reqs.total
        kv_metrics.gpu_cache_usage_perc = self.stats.token_usage
        kv_metrics.gpu_prefix_cache_hit_rate = self.stats.cache_hit_rate
        kv_metrics.data_parallel_rank = self.dp_rank if self.dp_rank is not None else 0

        if not self.send_metrics_from_scheduler.closed:
            self.send_metrics_from_scheduler.send_pyobj(kv_metrics)

    def _publish_kv_events(self: Scheduler):
        if not self.enable_kv_cache_events:
            return

        events = self.tree_cache.take_events()
        if events:
            batch = KVEventBatch(ts=time.time(), events=events)
            self.kv_event_publisher.publish(batch)

    def _log_hicache_stats(self: Scheduler):
        """Populate HiCache host-tier stats on self.stats.

        These are pushed to Prometheus by SchedulerMetricsCollector.log_stats().
        """
        if not self.enable_hierarchical_cache:
            return

        host_pool = getattr(self.tree_cache, "token_to_kv_pool_host", None) or getattr(
            self.tree_cache, "full_kv_pool_host", None
        )
        assert host_pool is not None, "Host pool not found"
        self.stats.hicache_host_used_tokens = (
            host_pool.size - host_pool.available_size()
        )
        self.stats.hicache_host_total_tokens = host_pool.size

    def update_lora_metrics(self: Scheduler):
        """Update LoRA pool metrics for monitoring and autoscaling."""
        if not self.enable_lora:
            return

        try:
            # Get LoRA memory pool stats
            lora_manager = self.tp_worker.model_runner.lora_manager
            if lora_manager is None or lora_manager.memory_pool is None:
                return

            mem_pool = lora_manager.memory_pool
            slots_total = mem_pool.max_loras_per_batch

            # Calculate active adapters from running batch
            # This gives a true measure of current load for autoscaling purposes
            active_lora_ids = set()

            # For PP mode, check all running micro batches
            if hasattr(self, "running_mbs") and self.running_mbs:
                for batch in self.running_mbs:
                    if batch and hasattr(batch, "reqs"):
                        for req in batch.reqs:
                            if hasattr(req, "lora_id") and req.lora_id is not None:
                                active_lora_ids.add(req.lora_id)
            # For normal mode, check running_batch
            elif hasattr(self, "running_batch") and self.running_batch:
                if hasattr(self.running_batch, "reqs"):
                    for req in self.running_batch.reqs:
                        if hasattr(req, "lora_id") and req.lora_id is not None:
                            active_lora_ids.add(req.lora_id)

            # Count active adapters (excluding None for base model)
            slots_used = len(active_lora_ids)
            utilization = slots_used / slots_total if slots_total > 0 else 0.0

            # Update stats
            self.stats.lora_pool_slots_used = slots_used
            self.stats.lora_pool_slots_total = slots_total
            self.stats.lora_pool_utilization = utilization

        except Exception as e:
            logger.warning(f"Failed to update LoRA metrics: {e}")

    def calculate_utilization(self: Scheduler):
        if self.disaggregation_mode == DisaggregationMode.PREFILL:
            self.stats.utilization = -1
        else:
            if (
                self.stats.max_running_requests_under_SLO is not None
                and self.stats.max_running_requests_under_SLO > 0
            ):
                self.stats.utilization = max(
                    self.stats.num_running_reqs.total
                    / self.stats.max_running_requests_under_SLO,
                    self.stats.token_usage / 0.9,
                )

    def _get_num_pending_tokens(self: Scheduler, chunk_deduct: int = 0) -> int:
        """Get the total number of tokens pending prefill.

        This includes tokens from waiting queue requests plus remaining tokens
        from the currently chunked request.

        Args:
            chunk_deduct: extra tokens to subtract from the chunked request's
                remaining count. At batch-scheduling time the current chunk
                has been planned but ``prefix_indices`` does not yet include it,
                so callers pass ``extend_input_len`` here. At load-reporting
                time ``prefix_indices`` is already up-to-date, so the default
                0 is correct.
        """
        num_pending_tokens = sum(req.seqlen for req in self.waiting_queue)
        if self.chunked_req is not None:
            req = self.chunked_req
            num_pending_tokens += req.seqlen - len(req.prefix_indices) - chunk_deduct
        return num_pending_tokens

    def get_loads(self: Scheduler, req: GetLoadsReqInput = None) -> GetLoadsReqOutput:
        """
        Get comprehensive load metrics for /v1/loads endpoint.

        Args:
            req: Request containing include list and optional dp_rank filter

        Returns:
            GetLoadsReqOutput with core metrics and optional detailed sections
        """
        if req is None:
            req = GetLoadsReqInput()

        include = set(req.include) if req.include else {"core"}
        include_all = "all" in include

        num_running_reqs = len(self.running_batch.reqs)

        waiting_queues = [self.waiting_queue]
        if self.disaggregation_mode == DisaggregationMode.PREFILL:
            waiting_queues.append(self.disagg_prefill_bootstrap_queue.queue)
        elif self.disaggregation_mode == DisaggregationMode.DECODE:
            waiting_queues.append(self.disagg_decode_prealloc_queue.queue)
            waiting_queues.append(self.disagg_decode_transfer_queue.queue)
            waiting_queues.append(self.disagg_decode_prealloc_queue.retracted_queue)

        num_waiting_reqs = sum(len(queue) for queue in waiting_queues)
        num_used_tokens, kv_token_usage = self.get_pool_stats().get_kv_token_stats()
        num_total_tokens = num_used_tokens + sum(
            req.seqlen for queue in waiting_queues for req in queue
        )

        memory = None
        if include_all or "memory" in include:
            try:
                memory = MemoryMetrics(
                    weight_gb=round(
                        self.tp_worker.model_runner.weight_load_mem_usage, 3
                    ),
                    kv_cache_gb=round(
                        self.token_to_kv_pool_allocator.get_kvcache().mem_usage, 3
                    ),
                    graph_gb=round(self.tp_worker.model_runner.graph_mem_usage, 3),
                    token_capacity=int(self.max_total_num_tokens),
                )
            except AttributeError as e:
                logger.debug(f"Memory metrics not available: {e}")

        speculative = None
        if include_all or "spec" in include:
            if not self.spec_algorithm.is_none() and self.spec_total_num_forward_ct > 0:
                speculative = SpeculativeMetrics(
                    accept_length=(
                        self.spec_total_num_accepted_tokens
                        / self.spec_total_num_forward_ct
                    ),
                    accept_rate=self.stats.spec_accept_rate,
                )

        lora = None
        if include_all or "lora" in include:
            if hasattr(self, "lora_scheduler") and self.lora_scheduler is not None:
                lora = LoRAMetrics(
                    slots_used=self.stats.lora_pool_slots_used,
                    slots_total=self.stats.lora_pool_slots_total,
                    utilization=self.stats.lora_pool_utilization,
                )

        disaggregation = None
        if include_all or "disagg" in include:
            mode_str = "null"
            prefill_prealloc = 0
            prefill_inflight = 0
            decode_prealloc = 0
            decode_transfer = 0
            decode_retracted = 0

            if self.disaggregation_mode == DisaggregationMode.PREFILL:
                mode_str = "prefill"
                prefill_prealloc = len(self.disagg_prefill_bootstrap_queue.queue)
                prefill_inflight = len(self.disagg_prefill_inflight_queue)
            elif self.disaggregation_mode == DisaggregationMode.DECODE:
                mode_str = "decode"
                decode_prealloc = len(self.disagg_decode_prealloc_queue.queue)
                decode_transfer = len(self.disagg_decode_transfer_queue.queue)
                decode_retracted = len(
                    self.disagg_decode_prealloc_queue.retracted_queue
                )

            disaggregation = DisaggregationMetrics(
                mode=mode_str,
                prefill_prealloc_queue_reqs=prefill_prealloc,
                prefill_inflight_queue_reqs=prefill_inflight,
                decode_prealloc_queue_reqs=decode_prealloc,
                decode_transfer_queue_reqs=decode_transfer,
                decode_retracted_queue_reqs=decode_retracted,
                kv_transfer_speed_gb_s=self.stats.kv_transfer_speed_gb_s,
                kv_transfer_latency_ms=self.stats.kv_transfer_latency_ms,
            )

        queues = None
        if include_all or "queues" in include:
            queues = QueueMetrics(
                waiting=len(self.waiting_queue),
                grammar=self.stats.num_grammar_queue_reqs,
                paused=self.stats.num_paused_reqs,
                retracted=self.stats.num_retracted_reqs,
            )

        return GetLoadsReqOutput(
            dp_rank=self.dp_rank,
            timestamp=time.time(),
            num_running_reqs=num_running_reqs,
            num_waiting_reqs=num_waiting_reqs,
            num_used_tokens=num_used_tokens,
            num_total_tokens=num_total_tokens,
            max_total_num_tokens=self.max_total_num_tokens,
            token_usage=round(kv_token_usage, 4),
            gen_throughput=round(self.stats.gen_throughput, 2),
            cache_hit_rate=round(self.stats.cache_hit_rate, 4),
            utilization=round(self.stats.utilization, 4),
            max_running_requests=self.max_running_requests,
            memory=memory,
            speculative=speculative,
            lora=lora,
            disaggregation=disaggregation,
            queues=queues,
        )

    @contextmanager
    def record_forward_metrics(self: Scheduler, batch: ScheduleBatch):
        if not (self.enable_metrics and ENABLE_METRICS_DEVICE_TIMER):
            yield
            return

        category = "forward_" + batch.forward_mode.name.lower()
        with self.forward_pass_device_timer.wrap(
            metadata=dict(
                category=category,
                dp_cooperation_info=batch.dp_cooperation_info,
            ),
        ):
            yield

    @contextmanager
    def record_bubble_metrics(self: Scheduler, batch: ScheduleBatch):
        if not (self.enable_metrics and ENABLE_METRICS_DEVICE_TIMER):
            yield
            return

        category = "forward_" + batch.forward_mode.name.lower()
        with self.bubble_timer.wrap(
            metadata=dict(
                category=category,
                dp_cooperation_info=batch.dp_cooperation_info,
            ),
        ):
            yield

    def cancel_bubble_timer(self: Scheduler):
        if self.enable_metrics and ENABLE_METRICS_DEVICE_TIMER:
            self.bubble_timer.cancel()
