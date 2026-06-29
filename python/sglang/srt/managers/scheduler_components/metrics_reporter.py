from __future__ import annotations

import dataclasses
import logging
import tempfile
import time
from collections import defaultdict
from dataclasses import dataclass
from typing import (
    TYPE_CHECKING,
    List,
    Optional,
    Tuple,
    Union,
)

from sglang.srt.disaggregation.utils import DisaggregationMode
from sglang.srt.environ import envs
from sglang.srt.managers.schedule_batch import ScheduleBatch
from sglang.srt.managers.utils import GenerationBatchResult
from sglang.srt.observability.metrics_collector import (
    DPCooperationInfo,
    QueueCount,
    SchedulerMetricsCollector,
    SchedulerMetricsCollectorContext,
    SchedulerStats,
    compute_routing_key_stats,
)
from sglang.srt.utils.device_timer import DeviceTimer
from sglang.srt.utils.scheduler_status_logger import SchedulerStatusLogger

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import Req
    from sglang.srt.managers.schedule_policy import PrefillAdder
    from sglang.srt.managers.scheduler import Scheduler
    from sglang.srt.managers.utils import EmbeddingBatchResult


logger = logging.getLogger(__name__)


RECORD_STEP_TIME = envs.SGLANG_RECORD_STEP_TIME.get()
LOG_FORWARD_ITERS = envs.SGLANG_LOG_FORWARD_ITERS.get()
ENABLE_METRICS_DEVICE_TIMER = envs.SGLANG_ENABLE_METRICS_DEVICE_TIMER.get()


def _decode_total_seq_lens(batch: ScheduleBatch) -> int:
    """Sync-free sum of seq_lens for decode metrics."""
    if batch.seq_lens_cpu is not None:
        return int(batch.seq_lens_cpu.sum().item())
    return sum(req.seqlen for req in batch.reqs)


@dataclasses.dataclass
class PrefillStats:
    """Stats for logging prefill batch metrics."""

    log_input_tokens: int
    log_hit_tokens: int
    new_token_ratio: float
    num_running_reqs: QueueCount
    num_new_seqs: int  # len(can_run_list)
    reprocessed_log_input_tokens: int = 0
    reprocessed_log_hit_tokens: int = 0
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
            reprocessed_log_input_tokens=adder.reprocessed_log_input_tokens,
            reprocessed_log_hit_tokens=adder.reprocessed_log_hit_tokens,
            new_token_ratio=adder.new_token_ratio,
            num_running_reqs=QueueCount.from_reqs(
                running_reqs, enable_priority_scheduling
            ),
            num_new_seqs=len(adder.can_run_list),
            num_pending_tokens=num_pending_tokens,
        )


@dataclass(kw_only=True)
class SchedulerMetricsReporter:
    scheduler: Scheduler
    tp_rank: int
    pp_rank: int
    dp_rank: Optional[int]
    metrics_collector_context: SchedulerMetricsCollectorContext
    metrics_collector: Optional[SchedulerMetricsCollector]
    num_retracted_reqs: int = 0
    num_paused_reqs: int = 0

    def __post_init__(self) -> None:
        self.enable_metrics = self.metrics_collector_context.enable_metrics
        self.is_stats_logging_rank = (
            self.metrics_collector_context.is_stats_logging_rank
        )
        self.current_scheduler_metrics_enabled = (
            self.metrics_collector_context.current_scheduler_metrics_enabled
        )
        self.enable_kv_cache_events = (
            self.metrics_collector_context.enable_kv_cache_events
        )
        self._init_metrics(self.tp_rank, self.pp_rank, self.dp_rank)
        self._install_device_timer_on_runners()

    def _init_metrics(
        self,
        tp_rank: int,
        pp_rank: int,
        dp_rank: Optional[int],
    ):
        # Basic stats
        self.forward_ct_decode = 0
        self.num_generated_tokens = 0
        self.last_decode_stats_tic = time.perf_counter()
        self.last_prefill_stats_tic = time.perf_counter()
        self.last_gen_throughput: float = 0.0
        self.last_input_throughput: float = 0.0
        self.step_time_dict = defaultdict(list)  # Dict[batch size -> step time]
        self.stats = SchedulerStats()
        self._graph_backend_label = {
            "cpu": "cpu graph",
            "npu": "npu graph",
            "musa": "musa graph",
        }.get(getattr(self.scheduler, "device", ""), "cuda graph")

        # Cumulative spec-decoding counters (reset every decode_log_interval).
        # Each update adds (num_correct_drafts + bs, bs).
        # `*_accept_tokens` = drafts + bonus; `*_correct_drafts` = drafts-only.
        self.spec_num_accept_tokens = 0  # per-log-interval
        self.spec_num_forward_ct = 0
        self.spec_total_num_accept_tokens = 0  # lifetime
        self.spec_total_num_forward_ct = 0

        # For PD disaggregation
        self.kv_transfer_speed_gb_s: float = 0.0
        self.kv_transfer_latency_ms: float = 0.0

        self.enable_mfu_metrics = False

        if self.enable_metrics:
            self.enable_mfu_metrics = self.scheduler.server_args.enable_mfu_metrics
            if self.enable_mfu_metrics:
                self._init_estimated_perf_constants()
                self._mfu_log_flops = 0.0
                self._mfu_log_read_bytes = 0.0
                self._mfu_log_write_bytes = 0.0

        self.fwd_occupancy = float("nan")

        self.forward_pass_device_timer: Optional[DeviceTimer] = None

        if ENABLE_METRICS_DEVICE_TIMER:
            self._device_timer_window_batch_count = 0
            self._device_timer_window_gpu_time = 0.0
            self._device_timer_window_start = None

            def _wrap_execution_reporter(**kwargs):
                self._device_timer_window_gpu_time += kwargs["t"]
                if self.enable_metrics:
                    self.metrics_collector.increment_forward_execution_seconds(**kwargs)

            self.forward_pass_device_timer = DeviceTimer(
                reporter=_wrap_execution_reporter,
            )

        self._init_fpm()

        self.scheduler_status_logger = SchedulerStatusLogger.maybe_create(
            enable_metrics=self.enable_metrics
        )

    def _install_device_timer_on_runners(self):
        if self.forward_pass_device_timer is None:
            return
        timer = self.forward_pass_device_timer
        self.scheduler.tp_worker.model_runner.device_timer = timer
        if self.scheduler.draft_worker is not None:
            dw = getattr(self.scheduler.draft_worker, "draft_worker", None)
            if dw is not None:
                if hasattr(dw, "draft_runner"):
                    dw.draft_runner.device_timer = timer
                for r in getattr(dw, "draft_runner_list", []):
                    r.device_timer = timer

    def _init_fpm(self):
        """Initialize Forward Pass Metrics (FPM) publisher if configured."""
        self.scheduler.enable_fpm = False
        if (
            self.scheduler.server_args.enable_forward_pass_metrics
            and self.scheduler.ps.attn_tp_rank == 0
            and self.scheduler.ps.pp_rank == self.scheduler.ps.pp_size - 1
        ):
            from sglang.srt.observability.forward_pass_metrics import (
                _FpmPublisherThread,
            )

            self.scheduler._fpm_dp_rank = (
                self.scheduler.ps.dp_rank
                if self.scheduler.ps.dp_rank is not None
                else 0
            )
            self.scheduler._fpm_worker_id = (
                self.scheduler.server_args.forward_pass_metrics_worker_id
            )
            base_endpoint = self.scheduler.server_args.forward_pass_metrics_ipc_name
            if base_endpoint is None:
                ipc_path = tempfile.NamedTemporaryFile(delete=False).name
                base_endpoint = f"ipc://{ipc_path}"
                self.scheduler.server_args.forward_pass_metrics_ipc_name = base_endpoint
            endpoint = f"{base_endpoint}.{self.scheduler._fpm_dp_rank}"
            self.scheduler._fpm_publisher = _FpmPublisherThread(
                endpoint,
                worker_id=self.scheduler._fpm_worker_id,
                dp_rank=self.scheduler._fpm_dp_rank,
            )
            self.scheduler._fpm_gpu_time_acc = 0.0

            def _fpm_device_timer_reporter(t, **_kwargs):
                self.scheduler._fpm_gpu_time_acc += t

            if self.forward_pass_device_timer is not None:
                self.forward_pass_device_timer.add_reporter(_fpm_device_timer_reporter)
            else:
                self.forward_pass_device_timer = DeviceTimer(
                    reporter=_fpm_device_timer_reporter,
                )
            self.scheduler._fpm_uses_device_timer = True
            self.scheduler.enable_fpm = True
            logger.info(
                "FPM: ZMQ PUB bound on %s (dp_rank=%d, device_timer=%s)",
                endpoint,
                self.scheduler._fpm_dp_rank,
                self.scheduler._fpm_uses_device_timer,
            )

    def _build_scheduled_request_metrics(self, batch: ScheduleBatch):
        from sglang.srt.observability.forward_pass_metrics import (
            ScheduledRequestMetrics,
            WelfordAccumulator,
        )

        num_prefill_requests = 0
        sum_prefill_tokens = 0
        sum_prefill_kv_tokens = 0
        prefill_lengths = WelfordAccumulator()

        if batch.forward_mode.is_mixed():
            decode_req_ids = {id(req) for req in batch.decoding_reqs or []}
            prefill_reqs = [req for req in batch.reqs if id(req) not in decode_req_ids]
        elif batch.forward_mode.is_extend():
            prefill_reqs = batch.reqs
        else:
            prefill_reqs = []

        if prefill_reqs:
            stats = batch.prefill_stats
            for req in prefill_reqs:
                prefill_lengths.add(len(req.origin_input_ids))
            num_prefill_requests = stats.num_new_seqs if stats else len(prefill_reqs)
            sum_prefill_tokens = stats.log_input_tokens if stats else 0
            sum_prefill_kv_tokens = sum(len(req.prefix_indices) for req in prefill_reqs)

        decode_kv = WelfordAccumulator()
        if batch.forward_mode.is_mixed():
            for req in batch.decoding_reqs or []:
                decode_kv.add(req.seqlen)
        elif batch.forward_mode.is_decode():
            for sl in batch.seq_lens_cpu:
                decode_kv.add(int(sl))

        return ScheduledRequestMetrics(
            num_prefill_requests=num_prefill_requests,
            sum_prefill_tokens=sum_prefill_tokens,
            var_prefill_length=prefill_lengths.variance(),
            sum_prefill_kv_tokens=sum_prefill_kv_tokens,
            num_decode_requests=decode_kv.count,
            sum_decode_kv_tokens=decode_kv.total,
            var_decode_kv_tokens=decode_kv.variance(),
        )

    def _build_queued_request_metrics(self):
        from sglang.srt.observability.forward_pass_metrics import (
            QueuedRequestMetrics,
            WelfordAccumulator,
        )

        prefill_q = WelfordAccumulator()
        decode_q = WelfordAccumulator()
        if self.scheduler.disaggregation_mode == DisaggregationMode.PREFILL:
            for req in self.scheduler.disagg_prefill_bootstrap_queue.queue:
                prefill_q.add(len(req.origin_input_ids))
        elif self.scheduler.disaggregation_mode == DisaggregationMode.DECODE:
            for req in self.scheduler.disagg_decode_prealloc_queue.queue:
                decode_q.add(req.seqlen)
            for req in self.scheduler.disagg_decode_transfer_queue.queue:
                decode_q.add(req.seqlen)
        else:
            for req in self.scheduler.waiting_queue:
                if len(req.output_ids) > 0:
                    decode_q.add(req.seqlen)
                else:
                    prefill_q.add(len(req.origin_input_ids))

        return QueuedRequestMetrics(
            num_prefill_requests=prefill_q.count,
            sum_prefill_tokens=prefill_q.total,
            var_prefill_length=prefill_q.variance(),
            num_decode_requests=decode_q.count,
            sum_decode_kv_tokens=decode_q.total,
            var_decode_kv_tokens=decode_q.variance(),
        )

    def _active_spec_config_snapshot(self) -> dict[str, int]:
        """Read the currently active speculative decoding configuration."""
        draft_worker = self.scheduler.draft_worker
        if draft_worker is None:
            return {
                "num_steps": 0,
                "num_draft_tokens": 0,
            }

        # Fallback to server_args if draft_worker does not have the attributes.
        server_args = self.scheduler.server_args
        num_steps = getattr(
            draft_worker, "speculative_num_steps", server_args.speculative_num_steps
        )
        num_draft_tokens = getattr(
            draft_worker,
            "speculative_num_draft_tokens",
            server_args.speculative_num_draft_tokens,
        )

        return {
            "num_steps": num_steps or 0,
            "num_draft_tokens": num_draft_tokens or 0,
        }

    def update_spec_metrics(self, bs: int, num_correct_drafts: int):
        self.spec_num_accept_tokens += num_correct_drafts + bs
        self.spec_num_forward_ct += bs

        # Bonus tokens updated elsewhere
        self.num_generated_tokens += num_correct_drafts

    def _init_estimated_perf_constants(self) -> None:
        model_config = self.scheduler.model_config
        hf_text_config = model_config.hf_text_config

        hidden_size = float(model_config.hidden_size)
        num_layers = float(getattr(model_config, "num_attention_layers", 0))
        head_dim = float(getattr(model_config, "head_dim", 0))
        num_attn_heads = float(
            model_config.get_num_attention_heads(self.scheduler.ps.tp_size)
        )
        num_kv_heads = float(model_config.get_num_kv_heads(self.scheduler.ps.tp_size))
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

    def _estimate_prefill_perf(self, batch) -> Tuple[float, float, float]:
        if batch is None or batch.extend_lens is None:
            return 0.0, 0.0, 0.0
        tokens = max(0, int(sum(batch.extend_lens)))
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
        self, batch: ScheduleBatch, num_tokens: int
    ) -> Tuple[float, float, float]:
        tokens = max(0, int(num_tokens))
        if tokens == 0:
            return 0.0, 0.0, 0.0

        total_context = float(_decode_total_seq_lens(batch))
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

    def _prefill_sol_suffix(self, batch, elapsed_s: float) -> str:
        """Hook: model-specific speed-of-light % suffix for the prefill log line.
        ``batch`` carries the per-request extend/prefix lengths a subclass needs
        for an exact attention pair-count. No model arch here, so returns "";
        a subclass may override it."""
        return ""

    def _decode_sol_suffix(self, batch, elapsed_s: float) -> str:
        """Hook: model-specific speed-of-light % suffix for the decode log line.
        ``elapsed_s`` is per-iteration. No model arch here, so returns "";
        a subclass may override it."""
        return ""

    def reset_metrics(self):
        self.forward_ct_decode = 0
        self.num_generated_tokens = 0
        self.spec_num_accept_tokens = 0
        self.spec_num_forward_ct = 0
        self.spec_total_num_accept_tokens = 0
        self.spec_total_num_forward_ct = 0

    def report_prefill_stats(
        self,
        batch: Optional[ScheduleBatch],
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

        pool_stats = self.scheduler.pool_stats_observer.get_pool_stats()
        token_usage_msg = ", ".join(pool_stats.get_prefill_usage_msg_parts()) + ", "

        self.stats.new_token_ratio = prefill_stats.new_token_ratio
        batch_iter = (
            batch.forward_iter
            if batch is not None and batch.forward_iter is not None
            else self.scheduler.forward_ct
        )
        iter_msg = f" [{batch_iter}]" if LOG_FORWARD_ITERS else ""

        msg = (
            f"Prefill batch{iter_msg}, "
            f"#new-seq: {prefill_stats.num_new_seqs}, "
            f"#new-token: {prefill_stats.log_input_tokens}, "
            f"#cached-token: {prefill_stats.log_hit_tokens}, "
            f"{token_usage_msg}"
            f"#running-req: {prefill_stats.num_running_reqs.total}, "
            f"#queue-req: {len(self.scheduler.waiting_queue)}, "
            f"#pending-token: {prefill_stats.num_pending_tokens}, "
        )

        if self.scheduler.disaggregation_mode == DisaggregationMode.PREFILL:
            msg += f"#bootstrap-req: {len(self.scheduler.disagg_prefill_bootstrap_queue.queue)}, "
            msg += (
                f"#inflight-req: {len(self.scheduler.disagg_prefill_inflight_queue)}, "
            )

        if (
            self.scheduler.server_args.language_only
            and self.scheduler.server_args.encoder_transfer_backend
            == "zmq_to_scheduler"
        ):
            msg += (
                f"waiting-image-req: {len(self.scheduler.mm_receiver.waiting_list)}, "
            )

        msg += f"{self._graph_backend_label}: {can_run_cuda_graph}, "
        msg += f"input throughput (token/s): {self.last_input_throughput:.2f}"

        if self.enable_mfu_metrics and gap_latency > 0:
            # Prefer the SoL suffix when it carries content: it scores FLOPs against
            # each forward's actual GPU span (device timer). The wall-clock est.
            # TFLOPS below divides FLOPs by gap_latency -- the inter-log interval on
            # the async scheduler loop, which is decoupled from this forward's
            # execution -- so it disagrees with the SoL. Omit it when SoL is present.
            sol_suffix = self._prefill_sol_suffix(batch, gap_latency)
            if sol_suffix:
                msg += sol_suffix
            else:
                flops, _, _ = self._estimate_prefill_perf(batch)
                tflops_per_s = flops / gap_latency / 1e12
                msg += f", est. prefill TFLOPS/s (per GPU): {tflops_per_s:.2f}"

        if ENABLE_METRICS_DEVICE_TIMER:
            msg += f", fwd occupancy: {self.fwd_occupancy:.2f}%"

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
                flops, read_bytes, write_bytes = self._estimate_prefill_perf(batch)
                self.metrics_collector.increment_estimated_perf(
                    num_flops_per_gpu=flops,
                    num_read_bytes_per_gpu=read_bytes,
                    num_write_bytes_per_gpu=write_bytes,
                )

            priority_enabled = self.scheduler.enable_priority_scheduling
            effective_input_tokens = (
                prefill_stats.log_input_tokens
                - prefill_stats.reprocessed_log_input_tokens
            )
            effective_hit_tokens = (
                prefill_stats.log_hit_tokens - prefill_stats.reprocessed_log_hit_tokens
            )
            total_tokens = effective_input_tokens + effective_hit_tokens
            cache_hit_rate = (
                effective_hit_tokens / total_tokens if total_tokens > 0 else 0.0
            )

            # Basics
            self.stats.num_running_reqs = prefill_stats.num_running_reqs
            self.stats.num_queue_reqs = QueueCount.from_reqs(
                self.scheduler.waiting_queue, priority_enabled
            )
            self.stats.num_grammar_queue_reqs = len(self.scheduler.grammar_manager)
            self.stats.cache_hit_rate = cache_hit_rate

            # Memory pool usage ratios / Absolute token counts
            pool_stats.update_scheduler_stats(self.stats)

            # Retract
            self.stats.num_retracted_reqs = self.num_retracted_reqs
            self.stats.num_paused_reqs = self.num_paused_reqs
            self.num_retracted_reqs = self.num_paused_reqs = 0

            # PD disaggregation
            if self.scheduler.disaggregation_mode == DisaggregationMode.PREFILL:
                self.stats.num_prefill_bootstrap_queue_reqs = QueueCount.from_reqs(
                    self.scheduler.disagg_prefill_bootstrap_queue.queue,
                    priority_enabled,
                )
                self.stats.num_prefill_inflight_queue_reqs = QueueCount.from_reqs(
                    self.scheduler.disagg_prefill_inflight_queue, priority_enabled
                )
                self.stats.kv_transfer_speed_gb_s = self.kv_transfer_speed_gb_s
                self.stats.kv_transfer_latency_ms = self.kv_transfer_latency_ms
            elif self.scheduler.disaggregation_mode == DisaggregationMode.DECODE:
                self.stats.num_decode_prealloc_queue_reqs = QueueCount.from_reqs(
                    self.scheduler.disagg_decode_prealloc_queue.queue, priority_enabled
                )
                self.stats.num_decode_transfer_queue_reqs = QueueCount.from_reqs(
                    self.scheduler.disagg_decode_transfer_queue.queue, priority_enabled
                )

            # Utilization / LoRA / HiCache
            self._calculate_utilization()
            self.stats.fwd_occupancy = self.fwd_occupancy
            self._update_lora_metrics()
            self._log_hicache_stats()
            self.metrics_collector.log_stats(self.stats)
            self.scheduler.kv_events_publisher.emit_kv_metrics()
        self.scheduler.kv_events_publisher.publish_kv_events()

    def report_decode_stats(
        self,
        can_run_cuda_graph: bool,
        running_batch: ScheduleBatch = None,
        num_correct_drafts: int = 0,
    ):
        batch = running_batch or self.scheduler.running_batch

        # Every-iteration work: realtime token counting + status logger
        if self.current_scheduler_metrics_enabled:
            decode_tokens = batch.batch_size() + num_correct_drafts
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
                x.maybe_dump(batch, self.scheduler.waiting_queue)

        # Periodic work: log + heavy metrics at decode_log_interval
        if self.forward_ct_decode % self.scheduler.server_args.decode_log_interval != 0:
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

        pool_stats = self.scheduler.pool_stats_observer.get_pool_stats()
        token_usage_msg = ", ".join(pool_stats.get_decode_usage_msg_parts()) + ", "

        if RECORD_STEP_TIME:
            self.step_time_dict[num_running_reqs].append(
                gap_latency / self.scheduler.server_args.decode_log_interval
            )

        batch_iter = (
            batch.forward_iter
            if batch is not None and batch.forward_iter is not None
            else self.scheduler.forward_ct
        )
        iter_msg = f" [{batch_iter}]" if LOG_FORWARD_ITERS else ""
        msg = f"Decode batch{iter_msg}, #running-req: {num_running_reqs}, {token_usage_msg}"

        spec_num_steps = 0
        spec_num_draft_tokens = 0
        if self.scheduler.spec_algorithm.is_none():
            spec_accept_length = 0
            spec_accept_rate = 0
        else:
            spec_accept_length = self.spec_num_accept_tokens / self.spec_num_forward_ct
            num_correct_drafts = self.spec_num_accept_tokens - self.spec_num_forward_ct
            if self.scheduler.server_args.speculative_num_draft_tokens:
                draft_per_round = (
                    self.scheduler.server_args.speculative_num_draft_tokens - 1
                )
            else:
                draft_per_round = self.scheduler.server_args.speculative_num_steps or 0
            total_draft_tokens = self.spec_num_forward_ct * draft_per_round
            spec_accept_rate = (
                num_correct_drafts / total_draft_tokens if total_draft_tokens > 0 else 0
            )
            self.spec_total_num_accept_tokens += self.spec_num_accept_tokens
            self.spec_total_num_forward_ct += self.spec_num_forward_ct
            self.spec_num_accept_tokens = self.spec_num_forward_ct = 0
            msg += f"accept len: {spec_accept_length:.2f}, accept rate: {spec_accept_rate:.2f}, "

            if self.current_scheduler_metrics_enabled:
                spec_snapshot = self._active_spec_config_snapshot()
                spec_num_steps = spec_snapshot["num_steps"]
                spec_num_draft_tokens = spec_snapshot["num_draft_tokens"]

        cache_hit_rate = 0.0

        if self.scheduler.disaggregation_mode == DisaggregationMode.DECODE:
            msg += f"pre-allocated usage: {self.scheduler.disagg_decode_prealloc_queue.num_tokens_pre_allocated / self.scheduler.max_total_num_tokens:.2f}, "
            msg += f"#prealloc-req: {len(self.scheduler.disagg_decode_prealloc_queue.queue)}, "
            msg += f"#transfer-req: {len(self.scheduler.disagg_decode_transfer_queue.queue)}, "
            msg += f"#retracted-req: {len(self.scheduler.disagg_decode_prealloc_queue.retracted_queue)}, "

        if (
            self.scheduler.server_args.language_only
            and self.scheduler.server_args.encoder_transfer_backend
            == "zmq_to_scheduler"
        ):
            msg += (
                f"waiting-image-req: {len(self.scheduler.mm_receiver.waiting_list)}, "
            )

        msg += (
            f"{self._graph_backend_label}: {can_run_cuda_graph}, "
            f"gen throughput (token/s): {self.last_gen_throughput:.2f}, "
            f"#queue-req: {len(self.scheduler.waiting_queue)}"
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
            msg += self._decode_sol_suffix(
                batch,
                gap_latency / max(1, self.scheduler.server_args.decode_log_interval),
            )
            self._mfu_log_flops = 0.0
            self._mfu_log_read_bytes = 0.0
            self._mfu_log_write_bytes = 0.0

        if ENABLE_METRICS_DEVICE_TIMER:
            msg += f", fwd occupancy: {self.fwd_occupancy:.2f}%"

        if self.is_stats_logging_rank:
            logger.info(msg)
        if self.current_scheduler_metrics_enabled:
            priority_enabled = self.scheduler.enable_priority_scheduling

            # Basics
            self.stats.num_running_reqs = QueueCount.from_reqs(
                batch.reqs, priority_enabled
            )
            self.stats.num_queue_reqs = QueueCount.from_reqs(
                self.scheduler.waiting_queue, priority_enabled
            )
            self.stats.num_grammar_queue_reqs = len(self.scheduler.grammar_manager)
            self.stats.gen_throughput = self.last_gen_throughput
            self.stats.cache_hit_rate = cache_hit_rate
            self.stats.decode_sum_seq_lens = _decode_total_seq_lens(batch)

            # Memory pool usage ratios / Absolute token counts
            pool_stats.update_scheduler_stats(self.stats)

            # Speculative decoding
            self.stats.spec_accept_length = spec_accept_length
            self.stats.spec_accept_rate = spec_accept_rate
            self.stats.spec_num_steps = spec_num_steps
            self.stats.spec_num_draft_tokens = spec_num_draft_tokens

            # Retract
            self.stats.num_retracted_reqs = self.num_retracted_reqs
            self.stats.num_paused_reqs = self.num_paused_reqs
            self.num_retracted_reqs = self.num_paused_reqs = 0

            # PD disaggregation
            if self.scheduler.disaggregation_mode == DisaggregationMode.PREFILL:
                self.stats.num_prefill_bootstrap_queue_reqs = QueueCount.from_reqs(
                    self.scheduler.disagg_prefill_bootstrap_queue.queue,
                    priority_enabled,
                )
                self.stats.num_prefill_inflight_queue_reqs = QueueCount.from_reqs(
                    self.scheduler.disagg_prefill_inflight_queue, priority_enabled
                )
            elif self.scheduler.disaggregation_mode == DisaggregationMode.DECODE:
                self.stats.num_decode_prealloc_queue_reqs = QueueCount.from_reqs(
                    self.scheduler.disagg_decode_prealloc_queue.queue, priority_enabled
                )
                self.stats.num_decode_transfer_queue_reqs = QueueCount.from_reqs(
                    self.scheduler.disagg_decode_transfer_queue.queue, priority_enabled
                )

            # Streaming session metrics
            self.stats.num_streaming_sessions = (
                self.scheduler.pool_stats_observer.streaming_session_count()
            )
            self.stats.streaming_session_held_tokens = (
                self.scheduler.pool_stats_observer.session_held_tokens()
            )

            # Routing key metrics
            # (to reduce the overhead, we only compute this when all requests have routing_key)
            if all(r.routing_key is not None for r in batch.reqs):
                running_routing_keys = [r.routing_key for r in batch.reqs]
                waiting_routing_keys = [
                    r.routing_key for r in self.scheduler.waiting_queue
                ]
                (
                    self.stats.num_unique_running_routing_keys,
                    self.stats.routing_key_running_req_counts,
                ) = compute_routing_key_stats(running_routing_keys)
                _, self.stats.routing_key_all_req_counts = compute_routing_key_stats(
                    running_routing_keys + waiting_routing_keys
                )

            # Utilization / LoRA / HiCache
            self._calculate_utilization()
            self.stats.fwd_occupancy = self.fwd_occupancy
            self._update_lora_metrics()
            self._log_hicache_stats()
            self.metrics_collector.log_stats(self.stats)
            self.scheduler.kv_events_publisher.emit_kv_metrics()
        self.scheduler.kv_events_publisher.publish_kv_events()

    def log_batch_result_stats(
        self,
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

    def _emit_forward_pass_metrics(
        self,
        batch: ScheduleBatch,
        result=None,
    ):
        """Emit per-iteration ForwardPassMetrics over ZMQ PUB.

        Prefers GPU-accurate timing from DeviceTimer (which wraps
        model_runner.forward / cuda_graph.replay via PR #24197).
        Falls back to monotonic clock when DeviceTimer is not enabled.
        """
        if not self.scheduler.enable_fpm:
            return

        from sglang.srt.observability.forward_pass_metrics import (
            ForwardPassMetrics,
        )

        if self.scheduler._fpm_uses_device_timer:
            self.forward_pass_device_timer._report()
            wall_time = self.scheduler._fpm_gpu_time_acc
            self.scheduler._fpm_gpu_time_acc = 0.0
            if wall_time == 0.0:
                return
        else:
            wall_time = max(0.0, time.monotonic() - batch.fpm_start_time)

        fpm = ForwardPassMetrics(
            worker_id=self.scheduler._fpm_worker_id,
            dp_rank=self.scheduler._fpm_dp_rank,
            wall_time=wall_time,
            scheduled_requests=self._build_scheduled_request_metrics(batch),
            queued_requests=self._build_queued_request_metrics(),
        )
        self.scheduler._fpm_publisher.publish(fpm)

    def _shutdown_fpm(self):
        """Shut down the FPM publisher thread."""
        if self.scheduler.enable_fpm:
            self.scheduler._fpm_publisher.shutdown()

    def _log_hicache_stats(self):
        """Populate HiCache host-tier stats on self.stats.

        These are pushed to Prometheus by SchedulerMetricsCollector.log_stats().
        """
        if not self.scheduler.enable_hierarchical_cache:
            return

        host_pool = getattr(
            self.scheduler.tree_cache, "token_to_kv_pool_host", None
        ) or getattr(self.scheduler.tree_cache, "full_kv_pool_host", None)
        assert host_pool is not None, "Host pool not found"
        self.stats.hicache_host_used_tokens = (
            host_pool.size - host_pool.available_size()
        )
        self.stats.hicache_host_total_tokens = host_pool.size

    def _update_lora_metrics(self):
        """Update LoRA pool metrics for monitoring and autoscaling."""
        if not self.scheduler.enable_lora:
            return

        try:
            # Get LoRA memory pool stats
            lora_manager = self.scheduler.tp_worker.model_runner.lora_manager
            if lora_manager is None or lora_manager.memory_pool is None:
                return

            mem_pool = lora_manager.memory_pool
            slots_total = mem_pool.max_loras_per_batch

            # Calculate active adapters from running batch
            # This gives a true measure of current load for autoscaling purposes
            active_lora_ids = set()

            # For PP mode, check all running micro batches
            if self.scheduler.server_args.pp_size > 1:
                for batch in self.scheduler.running_mbs:
                    if batch and hasattr(batch, "reqs"):
                        for req in batch.reqs:
                            if hasattr(req, "lora_id") and req.lora_id is not None:
                                active_lora_ids.add(req.lora_id)
            # For normal mode, check running_batch
            elif self.scheduler.running_batch:
                if hasattr(self.scheduler.running_batch, "reqs"):
                    for req in self.scheduler.running_batch.reqs:
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

    def _calculate_utilization(self):
        if self.scheduler.disaggregation_mode == DisaggregationMode.PREFILL:
            self.stats.utilization = -1
        else:
            # Prefer the SLO-aware capacity if it has been populated (non-OSS
            # or future OSS paths). Fall back to max_running_requests so that
            # sglang:utilization reflects real load in the OSS codepath.
            # See: https://github.com/sgl-project/sglang/issues/22713
            max_under_slo = getattr(
                self.scheduler, "max_running_requests_under_SLO", None
            )
            capacity = (
                max_under_slo
                if (max_under_slo is not None and max_under_slo > 0)
                else getattr(self.scheduler, "max_running_requests", None)
            )
            if capacity is not None and capacity > 0:
                self.stats.utilization = max(
                    self.stats.num_running_reqs.total / capacity,
                    self.stats.token_usage / 0.9,
                )

    def update_device_timer(self):
        if not ENABLE_METRICS_DEVICE_TIMER:
            return
        self.forward_pass_device_timer._report()
        now = time.perf_counter()
        if self._device_timer_window_batch_count == 0:
            # Window start: keep the last published value instead of NaN-ing
            # the gauge. Readers sample it asynchronously, and the window
            # boundary can phase-lock with the decode-log cadence, turning a
            # one-tick NaN into NaN on every log line. NaN is published only
            # when truly stale (reset_device_timer_window after idle).
            self._device_timer_window_start = now
            self._device_timer_window_gpu_time = 0.0
        else:
            cpu_time = now - self._device_timer_window_start
            if cpu_time > 0:
                self.fwd_occupancy = min(
                    self._device_timer_window_gpu_time / cpu_time * 100, 100
                )
        self._device_timer_window_batch_count += 1
        if (
            self._device_timer_window_batch_count
            >= self.scheduler.server_args.decode_log_interval
        ):
            self._device_timer_window_batch_count = 0

    def reset_device_timer_window(self):
        if ENABLE_METRICS_DEVICE_TIMER:
            self._device_timer_window_batch_count = 0
            self.fwd_occupancy = float("nan")

    def _maybe_log_idle_metrics(self):
        """Collect and log metrics every 30 seconds during idle."""
        if (
            not self.current_scheduler_metrics_enabled
            or time.perf_counter() <= self.metrics_collector.last_log_time + 30
        ):
            return

        self.scheduler.pool_stats_observer.get_pool_stats().update_scheduler_stats(
            self.stats
        )
        self.stats.num_streaming_sessions = (
            self.scheduler.pool_stats_observer.streaming_session_count()
        )
        self.stats.streaming_session_held_tokens = (
            self.scheduler.pool_stats_observer.session_held_tokens()
        )

        priority_enabled = self.scheduler.enable_priority_scheduling
        self.stats.num_running_reqs = QueueCount.from_reqs(
            self.scheduler.running_batch.reqs, priority_enabled
        )
        self.stats.gen_throughput = 0
        self.stats.num_queue_reqs = QueueCount.from_reqs(
            self.scheduler.waiting_queue, priority_enabled
        )
        self.stats.num_grammar_queue_reqs = len(self.scheduler.grammar_manager)
        if self.scheduler.disaggregation_mode == DisaggregationMode.PREFILL:
            self.stats.num_prefill_bootstrap_queue_reqs = QueueCount.from_reqs(
                self.scheduler.disagg_prefill_bootstrap_queue.queue, priority_enabled
            )
            self.stats.num_prefill_inflight_queue_reqs = QueueCount.from_reqs(
                self.scheduler.disagg_prefill_inflight_queue, priority_enabled
            )
        if self.scheduler.disaggregation_mode == DisaggregationMode.DECODE:
            self.stats.num_decode_prealloc_queue_reqs = QueueCount.from_reqs(
                self.scheduler.disagg_decode_prealloc_queue.queue, priority_enabled
            )
            self.stats.num_decode_transfer_queue_reqs = QueueCount.from_reqs(
                self.scheduler.disagg_decode_transfer_queue.queue, priority_enabled
            )
        self.metrics_collector.log_stats(self.stats)
