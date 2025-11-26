from __future__ import annotations

import logging
import time
from collections import defaultdict
from typing import TYPE_CHECKING, List, Optional

from sglang.srt.disaggregation.kv_events import EventPublisherFactory, KVEventBatch
from sglang.srt.disaggregation.utils import DisaggregationMode
from sglang.srt.environ import envs
from sglang.srt.managers.schedule_policy import PrefillAdder
from sglang.srt.managers.scheduler import Req, ScheduleBatch
from sglang.srt.metrics.collector import SchedulerMetricsCollector, SchedulerStats
from sglang.srt.utils import get_bool_env_var

if TYPE_CHECKING:
    from sglang.srt.managers.scheduler import Scheduler

logger = logging.getLogger(__name__)

RECORD_STEP_TIME = get_bool_env_var("SGLANG_RECORD_STEP_TIME")
LOG_FORWARD_ITERS = envs.SGLANG_LOG_FORWARD_ITERS.get()


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
        self.kv_transfer_speed_gb_s: float = 0.0
        self.kv_transfer_latency_ms: float = 0.0
        self.kv_transfer_bootstrap_ms: float = 0.0
        self.kv_transfer_alloc_ms: float = 0.0

        self.stats = SchedulerStats()

        if self.enable_metrics:
            engine_type = "unified"
            labels = {
                "model_name": self.server_args.served_model_name,
                "engine_type": engine_type,
                "tp_rank": tp_rank,
                "pp_rank": pp_rank,
            }
            if dp_rank is not None:
                labels["dp_rank"] = dp_rank
            self.metrics_collector = SchedulerMetricsCollector(labels=labels)

    def init_kv_events(self: Scheduler, kv_events_config: Optional[str]):
        if self.enable_kv_cache_events:
            self.kv_event_publisher = EventPublisherFactory.create(
                kv_events_config, self.attn_dp_rank
            )

    def update_spec_metrics(self: Scheduler, bs: int, num_accepted_tokens: int):
        self.spec_num_accepted_tokens += num_accepted_tokens + bs
        self.spec_num_forward_ct += bs
        self.num_generated_tokens += num_accepted_tokens

    def log_prefill_stats(
        self: Scheduler,
        adder: PrefillAdder,
        can_run_list: List[Req],
        running_bs: int,
        running_bs_offline_batch: int,
    ):
        gap_latency = time.perf_counter() - self.last_prefill_stats_tic
        self.last_prefill_stats_tic = time.perf_counter()
        self.last_input_throughput = self.last_prefill_tokens / gap_latency
        self.last_prefill_tokens = adder.log_input_tokens

        # TODO: generalize this for various memory pools
        if self.is_hybrid:
            (
                full_num_used,
                swa_num_used,
                full_token_usage,
                swa_token_usage,
                _,
                _,
                _,
                _,
            ) = self._get_swa_token_info()
            num_used = max(full_num_used, swa_num_used)
            token_usage = max(full_token_usage, swa_token_usage)
            token_usage_msg = (
                f"full token usage: {full_token_usage:.2f}, "
                f"swa token usage: {swa_token_usage:.2f}, "
            )
        elif self.is_hybrid_gdn:
            (
                full_num_used,
                _,
                full_token_usage,
                mamba_usage,
                _,
                _,
                _,
                _,
            ) = self._get_mamba_token_info()
            num_used = full_num_used
            token_usage = full_token_usage
            token_usage_msg = (
                f"full token usage: {full_token_usage:.2f}, "
                f"mamba usage: {mamba_usage:.2f}, "
            )
        else:
            num_used, token_usage, _, _ = self._get_token_info()
            token_usage_msg = f"token usage: {token_usage:.2f}, "

        self.stats.new_token_ratio = adder.new_token_ratio
        iter_msg = f" [{self.forward_ct + 1}]" if LOG_FORWARD_ITERS else ""

        f = (
            f"Prefill batch{iter_msg}, "
            f"#new-seq: {len(can_run_list)}, "
            f"#new-token: {adder.log_input_tokens}, "
            f"#cached-token: {adder.log_hit_tokens}, "
            f"{token_usage_msg}"
            f"#running-req: {running_bs}, "
            f"#queue-req: {len(self.waiting_queue)}, "
        )

        if self.disaggregation_mode == DisaggregationMode.PREFILL:
            f += f"#prealloc-req: {len(self.disagg_prefill_bootstrap_queue.queue)}, "
            f += f"#inflight-req: {len(self.disagg_prefill_inflight_queue)}, "
            f += f"input throughput (token/s): {self.last_input_throughput:.2f}, "

        logger.info(f)

        if self.enable_metrics:
            # Basics
            total_tokens = adder.log_input_tokens + adder.log_hit_tokens
            cache_hit_rate = (
                adder.log_hit_tokens / total_tokens if total_tokens > 0 else 0.0
            )

            self.stats.num_running_reqs = running_bs
            self.stats.num_running_reqs_offline_batch = running_bs_offline_batch
            self.stats.num_used_tokens = num_used
            self.stats.token_usage = token_usage
            if self.is_hybrid:
                self.stats.swa_token_usage = swa_token_usage
            if self.is_hybrid_gdn:
                self.stats.mamba_usage = mamba_usage
            self.stats.num_queue_reqs = len(self.waiting_queue)
            self.stats.num_grammar_queue_reqs = len(self.grammar_queue)
            self.stats.cache_hit_rate = cache_hit_rate

            # Retract
            self.stats.num_retracted_reqs = self.num_retracted_reqs
            self.stats.num_paused_reqs = self.num_paused_reqs
            self.num_retracted_reqs = self.num_paused_reqs = 0

            # PD disaggregation
            if self.disaggregation_mode == DisaggregationMode.PREFILL:
                self.stats.num_prefill_prealloc_queue_reqs = len(
                    self.disagg_prefill_bootstrap_queue.queue
                )
                self.stats.num_prefill_inflight_queue_reqs = len(
                    self.disagg_prefill_inflight_queue
                )
                self.stats.kv_transfer_speed_gb_s = self.kv_transfer_speed_gb_s
                self.stats.kv_transfer_latency_ms = self.kv_transfer_latency_ms
                self.stats.kv_transfer_bootstrap_ms = self.kv_transfer_bootstrap_ms
                self.stats.kv_transfer_alloc_ms = self.kv_transfer_alloc_ms
            elif self.disaggregation_mode == DisaggregationMode.DECODE:
                self.stats.num_decode_prealloc_queue_reqs = len(
                    self.disagg_decode_prealloc_queue.queue
                )
                self.stats.num_decode_transfer_queue_reqs = len(
                    self.disagg_decode_transfer_queue.queue
                )

            # Others
            self.calculate_utilization()
            self.metrics_collector.log_stats(self.stats)
            self._emit_kv_metrics()
        self._publish_kv_events()

    def log_decode_stats(
        self: Scheduler, can_run_cuda_graph: bool, running_batch: ScheduleBatch = None
    ):
        batch = running_batch or self.running_batch

        gap_latency = time.perf_counter() - self.last_decode_stats_tic
        self.last_decode_stats_tic = time.perf_counter()
        self.last_gen_throughput = self.num_generated_tokens / gap_latency

        self.num_generated_tokens = 0
        num_running_reqs = len(batch.reqs)
        num_running_reqs_offline_batch = 0

        # TODO: generalize this for various memory pools
        if self.is_hybrid:
            (
                full_num_used,
                swa_num_used,
                full_token_usage,
                swa_token_usage,
                _,
                _,
                _,
                _,
            ) = self._get_swa_token_info()
            num_used = max(full_num_used, swa_num_used)
            token_usage = max(full_token_usage, swa_token_usage)
            token_usage_msg = (
                f"#full token: {full_num_used}, "
                f"full token usage: {full_token_usage:.2f}, "
                f"#swa token: {swa_num_used}, "
                f"swa token usage: {swa_token_usage:.2f}, "
            )
        elif self.is_hybrid_gdn:
            (
                full_num_used,
                mamba_used,
                full_token_usage,
                mamba_usage,
                _,
                _,
                _,
                _,
            ) = self._get_mamba_token_info()
            num_used = full_num_used
            token_usage = full_token_usage
            token_usage_msg = (
                f"#full token: {full_num_used}, "
                f"full token usage: {full_token_usage:.2f}, "
                f"mamba num: {mamba_used}, "
                f"mamba usage: {mamba_usage:.2f}, "
            )
        else:
            num_used, token_usage, _, _ = self._get_token_info()
            token_usage_msg = f"#token: {num_used}, token usage: {token_usage:.2f}, "

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

        msg += (
            f"{'cuda graph' if self.device == 'cuda' else 'cpu graph'}: {can_run_cuda_graph}, "
            f"gen throughput (token/s): {self.last_gen_throughput:.2f}, "
            f"#queue-req: {len(self.waiting_queue)}, "
        )

        logger.info(msg)
        if self.enable_metrics:
            # Basics
            self.stats.num_running_reqs = num_running_reqs
            self.stats.num_running_reqs_offline_batch = num_running_reqs_offline_batch
            self.stats.num_used_tokens = num_used
            self.stats.token_usage = token_usage
            if self.is_hybrid:
                self.stats.swa_token_usage = swa_token_usage
            if self.is_hybrid_gdn:
                self.stats.mamba_usage = mamba_usage
            self.stats.gen_throughput = self.last_gen_throughput
            self.stats.num_queue_reqs = len(self.waiting_queue)
            self.stats.num_grammar_queue_reqs = len(self.grammar_queue)
            self.stats.cache_hit_rate = cache_hit_rate

            # Speculative decoding
            self.stats.spec_accept_rate = spec_accept_rate
            self.stats.spec_accept_length = spec_accept_length

            # Retract
            self.stats.num_retracted_reqs = self.num_retracted_reqs
            self.stats.num_paused_reqs = self.num_paused_reqs
            self.num_retracted_reqs = self.num_paused_reqs = 0

            # PD disaggregation
            if self.disaggregation_mode == DisaggregationMode.PREFILL:
                self.stats.num_prefill_prealloc_queue_reqs = len(
                    self.disagg_prefill_bootstrap_queue.queue
                )
                self.stats.num_prefill_inflight_queue_reqs = len(
                    self.disagg_prefill_inflight_queue
                )
            elif self.disaggregation_mode == DisaggregationMode.DECODE:
                self.stats.num_decode_prealloc_queue_reqs = len(
                    self.disagg_decode_prealloc_queue.queue
                )
                self.stats.num_decode_transfer_queue_reqs = len(
                    self.disagg_decode_transfer_queue.queue
                )

            # Others
            self.calculate_utilization()
            self.metrics_collector.log_stats(self.stats)
            self._emit_kv_metrics()
        self._publish_kv_events()

    def _emit_kv_metrics(self: Scheduler):
        if not self.enable_kv_cache_events:
            return

        kv_metrics = KvMetrics()
        kv_metrics.request_active_slots = self.stats.num_running_reqs
        kv_metrics.request_total_slots = self.max_running_requests
        kv_metrics.kv_active_blocks = int(
            self.stats.token_usage * self.max_total_num_tokens
        )
        kv_metrics.kv_total_blocks = self.max_total_num_tokens
        kv_metrics.num_requests_waiting = self.stats.num_queue_reqs
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

    def calculate_utilization(self):
        if self.disaggregation_mode == DisaggregationMode.PREFILL:
            self.stats.utilization = -1
        else:
            if (
                self.stats.max_running_requests_under_SLO is not None
                and self.stats.max_running_requests_under_SLO > 0
            ):
                self.stats.utilization = max(
                    self.stats.num_running_reqs
                    / self.stats.max_running_requests_under_SLO,
                    self.stats.token_usage / 0.9,
                )
