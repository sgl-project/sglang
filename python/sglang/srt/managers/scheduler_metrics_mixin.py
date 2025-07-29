import logging
import time
from collections import defaultdict
from typing import List, Optional

from sglang.srt.disaggregation.kv_events import EventPublisherFactory, KVEventBatch
from sglang.srt.disaggregation.utils import DisaggregationMode
from sglang.srt.managers.schedule_policy import PrefillAdder
from sglang.srt.managers.scheduler import Req, ScheduleBatch
from sglang.srt.metrics.collector import SchedulerMetricsCollector, SchedulerStats
from sglang.srt.utils import get_bool_env_var

logger = logging.getLogger(__name__)

RECORD_STEP_TIME = get_bool_env_var("SGLANG_RECORD_STEP_TIME")


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
    def init_metrics(self, tp_rank: int, pp_rank: int, dp_rank: Optional[int]):
        self.last_gen_throughput: float = 0.0
        self.last_input_throughput: float = 0.0
        self.step_time_dict = defaultdict(list)  # Dict[batch size -> step time]
        self.spec_num_total_accepted_tokens = 0
        self.spec_num_total_forward_ct = 0
        self.cum_spec_accept_length = 0
        self.cum_spec_accept_count = 0
        self.total_retracted_reqs = 0
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

    def init_kv_events(self, kv_events_config: Optional[str]):
        if self.enable_kv_cache_events:
            self.kv_event_publisher = EventPublisherFactory.create(
                kv_events_config, self.attn_dp_rank
            )

    def log_prefill_stats(
        self,
        adder: PrefillAdder,
        can_run_list: List[Req],
        running_bs: int,
    ):
        gap_latency = time.perf_counter() - self.last_prefill_stats_tic
        self.last_prefill_stats_tic = time.perf_counter()
        self.last_input_throughput = self.last_prefill_tokens / gap_latency
        self.last_prefill_tokens = adder.log_input_tokens

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
            token_msg = (
                f"full token usage: {full_token_usage:.2f}, "
                f"swa token usage: {swa_token_usage:.2f}, "
            )
        else:
            num_used, token_usage, _, _ = self._get_token_info()
            token_msg = f"token usage: {token_usage:.2f}, "

        num_new_seq = len(can_run_list)
        f = (
            f"Prefill batch. "
            f"#new-seq: {num_new_seq}, "
            f"#new-token: {adder.log_input_tokens}, "
            f"#cached-token: {adder.log_hit_tokens}, "
            f"{token_msg}"
        )

        if self.disaggregation_mode == DisaggregationMode.PREFILL:
            f += f"#unbootstrapped-req: {len(self.disagg_prefill_bootstrap_queue.queue)}, "
            f += f"#queue-req: {len(self.waiting_queue)}, "
            f += f"#transferring-req: {len(self.disagg_prefill_inflight_queue)}, "
            f += f"input throughput (token/s): {self.last_input_throughput:.2f}, "
        else:
            f += f"#running-req: {running_bs}, "
            f += f"#queue-req: {len(self.waiting_queue)}, "

        logger.info(f)

        if self.enable_metrics:
            total_tokens = adder.log_input_tokens + adder.log_hit_tokens

            cache_hit_rate = (
                adder.log_hit_tokens / total_tokens if total_tokens > 0 else 0.0
            )
            self.stats.num_running_reqs = running_bs
            self.stats.num_used_tokens = num_used
            self.stats.token_usage = round(token_usage, 2)
            self.stats.num_queue_reqs = len(self.waiting_queue)
            self.stats.cache_hit_rate = cache_hit_rate

            total_queue_latency = 0
            for req in can_run_list:
                total_queue_latency += req.queue_time_end - req.queue_time_start
            self.stats.avg_request_queue_latency = total_queue_latency / num_new_seq

            self.metrics_collector.log_stats(self.stats)
            self._emit_kv_metrics()
        self._publish_kv_events()

    def log_decode_stats(
        self, can_run_cuda_graph: bool, running_batch: ScheduleBatch = None
    ):
        batch = running_batch or self.running_batch

        gap_latency = time.perf_counter() - self.last_decode_stats_tic
        self.last_decode_stats_tic = time.perf_counter()
        self.last_gen_throughput = self.num_generated_tokens / gap_latency
        self.num_generated_tokens = 0
        num_running_reqs = len(batch.reqs)
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
            token_msg = (
                f"#full token: {full_num_used}, "
                f"full token usage: {full_token_usage:.2f}, "
                f"#swa token: {swa_num_used}, "
                f"swa token usage: {swa_token_usage:.2f}, "
            )
        else:
            num_used, token_usage, _, _ = self._get_token_info()
            token_msg = f"#token: {num_used}, " f"token usage: {token_usage:.2f}, "

        if RECORD_STEP_TIME:
            self.step_time_dict[num_running_reqs].append(
                gap_latency / self.server_args.decode_log_interval
            )

        msg = f"Decode batch. #running-req: {num_running_reqs}, {token_msg}"

        if self.spec_algorithm.is_none():
            spec_accept_length = 0
        else:
            spec_accept_length = (
                self.spec_num_total_accepted_tokens / self.spec_num_total_forward_ct
            )
            self.cum_spec_accept_length += self.spec_num_total_accepted_tokens
            self.cum_spec_accept_count += self.spec_num_total_forward_ct
            self.spec_num_total_accepted_tokens = self.spec_num_total_forward_ct = 0
            msg += f"accept len: {spec_accept_length:.2f}, "

        if self.disaggregation_mode == DisaggregationMode.DECODE:
            msg += f"pre-allocated usage: {self.disagg_decode_prealloc_queue.num_tokens_pre_allocated / self.max_total_num_tokens:.2f}, "
            msg += f"#retracted-req: {len(self.disagg_decode_prealloc_queue.retracted_queue)}, "

        msg += (
            f"cuda graph: {can_run_cuda_graph}, "
            f"gen throughput (token/s): {self.last_gen_throughput:.2f}, "
            f"#queue-req: {len(self.waiting_queue)}, "
        )

        logger.info(msg)
        if self.enable_metrics:
            self.stats.num_running_reqs = num_running_reqs
            self.stats.num_used_tokens = num_used
            self.stats.token_usage = round(token_usage, 2)
            self.stats.cache_hit_rate = 0.0
            self.stats.gen_throughput = self.last_gen_throughput
            self.stats.num_queue_reqs = len(self.waiting_queue)
            self.stats.num_grammar_queue_reqs = len(self.grammar_queue)
            self.stats.spec_accept_length = spec_accept_length
            self.stats.total_retracted_reqs = self.total_retracted_reqs
            self.metrics_collector.log_stats(self.stats)
            self._emit_kv_metrics()
        self._publish_kv_events()

    def _emit_kv_metrics(self):
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

    def _publish_kv_events(self):
        if self.enable_kv_cache_events:
            events = self.tree_cache.take_events()
            if events:
                batch = KVEventBatch(ts=time.time(), events=events)
                self.kv_event_publisher.publish(batch)
