from __future__ import annotations

import logging
import time
from collections import defaultdict
from typing import TYPE_CHECKING, Dict, List, Optional, Union

import torch

from sglang.srt.disaggregation.kv_events import EventPublisherFactory, KVEventBatch
from sglang.srt.disaggregation.utils import DisaggregationMode
from sglang.srt.managers.io_struct import TokenizedGenerateReqInput
from sglang.srt.managers.schedule_policy import PrefillAdder
from sglang.srt.managers.scheduler import Req, ScheduleBatch
from sglang.srt.managers.utils import DPBalanceMeta
from sglang.srt.metrics.collector import SchedulerMetricsCollector, SchedulerStats
from sglang.srt.utils import get_bool_env_var

if TYPE_CHECKING:
    from sglang.srt.managers.scheduler import Scheduler

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
    def init_metrics(
        self: Scheduler, tp_rank: int, pp_rank: int, dp_rank: Optional[int]
    ):
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

    def init_dp_balance(self: Scheduler, dp_balance_meta: Optional[DPBalanceMeta]):
        self.balance_meta = dp_balance_meta
        if (
            self.server_args.enable_dp_attention
            and self.server_args.load_balance_method == "minimum_tokens"
        ):
            assert dp_balance_meta is not None

        self.recv_dp_balance_id_this_term = []

    def init_kv_events(self: Scheduler, kv_events_config: Optional[str]):
        if self.enable_kv_cache_events:
            self.kv_event_publisher = EventPublisherFactory.create(
                kv_events_config, self.attn_dp_rank
            )

    def log_prefill_stats(
        self: Scheduler,
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

            if self.disaggregation_mode == DisaggregationMode.PREFILL:
                self.stats.num_prefill_prealloc_queue_reqs = len(
                    self.disagg_prefill_bootstrap_queue.queue
                )
                self.stats.num_prefill_inflight_queue_reqs = len(
                    self.disagg_prefill_inflight_queue
                )

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
            f"{'cpu graph' if self.device == 'cpu' else 'cuda graph'}: {can_run_cuda_graph}, "
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
            if self.disaggregation_mode == DisaggregationMode.DECODE:
                self.stats.num_decode_prealloc_queue_reqs = len(
                    self.disagg_decode_prealloc_queue.queue
                )
                self.stats.num_decode_transfer_queue_reqs = len(
                    self.disagg_decode_transfer_queue.queue
                )
            self._emit_kv_metrics()
        self._publish_kv_events()

    def _emit_kv_metrics(self: Scheduler):
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
        if self.enable_kv_cache_events:
            events = self.tree_cache.take_events()
            if events:
                batch = KVEventBatch(ts=time.time(), events=events)
                self.kv_event_publisher.publish(batch)

    def maybe_update_dp_balance_data(
        self: Scheduler, recv_req: TokenizedGenerateReqInput
    ):
        if (
            self.server_args.enable_dp_attention
            and self.server_args.load_balance_method == "minimum_tokens"
        ):
            self.recv_dp_balance_id_this_term.append(recv_req.dp_balance_id)

    def maybe_handle_dp_balance_data(self: Scheduler):
        if (
            self.server_args.load_balance_method == "minimum_tokens"
            and self.forward_ct % 40 == 0
        ):
            holding_tokens = self.get_load()

            new_recv_dp_balance_id_list, holding_token_list = (
                self.gather_dp_balance_info(holding_tokens)
            )

            self.recv_dp_balance_id_this_term.clear()
            if self.tp_rank == 0:  # only first worker write info
                self.write_shared_dp_balance_info(
                    new_recv_dp_balance_id_list, holding_token_list
                )

    def gather_dp_balance_info(
        self: Scheduler, holding_tokens_list
    ) -> Union[None, List[List[int]]]:
        """gather recv_dp_balance_id_this_term and holding tokens per worker for dp balance"""
        recv_list = self.recv_dp_balance_id_this_term
        assert len(recv_list) <= 511, (
            "The number of requests received this round is too large. "
            "Please increase gather_tensor_size and onfly_info_size."
        )
        # The maximum size of the tensor used for gathering data from all workers.
        gather_tensor_size = 512

        # recv_tensor: | holding_tokens | len(recv_dp_balance_id) | recv_dp_balance_ids
        recv_tensor = torch.zeros(gather_tensor_size, dtype=torch.int32)
        recv_tensor[0] = holding_tokens_list
        recv_tensor[1] = len(recv_list)  # The first element is the length of the list.
        recv_tensor[2 : len(recv_list) + 2] = torch.tensor(recv_list, dtype=torch.int32)

        if self.tp_rank == 0:
            gathered_list = [
                torch.zeros(gather_tensor_size, dtype=torch.int32)
                for _ in range(self.balance_meta.num_workers)
            ]
        else:
            gathered_list = None

        torch.distributed.gather(recv_tensor, gathered_list, group=self.tp_cpu_group)

        gathered_id_list_per_worker = None
        if self.tp_rank == 0:
            gathered_id_list_per_worker = []
            holding_tokens_list = []
            for tensor in gathered_list:
                holding_tokens_list.append(tensor[0].item())
                list_length = tensor[1].item()
                gathered_id_list_per_worker.append(tensor[2 : list_length + 2].tolist())

        return gathered_id_list_per_worker, holding_tokens_list

    def write_shared_dp_balance_info(self: Scheduler, new_recv_rid_lists, local_tokens):
        meta = self.balance_meta

        with meta.mutex:
            onfly_list: List[Dict[int, int]] = meta.get_shared_onfly()
            assert len(new_recv_rid_lists) == len(onfly_list), "num_worker not equal"
            # 1.Check if the rid received by each worker this round is present in onfly.
            #   If it is, remove the corresponding onfly item.
            worker_id = 0
            for new_recv_rids, on_fly_reqs in zip(new_recv_rid_lists, onfly_list):
                for new_recv_rid in new_recv_rids:
                    assert (
                        new_recv_rid in on_fly_reqs
                    ), f"{new_recv_rid=} not in {worker_id=} {on_fly_reqs=}, data consistency is wrong"
                    del on_fly_reqs[new_recv_rid]
                worker_id += 1
            # 2. Atomically write local_tokens and onfly into shm under the mutex
            meta.set_shared_onfly_info(onfly_list)
            meta.set_shared_local_tokens(local_tokens)
