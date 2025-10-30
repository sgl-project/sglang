from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

import torch

from sglang.srt.disaggregation.base.conn import KVPoll
from sglang.srt.disaggregation.utils import DisaggregationMode
from sglang.srt.distributed.parallel_state import P2PWork
from sglang.srt.managers.schedule_batch import ScheduleBatch
from sglang.srt.managers.utils import (
    GenerationBatchResult,
    get_logprob_dict_from_result,
    get_logprob_from_pp_outputs,
)
from sglang.srt.model_executor.forward_batch_info import PPProxyTensors
from sglang.srt.utils import DynamicGradMode, broadcast_pyobj, point_to_point_pyobj

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from sglang.srt.managers.scheduler import Scheduler


@dataclass
class PPBatchMetadata:
    can_run_cuda_graph: bool


class SchedulerPPMixin:
    def _pp_commit_comm_work(self: Scheduler, work: List[P2PWork]) -> None:
        for p2p_work in work:
            p2p_work.work.wait()
        work.clear()

    def _pp_send_pyobj_to_next_stage(self: Scheduler, data, async_send: bool = False):
        p2p_work = []
        if self.attn_tp_rank == 0:
            dp_offset = self.attn_dp_rank * self.attn_tp_size
            p2p_work = point_to_point_pyobj(
                data,
                self.pp_rank * self.tp_size + dp_offset,
                self.world_group.cpu_group,
                self.pp_rank * self.tp_size + dp_offset,
                ((self.pp_rank + 1) % self.pp_size) * self.tp_size + dp_offset,
                async_send=async_send,
            )
        return p2p_work

    def recv_pyobj_from_prev_stage(self: Scheduler):
        if self.attn_tp_rank == 0:
            dp_offset = self.dp_rank * self.attn_tp_size
            data = point_to_point_pyobj(
                [],
                self.pp_rank * self.tp_size + dp_offset,
                self.world_group.cpu_group,
                ((self.pp_rank - 1) % self.pp_size) * self.tp_size + dp_offset,
                self.pp_rank * self.tp_size + dp_offset,
            )
        else:
            data = None

        if self.tp_size != 1:
            data = broadcast_pyobj(
                data, self.tp_group.rank, self.tp_cpu_group, src=self.tp_group.ranks[0]
            )

        return data

    def _pp_prepare_tensor_dict(
        self: Scheduler, result: GenerationBatchResult, batch: ScheduleBatch
    ) -> Dict[str, torch.Tensor]:
        tensor_dict = {
            "next_token_ids": result.next_token_ids,
        }

        if batch.return_logprob:
            logprob_dict = get_logprob_dict_from_result(result)
            tensor_dict = {
                **tensor_dict,
                **logprob_dict,
            }
        return tensor_dict

    def _pp_send_dict_to_next_stage(
        self: Scheduler,
        tensor_dict: Dict[str, torch.Tensor],
        async_send: bool = True,
    ):
        p2p_work = []
        p2p_work.extend(
            self.pp_group.send_tensor_dict(
                tensor_dict=tensor_dict,
                all_gather_group=self.attn_tp_group,
                async_send=async_send,
            )
        )
        return p2p_work

    def _pp_recv_proxy_tensors(self: Scheduler) -> Optional[PPProxyTensors]:
        pp_proxy_tensors = None
        if not self.pp_group.is_first_rank:
            pp_proxy_tensors = PPProxyTensors(
                self.pp_group.recv_tensor_dict(all_gather_group=self.attn_tp_group)
            )
        return pp_proxy_tensors

    def _pp_recv_dict_from_prev_stage(
        self: Scheduler,
    ) -> Dict[str, torch.Tensor]:
        res = self.pp_group.recv_tensor_dict(
            all_gather_group=self.attn_tp_group,
        )
        return res

    def _pp_prep_batch_result(
        self: Scheduler,
        batch: ScheduleBatch,
        mb_metadata: PPBatchMetadata,
        pp_outputs: PPProxyTensors,
    ):
        from sglang.srt.managers.scheduler import GenerationBatchResult

        logits_output = None
        extend_input_len_per_req = None
        extend_logprob_start_len_per_req = None

        if batch.return_logprob:
            (
                logits_output,
                extend_input_len_per_req,
                extend_logprob_start_len_per_req,
            ) = get_logprob_from_pp_outputs(pp_outputs)
        batch.output_ids = pp_outputs["next_token_ids"]
        output_result = GenerationBatchResult(
            logits_output=logits_output,
            pp_hidden_states_proxy_tensors=None,
            next_token_ids=pp_outputs["next_token_ids"],
            extend_input_len_per_req=extend_input_len_per_req,
            extend_logprob_start_len_per_req=extend_logprob_start_len_per_req,
            can_run_cuda_graph=mb_metadata.can_run_cuda_graph,
        )
        return output_result

    def _pp_process_batch_result(
        self: Scheduler, batch: ScheduleBatch, output_result: GenerationBatchResult
    ):
        if self.disaggregation_mode == DisaggregationMode.PREFILL:
            self.process_batch_result_disagg_prefill(batch, output_result)
        else:
            self.process_batch_result(batch, output_result)

    def _pp_send_output_to_next_stage(
        self: Scheduler,
        next_first_rank_mb_id: int,
        mbs: List[ScheduleBatch],
        last_rank_comm_queue: deque[Tuple[torch.cuda.Event, PPProxyTensors]],
        pp_outputs: PPProxyTensors | None,
    ) -> List[P2PWork]:
        send_output_work = []
        if self.pp_group.is_last_rank:
            # send ready PP output to rank 0
            if mbs[next_first_rank_mb_id] is not None:
                q_event, pp_outputs_to_send = last_rank_comm_queue.popleft()
                torch.cuda.current_stream().wait_event(q_event)
                with torch.profiler.record_function("send_res_dict_to_next_stage"):
                    send_output_work = self._pp_send_dict_to_next_stage(
                        pp_outputs_to_send.tensors,
                        async_send=True,
                    )
        # send the outputs from the last round to let the next stage worker run post processing
        if not self.pp_group.is_last_rank:
            if pp_outputs:
                with torch.profiler.record_function("send_res_dict_to_next_stage"):
                    send_output_work = self._pp_send_dict_to_next_stage(
                        pp_outputs.tensors,
                        async_send=True,
                    )
        return send_output_work

    def _pp_send_recv_and_preprocess_output_tensors(
        self: Scheduler,
        next_first_rank_mb_id: int,
        next_mb_id: int,
        mbs: List[ScheduleBatch],
        mb_metadata: List[PPBatchMetadata],
        last_rank_comm_queue: deque[Tuple[torch.cuda.Event, PPProxyTensors]],
        pp_outputs: PPProxyTensors | None,
    ) -> Tuple[PPProxyTensors, List[P2PWork], torch.cuda.Event]:
        next_pp_outputs = None
        d2h_event = None
        batch_result = None
        send_output_work = self._pp_send_output_to_next_stage(
            next_first_rank_mb_id,
            mbs,
            last_rank_comm_queue,
            pp_outputs,
        )

        if mbs[next_mb_id] is not None:
            with torch.profiler.record_function("recv_res_dict_from_prev_stage"):
                next_pp_outputs = PPProxyTensors(self._pp_recv_dict_from_prev_stage())
            with self.copy_stream_ctx:
                self.copy_stream.wait_stream(self.default_stream)
                batch_result = self._pp_prep_batch_result(
                    mbs[next_mb_id], mb_metadata[next_mb_id], next_pp_outputs
                )
                d2h_event = torch.cuda.Event()
                d2h_event.record(torch.cuda.current_stream())

        return next_pp_outputs, batch_result, d2h_event, send_output_work

    def _pp_launch_batch(
        self: Scheduler,
        mb_id: int,
        pp_proxy_tensors: PPProxyTensors,
        mb_metadata: List[Optional[PPBatchMetadata]],
        last_rank_comm_queue: deque[Tuple[torch.cuda.Event, PPProxyTensors]],
    ):
        with torch.profiler.record_function("run_batch"):
            with self.forward_stream_ctx:
                self.forward_stream.wait_stream(self.default_stream)
                result = self.run_batch(self.cur_batch, pp_proxy_tensors)
                mb_metadata[mb_id] = PPBatchMetadata(
                    can_run_cuda_graph=result.can_run_cuda_graph,
                )
                event = torch.cuda.Event()
                event.record(torch.cuda.current_stream())
                if self.pp_group.is_last_rank:
                    # (last rank) buffer the outputs for async batch depth
                    last_rank_comm_queue.append(
                        (
                            event,
                            PPProxyTensors(
                                self._pp_prepare_tensor_dict(result, self.cur_batch)
                            ),
                        )
                    )
        return result, event

    @DynamicGradMode()
    def event_loop_pp(self: Scheduler):
        """
        A scheduler loop for pipeline parallelism.
        Notes:
        1. Each stage runs in the same order and is notified by the previous stage.
        2. We use async send but sync recv to avoid desynchronization while minimizing the communication overhead.
        3. We can use async batch depth to buffer the outputs in the last stage for to allow overlapping the GPU computation and CPU processing and avoid last PP rank staggler.

        Unified Schedule:
        ====================================================================
        Stage P
        recv ith req from previous stage
        recv ith proxy from previous stage
        run ith batch
        recv prev (i+1)% mb_size th outputs
        process batch result of prev (i+1)% mb_size th batch (can be run in parallel with the curr batch GPU computation)
        send ith req to next stage
        send ith proxy to next stage
        send current stage's outputs to next stage(can be stashed and delayed to send later)

        the above order can be optimized and reordered to minimize communication-related CPU stall and overhead bubbles.

        ====================================================================
        """
        self.pp_loop_size: int = self.pp_size + self.server_args.pp_async_batch_depth
        mbs = [None] * self.pp_loop_size
        last_mbs = [None] * self.pp_loop_size
        self.running_mbs = [
            ScheduleBatch(reqs=[], batch_is_full=False)
            for _ in range(self.pp_loop_size)
        ]
        mb_metadata: List[Optional[PPBatchMetadata]] = [None] * self.pp_loop_size
        pp_outputs: Optional[PPProxyTensors] = None
        last_rank_comm_queue: deque[Tuple[torch.cuda.Event, PPProxyTensors]] = deque()
        send_req_work = []
        send_proxy_work = []
        send_output_work = []
        event = None
        while True:
            server_is_idle = True
            for mb_id in range(self.pp_loop_size):
                self.running_batch = self.running_mbs[mb_id]
                self.last_batch = last_mbs[mb_id]
                next_first_rank_mb_id = (mb_id + self.pp_size) % self.pp_loop_size
                next_mb_id = (mb_id + 1) % self.pp_loop_size
                with torch.profiler.record_function("recv_requests"):
                    recv_reqs = self.recv_requests()
                    self.process_input_requests(recv_reqs)
                if not self.pp_group.is_last_rank:
                    self._pp_commit_comm_work(send_req_work)
                    with torch.profiler.record_function("send_reqs_to_next_stage"):
                        send_req_work = self._pp_send_pyobj_to_next_stage(
                            recv_reqs,
                            async_send=True,
                        )
                with torch.profiler.record_function("get_next_batch_to_run"):
                    mbs[mb_id] = self.get_next_batch_to_run()
                self.running_mbs[mb_id] = self.running_batch
                self.cur_batch: Optional[ScheduleBatch] = mbs[mb_id]
                if self.cur_batch:
                    server_is_idle = False
                    pp_proxy_tensors = self._pp_recv_proxy_tensors()
                next_pp_outputs = None
                next_batch_result = None
                d2h_event = None
                if self.server_args.pp_async_batch_depth > 0:
                    self._pp_commit_comm_work(work=send_output_work)
                    next_pp_outputs, next_batch_result, d2h_event, send_output_work = (
                        self._pp_send_recv_and_preprocess_output_tensors(
                            next_first_rank_mb_id,
                            next_mb_id,
                            mbs,
                            mb_metadata,
                            last_rank_comm_queue,
                            pp_outputs,
                        )
                    )
                self._pp_commit_comm_work(send_proxy_work)
                if self.cur_batch:
                    result, event = self._pp_launch_batch(
                        mb_id, pp_proxy_tensors, mb_metadata, last_rank_comm_queue
                    )
                if self.server_args.pp_async_batch_depth == 0:
                    self._pp_commit_comm_work(work=send_output_work)
                    next_pp_outputs, next_batch_result, d2h_event, send_output_work = (
                        self._pp_send_recv_and_preprocess_output_tensors(
                            next_first_rank_mb_id,
                            next_mb_id,
                            mbs,
                            mb_metadata,
                            last_rank_comm_queue,
                            pp_outputs,
                        )
                    )
                if mbs[next_mb_id] is not None:
                    d2h_event.synchronize()
                    with torch.profiler.record_function("process_batch_result"):
                        self._pp_process_batch_result(
                            mbs[next_mb_id],
                            next_batch_result,
                        )
                    last_mbs[next_mb_id] = mbs[next_mb_id]
                if not self.pp_group.is_last_rank:
                    if self.cur_batch:
                        torch.cuda.current_stream().wait_event(event)
                        with torch.profiler.record_function(
                            "send_proxy_dict_to_next_stage"
                        ):
                            send_proxy_work = self._pp_send_dict_to_next_stage(
                                result.pp_hidden_states_proxy_tensors.tensors,
                                async_send=True,
                            )

                # if self.delayed_weight_sync_fn:
                #     self.delayed_weight_sync_fn()
                #     self.delayed_weight_sync_fn = None

                pp_outputs = next_pp_outputs

            # When the server is idle, self-check and re-init some states
            if server_is_idle:
                self.check_memory()
                self.check_tree_cache()
                self.new_token_ratio = self.init_new_token_ratio
                self.maybe_sleep_on_idle()

    def process_bootstrapped_queue(
        self: Scheduler, bootstrapped_rids: Optional[List[str]]
    ):
        # finished consensus bootstrapped reqs and prepare the waiting queue
        if bootstrapped_rids is not None:
            (
                good_consensus_bootstrapped_rids,
                bad_consensus_bootstrapped_rids,
            ) = bootstrapped_rids
            good_reqs, failed_reqs = (
                self.disagg_prefill_bootstrap_queue.pop_bootstrapped(
                    return_failed_reqs=True,
                    rids_to_check=good_consensus_bootstrapped_rids,
                    bad_rids_to_check=bad_consensus_bootstrapped_rids,
                )
            )
            self.waiting_queue.extend(good_reqs)
            return [[req.rid for req in good_reqs], [req.rid for req in failed_reqs]]
        return None

    def _pp_pd_get_bootstrapped_ids(self: Scheduler):
        # communicate pre-consensus bootstrapp reqs
        if self.pp_group.is_first_rank:
            # First rank, pop the bootstrap reqs from the bootstrap queue
            good_bootstrapped_rids, bad_bootstrapped_rids = self.get_rids(
                self.disagg_prefill_bootstrap_queue.queue,
                [KVPoll.WaitingForInput],
                [KVPoll.Failed],
            )
        else:
            # Other ranks, receive the bootstrap reqs info from the previous rank and ensure the consensus
            prev_bootstrapped_rids = self.recv_pyobj_from_prev_stage()
            prev_good_bootstrapped_rids, prev_bad_bootstrapped_rids = (
                prev_bootstrapped_rids
            )
            curr_good_bootstrapped_rids, curr_bad_bootstrapped_rids = self.get_rids(
                self.disagg_prefill_bootstrap_queue.queue,
                [KVPoll.WaitingForInput],
                [KVPoll.Failed],
            )
            good_bootstrapped_rids = list(
                set(prev_good_bootstrapped_rids) & set(curr_good_bootstrapped_rids)
            )
            bad_bootstrapped_rids = list(
                set(prev_bad_bootstrapped_rids) | set(curr_bad_bootstrapped_rids)
            )
        return [good_bootstrapped_rids, bad_bootstrapped_rids]

    def _pp_pd_get_transferred_ids(self: Scheduler):
        # get the current stage transfer success
        if self.pp_group.is_first_rank:
            transferred_rids = self.get_rids(
                self.disagg_prefill_inflight_queue,
                [KVPoll.Success, KVPoll.Failed],
            )
        # if other ranks, do intersection with the previous rank's transferred rids
        else:
            # 2 (Release): Receive the transferred rids from the previous rank
            # 1. recv previous stage's transferred reqs info
            prev_transferred_rids = self.recv_pyobj_from_prev_stage()
            # 2. get the current stage's transferred reqs info
            curr_transferred_rids = self.get_rids(
                self.disagg_prefill_inflight_queue,
                [KVPoll.Success, KVPoll.Failed],
            )
            # 3. new consensus rids = intersection(previous consensus rids, transfer finished rids)
            transferred_rids = list(
                set(prev_transferred_rids) & set(curr_transferred_rids)
            )
        return transferred_rids

    def _pp_pd_send_consensus_bootstrapped_ids(
        self: Scheduler,
        bmbs: List[List[str]],
        next_first_rank_mb_id: int,
        consensus_bootstrapped_rids: List[str],
        bootstrapped_rids: List[str],
    ):
        # 3 (Release): send the release rids from last stage to the first stage
        send_consensus_bootstrapped_work = []
        if self.pp_group.is_last_rank:
            if bmbs[next_first_rank_mb_id] is not None:
                consensus_bootstrapped_rids = bootstrapped_rids
                send_consensus_bootstrapped_work = self._pp_send_pyobj_to_next_stage(
                    consensus_bootstrapped_rids, async_send=True
                )
        # 4 (Release): send the release rids from non last rank to the next rank
        else:
            if consensus_bootstrapped_rids is not None:
                send_consensus_bootstrapped_work = self._pp_send_pyobj_to_next_stage(
                    consensus_bootstrapped_rids, async_send=True
                )
        return send_consensus_bootstrapped_work, consensus_bootstrapped_rids

    def _pp_pd_send_consensus_release_ids(
        self: Scheduler,
        tmbs: List[List[str]],
        next_first_rank_mb_id: int,
        release_rids: List[str],
        transferred_rids: List[str],
    ):
        send_release_work = []
        if self.pp_group.is_last_rank:
            if tmbs[next_first_rank_mb_id] is not None:
                release_rids = transferred_rids
                send_release_work = self._pp_send_pyobj_to_next_stage(
                    release_rids, async_send=True
                )
        # 4 (Release): send the release rids from non last rank to the next rank
        else:
            if release_rids is not None:
                send_release_work = self._pp_send_pyobj_to_next_stage(
                    release_rids, async_send=True
                )
        return send_release_work, release_rids

    @DynamicGradMode()
    def event_loop_pp_disagg_prefill(self: Scheduler):
        """
        This is the prefill server event loop for pipeline parallelism.

        Notes:
        1. Following the same rules as the event_loop_pp.
        2. Adds extra steps for KV transfer process: bootstrap + release.

        Prefill Server Schedule:
        ====================================================================
        Stage P
        recv ith req from previous stage
        recv ith bootstrap req from previous stage
        recv ith transferred req from previous stage
        recv ith proxy from previous stage
        run ith batch
        recv prev (i+1) % mb_size th consensus bootstrapped req from previous stage
        local consensus on bootstrapped req
        recv prev (i+1) % mb_size th release req from previous stage
        local consensus on release req
        recv prev (i+1) % mb_size th outputs
        process batch result of prev (i+1)% mb_size th batch (can be run in parallel with the curr batch GPU computation)
        send ith req to next stage
        send ith bootstrap req to next stage
        send ith transferred req to next stage
        send ith proxy to next stage
        send current stage's outputs to next stage (can be stashed and delayed to send later)

        the above order can be optimized and reordered to minimize communication-related CPU stall and overhead bubbles.
        ====================================================================

        There are two additional elements compared to the regular schedule:

        Bootstrap Requests + Release Requests:
        - Both can have local failure and need to be consensus on. PP needs to guarantee eventual consistency of local failure and flush malfunc requests out as soft error.

        """
        self.pp_loop_size: int = self.pp_size + self.server_args.pp_async_batch_depth
        mbs = [None] * self.pp_loop_size
        last_mbs = [None] * self.pp_loop_size
        self.running_mbs = [
            ScheduleBatch(reqs=[], batch_is_full=False)
            for _ in range(self.pp_loop_size)
        ]
        mb_metadata: List[Optional[PPBatchMetadata]] = [None] * self.pp_loop_size
        pp_outputs: Optional[PPProxyTensors] = None
        last_rank_comm_queue: deque[Tuple[torch.cuda.Event, PPProxyTensors]] = deque()

        # PD additional
        consensus_bootstrapped_rids: Optional[List[str]] = None
        transferred_rids: List[str] = []
        release_rids: Optional[List[str]] = None
        tmbs = [None] * self.pp_loop_size
        bmbs = [None] * self.pp_loop_size

        send_req_work = []
        send_bootstrapped_work = []
        send_consensus_bootstrapped_work = []
        send_proxy_work = []
        send_release_work = []
        send_transfer_work = []

        while True:
            server_is_idle = True
            for mb_id in range(self.pp_loop_size):
                self.running_batch = self.running_mbs[mb_id]
                self.last_batch = last_mbs[mb_id]
                next_first_rank_mb_id = (mb_id + self.pp_size) % self.pp_loop_size
                next_mb_id = (mb_id + 1) % self.pp_loop_size

                next_pp_outputs = None
                next_release_rids = None
                next_consensus_bootstrapped_rids = None
                d2h_event = None
                next_batch_result = None

                recv_reqs = self.recv_requests()
                self._pp_commit_comm_work(send_req_work)
                self.process_input_requests(recv_reqs)

                bootstrapped_rids = self._pp_pd_get_bootstrapped_ids()
                bmbs[mb_id] = bootstrapped_rids
                self._pp_commit_comm_work(send_bootstrapped_work)

                transferred_rids = self._pp_pd_get_transferred_ids()
                self._pp_commit_comm_work(send_transfer_work)
                tmbs[mb_id] = transferred_rids

                self.process_prefill_chunk_pp()
                mbs[mb_id] = self.get_new_batch_prefill()
                self.running_mbs[mb_id] = self.running_batch

                self.cur_batch: Optional[ScheduleBatch] = mbs[mb_id]
                if self.cur_batch:
                    server_is_idle = False
                    pp_proxy_tensors = self._pp_recv_proxy_tensors()
                self._pp_commit_comm_work(send_proxy_work)
                if self.server_args.pp_async_batch_depth > 0:
                    next_pp_outputs, next_batch_result, d2h_event = (
                        self._pp_send_recv_and_preprocess_output_tensors(
                            next_first_rank_mb_id,
                            next_mb_id,
                            mbs,
                            mb_metadata,
                            last_rank_comm_queue,
                            pp_outputs,
                        )
                    )
                if self.cur_batch:
                    result, event = self._pp_launch_batch(
                        mb_id, pp_proxy_tensors, mb_metadata, last_rank_comm_queue
                    )
                if self.server_args.pp_async_batch_depth == 0:
                    next_pp_outputs, next_batch_result, d2h_event = (
                        self._pp_send_recv_and_preprocess_output_tensors(
                            next_first_rank_mb_id,
                            next_mb_id,
                            mbs,
                            mb_metadata,
                            last_rank_comm_queue,
                            pp_outputs,
                        )
                    )
                send_consensus_bootstrapped_work, consensus_bootstrapped_rids = (
                    self._pp_pd_send_consensus_bootstrapped_ids(
                        bmbs,
                        next_first_rank_mb_id,
                        consensus_bootstrapped_rids,
                        bootstrapped_rids,
                    )
                )
                send_release_work, release_rids = (
                    self._pp_pd_send_consensus_release_ids(
                        tmbs, next_first_rank_mb_id, release_rids, transferred_rids
                    )
                )

                if bmbs[next_mb_id] is not None:
                    next_consensus_bootstrapped_rids = self.recv_pyobj_from_prev_stage()
                    next_consensus_bootstrapped_rids = self.process_bootstrapped_queue(
                        next_consensus_bootstrapped_rids
                    )
                self._pp_commit_comm_work(send_consensus_bootstrapped_work)
                if tmbs[next_mb_id] is not None:
                    next_release_rids = self.recv_pyobj_from_prev_stage()
                self._pp_commit_comm_work(send_release_work)
                # post-process the coming microbatch
                if mbs[next_mb_id] is not None:
                    d2h_event.synchronize()
                    self._pp_process_batch_result(
                        mbs[next_mb_id],
                        next_batch_result,
                    )
                    last_mbs[next_mb_id] = mbs[next_mb_id]

                if tmbs[next_mb_id] is not None:
                    self.process_disagg_prefill_inflight_queue(next_release_rids)
                if not self.pp_group.is_last_rank:
                    send_req_work = self._pp_send_pyobj_to_next_stage(
                        recv_reqs, async_send=True
                    )
                    send_bootstrapped_work = self._pp_send_pyobj_to_next_stage(
                        bootstrapped_rids, async_send=True
                    )
                    send_transfer_work = self._pp_send_pyobj_to_next_stage(
                        transferred_rids, async_send=True
                    )
                    if self.cur_batch:
                        torch.cuda.current_stream().wait_event(event)
                        send_proxy_work = self._pp_send_dict_to_next_stage(
                            result.pp_hidden_states_proxy_tensors.tensors,
                            async_send=True,
                        )

                if self.delayed_weight_sync_fn:
                    self.delayed_weight_sync_fn()
                    self.delayed_weight_sync_fn = None

                pp_outputs = next_pp_outputs
                release_rids = next_release_rids
                consensus_bootstrapped_rids = next_consensus_bootstrapped_rids

                self.running_batch.batch_is_full = False

            # When the server is idle, self-check and re-init some states
            if server_is_idle and len(self.disagg_prefill_inflight_queue) == 0:
                self.check_memory()
                self.check_tree_cache()
                self.new_token_ratio = self.init_new_token_ratio
                self.maybe_sleep_on_idle()
