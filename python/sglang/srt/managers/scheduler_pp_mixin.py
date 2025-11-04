from typing import List, Optional

from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.managers.schedule_batch import ScheduleBatch
from sglang.srt.managers.utils import GenerationBatchResult
from sglang.srt.model_executor.forward_batch_info import PPProxyTensors
from sglang.srt.utils import DynamicGradMode, point_to_point_pyobj, require_mlp_sync


class SchedulerPPMixin:

    @DynamicGradMode()
    def event_loop_pp(self):
        """A non-overlap scheduler loop for pipeline parallelism."""
        mbs = [None] * self.pp_size
        last_mbs = [None] * self.pp_size
        self.running_mbs = [
            ScheduleBatch(reqs=[], batch_is_full=False) for _ in range(self.pp_size)
        ]
        pp_outputs: Optional[PPProxyTensors] = None
        while True:
            server_is_idle = True
            for mb_id in range(self.pp_size):
                self.running_batch = self.running_mbs[mb_id]
                self.last_batch = last_mbs[mb_id]

                recv_reqs = self.recv_requests()
                self.process_input_requests(recv_reqs)
                mbs[mb_id] = self.get_next_batch_to_run()
                self.running_mbs[mb_id] = self.running_batch

                self.cur_batch = mbs[mb_id]
                if self.cur_batch:
                    server_is_idle = False
                    result = self.run_batch(self.cur_batch)

                # (last rank) send the outputs to the next step
                if self.pp_group.is_last_rank:
                    if self.cur_batch:
                        next_token_ids = result.next_token_ids
                        if self.cur_batch.return_logprob:
                            pp_outputs = PPProxyTensors(
                                {
                                    "next_token_ids": next_token_ids,
                                    "extend_input_len_per_req": result.extend_input_len_per_req,
                                    "extend_logprob_start_len_per_req": result.extend_logprob_start_len_per_req,
                                }
                                | (
                                    {
                                        f"logits_output.{k}": v
                                        for k, v in result.logits_output.__dict__.items()
                                    }
                                    if result.logits_output is not None
                                    else {}
                                )
                            )
                        else:
                            pp_outputs = PPProxyTensors(
                                {
                                    "next_token_ids": next_token_ids,
                                }
                            )
                        # send the output from the last round to let the next stage worker run post processing
                        self.pp_group.send_tensor_dict(
                            pp_outputs.tensors,
                            all_gather_group=self.attn_tp_group,
                        )

                # receive outputs and post-process (filter finished reqs) the coming microbatch
                next_mb_id = (mb_id + 1) % self.pp_size
                next_pp_outputs = None
                if mbs[next_mb_id] is not None:
                    next_pp_outputs: Optional[PPProxyTensors] = PPProxyTensors(
                        self.pp_group.recv_tensor_dict(
                            all_gather_group=self.attn_tp_group
                        )
                    )
                    mbs[next_mb_id].output_ids = next_pp_outputs["next_token_ids"]
                    logits_output_args = {
                        k[len("logits_output.") :]: v
                        for k, v in next_pp_outputs.tensors.items()
                        if k.startswith("logits_output.")
                    }
                    if len(logits_output_args) > 0:
                        logits_output = LogitsProcessorOutput(**logits_output_args)
                    else:
                        logits_output = None

                    output_result = GenerationBatchResult.from_pp_proxy(
                        logits_output=logits_output,
                        next_pp_outputs=next_pp_outputs,
                        can_run_cuda_graph=result.can_run_cuda_graph,
                    )
                    self.process_batch_result(mbs[next_mb_id], output_result)
                    last_mbs[next_mb_id] = mbs[next_mb_id]

                # (not last rank)
                if not self.pp_group.is_last_rank:
                    # carry the outputs to the next stage
                    # send the outputs from the last round to let the next stage worker run post processing
                    if pp_outputs:
                        self.pp_group.send_tensor_dict(
                            pp_outputs.tensors,
                            all_gather_group=self.attn_tp_group,
                        )

                    # send out reqs to the next stage
                    dp_offset = self.attn_dp_rank * self.attn_tp_size
                    if self.attn_tp_rank == 0:
                        point_to_point_pyobj(
                            recv_reqs,
                            self.pp_rank * self.tp_size + dp_offset,
                            self.world_group.device_group,
                            self.pp_rank * self.tp_size + dp_offset,
                            (self.pp_rank + 1) * self.tp_size + dp_offset,
                        )

                    # send out proxy tensors to the next stage
                    if self.cur_batch:
                        # FIXME(lsyin): remove this assert
                        assert result.pp_hidden_states_proxy_tensors.tensors is not None
                        self.pp_group.send_tensor_dict(
                            result.pp_hidden_states_proxy_tensors.tensors,
                            all_gather_group=self.attn_tp_group,
                        )

                pp_outputs = next_pp_outputs

            # When the server is idle, self-check and re-init some states
            if server_is_idle:
                # When the server is idle, do self-check and re-init some states
                self.self_check_during_idle()

    @DynamicGradMode()
    def event_loop_pp_disagg_prefill(self):
        """
        An event loop for the prefill server in pipeline parallelism.

        Rules:
        1. Each stage runs in the same order and is notified by the previous stage.
        2. Each send/recv operation is blocking and matched by the neighboring stage.

        Regular Schedule:
        ====================================================================
        Stage i                   | Stage i+1
        send ith req              | recv ith req
        send ith proxy            | recv ith proxy
        send prev (i+1)th carry   | recv prev (i+1)th carry
        ====================================================================

        Prefill Server Schedule:
        ====================================================================
        Stage i                        | Stage i+1
        send ith req                   | recv ith req
        send ith bootstrap req         | recv ith bootstrap req
        send ith transferred req       | recv ith transferred req
        send ith proxy                 | recv ith proxy
        send prev (i+1)th carry        | recv prev (i+1)th carry
        send prev (i+1)th release req  | recv prev (i+1)th release req
        ====================================================================

        There are two additional elements compared to the regular schedule:

        1. Bootstrap Requests:
            a. Instead of polling the status on the current workers, we should wait for the previous stage to notify to avoid desynchronization.
            b. The first stage polls the status and propagates the bootstrapped requests down to all other stages.
            c. If the first stage polls successfully, by nature, other ranks are also successful because they performed a handshake together.

        2. Transferred Requests + Release Requests:
            a. The first stage polls the transfer finished requests, performs an intersection with the next stage's finished requests, and propagates down to the last stage.
            b. The last stage receives the requests that have finished transfer on all stages (consensus), then sends them to the first stage to release the memory.
            c. The first stage receives the release requests, releases the memory, and then propagates the release requests down to the last stage.
        """
        mbs = [None] * self.pp_size
        last_mbs = [None] * self.pp_size
        self.running_mbs = [
            ScheduleBatch(reqs=[], batch_is_full=False) for _ in range(self.pp_size)
        ]
        pp_outputs: Optional[PPProxyTensors] = None

        # Either success or failed
        bootstrapped_rids: List[str] = []
        transferred_rids: List[str] = []
        release_rids: Optional[List[str]] = None

        # transferred microbatch
        tmbs = [None] * self.pp_size

        ENABLE_RELEASE = True  # For debug

        while True:
            server_is_idle = True

            for mb_id in range(self.pp_size):
                self.running_batch = self.running_mbs[mb_id]
                self.last_batch = last_mbs[mb_id]

                recv_reqs = self.recv_requests()

                self.process_input_requests(recv_reqs)

                if self.pp_group.is_first_rank:
                    # First rank, pop the bootstrap reqs from the bootstrap queue
                    bootstrapped_reqs, failed_reqs = (
                        self.disagg_prefill_bootstrap_queue.pop_bootstrapped(
                            return_failed_reqs=True
                        )
                    )
                    bootstrapped_rids = [req.rid for req in bootstrapped_reqs] + [
                        req.rid for req in failed_reqs
                    ]
                    self.waiting_queue.extend(bootstrapped_reqs)
                else:
                    # Other ranks, receive the bootstrap reqs info from the previous rank and ensure the consensus
                    bootstrapped_rids = self.recv_pyobj_from_prev_stage()
                    bootstrapped_reqs = (
                        self.disagg_prefill_bootstrap_queue.pop_bootstrapped(
                            rids_to_check=bootstrapped_rids
                        )
                    )
                    self.waiting_queue.extend(bootstrapped_reqs)

                if self.pp_group.is_first_rank:
                    transferred_rids = self.get_transferred_rids()
                # if other ranks,
                else:
                    # 1. recv previous stage's transferred reqs info
                    prev_transferred_rids = self.recv_pyobj_from_prev_stage()
                    # 2. get the current stage's transferred reqs info
                    curr_transferred_rids = self.get_transferred_rids()
                    # 3. new consensus rids = intersection(previous consensus rids, transfer finished rids)
                    transferred_rids = list(
                        set(prev_transferred_rids) & set(curr_transferred_rids)
                    )

                tmbs[mb_id] = transferred_rids

                self.process_prefill_chunk()

                batch = self.get_new_batch_prefill()
                if require_mlp_sync(self.server_args):
                    batch = self.prepare_mlp_sync_batch(batch)
                mbs[mb_id] = batch

                self.running_mbs[mb_id] = self.running_batch

                self.cur_batch = mbs[mb_id]
                if self.cur_batch:
                    server_is_idle = False
                    result = self.run_batch(self.cur_batch)

                # send the outputs to the next step
                if self.pp_group.is_last_rank:
                    if self.cur_batch:
                        next_token_ids = result.next_token_ids
                        pp_outputs = PPProxyTensors(
                            {
                                "next_token_ids": next_token_ids,
                            }
                        )
                        # send the output from the last round to let the next stage worker run post processing
                        self.pp_group.send_tensor_dict(
                            pp_outputs.tensors,
                            all_gather_group=self.attn_tp_group,
                        )

                if ENABLE_RELEASE:
                    if self.pp_group.is_last_rank:
                        # At the last stage, all stages has reached the consensus to release memory for transferred_rids
                        release_rids = transferred_rids
                        # send to the first rank
                        self.send_pyobj_to_next_stage(release_rids)

                # receive outputs and post-process (filter finished reqs) the coming microbatch
                next_mb_id = (mb_id + 1) % self.pp_size
                next_pp_outputs = None
                next_release_rids = None

                if mbs[next_mb_id] is not None:
                    next_pp_outputs: Optional[PPProxyTensors] = PPProxyTensors(
                        self.pp_group.recv_tensor_dict(
                            all_gather_group=self.attn_tp_group
                        )
                    )
                    mbs[next_mb_id].output_ids = next_pp_outputs["next_token_ids"]
                    output_result = GenerationBatchResult(
                        logits_output=None,
                        pp_hidden_states_proxy_tensors=None,
                        next_token_ids=next_pp_outputs["next_token_ids"],
                        extend_input_len_per_req=None,
                        extend_logprob_start_len_per_req=None,
                        can_run_cuda_graph=result.can_run_cuda_graph,
                    )
                    self.process_batch_result_disagg_prefill(
                        mbs[next_mb_id], output_result
                    )

                    last_mbs[next_mb_id] = mbs[next_mb_id]

                if ENABLE_RELEASE:
                    if tmbs[next_mb_id] is not None:
                        # recv consensus rids from the previous rank
                        next_release_rids = self.recv_pyobj_from_prev_stage()
                        self.process_disagg_prefill_inflight_queue(next_release_rids)

                # carry the outputs to the next stage
                if not self.pp_group.is_last_rank:
                    if pp_outputs:
                        # send the outputs from the last round to let the next stage worker run post processing
                        self.pp_group.send_tensor_dict(
                            pp_outputs.tensors,
                            all_gather_group=self.attn_tp_group,
                        )
                    if ENABLE_RELEASE:
                        if release_rids is not None:
                            self.send_pyobj_to_next_stage(release_rids)

                if not self.pp_group.is_last_rank:
                    # send out reqs to the next stage
                    self.send_pyobj_to_next_stage(recv_reqs)
                    self.send_pyobj_to_next_stage(bootstrapped_rids)
                    self.send_pyobj_to_next_stage(transferred_rids)

                    # send out proxy tensors to the next stage
                    if self.cur_batch:
                        # FIXME(lsyin): remove this assert
                        assert result.pp_hidden_states_proxy_tensors.tensors is not None
                        self.pp_group.send_tensor_dict(
                            result.pp_hidden_states_proxy_tensors.tensors,
                            all_gather_group=self.attn_tp_group,
                        )

                pp_outputs = next_pp_outputs
                release_rids = next_release_rids

                self.running_batch.batch_is_full = False

            if not ENABLE_RELEASE:
                if len(self.disagg_prefill_inflight_queue) > 0:
                    self.process_disagg_prefill_inflight_queue()

            # When the server is idle, self-check and re-init some states
            if server_is_idle and len(self.disagg_prefill_inflight_queue) == 0:
                self.check_memory()
                self.check_tree_cache()
                self.new_token_ratio = self.init_new_token_ratio
