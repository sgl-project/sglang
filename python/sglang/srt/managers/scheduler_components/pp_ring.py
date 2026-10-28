from __future__ import annotations

import dataclasses
import logging
from collections import deque
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

import torch
import torch.distributed

from sglang.srt.distributed.parallel_state import P2PWork
from sglang.srt.environ import envs
from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.managers.overlap_utils import RelayPayload
from sglang.srt.managers.schedule_batch import ScheduleBatch
from sglang.srt.managers.scheduler_components.mode_stages import ScheduleSlot
from sglang.srt.managers.scheduler_components.pending_steps import PendingStep
from sglang.srt.managers.utils import (
    GenerationBatchResult,
    get_logprob_dict_from_result,
    get_logprob_from_pp_outputs,
)
from sglang.srt.model_executor.forward_batch_info import (
    ForwardMode,
    PPProxyTensors,
)
from sglang.srt.observability.req_time_stats import set_time_batch
from sglang.srt.runtime_context import get_parallel, get_spec
from sglang.srt.sampling.sampling_observer_pp import (
    add_auxiliary_output_to_pp_tensors,
    pop_auxiliary_output_from_pp_tensors,
)
from sglang.srt.speculative.forward_batch_view import derive_batch
from sglang.srt.speculative.pp_spec_relay import PPSpecRelay
from sglang.srt.utils.common import is_npu

_is_npu = is_npu()
logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from sglang.srt.managers.scheduler_components.bundles import (
        KvResources,
        Streams,
        Workers,
    )
    from sglang.srt.managers.scheduler_components.engine_state import (
        EngineFlags,
        LaunchCounters,
    )
    from sglang.srt.managers.scheduler_components.mode_stages import (
        DriverOps,
        ModeStages,
    )
    from sglang.srt.managers.scheduler_components.pending_steps import PendingSteps
    from sglang.srt.managers.scheduler_components.pp_channel import PPChannel
    from sglang.srt.managers.scheduler_components.pp_spec_forward import (
        PPSpecForward,
    )


def _pp_can_skip_output_comm(batch: ScheduleBatch) -> bool:
    """Check if output send/recv can be skipped for this batch."""
    return (
        envs.SGLANG_PP_SKIP_PURE_CHUNKED_OUTPUT_COMM.get()
        and batch is not None
        and batch.forward_mode == ForwardMode.EXTEND
        and len(batch.reqs) == 1
        and not batch.contains_last_prefill_chunk
        and not batch.return_logprob
    )


def _pp_exchange_outputs_before_forward(
    cur_batch: Optional[ScheduleBatch],
    spec_relay: bool,
    is_last_rank: bool,
    async_batch_depth: int,
) -> bool:
    """Extend microbatches launch first: they need nothing from the relay, and
    exchanging first caps every stage at (pp_size - 1) / pp_size. A verify round
    must exchange first or the ring deadlocks on its tree rebuild."""
    if async_batch_depth > 0:
        return True
    if not spec_relay or is_last_rank or cur_batch is None:
        return False
    return not (cur_batch.forward_mode.is_extend() or cur_batch.is_extend_in_batch)


@dataclasses.dataclass(slots=True)
class RingSlot:
    """One microbatch slot of the ring: what it schedules for, and what it
    launched."""

    # Its decoding requests.
    schedule_slot: ScheduleSlot
    # The record of the step it launched last; None when it launched nothing.
    # A record is pending until the ring commits it.
    record: Optional[PendingStep] = None
    # The step it launched most recently, whose mode weighs the slot's intake.
    last_step: Optional[ScheduleBatch] = None


class PPRing:
    """The PP ring driver of one rank: its microbatch slots in ring order, the
    steps they launched, the outputs the ring brings back for them, and the
    PP+spec trees relayed by request slot."""

    # The gated PP+spec relay flow's trees, per request slot; None without it.
    # Class-level default for rings built without __init__ (unit-test doubles).
    pp_spec_relay: Optional[PPSpecRelay] = None

    def __init__(
        self,
        *,
        channel: PPChannel,
        stages: ModeStages,
        pending: PendingSteps,
        recv_skipper,
        ops: DriverOps,
        counters: LaunchCounters,
        flags: EngineFlags,
        pp_spec_forward: Optional[PPSpecForward],
        kv: KvResources,
        streams: Streams,
        workers: Workers,
        future_map,
    ):
        pp_group = get_parallel().pp_group
        device = streams.device
        device_module = streams.device_module
        schedule_stream = streams.schedule_stream
        copy_stream = streams.copy_stream
        copy_stream_ctx = device_module.stream(copy_stream)
        forward_stream_ctx = device_module.stream(streams.forward_stream)
        token_to_kv_pool_allocator = kv.token_to_kv_pool_allocator
        self.channel = channel
        # The driver's stages and the state it shares.
        self.stages = stages
        self.pending = pending
        self.recv_skipper = recv_skipper
        self.ops = ops
        self.counters = counters
        self.flags = flags
        # What the forward and the output relay read.
        self.pp_group = pp_group
        self.device = device
        self.device_module = device_module
        self.token_to_kv_pool_allocator = token_to_kv_pool_allocator
        self.tp_worker = workers.tp_worker
        self.future_map = future_map
        self.schedule_stream = schedule_stream
        self.copy_stream = copy_stream
        self.copy_stream_ctx = copy_stream_ctx
        self.forward_stream_ctx = forward_stream_ctx

        self.pp_loop_size: int = (
            get_parallel().pp_size + get_parallel().pp_async_batch_depth
        )
        self.slots = [
            RingSlot(schedule_slot=ScheduleSlot(index=i))
            for i in range(self.pp_loop_size)
        ]
        self.pp_outputs: Optional[PPProxyTensors] = None
        self.last_rank_comm_queue: deque[Tuple[torch.Event, PPProxyTensors]] = deque()
        self.pp_spec_relay = (
            pp_spec_forward.pp_spec_relay if pp_spec_forward is not None else None
        )

        self.send_req_work = []
        self.send_output_work = []

    def run_ring_loop(self) -> None:
        """Run the microbatch slots in ring order. A slot launches its step,
        then commits the next slot's step, whose outputs the ring has brought
        back; a step stays pending in between. A PD mode adds its consensus
        messages around each slot."""
        stages = self.stages
        pd = (
            stages.ring_consensus(self.channel, self.pp_loop_size)
            if stages.ring_consensus is not None
            else None
        )
        while True:
            server_is_idle = True
            for mb_id in range(self.pp_loop_size):
                next_first_rank_mb_id = (
                    mb_id + get_parallel().pp_size
                ) % self.pp_loop_size
                next_mb_id = (mb_id + 1) % self.pp_loop_size
                if self.recv_skipper is not None:
                    self.recv_skipper.on_launch(self.slots[mb_id].last_step)

                # Receive requests; plain PP relays them on right away, PD
                # after this slot's consensus messages.
                with torch.profiler.record_function("recv_requests"):
                    recv_reqs = self.ops.ingest_requests()
                if not self.pp_group.is_last_rank:
                    self.channel.commit(self.send_req_work)
                    if pd is None:
                        with torch.profiler.record_function("send_reqs_to_next_stage"):
                            self.send_req_work = self.channel.send_pyobj(
                                recv_reqs,
                                async_send=True,
                            )

                # A shutdown or a pause launches nothing more, from the same
                # slot on every rank: the requests that carry it are relayed
                # on first. The slot still exchanges and commits what is in
                # flight, which the ranks pipeline across slots; once nothing
                # is, a shutdown ends the ring.
                exiting = self.flags.gracefully_exit
                if exiting and not self.pending:
                    if pd is not None and not self.pp_group.is_last_rank:
                        self.send_req_work = self.channel.send_pyobj(
                            recv_reqs, async_send=True
                        )
                    self.channel.commit(self.send_req_work)
                    self.channel.commit(self.send_output_work)
                    return
                held = exiting or self.ops.paused_pass()
                if held:
                    # Not idle: the idle checks wait for the resume, as in
                    # the flat loop.
                    server_is_idle = False

                if pd is not None:
                    pd.take_ids(mb_id)

                # Schedule the slot's step from the requests tagged with it;
                # each slot admits up to its own batch size.
                slot = self.slots[mb_id].schedule_slot
                step = None
                if not held:
                    with torch.profiler.record_function("get_next_batch_to_run"):
                        step = self.ops.schedule_step(slot)
                self.counters.cur_batch_for_debug = step
                if step:
                    self.slots[mb_id].last_step = step
                pp_proxy_tensors = None
                if step:
                    server_is_idle = False
                    pp_proxy_tensors = self.channel.recv_proxy()

                # Launch it, and exchange the outputs of the next slot's step
                # before or after the launch.
                exchange_first = _pp_exchange_outputs_before_forward(
                    cur_batch=step,
                    spec_relay=self.pp_spec_relay is not None,
                    is_last_rank=self.pp_group.is_last_rank,
                    async_batch_depth=get_parallel().pp_async_batch_depth,
                )
                if exchange_first:
                    outputs = (
                        self._commit_send_output_work_and_preprocess_output_tensors(
                            next_first_rank_mb_id,
                            next_mb_id,
                        )
                    )
                self.channel.commit_proxy_send()
                self.slots[mb_id].record = (
                    self.ops.launch_step(step, slot, pp_proxy_tensors=pp_proxy_tensors)
                    if step
                    else None
                )
                if not exchange_first:
                    outputs = (
                        self._commit_send_output_work_and_preprocess_output_tensors(
                            next_first_rank_mb_id,
                            next_mb_id,
                        )
                    )
                next_pp_outputs, next_batch_result, d2h_event = outputs
                if pd is not None:
                    pd.exchange_ids(
                        self.slots[next_mb_id].schedule_slot, next_first_rank_mb_id
                    )

                # Commit the next slot's step, launched one ring round ago.
                due = self.slots[next_mb_id].record
                if due is not None:
                    popped = self.pending.popleft()
                    assert popped is due, "the ring commits slots in launch order"
                    d2h_event.synchronize()
                    with torch.profiler.record_function("process_batch_result"):
                        self.ops.commit_step(
                            dataclasses.replace(due, result=next_batch_result)
                        )
                if pd is not None:
                    pd.after_commit(next_mb_id)

                # Relay this slot's requests, PD ids and hidden states on.
                if not self.pp_group.is_last_rank:
                    if pd is not None:
                        self.send_req_work = self.channel.send_pyobj(
                            recv_reqs, async_send=True
                        )
                        pd.send_ids()
                    launched = self.slots[mb_id].record
                    if launched:
                        self.channel.send_proxy(launched.result)

                self.pp_outputs = next_pp_outputs

            # When the server is idle, self-check and re-init some states;
            # on_idle checks the requests left in any list itself.
            if server_is_idle:
                self.ops.mark_idle()
                self.ops.on_idle()

    def _commit_send_output_work_and_preprocess_output_tensors(
        self,
        next_first_rank_mb_id: int,
        next_mb_id: int,
    ) -> Tuple[
        Optional[PPProxyTensors],
        Optional[GenerationBatchResult],
        Optional[torch.Event],
    ]:
        self.channel.commit(work=self.send_output_work)
        (
            next_pp_outputs,
            next_batch_result,
            d2h_event,
            self.send_output_work,
        ) = self._send_recv_and_preprocess_output_tensors(
            next_first_rank_mb_id,
            next_mb_id,
            self.slots,
            self.last_rank_comm_queue,
            self.pp_outputs,
        )
        return next_pp_outputs, next_batch_result, d2h_event

    def _prepare_tensor_dict(
        self, result: GenerationBatchResult, batch: ScheduleBatch
    ) -> Dict[str, torch.Tensor]:
        tensor_dict = {
            "next_token_ids": result.next_token_ids,
        }

        if self.pp_spec_relay is not None and result.accept_lens is not None:
            # PP+spec verify round: earlier stages need the accept results to
            # commit the accepted path and the bonus token to root the next
            # round's verify chain.
            tensor_dict["spec_accept_lens"] = result.accept_lens
            tensor_dict["spec_new_seq_lens"] = result.new_seq_lens
            tensor_dict["spec_bonus_tokens"] = result.next_draft_input.bonus_tokens
            if result.accept_index is not None:
                # Relayed recurrent commits also need chain accept indices.
                tensor_dict["spec_accept_index"] = result.accept_index
            if result.next_verify_chain is not None:
                # Tail-drafted tree for the next verify round (root = bonus),
                # with the topology its tokens were arranged by.
                tensor_dict["spec_next_chain"] = result.next_verify_chain
                tensor_dict["spec_next_parents"] = result.next_verify_parent_list
                tensor_dict["spec_next_top_scores"] = (
                    result.next_verify_top_scores_index
                )

        # Draft extend runs only on the last stage, but every rank needs its relayed
        # output to fill PD auxiliary buffers.
        draft_input = result.next_draft_input
        if (
            draft_input is not None
            and not batch.spec_algorithm.is_dspark()
            and draft_input.topk_p is not None
        ):
            tensor_dict["draft_topk_p"] = draft_input.topk_p.contiguous()
            tensor_dict["draft_topk_index"] = draft_input.topk_index.contiguous()
            tensor_dict["draft_hidden_states"] = draft_input.hidden_states.contiguous()
            # Preserve the DSA IndexShare seed when rebuilding spec_info on each rank.
            if draft_input.dsa_topk_indices is not None:
                tensor_dict["draft_dsa_topk_indices"] = (
                    draft_input.dsa_topk_indices.contiguous()
                )

        has_sampling_mask_output = (
            result.logits_output is not None
            and result.logits_output.sampling_mask_output is not None
        )
        if batch.return_logprob or has_sampling_mask_output:
            logits_output_dict = get_logprob_dict_from_result(result)
            tensor_dict = {
                **tensor_dict,
                **logits_output_dict,
            }
        auxiliary_output = (
            result.logits_output.auxiliary_device_output
            if result.logits_output is not None
            else None
        )
        add_auxiliary_output_to_pp_tensors(tensor_dict, auxiliary_output)
        return tensor_dict

    def _make_skip_output_result(
        self,
        batch: ScheduleBatch,
        launch_result: GenerationBatchResult,
    ):
        bs = len(batch.reqs)
        placeholder = torch.zeros(bs, dtype=torch.int64, device=self.device)
        # next_pp_outputs = None so non-last ranks skip forwarding
        # (pp_outputs is None gate). Placeholder carried in
        # batch_result.next_token_ids for process_batch_result_prefill.
        batch_result = GenerationBatchResult(
            logits_output=None,
            pp_hidden_states_proxy_tensors=None,
            next_token_ids=placeholder,
            can_run_cuda_graph=launch_result.can_run_cuda_graph,
            skipped_output_comm=True,
        )
        d2h_event = self.device_module.Event()
        d2h_event.record(self.device_module.current_stream())
        return None, batch_result, d2h_event

    def _prep_batch_result(
        self,
        batch: ScheduleBatch,
        launch_result: GenerationBatchResult,
        pp_outputs: PPProxyTensors,
    ):
        logits_output = None

        if (
            batch.return_logprob
            or pp_outputs.tensors.get("sampling_mask_token_ids") is not None
        ):
            logits_output = get_logprob_from_pp_outputs(pp_outputs)
        if self.pp_group.is_first_rank:
            observer = self.tp_worker.model_runner.sampling_observer
            auxiliary_output = pop_auxiliary_output_from_pp_tensors(
                pp_outputs.tensors,
                observer,
            )
            if auxiliary_output is not None:
                if logits_output is None:
                    logits_output = LogitsProcessorOutput(next_token_logits=None)
                logits_output.auxiliary_device_output = auxiliary_output

        if self.pp_spec_relay is not None and "spec_accept_lens" in pp_outputs.tensors:
            # PP+spec verify round, relayed in the step's row order: commit the
            # accepted path on this stage, keep the trees the last stage drafted
            # for the next round in the rows' slots, and rebuild a result the
            # spec output processor can consume (CPU tensors).
            self._spec_commit_relayed_accept(
                batch, launch_result.spec_verify_out_cache_loc, pp_outputs
            )
            chain = pp_outputs.tensors.get("spec_next_chain")
            if chain is None:
                # The last stage verified but skipped drafting (num_steps == 0
                # keeps draft KV warm without proposing). The bonus token it
                # sampled must still root the next round: the old tree would
                # re-verify a token that was already accepted.
                self.pp_spec_relay.write_roots(
                    batch.req_pool_indices, pp_outputs["spec_bonus_tokens"]
                )
            else:
                self.pp_spec_relay.write_trees(
                    batch.req_pool_indices,
                    chain,
                    pp_outputs["spec_next_parents"],
                    pp_outputs["spec_next_top_scores"],
                )
            output_result = GenerationBatchResult(
                logits_output=logits_output,
                pp_hidden_states_proxy_tensors=None,
                next_token_ids=pp_outputs["next_token_ids"].cpu(),
                accept_lens=pp_outputs["spec_accept_lens"].cpu(),
                new_seq_lens=pp_outputs["spec_new_seq_lens"],
                speculative_num_draft_tokens=get_spec().speculative_num_draft_tokens,
                can_run_cuda_graph=launch_result.can_run_cuda_graph,
            )
            return output_result

        next_token_ids = pp_outputs["next_token_ids"].to(torch.int64)

        # The last stage's draft state, rebuilt on every rank for the PD
        # result processor.
        next_draft_input = None
        if "draft_topk_p" in pp_outputs.tensors:
            from sglang.srt.speculative.eagle_info import EagleDraftInput

            next_draft_input = EagleDraftInput(
                topk_p=pp_outputs["draft_topk_p"],
                topk_index=pp_outputs["draft_topk_index"],
                hidden_states=pp_outputs["draft_hidden_states"],
                bonus_tokens=next_token_ids,
                num_tokens_per_req=1,
                num_tokens_for_logprob_per_req=1,
                dsa_topk_indices=pp_outputs.tensors.get("draft_dsa_topk_indices"),
            )
        elif batch.spec_algorithm.is_dspark():
            from sglang.srt.speculative.dspark_components.dspark_draft import (
                make_next_draft_input,
            )

            next_draft_input = make_next_draft_input(
                bonus_tokens=next_token_ids,
                new_seq_lens=batch.seq_lens,
            )

        if self.pp_spec_relay is not None:
            # Gated single-instance PP+spec: the sampled first token roots
            # round 1's tree. Only the chunk that finishes the prompt samples
            # a real token; a middle chunk's next_token_ids is a placeholder
            # no consumer reads, and storing it would seed the next verify
            # round with a token the model never emitted. The decode rounds
            # relay their own state, so the future_map stash is skipped.
            if batch.contains_last_prefill_chunk:
                self.pp_spec_relay.write_roots(batch.req_pool_indices, next_token_ids)
        else:
            # PP rank 0 also relays into output_tokens_buf so the next iter's
            # resolve_forward_inputs finds these tokens for the decode portion
            # of mixed-chunk batches (which gather via mix_running_indices).
            self.future_map.stash(
                batch.req_pool_indices,
                RelayPayload(
                    bonus_tokens=next_token_ids,
                    topk_p=(
                        None if next_draft_input is None else next_draft_input.topk_p
                    ),
                    topk_index=(
                        None
                        if next_draft_input is None
                        else next_draft_input.topk_index
                    ),
                    hidden_states=(
                        None
                        if next_draft_input is None
                        else next_draft_input.hidden_states
                    ),
                ),
            )
        output_result = GenerationBatchResult(
            logits_output=logits_output,
            pp_hidden_states_proxy_tensors=None,
            next_token_ids=pp_outputs["next_token_ids"],
            next_draft_input=next_draft_input,
            can_run_cuda_graph=launch_result.can_run_cuda_graph,
        )
        output_result.copy_auxiliary_output_to_cpu()
        return output_result

    def _spec_commit_relayed_accept(
        self,
        step: ScheduleBatch,
        verify_out_cache_loc: Optional[torch.Tensor],
        pp_outputs,
    ) -> None:
        """Commit relayed recurrent state, then compact tree KV when present.
        The step's lengths are the ones its verify started from."""
        accept_index = pp_outputs.tensors.get("spec_accept_index")
        if accept_index is None or step.forward_mode.is_idle():
            return
        from sglang.srt.speculative.spec_utils import (
            commit_mamba_states_after_verify,
            move_accept_tokens_to_target_kvcache,
        )

        device = (
            verify_out_cache_loc.device
            if verify_out_cache_loc is not None
            else step.seq_lens.device
        )
        accept_index = accept_index.to(device)
        accept_lens = pp_outputs["spec_accept_lens"].to(device)

        # The last stage already commits inside run_eagle_verify.
        if not self.pp_group.is_last_rank:
            commit_mamba_states_after_verify(
                self.tp_worker,
                step,
                accept_lens,
                accept_index,
                get_spec().speculative_num_draft_tokens,
            )

        if verify_out_cache_loc is None:
            return

        if get_spec().speculative_eagle_topk > 1:
            move_accept_tokens_to_target_kvcache(
                derive_batch(step, out_cache_loc=verify_out_cache_loc),
                accept_index,
                accept_lens - 1,
                self.token_to_kv_pool_allocator,
            )

    def _send_output_to_next_stage(
        self,
        next_first_rank_mb_id: int,
        slots: List[RingSlot],
        last_rank_comm_queue: deque,
        pp_outputs: PPProxyTensors | None,
    ) -> List[P2PWork]:
        send_output_work = []
        if self.pp_group.is_last_rank:
            # send ready PP output to rank 0
            record = slots[next_first_rank_mb_id].record
            if record is not None:
                q_event, pp_outputs_to_send = last_rank_comm_queue.popleft()
                if not _pp_can_skip_output_comm(record.step):
                    with torch.profiler.record_function("send_res_dict_to_next_stage"):
                        send_output_work = self.channel.send_dict(
                            pp_outputs_to_send.tensors,
                            async_send=True,
                            msg_type="output",
                            ready_event=q_event,
                        )
        # send the outputs from the last round to let the next stage worker run post processing
        if not self.pp_group.is_last_rank:
            if pp_outputs:
                with torch.profiler.record_function("send_res_dict_to_next_stage"):
                    send_output_work = self.channel.send_dict(
                        pp_outputs.tensors,
                        async_send=True,
                        msg_type="output",
                    )
        return send_output_work

    def _send_recv_and_preprocess_output_tensors(
        self,
        next_first_rank_mb_id: int,
        next_mb_id: int,
        slots: List[RingSlot],
        last_rank_comm_queue: deque[Tuple[torch.Event, PPProxyTensors]],
        pp_outputs: PPProxyTensors | None,
    ) -> Tuple[
        Optional[PPProxyTensors],
        Optional[GenerationBatchResult],
        Optional[torch.Event],
        List[P2PWork],
    ]:
        next_pp_outputs = None
        d2h_event = None
        batch_result = None
        send_output_work = []

        # On NPU (HCCL), isend/irecv may block until a matching peer op is
        # posted, so the parity-based send-first/recv-first ordering used
        # for NPU is replaced by batch_isend_irecv which submits all
        # send/recv operations atomically.
        if _is_npu and get_parallel().pp_size == 2:
            return self._pp2_only_send_recv_output_tensors_npu(
                next_first_rank_mb_id,
                next_mb_id,
                slots,
                last_rank_comm_queue,
                pp_outputs,
            )

        # Order send/recv by pp_rank parity (even: send->recv, odd: recv->send)
        # on every backend. A device send only completes on its own while the
        # peer's per-connection NCCL buffer has free slots (~8 ops over NVLink,
        # ~2 over NET/IB, i.e. any cross-node PP); beyond that it waits for the
        # peer's matching recv, and P2P ops are serialized on one stream per
        # peer pair. With two microbatches in flight, if every rank sends first
        # each rank's output sends queue ahead of the recv its peer's sends
        # need, and the ring deadlocks. Parity gives each adjacent pair one
        # rank that posts its recv first, which breaks the cycle for any
        # pp_size > 1 (an odd ring still has at least one recv-first rank).
        send_first = (get_parallel().pp_rank % 2) == 0

        def _do_send():
            return self._send_output_to_next_stage(
                next_first_rank_mb_id,
                slots,
                last_rank_comm_queue,
                pp_outputs,
            )

        def _do_recv():
            nonlocal next_pp_outputs, batch_result, d2h_event
            record = slots[next_mb_id].record
            if record is None:
                return
            target = record.step
            if _pp_can_skip_output_comm(target):
                next_pp_outputs, batch_result, d2h_event = (
                    self._make_skip_output_result(target, record.result)
                )
                return
            with torch.profiler.record_function("recv_res_dict_from_prev_stage"):
                tensor_dict, output_recv_event = self.channel.recv_output()
                next_pp_outputs = PPProxyTensors(tensor_dict)
            with self.copy_stream_ctx:
                self.copy_stream.wait_stream(self.schedule_stream)
                if output_recv_event is not None:
                    self.copy_stream.wait_event(output_recv_event)
                batch_result = self._prep_batch_result(
                    target, record.result, next_pp_outputs
                )
                d2h_event = self.device_module.Event()
                if (
                    batch_result.logits_output is not None
                    and batch_result.logits_output.sampling_mask_output is not None
                ):
                    batch_result.copy_done = d2h_event
                    batch_result.copy_to_cpu(
                        return_logprob=target.return_logprob,
                        return_hidden_states=False,
                    )
                else:
                    d2h_event.record(self.device_module.current_stream())

        if send_first:
            send_output_work = _do_send()
            _do_recv()
        else:
            _do_recv()
            send_output_work = _do_send()

        return next_pp_outputs, batch_result, d2h_event, send_output_work

    def _pp2_only_send_recv_output_tensors_npu(
        self,
        next_first_rank_mb_id: int,
        next_mb_id: int,
        slots: List[RingSlot],
        last_rank_comm_queue: deque[Tuple[torch.Event, PPProxyTensors]],
        pp_outputs: PPProxyTensors | None,
    ) -> Tuple[
        Optional[PPProxyTensors],
        Optional[GenerationBatchResult],
        Optional[torch.Event],
        List[P2PWork],
    ]:
        """NPU-specific output tensor send/recv using batch_isend_irecv.

        Pairs the send of output tensors to the next stage with the recv
        of output tensors from the previous stage in a single
        ``batch_isend_irecv`` call, avoiding the deadlock that can occur
        on HCCL when separate isend/irecv calls block waiting for each
        other.
        """
        next_pp_outputs = None
        d2h_event = None
        batch_result = None
        send_output_work = []

        all_gather_group = self.attn_tp_group

        # ---- Prepare send dict ----
        # On NPU, always send something (full output or a lightweight skip
        # marker) so the peer's recv in batch_isend_irecv always has a
        # matching send.  Without this, SGLANG_PP_SKIP_PURE_CHUNKED_OUTPUT_COMM
        # would cause asymmetric skip decisions between adjacent ranks
        # (send target and recv target are different micro-batches), leading
        # to deadlock or forcing the user to disable the optimisation
        # entirely (≈10 % throughput loss).
        send_dict: Optional[Dict[str, torch.Tensor]] = None
        if self.pp_group.is_last_rank:
            record_send = slots[next_first_rank_mb_id].record
            if record_send is not None:
                q_event, pp_outputs_to_send = last_rank_comm_queue.popleft()
                if _pp_can_skip_output_comm(record_send.step):
                    send_dict = {"__msg_type__": "output", "__skip__": True}
                else:
                    self.device_module.current_stream().wait_event(q_event)
                    send_dict = dict(pp_outputs_to_send.tensors)
                    send_dict["__msg_type__"] = "output"
        elif pp_outputs:
            if pp_outputs.tensors.get("__skip__"):
                send_dict = {"__msg_type__": "output", "__skip__": True}
            else:
                send_dict = dict(pp_outputs.tensors)
                send_dict["__msg_type__"] = "output"

        # ---- Determine recv conditions ----
        # NOTE: skip_recv is NOT computed here.  The skip decision is made
        # by the sender (last rank) and communicated via the __skip__
        # marker.  The receiver always participates in the
        # batch_isend_irecv and inspects the marker after recv.
        record_recv = slots[next_mb_id].record
        should_recv = record_recv is not None

        def _handle_recv_dict(recv_dict):
            nonlocal next_pp_outputs, batch_result, d2h_event
            if recv_dict.get("__skip__"):
                # _make_skip_output_result returns next_pp_outputs=None
                # (correct for the non-NPU path where the skip propagates
                # via pp_outputs=None).  On NPU we must propagate the skip
                # marker through the ring, so override it here.
                _, batch_result, d2h_event = self._make_skip_output_result(
                    record_recv.step, record_recv.result
                )
                next_pp_outputs = PPProxyTensors(recv_dict)
            else:
                next_pp_outputs = PPProxyTensors(recv_dict)
                with self.copy_stream_ctx:
                    self.copy_stream.wait_stream(self.schedule_stream)
                    batch_result = self._prep_batch_result(
                        record_recv.step, record_recv.result, next_pp_outputs
                    )
                    d2h_event = self.device_module.Event()
                    d2h_event.record(self.device_module.current_stream())

        # ---- Execute communication ----
        if send_dict is not None and should_recv:
            # Paired send + recv via batch_isend_irecv
            with torch.profiler.record_function("send_recv_res_dict"):
                recv_dict = self.pp_group.send_recv_tensor_dict(
                    send_tensor_dict=send_dict,
                    send_all_gather_group=all_gather_group,
                    recv_all_gather_group=all_gather_group,
                )
            _handle_recv_dict(recv_dict)
        elif send_dict is not None:
            # Send only (recv not needed — target is None or prebuilt)
            send_output_work = self.channel.send_dict(
                send_dict, async_send=True, msg_type="output"
            )
        elif should_recv:
            # Recv only (no send needed)
            with torch.profiler.record_function("recv_res_dict_from_prev_stage"):
                recv_dict, _ = self.channel.recv_output()
            _handle_recv_dict(recv_dict)

        return next_pp_outputs, batch_result, d2h_event, send_output_work

    def run_slot_batch(
        self,
        slot: ScheduleSlot,
        step: ScheduleBatch,
        pp_proxy_tensors: Optional[PPProxyTensors],
    ) -> GenerationBatchResult:
        """Run a ring slot's forward behind the PP stream fences; the last
        rank buffers its outputs for the async batch depth."""
        with torch.profiler.record_function("run_batch"):
            with self.forward_stream_ctx:
                self.channel.wait_forward_dependencies()
                set_time_batch(
                    step.reqs,
                    "set_run_batch_cpu_start_time",
                    trace_only=True,
                )
                result = self.ops.run_batch(step, pp_proxy_tensors)
                set_time_batch(
                    step.reqs,
                    "set_run_batch_cpu_end_time",
                    trace_only=True,
                    attrs={"pp_mb_id": slot.index},
                )
                self.channel.launch_event = self.device_module.Event()
                self.channel.launch_event.record(self.device_module.current_stream())
                if self.pp_group.is_last_rank:
                    # (last rank) buffer the outputs for async batch depth
                    self.last_rank_comm_queue.append(
                        (
                            self.channel.launch_event,
                            PPProxyTensors(self._prepare_tensor_dict(result, step)),
                        )
                    )
        return result
