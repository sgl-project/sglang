"""Fixed-TP VPP serving: admit a wave, execute local passes, process its outputs."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from sglang.srt.disaggregation.base.conn import KVPoll
from sglang.srt.disaggregation.mode import DisaggregationMode
from sglang.srt.distributed.vpp_activation import (
    VPPActivationReceiver,
    VPPSendWindow,
)
from sglang.srt.distributed.vpp_batch import VPPBatchLedger, VPPBatchState
from sglang.srt.distributed.vpp_schedule import VPPFixedSchedule, VPPTask
from sglang.srt.managers.scheduler_components.state_check import CheckPhase
from sglang.srt.managers.scheduler_pp_mixin import PPBatchMetadata
from sglang.srt.mem_cache.static_kv_storage import kv_storage_of
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, PPProxyTensors
from sglang.srt.runtime_context import get_exec, get_parallel
from sglang.srt.utils import DynamicGradMode

if TYPE_CHECKING:
    from sglang.srt.managers.scheduler import Scheduler

# Only CPU metadata/payloads are tagged; NCCL payloads follow the FIFO stream.
_VPP_ACTIVATION_TAG = 2


def _vpp_manifest(batch) -> tuple:
    return (
        batch.forward_mode.value,
        tuple(
            (
                req.rid,
                getattr(req, "session_generation", None),
                None if req.extend_end is None else req.prefix_len,
                req.extend_end,
            )
            for req in batch.reqs
        ),
        tuple(batch.seq_lens_cpu.tolist()),
        tuple(batch.extend_lens or ()),
    )


def _snapshot_tensors(tensors: dict) -> dict:
    # Both activations and logits can be views of model/backend workspaces.
    # Clone on the producer stream before recording the readiness event.
    return {
        key: value.clone() if isinstance(value, torch.Tensor) else value
        for key, value in tensors.items()
    }


class NativeVPPWaveMixin:
    def init_vpp_loop_state(self: Scheduler) -> None:
        layout = self.tp_worker.model_runner.model.vpp_layout
        parallel = get_parallel()
        if (
            layout.physical_size != parallel.pp_size
            or parallel.pp_size < 2
            or parallel.attn_dp_size != 1
            or parallel.attn_cp_size != 1
            or parallel.dp_size != 1
            or not str(self.device).startswith("cuda")
        ):
            raise NotImplementedError("VPP serving requires CUDA PP>1 with DP1/CP1")
        if (
            not self.is_generation
            or not self.spec_algorithm.is_none()
            or self.enable_overlap
            or self.enable_lora
            or self.enable_hierarchical_cache
            or self.enable_hisparse
            or self.dllm_config is not None
            or self.enable_unified_cache_external_linker
            or self.uocr_runtime is not None
            or get_exec().features.enable_return_routed_experts
        ):
            raise NotImplementedError(
                "VPP serving currently supports ordinary generation without "
                "speculation, overlap scheduling, LoRA, HiCache or output capture"
            )
        self.init_pp_loop_state()
        self.vpp_schedule = VPPFixedSchedule(
            layout, parallel.pp_rank, self.pp_loop_size
        )
        self.vpp_batches = VPPBatchLedger(layout, parallel.pp_rank, self.pp_loop_size)
        reverse_group = self.pp_group.vpp_reverse_group
        recv_group = send_group = self.pp_group
        if layout.physical_size == 2:
            if reverse_group is None:
                raise RuntimeError("PP2 VPP requires a separate reverse transport")
            if self.pp_group.is_first_rank:
                recv_group = reverse_group
            else:
                send_group = reverse_group
        self.vpp_receiver = VPPActivationReceiver(recv_group, tag=_VPP_ACTIVATION_TAG)
        self.vpp_sends = VPPSendWindow(
            send_group, self.pp_loop_size, tag=_VPP_ACTIVATION_TAG
        )
        self._vpp_next_batch_seq = 0
        self._vpp_next_slot = 0

    @DynamicGradMode()
    def event_loop_vpp(self: Scheduler) -> None:
        self.init_vpp_loop_state()
        while True:
            wave = self._vpp_admit_wave()
            if not wave:
                self._sched_idled = True
                self._pp_answer_verdict_requests()
                if self._vpp_pd_queues_empty():
                    self.on_idle()
                continue
            self._vpp_execute_wave(wave)
            self._vpp_process_wave(wave)

    def _vpp_pd_queues_empty(self: Scheduler) -> bool:
        if self.disaggregation_mode == DisaggregationMode.PREFILL:
            return not (
                self.disagg_prefill_bootstrap_queue.queue
                or self.disagg_prefill_transfer_queue
            )
        if self.disaggregation_mode == DisaggregationMode.DECODE:
            return not (
                self.waiting_queue
                or self.disagg_decode_prealloc_queue.queue
                or self.disagg_decode_prealloc_queue.retracted_queue
                or self.disagg_decode_transfer_queue.queue
                or self.disagg_decode_transfer_queue.has_pending_deferred_releases()
            )
        return True

    def _vpp_pd_consensus(self: Scheduler, collect):
        """Finish the existing PP reduction before any rank changes a queue."""
        outcome = collect()
        if not self.pp_group.is_last_rank:
            self._pp_commit_comm_work(
                self._pp_send_pyobj_to_next_stage(outcome, async_send=True)
            )
        return self.pp_group.broadcast_object(
            outcome if self.pp_group.is_last_rank else None,
            src=self.pp_group.world_size - 1,
        )

    def _vpp_pd_apply(self: Scheduler, process, outcome) -> None:
        """Relay actual admission/retirement outcomes, including local shortfalls.

        As in ordinary PP, later ranks consume what the preceding rank actually
        processed, rather than independently allocating from a stale poll.
        """
        if not self.pp_group.is_first_rank:
            outcome = self._pp_recv_pyobj_from_prev_stage()
        outcome = process(outcome)
        if not self.pp_group.is_last_rank:
            self._pp_commit_comm_work(
                self._pp_send_pyobj_to_next_stage(outcome, async_send=True)
            )
        # Fence all queue mutations before the wave's capacity agreement.
        self.pp_group.broadcast_object(
            outcome if self.pp_group.is_last_rank else None,
            src=self.pp_group.world_size - 1,
        )

    def _vpp_pd_release_prefill(self: Scheduler, outcomes):
        released = self.process_disagg_prefill_transfer_queue(outcomes)
        return {req.rid: outcomes[req.rid] for req in released}

    def _vpp_pd_get_decode_transferred_ids(self: Scheduler):
        queue = self.disagg_decode_transfer_queue
        # Raw transport Success does not imply that staged GPU scatters or
        # metadata publication have finished. Vote on the admission gate itself.
        polls = (
            queue._poll_with_staging()
            if queue.enable_staging
            else queue._poll_with_metadata_gate()
        )
        outcomes = {
            req.req.rid: poll == KVPoll.Failed
            for req, poll in zip(queue.queue.items(), polls)
            if poll in (KVPoll.Success, KVPoll.Failed)
        }
        if self.pp_group.is_first_rank:
            return outcomes
        previous = self._pp_recv_pyobj_from_prev_stage()
        return {
            rid: failed or previous[rid]
            for rid, failed in outcomes.items()
            if rid in previous
        }

    def _vpp_pd_advance(self: Scheduler) -> None:
        """PD owns queues only while no VPP batch is retained or executing."""
        if self.disaggregation_mode == DisaggregationMode.PREFILL:
            self.process_bootstrapped_queue()
            outcomes = self._vpp_pd_consensus(self._pp_pd_get_prefill_transferred_ids)
            self._vpp_pd_apply(self._vpp_pd_release_prefill, outcomes)
        elif self.disaggregation_mode == DisaggregationMode.DECODE:
            self.disagg_decode_transfer_queue.resolve_deferred_releases()
            self.disagg_decode_prealloc_queue.publish_pending_destinations()
            self.disagg_decode_transfer_queue.publish_pending_metadata()
            if self.pause_state.may_launch():
                retract = self._vpp_pd_consensus(
                    lambda: self._pp_pd_get_retract_ids(self._vpp_next_slot)
                )
                self._vpp_pd_apply(self.process_retract_queue, retract)
                prealloc = self._vpp_pd_consensus(self._pp_pd_get_prealloc_ids)
                self._vpp_pd_apply(self.process_prealloc_queue, prealloc)
                transferred = self._vpp_pd_consensus(
                    self._vpp_pd_get_decode_transferred_ids
                )
                self._vpp_pd_apply(self.process_decode_transfer_queue, transferred)

    def _vpp_plan_batch(self: Scheduler):
        if self.disaggregation_mode == DisaggregationMode.PREFILL:
            self.process_pending_chunked_abort()
            self.process_prefill_chunk(
                last_batch=self.last_batch, running_batch=self.running_batch
            )
            plan = self.get_new_batch_prefill(self.running_batch)
            plan.running_batch.batch_is_full = False
        elif self.disaggregation_mode == DisaggregationMode.DECODE:
            plan = self.get_next_disagg_decode_batch_to_run(
                running_batch=self.running_batch
            )
            self._flush_staging_scatters()
        else:
            plan = self.get_next_batch_to_run(
                running_batch=self.running_batch, last_batch=self.last_batch
            )
        return plan

    def _vpp_admit_wave(self: Scheduler) -> list[VPPBatchState]:
        """All physical ranks build the same batches using existing PP input relay.

        No allocation, input dispatch, abort processing or slot reuse happens
        during execution. A chunked request closes this wave immediately so its
        next chunk cannot mutate a retained request before output processing.
        """
        wave = []
        self._pp_mark_capacity_digest()
        recv_reqs = self._pp_ingest_requests()
        self._pp_check_capacity_digest()
        if not self.pp_group.is_last_rank:
            self._pp_commit_comm_work(self.send_req_work)
            self.send_req_work = self._pp_send_reqs_to_next_stage(recv_reqs)
            self._pp_send_capacity_digest()
        self.running_batch = self.running_mbs[self._vpp_next_slot]
        self.last_batch = self.last_mbs[self._vpp_next_slot]
        self._vpp_pd_advance()
        # Inputs (including aborts and pause) are dispatched only before any
        # batch is retained. Later slots consume the already replicated queue.
        for _ in range(self.pp_loop_size):
            slot_id = self._vpp_next_slot
            self._vpp_next_slot = (slot_id + 1) % self.pp_loop_size
            self.increment_scheduler_iterations()
            self.running_batch = self.running_mbs[slot_id]
            self.last_batch = self.last_mbs[slot_id]
            batch = None
            if self.pause_state.may_launch():
                plan = self._vpp_plan_batch()
                self.running_batch = plan.running_batch
                batch = plan.batch_to_run
            self.running_mbs[slot_id] = self.running_batch
            manifest = None if batch is None else _vpp_manifest(batch)
            # PP agreement only. TP lanes never vote on device readiness/tasks.
            manifests = self.pp_group.all_gather_object(manifest)
            if any(item != manifest for item in manifests):
                raise RuntimeError("VPP physical ranks built different batches")
            if batch is None:
                continue
            if batch.return_hidden_states or any(
                req.beam_group is not None for req in batch.reqs
            ):
                raise NotImplementedError("VPP hidden-state output and beam search")
            if any(
                req.rid in {old.rid for state in wave for old in state.batch.reqs}
                for req in batch.reqs
            ):
                raise RuntimeError("A request cannot belong to two VPP wave batches")
            state = VPPBatchState(self._vpp_next_batch_seq, slot_id, manifest, batch)
            self._vpp_next_batch_seq += 1
            self.vpp_batches.admit(state)
            self.mbs[slot_id] = batch
            wave.append(state)
            if self.pp_group.is_first_rank:
                self.vpp_schedule.enqueue(VPPTask(state.batch_seq, 0))
            if self.chunked_req is not None or batch.chunked_req is not None:
                break
        return wave

    def _vpp_execute_wave(self: Scheduler, wave: list[VPPBatchState]) -> None:
        layout = self.vpp_schedule.layout
        stages = layout.stages_for_rank(get_parallel().pp_rank)
        receives_left = len(wave) * (len(stages) - int(self.pp_group.is_first_rank))
        if receives_left:
            self.vpp_receiver.start()
        for stage in stages:
            for state in wave:
                task = VPPTask(state.batch_seq, stage.stage_id)
                activation = None
                if task.stage_id != 0:
                    # Rank zero chooses its FIFO head before waiting for input.
                    if self.pp_group.is_first_rank:
                        if self.vpp_schedule.next_task() != task:
                            raise RuntimeError("VPP rank-zero wave order mismatch")
                    receives_left -= 1
                    activation = self.vpp_receiver.take(post_next=receives_left > 0)
                    self.vpp_batches.check_activation(task, activation.tensors)
                    if not self.pp_group.is_first_rank:
                        self.vpp_schedule.enqueue(task)
                if self.vpp_schedule.next_task() != task:
                    raise RuntimeError("VPP task does not match the FIFO head")
                self._vpp_launch_stage(task, state, activation)
                self.vpp_schedule.record_dispatch(task)
                self.vpp_batches.record_dispatch(task)
        # Output collectives start after all local activation operations were
        # submitted. Fence readers on the forward stream before those collectives.
        with self.forward_stream_ctx:
            fence = self.vpp_sends.wait_for_reuse()
            if fence is not None:
                self.forward_stream.wait_event(fence)

    def _vpp_launch_stage(self: Scheduler, task, state, activation) -> None:
        batch = state.batch
        self.cur_batch_for_debug = batch
        final_local = task.stage_id == self.vpp_batches.stages[-1].stage_id
        final_global = task.stage_id == self.vpp_schedule.layout.logical_size - 1
        fence = None if final_global else self.vpp_sends.acquire()
        if state.forward_batch is None and getattr(self, "enable_staging", False):
            self.maybe_prefetch_staging_for_batch(batch)
        with self.forward_stream_ctx:
            self.forward_stream.wait_stream(self.schedule_stream)
            if fence is not None:
                self.forward_stream.wait_event(fence)
            if activation is not None:
                activation.wait_on(self.forward_stream)
            if state.forward_batch is None:
                launch = self._on_step_launch(batch)
                self.resolve_forward_inputs(batch, self.future_map)
                self.tp_worker.set_hicache_consumer(batch.hicache_consumer_index)
                state.forward_batch = ForwardBatch.init_new(
                    batch,
                    self.tp_worker.model_runner,
                    return_hidden_states_before_norm=False,
                )
                state.metadata = PPBatchMetadata(
                    can_run_cuda_graph=False,
                    forward_launch=launch,
                    forward_done=self.device_module.Event(),
                )
                # Register once per batch. This event is recorded at the final
                # local pass; admission has finished and the pool is untouched
                # until the whole wave's outputs have been synchronized.
                self._on_step_forward_recorded(state.metadata.forward_done)
            proxy = (
                None
                if activation is None
                else PPProxyTensors(
                    {
                        key: value
                        for key, value in activation.tensors.items()
                        if key != "vpp_identity"
                    }
                )
            )
            result = self.tp_worker.forward_batch_generation(
                None,
                forward_batch=state.forward_batch,
                pp_proxy_tensors=proxy,
                vpp_stage_id=task.stage_id,
            )
            if final_global:
                result.extend_input_len_per_req = (
                    list(batch.extend_lens) if batch.forward_mode.is_extend() else None
                )
                result.extend_logprob_start_len_per_req = (
                    batch.extend_logprob_start_lens if batch.return_logprob else None
                )
                state.output_tensors = _snapshot_tensors(
                    self._pp_prepare_tensor_dict(result, batch)
                )
            else:
                if result.pp_hidden_states_proxy_tensors is None:
                    raise RuntimeError("A non-final VPP stage produced no activation")
                tensors = _snapshot_tensors(
                    result.pp_hidden_states_proxy_tensors.tensors
                )
                tensors["vpp_identity"] = (
                    task.batch_seq,
                    task.stage_id + 1,
                    state.slot_id,
                    state.manifest,
                )
            event = self.device_module.Event()
            event.record(self.forward_stream)
            if final_local:
                state.metadata.forward_done.record(self.forward_stream)
                if self.enable_unified_memory:
                    with self.device_module.stream(self.schedule_stream):
                        kv_storage_of(
                            self.req_to_token_pool, self.token_to_kv_pool_allocator
                        ).record_forward(
                            state.metadata.forward_done,
                            written_tokens=batch.out_cache_loc,
                            token_writes_unknown=False,
                            req_pool_indices=batch.req_pool_indices,
                            state_slots=(),
                            state_writes_unknown=False,
                        )
                    state.metadata.unified_forward_done = state.metadata.forward_done
        if not final_global:
            self.vpp_sends.send(tensors, forward_event=event)
        self.launch_event = event
        # Coarse WAR barrier: next workspace/index writes follow this stage.
        self.schedule_stream.wait_event(event)

    def _vpp_process_wave(self: Scheduler, wave: list[VPPBatchState]) -> None:
        for state in wave:
            with self.forward_stream_ctx:
                tensors = self.pp_group.broadcast_tensor_dict(
                    state.output_tensors,
                    src=self.pp_group.world_size - 1,
                )
                output_ready = self.device_module.Event()
                output_ready.record(self.forward_stream)
            with self.copy_stream_ctx:
                self.copy_stream.wait_event(output_ready)
                self.copy_stream.wait_event(state.metadata.forward_done)
                result = self._pp_prep_batch_result(
                    state.batch, state.metadata, PPProxyTensors(tensors)
                )
                result.copy_done = self.device_module.Event()
                result.copy_to_cpu(
                    return_logprob=state.batch.return_logprob,
                    return_hidden_states=False,
                )
            result.copy_done.synchronize()
            self._pp_process_batch_result(state.batch, result)
            self.last_mbs[state.slot_id] = state.batch
            self.mbs[state.slot_id] = None
            self.mb_metadata[state.slot_id] = None
            self.vpp_batches.complete(state.batch_seq)
            self.invariant_checker.run(CheckPhase.BUSY, state.batch)
        self._pp_answer_verdict_requests()
