"""Scheduling VPP policy adapted to eager stage and request contracts."""

from __future__ import annotations

import hashlib
import pickle
from contextlib import contextmanager

import torch

from sglang.srt.disaggregation.base.conn import KVPoll
from sglang.srt.disaggregation.mode import DisaggregationMode
from sglang.srt.distributed.vpp_policy import PipelineControlEnvelope
from sglang.srt.managers.req_types import FINISH_ABORT
from sglang.srt.managers.schedule_batch import ScheduleBatch
from sglang.srt.managers.scheduler_pp_mixin import PPBatchMetadata, SchedulerPPMixin
from sglang.srt.managers.scheduler_vpp_core import SchedulingVPPCore
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, PPProxyTensors
from sglang.srt.model_executor.forward_mode import ForwardMode
from sglang.srt.runtime_context import get_exec, get_parallel
from sglang.srt.sampling.penalties import StepPenalties
from sglang.srt.sampling.watermarking.runtime import build_watermark_rows
from sglang.srt.utils import DynamicGradMode


def _snapshot_tensors(tensors: dict) -> dict:
    # Eager outputs and exported source pages own storage. Only backend index
    # and candidate metadata can be overwritten by the next virtual stage.
    return {
        key: value.clone()
        if isinstance(value, torch.Tensor)
        and key.startswith(("vpp_index_", "vpp_candidate_"))
        else value
        for key, value in tensors.items()
    }


class SchedulerVPPMixin(SchedulingVPPCore):
    @DynamicGradMode()
    def event_loop_vpp(self):
        self._vpp_layout = self.tp_worker.model_runner.model.vpp_layout
        if self._vpp_layout.physical_size != 2 or self._vpp_layout.virtual_stages != 2:
            raise NotImplementedError("VPP scheduling requires PP2/VPP2")
        parallel = get_parallel()
        if (
            parallel.attn_dp_size != 1
            or parallel.attn_cp_size != 1
            or parallel.dp_size != 1
            or not str(self.device).startswith("cuda")
            or not self.is_generation
            or not self.spec_algorithm.is_none()
            or self.enable_overlap
            or self.enable_lora
            or self.enable_unified_memory
            or self.enable_hierarchical_cache
            or self.enable_hisparse
            or self.dllm_config is not None
            or self.enable_unified_cache_external_linker
            or self.uocr_runtime is not None
            or get_exec().features.enable_return_routed_experts
        ):
            raise NotImplementedError(
                "VPP scheduling requires eager ordinary generation with DP1/CP1"
            )
        return self._event_loop_vpp_rank_local()

    def _pp_vpp_max_inflight(self):
        return 2

    def _pp_vpp_available_prefill_tokens(self):
        stats = self.pool_stats_observer.get_pool_stats()
        return min(
            [
                stats.full_available_size + stats.full_evictable_size,
                *(
                    window.available_size + window.evictable_size
                    for window in stats.windows
                ),
            ]
        )

    def _pp_vpp_req_retained(self, req):
        return any(
            batch is not None and any(row is req for row in batch.reqs)
            for batch in self.mbs
        )

    @contextmanager
    def _pp_vpp_select_chunk(self, request_ids=None, *, include_retained=False):
        # Keep ownership in chunked_reqs. Only the planning/result-processing
        # view changes, and a newly admitted chunk is visible within the scope.
        previous = getattr(self, "_vpp_chunk_selection", None)
        self._vpp_chunk_selection = (request_ids, include_retained)
        try:
            yield
        finally:
            self._vpp_chunk_selection = previous

    def _pp_vpp_selected_chunk(self):
        request_ids, include_retained = self._vpp_chunk_selection
        return next(
            (
                req
                for req in self.chunked_reqs
                if (request_ids is None or req.rid in request_ids)
                and (include_retained or not self._pp_vpp_req_retained(req))
            ),
            None,
        )

    def _init_scheduling_native_loop_state(self):
        SchedulerPPMixin.init_pp_loop_state(self)
        self.pp_loop_size = 2
        self.mbs = [None] * 2
        self.last_mbs = [None] * 2
        self.running_mbs = [
            ScheduleBatch(reqs=[], batch_is_full=False) for _ in range(2)
        ]
        self.mb_metadata = [None] * 2
        self._vpp_forward_batches = {}
        # Initialize both directed activation communicators before entering the
        # asynchronous control loop; first-use NCCL initialization is collective.
        probe = torch.zeros(1, dtype=torch.int32, device=self.device)
        for group in (
            self.pp_group,
            self.pp_group.vpp_reverse_group,
            self.pp_group.vpp_activation_tp_group,
        ):
            if group is None:
                raise RuntimeError("Missing native reverse activation group")
            torch.distributed.all_reduce(probe, group=group.device_group)
        self.device_module.synchronize()

    def _pp_vpp_start_receiver(self):
        # Scheduling fixed TP uses one FIFO per TP lane.
        return

    def _pp_vpp_activation_group(self, source_rank):
        if source_rank == 1:
            group = self.pp_group.vpp_reverse_group
            if group is None:
                raise RuntimeError("PP2 requires the main reverse activation group")
            return group
        return self.pp_group

    def _pp_vpp_layout_digest(self):
        # Hash the authoritative main layout, including every loaded layer range.
        return hashlib.sha256(
            pickle.dumps(
                tuple(
                    (s.stage_id, s.physical_rank, s.start_layer, s.end_layer)
                    for s in self._vpp_layout.stages
                ),
                protocol=4,
            )
        ).hexdigest()

    def _pp_vpp_start_control_receiver(self):
        if (
            self.attn_tp_group.rank_in_group == 0
            and self._pp_vpp_pending_control_recv is None
        ):
            self._pp_vpp_pending_control_recv = self.pp_group.recv_tensor_dict_async(
                tag=2
            )

    def _pp_vpp_poll_control_receiver(self):
        handle = self._pp_vpp_pending_control_recv
        if handle is None:
            self._pp_vpp_start_control_receiver()
            handle = self._pp_vpp_pending_control_recv
        if not handle.poll_metadata_ready():
            return None
        wire = handle.wait()  # Control messages contain CPU payloads only.
        envelope = PipelineControlEnvelope.from_dict(wire)
        if (
            envelope.protocol_version != 1
            or envelope.runtime_epoch != self._pp_vpp_runtime_epoch
            or envelope.layout_digest != self._pp_vpp_control_layout_digest
        ):
            raise RuntimeError("VPP control identity mismatch")
        self._pp_vpp_pending_control_recv = None
        self._pp_vpp_start_control_receiver()
        return envelope, wire

    def _pp_vpp_flush_control_outbox(self, pending_work, max_pending):
        if self.attn_tp_group.rank_in_group != 0:
            return
        self._pp_vpp_reap_send_work(pending_work)
        while self._pp_vpp_control_outbox and len(pending_work) < max_pending:
            work = self.pp_group.send_tensor_dict(
                self._pp_vpp_control_outbox.popleft(), async_send=True, tag=2
            )
            if work:
                pending_work.append(work)

    def _pp_vpp_reap_send_work(self, pending_work):
        # Poll CUDA completion on the stream that owns the sends.
        with self.pp_comm_stream_ctx:
            return super()._pp_vpp_reap_send_work(pending_work)

    def _vpp_transfer_outcomes(self):
        succeeded, failed = self.get_rids(
            self.disagg_prefill_transfer_queue, True, [KVPoll.Success], [KVPoll.Failed]
        )
        result = dict.fromkeys(succeeded, False)
        result.update(dict.fromkeys(failed, True))
        return result

    def _pp_vpp_decode_poll(self, operation, slot_id):
        self.disagg_decode_transfer_queue.resolve_deferred_releases()
        self.disagg_decode_prealloc_queue.publish_pending_destinations()
        self.disagg_decode_transfer_queue.publish_pending_metadata()
        if operation == "retract":
            queue = self.disagg_decode_prealloc_queue.retracted_queue
            for req in queue:
                if req.retraction_mb_id is None:
                    req.retraction_mb_id = slot_id
            return [req.rid for req in queue if req.retraction_mb_id == slot_id]
        if operation == "prealloc":
            queue = self.disagg_decode_prealloc_queue.queue
            good, bad = self.get_rids(
                list(queue.items()), False, [KVPoll.WaitingForInput], [KVPoll.Failed]
            )
            return self._route_aborts_to_bad(
                good,
                bad,
                {
                    req.rid
                    for req in queue
                    if isinstance(req.finished_reason, FINISH_ABORT)
                },
            )
        if operation == "transfer":
            queue = self.disagg_decode_transfer_queue
            # Success must include staged scatters and metadata publication.
            polls = (
                queue._poll_with_staging()
                if queue.enable_staging
                else queue._poll_with_metadata_gate()
            )
            return {
                req.req.rid: poll == KVPoll.Failed
                for req, poll in zip(queue.queue.items(), polls)
                if poll in (KVPoll.Success, KVPoll.Failed)
            }
        raise RuntimeError(f"Invalid VPP decode queue operation {operation}")

    def _pp_vpp_decode_apply(self, operation, outcomes):
        if operation == "retract":
            return self.process_retract_queue(outcomes)
        if operation == "prealloc":
            return self.process_prealloc_queue(outcomes)
        if operation == "transfer":
            return self.process_decode_transfer_queue(outcomes)
        raise RuntimeError(f"Invalid VPP decode queue operation {operation}")

    def _pp_vpp_decode_queues_pending(self):
        return bool(
            self.disagg_decode_prealloc_queue.queue
            or self.disagg_decode_prealloc_queue.retracted_queue
            or self.disagg_decode_transfer_queue.queue
            or self.disagg_decode_transfer_queue.has_pending_deferred_releases()
        )

    def _pp_vpp_bootstrap_prefix_boundaries(self, rids):
        wanted = set(rids)
        return {
            req.rid: req.prefix_len
            for req in self.disagg_prefill_bootstrap_queue.queue
            if req.rid in wanted
        }

    def _pp_vpp_apply_bootstrap_prefix_boundaries(self, boundaries, rids):
        for req in self.disagg_prefill_bootstrap_queue.queue:
            if req.rid not in rids:
                continue
            boundary = int(boundaries[req.rid])
            req.vpp_prefix_limit = boundary
            if req.prefix_len > boundary:
                req.init_next_round_input(self.tree_cache)
            if req.prefix_len > boundary:
                raise RuntimeError("VPP bootstrap prefix exceeds consensus boundary")

    def _pp_vpp_try_apply_bootstrap_status(
        self, boundaries, remaining_good, remaining_bad
    ):
        self._pp_vpp_apply_bootstrap_prefix_boundaries(boundaries, remaining_good)
        outcomes = dict.fromkeys(remaining_good, False)
        outcomes.update(dict.fromkeys(remaining_bad, True))
        self.disagg_prefill_bootstrap_queue.move_bootstrapped(
            self.waiting_queue, outcomes=outcomes
        )
        queued = {req.rid for req in self.disagg_prefill_bootstrap_queue.queue}
        remaining_good.intersection_update(queued)
        remaining_bad.intersection_update(queued)
        return not remaining_good and not remaining_bad

    def _pp_vpp_register_prefix_batch(self, batch):
        if batch.forward_mode.is_decode():
            return
        for req, start, length in zip(batch.reqs, batch.prefix_lens, batch.extend_lens):
            generation = int(getattr(req, "session_generation", None) or 0)
            entry = self._pp_vpp_prefix_registry.get(req.rid, generation)
            self._pp_vpp_prefix_registry.plan(
                rid=req.rid,
                request_generation=generation,
                residency_generation=generation,
                start=int(start) if entry is None else entry.planned_end,
                end=int(start + length),
                required_stages=range(self._vpp_layout.logical_size),
            )
            if not hasattr(req, "_vpp_original_skip_radix_cache_insert"):
                req._vpp_original_skip_radix_cache_insert = req.skip_radix_cache_insert
            req.skip_radix_cache_insert = True

    def _pp_vpp_advance_prefix_mapping(self, rid, request_generation, end):
        req = self._pp_vpp_find_req(rid)
        if (
            req is None
            or int(getattr(req, "session_generation", None) or 0) != request_generation
        ):
            return
        if not req.kv.holds_kv or end <= req.prefix_len:
            return
        if end > req.kv.kv_allocated_len:
            raise RuntimeError("VPP prefix publication exceeds allocated request row")
        # Publication is tracked by the protocol registry. Main advances prefix_len
        # through checkpoint_disagg_prefill after global completion, so result
        # processing still sees the span that actually ran.

    def _pp_vpp_prepare_wavefront_batch(
        self, batch_seq, slot_batch_seqs, expected_manifest=None
    ):
        slot_id = batch_seq % len(self.mbs)
        if slot_batch_seqs[slot_id] is not None:
            raise RuntimeError("VPP admission slot still owned")
        request_ids = (
            None
            if expected_manifest is None
            else {rid for rid, _, _ in expected_manifest["requests"]}
        )
        self.running_batch = self.running_mbs[slot_id]
        self.last_batch = self.last_mbs[slot_id]
        if (
            expected_manifest is not None
            and expected_manifest.get("forward_mode") != ForwardMode.DECODE.value
        ):
            spans = {
                rid: (start, end) for rid, start, end in expected_manifest["requests"]
            }
            for req in self.waiting_queue:
                if req.rid in spans:
                    req.vpp_prefix_limit = spans[req.rid][0]
        self._vpp_admit_manifest = expected_manifest
        try:
            with self._pp_vpp_select_chunk(request_ids):
                if self.disaggregation_mode == DisaggregationMode.PREFILL:
                    self.process_pending_chunked_abort()
                    self.process_prefill_chunk(
                        last_batch=self.last_batch, running_batch=self.running_batch
                    )
                    plan = self.get_new_batch_prefill(self.running_batch)
                elif self.disaggregation_mode == DisaggregationMode.DECODE:
                    plan = self.get_next_disagg_decode_batch_to_run(
                        running_batch=self.running_batch
                    )
                    self._flush_staging_scatters()
                else:
                    plan = self.get_next_batch_to_run(
                        running_batch=self.running_batch, last_batch=self.last_batch
                    )
        finally:
            self._vpp_admit_manifest = None
        batch = plan.batch_to_run
        if self.disaggregation_mode == DisaggregationMode.PREFILL:
            batch = self.dp_attn_adapter.maybe_prepare_mlp_sync_batch(batch)
            batch = self.ngram_embedding_manager.prepare_for_forward(
                batch, chunked_req=batch.chunked_req if batch is not None else None
            )
        if batch is not None:
            if batch.return_hidden_states or any(
                req.beam_group is not None for req in batch.reqs
            ):
                raise NotImplementedError("VPP hidden-state output and beam search")
            batch.disagg_prefill_chunk_end_by_rid = (
                {}
                if batch.forward_mode.is_decode()
                else {
                    req.rid: min(int(start + length), len(req.origin_input_ids))
                    for req, start, length in zip(
                        batch.reqs, batch.prefix_lens or (), batch.extend_lens or ()
                    )
                }
            )
            for req in batch.reqs:
                req.vpp_prefix_limit = None
        self.running_batch = plan.running_batch
        self.running_mbs[slot_id] = plan.running_batch
        self.mbs[slot_id] = batch
        self.mb_metadata[slot_id] = None
        slot_batch_seqs[slot_id] = batch_seq
        return batch

    def _pp_vpp_execute_wavefront_action(self, action, slot_batch_seqs):
        if action is None:
            return False, []
        if slot_batch_seqs[action.slot_id] != action.batch_seq:
            raise RuntimeError("VPP action does not own its batch slot")
        batch = self.mbs[action.slot_id]
        if batch is None or action.physical_rank != self.pp_group.rank_in_group:
            raise RuntimeError("VPP action has wrong rank or empty batch")
        proxy = None
        if action.stage_id > 0:
            proxy = PPProxyTensors(self._pp_tensor_dict_inbox["vpp_proxy"].popleft())
            if (
                int(proxy.tensors["vpp_batch_seq"]) != action.batch_seq
                or int(proxy.tensors["vpp_stage_id"]) != action.stage_id
            ):
                raise RuntimeError("VPP stage input identity mismatch")
        result, event, send_work = self._pp_launch_vpp_stage(action, batch, proxy)
        self._pp_vpp_stage_events[(action.batch_seq, action.stage_id)] = event
        return True, send_work

    def _pp_launch_vpp_stage(self, action, batch, proxy):
        self.cur_batch_for_debug = batch
        self.running_batch = self.running_mbs[action.slot_id]
        self.last_batch = self.last_mbs[action.slot_id]
        final_local = (
            action.stage_id
            == self._vpp_layout.stages_for_rank(action.physical_rank)[-1].stage_id
        )
        final_global = action.stage_id == self._vpp_layout.logical_size - 1
        with self.forward_stream_ctx:
            self.forward_stream.wait_stream(self.schedule_stream)
            if proxy is not None:
                recv_event = proxy.tensors.pop("_vpp_local_recv_event", None)
                if recv_event is not None:
                    self.forward_stream.wait_event(recv_event)
                for value in proxy.tensors.values():
                    if isinstance(value, torch.Tensor) and value.is_cuda:
                        value.record_stream(self.forward_stream)
            if action.batch_seq not in self._vpp_forward_batches:
                if getattr(self, "enable_staging", False):
                    self.maybe_prefetch_staging_for_batch(batch)
                launch = self._on_step_launch(batch)
                self.resolve_forward_inputs(batch, self.future_map)
                # run_batch normally binds the request-slot sampling state.
                # VPP bypasses run_batch and retains this launch across stages.
                num_reqs = len(batch.reqs)
                batch.sampling_info.set_sampler_params(
                    self.sampler_params_pool.gather(
                        batch.req_pool_indices[:num_reqs],
                        batch.req_pool_indices_cpu[:num_reqs],
                    )
                )
                batch.sampling_info.step_penalties = StepPenalties.build(
                    batch.reqs, self.penalty_status_pool
                )
                batch.sampling_info.watermark = build_watermark_rows(
                    batch.reqs, batch.device
                )
                batch.sampling_info.set_plugin_state(batch.reqs)
                batch.sampling_info.set_seeds_and_mask_rows(batch.reqs)
                self.tp_worker.set_hicache_consumer(batch.hicache_consumer_index)
                self._vpp_forward_batches[action.batch_seq] = ForwardBatch.init_new(
                    batch,
                    self.tp_worker.model_runner,
                    return_hidden_states_before_norm=False,
                )
                metadata = PPBatchMetadata(
                    can_run_cuda_graph=False,
                    forward_launch=launch,
                    forward_done=self.device_module.Event(),
                )
                self.mb_metadata[action.slot_id] = metadata
                self._on_step_forward_recorded(metadata.forward_done)
            result = self.tp_worker.forward_batch_generation(
                None,
                forward_batch=self._vpp_forward_batches[action.batch_seq],
                pp_proxy_tensors=proxy,
                vpp_stage_id=action.stage_id,
            )
            if final_global:
                result.extend_input_len_per_req = list(batch.extend_lens or ())
                result.extend_logprob_start_len_per_req = (
                    batch.extend_logprob_start_lens if batch.return_logprob else None
                )
                tensors = (
                    self._pp_prepare_tensor_dict(result, batch)
                    if self.attn_tp_group.rank_in_group == 0
                    else None
                )
                tensors = self.attn_tp_group.broadcast_tensor_dict(tensors, src=0)
                tensors["vpp_batch_seq"] = action.batch_seq
            else:
                if result.pp_hidden_states_proxy_tensors is None:
                    raise RuntimeError(
                        "Nonfinal native VPP stage produced no activation"
                    )
                tensors = _snapshot_tensors(
                    result.pp_hidden_states_proxy_tensors.tensors
                )
                tensors.update(
                    __msg_type__="vpp_proxy",
                    vpp_batch_seq=action.batch_seq,
                    vpp_stage_id=action.stage_id + 1,
                    vpp_src_stage_id=action.stage_id,
                    vpp_generation=action.batch_seq // len(self.mbs),
                    vpp_protocol_version=1,
                    vpp_runtime_epoch=self._pp_vpp_runtime_epoch,
                    vpp_layout_digest=self._pp_vpp_control_layout_digest,
                )
                if action.stage_id == 0:
                    tensors["vpp_batch_manifest"] = batch._vpp_fixed_manifest
            event = self.device_module.Event()
            event.record(self.forward_stream)
            if final_local:
                self.mb_metadata[action.slot_id].forward_done.record(
                    self.forward_stream
                )
        send_work = []
        if final_global:
            self.last_rank_comm_queue.append((event, PPProxyTensors(tensors)))
        else:
            group = self._pp_vpp_activation_group(action.physical_rank)
            with self.pp_comm_stream_ctx:
                self.device_module.current_stream().wait_event(event)
                send_work = group.send_tensor_dict(
                    tensors,
                    all_gather_group=self.attn_tp_group,
                    async_send=True,
                    tag=1,
                    batch_p2p=True,
                )
        self.launch_event = event
        self.schedule_stream.wait_event(event)
        return result, event, send_work

    def _pp_vpp_finalize_rank_local_batch(
        self, batch_seq, slot_batch_seqs, output_tensors, release_slot=True
    ):
        slot_id = batch_seq % len(self.mbs)
        if slot_batch_seqs[slot_id] != batch_seq:
            raise RuntimeError("VPP completion does not own slot")
        batch = self.mbs[slot_id]
        metadata = self.mb_metadata[slot_id]
        if batch is None or metadata is None:
            raise RuntimeError("VPP completion missing batch or forward metadata")
        # Order local writes and the relayed output on the copy stream, then
        # fence once before processing results and retiring launch leases.
        for req in batch.reqs:
            generation = int(getattr(req, "session_generation", None) or 0)
            entry = self._pp_vpp_prefix_registry.get(req.rid, generation)
            if (
                entry is not None
                and entry.committed_end >= entry.planned_end
                and not entry.locked
            ):
                req.skip_radix_cache_insert = req._vpp_original_skip_radix_cache_insert
        with self.copy_stream_ctx:
            self.copy_stream.wait_stream(self.schedule_stream)
            self.copy_stream.wait_event(metadata.forward_done)
            output_tensors = {
                key: value.to(self.device, non_blocking=True)
                if isinstance(value, torch.Tensor)
                else value
                for key, value in output_tensors.items()
            }
            result = self._pp_prep_batch_result(
                batch, metadata, PPProxyTensors(output_tensors)
            )
            copy_done = self.device_module.Event()
            copy_done.record(self.copy_stream)
        copy_done.synchronize()
        with self._pp_vpp_select_chunk(
            {req.rid for req in batch.reqs}, include_retained=True
        ):
            self._pp_process_batch_result(batch, result)
        if getattr(batch, "contains_last_prefill_chunk", False):
            for req in batch.reqs:
                if req is not batch.chunked_req:
                    self._pp_vpp_prefix_registry.release(
                        req.rid, int(getattr(req, "session_generation", None) or 0)
                    )
        self._vpp_forward_batches.pop(batch_seq)
        if self.disaggregation_mode != DisaggregationMode.PREFILL and all(
            req.finished() for req in batch.reqs
        ):
            # A follower only plans when another manifest arrives. Clear its
            # completed view now so cache flush need not wait for another batch.
            batch.filter_batch()
        self.last_mbs[slot_id] = batch
        self.mbs[slot_id] = None
        self.mb_metadata[slot_id] = None
        if release_slot:
            slot_batch_seqs[slot_id] = None
        return True
