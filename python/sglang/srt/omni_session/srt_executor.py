# SPDX-License-Identifier: Apache-2.0
"""Synchronous executor from omni session requests into an SRT scheduler.

This module is the execution boundary where omni_session borrows SRT internals:
it can enqueue small session requests and build temporary query batches that
read committed SRT KV cache without permanently appending generation query tokens.
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable

import torch

from sglang.omni.core.protocol import TemporaryForwardPrepared
from sglang.srt.omni_session.runtime_types import OmniSRTKVTokenBinding

if TYPE_CHECKING:
    from sglang.omni.runtime.srt_scheduler_state import OmniSchedulerExclusiveLease
    from sglang.srt.layers.attention.base_attn_backend import AttentionBackend
    from sglang.srt.managers.io_struct import OpenSessionReqInput, OpenSessionReqOutput
    from sglang.srt.managers.schedule_batch import Req, ScheduleBatch
    from sglang.srt.managers.scheduler import Scheduler
    from sglang.srt.mem_cache.allocator import BaseTokenToKVPoolAllocator
    from sglang.srt.mem_cache.base_prefix_cache import BasePrefixCache
    from sglang.srt.mem_cache.memory_pool import ReqToTokenPool
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch
    from sglang.srt.model_executor.model_runner import ModelRunner
    from sglang.srt.omni_session.runtime import OmniSessionRecord
    from sglang.srt.session.session_controller import SessionController
    from sglang.srt.session.streaming_session import SessionSlot


class OmniSRTSchedulerExecutorError(RuntimeError):
    """Raised when omni cannot synchronously execute an SRT scheduler request."""


@dataclass(slots=True)
class OmniSRTTemporaryReqSlot:
    """A conceptual temporary Req to help with memory allocation / release"""

    req_pool_idx: int | None = None
    kv_committed_len: int = 0
    is_chunked: int = 0
    mamba_pool_idx: torch.Tensor | None = None
    mamba_ping_pong_track_buffer: torch.Tensor | None = None
    mamba_next_track_idx: int | None = None


@dataclass
class OmniSRTTemporaryForwardBatch:
    """ForwardBatch wrapper that releases temporary context-query scheduler state."""

    forward_batch: "ForwardBatch"
    req_to_token_pool: "ReqToTokenPool"
    token_to_kv_pool_allocator: "BaseTokenToKVPoolAllocator"
    temp_req: OmniSRTTemporaryReqSlot
    out_cache_loc: torch.Tensor
    owned_cache_loc: torch.Tensor | None = None
    synchronize_before_release: bool = False
    scheduler_exclusive_lease: "OmniSchedulerExclusiveLease | None" = None
    released: bool = False

    def release(self) -> None:
        """release scratch KV allocated for one denoise context forward"""
        if self.released:
            return
        self.released = True
        try:
            if self.synchronize_before_release and self.out_cache_loc.is_cuda:
                # 1. wait for kernels that may still read temporary KV slots
                torch.cuda.current_stream(self.out_cache_loc.device).synchronize()
            if self.owned_cache_loc is not None:
                # 2. release temporary query KV after the denoise forward
                self.token_to_kv_pool_allocator.free(self.owned_cache_loc)
        finally:
            free_mamba_cache = getattr(self.req_to_token_pool, "free_mamba_cache", None)
            try:
                if (
                    callable(free_mamba_cache)
                    and self.temp_req.mamba_pool_idx is not None
                ):
                    free_mamba_cache(self.temp_req)
            finally:
                try:
                    self.req_to_token_pool.free(self.temp_req)
                finally:
                    if self.scheduler_exclusive_lease is not None:
                        self.scheduler_exclusive_lease.release()


class OmniSRTSchedulerExecutor:
    """Owned by OmniSessionRuntime, executes materialized omni session requests through the SRT scheduler.

    this executor enqueues the reqs created with runtime, runs them
    through scheduler-native batching when called from an async omni task,
    keeps the synchronous path for tests/control calls, and exposes committed
    KV bindings for generation.

    """

    finish_request_after_execute = False

    def __init__(
        self,
        scheduler: "Scheduler",
        *,
        run_synchronously: bool = True,
        max_sync_steps: int = 8,
        require_idle_scheduler: bool = True,
    ) -> None:
        if scheduler.session_controller is None:
            raise ValueError(
                "OmniSRTSchedulerExecutor requires scheduler.session_controller"
            )
        self.scheduler: Scheduler = scheduler
        # synchronous mode keeps omni AR/session state updated before mm-gen runs.
        self.run_synchronously = run_synchronously
        self.max_sync_steps = max_sync_steps
        self.require_idle_scheduler = require_idle_scheduler
        self.token_bindings: list[OmniSRTKVTokenBinding] = []
        self._request_token_bindings: dict[str, OmniSRTKVTokenBinding] = {}
        self.sync_step_count = 0
        self.temporary_context_forward_count = 0
        self.temporary_context_allocated_token_count = 0

    @property
    def session_controller(self) -> "SessionController":
        return self.scheduler.session_controller

    def open_session_on_scheduler_thread(
        self, recv_req: "OpenSessionReqInput"
    ) -> "OpenSessionReqOutput":
        """open an SRT session on the scheduler thread when omni is async"""

        def open_session() -> "OpenSessionReqOutput":
            return self.session_controller.open(recv_req)

        return self._run_scheduler_thread_call(
            callback=open_session,
            description=f"open omni SRT session {recv_req.session_id}",
        )

    def close_session_on_scheduler_thread(self, session_id: str) -> None:
        """close an SRT session on the scheduler thread when omni is async"""

        from sglang.srt.managers.io_struct import CloseSessionReqInput

        def close_session() -> None:
            if session_id in self.session_controller:
                self.session_controller.close(
                    CloseSessionReqInput(session_id=session_id)
                )

        self._run_scheduler_thread_call(
            callback=close_session,
            description=f"close omni SRT session {session_id}",
        )

    def execute_omni_request(
        self, *, record: "OmniSessionRecord", req: "Req", state: object | None
    ) -> None:
        if self._submit_native_omni_request(record=record, req=req, state=state):
            return
        self._check_scheduler_idle(req)
        self.scheduler.init_req_max_new_tokens(req)
        self.scheduler._add_request_to_queue(req)
        if self.run_synchronously:
            self._run_until_request_complete(req)

    def run_idle_cleanup(self) -> None:
        """drain finished scheduler state left by synchronous omni requests"""
        scheduler_state = self.scheduler.omni_scheduler_state
        if (
            scheduler_state is not None
            and scheduler_state.scheduler_thread_id is not None
            and not scheduler_state.is_scheduler_thread()
        ):
            return
        self._run_idle_cleanup()

    def get_request_token_binding(
        self,
        *,
        record: "OmniSessionRecord",
        req: "Req",
        state: object | None,
    ) -> OmniSRTKVTokenBinding | None:
        binding = self._request_token_bindings.get(req.rid)
        if binding is not None:
            return binding
        token_indices = self._request_token_indices(record, req)
        if token_indices is None:
            return None
        return OmniSRTKVTokenBinding(
            session_id=record.session_id,
            request_id=req.rid,
            token_count=int(token_indices.numel()),
            token_indices=token_indices,
            position_count=self._request_position_count(req),
        )

    def get_srt_model(self) -> Any:
        """Return the SRT model owned by the attached scheduler's ModelRunner."""
        model_runner = self._require_model_runner()
        srt_model = model_runner.model
        if srt_model is None:
            raise OmniSRTSchedulerExecutorError(
                "omni SRT executor requires model_runner.model"
            )
        return srt_model

    def get_latest_session_position_count(
        self,
        session_id: str,
        *,
        condition_path_role: str | None = None,
    ) -> int | None:
        binding = self.get_latest_session_token_binding(
            self._binding_session_id(session_id, condition_path_role)
        )
        if binding is None:
            return None
        if binding.position_count is not None:
            return int(binding.position_count)
        return int(binding.token_count)

    def get_latest_session_token_binding(
        self,
        session_id: str,
    ) -> OmniSRTKVTokenBinding | None:
        for binding in reversed(self.token_bindings):
            if binding.session_id == session_id:
                return binding
        return None

    def pad_input_ids(self, input_ids: list[int], mm_inputs: Any) -> list[int]:
        model_runner = self._require_model_runner()
        srt_model = model_runner.model
        pad_input_ids = getattr(srt_model, "pad_input_ids", None)
        if not callable(pad_input_ids):
            return list(input_ids)
        return pad_input_ids(list(input_ids), mm_inputs)

    def build_temporary_context_forward_batch(
        self,
        *,
        prepared: TemporaryForwardPrepared,
        scheduler_exclusive_lease: "OmniSchedulerExclusiveLease | None" = None,
    ) -> OmniSRTTemporaryForwardBatch:
        """build a temporary extend batch that reads a committed SRT context

        Colocated omni models such as U1 use this adapter when diffusion denoise
        needs SRT model attention over the live session KV without committing
        the denoise query tokens into that session.
        """

        model_runner: ModelRunner = self._require_model_runner()
        req_to_token_pool = model_runner.req_to_token_pool
        if req_to_token_pool is None:
            raise OmniSRTSchedulerExecutorError(
                "Temporary context forward requires model_runner.req_to_token_pool"
            )
        token_to_kv_pool_allocator = model_runner.token_to_kv_pool_allocator
        if token_to_kv_pool_allocator is None:
            raise OmniSRTSchedulerExecutorError(
                "Temporary context forward requires model_runner.token_to_kv_pool_allocator"
            )
        attn_backend = model_runner.attn_backend

        # 2. borrow committed context KV as prefix for a denoise query
        binding_session_id = self._binding_session_id(
            prepared.srt_session_id,
            prepared.condition_path_role,
        )
        binding = self.get_latest_session_token_binding(binding_session_id)
        if binding is None:
            raise OmniSRTSchedulerExecutorError(
                "Temporary context forward has no SRT KV token binding for session "
                f"{binding_session_id}"
            )
        prefix_indices = binding.token_indices
        if prefix_indices is None:
            raise OmniSRTSchedulerExecutorError(
                "Temporary context forward requires SRT token indices on the context binding"
            )
        prefix_len = int(binding.token_count or 0)
        if prefix_len < 0:
            raise OmniSRTSchedulerExecutorError(
                "Temporary context forward binding token_count cannot be negative"
            )
        if int(prefix_indices.numel()) < prefix_len:
            raise OmniSRTSchedulerExecutorError(
                "Temporary context forward binding token_count exceeds token_indices length"
            )

        # 1. materialize diffusion query shape before borrowing ModelRunner state
        generation_input = prepared.generation_input
        packed_seqlens = generation_input.get("packed_seqlens")
        packed_position_ids = generation_input.get("packed_position_ids")
        cross_attention_custom_mask = generation_input.get(
            "cross_attention_custom_mask"
        )
        attention_math_mode = generation_input.get("attention_math_mode")
        synchronize_before_release = bool(
            generation_input.get("synchronize_before_release", False)
        )
        if packed_seqlens is None or packed_position_ids is None:
            raise OmniSRTSchedulerExecutorError(
                "Temporary context forward requires packed_seqlens and packed_position_ids"
            )
        if int(packed_seqlens.numel()) != 1:
            raise OmniSRTSchedulerExecutorError(
                "Temporary context forward currently supports one packed query sequence"
            )
        position_token_count = (
            int(packed_position_ids.shape[-1])
            if int(packed_position_ids.ndim) > 1
            else int(packed_position_ids.numel())
        )
        extend_num_tokens = int(
            generation_input.get("extend_num_tokens", position_token_count)
        )
        if extend_num_tokens <= 0:
            raise OmniSRTSchedulerExecutorError(
                "Temporary context forward requires at least one query token"
            )
        if position_token_count != extend_num_tokens:
            raise OmniSRTSchedulerExecutorError(
                "Temporary context forward packed_position_ids must match packed_seqlens"
            )

        device = torch.device(model_runner.device)
        prefix_indices = prefix_indices[:prefix_len].to(
            device=device,
            dtype=torch.int64,
            non_blocking=True,
        )
        seq_len = prefix_len + extend_num_tokens
        self._check_context_capacity(req_to_token_pool, seq_len)
        if scheduler_exclusive_lease is None:
            scheduler_exclusive_lease = self._enter_temporary_context_region()
        try:
            # 2. ensure scheduler state will not mutate while denoise reads SRT KV
            self._check_scheduler_idle_for_temporary_context()
        except Exception:
            if scheduler_exclusive_lease is not None:
                scheduler_exclusive_lease.release()
            raise

        temp_req = None
        out_cache_loc = None
        owned_cache_loc = None
        context = None
        try:
            # 3. allocate scratch query KV without committing it to the session
            temp_req = self._alloc_temp_req_slot(req_to_token_pool)
            out_cache_loc, owned_cache_loc = self._alloc_temporary_context_cache(
                token_to_kv_pool_allocator=token_to_kv_pool_allocator,
                extend_num_tokens=extend_num_tokens,
                device=device,
            )
            self._write_temp_req_token_mapping(
                req_to_token_pool=req_to_token_pool,
                req_pool_idx=int(temp_req.req_pool_idx),
                prefix_indices=prefix_indices,
                out_cache_loc=out_cache_loc,
                prefix_len=prefix_len,
                seq_len=seq_len,
            )
            forward_batch = self._make_temporary_context_forward_batch(
                model_runner=model_runner,
                req_to_token_pool=req_to_token_pool,
                token_to_kv_pool_allocator=token_to_kv_pool_allocator,
                attn_backend=attn_backend,
                req_pool_idx=int(temp_req.req_pool_idx),
                out_cache_loc=out_cache_loc,
                packed_position_ids=packed_position_ids,
                prefix_len=prefix_len,
                seq_len=seq_len,
                extend_num_tokens=extend_num_tokens,
                binding=binding,
                cross_attention_custom_mask=cross_attention_custom_mask,
                attention_math_mode=attention_math_mode,
                device=device,
            )
            context = OmniSRTTemporaryForwardBatch(
                forward_batch=forward_batch,
                req_to_token_pool=req_to_token_pool,
                token_to_kv_pool_allocator=token_to_kv_pool_allocator,
                temp_req=temp_req,
                out_cache_loc=out_cache_loc,
                owned_cache_loc=owned_cache_loc,
                synchronize_before_release=synchronize_before_release,
                scheduler_exclusive_lease=scheduler_exclusive_lease,
            )
            scheduler_exclusive_lease = None
            # 4. initialize SRT attention metadata over prefix KV + scratch query KV
            prepare_forward_batch = getattr(
                model_runner.model, "prepare_forward_batch", None
            )
            if callable(prepare_forward_batch):
                prepare_forward_batch(forward_batch)
            attn_backend.init_forward_metadata(forward_batch)
            self.temporary_context_forward_count += 1
            self.temporary_context_allocated_token_count += extend_num_tokens
            return context
        except Exception:
            if context is not None:
                context.release()
            else:
                if owned_cache_loc is not None:
                    token_to_kv_pool_allocator.free(owned_cache_loc)
                if temp_req is not None:
                    self._free_temp_req_slot(req_to_token_pool, temp_req)
                if scheduler_exclusive_lease is not None:
                    scheduler_exclusive_lease.release()
            raise

    def run_temporary_context_forward(
        self,
        *,
        prepared: TemporaryForwardPrepared,
        forward: Callable[[Any], Any],
    ) -> Any:
        """run a temporary context forward inside the SRT scheduler boundary"""
        scheduler_exclusive_lease = self._enter_temporary_context_region()

        def run_forward() -> Any:
            nonlocal scheduler_exclusive_lease
            temporary_batch = self.build_temporary_context_forward_batch(
                prepared=prepared,
                scheduler_exclusive_lease=scheduler_exclusive_lease,
            )
            scheduler_exclusive_lease = None
            try:
                # 1. run denoise attention where srt model runner owns streams/KV
                return forward(temporary_batch.forward_batch)
            finally:
                # 2. release temporary query KV before scheduler accepts more AR work
                temporary_batch.release()

        try:
            return self._run_scheduler_thread_call(
                callback=run_forward,
                description=(
                    "run temporary omni context forward " f"{prepared.srt_session_id}"
                ),
            )
        finally:
            if scheduler_exclusive_lease is not None:
                scheduler_exclusive_lease.release()

    def capture_batch_token_bindings_before_process(
        self, batch: "ScheduleBatch"
    ) -> list[tuple[str, str, int | None]]:
        """capture active request KV before SRT output processing may release it"""
        self._capture_batch_token_bindings(batch)
        return self._batch_sessions(batch)

    def capture_session_token_bindings_after_process(
        self, sessions: list[tuple[str, str, int | None]]
    ) -> None:
        """capture committed session KV after SRT has updated the streaming slot"""
        self._capture_session_token_bindings(sessions)

    def _submit_native_omni_request(
        self,
        *,
        record: "OmniSessionRecord",
        req: "Req",
        state: object | None,
    ) -> bool:
        scheduler_state = self.scheduler.omni_scheduler_state
        if (
            scheduler_state is None
            or scheduler_state.scheduler_thread_id is None
            or scheduler_state.is_scheduler_thread()
        ):
            return False

        # 1. hand the native req back to scheduler so AR can batch normally
        pending = scheduler_state.submit_srt_request(
            executor=self,
            record=record,
            req=req,
            state=state,
        )
        # 2. wait until scheduler finalizes this segment's session/KV state
        pending.wait()
        return True

    def _enter_temporary_context_region(self) -> "OmniSchedulerExclusiveLease | None":
        scheduler_state = self.scheduler.omni_scheduler_state
        if (
            scheduler_state is None
            or scheduler_state.scheduler_thread_id is None
            or scheduler_state.is_scheduler_thread()
        ):
            return None

        # temporary context forward borrows SRT KV/attention and must not race scheduler
        return scheduler_state.enter_scheduler_exclusive_region(
            scheduler=self.scheduler,
            reason="temporary context forward",
        )

    def _run_scheduler_thread_call(
        self,
        *,
        callback: Callable[[], Any],
        description: str,
    ) -> Any:
        scheduler_state = self.scheduler.omni_scheduler_state
        if (
            scheduler_state is None
            or scheduler_state.scheduler_thread_id is None
            or scheduler_state.is_scheduler_thread()
        ):
            return callback()

        return scheduler_state.run_on_scheduler_thread(
            callback=callback,
            description=description,
        )

    def _run_until_request_complete(self, req: "Req") -> None:
        previous_last_batch = self.scheduler.last_batch
        try:
            for _ in range(self._sync_step_budget(req)):
                if req.finished():
                    self._run_idle_cleanup()
                    return
                batch = self._run_scheduler_step()
                if batch is None and not req.finished():
                    raise OmniSRTSchedulerExecutorError(
                        f"SRT scheduler produced no batch for omni request {req.rid}"
                    )

            if not req.finished():
                raise OmniSRTSchedulerExecutorError(
                    "SRT scheduler did not finish omni request "
                    f"{req.rid} within {self._sync_step_budget(req)} steps"
                )
            self._run_idle_cleanup()
        finally:
            self.scheduler.last_batch = previous_last_batch

    def _run_scheduler_step(self) -> "ScheduleBatch | None":
        batch = self.scheduler.get_next_batch_to_run()
        previous_cur_batch = self.scheduler.cur_batch
        self.scheduler.cur_batch = batch
        try:
            if batch:
                batch_sessions = self._batch_sessions(batch)
                result = self.scheduler.run_batch(batch)
                self.scheduler.launch_batch_sample_if_needed(result)
                self._capture_batch_token_bindings(batch)
                # 1. submit context KV while processing a req
                self.scheduler.process_batch_result(batch, result)
                self._capture_session_token_bindings(batch_sessions)
                self.sync_step_count += 1
            else:
                self.scheduler.on_idle()
            self.scheduler.last_batch = batch
        finally:
            self.scheduler.cur_batch = previous_cur_batch
        return batch

    def _run_idle_cleanup(self) -> None:
        self._drain_overlap_result_queue()
        for _ in range(self.max_sync_steps):
            if self._scheduler_fully_idle():
                return
            if self._scheduler_has_waiting_requests():
                # 1. drain already-admitted SRT work before borrowing KV slots
                if self._run_scheduler_step() is None:
                    return
                continue
            if self._scheduler_has_pending_requests():
                return
            if not self._scheduler_has_active_batches():
                return
            if self._scheduler_batches_all_finished():
                self._clear_finished_scheduler_batches()
                continue
            batch = self._run_scheduler_step()
            if batch is None:
                continue

    def _sync_step_budget(self, req: "Req") -> int:
        max_new_tokens = int(req.sampling_params.max_new_tokens or 0)
        return max(self.max_sync_steps, max_new_tokens + 4)

    def _drain_overlap_result_queue(self) -> None:
        drained = False
        while len(self.scheduler.result_queue) > 0:
            batch, result = self.scheduler.result_queue.popleft()
            self.scheduler.process_batch_result(batch, result)
            drained = True
        if drained:
            self.scheduler.last_batch = None

    def _scheduler_fully_idle(self) -> bool:
        return self.scheduler.is_fully_idle()

    def _scheduler_has_pending_requests(self) -> bool:
        if self._scheduler_has_waiting_requests():
            return True
        return len(self.scheduler.grammar_manager.grammar_queue) > 0

    def _scheduler_has_waiting_requests(self) -> bool:
        return len(self.scheduler.waiting_queue) > 0

    def _scheduler_has_active_batches(self) -> bool:
        for batch in (
            self.scheduler.last_batch,
            self.scheduler.running_batch,
            self.scheduler.cur_batch,
        ):
            if batch is None:
                continue
            if batch.is_empty():
                continue
            return True
        return False

    def _scheduler_batches_all_finished(self) -> bool:
        saw_req = False
        for batch in (
            self.scheduler.last_batch,
            self.scheduler.running_batch,
            self.scheduler.cur_batch,
        ):
            if batch is None:
                continue
            if batch.is_empty():
                continue
            for req in batch.reqs:
                saw_req = True
                if not req.finished():
                    return False
        return saw_req

    def _clear_finished_scheduler_batches(self) -> None:
        if self._batch_all_finished(self.scheduler.last_batch):
            self.scheduler.last_batch = None
        if self._batch_all_finished(self.scheduler.cur_batch):
            self.scheduler.cur_batch = None
        running_batch = self.scheduler.running_batch
        if self._batch_all_finished(running_batch):
            running_batch.filter_batch()
            running_batch.reqs = []
            running_batch.batch_is_full = False

    @staticmethod
    def _batch_all_finished(batch: "ScheduleBatch | None") -> bool:
        if batch is None:
            return False
        if not batch.reqs:
            return False
        for req in batch.reqs:
            if not req.finished():
                return False
        return True

    def _check_scheduler_idle(self, req: "Req") -> None:
        if not self.require_idle_scheduler:
            return
        if self.scheduler.is_fully_idle():
            return
        if self.run_synchronously:
            self._run_idle_cleanup()
        if self.scheduler.is_fully_idle():
            return
        if self._scheduler_batches_all_finished():
            self._clear_finished_scheduler_batches()
            if self.scheduler.is_fully_idle():
                return
        if (
            not self._scheduler_has_pending_requests()
            and not self._scheduler_has_active_batches()
        ):
            return
        raise OmniSRTSchedulerExecutorError(
            "omni synchronous scheduler execution requires an idle scheduler before "
            f"enqueuing request {req.rid}; {self._scheduler_idle_debug()}"
        )

    def _check_scheduler_idle_for_temporary_context(self) -> None:
        if not self.require_idle_scheduler:
            return
        if self.scheduler.is_fully_idle():
            return
        if self.run_synchronously:
            self._run_idle_cleanup()
        if self.scheduler.is_fully_idle():
            return
        if self._scheduler_batches_all_finished():
            self._clear_finished_scheduler_batches()
            if self.scheduler.is_fully_idle():
                return
        if (
            not self._scheduler_has_pending_requests()
            and not self._scheduler_has_active_batches()
        ):
            return
        raise OmniSRTSchedulerExecutorError(
            "Temporary context forward requires an idle scheduler before borrowing "
            f"ModelRunner KV slots; {self._scheduler_idle_debug()}"
        )

    def _scheduler_idle_debug(self) -> str:
        parts = [
            f"waiting={len(self.scheduler.waiting_queue)}",
            f"grammar={len(self.scheduler.grammar_manager.grammar_queue)}",
        ]
        for attr_name, batch in (
            ("last_batch", self.scheduler.last_batch),
            ("running_batch", self.scheduler.running_batch),
            ("cur_batch", self.scheduler.cur_batch),
        ):
            if batch is None:
                parts.append(f"{attr_name}=None")
                continue
            req_states = []
            for req in batch.reqs[:4]:
                req_states.append(f"{req.rid}:{req.finished()}")
            parts.append(f"{attr_name}(empty={batch.is_empty()}, reqs={req_states})")
        return ", ".join(parts)

    def _require_model_runner(self) -> "ModelRunner":
        model_runner = self._model_runner()
        if model_runner is None:
            raise OmniSRTSchedulerExecutorError(
                "omni native SRT execution requires a scheduler ModelRunner"
            )
        return model_runner

    @staticmethod
    def _check_context_capacity(
        req_to_token_pool: "ReqToTokenPool", seq_len: int
    ) -> None:
        if seq_len > int(req_to_token_pool.max_context_len):
            raise OmniSRTSchedulerExecutorError(
                "Temporary context forward exceeds req_to_token_pool.max_context_len: "
                f"{seq_len} > {req_to_token_pool.max_context_len}"
            )

    def _model_runner(self) -> "ModelRunner | None":
        model_runner = self.scheduler.model_worker.model_runner
        if model_runner is not None:
            return model_runner
        return self.scheduler.tp_worker.model_runner

    @staticmethod
    def _binding_session_id(
        session_id: str, condition_path_role: str | None = None
    ) -> str:
        if condition_path_role is None:
            return session_id
        return f"{session_id}:{condition_path_role}"

    @staticmethod
    def _alloc_temp_req_slot(
        req_to_token_pool: "ReqToTokenPool",
    ) -> OmniSRTTemporaryReqSlot:
        temp_req = OmniSRTTemporaryReqSlot()
        req_pool_indices = req_to_token_pool.alloc([temp_req])
        if req_pool_indices is None:
            raise OmniSRTSchedulerExecutorError(
                "Temporary context forward could not allocate a req_to_token slot"
            )
        if temp_req.req_pool_idx is None:
            temp_req.req_pool_idx = int(req_pool_indices[0])
        return temp_req

    @staticmethod
    def _free_temp_req_slot(
        req_to_token_pool: "ReqToTokenPool",
        temp_req: OmniSRTTemporaryReqSlot,
    ) -> None:
        free_mamba_cache = getattr(req_to_token_pool, "free_mamba_cache", None)
        try:
            if callable(free_mamba_cache) and temp_req.mamba_pool_idx is not None:
                free_mamba_cache(temp_req)
        finally:
            if temp_req.req_pool_idx is not None:
                req_to_token_pool.free(temp_req)

    def _alloc_temporary_context_cache(
        self,
        *,
        token_to_kv_pool_allocator: "BaseTokenToKVPoolAllocator",
        extend_num_tokens: int,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """allocate scratch KV for a temporary denoise query

        temporary context forward reads committed session KV through the same
        req_to_token row, so this path must not evict tree cache on its own.
        if scratch KV is unavailable, admission/backpressure should handle it
        above this model-runner boundary.
        """
        tree_cache = self.scheduler.tree_cache
        page_size = int(tree_cache.page_size)

        # 1. reserve only scratch KV; committed prefix KV remains session-owned
        owned_num_tokens = self._ceil_to_page(extend_num_tokens, page_size)
        owned_cache_loc = token_to_kv_pool_allocator.alloc(owned_num_tokens)
        if owned_cache_loc is None:
            available_size = int(token_to_kv_pool_allocator.available_size())
            raise OmniSRTSchedulerExecutorError(
                "Temporary context forward could not allocate scratch KV slots "
                "without evicting committed SRT context: "
                f"need={owned_num_tokens}, available={available_size}"
            )
        owned_cache_loc = owned_cache_loc.to(
            device=device,
            dtype=torch.int64,
            non_blocking=True,
        )
        out_cache_loc = owned_cache_loc[:extend_num_tokens]
        return out_cache_loc, owned_cache_loc

    @staticmethod
    def _ceil_to_page(num_tokens: int, page_size: int) -> int:
        if page_size <= 1:
            return num_tokens
        return ((num_tokens + page_size - 1) // page_size) * page_size

    @staticmethod
    def _write_temp_req_token_mapping(
        *,
        req_to_token_pool: "ReqToTokenPool",
        req_pool_idx: int,
        prefix_indices: torch.Tensor,
        out_cache_loc: torch.Tensor,
        prefix_len: int,
        seq_len: int,
    ) -> None:
        req_to_token = req_to_token_pool.req_to_token
        pool_device = req_to_token.device
        pool_dtype = req_to_token.dtype
        # 2. expose committed context KV through req_to_token
        if prefix_len > 0:
            req_to_token_pool.write(
                (req_pool_idx, slice(0, prefix_len)),
                prefix_indices.to(
                    device=pool_device,
                    dtype=pool_dtype,
                    non_blocking=True,
                ),
            )
        # 3. expose temporary query KV through req_to_token
        req_to_token_pool.write(
            (req_pool_idx, slice(prefix_len, seq_len)),
            out_cache_loc.to(device=pool_device, dtype=pool_dtype, non_blocking=True),
        )

    @staticmethod
    def _make_temporary_context_forward_batch(
        *,
        model_runner: "ModelRunner",
        req_to_token_pool: "ReqToTokenPool",
        token_to_kv_pool_allocator: "BaseTokenToKVPoolAllocator",
        attn_backend: "AttentionBackend",
        req_pool_idx: int,
        out_cache_loc: torch.Tensor,
        packed_position_ids: torch.Tensor,
        prefix_len: int,
        seq_len: int,
        extend_num_tokens: int,
        binding: OmniSRTKVTokenBinding,
        cross_attention_custom_mask: torch.Tensor | None,
        attention_math_mode: str | None,
        device: torch.device,
    ) -> "ForwardBatch":
        """adapt a diffusion denoise query into an SRT ForwardBatch"""
        from sglang.srt.model_executor.forward_batch_info import (
            CaptureHiddenMode,
            ForwardBatch,
            ForwardMode,
        )

        input_ids = torch.zeros(extend_num_tokens, dtype=torch.int64, device=device)
        req_pool_indices = torch.tensor(
            [req_pool_idx], dtype=torch.int64, device=device
        )
        seq_lens = torch.tensor([seq_len], dtype=torch.int64, device=device)
        seq_lens_cpu = torch.tensor([seq_len], dtype=torch.int64)
        extend_seq_lens = torch.tensor(
            [extend_num_tokens],
            dtype=torch.int32,
            device=device,
        )
        extend_prefix_lens = torch.tensor(
            [prefix_len], dtype=torch.int32, device=device
        )
        positions = packed_position_ids.to(
            device=device,
            dtype=torch.int64,
            non_blocking=True,
        )
        token_to_kv_pool = model_runner.token_to_kv_pool
        if token_to_kv_pool is None:
            token_to_kv_pool = token_to_kv_pool_allocator.get_kvcache()
        # 3. reuse SRT cache pool, attention backend, and position/mask semantics
        forward_batch = ForwardBatch(
            forward_mode=ForwardMode.EXTEND,
            batch_size=1,
            input_ids=input_ids,
            req_pool_indices=req_pool_indices,
            seq_lens=seq_lens,
            out_cache_loc=out_cache_loc,
            seq_lens_sum=seq_len,
            orig_seq_lens=torch.tensor([seq_len], dtype=torch.int32, device=device),
            seq_lens_cpu=seq_lens_cpu,
            positions=positions,
            extend_num_tokens=extend_num_tokens,
            extend_seq_lens=extend_seq_lens,
            extend_prefix_lens=extend_prefix_lens,
            extend_start_loc=torch.tensor([0], dtype=torch.int32, device=device),
            extend_prefix_lens_cpu=[prefix_len],
            extend_seq_lens_cpu=[extend_num_tokens],
            extend_logprob_start_lens_cpu=[0],
            req_to_token_pool=req_to_token_pool,
            token_to_kv_pool=token_to_kv_pool,
            attn_backend=attn_backend,
            spec_algorithm=model_runner.spec_algorithm,
            spec_info=None,
            capture_hidden_mode=CaptureHiddenMode.NULL,
            num_token_non_padded=torch.tensor(
                extend_num_tokens,
                dtype=torch.int32,
                device=device,
            ),
            num_token_non_padded_cpu=extend_num_tokens,
            is_extend_in_batch=True,
            all_extend_in_batch=True,
            global_forward_mode=ForwardMode.EXTEND,
            is_prefill_only=True,
            rids=[f"{binding.session_id}:temporary_context"],
            # U1 pixel-flow query tokens are bidirectional while prefix KV stays read-only
            temporary_context_attention_mode="full_query",
            attention_math_mode=attention_math_mode,
        )
        forward_batch.temporary_context_cu_seqlens_q = torch.tensor(
            [0, extend_num_tokens],
            dtype=torch.int32,
            device=device,
        )
        forward_batch.temporary_context_cu_seqlens_k = torch.tensor(
            [0, seq_len],
            dtype=torch.int32,
            device=device,
        )
        forward_batch.temporary_context_forward_metadata = {
            "session_id": binding.session_id,
            "request_id": binding.request_id,
            "req_pool_idx": req_pool_idx,
            "prefix_len": prefix_len,
            "extend_num_tokens": extend_num_tokens,
            "attention_mode": "full_query",
            "attention_mask_shape": (extend_num_tokens, seq_len),
        }
        if (
            cross_attention_custom_mask is not None
            and int(cross_attention_custom_mask.numel()) != extend_num_tokens * seq_len
        ):
            raise OmniSRTSchedulerExecutorError(
                "Temporary context forward custom mask has inconsistent size: "
                f"{int(cross_attention_custom_mask.numel())} != "
                f"{extend_num_tokens * seq_len}"
            )
        if cross_attention_custom_mask is not None:
            forward_batch.cross_attention_custom_mask = cross_attention_custom_mask.to(
                device=device,
                dtype=torch.bool,
                non_blocking=True,
            )
        return forward_batch

    def _request_token_indices(
        self, record: "OmniSessionRecord", req: "Req"
    ) -> torch.Tensor | None:
        tree_cache = self.scheduler.tree_cache
        req_to_token = tree_cache.req_to_token_pool.req_to_token

        pool_idx = req.req_pool_idx
        token_count = int(req.kv_committed_len or 0)
        if pool_idx is None or token_count <= 0:
            slot = self._streaming_session_slot(tree_cache, record.session_id)
            if slot is None:
                return None
            pool_idx = slot.req_pool_idx
            token_count = int(slot.kv_committed_len or 0)
        if pool_idx is None or token_count <= 0:
            return None

        token_indices = req_to_token[pool_idx, :token_count].to(dtype=torch.int64)
        return token_indices.clone()

    def _capture_batch_token_bindings(self, batch: "ScheduleBatch") -> None:
        for req in batch.reqs:
            session = req.session
            if session is None:
                continue
            session_id = session.session_id
            token_indices = self._request_token_indices_for_active_req(req)
            if token_indices is None:
                continue
            binding = OmniSRTKVTokenBinding(
                session_id=session_id,
                request_id=req.rid,
                token_count=int(token_indices.numel()),
                token_indices=token_indices,
                position_count=self._request_position_count(req),
            )
            self._request_token_bindings[req.rid] = binding
            self.token_bindings.append(binding)

    def _batch_sessions(
        self, batch: "ScheduleBatch"
    ) -> list[tuple[str, str, int | None]]:
        sessions: list[tuple[str, str, int | None]] = []
        for req in batch.reqs:
            session = req.session
            if session is None:
                continue
            session_id = session.session_id
            sessions.append(
                (
                    str(session_id),
                    str(req.rid),
                    self._request_position_count(req),
                )
            )
        return sessions

    def _capture_session_token_bindings(
        self, sessions: list[tuple[str, str, int | None]]
    ) -> None:
        for session_id, request_id, position_count in sessions:
            token_indices = self._request_token_indices_for_session(session_id)
            if token_indices is None:
                continue
            binding = OmniSRTKVTokenBinding(
                session_id=session_id,
                request_id=request_id,
                token_count=int(token_indices.numel()),
                token_indices=token_indices,
                position_count=position_count,
            )
            if request_id:
                self._request_token_bindings[request_id] = binding
            self.token_bindings.append(binding)

    def _request_token_indices_for_session(
        self, session_id: str
    ) -> torch.Tensor | None:
        tree_cache = self.scheduler.tree_cache
        req_to_token = tree_cache.req_to_token_pool.req_to_token
        slot = self._streaming_session_slot(tree_cache, session_id)
        if slot is None:
            return None
        pool_idx = slot.req_pool_idx
        token_count = int(slot.kv_committed_len or 0)
        if pool_idx is None or token_count <= 0:
            return None
        return req_to_token[pool_idx, :token_count].to(dtype=torch.int64).clone()

    def _request_token_indices_for_active_req(self, req: "Req") -> torch.Tensor | None:
        tree_cache = self.scheduler.tree_cache
        req_to_token = tree_cache.req_to_token_pool.req_to_token
        pool_idx = req.req_pool_idx
        token_count = int(req.kv_committed_len or 0)
        if (pool_idx is None or token_count <= 0) and req.session is not None:
            slot = self._streaming_session_slot(tree_cache, req.session.session_id)
            if slot is not None:
                pool_idx = slot.req_pool_idx
                token_count = int(slot.kv_committed_len or 0)
        if pool_idx is None or token_count <= 0:
            return None
        return req_to_token[pool_idx, :token_count].to(dtype=torch.int64).clone()

    @staticmethod
    def _request_position_count(req: "Req") -> int | None:
        if req.omni_srt_position_count is not None:
            return int(req.omni_srt_position_count)

        position_count = OmniSRTSchedulerExecutor._position_count_from_position_ids(
            req.custom_position_ids
        )
        if position_count is not None:
            return position_count

        if req.custom_decode_position_id is not None:
            return int(req.custom_decode_position_id) + 1
        return None

    @staticmethod
    def _position_count_from_position_ids(
        positions: torch.Tensor | list[int] | list[list[int]] | None,
    ) -> int | None:
        if positions is None:
            return None
        if isinstance(positions, torch.Tensor):
            positions = positions.detach().cpu().tolist()
        if not positions:
            return None
        first = positions[0]
        if isinstance(first, (list, tuple)):
            return max(int(position[0]) for position in positions) + 1
        return max(int(position) for position in positions) + 1

    @staticmethod
    def _streaming_session_slot(
        tree_cache: "BasePrefixCache", session_id: str
    ) -> "SessionSlot | None":
        return tree_cache.get_session_slot(session_id)
