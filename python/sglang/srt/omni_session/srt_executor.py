# SPDX-License-Identifier: Apache-2.0
"""Synchronous executor from omni session requests into an SRT scheduler.

This module is the execution boundary where omni_session borrows SRT internals:
it can enqueue small session requests and build temporary query batches that
read committed SRT KV cache without permanently appending generation query tokens.
"""

from dataclasses import dataclass
from typing import Any

import torch

from sglang.omni.protocol import TemporaryForwardPrepared
from sglang.srt.omni_session.runtime_protocol import OmniSRTKVTokenBinding


class OmniSRTSchedulerExecutorError(RuntimeError):
    """Raised when omni cannot synchronously execute an SRT scheduler request."""


@dataclass(slots=True)
class OmniSRTTemporaryReqSlot:
    req_pool_idx: int | None = None
    kv_committed_len: int = 0
    is_chunked: int = 0
    mamba_pool_idx: Any | None = None
    mamba_ping_pong_track_buffer: Any | None = None
    mamba_next_track_idx: Any | None = None


@dataclass
class OmniSRTTemporaryForwardBatch:
    """ForwardBatch wrapper that releases temporary context-query scheduler state."""

    forward_batch: Any
    req_to_token_pool: Any
    token_to_kv_pool_allocator: Any
    temp_req: OmniSRTTemporaryReqSlot
    out_cache_loc: torch.Tensor
    owned_cache_loc: torch.Tensor | None = None
    released: bool = False

    def release(self) -> None:
        if self.released:
            return
        self.released = True
        try:
            if self.owned_cache_loc is not None:
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
                self.req_to_token_pool.free(self.temp_req)


@dataclass(frozen=True, slots=True)
class OmniSRTBoundTemporaryForwardPrepared:
    generation_input: dict[str, Any]
    srt_session_id: str
    condition_path_role: str | None
    srt_kv_token_binding: OmniSRTKVTokenBinding


class OmniSRTSchedulerExecutor:
    """Execute materialized omni session requests through the SRT scheduler.

    this executor enqueues the reqs created with runtime, runs them
    synchronously when needed, and exposes committed KV bindings for generation.

    """

    finish_request_after_execute = False

    def __init__(
        self,
        scheduler: Any,
        *,
        run_synchronously: bool = True,
        max_sync_steps: int = 8,
        require_idle_scheduler: bool = True,
    ) -> None:
        if not hasattr(scheduler, "session_controller"):
            raise ValueError(
                "OmniSRTSchedulerExecutor requires scheduler.session_controller"
            )
        self.scheduler = scheduler
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
    def session_controller(self):
        return self.scheduler.session_controller

    def execute_omni_request(self, *, record, req, state) -> None:
        self._check_scheduler_idle(req)
        if hasattr(self.scheduler, "init_req_max_new_tokens"):
            self.scheduler.init_req_max_new_tokens(req)
        if not hasattr(self.scheduler, "_add_request_to_queue"):
            raise ValueError(
                "OmniSRTSchedulerExecutor requires scheduler._add_request_to_queue"
            )
        self.scheduler._add_request_to_queue(req)
        if self.run_synchronously:
            self._run_until_request_complete(req)

    def run_idle_cleanup(self) -> None:
        """drain finished scheduler state left by synchronous omni requests"""
        self._run_idle_cleanup()

    def get_request_token_binding(
        self,
        *,
        record,
        req,
        state,
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
        srt_model = getattr(model_runner, "model", None)
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
        position_count = getattr(binding, "position_count", None)
        if position_count is not None:
            return int(position_count)
        return int(getattr(binding, "token_count"))

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
        srt_model = getattr(model_runner, "model", None)
        pad_input_ids = getattr(srt_model, "pad_input_ids", None)
        if not callable(pad_input_ids):
            return list(input_ids)
        return pad_input_ids(list(input_ids), mm_inputs)

    def build_temporary_context_forward_batch_for_session(
        self,
        *,
        prepared: TemporaryForwardPrepared,
        generation_query_embeds: torch.Tensor,
        timestep: torch.Tensor,
    ) -> OmniSRTTemporaryForwardBatch:
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
        prepared_with_binding = OmniSRTBoundTemporaryForwardPrepared(
            generation_input=prepared.generation_input,
            srt_session_id=prepared.srt_session_id,
            condition_path_role=prepared.condition_path_role,
            srt_kv_token_binding=binding,
        )
        return self.build_temporary_context_forward_batch(
            prepared=prepared_with_binding,
            generation_query_embeds=generation_query_embeds,
            timestep=timestep,
        )

    def build_temporary_context_forward_batch(
        self,
        *,
        prepared: OmniSRTBoundTemporaryForwardPrepared,
        generation_query_embeds: torch.Tensor,
        timestep: torch.Tensor,
    ) -> OmniSRTTemporaryForwardBatch:
        """Build a temporary extend batch that reads a committed SRT context.

        The batch reuses the session's committed KV indices as prefix context, but
        allocates fresh request/query KV slots for this query and releases them as
        soon as the caller returns.
        """

        self._check_scheduler_idle_for_temporary_context()
        model_runner = self._require_model_runner()
        req_to_token_pool = self._require_attr(
            model_runner,
            "req_to_token_pool",
            "Temporary context forward requires model_runner.req_to_token_pool",
        )
        token_to_kv_pool_allocator = self._require_attr(
            model_runner,
            "token_to_kv_pool_allocator",
            "Temporary context forward requires model_runner.token_to_kv_pool_allocator",
        )
        attn_backend = self._require_attr(
            model_runner,
            "attn_backend",
            "Temporary context forward requires model_runner.attn_backend",
        )

        binding = prepared.srt_kv_token_binding
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

        generation_input = prepared.generation_input
        packed_seqlens = generation_input.get("packed_seqlens")
        packed_position_ids = generation_input.get("packed_position_ids")
        if packed_seqlens is None or packed_position_ids is None:
            raise OmniSRTSchedulerExecutorError(
                "Temporary context forward requires packed_seqlens and packed_position_ids"
            )
        if int(packed_seqlens.numel()) != 1:
            raise OmniSRTSchedulerExecutorError(
                "Temporary context forward currently supports one packed query sequence"
            )
        extend_num_tokens = int(packed_seqlens.to("cpu").sum().item())
        if extend_num_tokens <= 0:
            raise OmniSRTSchedulerExecutorError(
                "Temporary context forward requires at least one query token"
            )
        position_token_count = (
            int(packed_position_ids.shape[-1])
            if int(packed_position_ids.ndim) > 1
            else int(packed_position_ids.numel())
        )
        if position_token_count != extend_num_tokens:
            raise OmniSRTSchedulerExecutorError(
                "Temporary context forward packed_position_ids must match packed_seqlens"
            )

        device = torch.device(getattr(model_runner, "device", req_to_token_pool.device))
        prefix_indices = prefix_indices[:prefix_len].to(
            device=device,
            dtype=torch.int64,
            non_blocking=True,
        )
        seq_len = prefix_len + extend_num_tokens
        self._check_context_capacity(req_to_token_pool, seq_len)

        temp_req = self._alloc_temp_req_slot(req_to_token_pool)
        out_cache_loc = None
        owned_cache_loc = None
        context = None
        try:
            out_cache_loc, owned_cache_loc = self._alloc_temporary_context_cache(
                model_runner=model_runner,
                token_to_kv_pool_allocator=token_to_kv_pool_allocator,
                prefix_indices=prefix_indices,
                prefix_len=prefix_len,
                seq_len=seq_len,
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
                device=device,
            )
            context = OmniSRTTemporaryForwardBatch(
                forward_batch=forward_batch,
                req_to_token_pool=req_to_token_pool,
                token_to_kv_pool_allocator=token_to_kv_pool_allocator,
                temp_req=temp_req,
                out_cache_loc=out_cache_loc,
                owned_cache_loc=owned_cache_loc,
            )
            prepare_forward_batch = getattr(
                getattr(model_runner, "model", None),
                "prepare_forward_batch",
                None,
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
                self._free_temp_req_slot(req_to_token_pool, temp_req)
            raise

    def _run_until_request_complete(self, req: Any) -> None:
        self._require_scheduler_methods(
            "get_next_batch_to_run",
            "run_batch",
            "process_batch_result",
        )

        has_last_batch = hasattr(self.scheduler, "last_batch")
        previous_last_batch = getattr(self.scheduler, "last_batch", None)
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
            if has_last_batch:
                self.scheduler.last_batch = previous_last_batch

    def _run_scheduler_step(self):
        batch = self.scheduler.get_next_batch_to_run()
        has_cur_batch = hasattr(self.scheduler, "cur_batch")
        previous_cur_batch = getattr(self.scheduler, "cur_batch", None)
        if has_cur_batch:
            self.scheduler.cur_batch = batch
        try:
            if batch:
                batch_sessions = self._batch_sessions(batch)
                result = self.scheduler.run_batch(batch)
                launch_sample = getattr(
                    self.scheduler, "launch_batch_sample_if_needed", None
                )
                if callable(launch_sample):
                    launch_sample(result)
                self._capture_batch_token_bindings(batch)
                self.scheduler.process_batch_result(batch, result)
                self._capture_session_token_bindings(batch_sessions)
                self.sync_step_count += 1
            elif hasattr(self.scheduler, "on_idle"):
                self.scheduler.on_idle()
            if hasattr(self.scheduler, "last_batch"):
                self.scheduler.last_batch = batch
        finally:
            if has_cur_batch:
                self.scheduler.cur_batch = previous_cur_batch
        return batch

    def _run_idle_cleanup(self) -> None:
        self._drain_overlap_result_queue()
        for _ in range(self.max_sync_steps):
            if self._scheduler_fully_idle():
                return
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

    def _sync_step_budget(self, req: Any) -> int:
        sampling_params = getattr(req, "sampling_params", None)
        max_new_tokens = int(getattr(sampling_params, "max_new_tokens", 0) or 0)
        return max(self.max_sync_steps, max_new_tokens + 4)

    def _drain_overlap_result_queue(self) -> None:
        result_queue = getattr(self.scheduler, "result_queue", None)
        process_batch_result = getattr(self.scheduler, "process_batch_result", None)
        if result_queue is None or not callable(process_batch_result):
            return
        drained = False
        while self._safe_len(result_queue) > 0:
            batch, result = result_queue.popleft()
            process_batch_result(batch, result)
            drained = True
        if drained and hasattr(self.scheduler, "last_batch"):
            self.scheduler.last_batch = None

    def _scheduler_fully_idle(self) -> bool:
        is_fully_idle = getattr(self.scheduler, "is_fully_idle", None)
        return bool(callable(is_fully_idle) and is_fully_idle())

    def _scheduler_has_pending_requests(self) -> bool:
        waiting_queue = getattr(self.scheduler, "waiting_queue", None)
        if self._safe_len(waiting_queue) > 0:
            return True
        grammar_manager = getattr(self.scheduler, "grammar_manager", None)
        grammar_queue = getattr(grammar_manager, "grammar_queue", None)
        return self._safe_len(grammar_queue) > 0

    def _scheduler_has_active_batches(self) -> bool:
        for attr_name in ("last_batch", "running_batch", "cur_batch"):
            batch = getattr(self.scheduler, attr_name, None)
            if batch is None:
                continue
            if hasattr(batch, "is_empty") and batch.is_empty():
                continue
            reqs = getattr(batch, "reqs", None) or []
            if not reqs:
                continue
            return True
        return False

    def _scheduler_batches_all_finished(self) -> bool:
        saw_req = False
        for attr_name in ("last_batch", "running_batch", "cur_batch"):
            batch = getattr(self.scheduler, attr_name, None)
            if batch is None:
                continue
            if hasattr(batch, "is_empty") and batch.is_empty():
                continue
            for req in getattr(batch, "reqs", None) or []:
                saw_req = True
                finished = getattr(req, "finished", None)
                if not callable(finished) or not finished():
                    return False
        return saw_req

    def _clear_finished_scheduler_batches(self) -> None:
        for attr_name in ("last_batch", "cur_batch"):
            batch = getattr(self.scheduler, attr_name, None)
            if self._batch_all_finished(batch):
                setattr(self.scheduler, attr_name, None)
        running_batch = getattr(self.scheduler, "running_batch", None)
        if self._batch_all_finished(running_batch):
            filter_batch = getattr(running_batch, "filter_batch", None)
            if callable(filter_batch):
                filter_batch()
            if hasattr(running_batch, "reqs"):
                running_batch.reqs = []
            if hasattr(running_batch, "batch_is_full"):
                running_batch.batch_is_full = False

    @staticmethod
    def _batch_all_finished(batch: Any) -> bool:
        if batch is None:
            return False
        reqs = getattr(batch, "reqs", None) or []
        if not reqs:
            return False
        for req in reqs:
            finished = getattr(req, "finished", None)
            if not callable(finished) or not finished():
                return False
        return True

    def _require_scheduler_methods(self, *method_names: str) -> None:
        missing = [
            method_name
            for method_name in method_names
            if not hasattr(self.scheduler, method_name)
        ]
        if missing:
            raise OmniSRTSchedulerExecutorError(
                "OmniSRTSchedulerExecutor synchronous mode requires scheduler methods: "
                f"{missing}"
            )

    def _check_scheduler_idle(self, req: Any) -> None:
        if not self.require_idle_scheduler:
            return
        if self.run_synchronously:
            self._run_idle_cleanup()
        if not hasattr(self.scheduler, "is_fully_idle"):
            return
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
        if self.run_synchronously:
            self._run_idle_cleanup()
        if not hasattr(self.scheduler, "is_fully_idle"):
            return
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
            f"waiting={self._safe_len(getattr(self.scheduler, 'waiting_queue', None))}",
            "grammar="
            f"{self._safe_len(getattr(getattr(self.scheduler, 'grammar_manager', None), 'grammar_queue', None))}",
        ]
        for attr_name in ("last_batch", "running_batch", "cur_batch"):
            batch = getattr(self.scheduler, attr_name, None)
            if batch is None:
                parts.append(f"{attr_name}=None")
                continue
            is_empty = getattr(batch, "is_empty", None)
            reqs = getattr(batch, "reqs", None) or []
            req_states = []
            for req in reqs[:4]:
                finished = getattr(req, "finished", None)
                req_states.append(
                    f"{getattr(req, 'rid', '<unknown>')}:{callable(finished) and finished()}"
                )
            parts.append(
                f"{attr_name}(empty={callable(is_empty) and is_empty()}, reqs={req_states})"
            )
        return ", ".join(parts)

    @staticmethod
    def _safe_len(value: Any) -> int:
        try:
            return len(value) if value is not None else 0
        except TypeError:
            return 0

    def _require_model_runner(self) -> Any:
        model_runner = self._model_runner()
        if model_runner is None:
            raise OmniSRTSchedulerExecutorError(
                "omni native SRT execution requires a scheduler ModelRunner"
            )
        return model_runner

    @staticmethod
    def _require_attr(obj: Any, attr: str, message: str) -> Any:
        value = getattr(obj, attr, None)
        if value is None:
            raise OmniSRTSchedulerExecutorError(message)
        return value

    @staticmethod
    def _check_context_capacity(req_to_token_pool: Any, seq_len: int) -> None:
        max_context_len = getattr(req_to_token_pool, "max_context_len", None)
        if max_context_len is None:
            return
        if seq_len > int(max_context_len):
            raise OmniSRTSchedulerExecutorError(
                "Temporary context forward exceeds req_to_token_pool.max_context_len: "
                f"{seq_len} > {max_context_len}"
            )

    def _model_runner(self) -> Any | None:
        model_worker = getattr(self.scheduler, "model_worker", None)
        model_runner = getattr(model_worker, "model_runner", None)
        if model_runner is not None:
            return model_runner
        tp_worker = getattr(self.scheduler, "tp_worker", None)
        return getattr(tp_worker, "model_runner", None)

    @staticmethod
    def _binding_session_id(session_id: str, condition_path_role: str | None = None) -> str:
        if condition_path_role is None:
            return session_id
        return f"{session_id}:{condition_path_role}"

    @staticmethod
    def _alloc_temp_req_slot(req_to_token_pool: Any) -> OmniSRTTemporaryReqSlot:
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
        req_to_token_pool: Any,
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
        model_runner: Any,
        token_to_kv_pool_allocator: Any,
        prefix_indices: torch.Tensor,
        prefix_len: int,
        seq_len: int,
        extend_num_tokens: int,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        tree_cache = getattr(self.scheduler, "tree_cache", None)
        page_size = int(
            getattr(
                tree_cache,
                "page_size",
                getattr(token_to_kv_pool_allocator, "page_size", 1),
            )
        )
        owned_num_tokens = self._ceil_to_page(extend_num_tokens, page_size)
        self._evict_for_temporary_context(
            tree_cache,
            extend_num_tokens=owned_num_tokens,
            page_size=page_size,
        )
        owned_cache_loc = token_to_kv_pool_allocator.alloc(owned_num_tokens)
        if owned_cache_loc is None:
            raise OmniSRTSchedulerExecutorError(
                "Temporary context forward could not allocate KV cache slots"
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
    def _evict_for_temporary_context(
        tree_cache: Any,
        *,
        extend_num_tokens: int,
        page_size: int,
    ) -> None:
        if tree_cache is None:
            return
        if not hasattr(tree_cache, "evict"):
            return
        from sglang.srt.mem_cache.common import evict_from_tree_cache

        evict_from_tree_cache(
            tree_cache,
            extend_num_tokens + max(1, page_size),
        )

    @staticmethod
    def _write_temp_req_token_mapping(
        *,
        req_to_token_pool: Any,
        req_pool_idx: int,
        prefix_indices: torch.Tensor,
        out_cache_loc: torch.Tensor,
        prefix_len: int,
        seq_len: int,
    ) -> None:
        req_to_token = req_to_token_pool.req_to_token
        pool_device = req_to_token.device
        pool_dtype = req_to_token.dtype
        if prefix_len > 0:
            req_to_token_pool.write(
                (req_pool_idx, slice(0, prefix_len)),
                prefix_indices.to(
                    device=pool_device,
                    dtype=pool_dtype,
                    non_blocking=True,
                ),
            )
        req_to_token_pool.write(
            (req_pool_idx, slice(prefix_len, seq_len)),
            out_cache_loc.to(device=pool_device, dtype=pool_dtype, non_blocking=True),
        )

    @staticmethod
    def _make_temporary_context_forward_batch(
        *,
        model_runner: Any,
        req_to_token_pool: Any,
        token_to_kv_pool_allocator: Any,
        attn_backend: Any,
        req_pool_idx: int,
        out_cache_loc: torch.Tensor,
        packed_position_ids: torch.Tensor,
        prefix_len: int,
        seq_len: int,
        extend_num_tokens: int,
        binding: OmniSRTKVTokenBinding,
        device: torch.device,
    ) -> Any:
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
        token_to_kv_pool = getattr(model_runner, "token_to_kv_pool", None)
        if token_to_kv_pool is None:
            get_kvcache = getattr(token_to_kv_pool_allocator, "get_kvcache", None)
            token_to_kv_pool = get_kvcache() if callable(get_kvcache) else None
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
            spec_algorithm=getattr(model_runner, "spec_algorithm", None),
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
        )
        forward_batch.temporary_context_forward_metadata = {
            "session_id": binding.session_id,
            "request_id": binding.request_id,
            "prefix_len": prefix_len,
            "extend_num_tokens": extend_num_tokens,
            "attention_mode": "full_query",
            "attention_mask_shape": (extend_num_tokens, seq_len),
        }
        forward_batch.cross_attention_custom_mask = torch.ones(
            extend_num_tokens * seq_len,
            dtype=torch.bool,
            device=device,
        )
        return forward_batch

    def _request_token_indices(self, record: Any, req: Any) -> torch.Tensor | None:
        tree_cache = getattr(self.scheduler, "tree_cache", None)
        if tree_cache is None:
            return None
        req_to_token_pool = getattr(tree_cache, "req_to_token_pool", None)
        req_to_token = getattr(req_to_token_pool, "req_to_token", None)
        if req_to_token is None:
            return None

        pool_idx = getattr(req, "req_pool_idx", None)
        token_count = int(getattr(req, "kv_committed_len", 0) or 0)
        if pool_idx is None or token_count <= 0:
            slot = self._streaming_session_slot(tree_cache, record.session_id)
            if slot is None:
                return None
            pool_idx = getattr(slot, "req_pool_idx", None)
            token_count = int(getattr(slot, "kv_committed_len", 0) or 0)
        if pool_idx is None or token_count <= 0:
            return None

        token_indices = req_to_token[pool_idx, :token_count].to(dtype=torch.int64)
        return token_indices.clone()

    def _capture_batch_token_bindings(self, batch: Any) -> None:
        for req in getattr(batch, "reqs", []) or []:
            session = getattr(req, "session", None)
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

    def _batch_sessions(self, batch: Any) -> list[tuple[str, str, int | None]]:
        sessions: list[tuple[str, str, int | None]] = []
        for req in getattr(batch, "reqs", []) or []:
            session = getattr(req, "session", None)
            if session is None:
                continue
            session_id = session.session_id
            sessions.append(
                (
                    str(session_id),
                    str(getattr(req, "rid", "")),
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
        tree_cache = getattr(self.scheduler, "tree_cache", None)
        if tree_cache is None:
            return None
        req_to_token_pool = getattr(tree_cache, "req_to_token_pool", None)
        req_to_token = getattr(req_to_token_pool, "req_to_token", None)
        slot = self._streaming_session_slot(tree_cache, session_id)
        if req_to_token is None or slot is None:
            return None
        pool_idx = getattr(slot, "req_pool_idx", None)
        token_count = int(getattr(slot, "kv_committed_len", 0) or 0)
        if pool_idx is None or token_count <= 0:
            return None
        return req_to_token[pool_idx, :token_count].to(dtype=torch.int64).clone()

    def _request_token_indices_for_active_req(self, req: Any) -> torch.Tensor | None:
        tree_cache = getattr(self.scheduler, "tree_cache", None)
        if tree_cache is None:
            return None
        req_to_token_pool = getattr(tree_cache, "req_to_token_pool", None)
        req_to_token = getattr(req_to_token_pool, "req_to_token", None)
        pool_idx = getattr(req, "req_pool_idx", None)
        token_count = int(getattr(req, "kv_committed_len", 0) or 0)
        if (pool_idx is None or token_count <= 0) and getattr(req, "session", None):
            session_id = getattr(req.session, "session_id", None)
            slot = self._streaming_session_slot(tree_cache, session_id)
            if slot is not None:
                pool_idx = getattr(slot, "req_pool_idx", None)
                token_count = int(getattr(slot, "kv_committed_len", 0) or 0)
        if req_to_token is None or pool_idx is None or token_count <= 0:
            return None
        return req_to_token[pool_idx, :token_count].to(dtype=torch.int64).clone()

    @staticmethod
    def _request_position_count(req: Any) -> int | None:
        position_count = OmniSRTSchedulerExecutor._position_count_from_position_ids(
            getattr(req, "custom_position_ids", None)
        )
        if position_count is not None:
            return position_count

        position_count = getattr(req, "omni_srt_position_count", None)
        if position_count is not None:
            return int(position_count)

        decode_position_id = getattr(req, "custom_decode_position_id", None)
        if decode_position_id is not None:
            return int(decode_position_id) + 1
        return None

    @staticmethod
    def _position_count_from_position_ids(positions: Any) -> int | None:
        if positions is None:
            return None
        if hasattr(positions, "detach"):
            positions = positions.detach().cpu().tolist()
        if not positions:
            return None
        first = positions[0]
        if isinstance(first, (list, tuple)):
            return max(int(position[0]) for position in positions) + 1
        return max(int(position) for position in positions) + 1

    @staticmethod
    def _streaming_session_slot(tree_cache: Any, session_id: str) -> Any | None:
        slots = getattr(tree_cache, "slots", None)
        if isinstance(slots, dict) and session_id in slots:
            return slots[session_id]
        streaming_session = getattr(tree_cache, "session", None)
        slots = getattr(streaming_session, "slots", None)
        if isinstance(slots, dict):
            return slots.get(session_id)
        return None
