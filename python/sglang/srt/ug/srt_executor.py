# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

import torch

from sglang.srt.ug.context import UGSRTKVTokenBinding


class UGSRTSchedulerExecutorError(RuntimeError):
    """Raised when UG cannot synchronously execute an SRT scheduler request."""


@dataclass
class UGSRTTemporaryForwardBatch:
    """ForwardBatch wrapper that releases temporary G-step scheduler state."""

    forward_batch: Any
    req_to_token_pool: Any
    token_to_kv_pool_allocator: Any
    temp_req: Any
    out_cache_loc: torch.Tensor
    released: bool = False

    def release(self) -> None:
        if self.released:
            return
        self.released = True
        try:
            if self.out_cache_loc is not None:
                self.token_to_kv_pool_allocator.free(self.out_cache_loc)
        finally:
            free_mamba_cache = getattr(self.req_to_token_pool, "free_mamba_cache", None)
            try:
                if (
                    callable(free_mamba_cache)
                    and getattr(self.temp_req, "mamba_pool_idx", None) is not None
                ):
                    free_mamba_cache(self.temp_req)
            finally:
                self.req_to_token_pool.free(self.temp_req)


class UGSRTRequestBoundaryExecutor:
    """Records materialized UG SRT requests at the execution boundary."""

    finish_request_after_execute = True

    def __init__(self) -> None:
        self.events: list[tuple[str, str, int]] = []

    def execute_ug_request(self, *, record, req, state) -> None:
        del record
        self.events.append((state.value, req.rid, len(req.origin_input_ids)))


class UGSRTSchedulerExecutor:
    """Minimal adapter from UG materialized requests into an SRT Scheduler."""

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
                "UGSRTSchedulerExecutor requires scheduler.session_controller"
            )
        self.scheduler = scheduler
        self.run_synchronously = run_synchronously
        self.max_sync_steps = max_sync_steps
        self.require_idle_scheduler = require_idle_scheduler
        self.events: list[tuple[str, str, int]] = []
        self.token_bindings: list[UGSRTKVTokenBinding] = []
        self._request_token_bindings: dict[str, UGSRTKVTokenBinding] = {}
        self.ug_u_forward_observer = None
        self.sync_step_count = 0
        self.temp_g_forward_count = 0
        self.temp_g_allocated_token_count = 0

    @property
    def session_controller(self):
        return self.scheduler.session_controller

    def set_ug_u_forward_observer(self, observer) -> None:
        self.ug_u_forward_observer = observer
        model_runner = self._model_runner()
        if model_runner is None:
            return
        setter = getattr(model_runner, "set_ug_u_forward_observer", None)
        if callable(setter):
            setter(observer)
        else:
            model_runner.ug_u_forward_observer = observer

    def execute_ug_request(self, *, record, req, state) -> None:
        del record
        if self.ug_u_forward_observer is not None:
            self.set_ug_u_forward_observer(self.ug_u_forward_observer)
        self._check_scheduler_idle(req)
        self.events.append((state.value, req.rid, len(req.origin_input_ids)))
        if hasattr(self.scheduler, "init_req_max_new_tokens"):
            self.scheduler.init_req_max_new_tokens(req)
        if not hasattr(self.scheduler, "_add_request_to_queue"):
            raise ValueError(
                "UGSRTSchedulerExecutor requires scheduler._add_request_to_queue"
            )
        self.scheduler._add_request_to_queue(req)
        if self.run_synchronously:
            self._run_until_request_complete(req)

    def get_ug_request_token_binding(
        self,
        *,
        record,
        req,
        state,
    ) -> UGSRTKVTokenBinding | None:
        del state
        binding = self._request_token_bindings.get(req.rid)
        if binding is not None:
            return binding
        token_indices = self._request_token_indices(record, req)
        if token_indices is None:
            return None
        return UGSRTKVTokenBinding(
            session_id=record.session_id,
            request_id=req.rid,
            token_count=int(token_indices.numel()),
            token_indices=token_indices,
        )

    def create_bagel_native_srt_denoise_executor(self):
        """Create a BAGEL denoise executor backed by this scheduler's ModelRunner."""

        from sglang.srt.ug.bagel import BAGELNativeSRTDenoiseExecutor

        model_runner = self._require_model_runner()
        srt_model = getattr(model_runner, "model", None)
        if srt_model is None:
            raise UGSRTSchedulerExecutorError(
                "UG BAGEL native denoise requires model_runner.model"
            )
        return BAGELNativeSRTDenoiseExecutor(
            srt_model,
            forward_batch_provider=self.build_ug_g_forward_batch,
        )

    def build_ug_g_forward_batch(
        self,
        *,
        prepared: Any,
        latent_tokens: torch.Tensor,
        timestep: torch.Tensor,
    ) -> UGSRTTemporaryForwardBatch:
        """Build a temporary extend batch for one BAGEL G denoise timestep.

        The batch reuses the U session's committed KV indices as prefix context, but
        allocates fresh request/query KV slots for this timestep and releases them as
        soon as the native BAGEL velocity forward returns.
        """

        del latent_tokens, timestep
        self._check_scheduler_idle_for_temp_g()
        model_runner = self._require_model_runner()
        req_to_token_pool = self._require_attr(
            model_runner,
            "req_to_token_pool",
            "UG G forward requires model_runner.req_to_token_pool",
        )
        token_to_kv_pool_allocator = self._require_attr(
            model_runner,
            "token_to_kv_pool_allocator",
            "UG G forward requires model_runner.token_to_kv_pool_allocator",
        )
        attn_backend = self._require_attr(
            model_runner,
            "attn_backend",
            "UG G forward requires model_runner.attn_backend",
        )

        binding = getattr(prepared, "srt_kv_token_binding", None)
        if binding is None:
            raise UGSRTSchedulerExecutorError(
                "UG G forward requires prepared.srt_kv_token_binding"
            )
        prefix_indices = getattr(binding, "token_indices", None)
        if prefix_indices is None:
            raise UGSRTSchedulerExecutorError(
                "UG G forward requires SRT token indices on the U context binding"
            )
        prefix_len = int(getattr(binding, "token_count", 0) or 0)
        if prefix_len <= 0:
            raise UGSRTSchedulerExecutorError(
                "UG G forward requires a non-empty U context binding"
            )
        if int(prefix_indices.numel()) < prefix_len:
            raise UGSRTSchedulerExecutorError(
                "UG G forward binding token_count exceeds token_indices length"
            )

        generation_input = getattr(prepared, "generation_input", None) or {}
        packed_seqlens = generation_input.get("packed_seqlens")
        packed_position_ids = generation_input.get("packed_position_ids")
        if packed_seqlens is None or packed_position_ids is None:
            raise UGSRTSchedulerExecutorError(
                "UG G forward requires packed_seqlens and packed_position_ids"
            )
        if int(packed_seqlens.numel()) != 1:
            raise UGSRTSchedulerExecutorError(
                "UG G temporary batch currently supports one packed G sequence"
            )
        extend_num_tokens = int(packed_seqlens.to("cpu").sum().item())
        if extend_num_tokens <= 0:
            raise UGSRTSchedulerExecutorError(
                "UG G forward requires at least one timestep query token"
            )
        if int(packed_position_ids.numel()) != extend_num_tokens:
            raise UGSRTSchedulerExecutorError(
                "UG G packed_position_ids must match packed_seqlens"
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
        context = None
        try:
            out_cache_loc = self._alloc_temp_g_cache(
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
            forward_batch = self._make_temp_g_forward_batch(
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
            context = UGSRTTemporaryForwardBatch(
                forward_batch=forward_batch,
                req_to_token_pool=req_to_token_pool,
                token_to_kv_pool_allocator=token_to_kv_pool_allocator,
                temp_req=temp_req,
                out_cache_loc=out_cache_loc,
            )
            prepare_forward_batch = getattr(
                getattr(model_runner, "model", None),
                "prepare_forward_batch",
                None,
            )
            if callable(prepare_forward_batch):
                prepare_forward_batch(forward_batch)
            attn_backend.init_forward_metadata(forward_batch)
            self.temp_g_forward_count += 1
            self.temp_g_allocated_token_count += extend_num_tokens
            return context
        except Exception:
            if context is not None:
                context.release()
            else:
                if out_cache_loc is not None:
                    token_to_kv_pool_allocator.free(out_cache_loc)
                self._free_temp_req_slot(req_to_token_pool, temp_req)
            raise

    def _run_until_request_complete(self, req: Any) -> None:
        self._require_scheduler_methods(
            "get_next_batch_to_run",
            "run_batch",
            "process_batch_result",
        )

        for _ in range(self.max_sync_steps):
            if req.finished():
                self._run_idle_cleanup()
                return
            batch = self._run_scheduler_step()
            if batch is None and not req.finished():
                raise UGSRTSchedulerExecutorError(
                    f"SRT scheduler produced no batch for UG request {req.rid}"
                )

        if not req.finished():
            raise UGSRTSchedulerExecutorError(
                "SRT scheduler did not finish UG request "
                f"{req.rid} within {self.max_sync_steps} steps"
            )
        self._run_idle_cleanup()

    def _run_scheduler_step(self):
        batch = self.scheduler.get_next_batch_to_run()
        if hasattr(self.scheduler, "cur_batch"):
            self.scheduler.cur_batch = batch
        if batch:
            result = self.scheduler.run_batch(batch)
            self._capture_batch_token_bindings(batch)
            self.scheduler.process_batch_result(batch, result)
            self.sync_step_count += 1
        elif hasattr(self.scheduler, "on_idle"):
            self.scheduler.on_idle()
        if hasattr(self.scheduler, "last_batch"):
            self.scheduler.last_batch = batch
        return batch

    def _run_idle_cleanup(self) -> None:
        if not hasattr(self.scheduler, "last_batch"):
            return
        if self.scheduler.last_batch is None:
            return
        self._run_scheduler_step()

    def _require_scheduler_methods(self, *method_names: str) -> None:
        missing = [
            method_name
            for method_name in method_names
            if not hasattr(self.scheduler, method_name)
        ]
        if missing:
            raise UGSRTSchedulerExecutorError(
                "UGSRTSchedulerExecutor synchronous mode requires scheduler methods: "
                f"{missing}"
            )

    def _check_scheduler_idle(self, req: Any) -> None:
        if not self.require_idle_scheduler:
            return
        if not hasattr(self.scheduler, "is_fully_idle"):
            return
        if self.scheduler.is_fully_idle():
            return
        raise UGSRTSchedulerExecutorError(
            "UG synchronous scheduler execution requires an idle scheduler before "
            f"enqueuing request {req.rid}"
        )

    def _check_scheduler_idle_for_temp_g(self) -> None:
        if not self.require_idle_scheduler:
            return
        if not hasattr(self.scheduler, "is_fully_idle"):
            return
        if self.scheduler.is_fully_idle():
            return
        raise UGSRTSchedulerExecutorError(
            "UG temporary G forward requires an idle scheduler before borrowing "
            "ModelRunner KV slots"
        )

    def _require_model_runner(self) -> Any:
        model_runner = self._model_runner()
        if model_runner is None:
            raise UGSRTSchedulerExecutorError(
                "UG native SRT execution requires a scheduler ModelRunner"
            )
        return model_runner

    @staticmethod
    def _require_attr(obj: Any, attr: str, message: str) -> Any:
        value = getattr(obj, attr, None)
        if value is None:
            raise UGSRTSchedulerExecutorError(message)
        return value

    @staticmethod
    def _check_context_capacity(req_to_token_pool: Any, seq_len: int) -> None:
        max_context_len = getattr(req_to_token_pool, "max_context_len", None)
        if max_context_len is None:
            return
        if seq_len > int(max_context_len):
            raise UGSRTSchedulerExecutorError(
                "UG G temporary batch exceeds req_to_token_pool.max_context_len: "
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
    def _alloc_temp_req_slot(req_to_token_pool: Any) -> Any:
        temp_req = SimpleNamespace(
            req_pool_idx=None,
            kv_committed_len=0,
            is_chunked=0,
            mamba_pool_idx=None,
            mamba_ping_pong_track_buffer=None,
            mamba_next_track_idx=None,
        )
        req_pool_indices = req_to_token_pool.alloc([temp_req])
        if req_pool_indices is None:
            raise UGSRTSchedulerExecutorError(
                "UG G temporary batch could not allocate a req_to_token slot"
            )
        if temp_req.req_pool_idx is None:
            temp_req.req_pool_idx = int(req_pool_indices[0])
        return temp_req

    @staticmethod
    def _free_temp_req_slot(req_to_token_pool: Any, temp_req: Any) -> None:
        free_mamba_cache = getattr(req_to_token_pool, "free_mamba_cache", None)
        try:
            if (
                callable(free_mamba_cache)
                and getattr(temp_req, "mamba_pool_idx", None) is not None
            ):
                free_mamba_cache(temp_req)
        finally:
            if getattr(temp_req, "req_pool_idx", None) is not None:
                req_to_token_pool.free(temp_req)

    def _alloc_temp_g_cache(
        self,
        *,
        model_runner: Any,
        token_to_kv_pool_allocator: Any,
        prefix_indices: torch.Tensor,
        prefix_len: int,
        seq_len: int,
        extend_num_tokens: int,
        device: torch.device,
    ) -> torch.Tensor:
        tree_cache = getattr(self.scheduler, "tree_cache", None)
        page_size = int(
            getattr(
                tree_cache,
                "page_size",
                getattr(token_to_kv_pool_allocator, "page_size", 1),
            )
        )
        self._evict_for_temp_g(
            tree_cache,
            extend_num_tokens=extend_num_tokens,
            page_size=page_size,
        )
        if page_size > 1 and hasattr(token_to_kv_pool_allocator, "alloc_extend"):
            prefix_lens_cpu = torch.tensor([prefix_len], dtype=torch.int64)
            seq_lens_cpu = torch.tensor([seq_len], dtype=torch.int64)
            out_cache_loc = token_to_kv_pool_allocator.alloc_extend(
                prefix_lens=prefix_lens_cpu.to(device, non_blocking=True),
                prefix_lens_cpu=prefix_lens_cpu,
                seq_lens=torch.tensor([seq_len], dtype=torch.int64, device=device),
                seq_lens_cpu=seq_lens_cpu,
                last_loc=prefix_indices[-1:],
                extend_num_tokens=extend_num_tokens,
            )
        else:
            out_cache_loc = token_to_kv_pool_allocator.alloc(extend_num_tokens)
        if out_cache_loc is None:
            raise UGSRTSchedulerExecutorError(
                "UG G temporary batch could not allocate KV cache slots"
            )
        return out_cache_loc.to(device=device, dtype=torch.int64, non_blocking=True)

    @staticmethod
    def _evict_for_temp_g(
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
        req_to_token_pool.write(
            (req_pool_idx, slice(0, prefix_len)),
            prefix_indices.to(device=pool_device, dtype=pool_dtype, non_blocking=True),
        )
        req_to_token_pool.write(
            (req_pool_idx, slice(prefix_len, seq_len)),
            out_cache_loc.to(device=pool_device, dtype=pool_dtype, non_blocking=True),
        )

    @staticmethod
    def _make_temp_g_forward_batch(
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
        binding: UGSRTKVTokenBinding,
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
            rids=[f"{binding.session_id}:ug_g_temp"],
        )
        forward_batch.ug_g_forward_metadata = {
            "session_id": binding.session_id,
            "request_id": binding.request_id,
            "prefix_len": prefix_len,
            "extend_num_tokens": extend_num_tokens,
            "attention_mode": "non_causal_query",
            "attention_mask_shape": (extend_num_tokens, seq_len),
        }
        forward_batch.ug_g_non_causal_query_attention = True
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
            session_id = getattr(session, "session_id", None)
            if session_id is None:
                continue
            token_indices = self._request_token_indices_for_active_req(req)
            if token_indices is None:
                continue
            binding = UGSRTKVTokenBinding(
                session_id=session_id,
                request_id=req.rid,
                token_count=int(token_indices.numel()),
                token_indices=token_indices,
            )
            self._request_token_bindings[req.rid] = binding
            self.token_bindings.append(binding)

    def _request_token_indices_for_active_req(self, req: Any) -> torch.Tensor | None:
        tree_cache = getattr(self.scheduler, "tree_cache", None)
        if tree_cache is None:
            return None
        req_to_token_pool = getattr(tree_cache, "req_to_token_pool", None)
        req_to_token = getattr(req_to_token_pool, "req_to_token", None)
        pool_idx = getattr(req, "req_pool_idx", None)
        token_count = int(getattr(req, "kv_committed_len", 0) or 0)
        if req_to_token is None or pool_idx is None or token_count <= 0:
            return None
        return req_to_token[pool_idx, :token_count].to(dtype=torch.int64).clone()

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
