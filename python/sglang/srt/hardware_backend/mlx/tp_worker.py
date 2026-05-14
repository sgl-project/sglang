"""MLX-specific TpModelWorker subclass for Apple Silicon.

Routes forward passes through the MLX model runner, bypassing PyTorch
MPS.  A lightweight stub provides scheduler bookkeeping; the actual
KV data lives in MlxKVPool.

The worker also exposes an async (lazy-eval) surface used by the MLX
overlap scheduler: ``async_forward_batch_generation_mlx`` launches a
batch without blocking on the GPU, ``async_chained_decode_mlx`` builds
the next decode step on top of a still-lazy previous decode, and
``finalize_mlx_result`` blocks on the lazy outputs and produces a
normal ``GenerationBatchResult``.
"""

import logging
from typing import Optional, Union

import mlx.core as mx
import torch

from sglang.srt.hardware_backend.mlx.model_runner import (
    MlxPendingDecode,
    MlxPendingExtend,
    MlxPendingPrefill,
)
from sglang.srt.managers.schedule_batch import ModelWorkerBatch
from sglang.srt.managers.tp_worker import TpModelWorker
from sglang.srt.managers.utils import GenerationBatchResult
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, PPProxyTensors

logger = logging.getLogger(__name__)


class MlxTpModelWorker(TpModelWorker):
    """A tensor parallel model worker that routes inference through MLX.

    Inherits from TpModelWorker for scheduler integration, but replaces
    the standard ModelRunner with MlxModelRunnerStub (no PyTorch weights,
    zero-memory KV cache) and delegates all forward passes to a native
    MlxModelRunner.
    """

    def _init_model_runner(self):
        """Create MLX runner first (auto-sizes pool), then stub with matching size."""
        from sglang.srt.hardware_backend.mlx.model_runner import MlxModelRunner
        from sglang.srt.hardware_backend.mlx.model_runner_stub import (
            MlxModelRunnerStub,
        )

        logger.info("Initializing MlxModelRunner for end-to-end MLX inference")
        init_kwargs = dict(
            model_path=self.server_args.model_path,
            trust_remote_code=self.server_args.trust_remote_code,
            disable_radix_cache=self.server_args.disable_radix_cache,
            mem_fraction_static=self.server_args.mem_fraction_static,
        )
        if self.server_args.max_total_tokens is not None:
            init_kwargs["pool_size"] = self.server_args.max_total_tokens
        self._mlx_runner = MlxModelRunner(**init_kwargs)

        self._model_runner = MlxModelRunnerStub(
            model_config=self.model_config,
            mem_fraction_static=self.server_args.mem_fraction_static,
            gpu_id=self.gpu_id,
            tp_rank=self.tp_rank,
            tp_size=self.tp_size,
            moe_ep_rank=self.moe_ep_rank,
            moe_ep_size=self.ep_size,
            pp_rank=self.pp_rank,
            pp_size=self.pp_size,
            nccl_port=self.nccl_port,
            dp_rank=self.dp_rank,
            server_args=self.server_args,
            is_draft_worker=self.is_draft_worker,
            req_to_token_pool=self.req_to_token_pool,
            token_to_kv_pool_allocator=self.token_to_kv_pool_allocator,
            memory_pool_config=self.memory_pool_config,
            mlx_pool_size=self._mlx_runner.pool_size,
        )

        self._mlx_active_rids: set[str] = set()
        self._mlx_pool_initialized = False

    def get_pad_input_ids_func(self):
        """Override since the stub ModelRunner has no real model."""
        return None

    def _ensure_mlx_pool_initialized(self):
        """Lazily initialize the MlxKVPool after the stub's pools are ready."""
        if not self._mlx_pool_initialized:
            self._mlx_runner.init_kv_pool(self._model_runner.req_to_token_pool)
            self._mlx_pool_initialized = True

    def forward_batch_generation(
        self,
        model_worker_batch: ModelWorkerBatch,
        forward_batch: Optional[ForwardBatch] = None,
        pp_proxy_tensors: Optional[PPProxyTensors] = None,
        is_verify: bool = False,
        skip_attn_backend_init=False,
    ) -> GenerationBatchResult:
        """Override to route through MLX model runner."""
        if model_worker_batch is not None:
            self._ensure_mlx_pool_initialized()
            return self._forward_batch_generation_mlx(model_worker_batch)

        # Fallback to standard path for None batches
        return super().forward_batch_generation(
            model_worker_batch,
            forward_batch,
            pp_proxy_tensors,
            is_verify,
            skip_attn_backend_init,
        )

    def _cleanup_stale_rids(self, forward_mode, current_rids: set[str]) -> None:
        """Remove MLX state for decode-mode requests that dropped out of the batch."""
        if forward_mode.is_decode():
            stale_rids = self._mlx_active_rids - current_rids
            for rid in stale_rids:
                self._mlx_runner.remove_request(rid)
            self._mlx_active_rids = current_rids
        else:
            self._mlx_active_rids |= current_rids

    def _forward_batch_generation_mlx(
        self,
        model_worker_batch: ModelWorkerBatch,
    ) -> GenerationBatchResult:
        """Run forward pass through the MLX model runner (greedy only)."""
        from sglang.srt.layers.logits_processor import LogitsProcessorOutput

        forward_mode = model_worker_batch.forward_mode
        reqs = model_worker_batch.reqs

        if forward_mode.is_idle():
            return GenerationBatchResult(
                logits_output=LogitsProcessorOutput(next_token_logits=None),
                can_run_cuda_graph=False,
            )

        self._cleanup_stale_rids(forward_mode, {req.rid for req in reqs})

        next_token_ids_list: list[int] = []

        if forward_mode.is_extend():
            # Ensure pool is up-to-date before PoolBackedCache reads it
            # for prefix-cached prefills.  Only runs on extend batches.
            self._mlx_runner.flush_all_decode_kv()
            input_ids_cpu = model_worker_batch.input_ids.cpu().tolist()
            out_cache_loc_cpu = model_worker_batch.out_cache_loc.cpu().tolist()
            extend_seq_lens = model_worker_batch.extend_seq_lens

            offset = 0  # into input_ids_cpu
            slot_offset = 0  # into out_cache_loc_cpu
            prefill_rids: list[tuple[str, int]] = []
            extend_rids: list[tuple[str, int]] = []
            decode_rids: list[str] = []

            for i, req in enumerate(reqs):
                seq_len = extend_seq_lens[i]
                req_token_ids = input_ids_cpu[offset : offset + seq_len]
                req_new_slots = out_cache_loc_cpu[slot_offset : slot_offset + seq_len]
                offset += seq_len
                slot_offset += seq_len

                if self._mlx_runner.has_request(req.rid):
                    if seq_len > 1:
                        # Chunked prefill continuation
                        next_token = self._mlx_runner.extend(
                            req.rid, req_token_ids, req_new_slots
                        )
                        extend_rids.append((req.rid, next_token))
                    else:
                        # MIXED mode: single-token decode
                        decode_rids.append(req.rid)
                else:
                    # New prefill
                    prefix_slot_ids = req.prefix_indices.tolist()
                    full_token_ids = list(req.fill_ids)
                    next_token = self._mlx_runner.prefill(
                        req_id=req.rid,
                        new_token_ids=req_token_ids,
                        full_token_ids=full_token_ids,
                        prefix_slot_ids=prefix_slot_ids,
                        new_slot_ids=req_new_slots,
                        req_pool_idx=req.req_pool_idx,
                    )
                    prefill_rids.append((req.rid, next_token))

            # Batch decode all existing requests at once
            if decode_rids:
                decode_results = self._mlx_runner.decode_batch(decode_rids)
                decode_map = dict(zip(decode_rids, decode_results))
            else:
                decode_map = {}

            prefill_map = dict(prefill_rids)
            extend_map = dict(extend_rids)

            for req in reqs:
                if req.rid in decode_map:
                    next_token_ids_list.append(decode_map[req.rid])
                elif req.rid in extend_map:
                    next_token_ids_list.append(extend_map[req.rid])
                else:
                    next_token_ids_list.append(prefill_map[req.rid])

        elif forward_mode.is_decode():
            req_ids = [req.rid for req in reqs]
            next_token_ids_list = self._mlx_runner.decode_batch(req_ids)

        else:
            raise ValueError(
                f"MLX runner does not support forward mode: {forward_mode}"
            )

        next_token_ids = torch.tensor(
            next_token_ids_list, dtype=torch.long, device="cpu"
        )

        return GenerationBatchResult(
            logits_output=LogitsProcessorOutput(next_token_logits=None),
            next_token_ids=next_token_ids,
            can_run_cuda_graph=False,
        )

    def async_forward_batch_generation_mlx(
        self,
        model_worker_batch: ModelWorkerBatch,
    ) -> tuple[
        Union[mx.array, None],
        list[MlxPendingPrefill],
        list[MlxPendingExtend],
        Optional[MlxPendingDecode],
        str,
    ]:
        """Start an async (lazy) forward pass through the MLX model runner.

        Returns ``(lazy_result, prefills, extends, decode, mode)``:

        * ``lazy_result`` — an ``mx.array`` that, when evaluated, forces
          materialisation of the whole batch's outputs.  ``None`` for
          idle batches.
        * ``prefills`` — list of :class:`MlxPendingPrefill` for new
          requests in an extend batch.
        * ``extends`` — list of :class:`MlxPendingExtend` for chunked
          prefill continuations in an extend batch.
        * ``decode`` — :class:`MlxPendingDecode` for the decode
          sub-batch (covers full decode mode AND mixed decodes inside
          an extend batch).
        * ``mode`` — one of ``"idle"``, ``"decode"``, ``"extend"``.

        The caller must make sure the returned pendings are fed into a
        subsequent ``mx.async_eval`` or ``.item()`` / ``.tolist()`` call
        — :meth:`finalize_mlx_result` does that.
        """
        self._ensure_mlx_pool_initialized()

        forward_mode = model_worker_batch.forward_mode
        reqs = model_worker_batch.reqs

        if forward_mode.is_idle():
            return None, [], [], None, "idle"

        self._cleanup_stale_rids(forward_mode, {req.rid for req in reqs})

        if forward_mode.is_decode():
            req_ids = [req.rid for req in reqs]
            pending_decode = self._mlx_runner.decode_batch_start(req_ids)
            mx.async_eval(pending_decode.lazy_tokens)
            return pending_decode.lazy_tokens, [], [], pending_decode, "decode"

        if forward_mode.is_extend():
            # TODO (changminbark): Implement per-batch flushing using prefix_slot_ids
            # Ensure the pool is up-to-date before any PoolBackedCache
            # reads it for prefix-cached prefills. Mirror the sync path.
            self._mlx_runner.flush_all_decode_kv()
            return self._async_extend_batch(model_worker_batch)

        raise ValueError(
            f"MLX async runner does not support forward mode: {forward_mode}"
        )

    def _async_extend_batch(
        self,
        model_worker_batch: ModelWorkerBatch,
    ) -> tuple[
        Union[mx.array, None],
        list[MlxPendingPrefill],
        list[MlxPendingExtend],
        Optional[MlxPendingDecode],
        str,
    ]:
        """Launch each request in an EXTEND batch lazily and kick GPU work."""
        reqs = model_worker_batch.reqs
        input_ids_cpu = model_worker_batch.input_ids.cpu().tolist()
        out_cache_loc_cpu = model_worker_batch.out_cache_loc.cpu().tolist()
        extend_seq_lens = model_worker_batch.extend_seq_lens

        offset = 0
        slot_offset = 0
        pending_prefills: list[MlxPendingPrefill] = []
        pending_extends: list[MlxPendingExtend] = []
        mixed_decode_rids: list[str] = []

        for i, req in enumerate(reqs):
            seq_len = extend_seq_lens[i]
            req_token_ids = input_ids_cpu[offset : offset + seq_len]
            req_new_slots = out_cache_loc_cpu[slot_offset : slot_offset + seq_len]
            offset += seq_len
            slot_offset += seq_len

            if self._mlx_runner.has_request(req.rid):
                if seq_len > 1:
                    # Chunked prefill continuation
                    pending_extends.append(
                        self._mlx_runner.extend_start(
                            req_id=req.rid,
                            new_token_ids=req_token_ids,
                            new_slot_ids=req_new_slots,
                        )
                    )
                else:
                    # MIXED mode: single-token decode
                    mixed_decode_rids.append(req.rid)
            else:
                # New prefill
                prefix_slot_ids = req.prefix_indices.tolist()
                full_token_ids = list(req.fill_ids)
                pending_prefills.append(
                    self._mlx_runner.prefill_start(
                        req_id=req.rid,
                        new_token_ids=req_token_ids,
                        full_token_ids=full_token_ids,
                        prefix_slot_ids=prefix_slot_ids,
                        new_slot_ids=req_new_slots,
                        req_pool_idx=req.req_pool_idx,
                    )
                )

        pending_mixed_decode: Optional[MlxPendingDecode] = None
        if mixed_decode_rids:
            pending_mixed_decode = self._mlx_runner.decode_batch_start(
                mixed_decode_rids
            )

        # Stack lazy tokens so the caller has a single handle to evaluate
        # after CPU scheduling work.  We also hand every cache buffer
        # (and the decode cache arrays) to mx.async_eval so the GPU
        # kernel-launch stream sees everything the next step depends on
        # before we actually block on anything.
        prefill_ext_tokens: list[mx.array] = [p.lazy_token for p in pending_prefills]
        prefill_ext_tokens.extend(e.lazy_token for e in pending_extends)

        async_args: list[mx.array] = []
        if prefill_ext_tokens:
            lazy_stacked = mx.stack(prefill_ext_tokens, axis=0)
            async_args.append(lazy_stacked)
        else:
            lazy_stacked = None

        for p in pending_prefills:
            async_args.extend(self._cache_state(p.cache))
        for e in pending_extends:
            async_args.extend(self._cache_state(self._mlx_runner._req_caches[e.req_id]))
        if pending_mixed_decode is not None:
            async_args.append(pending_mixed_decode.lazy_tokens)
            for c_list in pending_mixed_decode.caches:
                async_args.extend(self._cache_state(c_list))

        if async_args:
            mx.async_eval(*async_args)

        return (
            lazy_stacked,
            pending_prefills,
            pending_extends,
            pending_mixed_decode,
            "extend",
        )

    @staticmethod
    def _cache_state(cache_list) -> list[mx.array]:
        """Flatten a per-layer cache list to its ``state`` arrays."""
        return [s for c in cache_list for s in c.state]

    def async_chained_decode_mlx(
        self,
        prev_pending: MlxPendingDecode,
    ) -> tuple[mx.array, list, list, MlxPendingDecode, str]:
        """Launch a decode step that chains off a still-lazy previous decode.

        This is the "no idle gap" pipelining primitive: build the next
        decode's compute graph using ``prev_pending.lazy_tokens`` (still
        unevaluated) as its input ids, hand the combined graph to
        ``mx.async_eval``, and return.  The GPU runs the new step
        immediately after ``prev_pending`` with no scheduling gap, while
        the caller is free to block on ``prev_pending`` and run CPU-side
        bookkeeping.

        Preconditions (caller must ensure):

        * ``prev_pending`` was produced by a previous decode start
          (either :meth:`async_forward_batch_generation_mlx` in decode
          mode or a previous :meth:`async_chained_decode_mlx`).
        * The batch composition for this step is identical to
          ``prev_pending`` — same requests, same order.  Composition
          changes (finished reqs, new prefills) must break the chain.
        * ``prev_pending`` should be finalised BEFORE the returned
          pending, so per-request token lists are appended in order.

        Returns a 5-tuple matching
        :meth:`async_forward_batch_generation_mlx` for the decode case:
        ``(lazy_tokens, [], [], pending_decode, "decode")``.  The empty
        prefill/extend lists are always absent for chained decodes.
        """
        pending = self._mlx_runner.decode_batch_start_chained(prev_pending)
        mx.async_eval(pending.lazy_tokens)
        return pending.lazy_tokens, [], [], pending, "decode"

    def finalize_mlx_result(
        self,
        prefills: list[MlxPendingPrefill],
        extends: list[MlxPendingExtend],
        decode: Optional[MlxPendingDecode],
        mode: str,
        reqs: list,
    ) -> GenerationBatchResult:
        """Materialise a lazy MLX result into a :class:`GenerationBatchResult`.

        The blocking wait happens inside ``decode_batch_finalize`` /
        ``prefill_finalize`` / ``extend_finalize`` via ``.tolist()`` /
        ``.item()`` on the specific lazy outputs.
        """
        from sglang.srt.layers.logits_processor import LogitsProcessorOutput

        if mode == "idle":
            return GenerationBatchResult(
                logits_output=LogitsProcessorOutput(next_token_logits=None),
                can_run_cuda_graph=False,
            )

        if mode == "decode":
            assert decode is not None
            next_tokens_list = self._mlx_runner.decode_batch_finalize(decode)

        elif mode == "extend":
            prefill_map: dict[str, int] = {}
            for pending_p in prefills:
                prefill_map[pending_p.req_id] = self._mlx_runner.prefill_finalize(
                    pending_p
                )

            extend_map: dict[str, int] = {}
            for pending_e in extends:
                extend_map[pending_e.req_id] = self._mlx_runner.extend_finalize(
                    pending_e
                )

            decode_map: dict[str, int] = {}
            if decode is not None:
                mixed_tokens = self._mlx_runner.decode_batch_finalize(decode)
                decode_map = {
                    rid: tok for rid, tok in zip(decode.req_ids, mixed_tokens)
                }

            next_tokens_list = []
            for req in reqs:
                if req.rid in decode_map:
                    next_tokens_list.append(decode_map[req.rid])
                elif req.rid in extend_map:
                    next_tokens_list.append(extend_map[req.rid])
                else:
                    next_tokens_list.append(prefill_map[req.rid])

        else:
            raise ValueError(f"Unknown MLX async mode: {mode}")

        next_token_ids = torch.tensor(next_tokens_list, dtype=torch.long, device="cpu")
        return GenerationBatchResult(
            logits_output=LogitsProcessorOutput(next_token_logits=None),
            next_token_ids=next_token_ids,
            can_run_cuda_graph=False,
        )
