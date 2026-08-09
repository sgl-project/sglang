"""MLX-specific TpModelWorker subclass for Apple Silicon.

Routes forward passes through the MLX model runner, bypassing PyTorch
MPS.  A lightweight stub provides scheduler bookkeeping; the actual
attention KV data lives in MlxAttentionKVPool.

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
from sglang.srt.hardware_backend.mlx.sampling import (
    MlxLogprobSpec,
    MlxStepLogprobs,
    lazy_logprob_arrays,
)
from sglang.srt.managers.schedule_batch import ScheduleBatch
from sglang.srt.managers.tp_worker import TpModelWorker
from sglang.srt.managers.utils import GenerationBatchResult
from sglang.srt.model_executor.forward_batch_info import (
    CaptureHiddenMode,
    ForwardBatch,
    PPProxyTensors,
)
from sglang.srt.runtime_context import get_device, get_exec

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
            quantization=self.server_args.quantization,
            enable_sampling=get_device().mlx_enable_sampling,
            sampling_rng_seed=get_device().random_seed,
            deterministic_seeding=(
                get_exec().deterministic.enable_deterministic_inference
            ),
        )
        if self.server_args.max_total_tokens is not None:
            init_kwargs["pool_size"] = self.server_args.max_total_tokens
        self._mlx_runner = MlxModelRunner(**init_kwargs)

        self._model_runner = MlxModelRunnerStub(
            model_config=self.model_config,
            mem_fraction_static=self.server_args.mem_fraction_static,
            gpu_id=self.gpu_id,
            ps=self.ps,
            nccl_port=self.nccl_port,
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
        """Lazily initialize MLX cache pools after the stub pools are ready."""
        if not self._mlx_pool_initialized:
            self._mlx_runner.init_cache_pools(self._model_runner.req_to_token_pool)
            self._mlx_pool_initialized = True

    def forward_batch_generation(
        self,
        batch: Optional[ScheduleBatch],
        forward_batch: Optional[ForwardBatch] = None,
        pp_proxy_tensors: Optional[PPProxyTensors] = None,
        is_verify: bool = False,
        skip_attn_backend_init: Optional[bool] = None,  # deprecated
        *,
        capture_hidden_mode: Optional[CaptureHiddenMode] = None,
    ) -> GenerationBatchResult:
        """Override to route through MLX model runner."""
        if batch is not None:
            self._ensure_mlx_pool_initialized()
            return self._forward_batch_generation_mlx(batch)

        # Fallback to standard path for None batches
        return super().forward_batch_generation(
            batch,
            forward_batch,
            pp_proxy_tensors,
            is_verify,
            skip_attn_backend_init,
            capture_hidden_mode=capture_hidden_mode,
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

    def prepare_for_kv_cache_release(self, req) -> None:
        """Snapshot MLX auxiliary state at the scheduler's radix insert point."""
        if self._mlx_runner.has_request(req.rid):
            self._mlx_runner.store_auxiliary_state_for_request(req.rid)
            # Prefer the just-snapshotted live auxiliary state for the final
            # insert. Any older tracked slot is released during component cleanup.
            req.mamba_last_track_seqlen = None

    def _route_extend_request(self, rid: str, decoding_rids: set[str]) -> str:
        """Classify a request within an extend / mixed batch.

        Shared by the sync (:meth:`_forward_batch_generation_mlx`) and async
        (:meth:`_async_extend_batch`) paths so both route identically.

        Returns one of:

        * ``"prefill"``      -- not seen before; start a fresh prefill.
        * ``"decode"``       -- a genuine single-token decode step mixed into
          this batch (present in ``batch.decoding_reqs``).
        * ``"continuation"`` -- a chunked-prefill continuation.  Routing keys on
          request state, **not** ``seq_len``: a final continuation chunk can be
          exactly one token, which must still extend.  Routing it as a decode
          would drop the real token and feed the model its own previous-chunk
          prediction, silently corrupting the output.
        """
        if not self._mlx_runner.has_request(rid):
            return "prefill"
        if rid in decoding_rids:
            return "decode"
        return "continuation"

    @staticmethod
    def _chunk_needs_logits(req) -> bool:
        """False iff this extend chunk is a non-final chunked-prefill chunk.

        The scheduler truncates a chunked request's extend range below the
        tokens it already knows about; such a chunk's next-token output is
        discarded (the runner pops it as the stale intermediate token), so
        the runner may skip the logit head for it.
        """
        if req.extend_range is None:
            return True
        return req.extend_range.end >= len(req.full_untruncated_fill_ids)

    @staticmethod
    def _sampling_active(batch: ScheduleBatch) -> bool:
        return get_device().mlx_enable_sampling and batch.sampling_info is not None

    def _build_logit_edit_rows(
        self, batch: ScheduleBatch
    ) -> dict[str, mx.array] | None:
        """Pre-combine grammar vocab masks and logit_bias into one additive
        [vocab] float32 row per request, ready to enter the lazy graph.

        Grammar FSM state is current at every fresh launch — the previous
        token was finalized before this batch was scheduled — so the mask
        is knowable at graph-build time with no device sync.  The
        scheduler never chains grammar batches
        (:attr:`MlxPendingJob.chain_safe`), so a chained step never needs
        a stale mask.  Mask application reuses the grammar backend's own
        ``apply_vocab_mask`` on a zeros tensor, which keeps this
        backend-agnostic (xgrammar / llguidance / outlines).
        """
        if not self._sampling_active(batch):
            return None
        sinfo = batch.sampling_info
        # Mirror ForwardBatch.init_new's grammars population — the MLX paths
        # never build a ForwardBatch, so without this the list stays None
        # even when requests carry live grammar objects.
        sinfo.grammars = (
            [req.grammar for req in batch.reqs] if batch.has_grammar else None
        )
        has_grammar = bool(sinfo.grammars)
        if not has_grammar and sinfo.logit_bias is None:
            return None
        combined = torch.zeros(len(batch.reqs), sinfo.vocab_size, dtype=torch.float32)
        if has_grammar:
            sinfo.update_regex_vocab_mask()
            if sinfo.grammar_mask is not None:
                grammar_mask = sinfo.grammar_mask
                grammar_mask.grammar.apply_vocab_mask(
                    logits=combined,
                    vocab_mask=grammar_mask.vocab_mask.to("cpu"),
                )
                # Release promptly; mirrors the VRAM-leak note in the CUDA
                # ModelRunner._preprocess_logits.
                sinfo.grammar_mask = None
        if sinfo.logit_bias is not None:
            combined += sinfo.logit_bias.to("cpu")
        rows = mx.array(combined.numpy())
        return {req.rid: rows[i] for i, req in enumerate(batch.reqs)}

    def _logprob_rows(
        self, batch: ScheduleBatch
    ) -> dict[str, tuple[int, tuple[int, ...] | None]] | None:
        """Per-request (top_logprobs_num, token_ids) for logprob output."""
        if not self._sampling_active(batch) or not batch.return_logprob:
            return None
        tops = batch.top_logprobs_nums or [0] * len(batch.reqs)
        tids = batch.token_ids_logprobs or [None] * len(batch.reqs)
        rows = {}
        for req, top_k, token_ids in zip(batch.reqs, tops, tids):
            if req.return_logprob:
                rows[req.rid] = (
                    int(top_k or 0),
                    tuple(token_ids) if token_ids else None,
                )
        return rows or None

    @staticmethod
    def _logprob_spec_for(
        rows: dict[str, tuple[int, tuple[int, ...] | None]] | None,
        rids: list[str],
    ) -> MlxLogprobSpec | None:
        if rows is None or not any(rid in rows for rid in rids):
            return None
        return MlxLogprobSpec(
            top_ks=tuple(rows.get(rid, (0, None))[0] for rid in rids),
            token_ids=tuple(rows.get(rid, (0, None))[1] for rid in rids),
        )

    def _custom_logits_hook(self, batch: ScheduleBatch):
        """CPU edit hook for custom logit processors, or None.

        Only built for fresh pure-decode launches; the runner materializes
        the logits for the hook, so these batches never chain.
        """
        if not (
            self._sampling_active(batch)
            and batch.sampling_info.has_custom_logit_processor
        ):
            return None
        sinfo = batch.sampling_info

        def hook(logits_np):
            from sglang.srt.layers.sampler import apply_custom_logit_processor

            # torch.from_numpy shares memory with logits_np, so the
            # processors' in-place edits land in the returned array.
            logits_t = torch.from_numpy(logits_np)
            apply_custom_logit_processor(logits_t, sinfo)
            return logits_np

        return hook

    @staticmethod
    def _assemble_logprob_output(step_rows: dict[str, tuple], reqs: list):
        """Batch-ordered LogitsProcessorOutput from per-request logprob rows.

        Field shapes follow what ``move_logprobs_to_cpu`` and
        ``add_logprob_return_values`` consume: tensors for values the
        scheduler ``.tolist()``s, plain lists for token-id indices.
        """
        from sglang.srt.layers.logits_processor import LogitsProcessorOutput

        chosen, top_val, top_idx, tid_val, tid_idx = [], [], [], [], []
        for req in reqs:
            row = step_rows.get(req.rid)
            if row is None:
                row = (0.0, [], [], [], [])
            chosen.append(row[0])
            top_val.append(torch.tensor(row[1], dtype=torch.float32))
            top_idx.append(torch.tensor(row[2], dtype=torch.long))
            tid_val.append(torch.tensor(row[3], dtype=torch.float32))
            tid_idx.append(list(row[4]))
        return LogitsProcessorOutput(
            next_token_logits=None,
            next_token_logprobs=torch.tensor(chosen, dtype=torch.float32),
            next_token_top_logprobs_val=top_val,
            next_token_top_logprobs_idx=top_idx,
            next_token_token_ids_logprobs_val=tid_val,
            next_token_token_ids_logprobs_idx=tid_idx,
        )

    @staticmethod
    def _step_logprob_rows(
        step: Optional[MlxStepLogprobs], rids: list[str]
    ) -> dict[str, tuple]:
        """Split a step's batch logprobs into per-request rows."""
        if step is None:
            return {}
        return {
            rid: (
                step.chosen[i],
                step.top_val[i],
                step.top_idx[i],
                step.token_ids_val[i],
                step.token_ids_idx[i],
            )
            for i, rid in enumerate(rids)
        }

    def _collect_step_logprobs(
        self,
        step_rows: dict[str, tuple],
        lazy_logprobs,
        rids: list[str],
    ) -> None:
        """Materialize one pending's lazy logprobs into ``step_rows``."""
        step = self._mlx_runner.collect_logprobs(lazy_logprobs)
        step_rows.update(self._step_logprob_rows(step, rids))

    def _forward_batch_generation_mlx(
        self, batch: ScheduleBatch
    ) -> GenerationBatchResult:
        """Run forward pass through the MLX model runner.

        Token selection (greedy, or in-graph sampling when
        ``--mlx-enable-sampling`` is set) happens inside the runner's
        lazy graph; this method only routes requests.
        """
        from sglang.srt.layers.logits_processor import LogitsProcessorOutput

        forward_mode = batch.forward_mode
        reqs = batch.reqs

        if forward_mode.is_idle():
            return GenerationBatchResult(
                logits_output=LogitsProcessorOutput(next_token_logits=None),
                can_run_cuda_graph=False,
            )

        self._cleanup_stale_rids(forward_mode, {req.rid for req in reqs})

        next_token_ids_list: list[int] = []
        edit_rows = self._build_logit_edit_rows(batch)
        logprob_rows = self._logprob_rows(batch)
        step_logprob_rows: dict[str, tuple] = {}

        if forward_mode.is_extend():
            # Ensure pool is up-to-date before pool-backed attention reads it
            # for prefix-cached prefills.  Only runs on extend batches.
            self._mlx_runner.flush_all_decode_kv()
            input_ids_cpu = batch.input_ids.cpu().tolist()
            out_cache_loc_cpu = batch.out_cache_loc.cpu().tolist()
            extend_seq_lens = batch.extend_lens

            offset = 0  # into input_ids_cpu
            slot_offset = 0  # into out_cache_loc_cpu
            prefill_rids: list[tuple[str, int]] = []
            extend_rids: list[tuple[str, int]] = []
            decode_rids: list[str] = []
            # Genuine decode steps mixed into this extend batch; see
            # _route_extend_request.
            decoding_rids = {r.rid for r in (batch.decoding_reqs or [])}

            for i, req in enumerate(reqs):
                seq_len = extend_seq_lens[i]
                req_token_ids = input_ids_cpu[offset : offset + seq_len]
                req_new_slots = out_cache_loc_cpu[slot_offset : slot_offset + seq_len]
                offset += seq_len
                slot_offset += seq_len

                route = self._route_extend_request(req.rid, decoding_rids)
                if route == "continuation":
                    pending_e = self._mlx_runner.extend_start(
                        req_id=req.rid,
                        new_token_ids=req_token_ids,
                        new_slot_ids=req_new_slots,
                        needs_logits=self._chunk_needs_logits(req),
                        logit_edit_row=edit_rows[req.rid] if edit_rows else None,
                        logprob_spec=self._logprob_spec_for(logprob_rows, [req.rid]),
                    )
                    self._mlx_runner.eval_extend(pending_e)
                    next_token = self._mlx_runner.extend_finalize(pending_e)
                    self._collect_step_logprobs(
                        step_logprob_rows, pending_e.lazy_logprobs, [req.rid]
                    )
                    extend_rids.append((req.rid, next_token))
                elif route == "decode":
                    decode_rids.append(req.rid)
                else:  # "prefill"
                    prefix_slot_ids = req.prefix_indices.tolist()
                    full_token_ids = list(req.get_fill_ids())
                    pending_p = self._mlx_runner.prefill_start(
                        req_id=req.rid,
                        new_token_ids=req_token_ids,
                        full_token_ids=full_token_ids,
                        prefix_slot_ids=prefix_slot_ids,
                        new_slot_ids=req_new_slots,
                        req_pool_idx=req.req_pool_idx,
                        req=req,
                        needs_logits=self._chunk_needs_logits(req),
                        logit_edit_row=edit_rows[req.rid] if edit_rows else None,
                        logprob_spec=self._logprob_spec_for(logprob_rows, [req.rid]),
                    )
                    self._mlx_runner.eval_prefill(pending_p)
                    next_token = self._mlx_runner.prefill_finalize(pending_p)
                    self._collect_step_logprobs(
                        step_logprob_rows, pending_p.lazy_logprobs, [req.rid]
                    )
                    prefill_rids.append((req.rid, next_token))

            # Batch decode all existing requests at once
            if decode_rids:
                decode_map = dict(
                    zip(
                        decode_rids,
                        self._sync_decode(
                            batch,
                            decode_rids,
                            edit_rows,
                            logprob_rows,
                            step_logprob_rows,
                            allow_hook=False,
                        ),
                    )
                )
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
            next_token_ids_list = self._sync_decode(
                batch,
                req_ids,
                edit_rows,
                logprob_rows,
                step_logprob_rows,
                allow_hook=True,
            )

        else:
            raise ValueError(
                f"MLX runner does not support forward mode: {forward_mode}"
            )

        next_token_ids = torch.tensor(
            next_token_ids_list, dtype=torch.long, device="cpu"
        )

        logits_output = (
            self._assemble_logprob_output(step_logprob_rows, reqs)
            if step_logprob_rows
            else LogitsProcessorOutput(next_token_logits=None)
        )
        return GenerationBatchResult(
            logits_output=logits_output,
            next_token_ids=next_token_ids,
            can_run_cuda_graph=False,
        )

    def _sync_decode(
        self,
        batch: ScheduleBatch,
        req_ids: list[str],
        edit_rows: dict[str, mx.array] | None,
        logprob_rows: dict[str, tuple[int, tuple[int, ...] | None]] | None,
        step_logprob_rows: dict[str, tuple],
        allow_hook: bool,
    ) -> list[int]:
        """Synchronous decode step with logit edits and logprob collection.

        ``allow_hook`` is False for decode rows mixed into an extend batch:
        ``apply_custom_logit_processor`` asserts its logits rows match the
        full ``sampling_info``, which only holds for pure-decode batches.
        """
        pending = self._mlx_runner.decode_batch_start(
            req_ids,
            edit_rows=self._stacked_edit_rows(edit_rows, req_ids),
            logprob_spec=self._logprob_spec_for(logprob_rows, req_ids),
            logits_hook=self._custom_logits_hook(batch) if allow_hook else None,
        )
        self._mlx_runner.eval_decode(pending)
        next_tokens = self._mlx_runner.decode_batch_finalize(pending)
        self._collect_step_logprobs(step_logprob_rows, pending.lazy_logprobs, req_ids)
        return next_tokens

    @staticmethod
    def _stacked_edit_rows(
        edit_rows: dict[str, mx.array] | None, req_ids: list[str]
    ) -> Optional[mx.array]:
        """Stack the per-request additive edit rows for a decode sub-batch."""
        if not edit_rows:
            return None
        return mx.stack([edit_rows[rid] for rid in req_ids])

    def async_forward_batch_generation_mlx(self, batch: ScheduleBatch) -> tuple[
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

        forward_mode = batch.forward_mode
        reqs = batch.reqs

        if forward_mode.is_idle():
            return None, [], [], None, "idle"

        self._cleanup_stale_rids(forward_mode, {req.rid for req in reqs})

        if forward_mode.is_decode():
            req_ids = [req.rid for req in reqs]
            pending_decode = self._mlx_runner.decode_batch_start(
                req_ids,
                edit_rows=self._stacked_edit_rows(
                    self._build_logit_edit_rows(batch), req_ids
                ),
                logprob_spec=self._logprob_spec_for(self._logprob_rows(batch), req_ids),
                logits_hook=self._custom_logits_hook(batch),
            )
            mx.async_eval(
                pending_decode.lazy_tokens,
                *lazy_logprob_arrays(pending_decode.lazy_logprobs),
            )
            return pending_decode.lazy_tokens, [], [], pending_decode, "decode"

        if forward_mode.is_extend():
            # TODO (changminbark): Implement per-batch flushing using prefix_slot_ids
            # Ensure the pool is up-to-date before pool-backed attention
            # reads it for prefix-cached prefills. Mirror the sync path.
            self._mlx_runner.flush_all_decode_kv()
            return self._async_extend_batch(batch)

        raise ValueError(
            f"MLX async runner does not support forward mode: {forward_mode}"
        )

    def _async_extend_batch(self, batch: ScheduleBatch) -> tuple[
        Union[mx.array, None],
        list[MlxPendingPrefill],
        list[MlxPendingExtend],
        Optional[MlxPendingDecode],
        str,
    ]:
        """Launch each request in an EXTEND batch lazily and kick GPU work."""
        reqs = batch.reqs
        input_ids_cpu = batch.input_ids.cpu().tolist()
        out_cache_loc_cpu = batch.out_cache_loc.cpu().tolist()
        extend_seq_lens = batch.extend_lens
        edit_rows = self._build_logit_edit_rows(batch)
        logprob_rows = self._logprob_rows(batch)

        offset = 0
        slot_offset = 0
        pending_prefills: list[MlxPendingPrefill] = []
        pending_extends: list[MlxPendingExtend] = []
        mixed_decode_rids: list[str] = []
        # Genuine decode steps mixed into this extend batch; see
        # _route_extend_request.
        decoding_rids = {r.rid for r in (batch.decoding_reqs or [])}

        for i, req in enumerate(reqs):
            seq_len = extend_seq_lens[i]
            req_token_ids = input_ids_cpu[offset : offset + seq_len]
            req_new_slots = out_cache_loc_cpu[slot_offset : slot_offset + seq_len]
            offset += seq_len
            slot_offset += seq_len

            route = self._route_extend_request(req.rid, decoding_rids)
            if route == "continuation":
                pending_extends.append(
                    self._mlx_runner.extend_start(
                        req_id=req.rid,
                        new_token_ids=req_token_ids,
                        new_slot_ids=req_new_slots,
                        needs_logits=self._chunk_needs_logits(req),
                        logit_edit_row=edit_rows[req.rid] if edit_rows else None,
                        logprob_spec=self._logprob_spec_for(logprob_rows, [req.rid]),
                    )
                )
            elif route == "decode":
                mixed_decode_rids.append(req.rid)
            else:  # "prefill"
                prefix_slot_ids = req.prefix_indices.tolist()
                full_token_ids = list(req.get_fill_ids())
                pending_prefills.append(
                    self._mlx_runner.prefill_start(
                        req_id=req.rid,
                        new_token_ids=req_token_ids,
                        full_token_ids=full_token_ids,
                        prefix_slot_ids=prefix_slot_ids,
                        new_slot_ids=req_new_slots,
                        req_pool_idx=req.req_pool_idx,
                        req=req,
                        needs_logits=self._chunk_needs_logits(req),
                        logit_edit_row=edit_rows[req.rid] if edit_rows else None,
                        logprob_spec=self._logprob_spec_for(logprob_rows, [req.rid]),
                    )
                )

        pending_mixed_decode: Optional[MlxPendingDecode] = None
        if mixed_decode_rids:
            pending_mixed_decode = self._mlx_runner.decode_batch_start(
                mixed_decode_rids,
                edit_rows=self._stacked_edit_rows(edit_rows, mixed_decode_rids),
                logprob_spec=self._logprob_spec_for(logprob_rows, mixed_decode_rids),
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
            async_args.extend(lazy_logprob_arrays(p.lazy_logprobs))
        for e in pending_extends:
            async_args.extend(self._mlx_runner.request_cache_arrays(e.req_id))
            async_args.extend(lazy_logprob_arrays(e.lazy_logprobs))
        if pending_mixed_decode is not None:
            async_args.append(pending_mixed_decode.lazy_tokens)
            async_args.extend(lazy_logprob_arrays(pending_mixed_decode.lazy_logprobs))
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
        arrays: list[mx.array] = []

        def collect(value):
            if isinstance(value, mx.array):
                arrays.append(value)
            elif value is None:
                return
            elif isinstance(value, (list, tuple)):
                for item in value:
                    collect(item)
            elif isinstance(value, dict):
                for item in value.values():
                    collect(item)

        for cache in cache_list:
            collect(getattr(cache, "state", ()))
        return arrays

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

        step_logprob_rows: dict[str, tuple] = {}

        if mode == "decode":
            assert decode is not None
            next_tokens_list = self._mlx_runner.decode_batch_finalize(decode)
            self._collect_step_logprobs(
                step_logprob_rows, decode.lazy_logprobs, decode.req_ids
            )

        elif mode == "extend":
            prefill_map: dict[str, int] = {}
            for pending_p in prefills:
                prefill_map[pending_p.req_id] = self._mlx_runner.prefill_finalize(
                    pending_p
                )
                self._collect_step_logprobs(
                    step_logprob_rows, pending_p.lazy_logprobs, [pending_p.req_id]
                )

            extend_map: dict[str, int] = {}
            for pending_e in extends:
                extend_map[pending_e.req_id] = self._mlx_runner.extend_finalize(
                    pending_e
                )
                self._collect_step_logprobs(
                    step_logprob_rows, pending_e.lazy_logprobs, [pending_e.req_id]
                )

            decode_map: dict[str, int] = {}
            if decode is not None:
                mixed_tokens = self._mlx_runner.decode_batch_finalize(decode)
                decode_map = {
                    rid: tok for rid, tok in zip(decode.req_ids, mixed_tokens)
                }
                self._collect_step_logprobs(
                    step_logprob_rows, decode.lazy_logprobs, decode.req_ids
                )

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
        logits_output = (
            self._assemble_logprob_output(step_logprob_rows, reqs)
            if step_logprob_rows
            else LogitsProcessorOutput(next_token_logits=None)
        )
        return GenerationBatchResult(
            logits_output=logits_output,
            next_token_ids=next_token_ids,
            can_run_cuda_graph=False,
        )
