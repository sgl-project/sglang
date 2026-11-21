"""EAGLE owns admission, requests and outputs. The executor owns graph replay."""

from __future__ import annotations

import torch

from sglang.kernels.plugin_kernels.npu.spec import (
    DraftExecutor,
    DraftExtendIO,
    DraftLoopIO,
)
from sglang.srt.kernel_plugins.npu.admission import all_admit, candidate_present
from sglang.srt.model_executor.forward_context import ForwardContext, forward_context
from sglang.srt.speculative.eagle_megakernel import EagleDraftRuntime
from sglang.srt.speculative.eagle_utils import per_step_draft_out_cache_loc


class NpuEagleDraftRuntime(EagleDraftRuntime):
    def __init__(self, worker, binding):
        self.binding = binding
        self.decode_executor: DraftExecutor = binding
        self.extend_executor: DraftExecutor = binding
        self.worker = worker
        self.steps = worker.speculative_num_steps
        self.hidden_size = worker.draft_runner.model_config.hidden_size

    def bind_executors(self, *, decode: DraftExecutor, extend: DraftExecutor):
        """Called during runner setup on every rank, before serving requests."""
        self.decode_executor = decode
        self.extend_executor = extend

    def propose_chain(self, batch, forward_batch):
        if forward_batch.forward_mode.is_idle():
            return None
        req = batch.reqs[0] if len(batch.reqs) == 1 else None
        if self.binding.mode == "fused_extend":
            lease = getattr(req, "ascend_candidate", None)
            valid = lease is None or (
                lease[0].shape == (1, self.steps)
                and lease[2] == req.kv.req_pool_idx
                and torch.equal(lease[1], forward_batch.positions)
            )
            if not candidate_present(lease is not None, valid):
                return None
            req.ascend_candidate = None
            self.binding.hit("consume_fused_chain")
            return lease[0]

        if not all_admit(
            (not self.binding.pd_decode or getattr(req, "npu_draft_kv_ready", False))
            and self.binding.supports_decode(
                batch, forward_batch, self.worker.hot_token_id
            )
            and self.decode_executor.covers("decode", batch)
        ):
            return None
        seed = forward_batch.spec_info
        slots = per_step_draft_out_cache_loc(
            forward_batch.out_cache_loc, 1, 1, self.steps
        )
        io = DraftLoopIO(
            forward_batch,
            seed.topk_index.reshape(-1),
            forward_batch.positions,
            seed.hidden_states,
            slots[: self.steps - 1],
        )
        backend = self.worker.draft_attn_backend.attn_backends[0]
        with forward_context(
            ForwardContext(attn_backend=backend), forward_batch=forward_batch
        ):
            tokens = self.decode_executor.execute_decode(io)
        if tokens.shape != (1, self.steps):
            raise RuntimeError("Invalid NextN suffix width")
        return tokens.to(dtype=seed.topk_index.dtype, copy=True)

    def plan_fused_extend(self, batch, result):
        if self.binding.mode != "fused_extend" or batch.forward_mode.is_idle():
            return None
        if not all_admit(
            self.binding.supports_extend(batch, result, self.worker.hot_token_id)
            and self.extend_executor.covers("extend", batch),
            outstanding_chain=any(
                getattr(r, "ascend_candidate", None) is not None for r in batch.reqs
            ),
        ):
            return None
        return self._execute_extend

    def _execute_extend(self, batch, forward_batch, select, next_input):
        backend = (
            self.worker.draft_extend_attn_backend
            or self.worker.draft_runner.attn_backend
        )
        with forward_context(
            ForwardContext(attn_backend=backend), forward_batch=forward_batch
        ):
            result = self.extend_executor.execute_extend(DraftExtendIO(forward_batch, select))
        hidden = result.first_hidden_states
        if (
            result.token_ids.shape != (1, self.steps)
            or hidden is None
            or hidden.shape != (1, self.hidden_size)
        ):
            raise RuntimeError("Invalid NextN extend output")
        # Replay may overwrite both outputs before the overlap scheduler consumes them.
        tokens = result.token_ids.to(torch.int64, copy=True)
        next_input.topk_index = tokens[:, :1]
        next_input.topk_p = torch.ones_like(next_input.topk_index, dtype=torch.float32)
        next_input.hidden_states = hidden.clone()
        next_input.dsa_topk_indices = None
        expected = forward_batch.positions.index_select(0, select.long()) + 1
        req = batch.reqs[0]
        req.ascend_candidate = (tokens, expected, req.kv.req_pool_idx)
