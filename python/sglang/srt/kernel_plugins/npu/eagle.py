"""EAGLE owns admission, requests and outputs. The executor owns graph replay."""

from __future__ import annotations

from sglang.kernels.plugin_kernels.npu.spec import (
    DraftLoopIO,
)
from sglang.srt.kernel_plugins.npu.admission import all_admit
from sglang.srt.model_executor.forward_context import ForwardContext, forward_context
from sglang.srt.speculative.eagle_megakernel import EagleDraftRuntime
from sglang.srt.speculative.eagle_utils import per_step_draft_out_cache_loc


class NpuEagleDraftRuntime(EagleDraftRuntime):
    def __init__(self, worker, binding):
        self.binding = binding
        self.worker = worker
        self.steps = worker.speculative_num_steps

    def propose_chain(self, batch, forward_batch, shape_key=None):
        idle = forward_batch.forward_mode.is_idle()
        req = batch.reqs[0] if len(batch.reqs) == 1 else None
        runner = self.worker.cuda_graph_runner
        backend = getattr(runner, "backend", None)
        attention_backends = getattr(
            getattr(self.worker, "draft_attn_backend", None), "attn_backends", None
        )
        attention = attention_backends[0] if attention_backends else None
        can_run = False
        if not idle and attention is not None and hasattr(backend, "execute_draft_graph"):
            can_run = (
                not self.binding.pd_decode
                or (
                    req is not None
                    and getattr(req, "npu_draft_kv_generation", None)
                    == req.retraction_count
                )
            ) and self.binding.supports_decode(
                batch, forward_batch, self.worker.hot_token_id
            ) and getattr(backend, "has_draft_graph", lambda: False)()
        # Every rank that entered this call has to vote, including idle ranks.
        if not all_admit(can_run):
            return None
        if not can_run:
            return None
        seed = forward_batch.spec_info
        # topk=1 plans a write window of steps+1. The chain writes the first
        # ``steps`` columns; the last column belongs to verify.
        window = forward_batch.out_cache_loc[: self.steps]
        slots = per_step_draft_out_cache_loc(window, 1, 1, self.steps)
        io = DraftLoopIO(
            forward_batch,
            seed.topk_index.reshape(-1),
            forward_batch.positions,
            seed.hidden_states,
            slots[: self.steps - 1],
        )
        with forward_context(
            ForwardContext(attn_backend=attention), forward_batch=forward_batch
        ):
            tokens = backend.execute_draft_graph(io, shape_key)
        if tokens is None:
            return None
        if tokens.shape != (1, self.steps):
            raise RuntimeError("Invalid NextN suffix width")
        return tokens.to(dtype=seed.topk_index.dtype, copy=True)
