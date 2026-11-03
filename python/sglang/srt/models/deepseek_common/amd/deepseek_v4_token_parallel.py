"""Token-parallel prefill for DeepSeek-V4 on ROCm (opt-in).

Under pure TP every rank recomputes the replicated per-token work of a layer for
all T prefill tokens: the mHC mixing and norms, the input projections (wqkv_a,
both compressors' wkv_gate, the indexer weights_proj), the indexer scoring and
the router. Token-parallel prefill keeps 1/tp of the [T, hc_mult, hidden]
residual on each rank and runs that work on the local rows only. Attention heads
and MoE experts stay TP-sharded: their inputs are all-gathered, and the two
per-layer all-reduces become reduce-scatters, so the communication volume per
layer is unchanged.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Dict, Optional, Tuple

import torch
import torch.nn.functional as F

from sglang.srt.environ import envs
from sglang.srt.runtime_context import get_forward, get_parallel
from sglang.srt.utils import is_gfx95_supported, is_gfx1250_supported, is_hip

if TYPE_CHECKING:
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch

# The ForwardBatch attribute that carries the per-forward layout.
_STATE_ATTR = "_dsv4_token_parallel"


class TokenParallelState:
    """Per-forward token-parallel layout over the TP group."""

    def __init__(self, num_tokens: int, input_ids: torch.Tensor):
        self.group = get_parallel().tp_group
        tp_size = self.group.world_size
        self.num_tokens = num_tokens
        self.padded = -(-num_tokens // tp_size) * tp_size
        self.local = self.padded // tp_size
        self.lo = self.group.rank_in_group * self.local
        # This rank's real rows; the last rank's tail rows past num_tokens are padding.
        self.hi = min(self.lo + self.local, num_tokens)
        self.input_ids_padded = F.pad(input_ids, (0, self.padded - num_tokens))
        # Outputs computed on local rows and all-gathered, keyed by the consuming module.
        self._pending: Dict[int, torch.Tensor] = {}

    def scatter(self, x: torch.Tensor) -> torch.Tensor:
        """Replicated [T, ...] -> this rank's [local, ...] rows."""
        if self.padded != self.num_tokens:
            x = torch.cat(
                [x, x.new_zeros((self.padded - self.num_tokens, *x.shape[1:]))]
            )
        return x[self.lo : self.lo + self.local].contiguous()

    def all_gather(self, x: torch.Tensor, narrow: bool = True) -> torch.Tensor:
        """[local, ...] on every rank -> [T, ...] (or [padded, ...])."""
        out = x.new_empty((self.padded, *x.shape[1:]))
        self.group.all_gather_into_tensor(out, x.contiguous())
        return out[: self.num_tokens] if narrow else out

    def reduce_scatter(self, x: torch.Tensor) -> torch.Tensor:
        """Per-rank partial sums [T or padded, ...] -> summed [local, ...] rows."""
        if x.shape[0] != self.padded:
            x = F.pad(x, (0, 0) * (x.dim() - 1) + (0, self.padded - x.shape[0]))
        out = x.new_empty((self.local, *x.shape[1:]))
        self.group.reduce_scatter_tensor(out, x.contiguous())
        return out

    def put(self, module: torch.nn.Module, value: torch.Tensor) -> None:
        self._pending[id(module)] = value

    def take(self, module: torch.nn.Module) -> Optional[torch.Tensor]:
        return self._pending.pop(id(module), None)


def token_parallel_state(forward_batch) -> Optional[TokenParallelState]:
    return getattr(forward_batch, _STATE_ATTR, None)


def enter_token_parallel(
    model,
    forward_batch: ForwardBatch,
    hidden_states: torch.Tensor,
    input_ids: torch.Tensor,
    *,
    eligible: bool,
) -> Optional[TokenParallelState]:
    """Attach a TokenParallelState to forward_batch when this prefill can run token-parallel;
    eligible is the caller's own gate (the plain layer loop, no TBO / DSpark capture)."""
    from sglang.srt.layers.cp.utils import is_cp_active
    from sglang.srt.layers.moe import get_moe_a2a_backend
    from sglang.srt.model_executor.forward_batch_info import ForwardMode
    from sglang.srt.model_executor.runner import get_is_capture_mode
    from sglang.srt.model_executor.runner_backend_utils.breakable_cuda_graph.context import (
        is_in_breakable_cuda_graph,
    )

    setattr(forward_batch, _STATE_ATTR, None)
    num_tokens = hidden_states.shape[0]
    par = get_parallel()
    layer = model.layers[model.start_layer]
    if not (
        eligible
        and envs.SGLANG_DSV4_TOKEN_PARALLEL_PREFILL.get()
        # gfx95 / gfx1250 quantize the attention input inside the norm, on full rows.
        and is_hip()
        and not (is_gfx95_supported() or is_gfx1250_supported())
        and forward_batch.forward_mode == ForwardMode.EXTEND
        and par.tp_size > 1
        # tp * tp tokens leave every rank at least one real row.
        and num_tokens
        >= max(envs.SGLANG_DSV4_TOKEN_PARALLEL_MIN_TOKENS.get(), par.tp_size**2)
        and par.attn_tp_size == par.tp_size
        and model.pp_group.world_size == 1
        and get_moe_a2a_backend().is_none()
        and not is_cp_active(forward_batch)
        and not get_is_capture_mode()
        and not is_in_breakable_cuda_graph()
        and not get_forward().sp_active
        and model.config.model_type == "deepseek_v4"
        and layer.self_attn.fuse_wqa_wkv
        and not layer.mlp._shared_expert_tp1
    ):
        return None
    state = TokenParallelState(num_tokens, input_ids)
    setattr(forward_batch, _STATE_ATTR, state)
    return state


def exit_token_parallel(forward_batch: ForwardBatch) -> None:
    setattr(forward_batch, _STATE_ATTR, None)


def project_inputs(
    attn, x_local: torch.Tensor, state: TokenParallelState
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Run MQALayer's replicated input projections (wqkv_a, both compressors' wkv_gate,
    the indexer weights_proj) on this rank's rows and all-gather their outputs, which
    are narrower than x.

    Returns a zero-stride [T, hidden] stand-in for x and the gathered qkv_a. On the
    prefill path every reader of x is served from these outputs (the compressors and
    the indexer take() theirs), so the stand-in only supplies the token count.
    """
    qkv_a = state.all_gather(attn.wqkv_a(x_local)[0])
    if attn.compressor is not None:
        state.put(
            attn.compressor,
            state.all_gather(attn.compressor._compute_wkv_gate(x_local)),
        )
    if attn.indexer is not None:
        indexer = attn.indexer
        state.put(indexer, state.all_gather(indexer.weights_proj(x_local)[0]))
        state.put(
            indexer.compressor,
            state.all_gather(indexer.compressor._compute_wkv_gate(x_local)),
        )
    return x_local[:1].expand(state.num_tokens, -1), qkv_a


def run_moe(
    moe,
    hidden_states: torch.Tensor,
    forward_batch: ForwardBatch,
    state: TokenParallelState,
) -> torch.Tensor:
    """The MoE block on this rank's rows. The router is replicated, so it routes the
    local rows and the per-row top-k is all-gathered. Experts and the shared expert
    stay TP-sharded: they run on all (padded) tokens with the post-experts all-reduce
    left out, and the partial sums are reduce-scattered back to this rank's rows."""
    local_ids = state.input_ids_padded[state.lo : state.lo + state.local]
    topk_local = moe._route(hidden_states, input_ids_global=local_ids)
    topk_output = type(topk_local)(
        *(
            state.all_gather(f, narrow=False)
            if isinstance(f, torch.Tensor) and f.dim() > 0 and f.shape[0] == state.local
            else f
            for f in topk_local
        )
    )
    hidden_states = state.all_gather(hidden_states, narrow=False)
    with get_forward().scoped(mlp_reduce_scatter=True):
        hidden_states = moe.forward_normal(
            hidden_states,
            input_ids=state.input_ids_padded,
            input_ids_global=state.input_ids_padded,
            topk_output=topk_output,
        )
    return state.reduce_scatter(hidden_states)
