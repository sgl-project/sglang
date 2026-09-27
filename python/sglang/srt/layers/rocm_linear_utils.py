from typing import Optional, Tuple

import torch
from aiter.ops.triton.fused_kv_cache import fused_qk_rope_cat_and_cache_mla
from aiter.ops.triton.fused_qk_concat import fused_qk_rope_cat
from aiter.tuned_gemm import tgemm

from sglang.kernels.ops.moe.rocm_router_gate import (
    rocm_router_gemv_split_k,
    rocm_router_max_tokens,
)
from sglang.srt.layers.attention.dsa.utils import dsa_use_prefill_cp
from sglang.srt.layers.utils.cp_utils import mla_use_prefill_cp
from sglang.srt.runtime_context import get_exec

__all__ = ["fused_qk_rope_cat", "fused_qk_rope_cat_and_cache_mla"]


def aiter_dsv3_router_gemm(
    hidden_states: torch.Tensor,
    weight: torch.Tensor,
):
    """Use aiter tuned GEMM dispatcher (tgemm.mm) to automatically select the GEMM kernel."""
    return tgemm.mm(hidden_states, weight.detach(), otype=hidden_states.dtype)


def aiter_dsv3_router_split_k_max_tokens(config, weight_dtype: torch.dtype) -> int:
    """Rows up to which ``aiter_dsv3_router_split_k`` serves the router of ``config``."""
    return rocm_router_max_tokens(
        num_experts=config.n_routed_experts,
        hidden_size=config.hidden_size,
        topk=config.num_experts_per_tok,
        weight_dtype=weight_dtype,
    )


def aiter_dsv3_router_split_k(
    gate, hidden_states: torch.Tensor, forward_batch=None
) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
    """The ROCm decode router for ``gate`` (a ``MoEGate``): an fp32 logits buffer plus
    the split-K partials whose fixed-order sum fills it, or None when ``gate.forward``
    applies. Only ``TopK.forward_cuda(..., router_logits_partials=partials)`` may read
    the buffer: it sums the partials into it inside the fused gate launch."""
    num_tokens = hidden_states.shape[0]
    if not 0 < num_tokens <= gate.rocm_router_max_tokens:
        return None
    if get_exec().deterministic.enable_deterministic_inference:
        return None
    if (
        not gate.is_deepseek_v4
        and forward_batch is not None
        and (
            dsa_use_prefill_cp(forward_batch, gate.dsa_enable_prefill_cp)
            or mla_use_prefill_cp(forward_batch, gate.mla_enable_prefill_cp)
        )
    ):
        return None
    partials = rocm_router_gemv_split_k(hidden_states, gate.weight)
    logits = torch.empty(
        (num_tokens, gate.weight.shape[0]),
        dtype=torch.float32,
        device=hidden_states.device,
    )
    return logits, partials


def get_dsv3_gemm_output_zero_allocator_size(
    n_routed_experts: int, num_moe_layers: int, allocate_size: int, embedding_dim: int
):
    if embedding_dim != 7168 or n_routed_experts != 256:
        return 0

    per_layer_size = 256 * (allocate_size + n_routed_experts)

    return num_moe_layers * per_layer_size
