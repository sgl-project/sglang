import torch
from aiter.ops.triton.fused_kv_cache import fused_qk_rope_cat_and_cache_mla
from aiter.ops.triton.fused_qk_concat import fused_qk_rope_cat
from aiter.tuned_gemm import tgemm

from sglang.srt.utils import is_gfx95_supported, is_gfx942_supported

_IS_GFX942 = is_gfx942_supported()

__all__ = [
    "fused_fp8_bmm_rope_cat_and_cache_mla",
    "fused_qk_rope_cat",
    "fused_qk_rope_cat_and_cache_mla",
]

# This module is imported wherever AITER is on, gfx942 included, but the fused
# bmm+rope+cache op is gfx95-only. Import it behind the same predicate its one
# caller gates on, so an aiter build without the op cannot take down every
# DeepSeek import on another card. The name stays bound either way.
if is_gfx95_supported():
    from aiter.ops.triton.fusions.fused_bmm_rope_kv_cache import (
        fused_fp8_bmm_rope_cat_and_cache_mla,
    )
else:
    fused_fp8_bmm_rope_cat_and_cache_mla = None


def aiter_dsv3_router_gemm(
    hidden_states: torch.Tensor,
    weight: torch.Tensor,
):
    """Compute router logits without gfx942 split-K BF16 atomic accumulation."""
    if _IS_GFX942:
        # AITER's tuned router GEMM can accumulate split-K partials in BF16,
        # changing logits across identical forwards. Router score perturbations
        # change discrete expert selection and compound across decoder layers.
        # Routing is deliberately high precision, independently of FP8 experts.
        return torch.nn.functional.linear(
            hidden_states.float(), weight.detach().float()
        )
    return tgemm.mm(hidden_states, weight.detach(), otype=hidden_states.dtype)


def get_dsv3_gemm_output_zero_allocator_size(
    n_routed_experts: int, num_moe_layers: int, allocate_size: int, embedding_dim: int
):
    if embedding_dim != 7168 or n_routed_experts != 256:
        return 0

    per_layer_size = 256 * (allocate_size + n_routed_experts)

    return num_moe_layers * per_layer_size
