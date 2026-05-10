import torch
from aiter.ops.triton.fused_kv_cache import fused_qk_rope_cat_and_cache_mla
from aiter.ops.triton.fused_qk_concat import fused_qk_rope_cat
from aiter.tuned_gemm import tgemm

from sglang.srt.layers.quantization.fp8_kernel import fp8_dtype
from sglang.srt.utils.common import get_torch_compile_disable_decorator

__all__ = ["fused_qk_rope_cat", "fused_qk_rope_cat_and_cache_mla"]


def aiter_dsv3_router_gemm(
    hidden_states: torch.Tensor,
    weight: torch.Tensor,
):
    """Use aiter tuned GEMM dispatcher (tgemm.mm) to automatically select the GEMM kernel."""
    return tgemm.mm(hidden_states, weight.detach(), otype=hidden_states.dtype)


def get_dsv3_gemm_output_zero_allocator_size(
    n_routed_experts: int, num_moe_layers: int, allocate_size: int, embedding_dim: int
):
    if embedding_dim != 7168 or n_routed_experts != 256:
        return 0

    per_layer_size = 256 * (allocate_size + n_routed_experts)

    return num_moe_layers * per_layer_size


# fused_qk_rope_cat_and_cache_mla plus attn_mqa both modify KV cache
# hit out-of-place issue with KV cache the large tensor
@get_torch_compile_disable_decorator(True)
def rope_plus_attn_mqa(
    mla_inst,
    q_nope_out,
    q_pe,
    k_nope,
    k_pe,
    positions,
    forward_batch,
    topk_indices,
) -> torch.Tensor:

    kv_cache_dtype = (
        fp8_dtype if mla_inst.kv_cache_dtype == "fp8_e4m3" else q_nope_out.dtype
    )

    k = forward_batch.token_to_kv_pool.get_key_buffer(mla_inst.attn_mqa.layer_id)

    q = fused_qk_rope_cat_and_cache_mla(
        q_nope_out,
        q_pe,
        k_nope,
        k_pe,
        k,
        forward_batch.out_cache_loc,
        positions,
        mla_inst.rotary_emb.cos_cache,
        mla_inst.rotary_emb.sin_cache,
        mla_inst.attn_mqa.k_scale,
        mla_inst.rotary_emb.is_neox_style,
        q_out_dtype=kv_cache_dtype,
    )[0]

    attn_output = mla_inst.attn_mqa(
        q,
        k,
        k_nope,
        forward_batch,
        save_kv_cache=False,
        **(dict(topk_indices=topk_indices) if topk_indices is not None else {}),
    )

    return attn_output
