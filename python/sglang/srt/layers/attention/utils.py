import torch

from sglang.jit_kernel.utils import is_arch_support_pdl
from sglang.srt.layers.attention.triton_ops.cache_ops import (
    concat_and_cast_mha_k_kernel as concat_and_cast_mha_k_kernel,
)
from sglang.srt.layers.attention.triton_ops.cache_ops import (
    concat_and_cast_mha_k_triton as concat_and_cast_mha_k_triton,
)
from sglang.srt.layers.attention.triton_ops.cache_ops import (
    launch_reshape_and_cache_flash as launch_reshape_and_cache_flash,
)
from sglang.srt.layers.attention.triton_ops.cache_ops import (
    reshape_and_cache_flash as reshape_and_cache_flash,
)
from sglang.srt.layers.attention.triton_ops.kv_indices import (
    create_flashinfer_kv_indices_triton as create_flashinfer_kv_indices_triton,
)
from sglang.srt.layers.attention.triton_ops.kv_indices import (
    create_flashmla_kv_indices_triton as create_flashmla_kv_indices_triton,
)
from sglang.srt.layers.attention.triton_ops.kv_indices import (
    get_num_kv_index_blocks_flashmla as get_num_kv_index_blocks_flashmla,
)
from sglang.srt.layers.attention.triton_ops.kv_indices import (
    get_num_page_per_block_flashmla as get_num_page_per_block_flashmla,
)
from sglang.srt.layers.attention.triton_ops.pad import (
    pad_sequence_with_mask as pad_sequence_with_mask,
)
from sglang.srt.layers.attention.triton_ops.pad import (
    pad_sequence_with_mask_kernel as pad_sequence_with_mask_kernel,
)
from sglang.srt.layers.attention.triton_ops.pad import (
    seqlens_expand_kernel as seqlens_expand_kernel,
)
from sglang.srt.layers.attention.triton_ops.pad import (
    seqlens_expand_triton as seqlens_expand_triton,
)
from sglang.srt.layers.attention.triton_ops.rope_cache import (
    fused_qk_rope_reshape_and_cache as fused_qk_rope_reshape_and_cache,
)
from sglang.srt.utils import is_cuda

_is_cuda = is_cuda()

if _is_cuda:
    from sglang.jit_kernel.concat_mla import concat_mla_absorb_q


# When num_kv_heads=1, we have tensors with degenerate strides,
# For example, as below, where we have stride[-3] == stride[-2]:
# - shape: [num_pages, 1, 64, 128]
# - stride: [8192, 128, 128, 1]
# This will cause TMA desc validation fail in flashinfer (trtllm-mha backend).
#
# See: https://github.com/flashinfer-ai/flashinfer/issues/2232
def canonicalize_stride(tensor: torch.Tensor) -> torch.Tensor:
    """
    Adjust degenerate strides for a tensor, make it canonical.
    """
    sizes = tensor.size()
    strides = tensor.stride()
    ndim = tensor.dim()

    need_fix = any(
        sizes[i] == 1 and strides[i] == strides[i + 1] for i in range(ndim - 1)
    )

    if not need_fix:
        return tensor

    # canonicalize the stride
    # Example:
    # - shape: [num_pages, 1, 64, 128]
    # - stride: [8192, 128, 128, 1] (wrong!)
    # Gives new stride: [8192, 8192, 128 ,1] (correct!)
    new_strides = [0] * ndim
    new_strides[-1] = 1
    for i in range(ndim - 2, -1, -1):
        new_strides[i] = new_strides[i + 1] * sizes[i + 1]

    return tensor.as_strided(sizes, new_strides)


def mla_quantize_and_rope_for_fp8(
    q_nope: torch.Tensor,
    q_rope: torch.Tensor,
    k_nope: torch.Tensor,
    k_rope: torch.Tensor,
    pos_ids: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    is_neox: bool,
    kv_lora_rank: int,
    qk_rope_head_dim: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    import flashinfer.rope

    """Quantize and apply RoPE for FP8 attention path.

        This function handles the FP8 quantization and RoPE application for MLA attention.
        It takes separate query/key nope and rope components, applies RoPE to the rope parts,
        quantizes all components to FP8, and merges the query components into a single tensor.

        Args:
            q_nope: Query no-position-encoding component [seq_len, num_heads, kv_lora_rank]
                - expected dtype: torch.bfloat16
            q_rope: Query RoPE component [seq_len, num_heads, qk_rope_head_dim]
                - expected dtype: torch.bfloat16
            k_nope: Key no-position-encoding component [seq_len, num_heads, kv_lora_rank]
                - expected dtype: torch.bfloat16
            k_rope: Key RoPE component [seq_len, num_heads, qk_rope_head_dim]
                - expected dtype: torch.bfloat16
            pos_ids: Position indices for each token
                - expected dtype: torch.int64 or torch.int32
            cos_sin_cache: Precomputed cosine/sine cache for RoPE
                - expected dtype: matches q_/k_ input dtype (torch.bfloat16)
            is_neox: Whether to use NeoX-style RoPE (interleaved) or GPT-style (half rotation)
            kv_lora_rank: Dimension of the no-position-encoding component
            qk_rope_head_dim: Dimension of the RoPE component

        Returns:
            tuple: (merged_q_out, k_nope_out, k_rope_out) quantized to FP8
                - merged_q_out: [seq_len, num_heads, kv_lora_rank + qk_rope_head_dim], dtype=torch.float8_e4m3fn
                - k_nope_out:   [seq_len, num_heads, kv_lora_rank], dtype=torch.float8_e4m3fn
                - k_rope_out:   [seq_len, num_heads, qk_rope_head_dim], dtype=torch.float8_e4m3fn
        """
    attn_dtype = torch.float8_e4m3fn
    q_len, num_heads = q_rope.shape[0], q_rope.shape[1]

    # Allocate output tensors with FP8 dtype
    # Query output will contain merged nope + rope components
    q_out = q_rope.new_empty(
        q_len,
        num_heads,
        kv_lora_rank + qk_rope_head_dim,
        dtype=attn_dtype,
    )

    # Key outputs maintain original shapes but with FP8 dtype
    k_rope_out = k_rope.new_empty(k_rope.shape, dtype=attn_dtype)
    k_nope_out = k_nope.new_empty(k_nope.shape, dtype=attn_dtype)

    # Apply RoPE and quantize all components in a single fused kernel call
    # This kernel handles:
    # 1. RoPE application to q_rope and k_rope using cos_sin_cache and positions
    # 2. Quantization of all components to FP8 format
    # 3. Output placement into pre-allocated tensors
    flashinfer.rope.mla_rope_quantize_fp8(
        q_rope=q_rope,
        k_rope=k_rope,
        q_nope=q_nope,
        k_nope=k_nope,
        cos_sin_cache=cos_sin_cache,
        pos_ids=pos_ids,
        is_neox=is_neox,
        quantize_dtype=attn_dtype,
        # Output tensor slicing: q_out contains [nope_part, rope_part]
        q_rope_out=q_out[..., kv_lora_rank:],  # RoPE part goes to end
        k_rope_out=k_rope_out,
        q_nope_out=q_out[..., :kv_lora_rank],  # Nope part goes to beginning
        k_nope_out=k_nope_out,
        # Quantization scales (set to 1.0 for no additional scaling)
        quant_scale_q=1.0,
        quant_scale_kv=1.0,
        enable_pdl=is_arch_support_pdl(),
    )

    return q_out, k_nope_out, k_rope_out


def concat_mla_absorb_q_general(q_nope, q_rope):
    if _is_cuda and q_nope.shape[-1] == 512 and q_rope.shape[-1] == 64:
        return concat_mla_absorb_q(q_nope, q_rope)
    else:
        return torch.cat([q_nope, q_rope], dim=-1)


def assert_buffer_fits(used: int, capacity: int, what: str, **context) -> None:
    """Safety guard: a preallocated cuda-graph buffer must hold the runtime write.

    The kv_indices / page_table scatter kernels bound writes only per-row, not
    against the destination buffer, so an undersized buffer silently overflows
    into the adjacent row. Fail fast on the host-known extent instead. All args
    are host ints, so this is always-on (no device sync, unlike async probes).
    """
    assert used <= capacity, f"{what}: used {used} > capacity {capacity}" + (
        f" ({', '.join(f'{k}={v}' for k, v in context.items())})" if context else ""
    )
