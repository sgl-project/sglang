from typing import Optional, Tuple

import torch

from sglang.jit_kernel.utils import (
    cache_once,
    is_arch_support_pdl,
    load_jit,
    make_cpp_args,
)
from sglang.srt.utils import is_cuda, is_hip

from .utils import make_name

_is_cuda = is_cuda()
_is_hip = is_hip()


@cache_once
def _jit_fused_rope_module():
    args = make_cpp_args(is_arch_support_pdl())
    return load_jit(
        make_name("fused_rope"),
        *args,
        cuda_files=["deepseek_v4/rope.cuh"],
        cuda_wrappers=[("forward", f"FusedQKRopeKernel<{args}>::forward")],
    )


@cache_once
def _jit_main_q_norm_rope_module(
    dtype: torch.dtype,
    head_dim: int,
    rope_dim: int,
):
    """Main MLA path Q kernel: rmsnorm-self + RoPE, warp per (token, head)."""
    args = make_cpp_args(dtype, head_dim, rope_dim, is_arch_support_pdl())
    return load_jit(
        make_name("main_q_norm_rope"),
        *args,
        cuda_files=["deepseek_v4/main_norm_rope.cuh"],
        cuda_wrappers=[
            ("forward", f"FusedQNormRopeKernel<{args}>::forward"),
        ],
    )


@cache_once
def _jit_main_k_norm_rope_flashmla_module(
    dtype: torch.dtype,
    head_dim: int,
    rope_dim: int,
    page_size: int,
):
    """Main MLA path K kernel: rmsnorm + RoPE + write to FlashMLA paged cache."""
    args = make_cpp_args(dtype, head_dim, rope_dim, page_size, is_arch_support_pdl())
    return load_jit(
        make_name("main_k_norm_rope_flashmla"),
        *args,
        cuda_files=["deepseek_v4/main_norm_rope.cuh"],
        cuda_wrappers=[
            ("forward", f"FusedKNormRopeFlashMLAKernel<{args}>::forward"),
        ],
    )


@cache_once
def _jit_main_q_indexer_rope_hadamard_quant_module(dtype: torch.dtype):
    """C4 indexer Q kernel: RoPE + 128-pt Hadamard + fp8 act-quant"""
    args = make_cpp_args(dtype, is_arch_support_pdl())
    return load_jit(
        make_name("main_q_indexer_rope_hadamard_quant"),
        *args,
        cuda_files=["deepseek_v4/main_norm_rope.cuh"],
        cuda_wrappers=[
            ("forward", f"FusedQIndexerRopeHadamardQuantKernel<{args}>::forward"),
        ],
    )


# V3.2 lays q out as [rope | nope] (V4 is [nope | rope]) -> kRopeFirst=true, and
# drops the Hadamard rotation (kHadamard=false).
@cache_once
def _jit_main_q_indexer_rope_first_quant_module(dtype: torch.dtype):
    args = make_cpp_args(dtype, is_arch_support_pdl(), True, False)
    return load_jit(
        make_name("main_q_indexer_rope_first_quant"),
        *args,
        cuda_files=["deepseek_v4/main_norm_rope.cuh"],
        cuda_wrappers=[
            ("forward", f"FusedQIndexerRopeHadamardQuantKernel<{args}>::forward"),
        ],
    )


@cache_once
def _jit_main_q_indexer_rope_hadamard_fp4_quant_module(dtype: torch.dtype):
    args = make_cpp_args(dtype, is_arch_support_pdl())
    return load_jit(
        make_name("main_q_indexer_rope_hadamard_fp4_quant"),
        *args,
        cuda_files=["deepseek_v4/main_norm_rope.cuh"],
        cuda_wrappers=[
            ("forward", f"FusedQIndexerRopeHadamardFp4QuantKernel<{args}>::forward"),
        ],
    )


def fused_rope_inplace(
    q: torch.Tensor,
    k: Optional[torch.Tensor],
    freqs_cis: torch.Tensor,
    positions: torch.Tensor,
    inverse: bool = False,
) -> None:
    """Apply rotary embeddings to both Q and K in a single fused CUDA kernel.

    Args:
        q: [batch_size, num_q_heads, rope_dim] bfloat16
        k: [batch_size, num_k_heads, rope_dim] bfloat16 or None
        freqs_cis: [max_seq_len, rope_dim // 2] complex64 (full table)
        positions: [batch_size] int32 or int64, indices into freqs_cis
        inverse: if True, apply inverse rotation (conjugate freqs)
    """
    if _is_hip:
        from sglang.srt.layers.deepseek_v4_rope import apply_rotary_emb_triton

        apply_rotary_emb_triton(q, freqs_cis, positions=positions, inverse=inverse)
        if k is not None:
            apply_rotary_emb_triton(k, freqs_cis, positions=positions, inverse=inverse)
        return

    freqs_real = torch.view_as_real(freqs_cis).flatten(-2).contiguous()
    module = _jit_fused_rope_module()
    module.forward(q, k, freqs_real, positions, inverse)


def fused_q_norm_rope(
    q_input: torch.Tensor,
    q_output: torch.Tensor,
    eps: float,
    freqs_cis: torch.Tensor,
    positions: torch.Tensor,
) -> None:
    freqs_real = torch.view_as_real(freqs_cis).flatten(-2)
    head_dim = q_input.shape[-1]
    rope_dim = freqs_real.shape[-1]
    module = _jit_main_q_norm_rope_module(q_input.dtype, head_dim, rope_dim)
    module.forward(q_input, q_output, freqs_real, positions, eps)


def fused_q_indexer_rope_hadamard_quant(
    q_input: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: float,
    freqs_cis: torch.Tensor,
    positions: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    freqs_real = torch.view_as_real(freqs_cis).flatten(-2)
    q_fp8 = torch.empty(q_input.shape, dtype=torch.float8_e4m3fn, device=q_input.device)
    weights_out = torch.empty(
        (*q_input.shape[:-1], 1), dtype=torch.float32, device=q_input.device
    )
    if _is_hip:
        torch.ops.sgl_kernel.dsv4_fused_q_indexer_rope_hadamard_quant(
            q_input,
            q_fp8,
            weight,
            weights_out,
            float(weight_scale),
            freqs_real,
            positions,
        )
    elif _is_cuda:
        module = _jit_main_q_indexer_rope_hadamard_quant_module(q_input.dtype)
        module.forward(
            q_input,
            q_fp8,
            weight,
            weights_out,
            float(weight_scale),
            freqs_real,
            positions,
        )
    else:
        from .fused_q_indexer_rope_hadamard_quant_torch import (
            fused_q_indexer_rope_hadamard_quant_torch,
        )

        fused_q_indexer_rope_hadamard_quant_torch(
            q_input,
            q_fp8,
            weight,
            weights_out,
            float(weight_scale),
            freqs_real,
            positions,
        )
    return q_fp8, weights_out


def fused_q_indexer_rope_first_quant(
    q_input: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: float,
    cos_sin_cache: torch.Tensor,
    positions: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """DeepSeek-V3.2 only. Indexer Q: RoPE on the leading dims + fp8 act-quant. CUDA only."""
    q_fp8 = torch.empty(q_input.shape, dtype=torch.float8_e4m3fn, device=q_input.device)
    weights_out = torch.empty(
        (*q_input.shape[:-1], 1), dtype=torch.float32, device=q_input.device
    )
    module = _jit_main_q_indexer_rope_first_quant_module(q_input.dtype)
    module.forward(
        q_input,
        q_fp8,
        weight,
        weights_out,
        float(weight_scale),
        cos_sin_cache,
        positions,
    )
    return q_fp8, weights_out


def fused_q_indexer_rope_hadamard_fp4_quant(
    q_input: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: float,
    freqs_cis: torch.Tensor,
    positions: torch.Tensor,
) -> Tuple[Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
    if _is_hip:
        raise RuntimeError("DeepSeek V4 FP4 indexer requires the CUDA fused Q path.")
    freqs_real = torch.view_as_real(freqs_cis).flatten(-2)
    q_fp4 = torch.empty(
        (*q_input.shape[:-1], q_input.shape[-1] // 2),
        dtype=torch.int8,
        device=q_input.device,
    )
    q_sf = torch.empty(q_input.shape[:-1], dtype=torch.int32, device=q_input.device)
    weights_out = torch.empty(
        (*q_input.shape[:-1], 1), dtype=torch.float32, device=q_input.device
    )
    module = _jit_main_q_indexer_rope_hadamard_fp4_quant_module(q_input.dtype)
    module.forward(
        q_input,
        q_fp4,
        q_sf,
        weight,
        weights_out,
        float(weight_scale),
        freqs_real,
        positions,
    )
    return (q_fp4, q_sf), weights_out


def fused_k_norm_rope_flashmla(
    kv: torch.Tensor,
    kv_weight: torch.Tensor,
    eps: float,
    freqs_cis: torch.Tensor,
    positions: torch.Tensor,
    out_loc: torch.Tensor,
    kvcache: torch.Tensor,
    page_size: int,
) -> None:
    freqs_real = torch.view_as_real(freqs_cis).flatten(-2)
    head_dim = kv.shape[-1]
    rope_dim = freqs_real.shape[-1]
    module = _jit_main_k_norm_rope_flashmla_module(
        kv.dtype, head_dim, rope_dim, page_size
    )
    module.forward(kv, kv_weight, freqs_real, positions, out_loc, kvcache, eps)
