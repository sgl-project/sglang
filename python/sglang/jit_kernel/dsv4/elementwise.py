from typing import Optional, Tuple

import torch

from sglang.jit_kernel.utils import (
    cache_once,
    is_arch_support_pdl,
    load_jit,
    make_cpp_args,
)
from sglang.srt.utils import is_cuda_alike, is_hip

from .utils import make_name

_is_cuda_alike = is_cuda_alike()
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
    if _is_cuda_alike:
        module = _jit_main_q_norm_rope_module(q_input.dtype, head_dim, rope_dim)
        module.forward(q_input, q_output, freqs_real, positions, eps)
    else:
        fused_q_norm_rope_torch(q_input, q_output, freqs_real, positions, eps)


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
    else:
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


def fused_q_norm_rope_torch(
    q_input: torch.Tensor,  # (B, num_q_heads, head_dim)  any float dtype
    q_output: torch.Tensor,  # (B, num_q_heads, head_dim)  same dtype, pre-allocated
    freqs_cis: torch.Tensor,  # (max_pos, rope_dim) fp32, interleaved re/im
    positions: torch.Tensor,  # (B,) int32 or int64
    eps: float,
) -> None:
    """
    In-place fused warp-per-(token, head) RMSNorm-self + RoPE.
    Writes result into q_output (same shape as q_input).
    Pure-PyTorch reference implementation of FusedQNormRopeKernel.

    Matches the CUDA kernel in:
      python/sglang/jit_kernel/csrc/deepseek_v4/main_norm_rope.cuh

    Algorithm (per token, per head):
      1. RMSNorm-self  – normalize the full head_dim vector (no learned weight).
      2. NoPE region   – first (head_dim - rope_dim) elements written as-is.
      3. RoPE region   – last rope_dim elements rotated with freqs_cis.

    freqs_cis contract (from the call-site):
      torch.view_as_real(freqs_cis).flatten(-2)  →  (max_pos, rope_dim) fp32
      Layout is interleaved [re0, im0, re1, im1, ...] along the last axis,
      so rope_dim pairs map to rope_dim/2 complex rotations.
    """
    # ------------------------------------------------------------------ #
    # shapes / constants
    # ------------------------------------------------------------------ #
    B, H, head_dim = q_input.shape
    rope_dim = freqs_cis.shape[-1]  # interleaved: rope_dim = 2 * num_pairs
    nope_dim = head_dim - rope_dim

    assert rope_dim % 2 == 0, "rope_dim must be even (interleaved re/im)"
    assert freqs_cis.shape[-1] == rope_dim

    # ------------------------------------------------------------------ #
    # part 1: RMSNorm-self  (no learned weight)
    #   norm_factor = rsqrt(mean(x^2) + eps)
    # ------------------------------------------------------------------ #
    x = q_input.float()  # (B, H, head_dim)

    # mean of squares over the head dimension
    rms = x.pow(2).mean(dim=-1, keepdim=True)  # (B, H, 1)
    norm_factor = torch.rsqrt(rms + eps)  # (B, H, 1)
    x = x * norm_factor  # (B, H, head_dim)  – normalised

    # ------------------------------------------------------------------ #
    # part 2: RoPE on the last rope_dim elements
    #   freqs row for each token: shape (B, rope_dim)  [re0,im0,re1,im1,...]
    # ------------------------------------------------------------------ #
    # gather the per-token frequency rows
    freq_rows = freqs_cis[positions.long()]  # (B, rope_dim)
    # broadcast over heads: (B, 1, rope_dim)
    freq_rows = freq_rows.unsqueeze(1)

    # split interleaved [re, im, re, im, ...] into separate tensors
    # shape of each: (B, 1, rope_dim/2)
    freq_re = freq_rows[..., 0::2]  # cosines
    freq_im = freq_rows[..., 1::2]  # sines

    # rope tail from the normalised vector
    x_rope = x[..., nope_dim:]  # (B, H, rope_dim)
    x_re = x_rope[..., 0::2]  # (B, H, rope_dim/2)
    x_im = x_rope[..., 1::2]  # (B, H, rope_dim/2)

    # complex multiply: (x_re + i*x_im) * (freq_re + i*freq_im)
    rotated_re = x_re * freq_re - x_im * freq_im  # (B, H, rope_dim/2)
    rotated_im = x_re * freq_im + x_im * freq_re  # (B, H, rope_dim/2)

    # re-interleave back to (B, H, rope_dim)
    rotated = torch.stack([rotated_re, rotated_im], dim=-1).flatten(-2)

    # ------------------------------------------------------------------ #
    # part 3: assemble and write to q_output
    # ------------------------------------------------------------------ #
    out = torch.cat([x[..., :nope_dim], rotated], dim=-1)  # (B, H, head_dim)
    q_output.copy_(out.to(q_input.dtype))
