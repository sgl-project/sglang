"""JIT-compiled SM90 (Hopper) kernel for the Q8KV8 born-fp8 q-prep.

Fuses the per-head absorbed-q bmm (q_nope [T, H, K] bf16 x w_kc [H, K, N]
bf16, fp32 accumulate), the nope/rope concat, and the bf16 -> fp8_e4m3 cast
into one hand-written WGMMA kernel.  CUDA replacement for the Triton
``absorbed_bmm_concat_cast_q_fp8`` (triton_ops/cache_ops.py) with the
identical fp32 -> bf16 -> fp8 epilogue rounding chain; the rope half is
bit-exact vs ``concat_and_cast_q_fp8_pad``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from sglang.jit_kernel.utils import cache_once, load_jit, override_jit_cuda_arch
from sglang.kernel_api_logging import debug_kernel_api

if TYPE_CHECKING:
    from tvm_ffi.module import Module


N_LORA = 512  # kv_lora_rank (nope output dim)
ROPE_DIM = 64  # qk_rope_head_dim


@cache_once
def _jit_qprep_bf16_fp8_module() -> Module:
    if torch.cuda.get_device_capability()[0] != 9:
        raise RuntimeError("qprep_bf16_fp8_sm90 requires an SM90 (Hopper) GPU")
    with override_jit_cuda_arch(9, 0, "a"):
        return load_jit(
            "qprep_bf16_fp8_sm90",
            cuda_files=["qprep_bf16_fp8_sm90/entry.cuh"],
            cuda_wrappers=[("dispatch", "qprep_bf16_fp8_dispatch")],
            # Same minimal flag set as the sparse_mla_q8kv8_prefill_sm90 JIT
            # build (per-flag ablation there showed the rest are no-ops).
            extra_cuda_cflags=[
                "-O3",
                "-DNDEBUG",
                "-DCUTE_USE_PACKED_TUPLE=1",
                "-DCUTLASS_ENABLE_TENSOR_CORE_MMA=1",
                "--use_fast_math",
            ],
            extra_dependencies=["cutlass"],
        )


# torch._C._cuda_getCurrentRawStream returns the cudaStream_t pointer expected
# by the JIT wrapper (see sparse_mla_q8kv8_prefill_sm90.py).
_get_current_stream_raw = torch._C._cuda_getCurrentRawStream


@debug_kernel_api
def q8kv8_qprep_fwd(
    q_fp8_pad: torch.Tensor,
    q_nope: torch.Tensor,
    w_kc: torch.Tensor,
    q_rope: torch.Tensor,
    num_heads: int,
) -> None:
    """Fused absorbed-q bmm + nope/rope concat + bf16->fp8 cast ("born fp8" q).

    Mirrors the contract of ``absorbed_bmm_concat_cast_q_fp8``:

    * ``q_fp8_pad``: [num_tokens, pad_heads, N + ROPE] fp8_e4m3 destination;
      only ``[:, :num_heads, :]`` is written.
    * ``q_nope``: [num_tokens, H, K] bf16 pre-absorb q (strided views OK).
    * ``w_kc``: [H, K, N] bf16 absorbed weight with K contiguous
      (``stride(1) == 1``, the production N-major layout).
    * ``q_rope``: [num_tokens, H, ROPE] bf16 post-rope q (strided views OK).

    K (``qk_nope_head_dim``) must be 128 or 192.  Extra restrictions vs the
    Triton kernel (all satisfied by the production layouts): 16-byte aligned
    q_nope/w_kc base pointers, q_nope/w_kc strides that are multiples of 8
    elements, and even q_fp8_pad row/head strides.
    """
    num_tokens, _, k_dim = q_nope.shape
    n_dim = w_kc.shape[-1]
    rope_dim = q_rope.shape[-1]
    assert q_fp8_pad.dtype == torch.float8_e4m3fn
    assert q_nope.dtype == torch.bfloat16 and w_kc.dtype == torch.bfloat16
    assert q_rope.dtype == torch.bfloat16
    assert q_nope.is_cuda and w_kc.is_cuda and q_rope.is_cuda and q_fp8_pad.is_cuda
    assert q_nope.shape[1] == num_heads and q_rope.shape[1] == num_heads
    assert w_kc.shape[0] == num_heads and w_kc.shape[1] == k_dim
    assert q_fp8_pad.shape[0] >= num_tokens and q_fp8_pad.shape[1] >= num_heads
    assert q_fp8_pad.shape[2] == n_dim + rope_dim
    assert k_dim in (128, 192), "CUDA q-prep supports K in {128, 192}"
    assert n_dim == N_LORA and rope_dim == ROPE_DIM
    # Innermost-contiguous requirements (same as the Triton kernel).
    assert q_nope.stride(2) == 1 and q_rope.stride(2) == 1
    assert q_fp8_pad.stride(2) == 1
    # CUDA-kernel-specific layout requirements (production layouts satisfy
    # all of these; the Triton kernel stays the general-strides fallback).
    assert w_kc.stride(1) == 1, "w_kc must have K contiguous (N-major layout)"
    assert q_nope.data_ptr() % 16 == 0 and w_kc.data_ptr() % 16 == 0
    assert q_nope.stride(0) % 8 == 0 and q_nope.stride(1) % 8 == 0
    assert w_kc.stride(0) % 8 == 0 and w_kc.stride(2) % 8 == 0
    assert q_fp8_pad.stride(0) % 2 == 0 and q_fp8_pad.stride(1) % 2 == 0

    rope_vec16 = (
        q_rope.data_ptr() % 16 == 0
        and q_rope.stride(0) % 8 == 0
        and q_rope.stride(1) % 8 == 0
    )
    out_vec16 = (
        q_fp8_pad.data_ptr() % 16 == 0
        and q_fp8_pad.stride(0) % 16 == 0
        and q_fp8_pad.stride(1) % 16 == 0
    )

    module = _jit_qprep_bf16_fp8_module()
    module.dispatch(
        q_nope,
        w_kc,
        q_rope,
        q_fp8_pad,
        num_tokens,
        num_heads,
        k_dim,
        q_nope.stride(0),
        q_nope.stride(1),
        w_kc.stride(0),
        w_kc.stride(2),
        q_rope.stride(0),
        q_rope.stride(1),
        q_fp8_pad.stride(0),
        q_fp8_pad.stride(1),
        int(rope_vec16),
        int(out_vec16),
        _get_current_stream_raw(q_nope.device.index),
    )
