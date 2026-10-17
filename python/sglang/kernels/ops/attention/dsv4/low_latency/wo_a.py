"""Low-latency MXFP8 wo_a for DSv4.1 decode on SM100, T <= 64: the GEMM of

    deep_gemm.fp8_einsum("bhr,hdr->bhd", (o_fp8, o_s), (weight, weight_scale), out, recipe=(1, 1, 32))

on the mega-fp8 attention output and the same weight tensors, with its bf16 result act-quantized into the MXFP8 input
``wo_b_fp8`` takes. See ``deepseek_v4/low_latency/wo_a_fp8.cuh`` for the design. Meant to run as the PDL secondary of
the attention: the weight streams into smem before griddepcontrol.wait, and it releases its own dependents at start.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional, Tuple

import torch

from sglang.kernels.jit.utils import cache_once, load_jit

if TYPE_CHECKING:
    from tvm_ffi.module import Module

RANK = 1024
GROUP_K = 4096
MAX_TOKENS = 64


@cache_once
def _n_tile(num_tokens: int) -> int:
    for n in (8, 16, 32, 64):
        if num_tokens <= n:
            return n
    raise ValueError(f"wo_a_fp8 supports up to {MAX_TOKENS} tokens, got {num_tokens}")


@cache_once
def _jit_module(n: int) -> Module:
    return load_jit(
        f"wo_a_fp8_mxf8_ll_n{n}",
        cuda_files=["deepseek_v4/low_latency/wo_a_fp8.cuh"],
        cuda_wrappers=[("run", f"wo_a_fp8_mxf8_ll<{n}>")],
        extra_cuda_cflags=["-O3"],
    )


def wo_a_fp8(
    o_fp8: torch.Tensor,
    o_s: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    out: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """``o_fp8`` [T, G, 4096] e4m3 and ``o_s`` [T, G, 32] int32 (mega-fp8) times ``weight`` [G, 1024, 4096] e4m3 with
    ``weight_scale`` [G, 1024, 32] int32 (``quantize_wo_a``'s output). Returns (x [T, G * 1024] e4m3, x_scale), x_scale
    being the UE8M0 per-32 scales of x in FlashInfer's 128x4 swizzled layout (128 * G * 32 bytes)."""
    num_tokens, groups = o_fp8.shape[:2]
    if out is None:
        out = (
            torch.empty(
                num_tokens,
                groups * RANK,
                dtype=torch.float8_e4m3fn,
                device=o_fp8.device,
            ),
            torch.empty(
                128 * groups * RANK // 32, dtype=torch.uint8, device=o_fp8.device
            ),
        )
    module = _jit_module(_n_tile(num_tokens))
    module.run(o_fp8, o_s, weight, weight_scale, out[0], out[1])
    return out
