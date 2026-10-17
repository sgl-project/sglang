"""Low-latency FP8 wo_b GEMM for DeepSeek-V4.1-Flash TP4 decode (SM100), alone or fused with the TP all-reduce, the
mHC post-mix, the pre-collapse and the RMSNorm over the CustomAllReduceV2 push plane. See
``deepseek_v4/low_latency/{wo_b_fp8,all_reduce_mhc}.cuh`` for the design."""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import torch

from sglang.kernels.jit.utils import cache_once, load_jit

if TYPE_CHECKING:
    from tvm_ffi.module import Module

    from sglang.kernels.ops.communication.all_reduce import Communicator

N_OUT = 5120
K_IN = 2048
MAX_TOKENS = 64


@cache_once
def _n_tile(num_tokens: int) -> int:
    """MMA N: the smallest of 8 / 16 / 32 / 64 that holds every token."""
    for n in (8, 16, 32, 64):
        if num_tokens <= n:
            return n
    raise ValueError(f"wo_b_fp8 supports up to {MAX_TOKENS} tokens, got {num_tokens}")


@cache_once
def _jit_gemm_module(n: int, world_size: int, multicast: bool) -> Module:
    mc = "true" if multicast else "false"
    return load_jit(
        f"dsv41_wo_b_ll_n{n}_w{world_size}_mc{int(multicast)}",
        cuda_files=["deepseek_v4/low_latency/wo_b_fp8.cuh"],
        cuda_wrappers=[("run", f"dsv41_wo_b_ll<{n}, {world_size}, {mc}>")],
        extra_cuda_cflags=["-O3"],
    )


@cache_once
def _jit_comm_module(world_size: int) -> Module:
    return load_jit(
        f"dsv41_all_reduce_mhc_ll_w{world_size}",
        cuda_files=["deepseek_v4/low_latency/all_reduce_mhc.cuh"],
        cuda_wrappers=[("run", f"dsv41_all_reduce_mhc_ll<{world_size}, true>")],
        extra_cuda_cflags=["-O3"],
    )


def prepare_weight(weight: torch.Tensor, block_scale: torch.Tensor):
    """``(weight, weight_scale)`` as sglang's FlashInfer cute-dsl wo_b keeps them: the plain [5120, 2048] e4m3
    weight, and its [160, 64] UE8M0 32x32 block scales expanded per row and interleaved by FlashInfer's
    ``block_scale_interleave`` (the layer's ``weight_scale_inv_swizzled``). For tests; the kernel reads exactly
    these tensors."""
    from flashinfer import block_scale_interleave

    assert weight.shape == (N_OUT, K_IN) and weight.dtype == torch.float8_e4m3fn
    assert block_scale.shape == (N_OUT // 32, K_IN // 32)
    per_row = block_scale.contiguous().view(torch.uint8).repeat_interleave(32, dim=0)
    return weight, block_scale_interleave(per_row.contiguous()).contiguous().view(-1)


def wo_b_fp8(
    x: torch.Tensor,
    x_scale: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    *,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """``x @ W^T`` in bf16, T <= 64: ``x`` [T, 2048] e4m3 with ``x_scale``, its UE8M0 per-32 scales in FlashInfer's
    128x4 swizzled layout (``wo_a_fp8``'s output); ``weight`` [5120, 2048] e4m3 and ``weight_scale``, the 128x4 image
    of its per-row-per-32 scales (the layer's ``weight_scale_inv_swizzled``)."""
    num_tokens = x.shape[0]
    n_tile = _n_tile(num_tokens)
    if out is None:
        out = x.new_empty((num_tokens, N_OUT), dtype=torch.bfloat16)
    module = _jit_gemm_module(n_tile, 0, False)
    module.run(x, x_scale.view(-1), weight, weight_scale.view(-1), out, None)
    return out


def wo_b_fp8_all_reduce_mhc(
    x: torch.Tensor,
    x_scale: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    comm: Communicator,
    residual: torch.Tensor,
    post: torch.Tensor,
    comb: torch.Tensor,
    pre: torch.Tensor,
    norm_weight: torch.Tensor,
    eps: float,
    *,
    multicast: bool = True,
    mhc_out: Optional[torch.Tensor] = None,
    normalized: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """``wo_b_fp8`` summed over the TP ranks of ``comm`` (world size 2 / 4), then the HC=4 post-mix with ``residual``
    [T, 4, 5120] (``post`` [T, 4], ``comb`` [T, 4, 4]), the ``pre`` [T, 4] collapse and RMSNorm(``norm_weight``,
    ``eps``). Returns (``mhc_out`` [T, 4, 5120], ``normalized`` [T, 5120]), both bf16, the arithmetic of
    ``all_reduce_mhc_post_combine_norm`` (``normalized`` may differ by a bf16 ulp). ``multicast`` pushes through the
    plane's multicast address instead of one store per rank.

    Every rank calls with the same T on the same communicator. Both kernels read the plane's phase counters, and the
    second also ``residual`` / ``post`` / ``comb`` / ``pre`` / ``norm_weight``, before their PDL wait: a kernel that
    waits before it triggers (the attention) must run between two calls, after the last write of those inputs.
    Back-to-back calls without one hang."""
    num_tokens = x.shape[0]
    n_tile = _n_tile(num_tokens)
    module = _jit_gemm_module(n_tile, comm.world_size, multicast)
    module.run(x, x_scale.view(-1), weight, weight_scale.view(-1), None, comm)
    if mhc_out is None:
        mhc_out = torch.empty_like(residual)
    if normalized is None:
        normalized = residual.new_empty((num_tokens, N_OUT))
    module = _jit_comm_module(comm.world_size)
    module.run(comm, residual, post, comb, pre, norm_weight, mhc_out, normalized, eps)
    return mhc_out, normalized
