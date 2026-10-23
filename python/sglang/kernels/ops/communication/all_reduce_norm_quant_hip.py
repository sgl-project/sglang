"""MiniMax-M3.1's TP4 decode layer boundary in one kernel on AITER's custom all-reduce,
for the smallest batches: all-reduce + residual add + Gemma RMSNorm + the per-token fp8
quant of gemma_add_rmsnorm_fp8."""

import torch
from aiter.dist.device_communicators.custom_all_reduce import CustomAllreduce
from aiter.jit.core import AITER_CSRC_DIR

from sglang.kernels.jit.utils import cache_once, load_jit

HIDDEN_SIZE = 6144
WORLD_SIZE = 4


@cache_once
def _module():
    return load_jit(
        "all_reduce_norm_quant_hip",
        cuda_files=["distributed/all_reduce_norm_quant_hip.cuh"],
        cuda_wrappers=[("run", "all_reduce_norm_quant_hip::AllReduceNormQuant::run")],
        # AITER's CustomAllreduce, whose peer buffers and signals the kernel runs on
        extra_include_paths=[f"{AITER_CSRC_DIR}/include"],
    )


def all_reduce_gemma_norm_fp8(
    input: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    communicator: CustomAllreduce,
    keep_bf16: bool = False,
):
    """``gemma_add_rmsnorm_fp8(all_reduce(input), residual, weight, eps, keep_bf16=...)``
    in one kernel, returned the same way: ``((fp8, scale[, bf16 norm]), residual_out)``.

    ``input`` and ``residual`` are bf16 [m, 6144]; ``weight`` is GemmaRMSNorm's (applied
    as 1 + weight). Every rank reads all ranks' rows, so it suits a few rows only."""
    assert communicator.world_size == WORLD_SIZE and not communicator.disabled
    m = input.shape[0]
    q = torch.empty((m, HIDDEN_SIZE), dtype=torch.float8_e4m3fn, device=input.device)
    scale = torch.empty((m, 1), dtype=torch.float32, device=input.device)
    residual_out = torch.empty_like(residual)
    norm = torch.empty_like(residual) if keep_bf16 else residual_out
    capturing = torch.cuda.is_current_stream_capturing()
    # capture() registers the peer addresses when the enclosing graph scope exits.
    assert not capturing or (
        communicator._IS_CAPTURING and communicator.enable_register_for_capturing
    )
    pool = communicator._pool["input"]
    _module().run(
        communicator._ptr,
        input,
        residual,
        weight,
        residual_out,
        q.view(torch.uint8),
        scale,
        norm,
        eps,
        keep_bf16,
        0 if capturing else pool.data_ptr,
        0 if capturing else pool.max_size,
    )
    quantized = (q, scale, norm) if keep_bf16 else (q, scale)
    return quantized, residual_out
