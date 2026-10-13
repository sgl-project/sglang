# SPDX-License-Identifier: Apache-2.0
"""FP8 attention on SGLang's [S, H, 128] strided BF16 views.

fp8_attention() quantizes Q/K/V per head with the fused Triton passes, runs the CuTe-DSL
kernel and returns (output [S, H, 128] BF16, lse [H, S] FP32). The compiled kernel is
cached per (sequence, heads, device); the E4M3, scale, output and LSE buffers come from
the caching allocator on every call and belong to the caller, nothing is retained
between calls.
"""

import logging
import math

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import msgspec
import torch
from cutlass.cute.runtime import from_dlpack

from sglang.kernels.ops.attention.fp8_fa_sm120.fused_prep import fused_prepare
from sglang.kernels.ops.attention.fp8_fa_sm120.kernel import fp8_attention_host

logger = logging.getLogger(__name__)

HEAD_DIM = 128
# Q and K/V are padded separately; the kernel compiles its key mask only when S % 32 != 0.
QUERY_TILE = 128
KEY_TILE = 32

# Compiled kernels only, KBs of host state each and about 10 s to build; the buffers
# they run on are allocated per call.
_KERNEL_CACHE: dict[tuple, object] = {}


def validate_inputs(q, k, v, softmax_scale):
    """Strided [S, H, 128] BF16 views on one CUDA device; returns the scale as float."""
    if (
        q.ndim != 3
        or q.shape[-1] != HEAD_DIM
        or q.shape != k.shape
        or q.shape != v.shape
    ):
        raise ValueError("Q/K/V must have equal shape [sequence, heads, 128]")
    if min(q.shape[:2]) < 1:
        raise ValueError("Sequence and heads must be positive")
    for tensor in (q, k, v):
        if (
            not tensor.is_cuda
            or tensor.device != q.device
            or tensor.dtype != torch.bfloat16
        ):
            raise ValueError("Q/K/V must share a CUDA device and use BF16")
        if tensor.stride(-1) != 1 or tensor.stride(-2) != HEAD_DIM:
            raise ValueError(
                "Feature and head dimensions must be contiguous within each token"
            )
        if tensor.data_ptr() % 16 or tensor.stride(0) % 8:
            raise ValueError("Token rows must be aligned to 16 bytes")
        if tensor.stride() != q.stride():
            raise ValueError("Q/K/V must share strides")
    scale = HEAD_DIM**-0.5 if softmax_scale is None else float(softmax_scale)
    if not math.isfinite(scale):
        raise ValueError("Softmax scale must be finite")
    return scale


def kernel_key(q):
    """Cache key: the kernel bakes the buffer shapes, which follow from (S, H)."""
    return (q.shape[0], q.shape[1], q.device.index)


class Workspace(msgspec.Struct, frozen=True, kw_only=True):
    """Kernel-side buffers for one call; the kernel works in [H, S, 128]."""

    q_fp8: torch.Tensor
    k_fp8: torch.Tensor
    v_fp8: torch.Tensor
    scales: torch.Tensor
    maximums: torch.Tensor
    output: torch.Tensor
    lse: torch.Tensor


def _allocate_workspace(q):
    sequence = q.shape[0]
    heads = q.shape[1]
    padded_queries = (sequence + QUERY_TILE - 1) // QUERY_TILE * QUERY_TILE
    padded_keys = (sequence + KEY_TILE - 1) // KEY_TILE * KEY_TILE
    device = q.device
    return Workspace(
        q_fp8=torch.empty(
            (heads, padded_queries, HEAD_DIM), device=device, dtype=torch.float8_e4m3fn
        ),
        k_fp8=torch.empty(
            (heads, padded_keys, HEAD_DIM), device=device, dtype=torch.float8_e4m3fn
        ),
        v_fp8=torch.empty(
            (heads, HEAD_DIM, padded_keys), device=device, dtype=torch.float8_e4m3fn
        ),
        scales=torch.empty((3, heads), device=device, dtype=torch.float32),
        maximums=torch.zeros((3, heads), device=device, dtype=torch.float32),
        output=torch.empty(q.shape, device=device, dtype=torch.bfloat16),
        lse=torch.empty((heads, sequence), device=device, dtype=torch.float32),
    )


def _kernel_views(workspace):
    # Byte DLPack views work across Torch versions without FP8 DLPack support.
    mQ = from_dlpack(workspace.q_fp8.view(torch.uint8), assumed_align=16)
    mK = from_dlpack(workspace.k_fp8.view(torch.uint8), assumed_align=16)
    mV = from_dlpack(workspace.v_fp8.view(torch.uint8), assumed_align=16)
    mQ.element_type = cutlass.Float8E4M3FN
    mK.element_type = cutlass.Float8E4M3FN
    mV.element_type = cutlass.Float8E4M3FN
    mO = from_dlpack(workspace.output.permute(1, 0, 2), assumed_align=16)
    mLSE = from_dlpack(workspace.lse, assumed_align=16)
    mScales = from_dlpack(workspace.scales, assumed_align=16)
    return (mQ, mK, mV, mO, mLSE, mScales)


def fp8_attention(q, k, v, softmax_scale=None):
    """Attention over strided [S, H, 128] BF16 views; returns (output [S, H, 128], lse [H, S])."""
    scale = validate_inputs(q, k, v, softmax_scale)
    kernel_scale = cutlass.Float32(scale)
    key = kernel_key(q)

    with torch.cuda.device(q.device):
        workspace = _allocate_workspace(q)
        views = _kernel_views(workspace)
        stream = cuda.CUstream(torch.cuda.current_stream(q.device).cuda_stream)

        compiled = _KERNEL_CACHE.get(key)
        if compiled is None:
            logger.info(
                "fp8_fa_sm120 attention: compiling for S=%d H=%d (once per shape)",
                q.shape[0],
                q.shape[1],
            )
            compiled = cute.compile(fp8_attention_host, *views, kernel_scale, stream)
            _KERNEL_CACHE[key] = compiled

        fused_prepare(
            q=q.permute(1, 0, 2),
            k=k.permute(1, 0, 2),
            v=v.permute(1, 0, 2),
            workspace=workspace,
        )
        compiled(*views, kernel_scale, stream)
    return workspace.output, workspace.lse
