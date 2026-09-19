# SPDX-License-Identifier: Apache-2.0
"""Reusable FP8 attention plan on SGLang's [S, H, 128] strided BF16 views.

One plan owns one compiled kernel and its E4M3, scale, output and LSE buffers for a
fixed (sequence, heads, input strides, softmax scale, device). bind_inputs() points
the plan at new Q/K/V storage with the same shape and strides, so every DiT layer
reuses one compilation. prepare() quantizes with the fused Triton passes,
launch_prepared() runs the kernel. The BF16 output buffer [S, H, 128] belongs to
the plan and is overwritten by the next launch.
"""

import math

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import torch
from cutlass.cute.runtime import from_dlpack

from sglang.kernels.ops.attention.fp8_fa_sm120.fused_prep import (
    _attach_fused_state,
    fused_prepare,
)
from sglang.kernels.ops.attention.fp8_fa_sm120.kernel import (
    fp8_attention_host,
    key_order_for_positions,
)

HEAD_DIM = 128
# Q and K/V are padded separately; the kernel compiles its key mask only when S % 32 != 0.
QUERY_TILE = 128
KEY_TILE = 32


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


def plan_key(q, softmax_scale):
    """Cache key: everything the compiled kernel and the prep constants depend on."""
    return (
        q.shape[0],
        q.shape[1],
        tuple(q.stride()),
        q.device.index,
        float(softmax_scale),
    )


class FP8AttentionPlan:
    """Compiled kernel plus buffers for fixed strided [S, H, 128] BF16 inputs."""

    def __init__(self, q, k, v, softmax_scale=None, block_tokens=64):
        scale = validate_inputs(q, k, v, softmax_scale)
        self.sequence = q.shape[0]
        self.heads = q.shape[1]
        self.softmax_scale = scale
        self.kernel_scale = cutlass.Float32(scale)
        self.padded_queries = (
            (self.sequence + QUERY_TILE - 1) // QUERY_TILE * QUERY_TILE
        )
        self.padded_keys = (self.sequence + KEY_TILE - 1) // KEY_TILE * KEY_TILE
        self.device = q.device
        self.input_strides = tuple(q.stride())
        # The kernel side works in [H, S, 128]; these are views, not copies.
        self.inputs = (q.permute(1, 0, 2), k.permute(1, 0, 2), v.permute(1, 0, 2))

        with torch.cuda.device(q.device):
            self.q_fp8 = torch.zeros(
                (self.heads, self.padded_queries, HEAD_DIM),
                device=q.device,
                dtype=torch.float8_e4m3fn,
            )
            self.k_fp8 = torch.zeros(
                (self.heads, self.padded_keys, HEAD_DIM),
                device=q.device,
                dtype=torch.float8_e4m3fn,
            )
            self.v_fp8 = torch.zeros(
                (self.heads, HEAD_DIM, self.padded_keys),
                device=q.device,
                dtype=torch.float8_e4m3fn,
            )
            self.scales = torch.empty(
                (3, self.heads), device=q.device, dtype=torch.float32
            )
            # V^T is stored in the key order the kernel's register P pack expects.
            positions = torch.arange(self.padded_keys, device=q.device)
            self.key_order = key_order_for_positions(positions)
            self.output = torch.empty(q.shape, device=q.device, dtype=torch.bfloat16)
            self.lse = torch.empty(
                (self.heads, self.sequence), device=q.device, dtype=torch.float32
            )

            # Byte DLPack views work across Torch versions without FP8 DLPack support.
            mQ = from_dlpack(self.q_fp8.view(torch.uint8), assumed_align=16)
            mK = from_dlpack(self.k_fp8.view(torch.uint8), assumed_align=16)
            mV = from_dlpack(self.v_fp8.view(torch.uint8), assumed_align=16)
            mQ.element_type = cutlass.Float8E4M3FN
            mK.element_type = cutlass.Float8E4M3FN
            mV.element_type = cutlass.Float8E4M3FN
            mO = from_dlpack(self.output.permute(1, 0, 2), assumed_align=16)
            mLSE = from_dlpack(self.lse, assumed_align=16)
            mScales = from_dlpack(self.scales, assumed_align=16)
            self.tensor_views = (mQ, mK, mV, mO, mLSE, mScales)

            stream = cuda.CUstream(torch.cuda.current_stream(q.device).cuda_stream)
            self.compiled = cute.compile(
                fp8_attention_host,
                *self.tensor_views,
                self.kernel_scale,
                stream,
            )

        _attach_fused_state(self, block_tokens)
        self._prepared = False

    def bind_inputs(self, q, k, v):
        """Point the plan at new Q/K/V storage with the same shape and strides."""
        expected_shape = (self.sequence, self.heads, HEAD_DIM)
        for tensor in (q, k, v):
            if tuple(tensor.shape) != expected_shape:
                raise ValueError(
                    f"Plan expects shape {expected_shape}, got {tuple(tensor.shape)}"
                )
            if tuple(tensor.stride()) != self.input_strides:
                raise ValueError(
                    f"Plan expects strides {self.input_strides}, got {tuple(tensor.stride())}"
                )
            if tensor.device != self.device or tensor.dtype != torch.bfloat16:
                raise ValueError("Plan inputs must stay BF16 on the plan's device")
            if tensor.data_ptr() % 16:
                raise ValueError("Token rows must be aligned to 16 bytes")
        self.inputs = (q.permute(1, 0, 2), k.permute(1, 0, 2), v.permute(1, 0, 2))
        self._prepared = False

    def prepare(self):
        """Quantize the bound inputs: per-head amax, scales, packed Q/K and permuted V^T."""
        fused_prepare(self)

    def launch_prepared(self):
        """Run attention on the prepared buffers; returns (output [S, H, 128], lse [H, S])."""
        if not self._prepared:
            raise RuntimeError("Call prepare() before launch_prepared()")
        with torch.cuda.device(self.device):
            stream = cuda.CUstream(torch.cuda.current_stream(self.device).cuda_stream)
            self.compiled(*self.tensor_views, self.kernel_scale, stream)
        return self.output, self.lse

    def __call__(self):
        self.prepare()
        return self.launch_prepared()
