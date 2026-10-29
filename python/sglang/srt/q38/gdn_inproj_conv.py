# SPDX-License-Identifier: Apache-2.0
"""M4 FN BF16 in-projection + native width-four verify convolution.

Only target verify B1/T4, no tree and no accept-offset. Caller retains the
native path otherwise. This module owns no weights or state copies.
"""
from __future__ import annotations

import os

import cuda.bindings.driver as cuda
import cutlass
import torch
from cutlass.cute import experimental as cute_ext
from cutlass.cute.runtime import from_dlpack
from .gdn_inproj_conv_core import TgvGemmCuteExtKernel

_CACHE = {}
_FRONT_STAGES = 12 if os.environ.get("SGLANG_Q38C_GDN_FRONT12", "0") == "1" else 8


def _view(t, align=2):
    return from_dlpack(t.detach(), assumed_align=align)


def run(x, packed_weight, conv_weight, conv_bias, conv_state, state_indices,
        window, intermediate_indices, pad_slot=-1):
    """Return packed BF16[4,16480] and convolved BF16[4,10240].

    State/window are mutated exactly at their supplied strides. Padded sequence
    (-1 by default) leaves conv output/state/windows undefined/unmodified, as
    the native conv; z/b/a projection still executes. All 16480 packed output
    columns currently contain the raw BF16 projection, including q/k/v.
    PDL consumers must issue their existing grid-dependency wait.
    """
    if x.shape != (4, 2560) or packed_weight.shape != (16480, 2560):
        raise ValueError('inproj-conv supports only FN B1/T4, H2560, N16480')
    if x.dtype != torch.bfloat16 or packed_weight.dtype != torch.bfloat16:
        raise ValueError('native BF16 projection is required')
    if conv_weight.shape != (10240, 4) or conv_state.shape[1] != 10240:
        raise ValueError('native 10240-channel width-four convolution required')
    if conv_state.shape[2] < 3 or window.shape[1:4] != (4, 10240, 3):
        raise ValueError('native three-history/four-step window required')
    if any(t.dtype != torch.bfloat16 for t in (conv_weight, conv_state, window)):
        raise ValueError('native BF16 convolution operands required')
    if state_indices.numel() < 1 or intermediate_indices.numel() < 1:
        raise ValueError('one request state/window index required')
    packed = torch.empty((4, 16480), device=x.device, dtype=x.dtype)
    conv_out = torch.empty((4, 10240), device=x.device, dtype=x.dtype)
    # GEMM kernel-M = output channel; kernel-N = token. No repack/copy.
    a = packed_weight.unsqueeze(-1)
    b = x.unsqueeze(-1)
    c = packed.t().unsqueeze(-1)
    bias = conv_weight[:, 0] if conv_bias is None else conv_bias
    tensors = (a, b, c, c, conv_weight, bias, conv_state,
               state_indices, window, intermediate_indices, conv_out)
    key = (x.device.index, conv_bias is not None, int(pad_slot),
           tuple((tuple(t.shape), tuple(t.stride()), t.dtype) for t in tensors))
    args = tuple(_view(t, 16 if i in (0, 1, 2, 3) else 2)
                 for i, t in enumerate(tensors))
    stream = cuda.CUstream(torch.cuda.current_stream(x.device).cuda_stream)
    compiled = _CACHE.get(key)
    if compiled is None:
        op = TgvGemmCuteExtKernel(
            cta_m=64, cta_n=8, cta_k=128, num_ab_stage=_FRONT_STAGES,
            use_2cta=False, use_pdl=True, has_bias=False,
            out_dtype=cutlass.BFloat16)
        op.conv_has_bias = conv_bias is not None
        op.pad_slot = int(pad_slot)
        compiled = cute_ext.compile(op, *args, stream)
        _CACHE[key] = compiled
    compiled(*args, stream)
    return packed, conv_out
