# SPDX-License-Identifier: Apache-2.0
"""FN TP4/TP1 BF16 projection and T4 verify convolution for B1/B2/B3.

Only target verify B1/B2/B3, T4, no tree and no accept-offset. Caller retains the
native path otherwise. This module owns no weights or state copies.
"""
from __future__ import annotations

import os

import cuda.bindings.driver as cuda
import cutlass
import torch
from cutlass.cute import experimental as cute_ext
from cutlass.cute.runtime import from_dlpack
from .q38_mkgdn_front_core import MkGdnFront

_CACHE = {}


def _view(t, align=2):
    return from_dlpack(t.detach(), assumed_align=align)


def run(x, packed_weight, conv_weight, conv_bias, conv_state, state_indices,
        window, intermediate_indices, pad_slot=-1):
    """Return packed BF16[M,Npacked] and convolved BF16[M,Nqkv].

    State/window are mutated exactly at their supplied strides. Padded sequence
    (-1 by default) leaves conv output/state/windows undefined/unmodified, as
    the native conv; z/b/a projection still executes. All packed output
    columns currently contain the raw BF16 projection, including q/k/v.
    PDL consumers must issue their existing grid-dependency wait.
    """
    rows, width = x.shape
    channels = conv_weight.shape[0]
    if ((rows not in (4, 8, 12) and not (
            rows == 16 and channels == 2560
            and os.environ.get("SGLANG_Q38C_MK_GDN_WIDE", "0") == "1"))
            or width != 2560 or channels not in (2560, 10240)):
        raise ValueError('MK_GDN front requires M4/M8/M12; TP4 M16 needs MK_GDN_WIDE')
    hv = channels // 128 * 3 // 5
    packed_width = channels + hv * 128 + 2 * hv
    if packed_weight.shape != (packed_width, 2560):
        raise ValueError('incorrect packed projection shape')
    if x.dtype != torch.bfloat16 or packed_weight.dtype != torch.bfloat16:
        raise ValueError('native BF16 projection is required')
    if conv_weight.shape != (channels, 4) or conv_state.shape[1] != channels:
        raise ValueError('native width-four convolution required')
    if conv_state.shape[2] < 3 or window.shape[1:4] != (4, channels, 3):
        raise ValueError('native three-history/four-step window required')
    packed = torch.empty((rows, packed_width), device=x.device, dtype=x.dtype)
    conv_out = torch.empty((rows, channels), device=x.device, dtype=x.dtype)
    # GEMM kernel-M = output channel; kernel-N = token. No repack/copy.
    a = packed_weight.unsqueeze(-1)
    b = x.unsqueeze(-1)
    c = packed.t().unsqueeze(-1)
    bias = conv_weight[:, 0] if conv_bias is None else conv_bias
    tensors = (a, b, c, c, conv_weight, bias, conv_state,
               state_indices, window, intermediate_indices, conv_out)
    stages = (int(os.environ.get("Q38_GDNPROJ_FRONT_STAGES", "6"))
              if rows == 4 and channels == 2560 else 6)
    key = (stages, x.device.index, conv_bias is not None, int(pad_slot),
           tuple((tuple(t.shape), tuple(t.stride()), t.dtype) for t in tensors))
    args = tuple(_view(t, 16 if i in (0, 1, 2, 3) else 2)
                 for i, t in enumerate(tensors))
    stream = cuda.CUstream(torch.cuda.current_stream(x.device).cuda_stream)
    compiled = _CACHE.get(key)
    if compiled is None:
        import logging
        logging.getLogger(__name__).info("STACK8_FRONT_COMPILE stages=%d rows=%d channels=%d arch=%s", stages, rows, channels, torch.cuda.get_device_capability(x.device))
        op = MkGdnFront(
            cta_m=64, cta_n=8 if rows == 4 else 16, cta_k=128, num_ab_stage=stages,
            use_2cta=False, use_pdl=True, has_bias=False,
            out_dtype=cutlass.BFloat16)
        op.qkv_channels = channels
        op.verify_rows = rows
        op.conv_has_bias = conv_bias is not None
        op.pad_slot = int(pad_slot)
        compiled = cute_ext.compile(op, *args, stream)
        _CACHE[key] = compiled
    compiled(*args, stream)
    return packed, conv_out
