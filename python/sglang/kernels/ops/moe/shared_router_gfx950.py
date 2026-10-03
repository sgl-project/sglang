# SPDX-License-Identifier: Apache-2.0
"""Two-launch shared MLP/router front for DSV4.1 Flash TP4, M=6/12.

No expert sorting, routed GEMM, collective, or custom AITER API is included.
Outputs use the ordinary upstream top-k contract. Independent CTA branches
only communicate across the launch boundary, never via spin barriers.
"""

import torch
import triton
import triton.language as tl

from sglang.kernels.jit.utils import cache_once, load_jit
from sglang.kernels.ops.moe.rocm_router_gate import _gate_row
from sglang.kernels.ops.quantization.mxfp8_amd_gfx95 import fp8_grid_round


@cache_once
def _projection_module():
    return load_jit(
        "shared_router_gfx950",
        cuda_files=["deepseek_v4/shared_router_gfx950.cuh"],
        cuda_wrappers=[("run", "SharedRouterGfx950Kernel::run")],
    )


@triton.jit
def _shared_middle(G, rows, k, M: tl.constexpr, BK: tl.constexpr):
    mask = (rows[:, None] < M) & (k[None, :] < 576)
    gate = tl.load(G + rows[:, None] * 1152 + k[None, :], mask, 0).to(tl.float32)
    up = tl.load(G + rows[:, None] * 1152 + 576 + k[None, :], mask, 0).to(tl.float32)
    gate = tl.minimum(gate, 10.0)
    up = tl.minimum(tl.maximum(up, -10.0), 10.0)
    # Preserve the native BF16 activation boundary before per-32 MXFP8 rounding.
    y = (gate / (1.0 + tl.exp(-gate)) * up).to(tl.bfloat16).to(tl.float32)
    return tl.reshape(fp8_grid_round(tl.reshape(y, (16 * BK // 32, 32))), (16, BK)).to(
        tl.bfloat16
    )


@triton.jit
def _shared_down_router_topk(
    G, W, O, Part, Bias, Logits, Weights, Ids, M: tl.constexpr
):
    pid = tl.program_id(0)
    DOWN_CTAS: tl.constexpr = 80 * tl.cdiv(M, 4)
    if pid < DOWN_CTAS:
        lane = tl.arange(0, 16)
        rows = (pid // 80) * 4 + lane % 4
        n = (pid % 80) * 64 + tl.arange(0, 64)
        k = tl.arange(0, 512)
        a = _shared_middle(G, rows, k, M, 512)
        b = tl.load(W + n[None, :] * 576 + k[:, None])
        acc = tl.dot(a, b)
        tail = 512 + tl.arange(0, 64)
        at = _shared_middle(G, rows, tail, M, 64)
        bt = tl.load(W + n[None, :] * 576 + tail[:, None])
        acc = tl.dot(at, bt, acc)
        tl.store(
            O + rows[:, None] * 5120 + n[None, :],
            acc,
            (lane[:, None] < 4) & (rows[:, None] < M),
        )
    else:
        row = pid - DOWN_CTAS
        # Reuse upstream's AITER-equivalent score, tie ordering, fixed-order
        # split-K sum and renormalization. Also materialize valid router_logits.
        weights, ids = _gate_row(
            Logits, Part, Bias, row, 384, M * 384, 384, 1.5, 10, True, True, True, 6
        )
        gate_lane = tl.arange(0, 64)
        tl.store(Weights + row * 6 + gate_lane, weights, gate_lane < 6)
        tl.store(Ids + row * 6 + gate_lane, ids, gate_lane < 6)


def project(x, shared_weight, shared_scale, router_weight):
    """Stage one; native shared gate/up plus BF16 router split-K projection."""
    m = x.shape[0]
    assert x.shape == (m, 5120) and m in (6, 12)
    assert shared_weight.shape == (72, 40, 2048)
    assert shared_scale.shape == (36, 160) and router_weight.shape == (384, 5120)
    assert x.dtype == router_weight.dtype == torch.bfloat16
    assert shared_weight.dtype == shared_scale.dtype == torch.uint8
    assert all(
        t.is_cuda and t.device == x.device and t.is_contiguous()
        for t in (x, shared_weight, shared_scale, router_weight)
    )
    gate = torch.empty((m, 1152), dtype=x.dtype, device=x.device)
    partials = torch.empty((10, m, 384), dtype=torch.float32, device=x.device)
    _projection_module().run(
        shared_weight, shared_scale, x, gate, router_weight, partials
    )
    return gate, partials


def finish(gate, partials, down_weight, bias):
    """Stage two; shared middle/down plus native-compatible top-k."""
    m = gate.shape[0]
    assert m in (6, 12) and gate.shape == (m, 1152)
    assert partials.shape == (10, m, 384) and partials.dtype == torch.float32
    assert down_weight.shape == (5120, 576) and bias.shape == (384,)
    assert gate.dtype == down_weight.dtype == bias.dtype == torch.bfloat16
    assert all(
        t.is_cuda and t.device == gate.device and t.is_contiguous()
        for t in (gate, partials, down_weight, bias)
    )
    output = torch.empty((m, 5120), dtype=gate.dtype, device=gate.device)
    logits = torch.empty((m, 384), dtype=torch.float32, device=gate.device)
    weights = torch.empty((m, 6), dtype=torch.float32, device=gate.device)
    ids = torch.empty((m, 6), dtype=torch.int32, device=gate.device)
    _shared_down_router_topk[(80 * triton.cdiv(m, 4) + m,)](
        gate,
        down_weight,
        output,
        partials,
        bias,
        logits,
        weights,
        ids,
        m,
        num_warps=4,
        num_stages=1,
        matrix_instr_nonkdim=16,
        enable_fp_fusion=False,
    )
    return output, weights, ids, logits


def shared_router(x, shared_weight, shared_scale, router_weight, down_weight, bias):
    gate, partials = project(x, shared_weight, shared_scale, router_weight)
    return finish(gate, partials, down_weight, bias)


def unpack_shared_down(weight, scale):
    """One-time exact BF16 view of native lane-ordered FP8 weights.

    An FP8 value times a power-of-two scale is exactly representable in BF16.
    The padding to K640 is discarded; the checkpoint's logical K is576.
    The caller owns this immutable derived buffer, not a process-global cache.
    """
    assert weight.shape == (320, 5, 2048) and scale.shape == (160, 20)
    assert scale.dtype == torch.uint8 and weight.dtype == torch.float8_e4m3fn
    codes = weight.view(torch.uint8).view(320, 5, 2, 2, 16, 2, 16)
    codes = codes.permute(0, 4, 1, 5, 2, 3, 6).contiguous().view(5120, 640)
    power = (scale.to(torch.int32) << 23).view(torch.float32)
    power = power.repeat_interleave(32, 0).repeat_interleave(32, 1)
    return (
        (codes.view(torch.float8_e4m3fn).float() * power)[:, :576]
        .to(torch.bfloat16)
        .contiguous()
    )
