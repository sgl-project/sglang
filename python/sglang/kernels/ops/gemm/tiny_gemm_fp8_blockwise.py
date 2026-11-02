"""Small-M block-FP8 projections with fused UE8M0 activation quantization."""

from __future__ import annotations

import torch
import triton
import triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor

from sglang.srt.utils.common import get_device_sm

_MAX_M = 16

# (N, K): (weight rows per CTA, K splits, warps).
_CONFIGS = {
    (1536, 7168): (64, 14, 8),
    (2112, 7168): (64, 8, 8),
    (6144, 7168): (128, 8, 4),
    (7168, 1536): (128, 4, 8),
    (2304, 1536): (64, 6, 8),
}

# Split-K partials and arrival counters, one set per (device, stream): the
# counters are re-armed by the last CTA, so two launches must not share them
# unless they are stream-ordered.
_workspaces: dict[tuple[torch.device, int], tuple[torch.Tensor, torch.Tensor]] = {}
_descriptors: dict[tuple, TensorDescriptor] = {}


def can_use_tiny_gemm_fp8_blockwise(x, weight, weight_scale) -> bool:
    return (
        x.is_cuda
        and x.dtype == torch.bfloat16
        and x.ndim == 2
        and 1 <= x.shape[0] <= _MAX_M
        and x.is_contiguous()
        and weight.ndim == 2
        and tuple(weight.shape) in _CONFIGS
        and weight.shape[1] == x.shape[1]
        and weight.dtype == torch.float8_e4m3fn
        and weight.is_contiguous()
        and weight.device == x.device
        and weight_scale.device == x.device
        and weight_scale.dtype == torch.int32
        and weight_scale.shape == (weight.shape[0], triton.cdiv(weight.shape[1], 512))
        and get_device_sm() // 10 == 10
    )


@triton.jit
def _tiny_gemm_fp8_blockwise_kernel(
    X,
    W,
    S,
    Y,
    P,
    C,
    M,
    N: tl.constexpr,
    K: tl.constexpr,
    SN: tl.constexpr,
    SK: tl.constexpr,
    BW: tl.constexpr,
    SPLIT: tl.constexpr,
):
    # Swap-AB: weight rows fill the MMA M side, the tokens (padded to 16) the N side.
    pid_n = tl.program_id(0)
    pid_k = tl.program_id(1)
    GPS: tl.constexpr = K // 128 // SPLIT
    rows = pid_n * BW + tl.arange(0, BW)
    ms = tl.arange(0, 16)
    ks = tl.arange(0, 128)
    tl.extra.cuda.gdc_wait()
    acc = tl.zeros([BW, 16], tl.float32)
    for g in tl.range(pid_k * GPS, pid_k * GPS + GPS, num_stages=3):
        weight = W.load([pid_n * BW, g * 128])
        # A tile never straddles a 128-row scale block.
        packed_scale = tl.load(S + pid_n * BW * SN + (g // 4) * SK)
        weight_scale = (((packed_scale >> ((g % 4) * 8)) & 255) << 23).to(
            tl.float32, bitcast=True
        )
        x = tl.load(X + ms[:, None] * K + g * 128 + ks[None, :], ms[:, None] < M, 0).to(
            tl.float32
        )
        amax = tl.maximum(tl.max(tl.abs(x), 1), 1.0e-10)
        exponent = ((amax * (1.0 / 448.0)).to(tl.int32, bitcast=True) + 0x7FFFFF) >> 23
        x_scale = (exponent << 23).to(tl.float32, bitcast=True)
        inverse = ((254 - exponent) << 23).to(tl.float32, bitcast=True)
        quantized = (x * inverse[:, None]).to(tl.float8e4nv)
        acc += tl.dot(weight, tl.trans(quantized)) * (x_scale * weight_scale)[None, :]
    mask = (rows[:, None] < N) & (ms[None, :] < M)
    out = Y + ms[None, :] * N + rows[:, None]
    if SPLIT == 1:
        tl.extra.cuda.gdc_launch_dependents()
        tl.store(out, acc, mask)
    else:
        tl.store(P + (pid_k * 16 + ms[None, :]) * N + rows[:, None], acc, mask)
        tl.extra.cuda.gdc_launch_dependents()
        tl.debug_barrier()
        # The last split to arrive reduces in split order and re-arms the counter.
        if tl.atomic_add(C + pid_n, 1, sem="acq_rel", scope="gpu") == SPLIT - 1:
            total = tl.zeros([BW, 16], tl.float32)
            for s in range(SPLIT):
                total += tl.load(
                    P + (s * 16 + ms[None, :]) * N + rows[:, None],
                    mask,
                    0.0,
                    cache_modifier=".cg",
                )
            tl.store(out, total, mask)
            tl.atomic_xchg(C + pid_n, 0, sem="relaxed", scope="gpu")


def _workspace(device):
    key = (device, torch.cuda.current_stream(device).cuda_stream)
    if key not in _workspaces:
        partials = max(split * 16 * n for (n, _), (_, split, _) in _CONFIGS.items())
        counters = max(triton.cdiv(n, bw) for (n, _), (bw, _, _) in _CONFIGS.items())
        _workspaces[key] = (
            torch.empty(partials, device=device, dtype=torch.float32),
            torch.zeros(counters, device=device, dtype=torch.int32),
        )
    return _workspaces[key]


def _descriptor(weight, bw):
    n, k = weight.shape
    key = (weight.data_ptr(), n, k, bw)
    if key not in _descriptors:
        _descriptors[key] = TensorDescriptor(weight, [n, k], [k, 1], [bw, 128])
    return _descriptors[key]


def tiny_gemm_fp8_blockwise(x, weight, weight_scale, output=None):
    """Apply W8A8 arithmetic to up to 16 BF16 rows and packed UE8M0 scales."""
    if not can_use_tiny_gemm_fp8_blockwise(x, weight, weight_scale):
        raise ValueError("Unsupported tiny block-FP8 GEMM input")
    m = x.shape[0]
    n, k = weight.shape
    if output is None:
        output = torch.empty((m, n), device=x.device, dtype=torch.bfloat16)
    elif (
        output.shape != (m, n)
        or output.dtype != torch.bfloat16
        or output.device != x.device
        or not output.is_contiguous()
    ):
        raise ValueError(
            "Tiny block-FP8 GEMM output must be contiguous BF16 [M, N] on the input device"
        )
    bw, split, warps = _CONFIGS[(n, k)]
    partials, counters = _workspace(x.device)
    _tiny_gemm_fp8_blockwise_kernel[(triton.cdiv(n, bw), split)](
        x,
        _descriptor(weight, bw),
        weight_scale,
        output,
        partials,
        counters,
        m,
        n,
        k,
        weight_scale.stride(0),
        weight_scale.stride(1),
        bw,
        split,
        num_warps=warps,
        launch_pdl=True,
    )
    return output
