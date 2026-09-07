"""Pure-torch fp8 / fp4 quantization used by the DeepSeek V4.1 reference path.

Activations: fp8 e4m3 with one ue8m0 scale per 32 elements along K. Dense weights:
fp8 e4m3 with one ue8m0 scale per 32x32 block. Expert weights: fp4 e2m1, two values
per byte packed along K, one ue8m0 scale per 32 elements. Scales are the smallest
power of two that maps the block absmax onto the format's max finite value.
"""

import torch
import torch.nn.functional as F

from sglang.srt.layers.quantization.fp8 import DSV4_DEQUANT_FP4_TABLE

FP8_MAX = 448.0
FP4_MAX = 6.0
FP8_BLOCK_SIZE = 32
FP4_BLOCK_SIZE = 32
FP8_AMAX_FLOOR = 1e-4
FP4_AMAX_FLOOR = 6 * 2.0**-126


def ceil_pow2(x: torch.Tensor) -> torch.Tensor:
    """2 ** ceil(log2(x)) for positive fp32 x, computed on the IEEE bits so the
    result is exact at powers of two."""
    bits = x.contiguous().view(torch.int32)
    exponent = ((bits >> 23) & 0xFF) - 127
    has_mantissa = (bits & 0x7FFFFF) != 0
    exponent = exponent + has_mantissa.to(torch.int32)
    return ((exponent + 127) << 23).view(torch.float32)


def block_scale(x: torch.Tensor, block_size: int, fmax: float, amax_floor: float):
    """Per-block ue8m0 scale, as fp32 powers of two, shape [..., N // block_size]."""
    amax = x.float().unflatten(-1, (-1, block_size)).abs().amax(dim=-1)
    amax = amax.clamp_min(amax_floor)
    # The kernel multiplies by the fp32 reciprocal rather than dividing. A Python
    # scalar keeps this free of host tensors, so it can run under CUDA graph capture.
    return ceil_pow2(amax * (1.0 / fmax))


def quant_fp8_act(x: torch.Tensor, block_size: int = FP8_BLOCK_SIZE):
    """Returns (fp8 e4m3 values, ue8m0 scales as fp32) with per-block scaling along K."""
    scale = block_scale(x, block_size, FP8_MAX, FP8_AMAX_FLOOR)
    scaled = x.float().unflatten(-1, (-1, block_size)) / scale.unsqueeze(-1)
    values = scaled.clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn)
    return values.flatten(-2), scale


def fake_quant_fp8(x: torch.Tensor, block_size: int = FP8_BLOCK_SIZE) -> torch.Tensor:
    """Quantize to fp8 and back, in x's dtype."""
    values, scale = quant_fp8_act(x, block_size)
    deq = values.float().unflatten(-1, (-1, block_size)) * scale.unsqueeze(-1)
    return deq.flatten(-2).to(x.dtype)


def round_fp4(x: torch.Tensor) -> torch.Tensor:
    """Round fp32 values in [-6, 6] onto the e2m1 grid with round-to-nearest-even."""
    magnitude = x.abs()
    step = torch.where(magnitude < 2.0, 0.5, torch.where(magnitude < 4.0, 1.0, 2.0))
    return torch.round(magnitude / step) * step * torch.sign(x)


def fake_quant_fp4(x: torch.Tensor, block_size: int = FP4_BLOCK_SIZE) -> torch.Tensor:
    """Quantize to fp4 (per-block ue8m0 scale) and back, in x's dtype."""
    scale = block_scale(x, block_size, FP4_MAX, FP4_AMAX_FLOOR)
    scaled = x.float().unflatten(-1, (-1, block_size)) / scale.unsqueeze(-1)
    deq = round_fp4(scaled.clamp(-FP4_MAX, FP4_MAX)) * scale.unsqueeze(-1)
    return deq.flatten(-2).to(x.dtype)


def dequant_fp8_weight(
    weight: torch.Tensor, scale: torch.Tensor, block_size: int = FP8_BLOCK_SIZE
):
    """fp8 [N, K] with ue8m0 scale [ceil(N / bs), ceil(K / bs)] -> fp32 [N, K]."""
    n, k = weight.shape
    scale = scale.float().repeat_interleave(block_size, dim=0)[:n]
    scale = scale.repeat_interleave(block_size, dim=1)[:, :k]
    return weight.float() * scale


def unpack_fp4_weight(weight: torch.Tensor) -> torch.Tensor:
    """float4_e2m1fn_x2 [N, K // 2] -> fp32 [N, K]."""
    bytes_ = weight.view(torch.uint8)
    low = bytes_ & 0x0F
    high = (bytes_ >> 4) & 0x0F
    table = DSV4_DEQUANT_FP4_TABLE.to(weight.device)
    return torch.stack([table[low.long()], table[high.long()]], dim=-1).flatten(-2)


def dequant_fp4_weight(
    weight: torch.Tensor, scale: torch.Tensor, block_size: int = FP4_BLOCK_SIZE
):
    """float4_e2m1fn_x2 [N, K // 2] with ue8m0 scale [N, K // bs] -> fp32 [N, K]."""
    values = unpack_fp4_weight(weight)
    return values * scale.float().repeat_interleave(block_size, dim=-1)


def naive_linear(
    x: torch.Tensor, weight: torch.Tensor, scale: torch.Tensor | None = None
):
    """x @ weight.T with the reference dtype rules: quantized weights take an fp8
    activation (per-32 ue8m0 scale) and accumulate in fp32; plain weights use F.linear."""
    if weight.dtype == torch.float4_e2m1fn_x2:
        w = dequant_fp4_weight(weight, scale)
    elif weight.dtype == torch.float8_e4m3fn:
        w = dequant_fp8_weight(weight, scale)
    else:
        return F.linear(x, weight)
    values, act_scale = quant_fp8_act(x)
    x_deq = values.float().unflatten(-1, (-1, FP8_BLOCK_SIZE)) * act_scale.unsqueeze(-1)
    return F.linear(x_deq.flatten(-2), w).to(x.dtype)
