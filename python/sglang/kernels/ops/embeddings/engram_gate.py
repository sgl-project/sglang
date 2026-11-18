"""Fused FP32 Engram gate with a single final cast to the activation dtype."""

from typing import Optional

import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice


@triton.jit
def _engram_gate_kernel(
    X,
    KV,
    QW,
    KW,
    IDS,
    O,
    IMAGE_ID,
    D: tl.constexpr,
    HC: tl.constexpr,
    EPS: tl.constexpr,
    CLAMP: tl.constexpr,
    B: tl.constexpr,
    HAS_IDS: tl.constexpr,
):
    row = tl.program_id(0)
    token = row // HC
    hc = row % HC
    col = tl.arange(0, B)
    mask = col < D
    x = tl.load(X + row * D + col, mask, 0).to(tl.float32)
    key = tl.load(KV + token * (HC + 1) * D + hc * D + col, mask, 0).to(tl.float32)
    q_weight = tl.load(QW + hc * D + col, mask, 0).to(tl.float32)
    k_weight = tl.load(KW + hc * D + col, mask, 0).to(tl.float32)
    weight = q_weight * k_weight
    rstd = tl.rsqrt(tl.sum(x * x, 0) / D + EPS) * tl.rsqrt(
        tl.sum(key * key, 0) / D + EPS
    )
    dot = tl.sum((x * weight) * key, 0) * rstd * (D**-0.5)
    gate = tl.sigmoid(libdevice.copysign(tl.sqrt(tl.maximum(tl.abs(dot), CLAMP)), dot))
    value = tl.load(KV + token * (HC + 1) * D + HC * D + col, mask, 0).to(tl.float32)
    out = x + gate * value
    if HAS_IDS:
        out = tl.where(tl.load(IDS + token) == IMAGE_ID, x, out)
    tl.store(O + row * D + col, out, mask)


def fused_engram_gate(
    x: torch.Tensor,
    kv: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    eps: float,
    clamp_value: float,
    input_ids: Optional[torch.Tensor] = None,
    image_token_id: Optional[int] = None,
) -> torch.Tensor:
    """Rows whose ``input_ids`` entry is ``image_token_id`` keep ``x`` unchanged."""
    assert x.ndim == 3 and kv.ndim == 2
    t, hc, d = x.shape
    assert kv.shape == (t, (hc + 1) * d)
    assert q_weight.shape == k_weight.shape == (hc, d)
    assert input_ids is None or (
        input_ids.shape == (t,) and input_ids.is_cuda and image_token_id is not None
    )
    assert all(
        a.is_cuda and a.is_contiguous() and a.device == x.device
        for a in (x, kv, q_weight, k_weight)
    )
    assert all(
        a.dtype in (torch.bfloat16, torch.float32) for a in (x, kv, q_weight, k_weight)
    )
    out = torch.empty_like(x)
    if t:
        _engram_gate_kernel[(t * hc,)](
            x,
            kv,
            q_weight,
            k_weight,
            input_ids,
            out,
            image_token_id,
            d,
            hc,
            eps,
            clamp_value,
            triton.next_power_of_2(d),
            input_ids is not None,
            num_warps=4,
            enable_fp_fusion=False,
        )
    return out
