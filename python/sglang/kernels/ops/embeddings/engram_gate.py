"""Fused FP32 Engram gate with a single final cast to the activation dtype."""

from typing import Optional, Tuple

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
    O,
    IDS,
    image_token_id,
    D: tl.constexpr,
    HC: tl.constexpr,
    EPS: tl.constexpr,
    CLAMP: tl.constexpr,
    B: tl.constexpr,
    KEEP_IMAGE_ROWS: tl.constexpr,
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
    if KEEP_IMAGE_ROWS:
        # image tokens keep x: the model's where(input_ids == image_token_id, x, gated)
        is_image = tl.load(IDS + token) == image_token_id
        out = tl.where(is_image, x, out)
    tl.store(O + row * D + col, out, mask)


def fused_engram_gate(
    x: torch.Tensor,
    kv: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    eps: float,
    clamp_value: float,
    image_select: Optional[Tuple[torch.Tensor, int]] = None,
) -> torch.Tensor:
    """``image_select = (input_ids [T], image_token_id)`` keeps ``x`` on the rows whose input id is
    the image token, as the model's ``torch.where`` after the gate does (bitwise: the kept bf16
    rows round-trip through fp32 exactly)."""
    assert x.ndim == 3 and kv.ndim == 2
    t, hc, d = x.shape
    assert kv.shape == (t, (hc + 1) * d)
    assert q_weight.shape == k_weight.shape == (hc, d)
    assert all(
        a.is_cuda and a.is_contiguous() and a.device == x.device
        for a in (x, kv, q_weight, k_weight)
    )
    assert all(
        a.dtype in (torch.bfloat16, torch.float32) for a in (x, kv, q_weight, k_weight)
    )
    out = torch.empty_like(x)
    if image_select is not None:
        input_ids, image_token_id = image_select
        assert input_ids.shape == (t,) and input_ids.stride(0) == 1, input_ids.shape
    if t:
        _engram_gate_kernel[(t * hc,)](
            x,
            kv,
            q_weight,
            k_weight,
            out,
            input_ids if image_select is not None else out,
            image_token_id if image_select is not None else 0,
            d,
            hc,
            eps,
            clamp_value,
            triton.next_power_of_2(d),
            KEEP_IMAGE_ROWS=image_select is not None,
            num_warps=4,
            enable_fp_fusion=False,
        )
    return out
