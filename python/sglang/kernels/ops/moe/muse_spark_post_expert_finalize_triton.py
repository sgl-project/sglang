from typing import Optional

import torch
import triton
import triton.language as tl

_MIN_TOKENS_FOR_FUSED = 32


@triton.jit(do_not_specialize=["part"])
def _post_expert_norm_weighted_sum_kernel(
    x_ptr,
    w_ptr,
    s_ptr,
    out_ptr,
    ids_ptr,
    rows_ptr,
    part_ptr,
    part,
    TOPK: tl.constexpr,
    H: tl.constexpr,
    eps,
    APPLY_NORM: tl.constexpr,
    HAS_WEIGHT: tl.constexpr,
    WEIGHT_FIRST: tl.constexpr,
    HAS_IDS: tl.constexpr,
    HAS_ROWS: tl.constexpr,
    ROUND_PRODUCT: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    tok = tl.program_id(0)
    if HAS_ROWS:
        tok += tl.load(part_ptr + part)
        if tok >= tl.load(part_ptr + part + 1):
            return
    offs = tl.arange(0, BLOCK_H)
    mask = offs < H

    if HAS_WEIGHT:
        w = tl.load(w_ptr + offs, mask=mask, other=0.0).to(tl.float32)

    acc = tl.zeros([BLOCK_H], dtype=tl.float32)

    for k in tl.static_range(TOPK):
        base = (tok * TOPK + k) * H
        live = True
        if HAS_ROWS:
            base = tl.load(rows_ptr + tok * TOPK + k).to(tl.int64) * H
            live = base >= 0
        if HAS_IDS:
            live = tl.load(ids_ptr + tok * TOPK + k) >= 0
        if live:
            x = tl.load(x_ptr + base + offs, mask=mask, other=0.0).to(tl.float32)

            if APPLY_NORM:
                if HAS_WEIGHT and WEIGHT_FIRST:
                    y = x * w
                    var = tl.sum(y * y, axis=0) / H
                    y = y * tl.rsqrt(var + eps)
                elif HAS_WEIGHT:
                    var = tl.sum(x * x, axis=0) / H
                    y = x * tl.rsqrt(var + eps) * w
                else:
                    var = tl.sum(x * x, axis=0) / H
                    y = x * tl.rsqrt(var + eps)
            else:
                y = x

            s = tl.load(s_ptr + tok * TOPK + k).to(tl.float32)
            product = y * s
            if ROUND_PRODUCT:
                product = product.to(x_ptr.dtype.element_ty).to(tl.float32)
            acc += product

    tl.store(out_ptr + tok * H + offs, acc.to(out_ptr.dtype.element_ty), mask=mask)


def fused_post_expert_norm_weighted_sum(
    expert_output: torch.Tensor,
    topk_scores: torch.Tensor,
    norm_weight: Optional[torch.Tensor] = None,
    eps: float = 1e-8,
    weight_first: bool = True,
    apply_norm: bool = True,
    slot_expert_ids: Optional[torch.Tensor] = None,
    round_product: bool = False,
    rows: Optional[torch.Tensor] = None,
    out: Optional[torch.Tensor] = None,
    part: Optional[tuple[torch.Tensor, int]] = None,
) -> torch.Tensor:
    """Normalize and combine experts without reading inactive slots.

    round_product preserves input-dtype multiplication before FP32 accumulation
    when combining an already normalized low-precision tensor.
    """
    x = expert_output.contiguous()
    tokens, topk, hidden = x.shape if rows is None else (*rows.shape, x.shape[-1])
    # the eager path below handles only the plain layout: no rows / part, a fresh output
    plain = rows is None and part is None and out is None
    if out is None:
        out = torch.empty((tokens, hidden), dtype=x.dtype, device=x.device)
    if tokens == 0:
        return out
    if (
        tokens < _MIN_TOKENS_FOR_FUSED
        and slot_expert_ids is None
        and not round_product
        and plain
    ):
        y = x
        if apply_norm:
            f = y.float()
            if norm_weight is not None and weight_first:
                f = f * norm_weight.float()
                f = f * torch.rsqrt(f.pow(2).mean(-1, keepdim=True) + eps)
            elif norm_weight is not None:
                f = f * torch.rsqrt(f.pow(2).mean(-1, keepdim=True) + eps)
                f = f * norm_weight.float()
            else:
                f = f * torch.rsqrt(f.pow(2).mean(-1, keepdim=True) + eps)
            y = f
        else:
            y = y.float()
        y = y * topk_scores.float().unsqueeze(-1)
        return y.sum(dim=1).to(x.dtype)

    block_h = triton.next_power_of_2(hidden)
    _post_expert_norm_weighted_sum_kernel[(tokens,)](
        x,
        norm_weight.contiguous() if norm_weight is not None else x,
        topk_scores.contiguous(),
        out,
        slot_expert_ids.contiguous() if slot_expert_ids is not None else x,
        rows.contiguous() if rows is not None else x,
        *(part or (x, 0)),
        TOPK=topk,
        H=hidden,
        eps=eps,
        APPLY_NORM=apply_norm,
        HAS_WEIGHT=norm_weight is not None,
        WEIGHT_FIRST=weight_first,
        HAS_IDS=slot_expert_ids is not None,
        HAS_ROWS=rows is not None,
        ROUND_PRODUCT=round_product,
        BLOCK_H=block_h,
        num_warps=8 if block_h >= 4096 else 4,
        num_stages=2,
    )
    return out


__all__ = ["fused_post_expert_norm_weighted_sum"]
