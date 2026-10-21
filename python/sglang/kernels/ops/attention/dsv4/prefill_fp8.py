"""SM90 FP8 scorer for full, source and consumer DSV4.1 prefill index layers."""

import torch
import triton
import triton.language as tl


@triton.jit
def _prefill_fp8_score(
    Q,
    K,
    W,
    Lens,
    Out,
    length,
    stride_qr: tl.constexpr,
    stride_qh: tl.constexpr,
    stride_kr: tl.constexpr,
    stride_wr: tl.constexpr,
    stride_lens: tl.constexpr,
    tile: tl.constexpr,
):
    row = tl.program_id(0)
    column = tl.program_id(1) * tile + tl.arange(0, tile)
    head = tl.arange(0, 32)
    dim = tl.arange(0, 128)
    visible = tl.minimum(tl.load(Lens + row * stride_lens), length)
    valid = (column < length) & (column < visible)

    # tile >= 64 and 32 heads make this an SM90 WGMMA-shaped FP8 dot.
    keys = tl.load(
        K + column[:, None] * stride_kr + dim[None, :],
        mask=valid[:, None],
        other=0.0,
    )
    query = tl.load(Q + row * stride_qr + head[:, None] * stride_qh + dim[None, :])
    dot = tl.dot(keys, tl.trans(query)).to(tl.bfloat16).to(tl.float32)
    weights = tl.load(W + row * stride_wr + head).to(tl.float32)
    weighted = (tl.maximum(dot, 0.0) * weights[None, :]).to(tl.bfloat16)
    score = tl.sum(weighted.to(tl.float32), axis=1).to(tl.bfloat16).to(tl.float32)
    tl.store(
        Out + row * length + column,
        tl.where(valid, score, -float("inf")),
        mask=column < length,
    )


def prefill_fp8_scores(
    q: torch.Tensor,
    keys: torch.Tensor,
    weights: torch.Tensor,
    visible_lens: torch.Tensor,
) -> torch.Tensor:
    """Return dense FP32 scores with the reference indexer's BF16 round points.

    ``q`` is ``[rows, 32, 128]`` and ``keys`` is request-shared ``[length, 128]``.
    Q and K are unit-scale E4M3 values; weights remain BF16. Invalid/causally hidden
    columns are written as ``-inf``.
    """
    if q.ndim != 3 or q.shape[1:] != (32, 128):
        raise ValueError("q must have shape [rows, 32, 128]")
    if keys.ndim != 2 or keys.shape[1] != 128:
        raise ValueError("keys must have shape [length, 128]")
    if weights.shape != q.shape[:2]:
        raise ValueError("weights must have shape [rows, 32]")
    if visible_lens.shape != (q.shape[0],):
        raise ValueError("visible_lens must have shape [rows]")
    if q.dtype != torch.float8_e4m3fn or keys.dtype != q.dtype:
        raise ValueError("q and keys must be float8_e4m3fn")
    if weights.dtype != torch.bfloat16:
        raise ValueError("weights must be bfloat16")
    if visible_lens.dtype not in (torch.int32, torch.int64):
        raise ValueError("visible_lens must be int32 or int64")
    if not q.is_cuda or any(
        tensor.device != q.device for tensor in (keys, weights, visible_lens)
    ):
        raise ValueError("all inputs must be on the same CUDA device")
    if torch.cuda.get_device_capability(q.device)[0] != 9:
        raise ValueError("prefill FP8 scorer requires SM90")
    if q.stride(-1) != 1 or keys.stride(-1) != 1 or weights.stride(-1) != 1:
        raise ValueError("the innermost Q/K/weight dimensions must be contiguous")

    rows, length = q.shape[0], keys.shape[0]
    out = torch.empty((rows, length), device=q.device, dtype=torch.float32)
    if not rows or not length:
        return out
    _prefill_fp8_score[(rows, triton.cdiv(length, 128))](
        q,
        keys,
        weights,
        visible_lens,
        out,
        length,
        q.stride(0),
        q.stride(1),
        keys.stride(0),
        weights.stride(0),
        visible_lens.stride(0),
        tile=128,
        num_warps=4,
        enable_fp_fusion=False,
    )
    return out
