import torch
import triton
import triton.language as tl

from sglang.kernels.ops.speculative.dfly import _check_tensor


@triton.jit
def _build_dfly_sparse_indices_stacked_kernel(
    Topk,
    Req,
    Anchors,
    Active,
    ReqToToken,
    SortedWindow,
    Out,
    REQ_STRIDE: tl.constexpr,
    TOPK_PLANE_STRIDE: tl.constexpr,
    CONTEXT_STRIDE: tl.constexpr,
    OUT_PLANE_STRIDE: tl.constexpr,
    TOPK: tl.constexpr,
    WINDOW: tl.constexpr,
    WIDTH: tl.constexpr,
    OUT_WIDTH: tl.constexpr,
    PHYSICAL: tl.constexpr,
    BLOCK: tl.constexpr,
    SEARCH_STEPS: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    query = tl.program_id(1)
    plane = tl.program_id(2).to(tl.int64)
    req = tl.load(Req + row * REQ_STRIDE).to(tl.int64)
    anchor = tl.load(Anchors + row)
    active = tl.load(Active + row)
    for start in range(tl.cdiv(OUT_WIDTH, BLOCK)):
        col = start * BLOCK + tl.arange(0, BLOCK)
        history = tl.load(
            Topk + plane * TOPK_PLANE_STRIDE + req * TOPK + col,
            (col < TOPK) & active,
            other=-1,
        ).to(tl.int64)
        if WINDOW > 0:
            if PHYSICAL:
                lo = tl.full((BLOCK,), 0, tl.int32)
                hi = tl.full((BLOCK,), WINDOW, tl.int32)
                for _ in range(SEARCH_STEPS):
                    mid = (lo + hi) // 2
                    value = tl.load(
                        SortedWindow + row * WINDOW + mid,
                        (mid < WINDOW) & (col < TOPK) & active,
                        other=0x7FFFFFFFFFFFFFFF,
                    )
                    smaller = value < history
                    lo = tl.where(smaller, mid + 1, lo)
                    hi = tl.where(smaller, hi, mid)
                found = tl.load(
                    SortedWindow + row * WINDOW + lo,
                    (lo < WINDOW) & (col < TOPK) & active,
                    other=-1,
                )
                in_window = (lo < WINDOW) & (found == history)
            else:
                in_window = (history >= anchor - WINDOW) & (history < anchor)
            history = tl.where(in_window, -1, history)
        history = tl.maximum(history, -1)
        w = col - TOPK
        b = w - WINDOW
        is_window = (w >= 0) & (w < WINDOW)
        is_block = (b >= 0) & (b <= query)
        pos = tl.where(is_window, anchor - WINDOW + w, anchor + b)
        valid = (is_window | is_block) & (pos >= 0) & active
        if PHYSICAL:
            ids = tl.load(ReqToToken + req * CONTEXT_STRIDE + pos, valid, other=-1)
        else:
            ids = tl.where(valid, pos, -1)
        value = tl.where(col < TOPK, history, tl.where(valid, ids, -1))
        tl.store(
            Out + plane * OUT_PLANE_STRIDE + (row * WIDTH + query) * OUT_WIDTH + col,
            tl.where(active, value, -1),
            col < OUT_WIDTH,
        )


def build_dfly_sparse_indices_stacked(
    *,
    target_topk,
    req_pool_indices,
    anchors,
    active,
    req_to_token,
    sorted_window,
    query_width,
    physical,
    out,
):
    """Fill all producer planes in one launch, retaining the full rounded width.

    CUDA/Triton; int32 contiguous target planes and output rows. A sliced output
    may retain its maximum-capacity producer stride. Physical history excludes
    window membership using the sorted IDs from the fixed-output query plan.
    """
    bs = anchors.numel()
    producers, _, topk = target_topk.shape
    window = sorted_window.shape[1]
    device = anchors.device
    ints = (torch.int32, torch.int64)
    for name, tensor, shape, dtype in (
        (
            "target_topk",
            target_topk,
            (producers, req_to_token.shape[0], topk),
            (torch.int32,),
        ),
        ("anchors", anchors, (bs,), (torch.int64,)),
        ("active", active, (bs,), (torch.bool,)),
        (
            "req_to_token",
            req_to_token,
            (target_topk.shape[1], req_to_token.shape[1]),
            ints,
        ),
        ("sorted_window", sorted_window, (bs, window), (torch.int64,)),
    ):
        _check_tensor(tensor, shape=shape, dtype=dtype, device=device, name=name)
    _check_tensor(
        req_pool_indices,
        shape=(bs,),
        dtype=ints,
        device=device,
        name="req_pool_indices",
        contiguous=False,
    )
    _check_tensor(
        out,
        shape=(producers, bs * query_width, out.shape[2]),
        dtype=(torch.int32,),
        device=device,
        name="out",
        contiguous=False,
    )
    if (
        out.stride(2) != 1
        or out.stride(1) != out.shape[2]
        or out.stride(0) < bs * query_width * out.shape[2]
    ):
        raise ValueError(
            "DFly sparse output must have contiguous, nonoverlapping rows and planes"
        )
    if topk + window + query_width > out.shape[2]:
        raise ValueError(
            "DFly sparse output is too narrow for history, window and query block"
        )
    if not 0 < query_width <= 128 or not 0 <= window <= 1024:
        raise ValueError(
            "DFly fused indices support query widths 1..128 and windows 0..1024"
        )
    if bs == 0:
        return
    _build_dfly_sparse_indices_stacked_kernel[(bs, query_width, producers)](
        target_topk,
        req_pool_indices,
        anchors,
        active,
        req_to_token,
        sorted_window,
        out,
        req_pool_indices.stride(0),
        target_topk.stride(0),
        req_to_token.stride(0),
        out.stride(0),
        topk,
        window,
        query_width,
        out.shape[2],
        physical,
        256,
        (window + 1).bit_length(),
    )
