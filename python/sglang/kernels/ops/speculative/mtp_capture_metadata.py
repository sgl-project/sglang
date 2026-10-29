# SPDX-License-Identifier: Apache-2.0
"""Integer-only MTP seed publication with opt-in PDL launch dependency."""
import torch
import triton
import triton.language as tl

from sglang.kernels.ops.pdl_small import ENABLED as _PDL_SMALL


@triton.jit
def _mtp_capture_metadata(
    TOPK, EXTEND, ACCEPT, REQ, LENGTHS, SEED, CAPTURED,
    ROWS: tl.constexpr, COLS: tl.constexpr,
    TOPK_STRIDE: tl.constexpr, SEED_STRIDE: tl.constexpr,
    FRONT: tl.constexpr, TRASH: tl.constexpr, BLOCK: tl.constexpr,
    PDL_SMALL: tl.constexpr = False,
):
    # BS1 native anchor: cumsum(extend)-1-(extend-front-accept).
    # Keep int64 arithmetic and clamp exactly as the original tensor expression.
    # Protect dynamic reads and reused-buffer writes before relaxing this edge.
    if PDL_SMALL:
        tl.extra.cuda.gdc_wait()
    extend = tl.load(EXTEND).to(tl.int64)
    accept = tl.load(ACCEPT).to(tl.int64)
    anchor = tl.minimum(tl.maximum(FRONT + accept - 1, 0), ROWS - 1)
    req = tl.load(REQ).to(tl.int64)
    row = tl.where((extend > 0) & (req > 0), req, TRASH)
    col = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    values = tl.load(TOPK + anchor * TOPK_STRIDE + col, col < COLS, 0)
    tl.store(SEED + row * SEED_STRIDE + col, values, col < COLS)
    if tl.program_id(0) == 0:
        length = tl.load(LENGTHS + anchor)
        tl.store(CAPTURED + row, length)
    if PDL_SMALL:
        tl.extra.cuda.gdc_launch_dependents()


def try_capture_metadata(topk, extend, accept, req, lengths, seed, captured, front, trash):
    """BS1 only: M8/M12 and non-native layouts keep the complete parent path."""
    if not (
        topk.is_cuda and topk.ndim == 2 and topk.shape[0] > 0
        and topk.dtype == torch.int32 and topk.stride(1) == 1
        and extend.numel() == accept.numel() == req.numel() == 1
        and lengths.numel() >= topk.shape[0] and lengths.is_contiguous()
        and seed.dtype == captured.dtype == torch.int32
        and seed.ndim == 2 and seed.stride(1) == 1
        and topk.shape[1] <= seed.shape[1]
    ):
        return False
    _mtp_capture_metadata[(triton.cdiv(topk.shape[1], 256),)](
        topk, extend, accept, req, lengths, seed, captured,
        ROWS=topk.shape[0], COLS=topk.shape[1], TOPK_STRIDE=topk.stride(0),
        SEED_STRIDE=seed.stride(0), FRONT=int(front), TRASH=int(trash),
        BLOCK=256, num_warps=4, launch_pdl=_PDL_SMALL,
        PDL_SMALL=_PDL_SMALL,
    )
    return True
