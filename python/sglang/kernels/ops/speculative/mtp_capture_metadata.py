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


@triton.jit
def _mtp_capture_metadata_batched(
    TOPK, EXTEND, ACCEPT, REQ, SEQLENS, T2B, SEED, CAPTURED, BS, ROWS,
    BS_P2: tl.constexpr, COLS: tl.constexpr,
    TOPK_STRIDE: tl.constexpr, SEED_STRIDE: tl.constexpr,
    FRONT: tl.constexpr, TRASH: tl.constexpr, HAS_ACCEPT: tl.constexpr,
    BLOCK: tl.constexpr,
):
    # One program per (request, column block). Integer-only, the torch expression of
    # _capture_mtp_sparse_indices_from_extend_lens: block_end = cumsum(extend) - 1 (int64),
    # anchor = block_end - (extend - front - accept) when accept counts exist, clamped to
    # [0, ROWS - 1]; row = req if extend > 0 and req > 0 else the trash row; the seed row
    # takes topk[anchor, :COLS] and captured_len takes int32(seq_lens)[t2b[anchor]].
    b = tl.program_id(0)
    offs = tl.arange(0, BS_P2)
    extend_all = tl.load(EXTEND + offs, offs < BS, other=0).to(tl.int64)
    block_end = tl.sum(tl.where(offs <= b, extend_all, 0), 0) - 1
    extend = tl.load(EXTEND + b).to(tl.int64)
    if HAS_ACCEPT:
        accept = tl.load(ACCEPT + b).to(tl.int64)
        anchor = block_end - (extend - FRONT - accept)
    else:
        anchor = block_end
    anchor = tl.minimum(tl.maximum(anchor, 0), ROWS - 1)
    req = tl.load(REQ + b).to(tl.int64)
    row = tl.where((extend > 0) & (req > 0), req, TRASH)
    col = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    values = tl.load(TOPK + anchor * TOPK_STRIDE + col, col < COLS, 0)
    tl.store(SEED + row * SEED_STRIDE + col, values, col < COLS)
    if tl.program_id(1) == 0:
        seq = tl.load(T2B + anchor).to(tl.int64)
        length = tl.load(SEQLENS + seq).to(tl.int32)
        tl.store(CAPTURED + row, length)


_BATCHED_MAX_REQUESTS = 256


def try_capture_metadata_batched(
    topk, extend, accept, front, req, seq_lens, token_to_batch, seed, captured, trash
):
    """Any batch (1..256 requests): the seed capture of one MTP draft-extend layer in one
    launch. Returns False, launching nothing, when the inputs fall outside the contract;
    the caller then runs the torch expression."""
    bs = req.numel()
    if not (
        topk.is_cuda and topk.ndim == 2 and topk.shape[0] > 0
        and topk.dtype == torch.int32 and topk.stride(1) == 1
        and 0 < bs <= _BATCHED_MAX_REQUESTS
        and extend.numel() == bs and extend.dtype in (torch.int32, torch.int64)
        and extend.is_contiguous()
        and (accept is None or (accept.numel() == bs and accept.is_contiguous()
                                and accept.dtype in (torch.int32, torch.int64)))
        and req.dtype in (torch.int32, torch.int64) and req.is_contiguous()
        and seq_lens.dtype in (torch.int32, torch.int64) and seq_lens.is_contiguous()
        and token_to_batch.dtype in (torch.int32, torch.int64)
        and token_to_batch.is_contiguous()
        and token_to_batch.numel() >= topk.shape[0]
        and seed.dtype == captured.dtype == torch.int32
        and seed.ndim == 2 and seed.stride(1) == 1 and captured.is_contiguous()
        and topk.shape[1] <= seed.shape[1]
        and int(trash) < seed.shape[0] and int(trash) < captured.numel()
    ):
        return False
    block = 256
    _mtp_capture_metadata_batched[(bs, triton.cdiv(topk.shape[1], block))](
        topk, extend, accept if accept is not None else extend, req, seq_lens,
        token_to_batch, seed, captured, bs, topk.shape[0],
        BS_P2=triton.next_power_of_2(bs), COLS=topk.shape[1],
        TOPK_STRIDE=topk.stride(0), SEED_STRIDE=seed.stride(0), FRONT=int(front),
        TRASH=int(trash), HAS_ACCEPT=accept is not None, BLOCK=block, num_warps=4,
    )
    return True
