"""The full-distribution top-p bound of a compact top-k verify support in two launches.

The same bound as ``dflash_utils._adjust_top_p_for_top_k``: top-p times the full vocabulary's
mass over the retained top-k mass, each term exp(x / T - max). The first launch splits every
row's vocabulary over several programs; the second adds the parts, the top-k mass and the
clamp. Only the summation order and the exp / divide rounding differ from the compiled path.
"""

import torch
import triton
import triton.language as tl

_BLOCK = 4096


@triton.jit
def _full_mass_part_kernel(
    logits_ptr,
    topk_ptr,
    temps_ptr,
    part_ptr,
    vocab,
    logits_stride,
    topk_stride,
    width,
    chunk,
    SPLIT: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    part = tl.program_id(1)
    temp = tl.load(temps_ptr + row // width)
    peak = tl.load(topk_ptr + row.to(tl.int64) * topk_stride).to(tl.float32) / temp
    base = logits_ptr + row.to(tl.int64) * logits_stride
    start = part * chunk
    acc = tl.zeros([BLOCK], dtype=tl.float32)
    for offset in range(0, chunk, BLOCK):
        lane = offset + tl.arange(0, BLOCK)
        cols = start + lane
        x = tl.load(
            base + cols, mask=(lane < chunk) & (cols < vocab), other=-float("inf")
        )
        acc += tl.exp(x.to(tl.float32) / temp - peak)
    tl.store(part_ptr + row * SPLIT + part, tl.sum(acc, axis=0))


@triton.jit
def _top_p_finish_kernel(
    part_ptr,
    topk_ptr,
    temps_ptr,
    top_ks_ptr,
    top_ps_ptr,
    out_ptr,
    topk_stride,
    width,
    max_top_k,
    SPLIT: tl.constexpr,
    K: tl.constexpr,
):
    row = tl.program_id(0)
    request = row // width
    temp = tl.load(temps_ptr + request)
    ranks = tl.arange(0, K)
    keep = (ranks < max_top_k) & (ranks < tl.load(top_ks_ptr + request))
    values = tl.load(
        topk_ptr + row.to(tl.int64) * topk_stride + ranks, mask=keep, other=0.0
    ).to(tl.float32)
    scaled = values / temp
    peak = tl.max(tl.where(keep, scaled, -float("inf")), axis=0)
    topk_mass = tl.sum(tl.where(keep, tl.exp(scaled - peak), 0.0), axis=0)
    full_mass = tl.sum(tl.load(part_ptr + row * SPLIT + tl.arange(0, SPLIT)), axis=0)
    top_p = tl.load(top_ps_ptr + request)
    tl.store(out_ptr + row, tl.minimum(top_p * (full_mass / topk_mass), 1.0))


def _split_for(rows: int, device: torch.device) -> int:
    # Enough programs for every SM to hold several; the rows alone give one wave at most.
    sms = torch.cuda.get_device_properties(device).multi_processor_count
    split = 4
    while split < 32 and rows * split < 8 * sms:
        split *= 2
    return split


def top_p_for_top_k_support(
    topk_logits: torch.Tensor,
    full_logits: torch.Tensor,
    temperatures: torch.Tensor,
    top_ks: torch.Tensor,
    top_ps: torch.Tensor,
    width: int,
) -> torch.Tensor:
    """[rows, 1] fp32 top-p bounds on the sorted top-k support, as
    ``_adjust_top_p_for_top_k`` computes them. ``topk_logits`` [rows, K] sorted descending,
    ``full_logits`` [rows, vocab]; ``temperatures`` / ``top_ks`` / ``top_ps`` per request
    (rows = requests x ``width``)."""
    rows, vocab = full_logits.shape
    max_top_k = topk_logits.shape[1]
    split = _split_for(rows, full_logits.device)
    chunk = triton.cdiv(triton.cdiv(vocab, split), _BLOCK) * _BLOCK
    part = torch.empty((rows, split), dtype=torch.float32, device=full_logits.device)
    out = torch.empty((rows, 1), dtype=torch.float32, device=full_logits.device)
    temps = temperatures.reshape(-1).float().contiguous()
    _full_mass_part_kernel[(rows, split)](
        full_logits,
        topk_logits,
        temps,
        part,
        vocab,
        full_logits.stride(0),
        topk_logits.stride(0),
        width,
        chunk,
        SPLIT=split,
        BLOCK=_BLOCK,
        num_warps=4,
    )
    _top_p_finish_kernel[(rows,)](
        part,
        topk_logits,
        temps,
        top_ks.reshape(-1).contiguous(),
        top_ps.reshape(-1).float().contiguous(),
        out,
        topk_logits.stride(0),
        width,
        max_top_k,
        SPLIT=split,
        K=triton.next_power_of_2(max_top_k),
        num_warps=4,
    )
    return out
