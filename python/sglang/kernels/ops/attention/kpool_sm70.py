"""SM70 indexer K-pool write.

The Triton pool kernels store fp8 and round through bf16, neither of which
Volta Triton can compile. The math is the same: per-channel softmax over the
pool, normalized Hadamard, then block fp8 with an optional power-of-two scale.
The packed page is 64 fp8 rows followed by 64 fp32 scales. The decode update
the target-verify write and the q quant run as kernels in attention/kpool_sm70.cuh; prefill stays torch.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional, Tuple

import torch

from sglang.kernels.jit.utils import cache_once, load_jit

if TYPE_CHECKING:
    from tvm_ffi.module import Module

_FP8_MAX = 448.0
_HADAMARD_SCALE = 0.08838834764831845  # 128 ** -0.5


@cache_once
def _module() -> Module:
    return load_jit(
        "kpool_sm70",
        cuda_files=["attention/kpool_sm70.cuh"],
        cuda_wrappers=[
            ("decode_update", "kpool_sm70::decode_update"),
            ("verify_write", "kpool_sm70::verify_write"),
            ("act_quant", "kpool_sm70::act_quant"),
        ],
        # Bitwise parity with the torch ops needs unfused multiply-adds.
        extra_cuda_cflags=["--fmad=false"],
    )


def _require_sm70(t: torch.Tensor, what: str) -> None:
    if t.device.type != "cuda" or torch.cuda.get_device_capability(t.device) != (7, 0):
        raise ValueError(f"{what} requires an SM70 CUDA device")


def _hadamard128(x: torch.Tensor) -> torch.Tensor:
    """Sylvester Hadamard on the last dimension, scaled by 128**-0.5."""
    h = x.float().reshape(*x.shape[:-1], 128)
    span = 1
    while span < 128:
        h = h.reshape(*h.shape[:-1], -1, 2, span)
        a = h[..., 0, :]
        b = h[..., 1, :]
        h = torch.stack((a + b, a - b), dim=-2).reshape(*x.shape[:-1], 128)
        span *= 2
    return h * _HADAMARD_SCALE


def _pow2_scale(amax: torch.Tensor) -> torch.Tensor:
    ratio = (amax / _FP8_MAX).clamp(min=torch.finfo(torch.float32).tiny)
    mant, exp = torch.frexp(ratio)
    # frexp: ratio = mant * 2^exp with mant in [0.5, 1). An exact power of
    # two has mant == 0.5, and ceil(log2(ratio)) is then exp - 1.
    pow2 = torch.where(mant == 0.5, exp - 1, exp)
    return torch.ldexp(torch.ones_like(ratio), pow2)


def _quantize(x: torch.Tensor, round_scale: bool) -> Tuple[torch.Tensor, torch.Tensor]:
    amax = x.abs().amax(dim=-1).clamp(min=1e-4)
    scale = _pow2_scale(amax) if round_scale else amax / _FP8_MAX
    q = (x / scale.unsqueeze(-1)).clamp(-_FP8_MAX, _FP8_MAX)
    return q.to(torch.float8_e4m3fn), scale


def _compress(values: torch.Tensor, round_scale: bool) -> Tuple[torch.Tensor, torch.Tensor]:
    """values is the pooled vector, fp32 [..., 128]."""
    rounded = values.float().to(torch.bfloat16).float()
    rotated = _hadamard128(rounded).to(torch.bfloat16).float()
    return _quantize(rotated, round_scale)


def _online_softmax(keys: torch.Tensor, scores: torch.Tensor) -> torch.Tensor:
    """Online softmax over the pool axis. keys/scores are [pool, rows, dim]."""
    acc = torch.zeros_like(keys[0], dtype=torch.float32)
    denom = torch.zeros(keys.shape[1], keys.shape[2], device=keys.device, dtype=torch.float32)
    running = torch.full_like(denom, float("-inf"))
    for slot in range(keys.shape[0]):
        score = scores[slot].float()
        new_max = torch.maximum(running, score)
        rescale = torch.exp(running - new_max)
        prob = torch.exp(score - new_max)
        denom = denom * rescale + prob
        acc = acc * rescale + keys[slot].float() * prob
        running = new_max
    return acc / denom


def _two_pass_softmax(keys: torch.Tensor, scores: torch.Tensor) -> torch.Tensor:
    """Two-pass softmax over the pool axis. keys/scores are [pool, rows, dim]."""
    scores = scores.float()
    peak = scores.amax(dim=0)
    weight = torch.exp(scores - peak)
    denom = weight.sum(dim=0)
    return (keys.float() * weight).sum(dim=0) / denom


def _write_pages(
    buf: torch.Tensor,
    loc: torch.Tensor,
    quantized: torch.Tensor,
    scale: torch.Tensor,
    slots_per_page: int,
    head_dim: int,
) -> None:
    if loc.numel() == 0:
        return
    loc = loc.to(torch.long)
    page = torch.div(loc, slots_per_page, rounding_mode="floor")
    slot = loc - page * slots_per_page
    k_bytes = slots_per_page * head_dim
    n_pages = buf.shape[0]
    k_view = (
        buf[:, :k_bytes]
        .view(torch.float8_e4m3fn)
        .view(n_pages, slots_per_page, head_dim)
    )
    k_view[page, slot] = quantized
    buf.view(torch.float32)[page, k_bytes // 4 + slot] = scale.float()


def _active_rows(mask: Optional[torch.Tensor], n: int, device: torch.device) -> torch.Tensor:
    if mask is None:
        return torch.ones(n, dtype=torch.bool, device=device)
    return mask.to(device=device, dtype=torch.bool)


def kpool_softmax_rotate_write_cache_sm70(
    pool,
    buf: torch.Tensor,
    slot_k: torch.Tensor,
    slot_score: torch.Tensor,
    ape: torch.Tensor,
    loc: torch.Tensor,
    write_mask: Optional[torch.Tensor],
    round_scale: bool,
    return_compressed: bool,
    write_cache: bool,
    has_write_mask: bool,
) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
    _require_sm70(slot_k, "kpool_softmax_rotate_write_cache")
    if slot_k.shape[0] == 0:
        if return_compressed:
            return (
                torch.empty((0, slot_k.shape[2]), dtype=torch.float8_e4m3fn, device=slot_k.device),
                torch.empty((0,), dtype=torch.float32, device=slot_k.device),
            )
        return None

    keys = slot_k.float()
    scores = slot_score.float() + ape.float().unsqueeze(0)
    # [pool, rows, dim] -> two-pass, matching the Triton prefill kernel.
    pooled = _two_pass_softmax(keys.transpose(0, 1), scores.transpose(0, 1))
    quantized, scale = _compress(pooled, round_scale)
    if write_cache:
        mask = _active_rows(write_mask if has_write_mask else None, slot_k.shape[0], slot_k.device)
        _write_pages(
            buf,
            loc[mask],
            quantized[mask],
            scale[mask],
            pool.slots_per_page,
            slot_k.shape[2],
        )
    if return_compressed:
        return quantized, scale
    return None


def kpool_assemble_softmax_rotate_write_cache_sm70(
    pool,
    buf: torch.Tensor,
    chunk_k: torch.Tensor,
    chunk_score: torch.Tensor,
    tail_k: torch.Tensor,
    tail_score: torch.Tensor,
    req_pool_idx: torch.Tensor,
    n_from_tail: torch.Tensor,
    chunk_src_start: torch.Tensor,
    tail_logical_base: torch.Tensor,
    ape: torch.Tensor,
    loc: torch.Tensor,
    write_mask: Optional[torch.Tensor],
    round_scale: bool,
    has_write_mask: bool,
) -> None:
    _require_sm70(chunk_k, "kpool_assemble_softmax_rotate_write_cache")
    n_pools = req_pool_idx.shape[0]
    if n_pools == 0:
        return
    pool_size = pool.index_kpool
    tail_size = tail_k.shape[1]
    req = req_pool_idx.to(torch.long)
    n_tail = n_from_tail.to(torch.long)
    src0 = chunk_src_start.to(torch.long)
    base = tail_logical_base.to(torch.long)
    keys = []
    scores = []
    for slot in range(pool_size):
        use_tail = slot < n_tail
        phys = torch.remainder(base + slot, tail_size)
        src = (src0 + (slot - n_tail)).clamp(min=0)
        k_tail = tail_k[req, phys].float()
        s_tail = tail_score[req, phys].float()
        k_chunk = chunk_k[src].float()
        s_chunk = chunk_score[src].float()
        k = torch.where(use_tail.unsqueeze(-1), k_tail, k_chunk)
        s = torch.where(use_tail.unsqueeze(-1), s_tail, s_chunk)
        keys.append(k)
        scores.append(s + ape[slot].float())
    pooled = _online_softmax(torch.stack(keys, 0), torch.stack(scores, 0))
    quantized, scale = _compress(pooled, round_scale)
    mask = _active_rows(write_mask if has_write_mask else None, n_pools, chunk_k.device)
    _write_pages(
        buf,
        loc[mask],
        quantized[mask],
        scale[mask],
        pool.slots_per_page,
        chunk_k.shape[-1],
    )


def scatter_kpool_tail_updates_sm70(
    chunk_k: torch.Tensor,
    chunk_score: torch.Tensor,
    tail_k: torch.Tensor,
    tail_score: torch.Tensor,
    req_pool_idx: torch.Tensor,
    dst_logical_start: torch.Tensor,
    chunk_src_start: torch.Tensor,
    n_write: torch.Tensor,
    pool_size: int,
) -> None:
    _require_sm70(chunk_k, "scatter_kpool_tail_updates")
    n_rows = req_pool_idx.shape[0]
    if n_rows == 0:
        return
    req = req_pool_idx.to(torch.long)
    dst0 = dst_logical_start.to(torch.long)
    src0 = chunk_src_start.to(torch.long)
    n_w = n_write.to(torch.long)
    tail_size = tail_k.shape[1]
    for slot in range(pool_size):
        active = slot < n_w
        if not bool(active.any()):
            continue
        phys = torch.remainder(dst0 + slot, tail_size)
        src = src0 + slot
        rows = active.nonzero(as_tuple=False).reshape(-1)
        tail_k[req[rows], phys[rows]] = chunk_k[src[rows]].to(tail_k.dtype)
        tail_score[req[rows], phys[rows]] = chunk_score[src[rows]].to(tail_score.dtype)


def kpool_decode_update_and_maybe_write_cache_sm70(
    pool,
    buf: torch.Tensor,
    tail_k: torch.Tensor,
    tail_score: torch.Tensor,
    key: torch.Tensor,
    slot_score: torch.Tensor,
    ape: torch.Tensor,
    block_tables: torch.Tensor,
    req_pool_indices: torch.Tensor,
    positions: torch.Tensor,
    seq_lens: torch.Tensor,
    out_cache_loc: torch.Tensor,
    round_scale: bool,
) -> None:
    _require_sm70(key, "kpool_decode_update_and_maybe_write_cache")
    if key.shape[0] == 0:
        return
    _module().decode_update(
        buf,
        tail_k,
        tail_score,
        key,
        slot_score,
        ape,
        block_tables,
        req_pool_indices,
        positions,
        seq_lens,
        out_cache_loc,
        pool.slots_per_page,
        round_scale,
    )


def kpool_write_tail_and_maybe_compress_sm70(
    pool,
    buf: torch.Tensor,
    key: torch.Tensor,
    score: torch.Tensor,
    tail_k: torch.Tensor,
    tail_score: torch.Tensor,
    ape: torch.Tensor,
    req_pool_indices: torch.Tensor,
    write_start: torch.Tensor,
    tail_logical_start: torch.Tensor,
    write_loc: torch.Tensor,
    out_cache_loc: torch.Tensor,
    num_draft_tokens: int,
    round_scale: bool,
    effective_n_per_batch: Optional[torch.Tensor],
) -> None:
    """Each closed pool matches the decode update on its last slot bitwise."""
    _require_sm70(key, "kpool_write_tail_and_maybe_compress")
    if key.shape[0] == 0:
        return
    if effective_n_per_batch is None:
        effective_n_per_batch = torch.empty(0, dtype=torch.int32, device=key.device)
    _module().verify_write(
        buf,
        tail_k,
        tail_score,
        key.contiguous(),
        score.contiguous(),
        ape.contiguous(),
        req_pool_indices,
        write_start,
        tail_logical_start,
        write_loc,
        out_cache_loc,
        effective_n_per_batch,
        num_draft_tokens,
        pool.slots_per_page,
        round_scale,
    )


def act_quant_sm70(
    x: torch.Tensor, round_scale: bool
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Block fp8 quant with 128-wide groups; x fp16/bf16 [..., K], contiguous."""
    y = torch.empty_like(x, dtype=torch.float8_e4m3fn)
    s = x.new_empty(*x.shape[:-1], x.shape[-1] // 128, dtype=torch.float32)
    _module().act_quant(y.view(torch.uint8), s, x, round_scale)
    return y, s
