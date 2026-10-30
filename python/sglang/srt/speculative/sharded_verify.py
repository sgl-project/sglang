# SPDX-License-Identifier: Apache-2.0
"""Greedy target sampling on vocabulary-sharded LM-head output.

Each rank reduces its ``[rows, V / tp]`` shard to int64 keys. A small cross-rank
exchange then picks the winning token, matching ``torch.argmax`` over gathered FP32
logits bit for bit. Alternate numeric policies can produce compatible local keys and
reuse :func:`reduce_sharded_token_keys` without owning the collective.
"""

from __future__ import annotations

import logging
from importlib import import_module
from typing import Optional

import msgspec
import torch
import triton
import triton.language as tl

logger = logging.getLogger(__name__)


_provider_spec = import_module("sglang.kernels.plugin_kernels.nvidia.spec")
_provider_vocab_shard = getattr(_provider_spec, "VocabShardView", None)
if _provider_vocab_shard is None:

    class VocabShard(msgspec.Struct, frozen=True):
        """Columns [0, valid) are global token ids ``start + column``."""

        start: int
        valid: int
        vocab_size: int

else:
    # When the optional binding contract is installed, expose its one canonical
    # shard-view type.  Runtime-only builds retain the structurally identical
    # local type above and never import a binding-only symbol eagerly.
    VocabShard = _provider_vocab_shard


# key of (-inf, token id 0xFFFFFFFF): below every real column's key
_KEY_FLOOR = -2139095041 << 32


@triton.jit
def _shard_key_kernel(
    logits_ptr,
    stride_row,
    keys_ptr,
    start,
    valid,
    vocab_size,
    span,
    KEY_FLOOR: tl.constexpr,
    BLOCK: tl.constexpr,
):
    # program (row, part): columns [part * span, (part + 1) * span) of the row's shard
    row = tl.program_id(0).to(tl.int64)
    part = tl.program_id(1)
    lo = part * span
    hi = tl.minimum(lo + span, valid)
    best = tl.full([BLOCK], KEY_FLOOR, tl.int64)
    for off in range(lo, hi, BLOCK):
        cols = off + tl.arange(0, BLOCK)
        gid = start + cols
        mask = (cols < hi) & (gid < vocab_size)
        x = tl.load(
            logits_ptr + row * stride_row + cols, mask=mask, other=float("-inf")
        )
        x = x.to(tl.float32)
        # NaN never wins, as after sanitize_nan_logits; -0.0 ties +0.0 as in torch.argmax
        x = tl.where(x == x, x, float("-inf")) + 0.0
        bits = x.to(tl.int32, bitcast=True)
        # monotone in x as a signed int32; the low word prefers the smaller token id
        ordered = tl.where(bits < 0, bits ^ 0x7FFFFFFF, bits).to(tl.int64)
        key = (ordered << 32) | (4294967295 - gid.to(tl.int64))
        key = tl.where(mask, key, KEY_FLOOR)
        best = tl.maximum(best, key)
    tl.store(keys_ptr + row * tl.num_programs(1) + part, tl.max(best, axis=0))


_KEY_BLOCK = 2048


def key_splits(valid: int) -> int:
    """Programs per row, padded even so a row spans whole multimem chunks."""
    n = -(-valid // _KEY_BLOCK)
    return n + (n & 1)


def shard_keys(
    logits: torch.Tensor,
    shard: VocabShard,
    splits: int = 1,
    *,
    temperatures: Optional[torch.Tensor] = None,
    seeds: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """One int64 key per row of this rank's shard: (order-preserving FP32 bits of the
    row's best value) << 32 | (0xFFFFFFFF - token id). ``splits > 1`` returns
    ``[rows, splits]``, the best key from each column range. Greedy ignores existing
    per-request seed material; numeric temperature policies use the local-key seam."""
    if temperatures is not None:
        raise ValueError("runtime shard_keys is greedy-only")
    assert logits.stride(-1) == 1
    rows = logits.shape[0]
    keys = torch.empty(
        (rows, splits) if splits > 1 else (rows,),
        dtype=torch.int64,
        device=logits.device,
    )
    if rows == 0:
        return keys
    _shard_key_kernel[(rows, splits)](
        logits,
        logits.stride(0),
        keys,
        shard.start,
        shard.valid,
        shard.vocab_size,
        -(-shard.valid // splits),
        KEY_FLOOR=_KEY_FLOOR,
        BLOCK=_KEY_BLOCK,
        num_warps=8,
    )
    return keys


@triton.jit
def _keys_to_tokens_kernel(
    keys_ptr,
    tokens_ptr,
    stride_rank,
    stride_row,
    world,
    parts,
    KEY_FLOOR: tl.constexpr,
    BLOCK_R: tl.constexpr,
    BLOCK_P: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    r = tl.arange(0, BLOCK_R)[:, None]
    p = tl.arange(0, BLOCK_P)[None, :]
    k = tl.load(
        keys_ptr + row * stride_row + r * stride_rank + p,
        mask=(r < world) & (p < parts),
        other=KEY_FLOOR,
    )
    best = tl.max(tl.max(k, axis=1), axis=0)
    tl.store(tokens_ptr + row, 4294967295 - (best & 4294967295))


def _reduce_keys(keys, rows, world, parts, stride_rank, stride_row) -> torch.Tensor:
    """[rows] token ids from the keys at keys[row * stride_row + rank * stride_rank + part]
    (int64 elements), the max over every rank and part of a row, in one launch."""
    tokens = torch.empty(rows, dtype=torch.int64, device=keys.device)
    if rows:
        _keys_to_tokens_kernel[(rows,)](
            keys,
            tokens,
            stride_rank,
            stride_row,
            world,
            parts,
            KEY_FLOOR=_KEY_FLOOR,
            BLOCK_R=triton.next_power_of_2(world),
            BLOCK_P=triton.next_power_of_2(parts),
            num_warps=1,
        )
    return tokens


def keys_to_tokens(
    gathered: torch.Tensor, tp_size: int, splits: int = 1
) -> torch.Tensor:
    """[tp * rows * splits] keys of all ranks (rank-major) -> [rows] int64 token ids."""
    if splits == 1:
        best = gathered.view(tp_size, -1).amax(dim=0)
        return 0xFFFFFFFF - (best & 0xFFFFFFFF)
    rows = gathered.numel() // (tp_size * splits)
    return _reduce_keys(gathered, rows, tp_size, splits, rows * splits, splits)


# The key exchange's multimem state is built by the first eager, rank-lockstep call.
# A row of one rank is ``width`` int64 values (four BF16 values each).
_MC_MAX_ROWS = 1024
_mc_state = None
_mc_width = 0


def _mc_build(tp_group, hidden_size: int, device):
    """The multimem state, or False (NCCL keeps the exchange) when the buffer has no
    multicast mapping, e.g. a TP group across nodes on torch symmetric memory."""
    from sglang.srt.distributed.device_communicators import triton_symm_mem_ag

    try:
        state = triton_symm_mem_ag.create_state(
            tp_group.device_group,
            tp_group.rank_in_group,
            max_tokens=_MC_MAX_ROWS,
            hidden_size=hidden_size,
            device=device,
        )
    except Exception as e:  # rendezvous refused: every rank sees the same failure
        logger.warning("Sharded verify multimem setup failed; using NCCL: %s", e)
        return False
    if state.symm_mem_hdl.multicast_ptr == 0:
        logger.warning("Sharded verify has no multicast mapping; using NCCL")
        return False
    return state


def _mc_gather_keys(keys: torch.Tensor, tp_group) -> Optional[torch.Tensor]:
    """[rows] or [rows, parts] int64 keys of every rank -> [rows] token ids, by one
    multimem all-gather (rows of at least 2 int64: a single key goes twice); None when
    there is no multicast state."""
    global _mc_state, _mc_width
    from sglang.srt.distributed.device_communicators import triton_symm_mem_ag

    world, rows = tp_group.world_size, keys.shape[0]
    if keys.dim() == 1:
        keys = torch.stack((keys, keys), dim=1)
    width = keys.shape[1]
    if _mc_state is None:
        _mc_state = _mc_build(tp_group, 4 * width * world, keys.device)
        _mc_width = width
    if _mc_state is False:
        return None
    assert width == _mc_width, (width, _mc_width)
    out = triton_symm_mem_ag.all_gather_inner(
        _mc_state, keys.view(torch.bfloat16), 4 * width * world, safe=False
    )
    # rows of the symmetric buffer; rank r's keys at int64 column width * r
    out = out.view(torch.int64)
    return _reduce_keys(out, rows, world, width, width, out.stride(0))


# Epoch-tagged exchange of one key per row. The low/high halves use two 64-bit
# words whose high halves carry a nonzero 32-bit tag.  A separate cursor rotates
# the three slots, so wrapping the wire tag never reuses the preceding slot.
_LL_SLOTS = 3
_LL_MAX_ROWS = 64
_LL_SPIN_LIMIT = 1 << 26
_ll_state = None


@triton.jit
def _ll_st_b64(addr, val, mask):
    return tl.inline_asm_elementwise(
        """
        {
            .reg .pred %p0;
            setp.eq.s32 %p0, $3, 1;
            @!%p0 bra end;
            multimem.st.relaxed.sys.global.b64 [$1], $2;
            end:
        }
        """,
        "=r,l,l,r",
        args=[addr, val, mask.to(tl.int32)],
        dtype=tl.uint32,
        is_pure=False,
        pack=1,
    )


@triton.jit
def _ll_ld_b64(addr, mask):
    return tl.inline_asm_elementwise(
        """
        {
            .reg .pred %p0;
            setp.eq.s32 %p0, $2, 1;
            mov.u64 $0, 0;
            @!%p0 bra end;
            ld.relaxed.sys.global.u64 $0, [$1];
            end:
        }
        """,
        "=l,l,r",
        args=[addr, mask.to(tl.int32)],
        dtype=tl.uint64,
        is_pure=False,
        pack=1,
    )


@triton.jit
def _ll_trap():
    tl.inline_asm_elementwise("trap;", "=r", [], dtype=tl.int32, is_pure=False, pack=1)


@triton.jit
def _ll_keys_kernel(
    keys_ptr,
    parts,
    rows,
    mc_addr,
    uc_addr,
    epoch_ptr,
    tokens_ptr,
    RANK: tl.constexpr,
    WORLD: tl.constexpr,
    MAXR: tl.constexpr,
    BLOCK_R: tl.constexpr,
    BLOCK_P: tl.constexpr,
    KEY_FLOOR: tl.constexpr,
    SPIN_LIMIT: tl.constexpr,
):
    # The wire format has exactly 32 tag bits.  Reinterpret the signed storage
    # before widening, skip zero (the buffer's initial value), and keep slot
    # rotation independent from tag wrap.  Both counters stay device-owned so
    # CUDA graph replay never introduces a host synchronization.
    last_tag = tl.load(epoch_ptr).to(tl.uint32, bitcast=True).to(tl.uint64)
    tag = last_tag + 1
    tag = tl.where(tag == 4294967296, 1, tag)
    slot = ((tl.load(epoch_ptr + 1) + 1) % 3).to(tl.int64)
    r = tl.arange(0, BLOCK_R)
    p = tl.arange(0, BLOCK_P)
    rmask = r < rows
    k = tl.load(
        keys_ptr + r[:, None] * parts + p[None, :],
        mask=rmask[:, None] & (p[None, :] < parts),
        other=KEY_FLOOR,
    )
    key = tl.max(k, axis=1).to(tl.uint64, bitcast=True)
    tagw = tag << 32
    mc = mc_addr.to(tl.pointer_type(tl.uint64))
    word = ((slot * WORLD + RANK) * MAXR + r) * 2
    _ll_st_b64(mc + word, (key & 0xFFFFFFFF) | tagw, rmask)
    _ll_st_b64(mc + word + 1, (key >> 32) | tagw, rmask)
    s = tl.arange(0, WORLD)
    m2 = rmask[None, :] & (s[:, None] < WORLD)
    uc = uc_addr.to(tl.pointer_type(tl.uint64))
    words = ((slot * WORLD + s[:, None]) * MAXR + r[None, :]) * 2
    # a CTA barrier before each block-wide count of the poll: without them some warps left it
    # on a stale count and took words a slot held from an earlier launch
    tl.debug_barrier()
    lo = _ll_ld_b64(uc + words, m2)
    hi = _ll_ld_b64(uc + words + 1, m2)
    ok = (((lo >> 32) == tag) & ((hi >> 32) == tag)) | (~m2)
    pending = tl.sum(1 - ok.to(tl.int32))
    spins = 0
    while pending > 0:
        tl.debug_barrier()
        lo = _ll_ld_b64(uc + words, m2)
        hi = _ll_ld_b64(uc + words + 1, m2)
        ok = (((lo >> 32) == tag) & ((hi >> 32) == tag)) | (~m2)
        pending = tl.sum(1 - ok.to(tl.int32))
        spins += 1
        if spins > SPIN_LIMIT:
            _ll_trap()
    got = (((hi & 0xFFFFFFFF) << 32) | (lo & 0xFFFFFFFF)).to(tl.int64, bitcast=True)
    best = tl.max(tl.where(m2, got, KEY_FLOOR), axis=0)
    tl.store(tokens_ptr + r, 4294967295 - (best & 4294967295), mask=rmask)
    tl.store(epoch_ptr, tag.to(tl.uint32).to(tl.int32, bitcast=True))
    tl.store(epoch_ptr + 1, slot.to(tl.int32))


def _ll_build(tp_group, device):
    """The LL exchange's symmetric buffer (zeroed before any peer can write it) and epoch, or
    False (the other exchanges keep the call) when it has no multicast mapping."""
    import torch.distributed as dist
    import torch.distributed._symmetric_memory as symm_mem

    world = tp_group.world_size
    try:
        with torch.inference_mode(False), torch.no_grad():
            buf = symm_mem.empty(
                (_LL_SLOTS * world * _LL_MAX_ROWS * 2,),
                dtype=torch.int64,
                device=device,
            )
        hdl = symm_mem.rendezvous(buf, group=tp_group.device_group)
    except Exception as e:
        logger.warning("Sharded verify LL setup failed; trying other exchanges: %s", e)
        return False
    if hdl.multicast_ptr == 0:
        logger.warning("Sharded verify LL exchange has no multicast mapping")
        return False
    buf.zero_()
    torch.cuda.synchronize(device)
    dist.barrier(group=tp_group.cpu_group)
    # [last nonzero wire tag bits, independent slot cursor]
    epoch = torch.zeros(2, dtype=torch.int32, device=device)
    return (buf, hdl, epoch)


def _ll_keys(keys: torch.Tensor, tp_group) -> Optional[torch.Tensor]:
    """[rows] or [rows, parts] int64 keys of every rank -> [rows] token ids in one launch, or None
    when the LL exchange is unavailable."""
    global _ll_state
    if _ll_state is None:
        _ll_state = _ll_build(tp_group, keys.device)
    if _ll_state is False:
        return None
    buf, hdl, epoch = _ll_state
    rows = keys.shape[0]
    parts = keys.shape[1] if keys.dim() == 2 else 1
    tokens = torch.empty(rows, dtype=torch.int64, device=keys.device)
    _ll_keys_kernel[(1,)](
        keys,
        parts,
        rows,
        hdl.multicast_ptr,
        hdl.buffer_ptrs[hdl.rank],
        epoch,
        tokens,
        RANK=hdl.rank,
        WORLD=tp_group.world_size,
        MAXR=_LL_MAX_ROWS,
        BLOCK_R=max(16, triton.next_power_of_2(rows)),
        BLOCK_P=max(2, triton.next_power_of_2(parts)),
        KEY_FLOOR=_KEY_FLOOR,
        SPIN_LIMIT=_LL_SPIN_LIMIT,
        num_warps=4,
    )
    return tokens


def sharded_target_tokens(
    logits: torch.Tensor,
    shard: VocabShard,
    tp_group,
) -> torch.Tensor:
    """[rows] token ids over the whole vocabulary from every rank's logits shard."""
    # The final TP rank may have fewer valid vocabulary columns than its padded
    # local LM-head output.  Every rank must nevertheless enter the collective
    # with the same key shape, so derive the split count from the rank-uniform
    # padded local width; ``shard.valid`` still masks the kernel's tail.
    splits = key_splits(logits.shape[-1])
    keys = shard_keys(logits, shard, splits)
    return reduce_sharded_token_keys(keys, tp_group, splits=splits)


def reduce_sharded_token_keys(
    keys: torch.Tensor,
    tp_group,
    *,
    splits: int,
) -> torch.Tensor:
    """Reduce provider-neutral local token keys to one global token per row.

    ``keys`` is contiguous int64 with shape ``[rows, splits]`` (or ``[rows]``
    for one split). Larger keys win. Their high 32 bits encode a monotone score;
    their low 32 bits are ``0xffffffff - global_token_id``, so ties pick the
    smaller token exactly as ``torch.argmax`` does. Runtime owns the
    LL/multimem/NCCL exchange while callers own the numeric policy that produced
    each local key.
    """
    if splits <= 0:
        raise ValueError(f"splits must be positive, got {splits}")
    if keys.dtype != torch.int64:
        raise TypeError(f"keys must have dtype torch.int64, got {keys.dtype}")
    if not keys.is_contiguous():
        raise ValueError("keys must be contiguous")
    if splits == 1 and keys.ndim != 1:
        raise ValueError(
            f"one split: keys shape {tuple(keys.shape)} is incompatible with splits=1"
        )
    if splits > 1 and (keys.ndim != 2 or keys.shape[1] != splits):
        raise ValueError(
            f"keys shape {tuple(keys.shape)} is incompatible with splits={splits}"
        )
    if 0 < keys.shape[0] <= _LL_MAX_ROWS and tp_group.world_size <= 16:
        tokens = _ll_keys(keys, tp_group)
        if tokens is not None:
            return tokens
    if 0 < keys.shape[0] <= _MC_MAX_ROWS:
        tokens = _mc_gather_keys(keys, tp_group)
        if tokens is not None:
            return tokens
    gathered = torch.empty(
        tp_group.world_size * keys.numel(), dtype=torch.int64, device=keys.device
    )
    tp_group.all_gather_into_tensor(gathered, keys.view(-1))
    return keys_to_tokens(gathered, tp_group.world_size, splits)


def gather_full_logits(
    logits: torch.Tensor, shard: VocabShard, tp_group
) -> torch.Tensor:
    """The FP32 [rows, vocab_size] logits the stock verify builds: all-gather, slice
    to the vocabulary, cast."""
    tp, rows = tp_group.world_size, logits.shape[0]
    # all_gather_into_tensor, not all_gather: the latter allocates in symmetric
    # memory, which the post-verify graph capture cannot record
    out = torch.empty(
        (tp,) + tuple(logits.shape), dtype=logits.dtype, device=logits.device
    )
    tp_group.all_gather_into_tensor(out.view(-1), logits.contiguous().view(-1))
    full = out.movedim(0, 1).reshape(rows, -1)
    return full[:, : shard.vocab_size].float()
