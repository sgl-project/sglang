# SPDX-License-Identifier: Apache-2.0
"""Target sampling of an EAGLE verify on the vocabulary-sharded LM head output
(SGLANG_Q35MF_SHARDED_VERIFY): every rank reduces its [rows, V / tp] shard to one int64
key per row, an all-gather of the keys and a max pick the token. Greedy rows match
torch.argmax over the gathered FP32 logits bit for bit; temperature-only rows draw a
Gumbel-max sample, the same distribution as softmax(logits / T)."""

from __future__ import annotations

from typing import Optional

import msgspec
import torch
import triton
import triton.language as tl

from sglang.srt.utils.common import print_info_once
from sglang.srt.environ import envs

_MC_GATHER = envs.SGLANG_PERFLL_VERIFY_MC_GATHER.get()
_SPLIT = envs.SGLANG_PERFLL_VERIFY_KEY_SPLIT.get()
_LL_KEYS = envs.SGLANG_PERFLL_VERIFY_LL_KEYS.get()


class VocabShard(msgspec.Struct, frozen=True):
    """Columns [0, valid) of this rank's logits are global token ids start + j."""

    start: int
    valid: int
    vocab_size: int


# key of (-inf, token id 0xFFFFFFFF): below every real column's key
_KEY_FLOOR = -2139095041 << 32


@triton.jit
def _shard_key_kernel(
    logits_ptr,
    stride_row,
    temp_ptr,
    seed_ptr,
    keys_ptr,
    start,
    valid,
    vocab_size,
    span,
    KEY_FLOOR: tl.constexpr,
    HAS_TEMP: tl.constexpr,
    BLOCK: tl.constexpr,
):
    # program (row, part): columns [part * span, (part + 1) * span) of the row's shard
    row = tl.program_id(0).to(tl.int64)
    part = tl.program_id(1)
    lo = part * span
    hi = tl.minimum(lo + span, valid)
    best = tl.full([BLOCK], KEY_FLOOR, tl.int64)
    if HAS_TEMP:
        temp = tl.load(temp_ptr + row)
        inv_t = 1.0 / tl.maximum(temp, 1e-20)
        seed = tl.load(seed_ptr + row)
    for off in range(lo, hi, BLOCK):
        cols = off + tl.arange(0, BLOCK)
        gid = start + cols
        mask = (cols < hi) & (gid < vocab_size)
        x = tl.load(
            logits_ptr + row * stride_row + cols, mask=mask, other=float("-inf")
        )
        x = x.to(tl.float32)
        if HAS_TEMP:
            if temp > 0.0:
                u = tl.rand(seed, gid)
                x = x * inv_t - tl.log(-tl.log(u))
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
    """SGLANG_PERFLL_VERIFY_KEY_SPLIT: programs a row, one per _KEY_BLOCK columns of the
    shard, even (a row of keys is whole 16-byte multimem chunks)."""
    n = -(-valid // _KEY_BLOCK)
    return n + (n & 1)


def shard_keys(
    logits: torch.Tensor,
    shard: VocabShard,
    temperatures: Optional[torch.Tensor] = None,
    seeds: Optional[torch.Tensor] = None,
    splits: int = 1,
) -> torch.Tensor:
    """One int64 key per row of this rank's shard: (order-preserving FP32 bits of the
    row's best value) << 32 | (0xFFFFFFFF - token id). temperatures: [rows] float32;
    rows at T <= 0 take the plain logit, rows at T > 0 logit / T + Gumbel noise drawn
    from seeds [rows] int64 and the global token id. splits > 1: [rows, splits], the
    best key of each of `splits` column ranges (the row's key is their max)."""
    assert logits.stride(-1) == 1
    rows = logits.shape[0]
    keys = torch.empty(
        (rows, splits) if splits > 1 else (rows,),
        dtype=torch.int64,
        device=logits.device,
    )
    if rows == 0:
        return keys
    has_temp = temperatures is not None
    _shard_key_kernel[(rows, splits)](
        logits,
        logits.stride(0),
        temperatures if has_temp else keys,
        seeds if has_temp else keys,
        keys,
        shard.start,
        shard.valid,
        shard.vocab_size,
        -(-shard.valid // splits),
        KEY_FLOOR=_KEY_FLOOR,
        HAS_TEMP=has_temp,
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


def keys_to_tokens(gathered: torch.Tensor, tp_size: int, splits: int = 1) -> torch.Tensor:
    """[tp * rows * splits] keys of all ranks (rank-major) -> [rows] int64 token ids."""
    if splits == 1:
        best = gathered.view(tp_size, -1).amax(dim=0)
        return 0xFFFFFFFF - (best & 0xFFFFFFFF)
    rows = gathered.numel() // (tp_size * splits)
    return _reduce_keys(gathered, rows, tp_size, splits, rows * splits, splits)


# SGLANG_PERFLL_VERIFY_MC_GATHER: the key exchange's multimem state, built by the first
# (eager, every rank in lockstep) call; a row of a rank is `width` int64 (4 bf16 each)
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
        print_info_once(f"SGLANG_PERFLL_VERIFY_MC_GATHER: multimem state failed ({e}); NCCL gather")
        return False
    if state.symm_mem_hdl.multicast_ptr == 0:
        print_info_once("SGLANG_PERFLL_VERIFY_MC_GATHER: no multicast mapping; NCCL gather")
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
    if not _SPLIT:
        return 0xFFFFFFFF - (
            out.contiguous().view(torch.int64).view(rows, world, width)[:, :, 0].amax(dim=1)
            & 0xFFFFFFFF
        )
    # rows of the symmetric buffer; rank r's keys at int64 column width * r
    out = out.view(torch.int64)
    return _reduce_keys(out, rows, world, width, width, out.stride(0))


# SGLANG_PERFLL_VERIFY_LL_KEYS: epoch-tagged exchange of one key per row (low / high 32 bits in two
# 64-bit words whose high halves carry the epoch), three slots rotating by epoch: a slot is rewritten
# three calls later, after two more exchanges that each needed every rank's words.
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
    tag = (tl.load(epoch_ptr) + 1).to(tl.uint64)
    slot = (tag % 3).to(tl.int64)
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
    tl.store(epoch_ptr, tag.to(tl.int32))


def _ll_build(tp_group, device):
    """The LL exchange's symmetric buffer (zeroed before any peer can write it) and epoch, or
    False (the other exchanges keep the call) when it has no multicast mapping."""
    import torch.distributed as dist
    import torch.distributed._symmetric_memory as symm_mem

    world = tp_group.world_size
    try:
        with torch.inference_mode(False), torch.no_grad():
            buf = symm_mem.empty(
                (_LL_SLOTS * world * _LL_MAX_ROWS * 2,), dtype=torch.int64, device=device
            )
        hdl = symm_mem.rendezvous(buf, group=tp_group.device_group)
    except Exception as e:
        print_info_once(f"SGLANG_PERFLL_VERIFY_LL_KEYS: symmetric buffer failed ({e}); not used")
        return False
    if hdl.multicast_ptr == 0:
        print_info_once("SGLANG_PERFLL_VERIFY_LL_KEYS: no multicast mapping; not used")
        return False
    buf.zero_()
    torch.cuda.synchronize(device)
    dist.barrier(group=tp_group.cpu_group if hasattr(tp_group, "cpu_group") else tp_group.device_group)
    epoch = torch.zeros(1, dtype=torch.int32, device=device)
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
    temperatures: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """[rows] token ids over the whole vocabulary from every rank's logits shard."""
    print_info_once(
        "SGLANG_Q35MF_SHARDED_VERIFY: target "
        + ("Gumbel-max" if temperatures is not None else "argmax")
        + " on the vocabulary shards"
        + (" (keys split, fused reduce)" if _SPLIT else "")
        + (" (multimem key gather)" if _MC_GATHER else "")
        + (" (LL key exchange <= 64 rows)" if _LL_KEYS else "")
    )
    seeds = None
    if temperatures is not None:
        seeds = torch.randint(
            0, 2**31 - 1, (logits.shape[0],), dtype=torch.int64, device=logits.device
        )
    splits = key_splits(shard.valid) if _SPLIT else 1
    keys = shard_keys(logits, shard, temperatures, seeds, splits)
    if _LL_KEYS and 0 < keys.shape[0] <= _LL_MAX_ROWS and tp_group.world_size <= 16:
        tokens = _ll_keys(keys, tp_group)
        if tokens is not None:
            return tokens
    if _MC_GATHER and 0 < keys.shape[0] <= _MC_MAX_ROWS:
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
    print_info_once(
        "SGLANG_Q35MF_SHARDED_VERIFY: a batch needs the full logits, gathered"
    )
    tp, rows = tp_group.world_size, logits.shape[0]
    # all_gather_into_tensor, not all_gather: the latter allocates in symmetric
    # memory, which the post-verify graph capture cannot record
    out = torch.empty(
        (tp,) + tuple(logits.shape), dtype=logits.dtype, device=logits.device
    )
    tp_group.all_gather_into_tensor(out.view(-1), logits.contiguous().view(-1))
    full = out.movedim(0, 1).reshape(rows, -1)
    return full[:, : shard.vocab_size].float()
