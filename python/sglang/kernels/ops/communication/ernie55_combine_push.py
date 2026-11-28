# Copyright 2026 Baidu ERNIE Team and The SGLang Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import torch
import triton
from triton.experimental import gluon
from triton.experimental.gluon import language as gl

from sglang.kernels.jit.utils import is_arch_support_pdl
from sglang.kernels.ops.attention.ernie55.swapab_gluon_partial import (
    _is_late,
    _pdl_launch_dependents,
    _pdl_wait,
)

# An all-zero 8-byte atom marks an empty Lamport slot, so a sender stores it
# with bf16 -0 in its first element.
NEG_ZERO = gl.constexpr(0x8000)


@gluon.jit
def _ld_acquire(ptr):
    return gl.inline_asm_elementwise(
        "ld.acquire.gpu.global.b32 $0, [$1];",
        "=r,l",
        [ptr],
        dtype=gl.int32,
        is_pure=False,
        pack=1,
    )


@gluon.jit
def _push(addr, lo, hi, owner):
    # Threads holding a replicated element share its value; only one stores.
    gl.inline_asm_elementwise(
        "{ .reg .pred p; .reg .u32 t; mov.u32 t, %tid.x; setp.eq.u32 p, t, $4; "
        "@p st.relaxed.sys.global.v2.b64 [$1], {$2, $3}; mov.u32 $0, 0; }",
        "=r,l,l,l,r",
        [addr, lo, hi, owner],
        dtype=gl.int32,
        is_pure=False,
        pack=1,
    )


@gluon.jit
def _ld_slot(addr, valid):
    return gl.inline_asm_elementwise(
        "{ .reg .pred q; mov.b64 $0, -1; mov.b64 $1, -1; setp.ne.s32 q, $3, 0; "
        "@q ld.relaxed.sys.global.v2.b64 {$0, $1}, [$2]; }",
        "=l,=l,l,r",
        [addr, valid],
        dtype=(gl.int64, gl.int64),
        is_pure=False,
        pack=1,
    )


@gluon.jit
def _drain_slot(lo, hi, addr, dst, valid):
    # Spin until neither atom is empty, copy out, then clear the slot for reuse.
    return gl.inline_asm_elementwise(
        "{ .reg .pred p, q; .reg .b64 x, y, z; "
        "mov.b64 x, $1; mov.b64 y, $2; mov.b64 z, 0; setp.ne.s32 q, $5, 0; "
        "WAIT: setp.eq.u64 p, x, 0; setp.eq.or.u64 p, y, 0, p; @!p bra DONE; "
        "ld.relaxed.sys.global.v2.b64 {x, y}, [$3]; bra WAIT; "
        "DONE: @q st.global.v2.b64 [$4], {x, y}; @q st.global.v2.b64 [$3], {z, z}; "
        "mov.u32 $0, 0; }",
        "=r,l,l,l,l,r",
        [lo, hi, addr, dst, valid],
        dtype=gl.int32,
        is_pure=False,
        pack=1,
    )


@gluon.jit
def _combine_row(
    PART,
    MAX,
    SUM,
    SINK,
    token,
    h,
    d0,
    NT: gl.constexpr,
    ST: gl.constexpr,
    CH: gl.constexpr,
    NL: gl.constexpr,
    D: gl.constexpr,
    RB: gl.constexpr,
    LAYOUT: gl.constexpr,
):
    """Merge one head's NT tiles over ``[d0, d0 + RB)``, as swapab_combine_local does."""
    t = gl.arange(0, ST, gl.SliceLayout(1, LAYOUT))
    stat = (token * NT + t) * NL + h
    den = gl.load(SUM + stat, t < NT, 0.0, cache_modifier=".cg")
    mx = gl.load(MAX + stat, t < NT, 0.0, cache_modifier=".cg")
    sink = gl.load(SINK + h)
    mx = gl.where(den > 0, mx, -float("inf"))
    m = gl.maximum(gl.max(mx, 0), sink)
    m = gl.where(gl.abs(m) == float("inf"), 0.0, m)
    denominator = gl.sum(den * gl.exp(mx - m), 0) + gl.exp(sink - m)
    keep = (denominator > 0) & (sink != float("inf"))
    inv = gl.where(keep, 1.0 / denominator, 0.0)

    tc = gl.arange(0, CH, gl.SliceLayout(1, LAYOUT))
    c = gl.arange(0, RB, gl.SliceLayout(0, LAYOUT))
    acc = gl.zeros([RB], gl.float32, gl.SliceLayout(0, LAYOUT))
    for c0 in gl.static_range(0, NT, CH):
        tt = c0 + tc
        stat = (token * NT + tt) * NL + h
        den = gl.load(SUM + stat, tt < NT, 0.0, cache_modifier=".cg")
        mx = gl.load(MAX + stat, tt < NT, 0.0, cache_modifier=".cg")
        factor = gl.where(den > 0, gl.exp(mx - m), 0.0)
        vals = gl.load(
            PART + stat.to(gl.int64)[:, None] * D + d0 + c[None, :],
            (tt < NT)[:, None],
            0.0,
            cache_modifier=".cg",
        )
        acc += gl.sum(vals * factor[:, None], 0)
    return acc * inv


@gluon.jit
def _pack(x, RB: gl.constexpr, PACK: gl.constexpr, LANE: gl.constexpr):
    """bf16 slice -> the (lo, hi) 8-byte atoms of each of its 16-byte vectors."""
    bits = gl.convert_layout(x.to(gl.bfloat16), PACK)
    bits = bits.to(gl.uint16, bitcast=True).to(gl.int64)
    even, odd = gl.split(gl.reshape(bits, [RB // 2, 2]))
    even, odd = gl.split(gl.reshape(even | (odd << 16), [RB // 4, 2]))
    atoms = even | (odd << 32)
    atoms = gl.where(atoms == 0, NEG_ZERO, atoms)
    lo, hi = gl.split(gl.reshape(atoms, [RB // 8, 2]))
    return gl.convert_layout(lo, LANE), gl.convert_layout(hi, LANE)


@gluon.jit(do_not_specialize=["RANK"])
def combine_push_gluon(
    PART,
    MAX,
    SUM,
    SINK,
    OUT,
    PEERS,
    RECV,
    EPOCH,
    SYNC,
    EL,
    POS,
    RANK,
    SLOT,
    T,
    W: gl.constexpr,
    NT: gl.constexpr,
    ST: gl.constexpr,
    CH: gl.constexpr,
    NL: gl.constexpr,
    D: gl.constexpr,
    RB: gl.constexpr,
    TPW_T: gl.constexpr,
    C_RATIO: gl.constexpr,
    TOKENS_PER_REQ: gl.constexpr,
):
    """Combine the local heads and Lamport-push them into every rank's slot.

    All programs co-resident: each pushes its share of the ``[T, NL, D]`` rows to
    the ``W`` peers (``RECV`` is this rank's own workspace), then polls its share
    of the received slots into ``OUT`` ``[T, W * NL, D]``, viewed as int64.
    ``EPOCH`` is the push plane's shared phase word.
    """
    NW: gl.constexpr = gl.num_warps()
    tile_layout: gl.constexpr = gl.BlockedLayout(
        [1, 4], [TPW_T, 32 // TPW_T], [1, NW], [1, 0]
    )
    pack_layout: gl.constexpr = gl.BlockedLayout([8], [32], [NW], [0])
    lane_layout: gl.constexpr = gl.BlockedLayout([1], [32], [NW], [0])
    poll_layout: gl.constexpr = gl.BlockedLayout([1, W], [32, 1], [NW, 1], [1, 0])
    ROWV: gl.constexpr = NL * D // 8
    PB: gl.constexpr = 32 * NW
    pid = gl.program_id(0)
    nprog = gl.num_programs(0)
    _pdl_wait()
    epoch = gl.load(EPOCH)
    phase = ((epoch >> 32) & 1) * SLOT * W
    send = phase + RANK.to(gl.int64) * SLOT
    k = gl.arange(0, RB // 8, lane_layout)
    for item in range(pid, T * NL * (D // RB), nprog):
        row = item // (D // RB)
        token = row // NL
        d0 = item % (D // RB) * RB
        if _is_late(EL, POS, token, C_RATIO, TOKENS_PER_REQ):
            # The late partial writes this token's last tile while we run.
            seq = gl.load((SYNC + 2).to(gl.pointer_type(gl.int64)), volatile=True)
            want = (seq >> 20).to(gl.int32) + 1
            while _ld_acquire(SYNC + 8 + token) != want:
                pass
        out = _combine_row(
            PART,
            MAX,
            SUM,
            SINK,
            token,
            row % NL,
            d0,
            NT,
            ST,
            CH,
            NL,
            D,
            RB,
            tile_layout,
        )
        lo, hi = _pack(out, RB, pack_layout, lane_layout)
        off = send + (row * D + d0) * 2 + k.to(gl.int64) * 16
        for i in gl.static_range(W):
            _push(PEERS[i] + off, lo, hi, k % PB)
    _pdl_launch_dependents()

    # Round-robin warps over programs so a small poll domain spreads over SMs.
    s = gl.arange(0, W, gl.SliceLayout(0, poll_layout))[None, :]
    r = gl.arange(0, PB, gl.SliceLayout(1, poll_layout))[:, None]
    lane_vid = (r // 32 * nprog + pid) * 32 + r % 32 + 0 * s
    recv = RECV + phase + s.to(gl.int64) * SLOT
    for v0 in range(0, T * ROWV, nprog * PB):
        vid = v0 + lane_vid
        valid = (vid < T * ROWV).to(gl.int32)
        addr = recv + vid.to(gl.int64) * 16
        dst = OUT + ((vid // ROWV * W + s) * ROWV + vid % ROWV) * 2
        lo, hi = _ld_slot(addr, valid)
        _drain_slot(lo, hi, addr, dst, valid)

    # Each launch's adds sum to 1 << 32 (phase) and 1 << 20 (seq): both move once
    # the whole grid is done.
    gl.barrier()
    first = (pid == 0).to(gl.int64)
    commit = 1 + first * ((first << 32) - nprog)
    gl.atomic_add(EPOCH, commit, sem="relaxed", scope="gpu")
    add = 1 + (pid == 0).to(gl.int64) * ((1 << 20) - nprog)
    gl.atomic_add(
        (SYNC + 2).to(gl.pointer_type(gl.int64)), add, sem="relaxed", scope="gpu"
    )


def swapab_combine_push(comm, partial, maximum, sums, sink, nt, late_sync):
    """Combine the local heads and gather every rank's heads over ``comm``'s push plane.

    ``partial`` is ``[T, nt, n_local, D]`` and the result ``[T, world * n_local, D]``.
    ``late_sync`` is ``(sync, extra_lengths, positions, tokens_per_req,
    compress_ratio)``: tokens with a late tile wait for the tile_mode=2 partial.
    """
    bs, _, num_heads, head_dim = partial.shape
    world_size = comm.world_size
    out = torch.empty(
        (bs, world_size * num_heads, head_dim),
        dtype=torch.bfloat16,
        device=partial.device,
    )
    nbytes = bs * num_heads * head_dim * out.element_size()
    assert nbytes <= comm._max_push_size, (nbytes, comm._max_push_size)
    sync, extra_lengths, positions, tokens_per_req, compress_ratio = late_sync
    lease = comm._lease
    slots = lease.data.push_slots
    num_programs = comm.config.num_push_blocks
    row_block = head_dim
    # Halve the slice while each program still gets at most one slice.
    while row_block > 64 and bs * num_heads * head_dim * 2 // row_block <= num_programs:
        row_block //= 2
    st = triton.next_power_of_2(nt)
    combine_push_gluon[(num_programs,)](
        partial,
        maximum,
        sums,
        sink,
        out.view(torch.int64),
        tuple(slots),
        slots[comm.rank],
        lease.control.epoch,
        sync,
        extra_lengths,
        positions,
        comm.rank,
        lease.capacity.push_slot_bytes,
        bs,
        W=world_size,
        NT=nt,
        ST=st,
        CH=min(st, 8),
        NL=num_heads,
        D=head_dim,
        RB=row_block,
        TPW_T=max(128 // row_block, 1),
        C_RATIO=compress_ratio,
        TOKENS_PER_REQ=tokens_per_req,
        num_warps=max(row_block // 128, 1),
        launch_pdl=is_arch_support_pdl(),
    )
    return out
