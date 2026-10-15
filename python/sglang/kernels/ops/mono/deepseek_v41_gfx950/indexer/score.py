# SPDX-License-Identifier: MIT
# Vendored from ROCm/ATOM @ 68506823: atom/models/deepseek_v41/mono/kernels/index_score.py
# (the selection trees and the FP4 scorer's driver; no FP8 scorer)
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""K2a's indexer (a layer that selects: FULL, and REINDEX within layer 20's
candidates): the scores of every visible index row, the top-512 of them and, on
the candidate-producing layer, the top-2048 blocks -- the answer of the
original ``score_topk_paged`` (logits: gluon ``deepgemm_fp8_paged_mqa_logits``;
selection: ``top_k_per_row_decode(stable=True)``; blocks: ``pick_candidate_blocks``)
at any context length. Which cells each stage writes and polls is decided per
step only in ``index_plan`` (DESIGN_v3 4.9); the CPU test there checks it.

    iscore  ``score_fp4.run_scores4`` (the row-group scorer's arithmetic) ->
            global ``ilog``; a candidate producer's 8-row block maxima (NaN
            ignored, capped at the largest finite, the newest block +inf) ->
            global ``ibmax``; then the flags (ISF) of its (token, chunk)s
    ilist   (a level's lists, per token): ``topk.lds_topk`` of a segment's
            values (level 0, after its chunks' flags) or of the lists it merges
            (level l > 0, from ISEL<l-1> / IBLK<l-1>), in index order. A list
            that is its token's last -> global (``isel``, lifted to real rows on
            a REINDEX layer; ``icout``) and, for the selection, the ready flag
            (IRDY) the attention waits on; else -> ISEL<l> / IBLK<l>.

Data too large for tagged pairs (the logits, up to max_model_len a token) goes
through global memory at device scope (SC1: past the per-XCD L2): stored,
``s_waitcnt vmcnt(0)``, then the flag: a task's drain and barrier, hence
several chunks a task.
"""

import flydsl.expr as fx
from aiter.ops.flydsl.kernels import buffer_ops as bo
from flydsl.expr import const_expr, gpu, range_constexpr
from flydsl.expr.typing import Int32, T

from sglang.kernels.ops.mono.common_hip.device.ops import (
    CM_DEV,
    ld_dev,
    ld_i32,
    rsrc,
    traced,
)
from sglang.kernels.ops.mono.common_hip.device.sync import publish
from sglang.kernels.ops.mono.common_hip.plan.arith import pick
from sglang.kernels.ops.mono.common_hip.plan.execution import (
    BLOCKS,
    THREADS,
    first_task,
)
from sglang.kernels.ops.mono.common_hip.plan.trace import enter_stage

from . import index_plan as ip
from .score_fp4 import run_scores4
from .topk import key_bits, lds_topk

TILE = ip.BLOCK_ROWS
KMETA = 4  # ATOM's key metadata words a row; K2i fills word 2 only: the row's page-table offset
PER_THREAD = ip.SEG // THREADS
NEG = float("-inf")
FLT_MAX = 3.4028234663852886e38


def scratch_pairs(tokens: int, bound_max: int) -> dict[str, int]:
    """The indexer's mailbox regions. The last level's list is always a token's
    final one (global), so only levels below it have a region."""
    out = {"isf": tokens * ip.chunks(bound_max), "irdy": tokens}
    for kind, k in ip.KINDS.items():
        for level in range(ip.top_level(kind, bound_max)):
            slots = ip.list_slots(kind, level, bound_max)
            out[f"i{kind}{level}"] = tokens * slots * k * 2
    return out


def _st_dev(v, ptr, i):
    bo.buffer_store(v, rsrc(ptr), i, cache_modifier=CM_DEV)


@traced
def stage_ilist(c, t, j, raw, kind, level, bound_max, max_chunks):
    """List j of ``level`` for token t (``raw`` its column bound); see the
    module. Only the lists ``index_plan.list_count`` gives a token run."""
    tid = c["tid"]
    a = c["args"]
    k = ip.KINDS[kind]
    slots = ip.list_slots(kind, level, bound_max)
    bound = ip.kind_bound(kind, raw)
    if j < ip.list_count(bound, level, k):
        n = ip.list_inputs(bound, level, j, k)
        final = ip.is_final(bound, level, k)
        # the top level has one list, a token's last: it has no region above it
        to_final = True if level == ip.top_level(kind, bound_max) else final
        got = []
        if const_expr(level == 0):
            if const_expr(kind == "sel"):
                first, count = ip.CHUNKS_PER_SEG * j, ip.seg_chunks(bound, j)
            else:
                first, count = ip.block_chunks(raw, j)
            if tid < count:
                c["poll"]([(c["isf"], t * max_chunks + first + tid, 1)])
            gpu.barrier()
            _stamp(c)
        else:
            below = ip.list_slots(kind, level - 1, bound_max)
            base = tid * PER_THREAD
            got = c["poll"](
                [
                    (
                        c[f"i{kind}{level - 1}"],
                        (
                            (t * below + j * ip.fan(k) + fx.min(base + d, n - 1) // k)
                            * k
                            + fx.min(base + d, n - 1) % k
                        )
                        * 2,
                        2,
                    )
                    for d in range(PER_THREAD)
                ]
            )
        src, stride = (
            (a["ilog"], a["lstride"]) if kind == "sel" else (a["ibmax"], a["bstride"])
        )

        def load(q, i):
            if const_expr(level == 0):
                # a quad stays in the row: strides are whole chunks / blocks
                v = fx.Vector(ld_dev(src, t * stride + j * ip.SEG + i, 4))
                return [fx.Int32(v[e]) for e in range(4)]
            return [got[4 * q + e][0] for e in range(4)]

        keys = c["sel_lds"]["keys"]
        first_i = tid * PER_THREAD

        def ident_of(d, i):
            return j * ip.SEG + i if level == 0 else got[d][1]

        # a selection's rows (``_selected_rows``), in flight during the passes
        def selected_rows():
            return [
                _selected_rows(c, t, ident_of(d, fx.min(first_i + d, n - 1)))
                for d in range(PER_THREAD)
            ]

        def emit(pos, d, i, rows):
            if const_expr(d is None):
                ident, bits = fx.Int32(-1), fx.Float32(NEG).bitcast(fx.Int32)
                # SGLang pads sparse_page_indices with -1 too
                picked = (fx.Int32(-1), fx.Int32(-1))
            else:
                ident = ident_of(d, i)
                bits = key_bits(fx.ptr_load(keys + i))
                picked = rows[d] if kind == "sel" else None
            if to_final:
                _final(c, kind, t, pos, ident, picked)
            else:
                c["put_words"](
                    c[f"i{kind}{level}"], ((t * slots + j) * k + pos) * 2, [bits, ident]
                )

        lds_topk(
            n, k, load, emit, c["sel_lds"], tid, PER_THREAD,
            selected_rows if kind == "sel" else None,
        )  # fmt: skip
        if const_expr(kind == "sel"):  # noqa: SIM102
            if final:
                publish(c["put"], c["irdy"], t, 1, tid == 0)


def _selected_rows(c, t, ident):
    """Column ``ident`` of token t as a selection holds it (a REINDEX layer's
    compacted column lifted back to the row it stands for) and as the
    attention reads it: its pool row, through the block table (``attention.
    _key_row``'s arithmetic; every layer sharing this selection shares its KV
    owner, so its row). Loaded for every element before the top-k
    (``stage_ilist``): inside the winners' branches each load was a round trip
    in series, and the attention's keys wait on none."""
    a = c["args"]
    cand = fx.Int32(
        bo.buffer_load(
            rsrc(a["icand"]),
            t * ip.TOPK_BLOCKS + fx.max(ident, 0) // TILE,
            vec_width=1,
            dtype=T.i32,
        )
    )
    lifted = cand * TILE + fx.max(ident, 0) % TILE
    row = fx.Int32(((a["ilift"] != 0) & (ident >= 0)).select(lifted, ident))
    toff = fx.Int32(
        bo.buffer_load(rsrc(a["kmeta"]), t * KMETA + 2, vec_width=1, dtype=T.i32)
    )
    rpp = a["rows_per_page"]
    live = fx.max(row, 0)
    page = fx.Int32(
        bo.buffer_load(rsrc(a["table"]), toff + live // rpp, vec_width=1, dtype=T.i32)
    )
    return row, page * a["page_rows"] + a["main_off"] + live % rpp


def _final(c, kind, t, pos, ident, rows=None):
    a = c["args"]
    if const_expr(kind == "sel"):
        row, pool_row = rows
        _st_dev(row, a["isel"], t * ip.TOPK + pos)
        _st_dev(pool_row, a["irow"], t * ip.TOPK + pos)
    else:
        _st_dev(Int32(ident), a["icout"], t * ip.TOPK_BLOCKS + pos)


@traced
def run_indexer(c, s, bid, bound_max):
    """Every indexer stage of a selecting layer, the tasks of each laid out from
    the step's bounds (``index_plan``): only live tasks. The bounds were written
    before this launch: plain loads."""
    a = c["args"]
    max_chunks = ip.chunks(bound_max)
    bounds = [ld_i32(a["ibound"], t) for t in range(s)]
    run_scores4(c, s, bid, max_chunks)
    _stamp(c)
    _run_tree(c, s, bid, "sel", bounds, bound_max, max_chunks)
    if a["iprod"] != 0:
        _run_tree(c, s, bid, "blk", bounds, bound_max, max_chunks)


@traced
def _run_tree(c, s, bid, kind, bounds, bound_max, max_chunks):
    k = ip.KINDS[kind]
    for level in range_constexpr(ip.top_level(kind, bound_max) + 1):
        counts = [ip.list_count(ip.kind_bound(kind, b), level, k) for b in bounds]
        enter_stage(f"k2a.i{kind}{level}")
        for task in range(first_task(bid, 0), ip.flat_total(counts), BLOCKS):
            t, j = ip.flat_task(counts, task)
            stage_ilist(c, t, j, pick(bounds, t), kind, level, bound_max, max_chunks)
        _stamp(c)


def _stamp(c):
    """A timeline build's next phase stamp (K2i ``K2iBuild.timeline``)."""
    if const_expr("stamp" in c):
        c["stamp"]()
