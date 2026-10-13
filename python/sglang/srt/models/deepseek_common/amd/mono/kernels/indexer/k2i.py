# SPDX-License-Identifier: Apache-2.0
"""K2i: ATOM K2a's indexer (``score.run_indexer``: the FP4 scorer, the exact
top-512 tree and, on the candidate producer, the top-2048 block tree) as a
launch of its own between K1 and K2 of an indexer layer (task 17 stage 1,
``SGLANG_MONO_DSV41_K2A=1``).

BLOCKS CTAs of THREADS threads, all co-resident (the trees poll the scorer's
flags). Its mailbox regions live in a scratch of its own, tagged with the layer
pair's epoch + 1: K2 moves the epoch on once a layer, so no two K2i launches
share a tag, replayed graphs included.

Arguments (ATOM's names; S rows, int32 unless noted):
  iq [S, 32, 64] E2M1, iqs [S, 4, 16, 4] E8M0, iw [S, 32] f32
  plane / pscale: the layer's FP4 index plane (page 64, ``score_fp4._keys``)
  iplan [BLOCKS, 4]: the FULL walk plan (``walk_plan``)
  itab [rows, itab_len] (row stride itab_stride): a FULL layer's page table,
      each entry 2 ** ishift (8) virtual pages; a REINDEX layer's candidate
      table, an entry a virtual page (ishift 0)
  ibatch [S]: the row whose itab row a FULL piece reads (its request's first)
  ibound [S]: columns a row's scorer reads
  ilog [S, lstride] f32, ibmax [S, bstride] f32: logits, block maxima
  ilift: REINDEX (lift a selection through icand), iprod: candidate producer
  kmeta [S, 4]: word 2 = the row's page-table offset (``_selected_rows``)
  table, rows_per_page, page_rows, main_off: a selected column's pool row
  isel / irow [S, 512]: SGLang's sparse_raw_indices / sparse_page_indices
  icand [S, 2048]: the candidate blocks a REINDEX layer reads, icout [S, 2048]:
      the blocks a producer writes
"""

from dataclasses import dataclass

import flydsl.compiler as flyc
import flydsl.expr as fx
from aiter.ops.flydsl.kernels import buffer_ops as bo
from flydsl.expr import const_expr
from flydsl.expr.typing import Int32, Int64, T

from sglang.srt.layers.mono.common.device.ops import rsrc
from sglang.srt.layers.mono.common.device.stamps import stamp, stamp_begin, stamp_flush
from sglang.srt.layers.mono.common.device.sync import Mailbox, sreg
from sglang.srt.layers.mono.common.plan.execution import BLOCKS, THREADS
from sglang.srt.layers.mono.common.plan.layout import pair_layout

from ..attention.device import kernel_symbol
from ..common.plan import key_tuple
from ..sources import SOURCES
from . import index_plan as ip
from .score import PER_THREAD, run_indexer, scratch_pairs
from .score_fp4 import run_scores4
from .topk import lds_words

_STREAM = fx.Stream(None)
_SEL = lds_words(PER_THREAD)
SEL_KEYS, SEL_HIST, SEL_BUF = _SEL["keys"], _SEL["hist"], _SEL["buf"]

ARGS = (
    "iq", "iqs", "iw", "plane", "pscale", "iplan", "itab", "itab_stride", "itab_len", "ibatch", "ishift",
    "ibound", "ilog", "lstride", "ilift", "iprod", "ibmax", "bstride", "kmeta", "table", "rows_per_page",
    "page_rows", "main_off", "isel", "irow", "icand", "icout",
)  # fmt: skip
I32_ARGS = frozenset(
    ("itab_stride", "itab_len", "ishift", "lstride", "ilift", "iprod", "bstride", "rows_per_page", "page_rows",
     "main_off")
)  # fmt: skip


@dataclass(frozen=True)
class K2iBuild:
    tokens: int  # S: the step's rows
    bound_max: (
        int  # the largest column bound a row's scorer reads (a multiple of CHUNK)
    )
    timeline: bool = (
        False  # per-CTA phase stamps into ``tl`` [BLOCKS, TL_POINTS] int64 (debug)
    )
    tree: bool = True  # False: the scorer alone (logits into ``ilog``; SGLang's top-k kernels select)


TL_POINTS = 16


def scratch_layout(key: K2iBuild) -> dict:
    return pair_layout(scratch_pairs(key.tokens, key.bound_max).items())


def scratch_bytes(key: K2iBuild) -> int:
    return max(o + n for o, n in scratch_layout(key).values())


def build_k2i(key: K2iBuild):
    s, bound_max = key.tokens, key.bound_max
    assert 0 < s <= 48 and bound_max % ip.CHUNK == 0, key
    assert (
        ip.WALK_ROWS * (ip.HEADS * ip.DIM // 8 + ip.HEADS * 3) + ip.WALK_ROWS
        <= _SEL["keys"]
    )
    layout = scratch_layout(key)
    name = kernel_symbol(
        "dsv41_mono_k2i", s=s, bm=bound_max, tree=key.tree, tl=key.timeline
    )
    keyed = key_tuple(key, SOURCES)

    timeline = key.timeline

    @fx.struct
    class SelLds:
        sk: fx.Array[fx.Int32, SEL_KEYS, 16]
        sh: fx.Array[fx.Int32, SEL_HIST, 16]
        sb: fx.Array[fx.Int32, SEL_BUF, 16]
        tls: fx.Array[fx.Int64, TL_POINTS if timeline else 1, 16]

    @fx.union
    class K2iLds:
        sel: SelLds

    @flyc.kernel(name=name, known_block_size=[THREADS, 1, 1])
    def k2i(
        iq: Int64, iqs: Int64, iw: Int64, plane: Int64, pscale: Int64, iplan: Int64, itab: Int64,
        itab_stride: Int32, itab_len: Int32, ibatch: Int64, ishift: Int32, ibound: Int64, ilog: Int64,
        lstride: Int32, ilift: Int32, iprod: Int32, ibmax: Int64, bstride: Int32, kmeta: Int64, table: Int64,
        rows_per_page: Int32, page_rows: Int32, main_off: Int32, isel: Int64, irow: Int64, icand: Int64,
        icout: Int64, scratch: Int64, epoch: Int64, tl: Int64,
    ):  # fmt: skip
        _ = keyed
        bid = fx.block_idx.x
        tid = fx.thread_idx.x
        lds = fx.SharedAllocator().allocate(K2iLds)
        sel = lds.sel.peek()
        mb = Mailbox(fx.Int32(bo.buffer_load(rsrc(epoch), 0, vec_width=1, dtype=T.i32)))
        args = {
            "iq": iq, "iqs": iqs, "iw": iw, "plane": plane, "pscale": pscale, "iplan": iplan, "itab": itab,
            "itab_stride": itab_stride, "itab_len": itab_len, "ibatch": ibatch, "ishift": ishift, "ibound": ibound,
            "ilog": ilog, "lstride": lstride, "ilift": ilift, "iprod": iprod, "ibmax": ibmax, "bstride": bstride,
            "kmeta": kmeta, "table": table, "rows_per_page": rows_per_page, "page_rows": page_rows,
            "main_off": main_off, "isel": isel, "irow": irow, "icand": icand, "icout": icout,
        }  # fmt: skip
        c = {
            "S": s, "tid": tid, "bid": bid, "lane": tid % 64, "wave": tid // 64,
            "put": mb.put, "put_words": mb.put_words, "poll": mb.poll, "args": args,
            "sel_lds": {"keys": sel.sk.ptr, "hist": sel.sh.ptr, "buf": sel.sb.ptr},
        }  # fmt: skip
        for region, (off, _n) in layout.items():
            c[region] = sreg(scratch, off, region)
        tls = sel.tls.ptr
        stamp_begin(timeline, tls, tid, TL_POINTS)
        if const_expr(timeline):
            point = [0]

            def next_stamp():
                point[0] += 1
                assert point[0] < TL_POINTS
                stamp(True, tls, tid, point[0])

            c["stamp"] = next_stamp
        if const_expr(key.tree):
            run_indexer(c, s, bid, bound_max)
        else:
            run_scores4(c, s, bid, ip.chunks(bound_max))
        stamp_flush(timeline, tls, tl, tid, bid, TL_POINTS)

    @flyc.jit
    def launch(
        iq: Int64, iqs: Int64, iw: Int64, plane: Int64, pscale: Int64, iplan: Int64, itab: Int64,
        itab_stride: Int32, itab_len: Int32, ibatch: Int64, ishift: Int32, ibound: Int64, ilog: Int64,
        lstride: Int32, ilift: Int32, iprod: Int32, ibmax: Int64, bstride: Int32, kmeta: Int64, table: Int64,
        rows_per_page: Int32, page_rows: Int32, main_off: Int32, isel: Int64, irow: Int64, icand: Int64,
        icout: Int64, scratch: Int64, epoch: Int64, tl: Int64, stream: fx.Stream = _STREAM,
    ):  # fmt: skip
        _ = keyed
        k2i(
            iq, iqs, iw, plane, pscale, iplan, itab, itab_stride, itab_len, ibatch, ishift, ibound, ilog, lstride,
            ilift, iprod, ibmax, bstride, kmeta, table, rows_per_page, page_rows, main_off, isel, irow, icand,
            icout, scratch, epoch, tl,
        ).launch(grid=(BLOCKS,), block=(THREADS,), stream=stream)  # fmt: skip

    return launch
