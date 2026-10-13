# SPDX-License-Identifier: Apache-2.0
"""The K2i launch's host side: persistent buffers (graph safe: every tensor a
replayed launch reads or writes lives here), the per-width builds, and the
per-step walk plans.

Two uses: ``launch`` runs ATOM's whole indexer (scores + the top-k trees, the
unit tests' reference build); ``score`` / ``score_reindex`` run the scorer
alone and leave the selection to SGLang's top-k kernels (``SGLANG_MONO_DSV41_K2A=1``)."""

import torch

from . import index_plan as ip
from .candidate_table import candidate_block_table, reindex_table
from .k2i import TL_POINTS, K2iBuild, build_k2i, scratch_bytes
from .walk_plan import BLOCKS, fill_walk_plan_device, new_plan

MAX_ROWS = 48
PLANE_ROWS = 64  # SGLang's FP4 index plane page
COMPACT = ip.TOPK_BLOCKS * ip.BLOCK_ROWS  # a REINDEX row's candidate columns


def _roundup(x: int, m: int) -> int:
    return -(-x // m) * m


class K2iRunner:
    """One per device: buffers sized for MAX_ROWS rows and columns up to ``bound_max``."""

    def __init__(self, device, bound_max: int, timeline: bool = False):
        self.device = torch.device(device)
        self.bound_max = _roundup(max(bound_max, COMPACT), ip.CHUNK)
        self.lstride = self.bound_max
        self.bstride = _roundup(ip.blocks(self.bound_max), 64)
        z = dict(dtype=torch.float32, device=self.device)
        # a level-0 select reads whole SEG quads: a row's last segment may run past it
        self.ilog = torch.empty(MAX_ROWS * self.lstride + ip.SEG, **z)
        self.ibmax = torch.empty(MAX_ROWS * self.bstride + ip.SEG, **z)
        self.iw = torch.empty(MAX_ROWS, ip.HEADS, **z)
        i32 = dict(dtype=torch.int32, device=self.device)
        # the producer's blocks (its icout), which the step's REINDEX layers read (their icand)
        self.icout = torch.full((MAX_ROWS, ip.TOPK_BLOCKS), -1, **i32)
        self.cand_ids = torch.full((MAX_ROWS, ip.TOPK_BLOCKS), -1, **i32)
        self.cand_table = torch.zeros(MAX_ROWS, ip.TOPK_BLOCKS, **i32)
        self.cand_bound = torch.zeros(MAX_ROWS, **i32)
        # a consumer's compact top-k (SGLang's transform over COMPACT columns: one page, at 0)
        self.compact_pos = torch.empty(MAX_ROWS, ip.TOPK, **i32)
        self.compact_table = torch.zeros(MAX_ROWS, 1, **i32)
        self.plans = {r: new_plan(self.device) for r in (1, 2)}
        self.epoch = torch.zeros(64, **i32)
        self.timeline = timeline
        self.tl = torch.zeros(BLOCKS, TL_POINTS, dtype=torch.int64, device=self.device)
        self._owners = {}
        self._kernels = {}
        self._scratch = {}

    def kernel(self, rows: int, tree: bool = True):
        key = (rows, tree)
        if key not in self._kernels:
            build = K2iBuild(rows, self.bound_max, self.timeline, tree)
            self._kernels[key] = build_k2i(build)
            self._scratch[key] = torch.zeros(
                scratch_bytes(build), dtype=torch.uint8, device=self.device
            )
        return self._kernels[key], self._scratch[key]

    def owners(self, rows: int, per_req: int, table_stride: int):
        """(owners, kmeta): a row's request's first row, and key metadata word 2 = its page-table offset."""
        key = (rows, per_req, table_stride)
        if key not in self._owners:
            first = (
                torch.arange(rows, dtype=torch.int32, device=self.device)
                // per_req
                * per_req
            )
            kmeta = torch.zeros(rows, 4, dtype=torch.int32, device=self.device)
            kmeta[:, 2] = first * table_stride
            self._owners[key] = (first, kmeta)
        return self._owners[key]

    def plan(
        self, ratio: int, bounds: torch.Tensor, owners: torch.Tensor, rows: int
    ) -> torch.Tensor:
        """The step's FULL walk plan of ratio ``ratio``; once a step and ratio (every FULL layer of the ratio reads
        it)."""
        fill_walk_plan_device(bounds, owners, self.plans[ratio], rows)
        return self.plans[ratio]

    def logits(self, rows: int) -> torch.Tensor:
        return self.ilog[: rows * self.lstride].view(rows, self.lstride)

    @staticmethod
    def _check(q_fp4, k_payload, k_scale, page_table, bounds):
        rows = q_fp4.shape[0]
        assert 0 < rows <= MAX_ROWS, rows
        assert q_fp4.is_contiguous() and q_fp4.numel() == rows * ip.HEADS * ip.DIM // 2
        assert k_payload.is_contiguous() and k_scale.is_contiguous()
        assert (
            k_payload.view(torch.uint8).reshape(-1, 4 * PLANE_ROWS * 16).shape[0]
            == k_scale.shape[0]
        )
        assert k_scale.reshape(k_scale.shape[0], -1).shape[1] == 4 * PLANE_ROWS
        assert (
            page_table.dtype == torch.int32
            and page_table.stride(1) == 1
            and page_table.shape[0] >= rows
        )
        assert bounds.dtype == torch.int32 and bounds.is_contiguous()
        return rows

    def _launch(self, rows, tree, *, q_fp4, q_scale, weights, k_payload, k_scale, plan, itab, itab_stride,
                itab_len, ishift, ibound, owners, reindex, producer, kmeta, table, raw_out, page_out, icand, epoch,
                stream):  # fmt: skip
        iw = self.iw[:rows]
        iw.copy_(weights)
        k, scratch = self.kernel(rows, tree)
        k(
            q_fp4.data_ptr(), q_scale.data_ptr(), iw.data_ptr(), k_payload.data_ptr(), k_scale.data_ptr(),
            plan.data_ptr(), itab.data_ptr(), itab_stride, itab_len, owners.data_ptr(), ishift, ibound.data_ptr(),
            self.ilog.data_ptr(), self.lstride, int(reindex), int(producer), self.ibmax.data_ptr(), self.bstride,
            kmeta.data_ptr(), table.data_ptr(), PLANE_ROWS, PLANE_ROWS, 0, raw_out.data_ptr(), page_out.data_ptr(),
            icand.data_ptr(), self.icout.data_ptr(), scratch.data_ptr(),
            (self.epoch if epoch is None else epoch).data_ptr(), self.tl.data_ptr(),
            stream=stream if stream is not None else torch.cuda.current_stream(),
        )  # fmt: skip

    def launch(self, *, q_fp4, q_scale, weights, k_payload, k_scale, page_table, bounds, plan, owners, kmeta,
               raw_out, page_out, kind="full", epoch=None, stream=None):  # fmt: skip
        """ATOM's whole indexer on one layer. ``kind``: "full" (score every visible column), "producer" (full, and
        its top-2048 8-row blocks into ``self.icout``), "reindex" (score only the producer's blocks)."""
        assert kind in ("full", "producer", "reindex"), kind
        rows = self._check(q_fp4, k_payload, k_scale, page_table, bounds)
        assert (
            raw_out.stride(0) == page_out.stride(0) == ip.TOPK
            and raw_out.dtype == page_out.dtype == torch.int32
        )
        if kind == "reindex":
            candidate_block_table(self.icout[:rows], page_table, owners, PLANE_ROWS // ip.BLOCK_ROWS, bounds,
                                  self.cand_table, self.cand_bound, rows_per_block=ip.BLOCK_ROWS)  # fmt: skip
            itab, itab_stride, itab_len, ishift, ibound = (self.cand_table, ip.TOPK_BLOCKS, ip.TOPK_BLOCKS, 0,
                                                           self.cand_bound)  # fmt: skip
        else:
            # a page-table entry is 8 virtual 8-row pages
            itab, itab_stride, itab_len, ishift, ibound = (
                page_table,
                page_table.stride(0),
                page_table.shape[1],
                3,
                bounds,
            )
        self._launch(rows, True, q_fp4=q_fp4, q_scale=q_scale, weights=weights, k_payload=k_payload,
                     k_scale=k_scale, plan=plan, itab=itab, itab_stride=itab_stride, itab_len=itab_len,
                     ishift=ishift, ibound=ibound, owners=owners, reindex=kind == "reindex",
                     producer=kind == "producer", kmeta=kmeta, table=page_table, raw_out=raw_out,
                     page_out=page_out, icand=self.icout, epoch=epoch, stream=stream)  # fmt: skip

    def score(self, *, q_fp4, q_scale, weights, k_payload, k_scale, page_table, bounds, plan, owners, kmeta,
              epoch=None, stream=None) -> torch.Tensor:  # fmt: skip
        """A FULL layer's logits, [rows, lstride] f32, valid below each row's bound."""
        rows = self._check(q_fp4, k_payload, k_scale, page_table, bounds)
        self._launch(rows, False, q_fp4=q_fp4, q_scale=q_scale, weights=weights, k_payload=k_payload,
                     k_scale=k_scale, plan=plan, itab=page_table, itab_stride=page_table.stride(0),
                     itab_len=page_table.shape[1], ishift=3, ibound=bounds, owners=owners, reindex=False,
                     producer=False, kmeta=kmeta, table=page_table, raw_out=self.cand_bound,
                     page_out=self.cand_bound, icand=self.icout, epoch=epoch, stream=stream)  # fmt: skip
        return self.logits(rows)

    def score_reindex(self, *, q_fp4, q_scale, weights, k_payload, k_scale, page_table, bounds, owners, kmeta,
                      cand_ids, epoch=None, stream=None):  # fmt: skip
        """A candidate consumer's logits on its candidates only: (compact logits [rows, COMPACT] f32 -- column
        j * 8 + t is row t of block ``ids[j]`` -- valid below ``lens``, lens [rows], ids [rows, TOPK_BLOCKS]: SGLang's
        ``cand_ids`` with the newest block moved last)."""
        rows = self._check(q_fp4, k_payload, k_scale, page_table, bounds)
        assert (
            cand_ids.shape == (rows, ip.TOPK_BLOCKS) and cand_ids.dtype == torch.int32
        )
        ids, lens = self.cand_ids[:rows], self.cand_bound[:rows]
        reindex_table(cand_ids, page_table, owners, PLANE_ROWS // ip.BLOCK_ROWS, bounds, ids, self.cand_table, lens,
                      rows_per_block=ip.BLOCK_ROWS)  # fmt: skip
        self._launch(rows, False, q_fp4=q_fp4, q_scale=q_scale, weights=weights, k_payload=k_payload,
                     k_scale=k_scale, plan=self.plans[1], itab=self.cand_table, itab_stride=ip.TOPK_BLOCKS,
                     itab_len=ip.TOPK_BLOCKS, ishift=0, ibound=lens, owners=owners, reindex=True, producer=False,
                     kmeta=kmeta, table=page_table, raw_out=lens, page_out=lens, icand=self.icout, epoch=epoch,
                     stream=stream)  # fmt: skip
        return self.logits(rows)[:, :COMPACT], lens, ids
