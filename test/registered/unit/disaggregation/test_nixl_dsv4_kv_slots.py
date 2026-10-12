"""NIXL prepared dlists over DeepSeek-V4's unequal-length KV entries.

DeepSeek-V4 registers C4 KV, C4 indexer and C128 KV pools for PD transfer, and
their row counts differ. Every entry must still map page p to its own page p.
"""

import unittest
from types import SimpleNamespace

import numpy as np
import torch

from sglang.srt.disaggregation.nixl.conn import NixlKVManager, num_kv_slots
from sglang.srt.mem_cache.deepseek_v4_memory_pool import (
    DeepSeekV4IndexerPool,
    DeepSeekV4SingleKVPool,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

PAGE = 256
FULL_PAGES = 40


def dsv4_kv_buffers(layers=3):
    """The PD KV entries of a DeepSeek-V4 pool, in wire order."""

    def kv(ratio, size):
        return DeepSeekV4SingleKVPool(
            size,
            PAGE // ratio,
            torch.uint8,
            448,
            64,
            layers,
            "cpu",
            False,
            global_page_size=PAGE,
        ).kv_buffer

    c4_size = FULL_PAGES * PAGE // 4
    indexer = DeepSeekV4IndexerPool(
        c4_size,
        PAGE // 4,
        torch.uint8,
        128,
        layers,
        "cpu",
        False,
        use_fp4_indexer=False,
    )
    return [
        *kv(4, c4_size),
        *indexer.contiguous_page_row_buffers(),
        *kv(128, FULL_PAGES * PAGE // 128),
    ]


def geometry(bufs):
    return (
        [b.data_ptr() for b in bufs],
        [b.nbytes for b in bufs],
        [b[0].nbytes for b in bufs],
    )


class FakeAgent:
    """Prepared dlists expand strided runs to per-block rows, as NIXL does.

    Transfer indices address the expanded descriptors, so the prepared handle
    must be the expanded array.
    """

    def __init__(self):
        self.posted = []

    def prep_xfer_dlist(self, peer_name, descs, mem_kind):
        return NixlKVManager._expand_stride_descs(np.asarray(descs))

    def make_prepped_xfer(self, op, src, src_indices, dst, dst_indices, notif):
        self.posted.append((src[src_indices], dst[dst_indices]))
        return len(self.posted)

    def transfer(self, handle):
        return "PROC"


class NixlDsv4KvSlotsTest(CustomTestCase):
    def setUp(self):
        self.src, self.dst = dsv4_kv_buffers(), dsv4_kv_buffers()
        rows = {b.shape[0] for b in self.src}
        self.assertGreater(len(rows), 1, "fixture must have unequal entries")

    def _manager(self):
        ptrs, lens, items = geometry(self.src)
        mgr = object.__new__(NixlKVManager)
        mgr.kv_args = SimpleNamespace(
            kv_data_ptrs=ptrs,
            kv_data_lens=lens,
            kv_item_lens=items,
            kv_data_mem_kinds=["VRAM"] * len(ptrs),
            gpu_id=0,
            kv_layer_ids=[],
        )
        mgr.pp_size = 1
        mgr.is_mla_backend = True
        mgr.is_hybrid_mla_backend = False
        mgr.attn_tp_size = 1
        mgr.src_mem_kind = "VRAM"
        mgr.prep_handles = {}
        mgr.prep_handles_segment_src = {}
        mgr._num_slots_src = num_kv_slots(lens, items)
        mgr.agent = FakeAgent()
        return mgr

    def _peer(self):
        ptrs, lens, items = geometry(self.dst)
        return SimpleNamespace(
            agent_name="decode",
            gpu_id=1,
            dst_kv_ptrs=ptrs,
            dst_kv_item_lens=items,
            dst_kv_mem_kinds=["VRAM"] * len(ptrs),
            requires_dcp_relayout=False,
            dst_kv_layer_ids=[],
            # What the decode advertises for its own entries.
            dst_num_slots=num_kv_slots(lens, items),
            dst_homogeneous_mem_kind=None,
            kv_xfer_segments=None,
        )

    def test_every_entry_reads_and_writes_its_own_page(self):
        mgr, peer = self._manager(), self._peer()
        mgr.decode_kv_args_table = {"decode": peer}
        mgr._prepare_payload_xfer(peer)
        # The highest page any allocator can hand out fits every entry.
        pages = np.array([1, 17, FULL_PAGES], dtype=np.int32)
        mgr.send_kvcache("decode", pages, peer.dst_kv_ptrs, pages[::-1].copy(), 1, "n")
        ((src_rows, dst_rows),) = mgr.agent.posted
        n = len(self.src)
        for i in range(n):
            for k, page in enumerate(pages):
                s, d = src_rows[i * len(pages) + k], dst_rows[i * len(pages) + k]
                item = self.src[i][0].nbytes
                self.assertEqual(
                    s[0], self.src[i].data_ptr() + int(page) * item, (i, page)
                )
                self.assertEqual(
                    d[0], self.dst[i].data_ptr() + int(pages[::-1][k]) * item, (i, page)
                )

    def test_destination_dlist_stays_inside_every_entry(self):
        mgr, peer = self._manager(), self._peer()
        mgr.decode_kv_args_table = {"decode": peer}
        mgr._prepare_payload_xfer(peer)
        rows = mgr.prep_handles["decode"].reshape(len(self.dst), -1, 3)
        for buf, entry in zip(self.dst, rows):
            end = buf.data_ptr() + buf.nbytes
            self.assertTrue((entry[:, 0] + entry[:, 1] <= end).all())


if __name__ == "__main__":
    unittest.main()
