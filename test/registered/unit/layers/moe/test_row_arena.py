"""Unit tests for ``RowArena`` (``layers/moe/row_arena.py``): a VMM-backed cache
of fixed-size rows whose physical memory can be handed back to the driver while
every row address stays constant.

Needs a CUDA device with ~150 MB free: a 100 MiB virtual reservation of which
36 MiB are backed, one pinned host row (200 KB) that stands in for the "cold"
rows, and a Triton gather kernel that reads rows through an address table.
  1. geometry: chunk / VA alignment, ``chunks_for_rows`` / ``bytes_to_reach`` /
     ``rows_for_bytes`` before and after backing;
  2. ``ensure_rows`` backs a prefix (driver-free memory drops by exactly the
     mapped bytes); a table gather through GPU and host addresses reads every
     row; a CUDA graph captured against the arena addresses replays after a
     ``shrink_rows`` (memory returns, evicted rows repointed to the host row)
     and after growing back (survivor rows intact); ``close`` returns the
     memory;
  3. ``ArenaOOM`` is raised when a chunk cannot be created (chunk larger than
     the device).
"""

import unittest

import torch

from sglang.srt.layers.moe.row_arena import ArenaOOM, RowArena
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=20, stage="base-b", runner_config="1-gpu-small")

ROW_BYTES, E, S = 204800, 512, 184  # a w13 int2 expert row; 184 resident of 512
SLACK = 8 << 20  # allocator noise allowed on the driver-free deltas


def _gather_kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def _gather(tab_ptr, idx_ptr, out_ptr, row_bytes, BLOCK: tl.constexpr):
        r = tl.program_id(0)
        e = tl.load(idx_ptr + r)
        src = tl.load(tab_ptr + e).to(tl.pointer_type(tl.uint8))
        for off in range(0, row_bytes, BLOCK):
            o = off + tl.arange(0, BLOCK)
            m = o < row_bytes
            tl.store(
                out_ptr + r * row_bytes + o, tl.load(src + o, mask=m, other=0), mask=m
            )

    return _gather


@unittest.skipUnless(torch.cuda.is_available(), "RowArena needs the CUDA driver API")
class TestRowArena(CustomTestCase):
    def test_geometry(self):
        a = RowArena(ROW_BYTES, E, 0, name="geom")
        try:
            self.assertEqual(a.chunk % a.gran, 0)
            self.assertGreaterEqual(a.chunk, 4 << 20)
            self.assertEqual(a.va_bytes % a.chunk, 0)
            self.assertGreaterEqual(a.va_bytes, ROW_BYTES * E)
            self.assertEqual(a.backed_rows, 0)
            self.assertEqual(a.chunks_for_rows(0), 0)
            self.assertEqual(a.chunks_for_rows(1), 1)
            self.assertEqual(a.chunks_for_rows(E), -(-(E * ROW_BYTES) // a.chunk))
            self.assertEqual(a.chunks_for_rows(E + 100), a.chunks_for_rows(E))
            self.assertEqual(a.bytes_to_reach(S), a.chunks_for_rows(S) * a.chunk)
            self.assertEqual(a.rows_for_bytes(a.chunk), min(E, a.chunk // ROW_BYTES))
            self.assertEqual(a.row_addr(7), a.base + 7 * ROW_BYTES)
            added = a.ensure_rows(S)
            self.assertEqual(added, a.chunks_for_rows(S) * a.chunk)
            self.assertGreaterEqual(a.backed_rows, S)
            self.assertEqual(a.bytes_to_reach(S), 0)
            self.assertEqual(a.ensure_rows(S), 0)  # idempotent
            self.assertEqual(
                a.bytes_to_reach(E),
                (a.chunks_for_rows(E) - a.chunks_for_rows(S)) * a.chunk,
            )
            with self.assertRaises(AssertionError):
                a.view(a.backed_rows + 1, (ROW_BYTES,))
        finally:
            a.close()

    def test_ensure_shrink_grow_with_graph_replay(self):
        gather = _gather_kernel()
        free0 = torch.cuda.mem_get_info()[0]
        a = RowArena(ROW_BYTES, E, 0, name="w13")
        try:
            added = a.ensure_rows(S)
            free1 = torch.cuda.mem_get_info()[0]
            self.assertGreaterEqual(a.backed_rows, S)
            self.assertLess(abs((free0 - free1) - added), SLACK)

            v = a.view(S, (ROW_BYTES,))
            for i in range(S):
                v[i].fill_(i % 251)
            # one "cold" row on the host stands in for every evicted expert
            host = torch.full((1, ROW_BYTES), 7, dtype=torch.uint8).pin_memory()
            tab = torch.empty(E, dtype=torch.int64)
            for e in range(E):
                tab[e] = a.row_addr(e) if e < S else host.data_ptr()
            tab = tab.cuda()
            idx = torch.tensor([0, 5, 183, 184, 511, 63, 64], dtype=torch.int64).cuda()
            out = torch.empty(len(idx), ROW_BYTES, dtype=torch.uint8, device="cuda")

            def run():
                gather[(len(idx),)](tab, idx, out, ROW_BYTES, BLOCK=4096)

            run()
            torch.cuda.synchronize()
            exp = [i % 251 if i < S else 7 for i in idx.tolist()]
            self.assertEqual(out[:, 0].tolist(), exp)
            self.assertEqual(out[:, -1].tolist(), exp)

            # CUDA graph captured against the arena addresses
            g = torch.cuda.CUDAGraph()
            s = torch.cuda.Stream()
            with torch.cuda.stream(s):
                run()
            torch.cuda.synchronize()
            with torch.cuda.graph(g, stream=s):
                run()
            torch.cuda.synchronize()

            # shrink to 64 rows: memory returns, addresses stay
            free_a = torch.cuda.mem_get_info()[0]
            freed = a.shrink_rows(64)
            free_b = torch.cuda.mem_get_info()[0]
            self.assertGreater(freed, 0)
            self.assertGreaterEqual(a.backed_rows, 64)
            self.assertLess(a.backed_rows, S)
            self.assertLess(abs((free_b - free_a) - freed), SLACK)
            # repoint evicted rows to the host, replay the graph
            tab_cpu = tab.cpu()
            for e in range(64, S):
                tab_cpu[e] = host.data_ptr()
            tab.copy_(tab_cpu)
            g.replay()
            torch.cuda.synchronize()
            exp2 = [i % 251 if i < 64 else 7 for i in idx.tolist()]
            self.assertEqual(out[:, 0].tolist(), exp2)

            # grow back, refill, replay: survivors intact
            a.ensure_rows(S)
            v = a.view(S, (ROW_BYTES,))
            for i in range(64, S):
                v[i].fill_(i % 251)
            for e in range(64, S):
                tab_cpu[e] = a.row_addr(e)
            tab.copy_(tab_cpu)
            g.replay()
            torch.cuda.synchronize()
            self.assertEqual(out[:, 0].tolist(), exp)
            self.assertEqual(v[:64, 0].tolist(), [i % 251 for i in range(64)])
            del g
            # close returns every backed chunk to the driver
            torch.cuda.synchronize()
            backed = a.backed_bytes
            free_c = torch.cuda.mem_get_info()[0]
        finally:
            a.close()
        self.assertEqual(a.backed_rows, 0)
        self.assertLess(abs((torch.cuda.mem_get_info()[0] - free_c) - backed), SLACK)

    def test_oom_raises_arena_oom(self):
        total = torch.cuda.mem_get_info()[1]
        # a single chunk larger than the whole device: cuMemCreate must fail
        big = RowArena(1 << 20, 4, 0, chunk_bytes=total + (1 << 30), name="oom-probe")
        try:
            with self.assertRaises(ArenaOOM):
                big.ensure_rows(1)
            self.assertEqual(big.backed_rows, 0)
        finally:
            big.close()


if __name__ == "__main__":
    unittest.main()
