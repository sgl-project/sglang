"""Pure-logic unit tests for the strict bit-exact SWA HiCache feature.

Covers the strict bit-exact SWA HiCache logic:
  * sizing: hybrid_pool_assembler._swa_host_num_pages and its host-DRAM budget
  * startup guards: the write-through requirement and the temporary gate that
    refuses to boot while the capture/restore halves are unwired
  * offload geometry: DeepSeekV4TokenToKVPool.swa_region_buffers page unit

No GPU / model is required; heavy collaborators are faked so we exercise only
the new logic.
"""

import math
import types
import unittest

from sglang.srt.mem_cache import unified_radix_cache as R
from sglang.srt.mem_cache.hybrid_cache import hybrid_pool_assembler as A
from sglang.srt.mem_cache.unified_cache.components import ComponentType
from sglang.test.ci.ci_register import register_amd_ci

register_amd_ci(est_time=30, suite="stage-b-test-1-gpu-small-amd")

FULL = R.BASE_COMPONENT_TYPE
SWA = ComponentType.SWA


def _sargs(stride=1):
    return types.SimpleNamespace(hicache_swa_offload_page_stride=stride)


class TestSwaHostSizing(unittest.TestCase):
    """Stride model: SWA host pool == ceil(full_host_pages / stride)
    + a device-ring-bounded tail allowance."""

    def _pages(
        self,
        *,
        stride=1,
        full_host_pages=100_000,
        device_ring_pages=65,
        page_bytes=1,
    ):
        return A._swa_host_num_pages(
            server_args=_sargs(stride),
            full_host_pages=full_host_pages,
            device_ring_pages=device_ring_pages,
            page_bytes=page_bytes,
        )

    def test_stride1_covers_all_pages(self):
        # stride 1 == per-page: one window per full page + tail allowance.
        self.assertEqual(self._pages(stride=1), 100_000 + 65)

    def test_larger_stride_shrinks_pool(self):
        # coarser stride -> fewer windows -> smaller SWA pool.
        self.assertLess(self._pages(stride=8), self._pages(stride=1))
        self.assertEqual(self._pages(stride=8), math.ceil(100_000 / 8) + 65)

    def test_tail_allowance_bounded_by_device_ring(self):
        # a huge stride collapses the strided part to a single window; the pool
        # is then just that window plus the device-ring tail allowance.
        self.assertEqual(self._pages(stride=10_000_000, device_ring_pages=65), 1 + 65)

    def test_floor_one_page(self):
        self.assertEqual(
            self._pages(stride=10_000_000, device_ring_pages=0, full_host_pages=1),
            1,
        )

    def test_no_84gb_regression(self):
        # A coarse stride keeps the pool a small fraction of the full host pool
        # -- NOT device_ring * ratio which over-allocated to ~84GB.
        pages = self._pages(stride=64, full_host_pages=100_000)
        self.assertLess(pages, 100_000 * 0.02)
        self.assertEqual(pages, math.ceil(100_000 / 64) + 65)

    def test_warn_above_16gb_but_no_clamp(self):
        # page_bytes chosen so the result exceeds the 16GB slow-launch threshold.
        expected = 100_000 + 65  # stride 1
        page_bytes = int(16e9 / expected) + 1_000_000  # push over 16GB
        with self.assertLogs(A.logger, level="WARNING") as cm:
            pages = self._pages(
                stride=1, full_host_pages=100_000, page_bytes=page_bytes
            )
        self.assertEqual(pages, expected)  # warned, not clamped
        self.assertTrue(any("may slow server launch" in m for m in cm.output))

    def test_no_warn_below_16gb(self):
        # Small page_bytes -> comfortably under threshold -> no warning emitted.
        with self.assertRaises(AssertionError):
            with self.assertLogs(A.logger, level="WARNING"):
                self._pages(stride=1, full_host_pages=100_000, page_bytes=1)

    def test_hard_ceiling_raises_and_names_the_stride(self):
        # The DRAM-derived tier owns real OOM rather than slow pinning, so it
        # fails instead of warning, and the message has to carry both the
        # cross-rank arithmetic and the one knob that shrinks the pool.
        with self.assertRaises(ValueError) as ctx:
            A._check_swa_host_pool_upper_bound(
                swa_gb=100.0,
                slow_gb=A._SWA_HICACHE_SLOW_LAUNCH_GB,
                hard_gb=10.0,
                full_host_pages=100_000,
                stride=1,
                page_bytes=1 << 20,
                avail_gb=64.0,
                ranks_per_node=8,
            )
        msg = str(ctx.exception)
        self.assertIn("hard limit", msg)
        self.assertIn("--hicache-swa-offload-page-stride", msg)
        self.assertIn("rank(s)/node", msg)

    def test_hard_limit_is_per_rank_on_the_node(self):
        # Every rank page-locks its own pool, so the ceiling is per rank: tp 8
        # over 2 nodes puts 4 ranks on one node's DRAM.
        hard_gb, avail_gb, ranks_per_node = A._swa_host_hard_limit_gb(
            types.SimpleNamespace(tp_size=8, nnodes=2)
        )
        self.assertEqual(ranks_per_node, 4)
        self.assertAlmostEqual(
            hard_gb,
            avail_gb * A._SWA_HICACHE_HARD_LIMIT_DRAM_FRACTION / 4,
            places=6,
        )


class TestWriteBackGuard(unittest.TestCase):
    """Strict bit-exact must fail fast at build() entry if write policy is not
    write_through (write_back can leave the SWA ring un-offloaded -> silent
    non-bit-exact reuse)."""

    import unittest.mock as _mock

    def _build(self, *, unified, write_policy, flag):
        strategy = A._DeepSeekV4Strategy()
        kv = types.SimpleNamespace(_unified_kv=unified)
        sa = types.SimpleNamespace(hicache_write_policy=write_policy)
        flag_obj = types.SimpleNamespace(get=lambda: flag)
        with self._mock.patch.object(
            A.envs, "SGLANG_UNIFIED_KV_BIT_EXACT_HICACHE", flag_obj
        ):
            strategy.build(
                cache=None,
                kvcache=kv,
                params=None,
                server_args=sa,
                load_cache_event=None,
            )

    def test_write_back_trips_guard(self):
        with self.assertRaises(ValueError) as ctx:
            self._build(unified=True, write_policy="write_back", flag=True)
        self.assertIn("write_through", str(ctx.exception))

    def test_write_through_passes_guard(self):
        # Passes the guard, then fails later on the None collaborators -- we only
        # assert it is NOT the guard ValueError.
        with self.assertRaises(Exception) as ctx:
            self._build(unified=True, write_policy="write_through", flag=True)
        self.assertNotIn("requires --hicache-write-policy", str(ctx.exception))

    def test_flag_off_no_guard(self):
        with self.assertRaises(Exception) as ctx:
            self._build(unified=True, write_policy="write_back", flag=False)
        self.assertNotIn("requires --hicache-write-policy", str(ctx.exception))

    def test_non_unified_no_guard(self):
        with self.assertRaises(Exception) as ctx:
            self._build(unified=False, write_policy="write_back", flag=True)
        self.assertNotIn("requires --hicache-write-policy", str(ctx.exception))

    def test_flag_on_refuses_to_start_while_unwired(self):
        # Past the write-policy guard nothing else stands between the flag and a
        # half-wired reuse path, so enabling it must refuse to boot until the
        # capture and restore halves are in.
        with self.assertRaises(ValueError) as ctx:
            self._build(unified=True, write_policy="write_through", flag=True)
        self.assertIn("not usable yet", str(ctx.exception))


class TestSwaRegionBuffers(unittest.TestCase):
    """The SWA-ring host pool must be page-granular with the sliding window as
    the page unit, so each indexed device row is exactly one host item_bytes.
    Row-granular device buffers (head_dim rows) declared with a page-granular
    item_bytes mismatch in transfer_kv_direct and crash."""

    def _fake_pool(self, *, num_slots, ring_size, head_dim, compress_rows, layers):
        import torch

        swa_pages = num_slots * ring_size
        rows = swa_pages + compress_rows
        kv_buffer = [
            torch.arange(rows * head_dim, dtype=torch.bfloat16).reshape(rows, head_dim)
            for _ in range(layers)
        ]
        unified_kv_pool = types.SimpleNamespace(
            swa_pages=swa_pages, head_dim=head_dim, kv_buffer=kv_buffer
        )
        return types.SimpleNamespace(
            _unified_kv=True,
            unified_swa_ring_size=ring_size,
            unified_kv_pool=unified_kv_pool,
        )

    def test_page_granular_geometry(self):
        import torch

        from sglang.srt.mem_cache.deepseek_v4_memory_pool import (
            DeepSeekV4TokenToKVPool as P,
        )

        ring_size, head_dim, num_slots, layers = 2, 4, 3, 2
        pool = self._fake_pool(
            num_slots=num_slots,
            ring_size=ring_size,
            head_dim=head_dim,
            compress_rows=8,
            layers=layers,
        )
        views, item_bytes = P.swa_region_buffers(pool)
        # one page == one sliding window == ring_size rows (bf16 = 2 bytes).
        self.assertEqual(item_bytes, ring_size * head_dim * 2)
        self.assertEqual(len(views), layers)
        for v in views:
            self.assertEqual(v.dtype, torch.uint8)
            # num_pages == swa_pages // ring_size == num_slots; row width == item_bytes.
            self.assertEqual(v.shape, (num_slots, item_bytes))
            self.assertEqual(v[0].nbytes, item_bytes)

    def test_view_preserves_ring_data(self):
        import torch

        from sglang.srt.mem_cache.deepseek_v4_memory_pool import (
            DeepSeekV4TokenToKVPool as P,
        )

        ring_size, head_dim = 2, 4
        pool = self._fake_pool(
            num_slots=3,
            ring_size=ring_size,
            head_dim=head_dim,
            compress_rows=8,
            layers=1,
        )
        views, _ = P.swa_region_buffers(pool)
        buf = pool.unified_kv_pool.kv_buffer[0]
        # host page 1 must be ring rows [ring_size, 2*ring_size) byte-identical.
        expected = buf.narrow(0, ring_size, ring_size).reshape(-1).view(torch.uint8)
        self.assertTrue(torch.equal(views[0][1], expected))

    def test_rejects_non_unified(self):
        from sglang.srt.mem_cache.deepseek_v4_memory_pool import (
            DeepSeekV4TokenToKVPool as P,
        )

        pool = types.SimpleNamespace(_unified_kv=False)
        with self.assertRaises(AssertionError):
            P.swa_region_buffers(pool)

    def test_rejects_non_divisible_ring(self):
        from sglang.srt.mem_cache.deepseek_v4_memory_pool import (
            DeepSeekV4TokenToKVPool as P,
        )

        # swa_pages must be a whole number of windows; a corrupt pool trips the guard.
        pool = self._fake_pool(
            num_slots=3, ring_size=2, head_dim=4, compress_rows=8, layers=1
        )
        pool.unified_kv_pool.swa_pages += 1  # 7, not a multiple of ring_size=2
        with self.assertRaises(AssertionError):
            P.swa_region_buffers(pool)

    def test_all_pages_map_to_their_window(self):
        import torch

        from sglang.srt.mem_cache.deepseek_v4_memory_pool import (
            DeepSeekV4TokenToKVPool as P,
        )

        ring_size, head_dim, num_slots = 2, 4, 5
        pool = self._fake_pool(
            num_slots=num_slots,
            ring_size=ring_size,
            head_dim=head_dim,
            compress_rows=6,
            layers=2,
        )
        views, _ = P.swa_region_buffers(pool)
        for layer, view in enumerate(views):
            buf = pool.unified_kv_pool.kv_buffer[layer]
            self.assertEqual(view.shape[0], num_slots)  # one page per window
            for page in range(num_slots):
                # page p must be exactly ring rows [p*ring, (p+1)*ring), byte-identical.
                expected = (
                    buf.narrow(0, page * ring_size, ring_size)
                    .reshape(-1)
                    .view(torch.uint8)
                )
                self.assertTrue(torch.equal(view[page], expected))


class TestSwaRingRegionDelegation(unittest.TestCase):
    """The assembler seam must delegate SWA-ring buffer resolution to the pool
    (which owns the unified_kv layout) and never re-derive geometry itself."""

    def test_delegates_to_pool(self):
        sentinel = (["buf0", "buf1"], 131072)
        kvcache = types.SimpleNamespace(
            _unified_kv=True, swa_region_buffers=lambda: sentinel
        )
        self.assertIs(A._dsv4_swa_ring_region_buffers(kvcache), sentinel)

    def test_rejects_non_unified(self):
        kvcache = types.SimpleNamespace(_unified_kv=False)
        with self.assertRaises(AssertionError):
            A._dsv4_swa_ring_region_buffers(kvcache)


# ---------------------------------------------------------------------------
# Commit-time c4/indexer STATE coupling guard, merged from
# test_swa_commit_coupling_guard.py. Defense-in-depth for non-file backends /
# partial per-pool get: the sidecar state pools ride the SWA window key family,
# so a loaded SWA window has one coupled state page per state pool. If a
# registered state pool loaded fewer pages than SWA, attaching would restore a
# desynced (dirty) state, so the whole window must be dropped (recompute).
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    unittest.main()
