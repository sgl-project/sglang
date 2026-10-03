"""page_first_direct HiCache transfers through the page gather kernel.

io_backend=direct with mem_layout=page_first_direct used to issue one memcpy
per (page, layer); on ROCm that is CPU-bound at a few GB/s. The MLA and DSA
indexer host pools now move whole pages with one gather launch per layer on ROCm.
The data must be bit-identical to the per-page memcpy path.
"""

import contextlib
import unittest
from unittest import mock

import torch

from sglang.srt.mem_cache.pool_host import common
from sglang.srt.utils import is_cuda, is_hip, is_npu, is_xpu
from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci

register_cuda_ci(est_time=15, stage="base-b", runner_config="1-gpu-small")
register_amd_ci(est_time=15, suite="stage-b-test-1-gpu-small-amd")


def _registered(buffer: torch.Tensor, ranges) -> torch.Tensor:
    setattr(buffer, common._CUDA_HOST_REGISTERED_RANGES_ATTR, ranges)
    return buffer


class TestPageIdsOfTokens(unittest.TestCase):
    def test_page_ids_of_whole_pages(self):
        pages = torch.tensor([5, 0, 9, 2])
        tokens = (pages[:, None] * 4 + torch.arange(4)).reshape(-1)
        got = common.page_ids_of_tokens(tokens, 4)
        self.assertEqual(got.dtype, torch.int64)
        self.assertTrue(torch.equal(got, pages))

    def test_rejects_partial_or_misaligned_pages(self):
        cases = {
            "partial page": torch.arange(6),
            "misaligned start": torch.arange(1, 9),
            "hole inside a page": torch.tensor([0, 1, 3, 2, 8, 9, 10, 11]),
        }
        for name, tokens in cases.items():
            with self.subTest(name):
                self.assertIsNone(common.page_ids_of_tokens(tokens, 4))

    def test_empty(self):
        got = common.page_ids_of_tokens(torch.empty(0, dtype=torch.int64), 4)
        self.assertEqual(got.numel(), 0)


def _tokens(pages, page_size=4):
    pages = torch.tensor(pages, dtype=torch.int64)
    return (pages[:, None] * page_size + torch.arange(page_size)).reshape(-1)


class TestDirectPageIndices(unittest.TestCase):
    def test_reuses_the_entries_of_the_current_load(self):
        pages = common.DirectPageIndices(4, "cpu", ((0, 16),))
        host = torch.arange(8, 16)
        dev = torch.arange(20, 28)
        first = pages.get(host, dev, reuse=True)
        self.assertEqual(len(first), 1)
        self.assertEqual(first[0][:2], (0, 16))
        self.assertTrue(torch.equal(first[0][2], torch.tensor([2, 3])))
        self.assertTrue(torch.equal(first[0][3], torch.tensor([5, 6])))
        self.assertIs(pages.get(host, dev, reuse=True), first)
        # A new load (new tensors, equal or not) converts again.
        self.assertIsNot(pages.get(host.clone(), dev, reuse=True), first)
        # One-shot conversions (backups) neither use nor replace the entry.
        again = pages.get(host, dev, reuse=False)
        self.assertIsNot(again, first)

    def test_misaligned_side_falls_back_and_warns_once(self):
        pages = common.DirectPageIndices(4, "cpu", ((0, 16),), pool_name="t")
        with self.assertLogs(common.logger, "WARNING") as logs:
            self.assertIsNone(
                pages.get(torch.arange(8), torch.arange(1, 9), reuse=True)
            )
            self.assertIsNone(
                pages.get(torch.arange(1, 9), torch.arange(8), reuse=True)
            )
        self.assertEqual(len(logs.records), 1)

    def test_empty_transfer_has_no_launch(self):
        pages = common.DirectPageIndices(4, "cpu", ((0, 8), (8, 16)))
        empty = torch.empty(0, dtype=torch.int64)
        self.assertEqual(pages.get(empty, empty, reuse=False), [])

    def test_routes_pages_to_their_registration(self):
        # Registrations hold host pages [0, 3), [3, 6), [6, 8).
        pages = common.DirectPageIndices(4, "cpu", ((0, 3), (3, 6), (6, 8)))
        host = _tokens([7, 0, 4, 2, 3])
        dev = _tokens([10, 11, 12, 13, 14])
        got = pages.get(host, dev, reuse=False)
        self.assertEqual([(g[0], g[1]) for g in got], [(0, 3), (3, 6), (6, 8)])
        # Host pages relative to the registration, device pages kept in order.
        self.assertEqual([g[2].tolist() for g in got], [[0, 2], [1, 0], [1]])
        self.assertEqual([g[3].tolist() for g in got], [[11, 13], [12, 14], [10]])

    def test_registrations_without_pages_are_skipped(self):
        pages = common.DirectPageIndices(4, "cpu", ((0, 3), (3, 6), (6, 8)))
        got = pages.get(_tokens([4, 5]), _tokens([0, 1]), reuse=False)
        self.assertEqual([(g[0], g[1]) for g in got], [(3, 6)])
        self.assertEqual(got[0][2].tolist(), [1, 2])


class TestDirectPageKernelSegments(unittest.TestCase):
    PAGE = 16  # bytes per pool page in these tests

    def _segments(
        self,
        *,
        hip,
        buffer,
        pin_memory=True,
        device="cuda",
        os_page=16,
        item_bytes=8,
    ):
        # Small test buffers: treat 16 B as the OS page unless a test says so.
        with (
            mock.patch.object(common, "_is_hip", hip),
            mock.patch.object(torch.cuda, "is_available", return_value=True),
            mock.patch.object(common, "_OS_PAGE_BYTES", os_page),
        ):
            return common.direct_page_kernel_segments(
                buffer,
                page_bytes=self.PAGE,
                item_bytes=item_bytes,
                pin_memory=pin_memory,
                target_device=device,
                pool_name="test",
            )

    def test_rocm_only(self):
        buf = torch.zeros(64, dtype=torch.uint8)
        buf = _registered(buf, [(buf.data_ptr(), 64)])
        self.assertEqual(self._segments(hip=True, buffer=buf), ((0, 4),))
        with self.assertLogs(common.logger, "INFO") as logs:
            self.assertIsNone(self._segments(hip=False, buffer=buf))
        self.assertIn(
            "per-page copies: the page gather kernel is used on ROCm only",
            logs.output[0],
        )

    def test_kernel_needs_blocks_of_8_bytes(self):
        # A 1-token DSA indexer page is 128 + 4 = 132 B per layer; the gather
        # kernel refuses it ("Item byte size must be divisible by 8").
        buf = torch.zeros(64, dtype=torch.uint8)
        buf = _registered(buf, [(buf.data_ptr(), 64)])
        with self.assertLogs(common.logger, "INFO") as logs:
            self.assertIsNone(self._segments(hip=True, buffer=buf, item_bytes=132))
        self.assertIn("per-page copies: the 132 B (page, layer) block", logs.output[0])
        self.assertEqual(
            self._segments(hip=True, buffer=buf, item_bytes=264), ((0, 4),)
        )

    def test_one_segment_per_page_aligned_registration(self):
        # A pool above SGLANG_HICACHE_HOST_REGISTER_CHUNK_GB is registered in
        # page-aligned pieces (e.g. 300 GB = 256 GiB + the rest); each piece is
        # its own kernel segment.
        buf = torch.zeros(80, dtype=torch.uint8)
        base = buf.data_ptr()
        _registered(buf, [(base, 32), (base + 32, 32), (base + 64, 16)])
        self.assertEqual(self._segments(hip=True, buffer=buf), ((0, 2), (2, 4), (4, 5)))

    def test_registrations_must_not_share_an_os_page(self):
        buf = torch.zeros(128, dtype=torch.uint8)
        base = buf.data_ptr()
        _registered(buf, [(base, 32), (base + 32, 96)])
        self.assertEqual(
            self._segments(hip=True, buffer=buf, os_page=32), ((0, 2), (2, 8))
        )
        self.assertIsNone(self._segments(hip=True, buffer=buf, os_page=64))

    def test_logs_the_path_and_reason(self):
        buf = torch.zeros(64, dtype=torch.uint8)
        base = buf.data_ptr()
        _registered(buf, [(base, 32), (base + 32, 32)])
        with self.assertLogs(common.logger, "INFO") as logs:
            self._segments(hip=True, buffer=buf)
        self.assertIn("page gather kernel over 2 host registration(s)", logs.output[0])
        _registered(buf, [(base, 24), (base + 24, 40)])
        with self.assertLogs(common.logger, "INFO") as logs:
            self._segments(hip=True, buffer=buf)
        self.assertIn(
            "per-page copies: a host registration ends inside a page", logs.output[0]
        )

    def test_rejects_registrations_that_do_not_tile_whole_pages(self):
        buf = torch.zeros(64, dtype=torch.uint8)
        base = buf.data_ptr()
        cases = {
            "not registered": [],
            "registration inside a page": [(base, 24), (base + 24, 40)],
            "short registration": [(base, 32)],
            "gap": [(base, 16), (base + 32, 32)],
            "other base": [(base + 16, 64)],
        }
        for name, ranges in cases.items():
            with self.subTest(name):
                _registered(buf, ranges)
                self.assertIsNone(self._segments(hip=True, buffer=buf))
        _registered(buf, [(base, 64)])
        self.assertIsNone(self._segments(hip=True, buffer=buf, pin_memory=False))
        self.assertIsNone(self._segments(hip=True, buffer=buf, device="cpu"))
        self.assertIsNone(self._segments(hip=True, buffer=None))
        with mock.patch.object(self, "PAGE", 48):  # 64 B is not whole pages
            self.assertIsNone(self._segments(hip=True, buffer=buf))


_gpu = (
    torch.cuda.is_available()
    and not is_npu()
    and not is_xpu()
    and (is_cuda() or is_hip())
)


@unittest.skipUnless(_gpu, "needs CUDA/ROCm")
class TestPageFirstDirectRoundTrip(unittest.TestCase):
    PAGE = 64
    LAYERS = 4

    def _pools(self, skip_topk_layers, kernel: bool, chunk_bytes=None):
        from sglang.srt.mem_cache.memory_pool import DSATokenToKVPool
        from sglang.srt.mem_cache.pool_host import dsa as dsa_module
        from sglang.srt.mem_cache.pool_host import mla as mla_module
        from sglang.srt.mem_cache.pool_host.common import direct_page_kernel_segments
        from sglang.srt.mem_cache.pool_host.dsa import (
            DSAIndexerPoolHost,
            make_dsa_indexer_pool_decl,
        )
        from sglang.srt.mem_cache.pool_host.mla import MLATokenToKVPoolHost

        try:
            device_pool = DSATokenToKVPool(
                size=self.PAGE * 16,
                page_size=self.PAGE,
                kv_lora_rank=512,
                dtype=torch.float8_e4m3fn,
                qk_rope_head_dim=64,
                layer_num=self.LAYERS,
                device="cuda",
                enable_memory_saver=False,
                kv_cache_dim=576,
                index_head_dim=128,
                skip_topk_layers=skip_topk_layers,
            )
        except AssertionError as e:  # e.g. HIP without the preshuffle indexer
            self.skipTest(f"DSATokenToKVPool(page_size={self.PAGE}) unavailable: {e}")
        chunk_limit = (
            mock.patch.object(
                common, "_host_register_chunk_limit_bytes", return_value=chunk_bytes
            )
            if chunk_bytes
            else contextlib.nullcontext()
        )
        # Pick the path under test: the segment helper decides on ROCm, so
        # present the platform as ROCm (kernel) or not (per-page copies).
        def pick_path(*args, **kwargs):
            with mock.patch.object(common, "_is_hip", kernel):
                return direct_page_kernel_segments(*args, **kwargs)

        with (
            mock.patch.object(mla_module, "direct_page_kernel_segments", pick_path),
            mock.patch.object(dsa_module, "direct_page_kernel_segments", pick_path),
            chunk_limit,
        ):
            mla_host = MLATokenToKVPoolHost(
                device_pool=device_pool,
                host_to_device_ratio=2.0,
                host_size=0,
                page_size=self.PAGE,
                layout="page_first_direct",
                pin_memory=True,
                device="cpu",
                allocator_type="default",
                override_kv_cache_dim=device_pool.kv_cache_dim,
            )
            indexer_host = DSAIndexerPoolHost(
                decl=make_dsa_indexer_pool_decl(device_pool),
                anchor_host=mla_host,
                pin_memory=True,
                device="cpu",
                allocator_type="default",
            )
        # Unregister before the mappings go away, as a server does at exit: a
        # later pool may be mapped at the same addresses.
        self.addCleanup(
            common._cuda_host_unregister, indexer_host.index_k_with_scale_buffer
        )
        self.addCleanup(mla_host.destroy)
        self.assertEqual(mla_host.use_direct_page_kernel, kernel)
        # The gather kernel needs (page, layer) blocks of a multiple of 8 B; a
        # DSA indexer token is 132 B.
        self.assertEqual(
            indexer_host.use_direct_page_kernel, kernel and self.PAGE * 132 % 8 == 0
        )
        self.assertEqual(mla_host.token_stride_size, 576)
        return device_pool, mla_host, indexer_host

    def _tokens(self, pages):
        pages = torch.tensor(pages, dtype=torch.int64)
        return (pages[:, None] * self.PAGE + torch.arange(self.PAGE)).reshape(-1)

    def _fill_device(self, device_pool, seed):
        g = torch.Generator(device="cuda").manual_seed(seed)
        for buf in list(device_pool.kv_buffer) + list(
            device_pool.index_k_with_scale_buffer
        ):
            if buf.numel():
                buf.view(torch.uint8).copy_(
                    torch.randint(
                        0, 256, buf.shape, dtype=torch.uint8, device="cuda", generator=g
                    ).view(buf.shape)
                )

    def _round_trip(self, skip_topk_layers, kernel, chunk_bytes=None):
        device_pool, mla_host, indexer_host = self._pools(
            skip_topk_layers, kernel, chunk_bytes
        )
        self._fill_device(device_pool, seed=7)
        mla_host.kv_buffer.fill_(0xA5)
        indexer_host.index_k_with_scale_buffer.fill_(0x5A)

        # Write-through backup of device pages 3, 1, 8 into host pages 6, 2, 11
        # (the direct backend keeps both index sets on the CPU).
        dev_tokens = self._tokens([3, 1, 8])
        host_tokens = self._tokens([6, 2, 11])
        mla_host.backup_from_device_all_layer(
            device_pool, host_tokens, dev_tokens, "direct"
        )
        indexer_host.backup_from_device_all_layer(
            device_pool, host_tokens, dev_tokens, "direct"
        )
        torch.cuda.synchronize()
        snapshot_kv = [b.view(torch.uint8).cpu().clone() for b in device_pool.kv_buffer]
        snapshot_idx = [b.cpu().clone() for b in device_pool.index_k_with_scale_buffer]

        # Load back into different device pages, one layer at a time.
        for buf in list(device_pool.kv_buffer) + list(
            device_pool.index_k_with_scale_buffer
        ):
            buf.view(torch.uint8).fill_(0)
        load_tokens = self._tokens([12, 4, 9])
        for layer_id in range(self.LAYERS):
            mla_host.load_to_device_per_layer(
                device_pool, host_tokens, load_tokens, layer_id, "direct"
            )
            indexer_host.load_to_device_per_layer(
                device_pool, host_tokens, load_tokens, layer_id, "direct"
            )
        torch.cuda.synchronize()

        for layer_id in range(self.LAYERS):
            got = device_pool.kv_buffer[layer_id].view(torch.uint8).cpu()
            exp = torch.zeros_like(got)
            exp[load_tokens] = snapshot_kv[layer_id][dev_tokens]
            self.assertTrue(torch.equal(got, exp), f"kv layer {layer_id}")
            idx = device_pool.index_k_with_scale_buffer[layer_id]
            if idx.shape[0] == 0:
                continue
            got = idx.cpu()
            exp = torch.zeros_like(got)
            exp[torch.tensor([12, 4, 9])] = snapshot_idx[layer_id][
                torch.tensor([3, 1, 8])
            ]
            self.assertTrue(torch.equal(got, exp), f"indexer layer {layer_id}")
        # Host pages not named by the backup are untouched.
        untouched = self._tokens([0, 5, 7])
        self.assertTrue(
            bool((mla_host.kv_buffer[untouched // self.PAGE] == 0xA5).all())
        )
        return mla_host, indexer_host

    def test_kernel_matches_memcpy_path(self):
        full = [False] * self.LAYERS
        kernel_pools = self._round_trip(full, kernel=True)
        memcpy_pools = self._round_trip(full, kernel=False)
        # Same device data (same seed) -> identical host pools on both paths.
        self.assertTrue(
            torch.equal(kernel_pools[0].kv_buffer, memcpy_pools[0].kv_buffer)
        )
        self.assertTrue(
            torch.equal(
                kernel_pools[1].index_k_with_scale_buffer,
                memcpy_pools[1].index_k_with_scale_buffer,
            )
        )

    def test_kernel_skips_zero_row_indexer_layers(self):
        # Layers that reuse the previous layer's top-k have 0-row index buffers.
        self._round_trip([False, True, False, True], kernel=True)

    def test_kernel_over_several_host_registrations(self):
        # A pool above SGLANG_HICACHE_HOST_REGISTER_CHUNK_GB is registered in
        # page-aligned pieces. A 540672 B limit is 3 MLA pages (147456 B) and
        # 16 indexer pages (33792 B), both whole OS pages: host pages 2, 6 and
        # 11 land in three different MLA registrations. The indexer host pool
        # keeps one host layer per layer that owns an index buffer, so with
        # all 4 layers it gets three registrations and with 2 (the other two
        # reuse the previous layer's top-k) a page is half as big and it gets two.
        chunk_bytes = 16 * self.LAYERS * self.PAGE * 132
        for skip in ([False] * self.LAYERS, [False, True, False, True]):
            with self.subTest(skip_topk_layers=skip):
                mla_host, indexer_host = self._round_trip(
                    skip, kernel=True, chunk_bytes=chunk_bytes
                )
                self.assertEqual(
                    mla_host._direct_page_indices.segments[:4],
                    ((0, 3), (3, 6), (6, 9), (9, 12)),
                )
                self.assertEqual(
                    len(indexer_host._direct_page_indices.segments),
                    {4: 3, 2: 2}[indexer_host.layer_num],
                )
        memcpy_pools = self._round_trip([False] * self.LAYERS, kernel=False)
        kernel_pools = self._round_trip(
            [False] * self.LAYERS, kernel=True, chunk_bytes=chunk_bytes
        )
        self.assertTrue(
            torch.equal(kernel_pools[0].kv_buffer, memcpy_pools[0].kv_buffer)
        )
        self.assertTrue(
            torch.equal(
                kernel_pools[1].index_k_with_scale_buffer,
                memcpy_pools[1].index_k_with_scale_buffer,
            )
        )


class TestPageFirstDirectRoundTripPageSize1(TestPageFirstDirectRoundTrip):
    """page_size=1: MLA pages (576 B) take the kernel, DSA indexer pages
    (132 B) keep the per-page copies, and both match the memcpy path."""

    PAGE = 1

    def test_kernel_skips_zero_row_indexer_layers(self):
        self.skipTest("covered at page_size=64")

    def test_kernel_over_several_host_registrations(self):
        self.skipTest("covered at page_size=64")


if __name__ == "__main__":
    unittest.main()
