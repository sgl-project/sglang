"""File-backed host storage for the offloaded Qwen4-Exp PLE table.

CPU part: the allocator builds a sparse file of exactly the table's size, hands
back a tensor with the requested shape/dtype whose writes land in the file and
survive a re-open, reuses the file across calls, replaces one of the wrong size,
and the prefetcher computes the right page set and honours its size floor. The
resident-set trimmer measures only its own mapping, drops its pages once over
budget without losing what was written through them, and is off when the budget
is zero or the mapping is pinned.

GPU part (skipped unless the device reads pageable host memory through the host
page tables, i.e. unified-memory parts such as GB10): the production Triton
gather kernel reading from the file-backed table matches a torch gather.
"""

import os
import tempfile
import unittest
from unittest import mock

import torch

from sglang.srt.models.qwen4_exp_ple_table import (
    PleFilePrefetcher,
    PleFileRssTrimmer,
    _mapping_rss_bytes,
    allocate_ple_host_table,
    default_ple_table_dir,
    device_uses_host_page_tables,
    make_ple_file_prefetcher,
    make_ple_file_rss_trimmer,
    ple_table_file_name,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


class TestPleFileTableAllocator(CustomTestCase):
    def test_file_is_sparse_and_sized_exactly(self):
        with tempfile.TemporaryDirectory() as d:
            table = allocate_ple_host_table((1000, 160), torch.float8_e4m3fn, "file", d)
            path = os.path.join(
                d, ple_table_file_name((1000, 160), torch.float8_e4m3fn)
            )
            self.assertTrue(os.path.exists(path))
            self.assertEqual(os.path.getsize(path), 1000 * 160)
            self.assertEqual(tuple(table.shape), (1000, 160))
            self.assertEqual(table.dtype, torch.float8_e4m3fn)
            # Sparse: nothing written yet, so (almost) no blocks allocated.
            self.assertLess(os.stat(path).st_blocks * 512, 64 * 1024)

    def test_writes_persist_and_file_is_reused(self):
        with tempfile.TemporaryDirectory() as d:
            shape, dtype = (64, 32), torch.bfloat16
            table = allocate_ple_host_table(shape, dtype, "file", d)
            row = torch.arange(32, dtype=torch.float32).to(dtype)
            table[7].copy_(row)  # what the weight loader does, row by row
            del table
            again = allocate_ple_host_table(shape, dtype, "file", d)
            self.assertTrue(torch.equal(again[7].float(), row.float()))
            self.assertEqual(len(os.listdir(d)), 1)

    def test_wrong_sized_file_is_replaced(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, ple_table_file_name((8, 8), torch.bfloat16))
            with open(path, "wb") as f:
                f.write(b"\x01" * 10)
            table = allocate_ple_host_table((8, 8), torch.bfloat16, "file", d)
            self.assertEqual(os.path.getsize(path), 8 * 8 * 2)
            self.assertEqual(tuple(table.shape), (8, 8))

    def test_tag_separates_tensor_parallel_shards(self):
        with tempfile.TemporaryDirectory() as d:
            a = allocate_ple_host_table(
                (8, 8), torch.bfloat16, "file", d, tag="rows0-8"
            )
            b = allocate_ple_host_table(
                (8, 8), torch.bfloat16, "file", d, tag="rows8-16"
            )
            a.fill_(1.0)
            self.assertEqual(len(os.listdir(d)), 2)
            self.assertTrue(torch.all(b.float() == 0.0))
            self.assertIn(
                "rows8-16", ple_table_file_name((8, 8), torch.bfloat16, "rows8-16")
            )

    def test_default_dir_is_per_checkpoint(self):
        with mock.patch.dict(os.environ, {"SGLANG_QWEN4_PLE_FILE_DIR": "/cache/ple"}):
            a = default_ple_table_dir("RadixArk/Qwen3.8-Flash-Next-NVFP4")
            b = default_ple_table_dir("/root/.cache/huggingface/flashnext-fp8/")
            self.assertEqual(a, "/cache/ple/RadixArk_Qwen3.8-Flash-Next-NVFP4")
            self.assertEqual(b, "/cache/ple/root_.cache_huggingface_flashnext-fp8")
            self.assertNotEqual(a, b)

    def test_unknown_backend_rejected(self):
        with self.assertRaises(ValueError):
            allocate_ple_host_table((4, 4), torch.bfloat16, "nvme", None)

    def test_pinned_backend_unchanged(self):
        if not torch.cuda.is_available():
            self.skipTest("pinned memory needs a CUDA runtime")
        table = allocate_ple_host_table((4, 4), torch.bfloat16, "pinned", None)
        self.assertTrue(table.is_pinned())
        self.assertIsNone(make_ple_file_prefetcher(table))


class TestPleFilePrefetcher(CustomTestCase):
    def test_page_set_covers_row_start_and_end(self):
        # 160-byte rows: row 25 spans bytes 4000-4159, i.e. pages 0 and 1.
        pages = PleFilePrefetcher.pages_for_rows(torch.tensor([25, 0]), 160)
        self.assertEqual(pages, [0, 1])
        pages = PleFilePrefetcher.pages_for_rows(torch.tensor([1000, 1000]), 160)
        self.assertEqual(pages, [39])  # dedup, single page

    def test_enqueue_respects_min_rows_and_advises_pages(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "t.bin")
            with open(path, "wb") as f:
                f.truncate(1 << 20)
            pf = PleFilePrefetcher(path, row_bytes=160, min_rows=4)
            try:
                self.assertFalse(pf.enqueue(torch.tensor([1, 2, 3])))
                with mock.patch("os.posix_fadvise") as fadvise:
                    self.assertTrue(pf.enqueue(torch.tensor([0, 1, 2, 30])))
                    pf._pool.shutdown(wait=True)
                    offsets = sorted(c.args[1] for c in fadvise.call_args_list)
                    # rows 0-2 live in page 0; row 30 (bytes 4800-4959) in page 1
                    self.assertEqual(offsets, [0, 4096])
            finally:
                pf.close()


class TestSmapsParsing(CustomTestCase):
    """The parser runs anywhere; only the live mapping needs Linux."""

    SMAPS = """00400000-00401000 r--p 00000000 08:01 1  /usr/bin/x
Size:                  4 kB
Rss:                   4 kB
7f0000000000-7f0004000000 rw-s 00000000 08:01 2  /cache/ple/table.bin
Size:              65536 kB
Rss:               32768 kB
7f0004000000-7f0008000000 rw-s 00000000 08:01 2  /cache/ple/table.bin
Size:              65536 kB
Rss:                1024 kB
7f0100000000-7f0100001000 rw-p 00000000 00:00 0
Size:                  4 kB
Rss:                   4 kB
"""

    def _rss(self, addr, nbytes):
        with tempfile.NamedTemporaryFile("w", suffix=".smaps", delete=False) as f:
            f.write(self.SMAPS)
            path = f.name
        try:
            return _mapping_rss_bytes(addr, nbytes, smaps_path=path)
        finally:
            os.unlink(path)

    def test_sums_every_vma_of_the_table_and_nothing_else(self):
        # The table spans both of its VMAs; the unrelated ones must not count.
        self.assertEqual(self._rss(0x7F0000000000, 0x8000000), (32768 + 1024) * 1024)

    def test_counts_a_partially_overlapping_vma(self):
        # A range ending inside the first VMA still needs that VMA's pages.
        self.assertEqual(self._rss(0x7F0000000000, 0x1000), 32768 * 1024)

    def test_ignores_unrelated_mappings(self):
        self.assertEqual(self._rss(0x7F0200000000, 0x1000), 0)

    def test_missing_smaps_is_reported_as_unknown(self):
        self.assertIsNone(
            _mapping_rss_bytes(0x1000, 0x1000, smaps_path="/nonexistent/smaps")
        )


class TestPleFileRssTrimmerConfig(CustomTestCase):
    def test_budget_zero_disables_the_trimmer(self):
        with tempfile.TemporaryDirectory() as d:
            table = allocate_ple_host_table((64, 32), torch.bfloat16, "file", d)
            with mock.patch.dict(
                os.environ, {"SGLANG_QWEN4_PLE_FILE_RSS_BUDGET_GB": "0"}
            ):
                self.assertIsNone(make_ple_file_rss_trimmer(table))

    def test_pinned_table_has_no_trimmer(self):
        if not torch.cuda.is_available():
            self.skipTest("pinned memory needs a CUDA runtime")
        table = allocate_ple_host_table((4, 4), torch.bfloat16, "pinned", None)
        self.assertIsNone(make_ple_file_rss_trimmer(table))


@unittest.skipUnless(
    os.path.exists("/proc/self/smaps"),
    "the resident set of a mapping is only readable on Linux",
)
class TestPleFileRssTrimmer(CustomTestCase):
    # 64 MiB: large enough that the Rss of the mapping stands out, small
    # enough to write in a CPU test.
    SHAPE = (32768, 1024)
    NBYTES = 32768 * 1024 * 2

    def _trimmer(self, table, budget_bytes):
        return PleFileRssTrimmer(
            addr=table.data_ptr(),
            nbytes=self.NBYTES,
            budget_bytes=budget_bytes,
            interval_s=3600.0,
            chunk_bytes=16 << 20,  # several chunks, as in production
        )

    def test_measures_its_own_mapping_only(self):
        with tempfile.TemporaryDirectory() as d:
            table = allocate_ple_host_table(self.SHAPE, torch.bfloat16, "file", d)
            trimmer = self._trimmer(table, 0)
            empty = trimmer.mapping_rss_bytes()
            table.fill_(1.0)  # touches every page
            touched = trimmer.mapping_rss_bytes()
            self.assertIsNotNone(touched)
            self.assertGreater(touched, empty)
            self.assertGreater(touched, self.NBYTES // 2)
            # Never the whole process: only the VMAs backing this table.
            self.assertLessEqual(touched, self.NBYTES + (16 << 20))

    def test_over_budget_drops_pages_and_keeps_the_data(self):
        with tempfile.TemporaryDirectory() as d:
            table = allocate_ple_host_table(self.SHAPE, torch.bfloat16, "file", d)
            table.fill_(1.0)
            table[7][3] = 2.0
            trimmer = self._trimmer(table, budget_bytes=1 << 20)
            before = trimmer.mapping_rss_bytes()
            freed = trimmer.trim_once()
            after = trimmer.mapping_rss_bytes()
            self.assertGreater(freed, 0)
            self.assertLess(after, before // 2)
            # MADV_DONTNEED on a shared file mapping drops the page-table
            # entries, not the page cache: the writes are still there.
            self.assertEqual(table[7][3].item(), 2.0)
            self.assertEqual(table[9][3].item(), 1.0)

    def test_under_budget_is_a_no_op(self):
        with tempfile.TemporaryDirectory() as d:
            table = allocate_ple_host_table(self.SHAPE, torch.bfloat16, "file", d)
            table.fill_(1.0)
            trimmer = self._trimmer(table, budget_bytes=self.NBYTES * 4)
            before = trimmer.mapping_rss_bytes()
            self.assertEqual(trimmer.trim_once(), 0)
            self.assertEqual(trimmer.mapping_rss_bytes(), before)

    def test_factory_starts_and_stops_a_thread(self):
        with tempfile.TemporaryDirectory() as d:
            table = allocate_ple_host_table((64, 32), torch.bfloat16, "file", d)
            with mock.patch.dict(
                os.environ,
                {
                    "SGLANG_QWEN4_PLE_FILE_RSS_BUDGET_GB": "1",
                    "SGLANG_QWEN4_PLE_FILE_RSS_INTERVAL_S": "3600",
                },
            ):
                trimmer = make_ple_file_rss_trimmer(table)
            self.assertIsNotNone(trimmer)
            try:
                self.assertTrue(trimmer._thread.is_alive())
            finally:
                trimmer.close()
                trimmer._thread.join(timeout=5)
            self.assertFalse(trimmer._thread.is_alive())


@unittest.skipUnless(
    torch.cuda.is_available() and device_uses_host_page_tables(0) is True,
    "needs a device that reads pageable host memory through the host page tables",
)
class TestPleFileTableGatherOnDevice(CustomTestCase):
    def test_triton_gather_reads_file_backed_table(self):
        import triton

        from sglang.srt.models.qwen4_exp import (
            _gather_ple_embedding_from_pinned_kernel,
        )

        rows, dim = 4096, 160
        with tempfile.TemporaryDirectory() as d:
            table = allocate_ple_host_table((rows, dim), torch.bfloat16, "file", d)
            table.copy_(torch.randn(rows, dim).to(torch.bfloat16))
            ids = torch.randint(0, rows, (2048,), device="cuda")
            out = torch.empty(2048, dim, dtype=torch.bfloat16, device="cuda")
            _gather_ple_embedding_from_pinned_kernel[(ids.numel(),)](
                table.data_ptr(),
                ids,
                out,
                embedding_dim=dim,
                tp_vocab_start=0,
                tp_vocab_end=rows,
                is_fp8=False,
                BLOCK_D=triton.next_power_of_2(dim),
            )
            torch.cuda.synchronize()
            expected = table[ids.cpu()].to("cuda")
            self.assertTrue(torch.equal(out, expected))


if __name__ == "__main__":
    unittest.main()
