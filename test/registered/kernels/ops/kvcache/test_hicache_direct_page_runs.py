"""Batched page-run direct copies move the bytes `transfer_kv_direct` moves, in stream order."""

import unittest
from unittest import mock

import numpy as np
import torch

from sglang.srt.mem_cache.pool_host import direct_copy
from sglang.srt.mem_cache.pool_host.direct_copy import transfer_kv_direct_runs
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=15, stage="base-b-kernel-unit", runner_config="1-gpu-small")

LAYERS = 3


def _units(pages, units_per_page):
    pages = torch.as_tensor(pages, dtype=torch.int64)
    return (pages[:, None] * units_per_page + torch.arange(units_per_page)).reshape(-1)


def _pages():
    """24 distinct device pages with one 10-page run; 24 host pages with one break."""
    rng = np.random.default_rng(0)
    run = np.arange(30, 40)
    others = rng.permutation(np.setdiff1d(np.arange(1, 64), run))[:14]
    host_pages = np.arange(5, 29)
    host_pages[12:] += 3
    return np.concatenate([run, others]), host_pages


def _no_fallback(**kwargs):
    # The stock copy moves the same bytes, so a silent fallback would pass every check.
    raise AssertionError("the batched page-run copy fell back to transfer_kv_direct")


def _copy(src, dst, src_idx, dst_idx, units_per_page, fallback=_no_fallback):
    transfer_kv_direct_runs(
        src_layers=src,
        dst_layers=dst,
        src_indices=src_idx,
        dst_indices=dst_idx,
        page_size=units_per_page,
        fallback=fallback,
    )


def _pinned(shape, value=0):
    return torch.full(shape, value, dtype=torch.uint8, pin_memory=True)


@unittest.skipUnless(torch.cuda.is_available(), "needs CUDA")
class TestDirectPageRuns(CustomTestCase):
    def setUp(self):
        # The failure latch is process-wide; one set by an earlier case skips the op.
        patcher = mock.patch.object(direct_copy, "_op_failed", False)
        patcher.start()
        self.addCleanup(patcher.stop)

    def _round_trip(self, *, units_per_page, row_elems, stream):
        from sgl_kernel.kvcacheio import transfer_kv_direct

        dev_pages, host_pages = _pages()
        dev_units = _units(dev_pages, units_per_page)
        host_units = _units(host_pages, units_per_page)
        rows = 80 * units_per_page
        with torch.cuda.stream(stream):
            dev = [
                torch.randint(
                    0, 256, (rows, 1, row_elems), dtype=torch.uint8, device="cuda"
                )
                for _ in range(LAYERS)
            ]
            host_ref = [_pinned((rows, 1, row_elems)) for _ in range(LAYERS)]
            host_run = [t.clone().pin_memory() for t in host_ref]
            transfer_kv_direct(dev, host_ref, dev_units, host_units, units_per_page)
            _copy(dev, host_run, dev_units, host_units, units_per_page)
            torch.cuda.synchronize()
            for a, b in zip(host_ref, host_run):
                self.assertTrue(torch.equal(a, b))

            back_ref = [torch.zeros_like(t) for t in dev]
            back_run = [torch.zeros_like(t) for t in dev]
            for layer in range(LAYERS):
                transfer_kv_direct(
                    [host_ref[layer]],
                    [back_ref[layer]],
                    host_units,
                    dev_units,
                    units_per_page,
                )
                _copy(
                    [host_ref[layer]],
                    [back_run[layer]],
                    host_units,
                    dev_units,
                    units_per_page,
                )
            torch.cuda.synchronize()
            for a, b in zip(back_ref, back_run):
                self.assertTrue(torch.equal(a, b))

    def test_token_rows(self):  # MLA KV: index unit = token
        self._round_trip(units_per_page=4, row_elems=96, stream=torch.cuda.Stream())

    def test_page_rows(self):  # DSA indexer: index unit = page
        self._round_trip(units_per_page=1, row_elems=264, stream=torch.cuda.Stream())

    def test_legacy_default_stream(self):
        """cudaMemcpyBatchAsync rejects the legacy stream; the op must still copy."""
        self._round_trip(
            units_per_page=4, row_elems=32, stream=torch.cuda.default_stream()
        )

    def test_copies_are_ordered_on_the_current_stream(self):
        """A copy issued on any other stream would run before the delayed fill before it."""
        dev_pages, host_pages = _pages()
        dev_units, host_units = _units(dev_pages, 4), _units(host_pages, 4)
        side = torch.cuda.Stream()
        with torch.cuda.stream(side):
            dev = [torch.zeros((320, 1, 64), dtype=torch.uint8, device="cuda")]
            host = [_pinned((320, 1, 64), 9)]
            torch.cuda.synchronize()
            torch.cuda._sleep(50_000_000)
            dev[0].fill_(7)
            _copy(dev, host, dev_units, host_units, 4)
            done = torch.cuda.Event()
            done.record()
        done.synchronize()
        self.assertTrue(bool((host[0][host_units] == 7).all()))

        with torch.cuda.stream(side):
            torch.cuda._sleep(50_000_000)
            dev[0].fill_(3)
            host[0].fill_(5)
            _copy(host, dev, host_units, dev_units, 4)
            done = torch.cuda.Event()
            done.record()
        done.synchronize()
        self.assertTrue(bool((dev[0][dev_units] == 5).all()))

    def _compare_with_stock(
        self, dev_idx, host_idx, page_size=4, fallback=_no_fallback
    ):
        from sgl_kernel.kvcacheio import transfer_kv_direct

        with torch.cuda.stream(torch.cuda.Stream()):
            dev = [torch.randint(0, 256, (64, 1, 32), dtype=torch.uint8, device="cuda")]
            host_ref, host_run = [_pinned((64, 1, 32))], [_pinned((64, 1, 32))]
            transfer_kv_direct(dev, host_ref, dev_idx, host_idx, page_size)
            _copy(dev, host_run, dev_idx, host_idx, page_size, fallback=fallback)
        torch.cuda.synchronize()
        self.assertTrue(torch.equal(host_ref[0], host_run[0]))

    def test_page_starting_mid_page_copies_its_own_rows(self):
        """Rows are addressed by index, not rounded down to a page boundary."""
        dev_idx = torch.tensor([8, 9, 10, 11, 12, 13, 14, 15])
        host_idx = torch.tensor([2, 3, 4, 5, 6, 7, 8, 9])
        self._compare_with_stock(dev_idx, host_idx)

    def test_page_with_a_gap_falls_back_to_stock(self):
        """A page whose rows are not consecutive cannot be one descriptor; stock copies it."""
        from sgl_kernel.kvcacheio import transfer_kv_direct

        host_idx = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7])
        pages = {
            "last rows apart": [8, 9, 10, 11, 20, 21, 40, 41],
            "interior gap": [8, 9, 10, 11, 20, 40, 22, 23],
            "interior permutation": [8, 9, 10, 11, 20, 22, 21, 23],
        }
        for name, dev in pages.items():
            with (
                self.subTest(name),
                mock.patch.object(direct_copy, "_op_failed", False),
                self.assertLogs(direct_copy.logger, "WARNING") as logs,
            ):
                self._compare_with_stock(
                    torch.tensor(dev), host_idx, fallback=transfer_kv_direct
                )
                self.assertTrue(direct_copy._op_failed)
                self.assertIn("does not span consecutive rows", "\n".join(logs.output))

    def test_indices_built_in_inference_mode(self):
        """Index tensors from inference mode have no version counter; they must still copy."""
        with torch.inference_mode():
            dev_idx, host_idx = torch.arange(8, 16), torch.arange(0, 8)
        for page_size in (4, 1):
            with self.subTest(page_size=page_size):
                self._compare_with_stock(dev_idx, host_idx, page_size)

    def test_out_of_range_page_raises_before_copying(self):
        """A row past its layer must not be written into the next layer's memory."""
        from sglang.kernels.ops.kvcache.page_runs import transfer_page_runs

        dev = torch.zeros((2, 16, 1, 32), dtype=torch.uint8, device="cuda")
        host = _pinned((16, 1, 32), 9)
        with self.assertRaisesRegex(Exception, "out of range"):
            transfer_page_runs(
                torch.arange(0, 4),
                torch.arange(16, 20),  # rows 0-3 of layer 1 if unchecked
                torch.tensor([host.data_ptr()], dtype=torch.uint64),
                torch.tensor([dev[0].data_ptr()], dtype=torch.uint64),
                host_rows=16,
                device_rows=16,
                rows_per_page=4,
                row_bytes=32,
                device=dev.device,
                to_host=False,
            )
        torch.cuda.synchronize()
        self.assertTrue(bool((dev == 0).all()))


if __name__ == "__main__":
    unittest.main()
