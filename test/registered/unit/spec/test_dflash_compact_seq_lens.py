"""Regression tests for DFLASH compact draft cache host seq_lens_cpu bounds.

FlashInfer's DFLASH verify fast plan consumes ``forward_batch.seq_lens_cpu`` as
the full device KV length (``compact_prefix + block_size``), matching the
non-compact path. ``_fill_compact_seq_lens_cpu_bound`` must add ``block_size``
after clamping the compact prefix bound across every length-source branch.
"""

import unittest

import torch

from sglang.srt.speculative.dflash_worker_v2 import DFlashWorkerV2
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _worker(window: int, page_size: int, block_size: int) -> DFlashWorkerV2:
    # Skip __init__, which requires live target and draft model runners.
    worker = object.__new__(DFlashWorkerV2)
    worker.device = torch.device("cpu")
    worker.draft_window_size = window
    worker.page_size = page_size
    worker._draft_compact_page_size = page_size
    worker.block_size = block_size
    return worker


class TestCompactDraftSeqLensCpuBound(CustomTestCase):
    def _fill(
        self,
        worker: DFlashWorkerV2,
        *,
        batch_seq_lens_cpu=None,
        nxt_kv_lens_cpu=None,
        length: int,
    ) -> torch.Tensor:
        out = torch.zeros(length, dtype=torch.int32)
        worker._fill_compact_seq_lens_cpu_bound(
            batch_seq_lens_cpu=batch_seq_lens_cpu,
            nxt_kv_lens_cpu=nxt_kv_lens_cpu,
            out=out,
        )
        return out

    def test_batch_seq_lens_cpu_matches_device_kv_len_for_page_size_one(self):
        # Covers both uncropped (<= window) and clamped (> window) prefixes:
        # folding block_size into the clamp bound instead of adding it after
        # clamping would still undershoot every uncropped prefix by block_size.
        window, page_size, block_size = 2048, 1, 16
        worker = _worker(window, page_size, block_size)
        prefix_lens = torch.tensor(
            [1, 1024, 2048, 2049, 5000, 28110], dtype=torch.int64
        )
        got = self._fill(
            worker,
            batch_seq_lens_cpu=prefix_lens,
            length=prefix_lens.numel(),
        )
        device_kv_lens = (
            worker._compute_compact_draft_seq_lens(prefix_lens) + block_size
        )
        torch.testing.assert_close(got, device_kv_lens.to(torch.int32))

    def test_page_aligned_compact_bound_covers_verify_block(self):
        window, page_size, block_size = 4096, 16, 16
        worker = _worker(window, page_size, block_size)
        prefix_lens = torch.tensor([1, 4096, 4100, 4160, 100000], dtype=torch.int64)
        got = self._fill(
            worker,
            batch_seq_lens_cpu=prefix_lens,
            length=prefix_lens.numel(),
        )
        device_kv_lens = (
            worker._compute_compact_draft_seq_lens(prefix_lens) + block_size
        )
        self.assertTrue(bool((got >= device_kv_lens.to(torch.int32)).all()))
        self.assertEqual(int(got[-1]), window + page_size + block_size)

    def test_nxt_kv_lens_cpu_and_fallback_cover_verify_block(self):
        window, page_size, block_size = 2048, 1, 16
        worker = _worker(window, page_size, block_size)
        prefix_lens = torch.tensor([1, 2048, 5000, 28110], dtype=torch.int64)
        device_kv_lens = (
            worker._compute_compact_draft_seq_lens(prefix_lens) + block_size
        ).to(torch.int32)

        # Overlap reserved lengths exceed window when L > window, so clamping
        # nxt_kv_lens_cpu without adding block_size still falls short.
        nxt_kv_lens = prefix_lens + block_size
        got_nxt = self._fill(
            worker,
            nxt_kv_lens_cpu=nxt_kv_lens,
            length=prefix_lens.numel(),
        )
        self.assertTrue(bool((got_nxt >= device_kv_lens).all()))
        self.assertEqual(int(got_nxt[-1]), window + block_size)

        got_fallback = self._fill(worker, length=prefix_lens.numel())
        self.assertTrue(bool((got_fallback >= device_kv_lens).all()))
        self.assertEqual(
            got_fallback.tolist(), [window + block_size] * prefix_lens.numel()
        )


if __name__ == "__main__":
    unittest.main()
