"""Regression test for the DFLASH compact draft cache host bound.

flashinfer_backend.py builds the DFLASH verify attention args from the host-side
``seq_lens_cpu`` and documents the requirement it relies on:

    "the batch carries seq_lens_cpu = prefix + draft_token_num ... which equals
     the device kv length generate_attn_arg_prefill produces"

The non-compact path upholds this by adding ``block_size``. The compact path
(``--speculative-draft-window-size``) fills ``seq_lens_cpu`` from
``_fill_compact_seq_lens_cpu_bound`` -> ``_compute_compact_draft_seq_lens_host``,
whose bound used to be ``min(len, window + page_size)`` and therefore fell
``block_size`` short of the device kv length. Its own docstring calls the value
"a safe upper bound", so the under-estimate was a contract violation, not a
conservative choice.

Measured effect before the fix (L40S, Qwen3-4B + Qwen3-4B-DFlash-b16, L=28110,
W=2048): host 2048 vs device 2064, draft accept rate 0.058 -> 0.002 and
acceptance length 1.76 -> 1.03, i.e. the windowed path was accepting essentially
nothing.

This test is CPU-only: it constructs the worker with ``object.__new__`` so no
model, GPU or server is needed.
"""

import unittest

import torch

from sglang.srt.speculative.dflash_worker_v2 import DFlashWorkerV2
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="stage-a-test-cpu")


def _worker(window, page_size, block_size):
    w = object.__new__(DFlashWorkerV2)  # __init__ needs a real model
    w.draft_window_size = window
    w.page_size = page_size
    w.block_size = block_size
    return w


class TestCompactDraftSeqLensHostBound(CustomTestCase):
    def _bound(self, w, lens):
        out = torch.zeros(len(lens), dtype=torch.int32)
        w._compute_compact_draft_seq_lens_host(
            torch.tensor(lens, dtype=torch.int64), out=out
        )
        return out.tolist()

    def test_bound_covers_the_verify_block(self):
        """The bound must be an upper bound on the DEVICE kv length, which for
        the draft is (compact prefix + block_size)."""
        window, page_size, block_size = 2048, 1, 16
        w = _worker(window, page_size, block_size)
        lens = [1, 1024, 2048, 2049, 5000, 28110]
        got = self._bound(w, lens)
        for L, b in zip(lens, got):
            device_kv_len = min(L, window) + block_size
            self.assertGreaterEqual(
                b,
                device_kv_len,
                f"host bound {b} < device kv length {device_kv_len} for len={L}",
            )

    def test_bound_equals_prefix_plus_block_when_uncropped(self):
        """With no cropping the host value must match the non-compact path,
        which is seq_lens + block_size."""
        window, page_size, block_size = 2048, 1, 16
        w = _worker(window, page_size, block_size)
        got = self._bound(w, [2048])
        self.assertEqual(got, [2048 + block_size])

    def test_bound_is_clamped_for_long_prefixes(self):
        window, page_size, block_size = 2048, 1, 16
        w = _worker(window, page_size, block_size)
        got = self._bound(w, [5000, 28110])
        self.assertEqual(got, [window + block_size, window + block_size])

    def test_page_alignment_still_covered(self):
        """page_size > 1 keeps its extra page of headroom, plus the block."""
        window, page_size, block_size = 4096, 16, 16
        w = _worker(window, page_size, block_size)
        got = self._bound(w, [100000])
        self.assertEqual(got, [window + page_size + block_size])


if __name__ == "__main__":
    unittest.main()
