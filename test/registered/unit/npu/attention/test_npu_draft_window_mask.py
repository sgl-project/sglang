"""Unit tests for the Ascend draft-decode sink+window KV mask.

`build_draft_window_mask` implements the same visible set as the
`generate_draft_decode_kv_indices` Triton kernel (the Triton/FlashInfer
draft-decode index builder) as a boolean mask for the Ascend backend:
attend the first `sink` prefix tokens plus the most recent `window` tokens,
while the draft tree tokens of each step stay visible. These tests pin that
equivalence against a pure-Python oracle of the kernel semantics.
"""

import random
import unittest

import torch

from sglang.srt.speculative.spec_utils import build_draft_window_mask
from sglang.test.ci.ci_register import register_npu_ci

register_npu_ci(est_time=2, suite="base-a-test-npu")


def _expected_visible(seq_len, window, sink, step):
    """Visible KV positions for one draft-decode step, per the Triton kernel.

    Mirrors _expected_base in test_draft_decode_window.py (StreamingLLM
    sink + recent window over the prefix) plus the always-visible tree
    tokens [seq_len, seq_len + step + 1).
    """
    cap = window + sink
    kept = min(seq_len, cap)
    s_eff = min(sink, seq_len)
    recent_start = seq_len - (kept - s_eff)
    visible = set(range(s_eff)) | set(range(recent_start, seq_len))
    visible |= set(range(seq_len, seq_len + step + 1))
    return visible


class TestDraftWindowMask(unittest.TestCase):
    def _check_row(self, mask_row, seq_len, window, sink, step, total_width):
        kv_len = seq_len + step + 1
        visible = _expected_visible(seq_len, window, sink, step)
        for pos in range(total_width):
            attend = not bool(mask_row[pos])
            if pos < kv_len:
                self.assertEqual(
                    attend,
                    pos in visible,
                    f"pos={pos} seq_len={seq_len} window={window} "
                    f"sink={sink} step={step}",
                )
            else:
                self.assertTrue(
                    mask_row[pos], f"pos >= kv_len must be masked (pos={pos})"
                )

    def test_fixed_cases(self):
        cases = [
            (5000, 1024, 64),  # band [64, 3976)
            (5000, 1024, 0),  # pure recent window
            (100, 4096, 64),  # cap exceeds seq_len -> full attention
            (50, 32, 64),  # seq_len < sink (s_eff caps at seq_len, full prefix visible)
            (128, 64, 64),  # seq_len == window + sink (exact boundary, empty band)
            (129, 64, 64),  # just past the boundary (band = [64, 65))
            (16, 8, 4),
        ]
        total_width = 6000
        for seq_len, window, sink in cases:
            for step in range(4):
                with self.subTest(seq_len=seq_len, window=window, sink=sink, step=step):
                    seq_lens = torch.tensor([seq_len], dtype=torch.int32)
                    kv_lens = seq_lens + step + 1
                    mask = build_draft_window_mask(
                        seq_lens, kv_lens, window, sink, total_width
                    )
                    self.assertEqual(mask.dtype, torch.bool)
                    self.assertEqual(mask.shape, (1, total_width))
                    self._check_row(mask[0], seq_len, window, sink, step, total_width)

    def test_batched_rows_independent(self):
        seq_lens = torch.tensor([129, 5000, 128, 64], dtype=torch.int32)
        window, sink, step = 64, 64, 2
        total_width = 6000
        kv_lens = seq_lens + step + 1
        mask = build_draft_window_mask(seq_lens, kv_lens, window, sink, total_width)
        for b, seq_len in enumerate(seq_lens.tolist()):
            self._check_row(mask[b], seq_len, window, sink, step, total_width)

    def test_random_against_oracle(self):
        random.seed(0)
        total_width = 512
        for _ in range(50):
            seq_len = random.randint(1, 400)
            window = random.randint(1, 300)
            sink = random.randint(0, 200)
            step = random.randint(0, 3)
            with self.subTest(seq_len=seq_len, window=window, sink=sink, step=step):
                seq_lens = torch.tensor([seq_len], dtype=torch.int32)
                kv_lens = seq_lens + step + 1
                mask = build_draft_window_mask(
                    seq_lens, kv_lens, window, sink, total_width
                )
                self._check_row(mask[0], seq_len, window, sink, step, total_width)

    def test_window_zero_is_noop(self):
        seq_lens = torch.tensor([100, 5000], dtype=torch.int32)
        mask = build_draft_window_mask(
            seq_lens, seq_lens + 3, window_size=0, sink_size=64, total_width=6000
        )
        self.assertTrue(torch.equal(mask, torch.zeros_like(mask)))

    def test_masked_band_is_contiguous(self):
        """The masked middle band must be one contiguous run per row."""
        for seq_len, window, sink in [(5000, 1024, 64), (300, 100, 10)]:
            seq_lens = torch.tensor([seq_len], dtype=torch.int32)
            # width == kv_len so the trailing always-masked region is excluded
            mask = build_draft_window_mask(
                seq_lens, seq_lens + 1, window, sink, seq_len + 1
            )[0]
            masked = torch.nonzero(mask).flatten().tolist()
            if not masked:
                continue
            self.assertEqual(
                masked,
                list(range(masked[0], masked[-1] + 1)),
                f"band not contiguous for seq_len={seq_len} window={window} sink={sink}",
            )


if __name__ == "__main__":
    unittest.main()
