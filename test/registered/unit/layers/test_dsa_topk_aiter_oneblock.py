"""Unit tests for routing DSA paged top-k through aiter's one-block kernel.

sgl_kernel's ``topk_transform_decode_kernel`` launches one workgroup per query
row (``grid = dim3{B}`` in topk.hip). At decode that is ``batch * next_n`` rows
-- 8 to 16 on a conc-4 GLM-5.2 run -- so it occupies 8/256 CUs on MI355X.
aiter's ``top_k_per_row_decode`` does the same selection with a wider radix (11
or 12 bits vs 8, so three passes instead of four) and is 3.3-3.9x faster at
these shapes; being one workgroup per row it also has no cross-block barrier,
unlike aiter's multi-block entries.

The dispatch has to recover two things the aiter entry needs but sglang does not
pass down: ``next_n`` and per-sequence ``seq_lens``. Both come from
``attn_metadata`` by shape alone, so no device sync is involved:

* decode sets ``dsa_seqlens_expanded = cache_seqlens_int32`` -> ``next_n == 1``
* target-verify expands it to ``batch * next_n`` with the MTP stagger
  ``lengths[b * next_n + j] = max(seq_lens[b] - next_n + j + 1, 0)``, which is
  exactly what ``top_k_per_row_decode`` reconstructs internally

These tests cover that derivation and the guards around it with the aiter call
mocked out -- pure dispatch logic, no GPU required.
"""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.layers.attention.dsa import dsa_topk_backend as dtb
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

TOPK = 2048
WIDTH = 4096  # page-table width, i.e. the largest index the gather can take


def _expanded_lengths(seq_lens, next_n):
    """Reproduce seqlens_expand_kernel: max(kv_len - qo_len + 1 + j, 0)."""
    return torch.tensor(
        [max(s - next_n + j + 1, 0) for s in seq_lens for j in range(next_n)],
        dtype=torch.int32,
    )


class _AiterSpy:
    """Stands in for the aiter module and records the decode call's arguments."""

    def __init__(self):
        self.calls = []

    def top_k_per_row_decode(
        self, logits, next_n, seq_lens, indices, num_rows, s0, s1, k
    ):
        self.calls.append(
            SimpleNamespace(
                next_n=next_n, seq_lens=seq_lens.clone(), num_rows=num_rows, k=k
            )
        )
        indices.fill_(0)


class TestAiterOneBlockDispatch(CustomTestCase):
    def _invoke(self, seq_lens, next_n, *, rows=None, row_starts=None, width=WIDTH):
        """Drive the dispatch with a batch shaped like decode / target-verify."""
        batch = len(seq_lens)
        rows = batch * next_n if rows is None else rows
        logits = torch.zeros(rows, 64, dtype=torch.float32)
        lengths = _expanded_lengths(seq_lens, next_n)
        page_table = torch.zeros(rows, width, dtype=torch.int32)
        meta = SimpleNamespace(
            cache_seqlens_int32=torch.tensor(seq_lens, dtype=torch.int32),
            dsa_extend_seq_lens_list=[next_n] * batch,
        )
        spy = _AiterSpy()
        with patch.dict("sys.modules", {"aiter": spy}), patch.object(
            dtb, "_aiter_topk_available", return_value=True
        ), patch.object(
            dtb,
            "_get_triton_gather",
            return_value=lambda idx, tab, w, out: out.zero_(),
        ):
            out = dtb._aiter_paged_topk_transform(
                logits, lengths, page_table, TOPK, row_starts, meta
            )
        return out, spy

    def test_decode_derives_next_n_one(self):
        """decode: lengths has one row per sequence, so next_n must come out 1."""
        out, spy = self._invoke([1000, 2000, 3000, 4000], next_n=1)
        self.assertIsNotNone(out)
        self.assertEqual(len(spy.calls), 1)
        self.assertEqual(spy.calls[0].next_n, 1)
        self.assertEqual(spy.calls[0].num_rows, 4)

    def test_target_verify_derives_next_n(self):
        """target-verify: rows == batch * next_n, so next_n is the ratio."""
        for next_n in (2, 4, 8):
            with self.subTest(next_n=next_n):
                out, spy = self._invoke([3000, 4000], next_n=next_n)
                self.assertIsNotNone(out)
                self.assertEqual(spy.calls[0].next_n, next_n)
                self.assertEqual(spy.calls[0].num_rows, 2 * next_n)

    def test_seq_lens_passed_through_unmodified_when_short(self):
        """Sequences inside the page table reach aiter as-is."""
        seq_lens = [100, WIDTH - 1]
        _, spy = self._invoke(seq_lens, next_n=4)
        self.assertTrue(
            torch.equal(
                spy.calls[0].seq_lens, torch.tensor(seq_lens, dtype=torch.int32)
            )
        )

    def test_seq_lens_clamped_to_page_table_width(self):
        """Selection must not be able to index past the gather's page table.

        sgl_kernel fuses selection and the page-table transform into one kernel
        and bounds selection internally. Split into select + gather, nothing
        keeps the two in step: under a CUDA graph the page table is a static
        capture-time width while lengths stay live, so a sequence that outgrows
        it would select past the end and fault in the gather.
        """
        _, spy = self._invoke([WIDTH * 10, 50], next_n=4)
        self.assertEqual(int(spy.calls[0].seq_lens.max()), WIDTH)
        self.assertEqual(int(spy.calls[0].seq_lens.min()), 50)

    def test_capture_layout_is_accepted(self):
        """The graph builder emits [1]*bs*next_n, the eager one [next_n]*bs.

        Both mean the same expansion; the derivation must accept either.
        """
        logits = torch.zeros(8, 64, dtype=torch.float32)
        lengths = _expanded_lengths([1000, 2000], 4)
        page_table = torch.zeros(8, WIDTH, dtype=torch.int32)
        meta = SimpleNamespace(
            cache_seqlens_int32=torch.tensor([1000, 2000], dtype=torch.int32),
            dsa_extend_seq_lens_list=[1] * 2 * 4,  # capture convention
        )
        spy = _AiterSpy()
        with patch.dict("sys.modules", {"aiter": spy}), patch.object(
            dtb, "_aiter_topk_available", return_value=True
        ), patch.object(
            dtb, "_get_triton_gather", return_value=lambda i, t, w, o: o.zero_()
        ):
            out = dtb._aiter_paged_topk_transform(
                logits, lengths, page_table, TOPK, None, meta
            )
        self.assertIsNotNone(out)
        self.assertEqual(spy.calls[0].next_n, 4)

    def test_padded_rows_do_not_break_derivation(self):
        """DP-padded / idle rows have seq_lens < next_n; lengths clamp to 0."""
        out, spy = self._invoke([3, 0], next_n=4)
        self.assertIsNotNone(out)
        self.assertEqual(spy.calls[0].next_n, 4)

    def test_falls_back_when_page_table_rows_mismatch(self):
        """Without one page-table row per logits row the gather has no row map."""
        logits = torch.zeros(8, 64, dtype=torch.float32)
        lengths = _expanded_lengths([1000, 2000], 4)
        page_table = torch.zeros(2, WIDTH, dtype=torch.int32)  # 2 rows, not 8
        meta = SimpleNamespace(
            cache_seqlens_int32=torch.tensor([1000, 2000], dtype=torch.int32),
            dsa_extend_seq_lens_list=[4, 4],
        )
        with patch.object(dtb, "_aiter_topk_available", return_value=True):
            self.assertIsNone(
                dtb._aiter_paged_topk_transform(
                    logits, lengths, page_table, TOPK, None, meta
                )
            )

    def test_falls_back_on_ragged_row_starts(self):
        """A non-trivial row_starts means extend/ragged rows, not covered here."""
        out, _ = self._invoke(
            [1000, 2000], next_n=4, row_starts=torch.zeros(8, dtype=torch.int32)
        )
        self.assertIsNone(out)

    def test_falls_back_when_rows_not_a_multiple_of_batch(self):
        """next_n would not be an integer, so the stagger cannot be reconstructed."""
        logits = torch.zeros(7, 64, dtype=torch.float32)
        lengths = torch.full((7,), 1000, dtype=torch.int32)
        page_table = torch.zeros(7, WIDTH, dtype=torch.int32)
        meta = SimpleNamespace(
            cache_seqlens_int32=torch.tensor([1000, 2000], dtype=torch.int32),
            dsa_extend_seq_lens_list=[4, 4],
        )
        with patch.object(dtb, "_aiter_topk_available", return_value=True):
            self.assertIsNone(
                dtb._aiter_paged_topk_transform(
                    logits, lengths, page_table, TOPK, None, meta
                )
            )

    def test_falls_back_on_non_fp32_logits(self):
        """aiter's entry takes fp32 scores; anything else keeps the old path."""
        logits = torch.zeros(8, 64, dtype=torch.bfloat16)
        lengths = _expanded_lengths([1000, 2000], 4)
        page_table = torch.zeros(8, WIDTH, dtype=torch.int32)
        meta = SimpleNamespace(
            cache_seqlens_int32=torch.tensor([1000, 2000], dtype=torch.int32),
            dsa_extend_seq_lens_list=[4, 4],
        )
        with patch.object(dtb, "_aiter_topk_available", return_value=True):
            self.assertIsNone(
                dtb._aiter_paged_topk_transform(
                    logits, lengths, page_table, TOPK, None, meta
                )
            )


if __name__ == "__main__":
    unittest.main()
