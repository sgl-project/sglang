"""Speculative top logprobs with one k for the whole batch come back as one
[rows, k] tensor pair. After the host copy and list conversion, every row must
equal the per-row path (mixed k) bit for bit: values, ids and tie order.
"""

import struct
import unittest
from types import SimpleNamespace

import torch

from sglang.srt.layers.logprob_processor import compute_spec_logprobs
from sglang.srt.managers.scheduler_components.batch_result_processor import (
    _top_logprobs_to_lists,
)
from sglang.srt.managers.utils import _rows_d2h
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

VOCAB = 64


def _bits(row):
    return struct.pack(f"<{len(row)}d", *row)


def _batch(nums, greedy):
    return SimpleNamespace(
        seq_lens=[8] * len(nums),
        top_logprobs_nums=list(nums),
        token_ids_logprobs=None,
        sampling_info=SimpleNamespace(
            is_all_greedy=greedy,
            temperatures=torch.tensor([0.6 + 0.1 * i for i in range(len(nums))]).view(
                -1, 1
            ),
        ),
    )


def _host_rows(nums, greedy, logits, predict, **layout):
    out = SimpleNamespace(
        next_token_logits=logits.clone(),
        next_token_top_logprobs_val=None,
        next_token_top_logprobs_idx=None,
    )
    compute_spec_logprobs(_batch(nums, greedy), out, predict.clone(), **layout)
    batched = torch.is_tensor(out.next_token_top_logprobs_val)
    val, idx = _top_logprobs_to_lists(
        _rows_d2h(out.next_token_top_logprobs_val),
        _rows_d2h(out.next_token_top_logprobs_idx),
    )
    return batched, val, idx


class TestSpecTopLogprobsBatched(CustomTestCase):
    def _case(self, *, k, greedy, ties, chain, bs=5, max_accept=4, seed=0):
        """Run bs requests with one k (batched path), and the same bs requests
        plus one extra request asking k=0 (per-row path): every accepted row of
        the first bs requests must be identical."""
        g = torch.Generator().manual_seed(seed)
        rows = (bs + 1) * max_accept
        logits = torch.randn(rows, VOCAB, generator=g)
        if ties:
            logits = (logits * 2).round() / 2
        predict = torch.randint(0, VOCAB, (rows,), generator=g)

        # Rows padded with -1 in accept_index are never read by the consumers.
        accepted = [
            i * max_accept + j
            for i in range(bs)
            for j in range(max_accept if chain else 1 + i % max_accept)
        ]

        def layout(n):
            if chain:
                return dict(chain_stride=max_accept)
            accept_index = torch.full((n, max_accept), -1, dtype=torch.int32)
            for i in range(n):
                accept_len = 1 + i % max_accept
                accept_index[i, :accept_len] = torch.arange(
                    i * max_accept, i * max_accept + accept_len
                )
            return dict(accept_index=accept_index)

        n_rows = bs * max_accept
        batched, val, idx = _host_rows(
            [k] * bs, greedy, logits[:n_rows], predict[:n_rows], **layout(bs)
        )
        per_row, ref_val, ref_idx = _host_rows(
            [k] * bs + [0], greedy, logits, predict, **layout(bs + 1)
        )
        self.assertTrue(batched)
        self.assertFalse(per_row)
        self.assertEqual(len(val), n_rows)
        for r in accepted:
            self.assertEqual(idx[r], ref_idx[r])
            self.assertEqual(_bits(val[r]), _bits(ref_val[r]))
        self.assertTrue(all(type(t) is int for row in idx for t in row))

    def test_temperature(self):
        self._case(k=5, greedy=False, ties=False, chain=False)

    def test_greedy_ties(self):
        self._case(k=8, greedy=True, ties=True, chain=False)

    def test_chain_layout(self):
        self._case(k=8, greedy=False, ties=True, chain=True)

    def test_top1_and_wide_k(self):
        self._case(k=1, greedy=False, ties=True, chain=False, seed=1)
        self._case(k=32, greedy=False, ties=True, chain=True, bs=7, seed=2)

    def test_mixed_k_keeps_row_lists(self):
        logits = torch.randn(4, VOCAB)
        out = SimpleNamespace(
            next_token_logits=logits,
            next_token_top_logprobs_val=None,
            next_token_top_logprobs_idx=None,
        )
        compute_spec_logprobs(
            _batch([3, 0], greedy=False),
            out,
            torch.tensor([1, 2, 3, 4]),
            chain_stride=2,
        )
        self.assertIsInstance(out.next_token_top_logprobs_val, list)
        self.assertEqual(
            [len(v) for v in out.next_token_top_logprobs_val], [3, 3, 0, 0]
        )


if __name__ == "__main__":
    unittest.main()
