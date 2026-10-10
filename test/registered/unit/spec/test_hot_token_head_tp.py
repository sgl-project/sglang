"""Unit tests for slice_hot_token_head: reduced draft vocab under vocab-parallel TP."""

import types
import unittest

import torch

from sglang.srt.speculative.eagle_worker_v2 import (
    check_hot_token_head_indexable,
    slice_hot_token_head,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


def _fake_lm_head(tp_size, rank, part, org_vocab_size):
    return types.SimpleNamespace(
        tp_size=tp_size,
        num_embeddings_per_partition=part,
        org_vocab_size=org_vocab_size,
        shard_indices=types.SimpleNamespace(org_vocab_start_index=rank * part),
    )


class TestSliceHotTokenHead(unittest.TestCase):
    def _check(self, vocab, tp_size, hot):
        part = -(-vocab // tp_size)
        # Row r of the full head stores r, so a gathered column's value is its id.
        full = torch.arange(tp_size * part, dtype=torch.float32)
        gathered_ref = None
        per_rank_logits = []
        for rank in range(tp_size):
            head = _fake_lm_head(tp_size, rank, part, vocab)
            local_rows, gathered = slice_hot_token_head(hot, head)
            shard = full[rank * part : (rank + 1) * part]
            self.assertTrue(bool((local_rows >= 0).all()))
            self.assertTrue(bool((local_rows < part).all()))
            per_rank_logits.append(shard[local_rows])
            if gathered_ref is None:
                gathered_ref = gathered
            else:
                self.assertTrue(torch.equal(gathered_ref, gathered))
        # All-gather along vocab == concatenation in rank order.
        logits = torch.cat(per_rank_logits)
        self.assertTrue(torch.equal(logits.long(), gathered_ref))
        # Every hot id is present, every column is a distinct real token.
        self.assertTrue(set(hot.tolist()) <= set(gathered_ref.tolist()))
        self.assertEqual(len(set(gathered_ref.tolist())), gathered_ref.numel())
        self.assertTrue(bool((gathered_ref < vocab).all()))
        return gathered_ref

    def test_tp2_unbalanced(self):
        # Mostly low ids (typical for BPE vocabularies): rank 0 owns more.
        hot = torch.cat([torch.arange(0, 900), torch.tensor([1500, 1999])])
        gathered = self._check(vocab=2000, tp_size=2, hot=hot)
        self.assertEqual(gathered.numel(), 2 * 900)

    def test_tp4_with_empty_shard(self):
        hot = torch.tensor([3, 7, 9, 250, 260])
        self._check(vocab=1000, tp_size=4, hot=hot)

    def test_uneven_last_shard(self):
        hot = torch.tensor([0, 1, 2, 998])
        self._check(vocab=999, tp_size=2, hot=hot)

    def test_tp1_passthrough(self):
        hot = torch.tensor([5, 1, 9])
        rows, gathered = slice_hot_token_head(hot, types.SimpleNamespace(tp_size=1))
        self.assertTrue(torch.equal(rows, hot))
        self.assertTrue(torch.equal(gathered, hot))
        rows, gathered = slice_hot_token_head(hot, None)
        self.assertTrue(torch.equal(rows, hot))

    def test_shard_too_small(self):
        hot = torch.arange(0, 60)
        with self.assertRaises(ValueError):
            slice_hot_token_head(hot, _fake_lm_head(2, 0, 50, 70))

    def test_block_quantized_head_rejected(self):
        check_hot_token_head_indexable(torch.zeros(4, 8, dtype=torch.bfloat16))
        with self.assertRaises(ValueError):
            check_hot_token_head_indexable(torch.zeros(4, 4, dtype=torch.uint8))


if __name__ == "__main__":
    unittest.main()
