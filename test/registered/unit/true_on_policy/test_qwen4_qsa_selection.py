import unittest

import torch

from sglang.srt.true_on_policy import qsa
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")

_LOW_WORD = 4294967295


def _unique_key_selection(logits, counts, topk):
    """Top-k of the unique keys (score bits << 32 | ~block id), emitted as ascending ids, -1 padded."""
    rows, width = logits.shape
    ids = torch.arange(width, dtype=torch.int64)
    keys = (logits.view(torch.int32).to(torch.int64) << 32) | (_LOW_WORD - ids)
    out = torch.full((rows, topk), -1, dtype=torch.int32)
    for row in range(rows):
        count = min(int(counts[row]), width)
        top = keys[row, :count].topk(min(count, topk)).values
        picked = (_LOW_WORD - (top & _LOW_WORD)).sort().values
        out[row, : picked.numel()] = picked.to(torch.int32)
    return out


class TestQsaBlockSelection(unittest.TestCase):
    def test_ties_go_to_the_lower_block_id_as_with_unique_keys(self):
        """The selection keeps the order of unique (score bits, lower id) keys on non-negative scores.

        Rows cover no visible block, fewer than, exactly and more than top-k, and a count past the score width.
        """
        topk, width = 8, 40
        counts = torch.tensor([0, 1, 7, 8, 9, 23, 40, 55, 40], dtype=torch.int32)
        g = torch.Generator().manual_seed(0)
        # Four distinct scores, +0.0 among them: most blocks tie; the last row is all ties.
        logits = torch.randint(0, 4, (counts.numel(), width), generator=g).float() / 4
        logits[-1] = 0.5
        got = qsa.select_blocks(logits, counts, topk)
        self.assertTrue(torch.equal(got, _unique_key_selection(logits, counts, topk)))
        self.assertTrue(torch.equal(got[-1], torch.arange(topk, dtype=torch.int32)))


if __name__ == "__main__":
    unittest.main()
