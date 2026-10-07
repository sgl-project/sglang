import unittest

import torch

from sglang.kernels.ops.speculative.topk1 import draft_topk1_postprocess
from sglang.srt.layers.mtp_head_tp import (
    head_tp4_rank_groups,
    local_argmax_pair,
    merge_argmax_pairs,
)


class TestMTPHeadTopology(unittest.TestCase):
    def test_node_local_groups_and_fallback(self):
        for size in (4, 8, 16):
            hosts = [(r, f"node{r // 4}", r % 4) for r in range(size)]
            expected = [list(range(i, i + 4)) for i in range(0, size, 4)]
            self.assertEqual(head_tp4_rank_groups(hosts), expected)
            hosts[-1] = (size - 1, "wrong-host", 3)
            self.assertIsNone(head_tp4_rank_groups(hosts))
        self.assertIsNone(head_tp4_rank_groups([(r, "node", 0) for r in range(4)]))


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class TestMTPHeadTP(unittest.TestCase):
    def test_four_shard_argmax(self):
        for vocab in (3, 32773):
            torch.manual_seed(168)
            logits = torch.randn((10, vocab), device="cuda", dtype=torch.bfloat16)
            logits[0].fill_(1)
            logits[1].fill_(-float("inf"))
            logits[2, [0, vocab - 1]] = float("inf")
            logits[3, [1, vocab - 1]] = float("nan")
            logits[4].fill_(float("nan"))
            logits[5].fill_(-float("inf"))
            logits[5, -1] = float("nan")
            logits[6].fill_(-1e30)
            logits[6, -1] = float("nan")
            logits[7].fill_(0)
            logits[7, 0] = 1
            logits[7, -1] = torch.nextafter(logits.new_tensor(1), logits.new_tensor(2))
            for sentinel in (False, True):
                expected = torch.argmax(logits, -1, keepdim=True)
                if sentinel:
                    _, expected = draft_topk1_postprocess(
                        logits, torch.zeros(10, device="cuda", dtype=torch.int64)
                    )
                width = (vocab + 3) // 4
                pairs = []
                for rank in range(4):
                    start, end = (
                        min(rank * width, vocab),
                        min((rank + 1) * width, vocab),
                    )
                    padded = torch.full((10, width + 3), float("nan"), device="cuda")
                    padded[:, : end - start] = logits[:, start:end]
                    pairs.append(
                        local_argmax_pair(padded, start, end - start, sentinel)
                    )
                actual = merge_argmax_pairs(torch.stack(pairs))
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                empty = torch.stack(
                    [local_argmax_pair(logits[:0], draft_nan_sentinel=sentinel)] * 4
                )
                self.assertEqual(tuple(merge_argmax_pairs(empty).shape), (0, 1))


if __name__ == "__main__":
    unittest.main()
