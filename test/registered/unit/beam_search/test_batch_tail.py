"""Member tails preserve request order and each leader's sequence lengths."""

import unittest
from types import SimpleNamespace

import torch

from sglang.srt.beam_search.batch_tail import append_beam_tail, strip_beam_tail
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


class TestBeamTail(unittest.TestCase):
    def test_mixed_width_tail_round_trip(self):
        for host_lengths in (True, False):
            with self.subTest(host_lengths=host_lengths):

                def group(rows):
                    return SimpleNamespace(
                        member_rows=torch.tensor(rows),
                        member_rows_cpu=torch.tensor(rows),
                        num_member_rows=len(rows),
                        retired=False,
                    )

                batch = SimpleNamespace(
                    reqs=[
                        SimpleNamespace(beam_group=group([3, 4, 5])),
                        SimpleNamespace(beam_group=None),
                        SimpleNamespace(beam_group=group([6])),
                    ],
                    beam_tail=None,
                    device=torch.device("cpu"),
                    req_pool_indices=torch.tensor([0, 1, 2]),
                    req_pool_indices_cpu=torch.tensor([0, 1, 2]),
                    seq_lens=torch.tensor([10, 20, 30], dtype=torch.int32),
                    seq_lens_cpu=(
                        torch.tensor([10, 20, 30], dtype=torch.int32)
                        if host_lengths
                        else None
                    ),
                    orig_seq_lens=torch.tensor([8, 18, 28], dtype=torch.int32),
                    seq_lens_sum=60,
                    input_ids=torch.arange(7),
                    out_cache_loc=torch.arange(7),
                )
                append_beam_tail(batch)
                self.assertEqual(batch.req_pool_indices.tolist(), list(range(7)))
                self.assertEqual(batch.seq_lens.tolist(), [10, 20, 30, 10, 10, 10, 30])
                self.assertEqual(batch.orig_seq_lens.tolist(), [8, 18, 28, 8, 8, 8, 28])
                if host_lengths:
                    self.assertTrue(torch.equal(batch.seq_lens, batch.seq_lens_cpu))
                self.assertEqual(
                    [(e.leader_idx, e.start, e.end) for e in batch.beam_tail.entries],
                    [(0, 0, 3), (2, 3, 4)],
                )
                strip_beam_tail(batch)
                self.assertIsNone(batch.beam_tail)
                self.assertEqual(batch.req_pool_indices.tolist(), [0, 1, 2])
                self.assertEqual(batch.req_pool_indices_cpu.tolist(), [0, 1, 2])
                self.assertEqual(batch.seq_lens.tolist(), [10, 20, 30])
                self.assertEqual(batch.orig_seq_lens.tolist(), [8, 18, 28])
                self.assertEqual(batch.input_ids.tolist(), [0, 1, 2])
                self.assertIsNone(batch.out_cache_loc)
                self.assertIsNone(batch.seq_lens_sum)


if __name__ == "__main__":
    unittest.main()
