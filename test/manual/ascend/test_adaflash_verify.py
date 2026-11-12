import unittest
from types import SimpleNamespace

import torch
from torch import nn

from sglang.srt.speculative.adaflash_verify import AdaFlashVerify


class FixedRatios(nn.Module):
    def forward(self, hidden):
        return hidden[:, :1, 0]


def verifier():
    result = AdaFlashVerify.__new__(AdaFlashVerify)
    result.model = SimpleNamespace(thresh_head_two_model=FixedRatios())
    result.block_size = 16
    result.num_steps = 0
    result.num_verify_tokens = 0
    result.num_full_width_tokens = 0
    result.last_verify_lens = None
    return result


class TestAdaFlashVerify(unittest.TestCase):
    def test_direct_lengths_keep_anchor_and_bound_each_request(self):
        for device in ["cpu", "npu"]:
            with self.subTest(device=device):
                adapter = verifier()
                hidden = torch.tensor([-1.0, 0.4375, 2.0], device=device).reshape(3, 1, 1)
                layout = adapter.layout(hidden)
                self.assertEqual(layout.verify_lens.cpu().tolist(), [1, 7, 16])
                self.assertEqual(layout.qo_indptr_device.cpu().tolist(), [0, 1, 8, 24])
                self.assertEqual(layout.total_verify_tokens, 24)
                self.assertEqual(layout.graph_num_tokens, 24)
                self.assertEqual(adapter.num_verify_tokens, 24)
                self.assertEqual(adapter.num_full_width_tokens, 48)

    def test_matching_padding_cannot_extend_acceptance_past_verified_prefix(self):
        for device in ["cpu", "npu"]:
            with self.subTest(device=device):
                adapter = verifier()
                hidden = torch.tensor([0.0, 0.4375, 1.0], device=device).reshape(3, 1, 1)
                layout = adapter.layout(hidden)
                # Token zero also wins every unmaterialized padding row. A plain
                # rectangular acceptance scan would incorrectly accept all 15.
                candidates = torch.zeros((3, 16), dtype=torch.int64, device=device)
                logits = torch.zeros((48, 4), device=device)
                logits[:, 0] = 1
                correct, bonus = adapter.accept(
                    candidates=candidates, logits=logits, layout=layout
                )
                self.assertEqual(correct.cpu().tolist(), [0, 6, 15])
                self.assertEqual(bonus.cpu().tolist(), [0, 0, 0])

    def test_bonus_comes_from_last_materialized_target_row(self):
        adapter = verifier()
        layout = adapter.layout(torch.tensor([0.4375], device="npu").reshape(1, 1, 1))
        candidates = torch.zeros((1, 16), dtype=torch.int64, device="npu")
        logits = torch.zeros((16, 4), device="npu")
        logits[:, 0] = 1
        logits[6, 2] = 3
        correct, bonus = adapter.accept(candidates=candidates, logits=logits, layout=layout)
        self.assertEqual(correct.cpu().tolist(), [6])
        self.assertEqual(bonus.cpu().tolist(), [2])


if __name__ == "__main__":
    unittest.main()
