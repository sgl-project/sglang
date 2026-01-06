import unittest

import torch

from sglang.srt.speculative.dflash_utils import compute_dflash_accept_len_and_bonus


class TestDFlashAcceptanceUnit(unittest.TestCase):
    def test_accept_len_and_bonus_basic(self):
        candidates = torch.tensor(
            [
                [10, 11, 12, 13],
                [20, 21, 22, 23],
            ],
            dtype=torch.long,
        )
        target_predict = torch.tensor(
            [
                [11, 12, 55, 0],  # accept 11,12 then bonus=55
                [99, 21, 22, 0],  # accept none then bonus=99
            ],
            dtype=torch.long,
        )

        accept_len, bonus = compute_dflash_accept_len_and_bonus(
            candidates=candidates,
            target_predict=target_predict,
        )
        self.assertEqual(accept_len.tolist(), [2, 0])
        self.assertEqual(bonus.tolist(), [55, 99])

    def test_accept_len_all_accepted(self):
        candidates = torch.tensor([[10, 11, 12, 13]], dtype=torch.long)
        target_predict = torch.tensor([[11, 12, 13, 77]], dtype=torch.long)

        accept_len, bonus = compute_dflash_accept_len_and_bonus(
            candidates=candidates,
            target_predict=target_predict,
        )
        self.assertEqual(accept_len.tolist(), [3])
        self.assertEqual(bonus.tolist(), [77])

    def test_shape_mismatch_raises(self):
        candidates = torch.zeros((2, 4), dtype=torch.long)
        target_predict = torch.zeros((2, 5), dtype=torch.long)
        with self.assertRaises(ValueError):
            compute_dflash_accept_len_and_bonus(
                candidates=candidates,
                target_predict=target_predict,
            )


if __name__ == "__main__":
    unittest.main()

