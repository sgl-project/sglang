import unittest
from types import SimpleNamespace

import torch

from sglang.srt.speculative.racer.worker import RACERWorker
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class _FakeReq:
    def __init__(self, fill_ids, prefix_len=0):
        self._fill_ids = list(fill_ids)
        self.prefix_indices = list(range(prefix_len))

    def get_fill_ids(self):
        return self._fill_ids


class TestRacerPromptWarmupSelect(CustomTestCase):
    def test_last_occurrence_per_request(self):
        tokens = torch.tensor([10, 11, 11, 20, 21], dtype=torch.int64)
        rows, lens = RACERWorker._select_prompt_last_occurrence_rows(
            tokens, [3, 2], vocab_size=32
        )
        self.assertEqual(lens, [2, 2])
        selected = set(rows.tolist())
        self.assertEqual(selected, {0, 2, 3, 4})

    def test_skips_oob_negative_and_hash_ids(self):
        # 1_000_000 is the multimodal pad shift; -1 is a graph/sentinel hole.
        tokens = torch.tensor([7, -1, 1_000_000, 7, 32], dtype=torch.int64)
        rows, lens = RACERWorker._select_prompt_last_occurrence_rows(
            tokens, [5], vocab_size=32
        )
        self.assertEqual(lens, [1])
        self.assertEqual(rows.tolist(), [3])

    def test_all_invalid_ids_select_nothing(self):
        tokens = torch.tensor([-1, 99, 1_000_007], dtype=torch.int64)
        rows, lens = RACERWorker._select_prompt_last_occurrence_rows(
            tokens, [3], vocab_size=8
        )
        self.assertEqual(lens, [0])
        self.assertEqual(rows.numel(), 0)

    def test_prompt_warmup_token_ids_from_fill_ids(self):
        batch = SimpleNamespace(
            reqs=[
                _FakeReq([1, 2, 3, 4], prefix_len=1),
                _FakeReq([9, 8], prefix_len=0),
            ],
            input_ids=torch.tensor([999, 999, 999, 999], dtype=torch.int64),
        )
        got = RACERWorker._prompt_warmup_token_ids(batch, [3, 2], device="cpu")
        self.assertEqual(got.tolist(), [2, 3, 4, 9, 8])

    def test_prompt_warmup_token_ids_falls_back_to_input_ids(self):
        batch = SimpleNamespace(
            reqs=[_FakeReq([1], prefix_len=0)],
            input_ids=torch.tensor([4, 5, 6], dtype=torch.int64),
        )
        got = RACERWorker._prompt_warmup_token_ids(batch, [3], device="cpu")
        self.assertEqual(got.tolist(), [4, 5, 6])


if __name__ == "__main__":
    unittest.main()
