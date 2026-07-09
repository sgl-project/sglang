import unittest

import torch

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _shift_in_place(input_ids, extend_lens, tail_tokens):
    input_ids = input_ids.clone()
    pt = 0
    for i, extend_len in enumerate(extend_lens):
        chunk = input_ids[pt : pt + extend_len]
        input_ids[pt : pt + extend_len] = torch.cat(
            (chunk[1:], tail_tokens[i].reshape(1))
        )
        pt += extend_len
    return input_ids


def _shift_out_of_place(input_ids, extend_lens, tail_tokens):
    new_input_ids = torch.empty_like(input_ids)
    pt = 0
    for i, extend_len in enumerate(extend_lens):
        chunk = input_ids[pt : pt + extend_len]
        new_input_ids[pt : pt + extend_len].copy_(
            torch.cat((chunk[1:], tail_tokens[i].reshape(1)))
        )
        pt += extend_len
    assert pt == input_ids.numel()
    return new_input_ids


class TestEagleDraftExtendInputIds(unittest.TestCase):
    def _assert_equivalent(self, extend_lens):
        total = sum(extend_lens)
        input_ids = torch.arange(100, 100 + total, dtype=torch.int64)
        tail_tokens = torch.arange(900, 900 + len(extend_lens), dtype=torch.int64)

        old_result = _shift_in_place(input_ids, extend_lens, tail_tokens)
        new_result = _shift_out_of_place(input_ids, extend_lens, tail_tokens)

        self.assertTrue(torch.equal(old_result, new_result))

    def test_single_request(self):
        """Out-of-place shift matches the old in-place shift for one request."""
        self._assert_equivalent([5])

    def test_multiple_requests_unequal_lengths(self):
        """Out-of-place shift matches in-place per-request boundaries for unequal extend lens."""
        self._assert_equivalent([3, 7, 1, 4])

    def test_extend_len_one(self):
        """A length-1 extend keeps only the tail token, identically in both forms."""
        self._assert_equivalent([1, 1])

    def test_source_tensor_not_mutated(self):
        """The out-of-place form must leave the source input_ids tensor untouched."""
        extend_lens = [2, 3]
        input_ids = torch.arange(5, dtype=torch.int64)
        snapshot = input_ids.clone()
        tail_tokens = torch.tensor([50, 51], dtype=torch.int64)

        result = _shift_out_of_place(input_ids, extend_lens, tail_tokens)

        self.assertTrue(torch.equal(input_ids, snapshot))
        self.assertEqual(result.numel(), input_ids.numel())


if __name__ == "__main__":
    unittest.main()
