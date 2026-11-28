"""CPU checks for the GPU result comparator; no model download required."""

import copy
import unittest

from validate_fast_dvlm import compare


def artifact():
    return {
        "schema": 1,
        "model": "Efficient-Large-Model/Fast_dVLM_3B",
        "revision": "a" * 40,
        "sampling": {"temperature": 0, "max_new_tokens": 8},
        "records": [
            {
                "id": str(i),
                "input_ids": [1, 2, i],
                "output_ids": [3, 4],
                "finish_reason": {"type": "length", "length": 8},
                "cached_tokens": 32,
            }
            for i in range(2)
        ],
    }


class TestComparison(unittest.TestCase):
    def test_identical_results_can_arrive_in_a_different_order(self):
        a, b = artifact(), artifact()
        b["records"].reverse()
        self.assertEqual(compare(a, b, True)["exact"], 2)

    def test_missing_result_is_an_error(self):
        b = artifact()
        b["records"].pop()
        with self.assertRaisesRegex(ValueError, "different prompt IDs"):
            compare(artifact(), b)

    def test_duplicate_id_is_an_error(self):
        b = artifact()
        b["records"][1]["id"] = b["records"][0]["id"]
        with self.assertRaisesRegex(ValueError, "unique"):
            compare(artifact(), b)

    def test_missing_token_ids_cannot_fall_back_to_text(self):
        b = artifact()
        del b["records"][0]["output_ids"]
        b["records"][0]["text"] = "same decoded text"
        with self.assertRaisesRegex(ValueError, "output token IDs"):
            compare(artifact(), b)

    def test_different_inputs_invalidate_comparison(self):
        b = artifact()
        b["records"][0]["input_ids"][0] = 5
        with self.assertRaisesRegex(ValueError, "input token IDs"):
            compare(artifact(), b)

    def test_checkpoint_and_sampling_must_match(self):
        for field, value in (("revision", "b" * 40), ("sampling", {"temperature": 1})):
            b = artifact()
            b[field] = value
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, field):
                compare(artifact(), b)

    def test_token_or_length_difference_is_reported(self):
        for tokens, first in (([3, 5], 1), ([3, 4, 5], 2)):
            b = artifact()
            b["records"][0]["output_ids"] = tokens
            report = compare(artifact(), b)
            self.assertEqual(report["exact"], 1)
            self.assertEqual(report["differences"][0]["first_difference"], first)

    def test_cache_hit_requirement_rejects_absent_zero_and_negative_counts(self):
        for count in (None, 0, -1, True):
            b = artifact()
            b["records"][0]["cached_tokens"] = count
            with (
                self.subTest(count=count),
                self.assertRaisesRegex(ValueError, "cache hit"),
            ):
                compare(artifact(), b, True)

    def test_aborted_request_is_not_an_exact_match(self):
        a = artifact()
        a["records"][0]["finish_reason"] = {"type": "abort"}
        with self.assertRaisesRegex(ValueError, "finish_reason"):
            compare(a, copy.deepcopy(a))


if __name__ == "__main__":
    unittest.main()
