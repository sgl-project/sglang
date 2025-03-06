"""
Usage:
python3 -m unittest test_overlap_schedule.TestOverlapSchedule.test_radix_attention_chunked_prefill
python3 test_overlap_schedule.py
"""

import unittest

from sglang.test.test_utils import run_mmlu_test


class TestOverlapSchedule(unittest.TestCase):
    def test_no_radix_attention_chunked_prefill(self):
        run_mmlu_test(
            disable_radix_cache=True, chunked_prefill_size=32, disable_overlap=True
        )

    def test_no_radix_attention_no_chunked_prefill(self):
        run_mmlu_test(
            disable_radix_cache=True, chunked_prefill_size=-1, disable_overlap=True
        )

    def test_radix_attention_chunked_prefill(self):
        run_mmlu_test(
            disable_radix_cache=False, chunked_prefill_size=32, disable_overlap=True
        )

    def test_radix_attention_no_chunked_prefill(self):
        run_mmlu_test(
            disable_radix_cache=False, chunked_prefill_size=-1, disable_overlap=True
        )


if __name__ == "__main__":
    unittest.main()
