"""CPU unit tests for memory-cache allocation sizing."""

import unittest
from types import SimpleNamespace

from sglang.srt.mem_cache.allocation_sizing import (
    get_alloc_len_per_decode,
    get_alloc_reserve_per_decode,
    get_req_to_token_extra_context_len,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


def _server_args(**overrides):
    values = {
        "speculative_algorithm": None,
        "speculative_num_steps": None,
        "speculative_eagle_topk": None,
        "max_speculative_num_draft_tokens": 0,
        "page_size": 1,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


class TestAllocationSizing(CustomTestCase):
    def test_non_speculative_defaults(self):
        server_args = _server_args()

        self.assertEqual(get_alloc_len_per_decode(server_args), 1)
        self.assertEqual(get_alloc_reserve_per_decode(server_args), 2)
        self.assertEqual(get_req_to_token_extra_context_len(server_args), 4)

    def test_missing_tree_args_default_to_one(self):
        server_args = _server_args(speculative_algorithm="EAGLE")

        self.assertEqual(get_alloc_len_per_decode(server_args), 1)

    def test_page_size_one_uses_tree_width_or_draft_cap(self):
        server_args = _server_args(
            speculative_algorithm="EAGLE",
            speculative_num_steps=3,
            speculative_eagle_topk=4,
            max_speculative_num_draft_tokens=20,
        )

        self.assertEqual(get_alloc_len_per_decode(server_args), 20)

    def test_topk_one_does_not_round_to_pages(self):
        server_args = _server_args(
            speculative_algorithm="EAGLE",
            speculative_num_steps=5,
            speculative_eagle_topk=1,
            max_speculative_num_draft_tokens=7,
            page_size=16,
        )

        self.assertEqual(get_alloc_len_per_decode(server_args), 7)

    def test_eagle_rounds_each_topk_branch_to_pages(self):
        server_args = _server_args(
            speculative_algorithm="EAGLE",
            speculative_num_steps=5,
            speculative_eagle_topk=4,
            max_speculative_num_draft_tokens=6,
            page_size=16,
        )

        self.assertEqual(get_alloc_len_per_decode(server_args), 128)
        self.assertEqual(get_alloc_reserve_per_decode(server_args), 256)

    def test_ngram_and_req_to_token_context_sizing(self):
        ngram_args = _server_args(
            speculative_algorithm="NGRAM",
            speculative_num_steps=5,
            speculative_eagle_topk=4,
            max_speculative_num_draft_tokens=6,
            page_size=16,
        )
        eagle_args = _server_args(
            speculative_algorithm="EAGLE",
            speculative_num_steps=5,
            speculative_eagle_topk=4,
            max_speculative_num_draft_tokens=6,
            page_size=16,
        )

        self.assertEqual(get_alloc_len_per_decode(ngram_args), 20)
        self.assertEqual(get_req_to_token_extra_context_len(eagle_args), 271)


if __name__ == "__main__":
    unittest.main()
