import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock

import torch

from sglang.srt.mem_cache.allocation import alloc_for_spec_decode
from sglang.srt.mem_cache.allocation_sizing import (
    decode_alloc_targets,
    get_req_to_token_row_width,
)
from sglang.srt.runtime_context import get_schedule, get_spec, reset_context
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase
from sglang.test.unified_allocator_fixtures import setup_allocator_context

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

# (algorithm, num_steps, topk, num_draft_tokens); None is non-speculative decode.
_DECODE_CONFIGS = [
    (None, None, None, None),
    ("EAGLE", 3, 1, 4),
    ("EAGLE", 5, 8, 64),
    ("NGRAM", 1, 8, 16),
    ("UNO", 1, 1, 8),
]


class TestRowHoldsDecodeAllocation(CustomTestCase):
    def setUp(self):
        super().setUp()
        setup_allocator_context()
        self.addCleanup(reset_context)

    def test_decode_allocation_fits_the_row(self):
        for algorithm, steps, topk, draft in _DECODE_CONFIGS:
            for page_size in (1, 16, 64):
                with (
                    self.subTest(
                        algorithm=algorithm,
                        steps=steps,
                        topk=topk,
                        draft=draft,
                        page_size=page_size,
                    ),
                    get_spec().override(
                        speculative_algorithm=algorithm,
                        speculative_num_steps=steps,
                        speculative_eagle_topk=topk,
                        speculative_num_draft_tokens=draft,
                    ),
                    get_schedule().override(page_size=page_size),
                ):
                    # A request at the context length, at every page phase of it.
                    context_lens = range(1000, 1000 + page_size)
                    reqs = [
                        SimpleNamespace(
                            kv=SimpleNamespace(
                                kv_committed_len=context_len,
                                kv_allocated_len=context_len,
                            )
                        )
                        for context_len in context_lens
                    ]
                    _, target_lens, _ = decode_alloc_targets(reqs, page_size=page_size)
                    for context_len, target_len in zip(context_lens, target_lens):
                        self.assertLessEqual(
                            target_len, get_req_to_token_row_width(context_len)
                        )


class TestSpecDecodeAllocationRowGuard(CustomTestCase):
    def _alloc(self, tree_cache, req_to_token, *, nxt_kv_len: int, num_needed: int):
        alloc_for_spec_decode(
            tree_cache,
            SimpleNamespace(req_to_token=req_to_token, row_width=req_to_token.shape[1]),
            reqs=[SimpleNamespace(kv=MagicMock())],
            req_pool_indices=torch.tensor([0]),
            cur_kv_lens=torch.tensor([0]),
            cur_kv_lens_cpu=torch.tensor([0]),
            nxt_kv_lens=torch.tensor([nxt_kv_len]),
            nxt_kv_lens_cpu=torch.tensor([nxt_kv_len]),
            num_needed_tokens=num_needed,
        )

    def test_allocation_past_the_row_raises_before_allocating(self):
        tree_cache = MagicMock()
        req_to_token = torch.zeros((2, 8), dtype=torch.int32)
        # Filling the row exactly is legal.
        self._alloc(tree_cache, req_to_token, nxt_kv_len=8, num_needed=0)
        with self.assertRaisesRegex(RuntimeError, "row width"):
            self._alloc(tree_cache, req_to_token, nxt_kv_len=9, num_needed=9)
        self.assertEqual(tree_cache.mock_calls, [])
        self.assertEqual(int(req_to_token.count_nonzero()), 0)


if __name__ == "__main__":
    unittest.main()
