"""Per-request EAGLE prefill tail-token substitution (PR #26329, pluralized).

For a non-final prefill chunk the draft model's chain must continue from the
next *prompt* token, not the sampled one. With concurrent chunked prefill,
several rows of one batch can need the substitution at once; the consumer
applies them all in a single index_copy_ instead of stopping at the first.
"""

import unittest
from types import SimpleNamespace

import torch

from sglang.srt.managers.schedule_batch import _compute_chunked_next_prompt_tokens
from sglang.srt.speculative.eagle_utils import _eagle_prefill_tail_tokens
from sglang.srt.utils.common import Range
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=6, suite="base-a-test-cpu")

VOCAB = 1000


def _batch(next_prompt_tokens, dtype=torch.int64):
    return SimpleNamespace(
        input_ids=torch.zeros(4, dtype=dtype),
        chunked_next_prompt_tokens=next_prompt_tokens,
    )


class _Req:
    """Hashable Req stand-in (the production code builds a set of chunked
    reqs; SimpleNamespace is unhashable)."""

    def __init__(self, fill_end, origin_ids):
        self.extend_range = Range(0, fill_end)
        self.origin_input_ids = origin_ids


def _req(fill_end, origin_ids):
    return _Req(fill_end, origin_ids)


class TestComputeChunkedNextPromptTokens(CustomTestCase):
    def test_no_chunked_reqs_returns_none(self):
        reqs = [_req(3, [1, 2, 3, 4])]
        self.assertIsNone(_compute_chunked_next_prompt_tokens(reqs, [], VOCAB))
        self.assertIsNone(_compute_chunked_next_prompt_tokens(reqs, None, VOCAB))

    def test_positionally_aligned_with_reqs(self):
        a, b, c, d = (
            _req(2, [10, 11, 12]),
            _req(2, [20, 21, 22]),
            _req(2, [30, 31, 32]),
            _req(2, [40, 41, 42]),
        )
        reqs = [a, b, c, d]
        tokens = _compute_chunked_next_prompt_tokens(reqs, [d, b], VOCAB)
        # Membership, not position or carry order, decides which rows map.
        self.assertEqual(tokens, [None, 22, None, 42])

    def test_final_chunk_boundary_maps_to_none(self):
        r = _req(3, [10, 11, 12])  # fill consumed the whole prompt
        self.assertIsNone(_compute_chunked_next_prompt_tokens([r], [r], VOCAB))

    def test_placeholder_token_maps_to_none(self):
        # Multimodal hash tokens lie outside the model vocab and must not be
        # fed to the draft chain.
        r = _req(1, [10, 5_000_000, 12])
        self.assertIsNone(_compute_chunked_next_prompt_tokens([r], [r], VOCAB))


class TestEaglePrefillTailTokens(CustomTestCase):
    def test_no_map_returns_uncopied_conversion(self):
        next_token_ids = torch.tensor([7, 8, 9, 10], dtype=torch.int64)
        result = _eagle_prefill_tail_tokens(_batch(None), next_token_ids)
        # .to() with a matching dtype is the identity: no clone on the
        # no-substitution path.
        self.assertIs(result, next_token_ids)

    def test_all_none_rows_returns_uncopied_conversion(self):
        next_token_ids = torch.tensor([7, 8, 9, 10], dtype=torch.int64)
        result = _eagle_prefill_tail_tokens(
            _batch([None, None, None, None]), next_token_ids
        )
        self.assertIs(result, next_token_ids)

    def test_single_row_matches_stock_n1_behavior(self):
        next_token_ids = torch.tensor([7, 8, 9, 10], dtype=torch.int64)
        result = _eagle_prefill_tail_tokens(
            _batch([None, None, 555, None]), next_token_ids
        )
        self.assertIsNot(result, next_token_ids)
        self.assertEqual(result.tolist(), [7, 8, 555, 10])
        # The source tensor is not mutated through the clone.
        self.assertEqual(next_token_ids.tolist(), [7, 8, 9, 10])

    def test_multiple_rows_all_substituted(self):
        # The N>1 case: every mid-prefill row must be substituted, not just
        # the first -- the stock loop broke after the single slot's row.
        next_token_ids = torch.tensor([7, 8, 9, 10], dtype=torch.int64)
        result = _eagle_prefill_tail_tokens(
            _batch([111, None, 555, 999]), next_token_ids
        )
        self.assertEqual(result.tolist(), [111, 8, 555, 999])

    def test_dtype_follows_input_ids(self):
        next_token_ids = torch.tensor([7, 8, 9, 10], dtype=torch.int64)
        result = _eagle_prefill_tail_tokens(
            _batch([111, None, None, None], dtype=torch.int32), next_token_ids
        )
        self.assertEqual(result.dtype, torch.int32)
        self.assertEqual(result.tolist(), [111, 8, 9, 10])


if __name__ == "__main__":
    unittest.main()
