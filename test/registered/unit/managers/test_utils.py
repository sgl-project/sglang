"""Unit tests for sglang.srt.managers.utils — no server, no model loading."""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

import unittest
from unittest.mock import MagicMock

from sglang.srt.managers.utils import (
    GenerationBatchResult,
    get_alloc_len_per_decode,
    validate_input_length,
)


class TestValidateInputLength(unittest.TestCase):
    """Test validate_input_length with various input and config combos."""

    def _make_req(self, input_len: int) -> MagicMock:
        req = MagicMock()
        req.origin_input_ids = list(range(input_len))
        return req

    # --- normal cases ---

    def test_within_limit_returns_none(self):
        req = self._make_req(100)
        result = validate_input_length(
            req, max_req_input_len=200, allow_auto_truncate=False
        )
        self.assertIsNone(result)

    def test_exactly_at_limit_returns_none(self):
        """Input length == max should be accepted."""
        req = self._make_req(200)
        result = validate_input_length(
            req, max_req_input_len=200, allow_auto_truncate=False
        )
        self.assertIsNone(result)

    def test_exceeds_limit_without_truncate_returns_error(self):
        req = self._make_req(300)
        result = validate_input_length(
            req, max_req_input_len=200, allow_auto_truncate=False
        )
        self.assertIsNotNone(result)
        self.assertIn("300", result)
        self.assertIn("200", result)

    # --- truncation ---

    def test_exceeds_limit_with_truncate_truncates(self):
        req = self._make_req(300)
        result = validate_input_length(
            req, max_req_input_len=200, allow_auto_truncate=True
        )
        self.assertIsNone(result)
        self.assertEqual(len(req.origin_input_ids), 200)

    def test_truncate_preserves_token_order(self):
        req = self._make_req(10)
        validate_input_length(req, max_req_input_len=5, allow_auto_truncate=True)
        self.assertEqual(req.origin_input_ids, [0, 1, 2, 3, 4])

    # --- edge cases ---

    def test_empty_input_passes(self):
        req = self._make_req(0)
        result = validate_input_length(
            req, max_req_input_len=200, allow_auto_truncate=False
        )
        self.assertIsNone(result)

    def test_single_token_at_limit_passes(self):
        req = self._make_req(1)
        result = validate_input_length(
            req, max_req_input_len=1, allow_auto_truncate=False
        )
        self.assertIsNone(result)


class TestGetAllocLenPerDecode(unittest.TestCase):
    """Test get_alloc_len_per_decode with various server_args configs."""

    def _make_server_args(self, **kwargs) -> MagicMock:
        args = MagicMock()
        args.speculative_algorithm = kwargs.get("speculative_algorithm", None)
        args.speculative_num_steps = kwargs.get("speculative_num_steps", None)
        args.speculative_eagle_topk = kwargs.get("speculative_eagle_topk", None)
        args.speculative_num_draft_tokens = kwargs.get(
            "speculative_num_draft_tokens", 0
        )
        args.page_size = kwargs.get("page_size", 1)
        return args

    def test_no_speculation_returns_one(self):
        args = self._make_server_args()
        result = get_alloc_len_per_decode(args)
        self.assertEqual(result, 1)

    def test_basic_speculation(self):
        args = self._make_server_args(
            speculative_algorithm="eagle",
            speculative_num_steps=5,
            speculative_eagle_topk=2,
            speculative_num_draft_tokens=8,
        )
        result = get_alloc_len_per_decode(args)
        # max(5*2, 8) = 10
        self.assertEqual(result, 10)

    def test_draft_tokens_larger(self):
        args = self._make_server_args(
            speculative_algorithm="eagle",
            speculative_num_steps=2,
            speculative_eagle_topk=1,
            speculative_num_draft_tokens=20,
        )
        result = get_alloc_len_per_decode(args)
        # max(2*1, 20) = 20
        self.assertEqual(result, 20)

    def test_topk_one_with_larger_page_size(self):
        args = self._make_server_args(
            speculative_algorithm="eagle",
            speculative_num_steps=3,
            speculative_eagle_topk=1,
            speculative_num_draft_tokens=5,
            page_size=4,
        )
        result = get_alloc_len_per_decode(args)
        # page_size > 1 but topk == 1, so no error
        self.assertEqual(result, 5)

    def test_page_size_gt1_and_topk_gt1_raises(self):
        args = self._make_server_args(
            speculative_algorithm="eagle",
            speculative_num_steps=3,
            speculative_eagle_topk=2,
            speculative_num_draft_tokens=5,
            page_size=4,
        )
        with self.assertRaises(NotImplementedError):
            get_alloc_len_per_decode(args)


class TestGenerationBatchResult(unittest.TestCase):
    """Test GenerationBatchResult dataclass basics."""

    def test_default_fields(self):
        result = GenerationBatchResult()
        self.assertIsNone(result.logits_output)
        self.assertIsNone(result.next_token_ids)
        self.assertEqual(result.num_correct_drafts, 0)
        self.assertFalse(result.can_run_cuda_graph)

    def test_custom_fields(self):
        result = GenerationBatchResult(
            num_correct_drafts=5,
            can_run_cuda_graph=True,
        )
        self.assertEqual(result.num_correct_drafts, 5)
        self.assertTrue(result.can_run_cuda_graph)


if __name__ == "__main__":
    unittest.main()
