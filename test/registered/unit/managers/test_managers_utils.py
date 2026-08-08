"""Unit tests for managers/utils.py — no server, no model loading."""

import unittest
from unittest.mock import Mock

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.managers.utils import get_alloc_len_per_decode, validate_input_length

register_cpu_ci(est_time=5, suite="stage-a-test-cpu")


def _make_req(input_ids):
    req = Mock()
    req.origin_input_ids = list(input_ids)
    return req


class TestValidateInputLength(CustomTestCase):

    def test_within_limit(self):
        req = _make_req(range(10))
        self.assertIsNone(
            validate_input_length(req, max_req_input_len=100, allow_auto_truncate=False)
        )

    def test_at_exact_boundary_returns_error(self):
        """The check is >=, so input_len == max triggers rejection."""
        req = _make_req(range(100))
        error = validate_input_length(
            req, max_req_input_len=100, allow_auto_truncate=False
        )
        self.assertIsNotNone(error)
        self.assertIn("100", error)

    def test_exceeds_limit_no_truncate(self):
        req = _make_req(range(200))
        error = validate_input_length(
            req, max_req_input_len=100, allow_auto_truncate=False
        )
        self.assertIn("200", error)

    def test_exceeds_limit_with_truncate(self):
        req = _make_req(range(200))
        result = validate_input_length(
            req, max_req_input_len=100, allow_auto_truncate=True
        )
        self.assertIsNone(result)
        self.assertEqual(len(req.origin_input_ids), 100)

    def test_within_limit_no_truncation_applied(self):
        req = _make_req(range(50))
        validate_input_length(req, max_req_input_len=100, allow_auto_truncate=True)
        self.assertEqual(len(req.origin_input_ids), 50)


def _make_server_args(
    speculative_algorithm=None,
    num_steps=None,
    eagle_topk=None,
    num_draft_tokens=1,
    page_size=1,
):
    args = Mock()
    args.speculative_algorithm = speculative_algorithm
    args.speculative_num_steps = num_steps
    args.speculative_eagle_topk = eagle_topk
    args.speculative_num_draft_tokens = num_draft_tokens
    args.page_size = page_size
    return args


class TestGetAllocLenPerDecode(CustomTestCase):

    def test_no_speculative_returns_one(self):
        args = _make_server_args()
        self.assertEqual(get_alloc_len_per_decode(args), 1)

    def test_steps_times_topk_larger(self):
        args = _make_server_args(
            speculative_algorithm="eagle",
            num_steps=3,
            eagle_topk=2,
            num_draft_tokens=5,
            page_size=1,
        )
        # max(3*2, 5) = 6
        self.assertEqual(get_alloc_len_per_decode(args), 6)

    def test_draft_tokens_larger(self):
        args = _make_server_args(
            speculative_algorithm="eagle",
            num_steps=2,
            eagle_topk=1,
            num_draft_tokens=10,
            page_size=1,
        )
        # max(2*1, 10) = 10
        self.assertEqual(get_alloc_len_per_decode(args), 10)

    def test_page_size_gt1_with_topk_gt1_raises(self):
        args = _make_server_args(
            speculative_algorithm="eagle",
            num_steps=2,
            eagle_topk=2,
            num_draft_tokens=5,
            page_size=4,
        )
        with self.assertRaises(NotImplementedError):
            get_alloc_len_per_decode(args)


if __name__ == "__main__":
    unittest.main()
