"""Unit tests for srt/mem_cache/allocation_sizing."""

import unittest
from types import SimpleNamespace

from sglang.srt.mem_cache.allocation_sizing import (
    get_alloc_reserve_per_decode,
    get_req_to_token_extra_context_len,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _spec_args(*, page_size, num_draft_tokens, num_steps=None, topk=1):
    return SimpleNamespace(
        speculative_algorithm="DSPARK",
        speculative_num_steps=num_steps,
        speculative_eagle_topk=topk,
        max_speculative_num_draft_tokens=num_draft_tokens,
        page_size=page_size,
    )


class TestReqToTokenRowHeadroom(CustomTestCase):
    def test_row_headroom_covers_decode_reserve_at_page_size_1(self):
        """Spec v2 reserves committed + 2*draft_tokens per decode step, so the
        req_to_token row must hold that watermark near the context limit.
        Regression: the reserve-sized headroom was gated on page_size > 1, so
        at page_size=1 a request generating to the context boundary made the
        row write spill into the neighbor row and the release slice clamp,
        stranding full-pool slots (post-eval leak on DSPARK, gamma=16)."""
        args = _spec_args(page_size=1, num_draft_tokens=17)
        self.assertGreaterEqual(
            get_req_to_token_extra_context_len(args),
            get_alloc_reserve_per_decode(args),
        )

    def test_row_headroom_covers_aligned_reserve_at_page_size_gt_1(self):
        # Paged reserve overshoots by up to page_size - 1 after alignment.
        args = _spec_args(page_size=64, num_draft_tokens=17)
        self.assertGreaterEqual(
            get_req_to_token_extra_context_len(args),
            get_alloc_reserve_per_decode(args) + args.page_size - 1,
        )

    def test_non_spec_headroom_unchanged(self):
        args = SimpleNamespace(
            speculative_algorithm=None,
            speculative_num_steps=None,
            speculative_eagle_topk=None,
            max_speculative_num_draft_tokens=None,
            page_size=1,
        )
        self.assertEqual(get_req_to_token_extra_context_len(args), 4)


if __name__ == "__main__":
    unittest.main()
