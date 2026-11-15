"""The req_to_token row has to hold what a request at its length cap allocates."""

import unittest
from types import SimpleNamespace

from sglang.srt.mem_cache.allocation_sizing import (
    get_alloc_reserve_per_decode,
    get_req_to_token_extra_context_len,
    page_aligned_decode_alloc_lens,
)
from sglang.srt.runtime_context import get_context, get_parallel
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

_CONTEXT_LEN = 4096
# A request that still decodes holds at most context_len - 3 tokens, and the
# newest one has no KV yet.
_MAX_KV_COMMITTED_LEN = _CONTEXT_LEN - 4


class TestReqToTokenRowHeadroom(CustomTestCase):
    def setUp(self):
        cm = get_parallel().override(attn_dcp_size=1)
        cm.__enter__()
        self.addCleanup(cm.__exit__, None, None, None)

    def test_row_holds_the_decode_allocation_at_the_length_cap(self):
        """Every speculative decode step allocates kv_committed_len plus the
        decode reserve, rounded up to a page, for a request at its length cap
        too. The reserve-sized headroom used to apply only at page_size > 1, so
        at page_size 1 the write ran past the row into the next request's row
        once the reserve outgrew 4 + num_draft_tokens: that request's KV
        mapping was overwritten, and the slots were never freed."""
        cases = (
            # algorithm, num_steps, eagle_topk, num_draft_tokens
            ("EAGLE", 3, 1, 4),
            ("EAGLE", 8, 1, 9),
            ("EAGLE3", 4, 4, 8),
            ("NGRAM", 12, 1, 12),
            ("DFLASH", None, None, 16),
            ("DSPARK", None, None, 8),
        )
        for page_size in (1, 16, 64):
            for algorithm, num_steps, eagle_topk, num_draft_tokens in cases:
                with (
                    self.subTest(
                        page_size=page_size,
                        algorithm=algorithm,
                        num_steps=num_steps,
                        eagle_topk=eagle_topk,
                    ),
                    get_context().override_server_args(
                        speculative_algorithm=algorithm,
                        speculative_num_steps=num_steps,
                        speculative_eagle_topk=eagle_topk,
                        speculative_num_draft_tokens=num_draft_tokens,
                        page_size=page_size,
                    ),
                ):
                    req = SimpleNamespace(
                        kv=SimpleNamespace(
                            kv_committed_len=_MAX_KV_COMMITTED_LEN, kv_allocated_len=0
                        )
                    )
                    _, (alloc_len,), _ = page_aligned_decode_alloc_lens(
                        [req],
                        reserve=get_alloc_reserve_per_decode(),
                        page_size=page_size,
                    )
                    self.assertLessEqual(
                        alloc_len, _CONTEXT_LEN + get_req_to_token_extra_context_len()
                    )


if __name__ == "__main__":
    unittest.main()
