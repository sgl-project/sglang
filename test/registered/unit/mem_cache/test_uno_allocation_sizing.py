"""Unit tests for UNO allocation sizing."""

import unittest

from sglang.srt.mem_cache.allocation_sizing import (
    get_alloc_len_per_decode,
    get_alloc_reserve_per_decode,
    get_req_to_token_extra_context_len,
)
from sglang.srt.runtime_context import get_context, get_parallel
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class TestUnoAllocationSizing(CustomTestCase):
    def test_page_size_one_row_covers_decode_reserve(self):
        with (
            get_context().override_server_args(
                speculative_algorithm="UNO",
                speculative_num_draft_tokens=8,
                page_size=1,
            ),
            get_parallel().override(attn_dcp_size=1),
        ):
            self.assertEqual(get_alloc_len_per_decode(), 9)
            self.assertEqual(get_alloc_reserve_per_decode(), 18)
            self.assertGreaterEqual(
                get_req_to_token_extra_context_len(),
                get_alloc_reserve_per_decode(),
            )


if __name__ == "__main__":
    unittest.main()
