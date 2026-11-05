"""A decode preallocation's required tokens: the owner's fill and each window's
tail, in whole pages, each plus the decode reservation. Every PD decode
request is sized this way before it is admitted."""

import unittest
from types import SimpleNamespace

from sglang.srt.disaggregation.decode import DecodePreallocQueue
from sglang.srt.runtime_context import get_context
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class TestPreallocRequired(CustomTestCase):
    def _queue(self):
        queue = DecodePreallocQueue.__new__(DecodePreallocQueue)
        queue.token_to_kv_pool_allocator = SimpleNamespace(page_size=4)
        queue.num_reserved_decode_tokens = 8
        queue._window_tails = lambda fill_len: {"window_16": min(fill_len, 16)}
        return queue

    def test_owner_fill_and_window_tails_in_whole_pages(self):
        """REGRESSION: the sizing called a fill-length method the queue does
        not have, so every PD decode request raised before admission."""
        # Ten prompt tokens and three outputs: the last output is the pending
        # decode input, so twelve tokens of KV are filled.
        req = SimpleNamespace(origin_input_ids=[0] * 10, output_ids=[1, 2, 3])
        for disable_radix_cache, window_reserved in ((False, 8), (True, 0)):
            with self.subTest(disable_radix_cache=disable_radix_cache):
                override = get_context().override_server_args(
                    disable_radix_cache=disable_radix_cache
                )
                override.install()
                try:
                    full, windows = self._queue()._prealloc_required(req)
                finally:
                    override.restore()
                self.assertEqual(full, 12 + 8)
                self.assertEqual(windows, {"window_16": 12 + window_reserved})


if __name__ == "__main__":
    unittest.main()
