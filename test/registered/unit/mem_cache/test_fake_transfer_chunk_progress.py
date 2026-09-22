"""Chunk continuation when fake PD transfer skips shared radix insertion."""

import unittest
from types import SimpleNamespace as NS
from unittest.mock import Mock

from sglang.srt.mem_cache.common import checkpoint_kv_cache
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestFakeTransferChunkProgress(CustomTestCase):
    def test_chunk_progress_without_shared_insert(self):
        cache = NS(checkpoint=Mock())
        req = NS(
            rid="fake-transfer",
            finished=lambda: False,
            skip_radix_cache_insert=True,
            kv=NS(cache_protected_len=0),
            extend_end=16384,
            prefix_len=0,
        )
        checkpoint_kv_cache(req, cache)
        self.assertEqual(16385 - req.prefix_len, 1)
        self.assertEqual(req.kv.cache_protected_len, 0)
        cache.checkpoint.assert_not_called()
        req.extend_end = 16385
        checkpoint_kv_cache(req, cache)
        self.assertEqual(req.prefix_len, 16385)
        cache.checkpoint.assert_not_called()

    def test_real_transfer_preserves_cache_path(self):
        req = NS(
            rid="real-transfer",
            finished=lambda: False,
            skip_radix_cache_insert=False,
            extend_end=16384,
            prefix_len=0,
        )
        cache = NS(checkpoint=Mock())
        checkpoint_kv_cache(req, cache)
        self.assertEqual(req.prefix_len, 16384)
        cache.checkpoint.assert_called_once_with(req, up_to=16384)


if __name__ == "__main__":
    unittest.main()
