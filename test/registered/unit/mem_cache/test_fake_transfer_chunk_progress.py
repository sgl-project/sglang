"""Chunk continuation when fake PD transfer skips shared radix insertion."""

import unittest
from types import SimpleNamespace as NS
from unittest.mock import Mock

import torch

from sglang.srt.mem_cache.common import maybe_cache_unfinished_req
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestFakeTransferChunkProgress(CustomTestCase):
    def test_chunk_progress_without_shared_insert(self):
        slots = torch.arange(16385, dtype=torch.int32).reshape(1, -1)
        cache = NS(
            req_to_token_pool=NS(req_to_token=slots), cache_unfinished_req=Mock()
        )
        req = NS(
            skip_radix_cache_insert=True,
            kv=NS(req_pool_idx=0, cache_protected_len=0),
            get_fill_ids=lambda: range(16384),
        )
        maybe_cache_unfinished_req(req, cache, chunked=True)
        self.assertEqual(16385 - len(req.prefix_indices), 1)
        self.assertEqual(req.kv.cache_protected_len, 0)
        cache.cache_unfinished_req.assert_not_called()
        self.assertEqual(req.prefix_indices.dtype, torch.int64)
        slots[0, 0] = -1
        self.assertEqual(req.prefix_indices[0].item(), 0)
        req.get_fill_ids = lambda: range(16385)
        maybe_cache_unfinished_req(req, cache, chunked=True)
        self.assertEqual(len(req.prefix_indices), 16385)

    def test_real_transfer_preserves_cache_path(self):
        req = NS(skip_radix_cache_insert=False)
        cache = NS(cache_unfinished_req=Mock())
        maybe_cache_unfinished_req(req, cache, chunked=True)
        cache.cache_unfinished_req.assert_called_once_with(req, chunked=True)


if __name__ == "__main__":
    unittest.main()
