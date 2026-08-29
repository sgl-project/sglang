"""Regression coverage for chunked prefill with radix insertion disabled."""

import unittest
from array import array
from types import SimpleNamespace
from unittest.mock import MagicMock

import torch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.mem_cache.chunk_cache import ChunkCache  # noqa: E402
from sglang.srt.mem_cache.common import maybe_cache_unfinished_req  # noqa: E402
from sglang.srt.mem_cache.pure_swa_radix_cache import PureSWARadixCache  # noqa: E402
from sglang.srt.mem_cache.radix_cache import RadixCache  # noqa: E402
from sglang.srt.mem_cache.swa_radix_cache import SWARadixCache  # noqa: E402

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


class TestSkipRadixCacheInsert(CustomTestCase):
    def test_chunked_request_advances_without_inserting(self):
        kv_indices = torch.tensor([[101, 102, 103, 104, 105, 106]])
        cache = RadixCache.__new__(RadixCache)
        cache.disable = False
        cache.disable_finished_insert = False
        cache.page_size = 1
        cache.is_eagle = False
        cache.req_to_token_pool = SimpleNamespace(req_to_token=kv_indices)
        cache.token_to_kv_pool_allocator = MagicMock()
        cache.insert = MagicMock()
        cache.match_prefix = MagicMock()
        cache.dec_lock_ref = MagicMock()
        cache.inc_lock_ref = MagicMock()

        last_node = object()
        req = SimpleNamespace(
            skip_radix_cache_insert=True,
            req_pool_idx=0,
            prefix_indices=kv_indices[0, :2].clone(),
            cache_protected_len=2,
            last_node=last_node,
            origin_input_ids=array("q", [1, 2, 3, 4, 5, 6]),
            output_ids=array("q"),
            extra_key=None,
            cache_salt=None,
            get_fill_ids=lambda: array("q", [1, 2, 3, 4, 5, 6]),
        )

        maybe_cache_unfinished_req(req, cache, chunked=True)

        torch.testing.assert_close(req.prefix_indices, kv_indices[0])
        self.assertEqual(req.cache_protected_len, 2)
        self.assertIs(req.last_node, last_node)
        cache.insert.assert_not_called()
        cache.match_prefix.assert_not_called()
        cache.token_to_kv_pool_allocator.free_segment.assert_not_called()
        cache.dec_lock_ref.assert_not_called()
        cache.inc_lock_ref.assert_not_called()

        cache.cache_finished_req(req, is_insert=False, kv_len_to_handle=6)

        freed_segments = cache.token_to_kv_pool_allocator.free_segments.call_args.args[
            0
        ]
        torch.testing.assert_close(freed_segments[0][0], kv_indices[0, 2:])
        self.assertEqual(freed_segments[0][1], 2)
        self.assertEqual(freed_segments[1][0].numel(), 0)
        cache.dec_lock_ref.assert_called_once_with(last_node)

    def test_chunk_cache_still_advances_with_skip_flag(self):
        kv_indices = torch.tensor([[201, 202, 203, 204, 205, 206]])
        cache = ChunkCache.__new__(ChunkCache)
        cache.req_to_token_pool = SimpleNamespace(req_to_token=kv_indices)
        req = SimpleNamespace(
            skip_radix_cache_insert=True,
            req_pool_idx=0,
            prefix_indices=kv_indices[0, :2].clone(),
            extend_range=SimpleNamespace(end=6),
        )

        maybe_cache_unfinished_req(req, cache, chunked=True)

        torch.testing.assert_close(req.prefix_indices, kv_indices[0])

    def test_swa_backends_advance_without_publishing(self):
        """Skipping publication must still advance chunked-prefill ownership."""
        kv_indices = torch.tensor([[301, 302, 303, 304, 305, 306]])

        for cache_cls in (PureSWARadixCache, SWARadixCache):
            with self.subTest(cache_cls=cache_cls.__name__):
                cache = cache_cls.__new__(cache_cls)
                cache.disable = False
                cache.req_to_token_pool = SimpleNamespace(req_to_token=kv_indices)
                cache.insert = MagicMock()
                req = SimpleNamespace(
                    skip_radix_cache_insert=True,
                    req_pool_idx=0,
                    prefix_indices=kv_indices[0, :2].clone(),
                    cache_protected_len=2,
                    extend_range=SimpleNamespace(end=6),
                    get_fill_ids=lambda: array("q", [1, 2, 3, 4, 5, 6]),
                )

                maybe_cache_unfinished_req(req, cache, chunked=True)

                torch.testing.assert_close(req.prefix_indices, kv_indices[0])
                self.assertEqual(req.cache_protected_len, 2)
                cache.insert.assert_not_called()


if __name__ == "__main__":
    unittest.main(verbosity=2)
