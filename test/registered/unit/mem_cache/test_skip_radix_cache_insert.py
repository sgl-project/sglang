"""Regression coverage for chunked prefill with radix insertion disabled."""

import unittest
from array import array
from types import SimpleNamespace
from unittest.mock import MagicMock

import torch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.managers.schedule_policy import PrefillAdder  # noqa: E402
from sglang.srt.mem_cache.chunk_cache import ChunkCache  # noqa: E402
from sglang.srt.mem_cache.common import maybe_cache_unfinished_req  # noqa: E402
from sglang.srt.mem_cache.pure_swa_radix_cache import PureSWARadixCache  # noqa: E402
from sglang.srt.mem_cache.radix_cache import RadixCache  # noqa: E402
from sglang.srt.mem_cache.swa_radix_cache import SWARadixCache  # noqa: E402

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


class TestSkipRadixCacheInsert(CustomTestCase):
    def test_chunked_request_advances_across_scheduling_rounds(self):
        chunk_size = 1024
        num_tokens = 3 * chunk_size
        kv_indices = torch.arange(num_tokens).unsqueeze(0)
        cache = RadixCache.__new__(RadixCache)
        cache.disable = False
        cache.evictable_size_ = 0
        cache.req_to_token_pool = SimpleNamespace(req_to_token=kv_indices)

        req = SimpleNamespace(
            skip_radix_cache_insert=True,
            kv=SimpleNamespace(req_pool_idx=0, cache_protected_len=0),
            prefix_indices=torch.empty(0, dtype=torch.int64),
            full_untruncated_fill_ids=array("q", range(num_tokens)),
            output_ids=array("q"),
            sampling_params=SimpleNamespace(max_new_tokens=1),
            retracted_stain=False,
        )
        req.set_extend_range = lambda start, end: setattr(
            req,
            "extend_range",
            SimpleNamespace(start=start, end=end, length=end - start),
        )
        req.get_fill_ids = lambda: req.full_untruncated_fill_ids[: req.extend_range.end]

        starts = []
        chunked_req = req
        for _ in range(3):
            allocator = MagicMock()
            allocator.available_size.return_value = num_tokens + chunk_size
            adder = PrefillAdder(
                page_size=1,
                tree_cache=cache,
                token_to_kv_pool_allocator=allocator,
                running_batch=None,
                new_token_ratio=0,
                rem_input_tokens=num_tokens,
                rem_chunk_tokens=chunk_size,
            )

            chunked_req = adder.add_chunked_req(chunked_req)
            starts.append(req.extend_range.start)
            if chunked_req is not None:
                maybe_cache_unfinished_req(req, cache, chunked=True)

        self.assertEqual(starts, [0, chunk_size, 2 * chunk_size])
        self.assertIsNone(chunked_req)

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
            kv=SimpleNamespace(req_pool_idx=0, cache_protected_len=2),
            prefix_indices=kv_indices[0, :2].clone(),
            last_node=last_node,
            origin_input_ids=array("q", [1, 2, 3, 4, 5, 6]),
            output_ids=array("q"),
            extra_key=None,
            cache_salt=None,
            get_fill_ids=lambda: array("q", [1, 2, 3, 4, 5, 6]),
        )

        maybe_cache_unfinished_req(req, cache, chunked=True)

        torch.testing.assert_close(req.prefix_indices, kv_indices[0])
        self.assertEqual(req.kv.cache_protected_len, 2)
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
            kv=SimpleNamespace(req_pool_idx=0),
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
                    kv=SimpleNamespace(req_pool_idx=0, cache_protected_len=2),
                    prefix_indices=kv_indices[0, :2].clone(),
                    extend_range=SimpleNamespace(end=6),
                    get_fill_ids=lambda: array("q", [1, 2, 3, 4, 5, 6]),
                )

                maybe_cache_unfinished_req(req, cache, chunked=True)

                torch.testing.assert_close(req.prefix_indices, kv_indices[0])
                self.assertEqual(req.kv.cache_protected_len, 2)
                cache.insert.assert_not_called()


if __name__ == "__main__":
    unittest.main(verbosity=2)
