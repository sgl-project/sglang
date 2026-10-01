"""--strip-thinking-cache caps Req.owned_kv_len() at the prompt, so a finished
reasoning request's output is freed instead of inserted. A retracted request is
re-prefilled over prompt + output, and that prefill's cache_unfinished_req puts
the output into the tree (cache_protected_len > prompt). Capping below
cache_protected_len then frees slots the tree still owns: they end up in the
free list twice and can be handed to two requests."""

import unittest
from array import array

import torch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.managers.schedule_batch import Req  # noqa: E402
from sglang.srt.mem_cache.allocator import TokenToKVPoolAllocator  # noqa: E402
from sglang.srt.mem_cache.allocator.paged import (  # noqa: E402
    PagedTokenToKVPoolAllocator,
)
from sglang.srt.mem_cache.base_prefix_cache import (  # noqa: E402
    DecLockRefParams,
    EvictParams,
)
from sglang.srt.mem_cache.cache_init_params import CacheInitParams  # noqa: E402
from sglang.srt.mem_cache.common import release_kv_cache  # noqa: E402
from sglang.srt.mem_cache.memory_pool import (  # noqa: E402
    MHATokenToKVPool,
    ReqToTokenPool,
)
from sglang.srt.mem_cache.unified_cache.components.base import (  # noqa: E402
    ComponentType,
)
from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache  # noqa: E402
from sglang.srt.runtime_context import get_serving  # noqa: E402
from sglang.srt.sampling.sampling_params import SamplingParams  # noqa: E402
from sglang.srt.server_args import (  # noqa: E402
    ServerArgs,
    set_global_server_args_for_scheduler,
)

register_cpu_ci(est_time=6, suite="base-a-test-cpu")

POOL = 128
PROMPT = 8
BEFORE_RETRACT = 6  # output tokens generated before the retraction
AFTER = 3  # decode steps after the re-prefill


class TestStripThinkingAfterRetraction(CustomTestCase):
    def setUp(self):
        set_global_server_args_for_scheduler(
            ServerArgs(model_path="dummy", page_size=1)
        )
        self.pool = ReqToTokenPool(
            size=4, max_context_len=64, device="cpu", enable_memory_saver=False
        )
        kvcache = MHATokenToKVPool(
            size=POOL,
            page_size=1,
            dtype=torch.float16,
            head_num=1,
            head_dim=8,
            layer_num=1,
            device="cpu",
            enable_memory_saver=False,
        )
        self.allocator = TokenToKVPoolAllocator(
            size=POOL,
            dtype=torch.float16,
            device="cpu",
            kvcache=kvcache,
            need_sort=False,
        )
        self.cache = UnifiedRadixCache(
            params=CacheInitParams(
                req_to_token_pool=self.pool,
                token_to_kv_pool_allocator=self.allocator,
                page_size=1,
                disable=False,
                tree_components=(ComponentType.FULL,),
            )
        )

    def _prefill_and_decode(self, earlier_output):
        """Prefill prompt + earlier_output (a re-prefill when non-empty), cache it
        as the scheduler does for an unfinished request, then decode AFTER tokens."""
        req = Req(
            "r",
            "",
            array("q", range(1, PROMPT + 1)),
            SamplingParams(temperature=0, max_new_tokens=32),
        )
        req.output_ids = array("q", earlier_output)
        req.reasoning_tokens = len(earlier_output) + 1  # still thinking
        req.last_node = self.cache.root_node_handle()
        req.lock_receipt = DecLockRefParams()
        req.extra_key = None

        self.pool.alloc([req])
        fill = PROMPT + len(earlier_output)
        req.full_untruncated_fill_ids = req.origin_input_ids + req.output_ids
        req.prefix_indices = torch.empty((0,), dtype=torch.int64)
        req.set_extend_range(0, fill)
        slots = self.allocator.alloc(fill)
        self.pool.write((req.kv.req_pool_idx, slice(0, fill)), slots.to(torch.int32))
        req.kv.kv_committed_len = req.kv.kv_allocated_len = fill
        self.cache.cache_unfinished_req(req)  # batch_result_processor, unfinished

        for _ in range(AFTER):
            slot = self.allocator.alloc(1)
            self.pool.write(
                (req.kv.req_pool_idx, req.kv.kv_committed_len), slot.to(torch.int32)
            )
            req.kv.kv_committed_len += 1
            req.kv.kv_allocated_len += 1
            req.output_ids.append(99)
        return req

    def _release_and_check(self, req, is_insert):
        tree_slots = set(
            self.pool.req_to_token[
                req.kv.req_pool_idx, : req.kv.cache_protected_len
            ].tolist()
        )
        with get_serving().override(strip_thinking_cache=True):
            release_kv_cache(req, self.cache, is_insert=is_insert)

        free = self.allocator.free_pages.tolist()
        self.assertEqual(sorted(tree_slots & set(free)), [])
        self.cache.evict(EvictParams(num_tokens=POOL))
        free = self.allocator.free_pages.tolist()
        self.assertEqual(len(free), len(set(free)), "a KV slot is free twice")
        self.assertEqual(self.allocator.available_size(), POOL)

    def test_finish_after_reprefill_frees_each_slot_once(self):
        req = self._prefill_and_decode(list(range(100, 100 + BEFORE_RETRACT)))
        self.assertGreater(req.kv.cache_protected_len, PROMPT)
        self._release_and_check(req, is_insert=True)

    def test_second_retraction_frees_each_slot_once(self):
        req = self._prefill_and_decode(list(range(100, 100 + BEFORE_RETRACT)))
        self._release_and_check(req, is_insert=False)

    def test_without_retraction_output_is_still_stripped(self):
        req = self._prefill_and_decode([])
        with get_serving().override(strip_thinking_cache=True):
            self.assertEqual(req.owned_kv_len(), PROMPT)
        self._release_and_check(req, is_insert=True)


class TestStripThinkingAfterRetractionPaged(CustomTestCase):
    """page_size=4. With an unaligned prompt the tree-owned page is freed by
    insert_req's unaligned-tail free instead of the over-allocated range; the
    same bound on owned_kv_len() covers both."""

    PAGE = 4

    def setUp(self):
        set_global_server_args_for_scheduler(
            ServerArgs(model_path="dummy", page_size=self.PAGE)
        )
        self.pool = ReqToTokenPool(
            size=4, max_context_len=64, device="cpu", enable_memory_saver=False
        )
        kvcache = MHATokenToKVPool(
            size=POOL,
            page_size=self.PAGE,
            dtype=torch.float16,
            head_num=1,
            head_dim=8,
            layer_num=1,
            device="cpu",
            enable_memory_saver=False,
        )
        self.allocator = PagedTokenToKVPoolAllocator(
            size=POOL,
            page_size=self.PAGE,
            dtype=torch.float16,
            device="cpu",
            kvcache=kvcache,
            need_sort=False,
        )
        self.cache = UnifiedRadixCache(
            params=CacheInitParams(
                req_to_token_pool=self.pool,
                token_to_kv_pool_allocator=self.allocator,
                page_size=self.PAGE,
                disable=False,
                tree_components=(ComponentType.FULL,),
            )
        )

    def _run(self, prompt, is_insert):
        req = Req(
            "r",
            "",
            array("q", range(1, prompt + 1)),
            SamplingParams(temperature=0, max_new_tokens=32),
        )
        req.output_ids = array("q", range(100, 100 + BEFORE_RETRACT))
        req.reasoning_tokens = BEFORE_RETRACT + 1
        req.last_node = self.cache.root_node_handle()
        req.lock_receipt = DecLockRefParams()
        req.extra_key = None

        self.pool.alloc([req])
        fill = prompt + BEFORE_RETRACT
        pages = -(-(fill + AFTER) // self.PAGE)
        slots = self.allocator.alloc(pages * self.PAGE)
        req.full_untruncated_fill_ids = req.origin_input_ids + req.output_ids
        req.prefix_indices = torch.empty((0,), dtype=torch.int64)
        req.set_extend_range(0, fill)
        self.pool.write(
            (req.kv.req_pool_idx, slice(0, fill)), slots[:fill].to(torch.int32)
        )
        req.kv.kv_committed_len = req.kv.kv_allocated_len = fill
        self.cache.cache_unfinished_req(req)
        self.assertGreater(
            req.kv.cache_protected_len, prompt
        )  # the re-prefill put output in the tree
        for k in range(AFTER):
            self.pool.write(
                (req.kv.req_pool_idx, req.kv.kv_committed_len),
                slots[fill + k : fill + k + 1].to(torch.int32),
            )
            req.kv.kv_committed_len += 1
            req.kv.kv_allocated_len += 1
            req.output_ids.append(99)

        with get_serving().override(strip_thinking_cache=True):
            release_kv_cache(req, self.cache, is_insert=is_insert)
        self.cache.evict(EvictParams(num_tokens=POOL))
        self.allocator.merge_and_sort_free()
        free = self.allocator.free_pages.tolist()
        self.assertEqual(len(free), len(set(free)), "a KV page is free twice")
        self.assertEqual(self.allocator.available_size(), POOL)

    def test_reprefill_release_frees_each_page_once(self):
        for prompt in (8, 10):  # aligned, unaligned
            for is_insert in (True, False):
                with self.subTest(prompt=prompt, is_insert=is_insert):
                    self.setUp()
                    self._run(prompt, is_insert)


if __name__ == "__main__":
    unittest.main()
