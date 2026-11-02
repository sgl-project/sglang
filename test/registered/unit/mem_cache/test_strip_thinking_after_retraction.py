"""--strip-thinking-cache caps Req.owned_kv_len() at the prompt. A retracted
request is re-prefilled over prompt + output, and the tree then holds part of
that output (the re-prefill's checkpoint inserts it, or its match finds it), so
cache_protected_len > prompt. Releasing it must never free those tree-owned
slots: no slot may be free twice, and every slot must come back."""

import contextlib
import unittest
from array import array

import torch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.managers.schedule_batch import FINISH_LENGTH, Req  # noqa: E402
from sglang.srt.mem_cache.allocator import TokenToKVPoolAllocator  # noqa: E402
from sglang.srt.mem_cache.allocator.paged import (  # noqa: E402
    PagedTokenToKVPoolAllocator,
)
from sglang.srt.mem_cache.base_prefix_cache import EvictParams  # noqa: E402
from sglang.srt.mem_cache.cache_init_params import CacheInitParams  # noqa: E402
from sglang.srt.mem_cache.common import (  # noqa: E402
    checkpoint_kv_cache,
    release_kv_cache,
)
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
EARLIER_OUTPUT = list(range(100, 106))  # generated before the retraction
AFTER = 3  # decode steps after the re-prefill


class _RetractionCase(CustomTestCase):
    PAGE = 1

    def setUp(self):
        set_global_server_args_for_scheduler(
            ServerArgs(model_path="dummy", page_size=self.PAGE)
        )
        # The server runs with the flag on, so the re-prefill sees it too.
        stack = contextlib.ExitStack()
        self.addCleanup(stack.close)
        stack.enter_context(get_serving().override(strip_thinking_cache=True))
        self._build()

    def _build(self):
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
        if self.PAGE == 1:
            self.allocator = TokenToKVPoolAllocator(
                size=POOL,
                dtype=torch.float16,
                device="cpu",
                kvcache=kvcache,
                need_sort=False,
            )
        else:
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
        self.page_tail = []  # unused slots of the row's last page

    def _slots(self, n):
        # Whole pages per row, so no two rows share a page.
        while len(self.page_tail) < n:
            self.page_tail += self.allocator.alloc(self.PAGE).tolist()
        slots, self.page_tail = self.page_tail[:n], self.page_tail[n:]
        return torch.tensor(slots, dtype=torch.int32)

    def _admit(self, prompt, output, reasoning=True):
        """Match prompt + output and lock the hit, as admission does."""
        req = Req(
            f"r-{len(output)}-{reasoning}",
            "",
            array("q", range(1, prompt + 1)),
            SamplingParams(temperature=0, max_new_tokens=32),
        )
        req.output_ids = array("q", output)
        req.reasoning_tokens = len(output) + 1 if reasoning else 0
        req.extra_key = None
        req.init_next_round_input(self.cache)
        req.lock = self.cache.lock(req.last_node)
        prefix = self.cache.prefix_device_indices(req)
        self.pool.alloc([req])
        self.pool.write(
            (req.kv.req_pool_idx, slice(0, req.prefix_len)), prefix.to(torch.int32)
        )
        req.kv.kv_committed_len = req.kv.kv_allocated_len = req.prefix_len
        return req

    def _extend(self, req, end):
        req.extend_end = end
        self.pool.write(
            (req.kv.req_pool_idx, slice(req.prefix_len, end)),
            self._slots(end - req.prefix_len),
        )
        req.kv.kv_committed_len = req.kv.kv_allocated_len = end

    def _prefill_and_decode(self, req, chunk_end=None):
        """Prefill prompt + output (a re-prefill when output is non-empty),
        checkpoint it as the scheduler does, then decode AFTER tokens."""
        if chunk_end is not None:
            self._extend(req, chunk_end)
            checkpoint_kv_cache(req, self.cache)  # stash_chunked_request
        self._extend(req, len(req.full_untruncated_fill_ids))
        checkpoint_kv_cache(req, self.cache)  # batch_result_processor, unfinished
        for _ in range(AFTER):
            self.pool.write(
                (req.kv.req_pool_idx, req.kv.kv_committed_len), self._slots(1)
            )
            req.kv.kv_committed_len += 1
            req.kv.kv_allocated_len += 1
            req.output_ids.append(99)
        return req

    def _release(self, req, finish):
        if finish:
            req.finished_reason = FINISH_LENGTH(length=len(req.output_ids))
        release_kv_cache(req, self.cache, checkpoint=finish)
        self.page_tail = []  # freed with the row's last page

    def _release_and_check(self, req, finish):
        tree_slots = set(
            self.pool.req_to_token[
                req.kv.req_pool_idx, : req.kv.cache_protected_len
            ].tolist()
        )
        self._release(req, finish)
        if self.PAGE == 1:
            free = self.allocator.free_pages.tolist()
            self.assertEqual(sorted(tree_slots & set(free)), [])
        self.cache.evict(EvictParams(num_tokens=POOL))
        if self.PAGE > 1:
            self.allocator.merge_and_sort_free()
        free = self.allocator.free_pages.tolist()
        self.assertEqual(len(free), len(set(free)), "a KV slot is free twice")
        self.assertEqual(self.allocator.available_size(), POOL)


class TestStripThinkingAfterRetraction(_RetractionCase):
    def test_finish_after_reprefill_frees_each_slot_once(self):
        req = self._prefill_and_decode(self._admit(PROMPT, EARLIER_OUTPUT))
        self.assertGreater(req.kv.cache_protected_len, PROMPT)
        self._release_and_check(req, finish=True)

    def test_second_retraction_frees_each_slot_once(self):
        req = self._prefill_and_decode(self._admit(PROMPT, EARLIER_OUTPUT))
        self._release_and_check(req, finish=False)

    def test_without_retraction_output_is_still_stripped(self):
        req = self._prefill_and_decode(self._admit(PROMPT, []))
        self.assertEqual(req.owned_kv_len(), PROMPT)
        self._release_and_check(req, finish=True)

    def test_chunked_reprefill_frees_each_slot_once(self):
        for finish in (True, False):
            with self.subTest(finish=finish):
                self._build()
                req = self._admit(PROMPT, EARLIER_OUTPUT)
                # The first chunk ends inside the earlier output.
                self._prefill_and_decode(req, chunk_end=PROMPT + 2)
                self._release_and_check(req, finish=finish)

    def test_reprefill_matching_cached_output_frees_each_slot_once(self):
        for finish in (True, False):
            with self.subTest(finish=finish):
                self._build()
                # A request without reasoning cached prompt + the first output
                # tokens, so the re-prefill's match alone passes the prompt.
                other = self._admit(PROMPT, EARLIER_OUTPUT[:4], reasoning=False)
                self._extend(other, PROMPT + 4)
                self._release(other, finish=True)
                req = self._admit(PROMPT, EARLIER_OUTPUT)
                self.assertEqual(req.prefix_len, PROMPT + 4)
                self._prefill_and_decode(req)
                self._release_and_check(req, finish=finish)


class TestStripThinkingAfterRetractionPaged(_RetractionCase):
    PAGE = 4

    def test_reprefill_release_frees_each_page_once(self):
        for prompt in (8, 10):  # page-aligned, unaligned
            for finish in (True, False):
                with self.subTest(prompt=prompt, finish=finish):
                    self._build()
                    req = self._admit(prompt, EARLIER_OUTPUT)
                    self._prefill_and_decode(req)
                    self.assertGreater(req.kv.cache_protected_len, prompt)
                    self._release_and_check(req, finish=finish)


if __name__ == "__main__":
    unittest.main()
