"""CPU request-lifecycle coverage; this does not validate NPU draft KV contents."""

import unittest
from array import array
from types import SimpleNamespace

import torch

from sglang.srt.arg_groups.pd_disaggregation_hook import (
    _validate_decode_input_only_cache,
    handle_pd_disaggregation,
)
from sglang.srt.environ import envs
from sglang.srt.managers.req_types import FINISH_LENGTH
from sglang.srt.managers.schedule_batch import Req
from sglang.srt.mem_cache.allocator.paged import PagedTokenToKVPoolAllocator
from sglang.srt.mem_cache.base_prefix_cache import EvictParams, MatchPrefixParams
from sglang.srt.mem_cache.cache_init_params import CacheInitParams
from sglang.srt.mem_cache.common import release_kv_cache
from sglang.srt.mem_cache.memory_pool import MHATokenToKVPool, ReqToTokenPool
from sglang.srt.mem_cache.radix_cache import RadixKey
from sglang.srt.mem_cache.unified_cache.components import ComponentType
from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache
from sglang.srt.runtime_context import get_context
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.srt.server_args import ServerArgs
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase
from sglang.test.tree_slots import tree_slots_of

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestDecodeInputOnlyConfig(CustomTestCase):
    def config(self, **changes):
        values = dict(
            disaggregation_mode="decode",
            disaggregation_decode_enable_radix_cache=True,
            speculative_algorithm="EAGLE",
            disaggregation_transfer_backend="mooncake",
            enable_hierarchical_cache=False,
            disaggregation_decode_enable_offload_kvcache=False,
            enable_session_radix_cache=False,
            enable_hisparse=False,
            enable_pd_role_switch=False,
            pp_size=1,
            attn_cp_size=1,
            dcp_size=1,
        )
        values.update(changes)
        return SimpleNamespace(**values)

    def test_default_off_and_other_roles_keep_existing_behavior(self):
        with envs.SGLANG_PD_DECODE_CACHE_INPUT_ONLY.override(False):
            self.assertFalse(_validate_decode_input_only_cache(self.config()))
        with envs.SGLANG_PD_DECODE_CACHE_INPUT_ONLY.override(True):
            for role in ("prefill", "null"):
                self.assertFalse(
                    _validate_decode_input_only_cache(
                        self.config(disaggregation_mode=role)
                    )
                )
            for algorithm in ("EAGLE", "NEXTN"):
                self.assertTrue(
                    _validate_decode_input_only_cache(
                        self.config(speculative_algorithm=algorithm)
                    )
                )

    def test_actual_pd_hook_accepts_the_opt_in(self):
        args = ServerArgs(model_path="dummy", **vars(self.config()))
        with envs.SGLANG_PD_DECODE_CACHE_INPUT_ONLY.override(True):
            handle_pd_disaggregation(args)

    def test_unsupported_combinations_fail_before_serving(self):
        changes = (
            {"disaggregation_decode_enable_radix_cache": False},
            {"speculative_algorithm": None},
            {"speculative_algorithm": "EAGLE3"},
            {"disaggregation_transfer_backend": "fake"},
            {"enable_hierarchical_cache": True},
            {"disaggregation_decode_enable_offload_kvcache": True},
            {"enable_session_radix_cache": True},
            {"enable_hisparse": True},
            {"enable_pd_role_switch": True},
            {"pp_size": 2},
            {"attn_cp_size": 2},
            {"dcp_size": 2},
        )
        with envs.SGLANG_PD_DECODE_CACHE_INPUT_ONLY.override(True):
            for change in changes:
                with (
                    self.subTest(change=change),
                    self.assertRaisesRegex(
                        ValueError, "SGLANG_PD_DECODE_CACHE_INPUT_ONLY"
                    ),
                ):
                    _validate_decode_input_only_cache(self.config(**change))


class TestDecodeInputOnlyLifecycle(CustomTestCase):
    capacity = 2048
    page_size = 128

    def make_cache(self, enabled=True, role="decode", is_eagle=True):
        self.enterContext(
            get_context().override_server_args(
                model_path="dummy",
                device="cpu",
                page_size=self.page_size,
                disaggregation_mode=role,
                speculative_algorithm="EAGLE",
            )
        )
        req_pool = ReqToTokenPool(4, 1024, "cpu", enable_memory_saver=False)
        kv_pool = MHATokenToKVPool(
            size=self.capacity,
            page_size=self.page_size,
            dtype=torch.float32,
            head_num=1,
            head_dim=4,
            layer_num=1,
            device="cpu",
            enable_memory_saver=False,
        )
        allocator = PagedTokenToKVPoolAllocator(
            size=self.capacity,
            page_size=self.page_size,
            dtype=torch.float32,
            device="cpu",
            kvcache=kv_pool,
            need_sort=False,
        )
        with (
            envs.SGLANG_PD_DECODE_CACHE_INPUT_ONLY.override(enabled),
            envs.SGLANG_UNIFIED_RADIX_TREE_CORE_BACKEND.override("python"),
        ):
            cache = UnifiedRadixCache(
                CacheInitParams(
                    disable=False,
                    req_to_token_pool=req_pool,
                    token_to_kv_pool_allocator=allocator,
                    page_size=self.page_size,
                    is_eagle=is_eagle,
                    tree_slots=tree_slots_of((ComponentType.FULL,)),
                )
            )
        return cache, allocator, req_pool

    def request(self, cache, allocator, pool, prompt_len, output_len=256):
        tokens = array("q", range(1, prompt_len + output_len + 1))
        req = Req(
            rid=f"input-only-{prompt_len}",
            origin_input_text="",
            origin_input_ids=tokens[:prompt_len],
            sampling_params=SamplingParams(temperature=0, max_new_tokens=output_len),
        )
        self.assertIsNotNone(pool.alloc([req]))
        req.output_ids = tokens[prompt_len:]
        req.full_untruncated_fill_ids = tokens
        req.extend_end = len(tokens)
        aligned = (len(tokens) + self.page_size - 1) // self.page_size * self.page_size
        indices = allocator.alloc(aligned)
        self.assertIsNotNone(indices)
        pool.write((req.kv.req_pool_idx, slice(0, len(tokens))), indices[: len(tokens)])
        req.kv.kv_committed_len = req.kv.kv_allocated_len = len(tokens)
        req.last_node = cache.root_node_handle()
        req.lock = None
        return req, tokens

    def finish(self, cache, req, insert=True):
        req.finished_reason = FINISH_LENGTH(length=len(req.output_ids))
        release_kv_cache(req, cache, checkpoint=insert)

    def checkpoint(self, cache, req):
        cache.checkpoint(req, up_to=req.extend_end)

    def matched(self, cache, tokens):
        result = cache.match_prefix(
            MatchPrefixParams(key=RadixKey(array("q", tokens), is_bigram=True))
        )
        return result.device_prefix_len

    def assert_reclaimed(self, cache, allocator, retained):
        cache.sanity_check()
        self.assertEqual(allocator.available_size(), self.capacity - retained)
        self.assertEqual(cache.protected_size(), 0)
        self.assertEqual(
            cache.evict(EvictParams(num_tokens=self.capacity)).num_tokens_evicted,
            retained,
        )
        self.assertEqual(allocator.available_size(), self.capacity)
        slots = allocator.alloc(self.capacity)
        self.assertEqual(slots.unique().numel(), self.capacity)

    def test_finish_keeps_input_only_across_bigram_page_boundaries(self):
        for prompt in (1, 127, 128, 129, 255, 256, 257):
            with self.subTest(prompt=prompt):
                cache, allocator, pool = self.make_cache()
                req, tokens = self.request(cache, allocator, pool, prompt)
                self.finish(cache, req)
                retained = (prompt - 1) // self.page_size * self.page_size
                self.assertEqual(self.matched(cache, tokens), retained)
                self.assert_reclaimed(cache, allocator, retained)

    def test_inflight_tail_stays_owned_and_prefix_stays_locked(self):
        cache, allocator, pool = self.make_cache()
        req, tokens = self.request(cache, allocator, pool, 257)
        free = allocator.available_size()
        self.checkpoint(cache, req)
        self.assertEqual(self.matched(cache, tokens), 256)
        self.assertEqual(allocator.available_size(), free)
        self.assertEqual(cache.evict(EvictParams(num_tokens=256)).num_tokens_evicted, 0)
        self.finish(cache, req)
        self.assert_reclaimed(cache, allocator, 256)

    def test_duplicate_insert_and_different_suffix_do_not_reuse_output(self):
        cache, allocator, pool = self.make_cache()
        for _ in range(2):
            req, tokens = self.request(cache, allocator, pool, 257)
            self.finish(cache, req)
            self.assertEqual(self.matched(cache, tokens), 256)
        changed = tokens[:257]
        changed[-1] = 99999
        self.assertEqual(self.matched(cache, changed), 128)
        self.assert_reclaimed(cache, allocator, 256)

    def test_no_insert_release_returns_every_slot(self):
        cache, allocator, pool = self.make_cache()
        req, tokens = self.request(cache, allocator, pool, 257)
        self.finish(cache, req, insert=False)
        self.assertEqual(self.matched(cache, tokens), 0)
        self.assert_reclaimed(cache, allocator, 0)

    def test_disabled_and_prefill_keep_output_caching(self):
        for enabled, role in ((False, "decode"), (True, "prefill")):
            with self.subTest(enabled=enabled, role=role):
                cache, allocator, pool = self.make_cache(enabled, role)
                req, tokens = self.request(cache, allocator, pool, 257)
                self.finish(cache, req)
                self.assertEqual(self.matched(cache, tokens), 512)
                self.assert_reclaimed(cache, allocator, 512)

    def test_non_eagle_cache_and_later_host_attachments_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "EAGLE FULL-only"):
            self.make_cache(is_eagle=False)
        cache, _, _ = self.make_cache()
        with self.assertRaisesRegex(ValueError, "does not support HiCache"):
            cache.init_hicache(None, None)
        with self.assertRaisesRegex(ValueError, "does not support external KV stores"):
            cache.init_cache_linker(None)


if __name__ == "__main__":
    unittest.main()
