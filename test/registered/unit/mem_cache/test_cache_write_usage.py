"""Cache creation usage must come from adopted, reusable prompt ranges."""

import importlib.util
import sys
import unittest
from array import array
from dataclasses import replace
from types import SimpleNamespace
from unittest import mock

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

from sglang.srt.mem_cache.cache_write_usage import (
    CacheWriteUsage,
    cache_write_usage_for_req,
)


class TestCacheWriteUsage(CustomTestCase):
    def test_unknown_is_not_zero(self):
        usage = CacheWriteUsage(100)
        self.assertIsNone(usage.tokens)
        usage.mark_supported()
        self.assertEqual(usage.tokens, 0)
        usage.mark_unknown()
        usage.mark_supported()
        usage.record([(0, 100)], reusable_prefix_len=100)
        self.assertIsNone(usage.tokens)

    def test_only_new_prompt_ranges_are_counted(self):
        usage = CacheWriteUsage(100)
        usage.mark_supported()
        # [0, 40) was a hit, and [100, 120) is generated output.
        usage.record([(40, 120)], reusable_prefix_len=120)
        self.assertEqual(usage.tokens, 60)
        # Neither an H2D hit nor a concurrent duplicate insert adopts KV.
        usage.record([], reusable_prefix_len=100)
        self.assertEqual(usage.tokens, 60)

    def test_chunks_checkpoints_and_retraction_are_a_union(self):
        usage = CacheWriteUsage(100)
        usage.mark_supported()
        for ranges in ([(0, 32)], [(32, 64)], [(0, 64)], [(50, 100)]):
            usage.record(ranges, reusable_prefix_len=100)
        self.assertEqual(usage.tokens, 100)

    def test_full_kv_waits_for_a_reusable_mamba_boundary(self):
        usage = CacheWriteUsage(100)
        usage.mark_supported()
        usage.record([(20, 100)], reusable_prefix_len=0)
        self.assertEqual(usage.tokens, 0)
        usage.record([], reusable_prefix_len=64)
        self.assertEqual(usage.tokens, 44)
        usage.record([], reusable_prefix_len=100)
        self.assertEqual(usage.tokens, 80)
        # An internal retry at an earlier boundary does not erase history.
        usage.record([(20, 64)], reusable_prefix_len=32)
        self.assertEqual(usage.tokens, 80)

    def test_disjoint_ranges_and_page_boundary(self):
        usage = CacheWriteUsage(100)
        usage.mark_supported()
        usage.record([(64, 100), (0, 32), (16, 48)], reusable_prefix_len=96)
        self.assertEqual(usage.tokens, 80)

    def test_original_cache_read_is_not_written_again_on_retraction(self):
        usage = CacheWriteUsage(100)
        usage.mark_supported()
        usage.record([(40, 64)], reusable_prefix_len=64, cached_prefix_len=40)
        usage.record([(0, 100)], reusable_prefix_len=100, cached_prefix_len=40)
        self.assertEqual(usage.tokens, 60)

    def test_prompt_length_follows_padding_or_truncation_before_publication(self):
        for initial, final in ((4, 8), (8, 4)):
            with self.subTest(initial=initial, final=final):
                usage = CacheWriteUsage(initial)
                usage.mark_supported()
                req = SimpleNamespace(
                    _cache_write_usage=usage, origin_input_ids=list(range(final))
                )
                cache_write_usage_for_req(req).record(
                    [(0, final)], reusable_prefix_len=final
                )
                self.assertEqual(usage.tokens, final)
                req.origin_input_ids.append(100)
                self.assertIsNone(cache_write_usage_for_req(req).tokens)
                req.origin_input_ids.pop()
                usage.mark_supported()
                self.assertIsNone(cache_write_usage_for_req(req).tokens)

    def test_reporting_window_intersects_each_reusable_range(self):
        for adopted, cached, expected in (
            ([(0, 512)], 0, 509),
            ([(256, 512)], 256, 253),
            ([(509, 512)], 0, 0),
        ):
            with self.subTest(adopted=adopted, cached=cached):
                usage = CacheWriteUsage(512, excluded_suffix_tokens=3)
                usage.mark_supported()
                usage.record(adopted, reusable_prefix_len=512, cached_prefix_len=cached)
                self.assertEqual(usage.tokens, expected)
                self.assertEqual(usage.prompt_tokens, 512)
                # Duplicate publication and retraction do not count it twice.
                usage.record(adopted, reusable_prefix_len=512, cached_prefix_len=cached)
                self.assertEqual(usage.tokens, expected)

    def test_reporting_window_follows_final_input_coordinates(self):
        for initial, final, expected in ((4, 8, 5), (8, 4, 1), (4, 2, 0)):
            with self.subTest(initial=initial, final=final):
                usage = CacheWriteUsage(initial, excluded_suffix_tokens=3)
                usage.mark_supported()
                usage.sync_prompt_tokens(final)
                usage.record([(0, final + 2)], reusable_prefix_len=final + 2)
                self.assertEqual(usage.tokens, expected)
                usage.sync_prompt_tokens(final + 1)
                self.assertIsNone(usage.tokens)
                self.assertEqual(usage.reported_prompt_tokens, max(0, final - 2))

    def test_late_reporting_window_change_keeps_unknown_sticky(self):
        usage = CacheWriteUsage(8)
        usage.mark_supported()
        usage.set_excluded_suffix_tokens(3)
        usage.record([(0, 8)], reusable_prefix_len=8)
        self.assertEqual(usage.tokens, 5)
        usage.set_excluded_suffix_tokens(2)
        usage.mark_supported()
        self.assertIsNone(usage.tokens)

    def test_reported_reads_do_not_change_physical_cache_prefix(self):
        from sglang.srt.managers.schedule_batch import Req

        for suffix, cached, expected in (
            (0, 511, 511),
            (3, 511, 509),
            (0, 512, 512),
            (3, 512, 509),
        ):
            with self.subTest(suffix=suffix, cached=cached):
                req = Req.__new__(Req)
                req.origin_input_ids = array("q", range(512))
                req.cached_tokens = cached
                req._cache_write_usage = CacheWriteUsage(
                    512, excluded_suffix_tokens=suffix
                )
                req._cache_write_usage.mark_supported()
                req._cache_write_usage.record(
                    [(cached, 512)], reusable_prefix_len=512, cached_prefix_len=cached
                )
                self.assertEqual(req.reported_cached_tokens, expected)
                self.assertEqual(req.cached_tokens, cached)
                self.assertEqual(
                    req.cache_write_tokens, 512 - cached if suffix == 0 else 0
                )
                req.origin_input_ids = array("q", range(256))
                self.assertEqual(req.reported_cached_tokens, 256 - suffix)
                self.assertIsNone(req.cache_write_tokens)
                self.assertEqual(req.cached_tokens, cached)


class TestUnifiedCacheWriteUsage(CustomTestCase):
    """Exercise the actual insert and canonical match on CPU, without a model."""

    backend = "python"

    def _cache(self, *, mamba=False, extra_buffer=False):
        from sglang.srt.mem_cache.unified_cache.component_type import ComponentType
        from sglang.test.kits.unified_radix_cache import fixtures
        from sglang.test.kits.unified_radix_cache.config import CacheConfig

        cfg = CacheConfig(
            components=(
                (ComponentType.FULL, ComponentType.MAMBA)
                if mamba
                else (ComponentType.FULL,)
            ),
            enable_mamba_extra_buffer=extra_buffer,
            num_layers=2,
            full_attention_layer_ids=(0,),
            kv_size=64,
            max_context_len=64,
            mamba_intermediate_size=16,
            mamba_num_heads=1,
            mamba_head_dim=4,
            mamba_state_size=4,
        )
        with mock.patch.object(fixtures, "get_device", return_value="cpu"):
            # Reuse CPU pool setup, but do not request the fixture's Rust
            # inspection factory: this test only needs production interfaces.
            result = fixtures.build_fixture(cfg, tree_core_backend="python")
        if self.backend == "rust":
            from sglang.srt.environ import envs
            from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache

            native_module = "sglang.srt.mem_cache.rust_tree_core.mem_cache"
            if (
                native_module not in sys.modules
                and importlib.util.find_spec(native_module) is None
            ):
                self.skipTest(
                    "production Rust binding is not bundled in this runtime; "
                    "cache-write tests do not build a Rust inspection extension"
                )
            params = replace(result[0].cache_init_params, tree_core_backend="rust")
            # Use the real registered Rust adapter and the installed binding.
            # Do not swallow constructor, insert, match or assertion failures.
            with envs.SGLANG_RUST_BUILD_MODE.override("never"):
                runtime_cache = UnifiedRadixCache(params=params)
            result = runtime_cache, result[1], result[2]
        if result[0]._tree_core_backend != self.backend:
            self.skipTest(f"{self.backend} TreeCore is unavailable on this runtime")
        return result

    def _req(self, cache, allocator, pool, prompt, output=(), *, mamba=False):
        from sglang.srt.managers.schedule_batch import Req
        from sglang.srt.sampling.sampling_params import SamplingParams

        req = Req(
            rid=str(id(prompt)),
            origin_input_text="",
            origin_input_ids=array("q", prompt),
            sampling_params=SamplingParams(temperature=0, max_new_tokens=2),
        )
        req._cache_write_tracking_enabled = True
        pool.alloc([req])
        req.output_ids = array("q", output)
        req.full_untruncated_fill_ids = array("q", (*prompt, *output))
        size = len(req.full_untruncated_fill_ids)
        pool.write((req.kv.req_pool_idx, slice(0, size)), allocator.alloc(size))
        req.kv.kv_committed_len = size
        req.kv.kv_allocated_len = size
        req.last_node = cache.root_node_handle()
        if mamba:
            req.kv.mamba_last_track_seqlen = size
        return req

    def test_cold_warm_and_partial_hits(self):
        from sglang.test.mem_cache_utils import finish_req

        for mamba in (False, True):
            with self.subTest(mamba=mamba):
                cache, allocator, pool = self._cache(mamba=mamba)
                prompt = list(range(8))
                cold = self._req(cache, allocator, pool, prompt, [80, 81], mamba=mamba)
                if mamba:
                    # Publish a prompt-only recurrent state before decode.
                    cache.checkpoint(cold, up_to=8)
                finish_req(cache, cold, cold.owned_kv_len())
                self.assertEqual(cold.cache_write_tokens, 8)
                warm = self._req(cache, allocator, pool, prompt, [90, 91], mamba=mamba)
                finish_req(cache, warm, warm.owned_kv_len())
                self.assertEqual(warm.cache_write_tokens, 0)
                partial = self._req(
                    cache, allocator, pool, prompt + [20, 21, 22, 23], mamba=mamba
                )
                finish_req(cache, partial, partial.owned_kv_len())
                self.assertEqual(partial.cache_write_tokens, 4)

    def test_mamba_state_after_output_does_not_make_prompt_reusable(self):
        from sglang.test.mem_cache_utils import finish_req

        cache, allocator, pool = self._cache(mamba=True)
        req = self._req(cache, allocator, pool, list(range(8)), [80, 81], mamba=True)
        finish_req(cache, req, req.owned_kv_len())
        self.assertEqual(req.cache_write_tokens, 0)

    def test_chunked_checkpoints_and_duplicate_insert(self):
        from sglang.test.mem_cache_utils import finish_req

        cache, allocator, pool = self._cache()
        prompt = list(range(8))
        req = self._req(cache, allocator, pool, prompt, [80, 81])
        duplicate = self._req(cache, allocator, pool, prompt, [80, 81])
        cache.checkpoint(req, up_to=4)
        self.assertEqual(req.cache_write_tokens, 4)
        cache.checkpoint(req, up_to=8)
        self.assertEqual(req.cache_write_tokens, 8)
        finish_req(cache, req, req.owned_kv_len())
        self.assertEqual(req.cache_write_tokens, 8)
        finish_req(cache, duplicate, duplicate.owned_kv_len())
        self.assertEqual(duplicate.cache_write_tokens, 0)

    def test_mamba_without_ready_checkpoint_is_zero(self):
        from sglang.test.mem_cache_utils import finish_req

        cache, allocator, pool = self._cache(mamba=True, extra_buffer=True)
        req = self._req(cache, allocator, pool, list(range(8)), mamba=True)
        req.kv.mamba_last_track_seqlen = None
        finish_req(cache, req, req.owned_kv_len())
        self.assertEqual(req.cache_write_tokens, 0)

    def test_disabled_zero_and_unsupported_unknown(self):
        from sglang.srt.runtime_context import get_disagg

        cache, allocator, pool = self._cache()
        req = self._req(cache, allocator, pool, list(range(8)))
        cache.disable = True
        cache.checkpoint(req, up_to=8)
        self.assertEqual(req.cache_write_tokens, 0)
        req.session = SimpleNamespace(streaming=False)
        cache._prepare_cache_write_usage(req)
        self.assertIsNone(req.cache_write_tokens)
        req.session = None
        cache.disable = False
        cache._prepare_cache_write_usage(req)
        self.assertIsNone(req.cache_write_tokens)

        req2 = self._req(cache, allocator, pool, list(range(8)))
        with get_disagg().override(disaggregation_decode_enable_offload_kvcache=True):
            cache._prepare_cache_write_usage(req2)
        self.assertIsNone(req2.cache_write_tokens)

    def test_tracking_flag_controls_adoption_without_a_second_match(self):
        """Accounting must not alter tree matching or run when disabled."""
        for enabled in (False, True):
            with self.subTest(enabled=enabled):
                cache, allocator, pool = self._cache()
                req = self._req(cache, allocator, pool, list(range(8)))
                req._cache_write_tracking_enabled = enabled
                with mock.patch.object(
                    cache.tree_core, "match_prefix", wraps=cache.tree_core.match_prefix
                ) as match:
                    with mock.patch.object(
                        cache, "insert", wraps=cache.insert
                    ) as insert:
                        cache.checkpoint(req, up_to=8)
                self.assertEqual(insert.call_args.args[0].track_adopted_ranges, enabled)
                match.assert_not_called()
                self.assertEqual(req.cache_write_tokens, 8 if enabled else None)

    def test_nonbase_token_slot_reports_unknown(self):
        """TreeCore emits ranges at BASE, so a different token id is unknown."""
        from sglang.srt.mem_cache.kv.plan import TREE_TOKEN
        from sglang.srt.mem_cache.unified_cache.component_type import ComponentType

        cache, allocator, pool = self._cache()
        req = self._req(cache, allocator, pool, list(range(8)))
        original = cache.slot_id
        cache.slot_id = lambda family: (
            ComponentType.SWA if family == TREE_TOKEN else original(family)
        )
        cache._prepare_cache_write_usage(req)
        self.assertIsNone(req.cache_write_tokens)


class TestRustUnifiedCacheWriteUsage(TestUnifiedCacheWriteUsage):
    backend = "rust"


if __name__ == "__main__":
    unittest.main()
