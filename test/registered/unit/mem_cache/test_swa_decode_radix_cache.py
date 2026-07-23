"""Unit tests for decode radix cache support on hybrid-SWA models (PR #30929).

The PR enables ``--disaggregation-decode-enable-radix-cache`` on hybrid-SWA
models whose allocator can preallocate the sliding-window tail (SWA-tail
prealloc, e.g. DeepSeek-V4). Two source changes are covered here:

1. ``mem_cache.kv_cache_builder.build_kv_cache`` — the startup guard now only
   rejects SWA models whose allocator lacks the SWA-tail-prealloc capability
   (``alloc_extend_swa_tail`` with ``page_size > 1``). Mamba/SSM stays rejected.

2. ``mem_cache.unified_radix_cache.UnifiedRadixCache.cache_unfinished_req`` —
   ``match_prefix()`` gates device indices on ALL components, so on a hybrid-SWA
   tree the SWA validator collapses ``device_indices`` to length 0 at the first
   out-of-window tombstone even though the full-attention KV for the whole
   prefix is still device-resident. The fix walks the just-inserted path
   read-only to recover the ungated FULL-component indices, avoiding a fatal
   ``new_prefix_len <= len(new_indices)`` assert.

Usage:
    python -m pytest \
        test/registered/unit/mem_cache/test_swa_decode_radix_cache.py -v
"""

from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci

register_cuda_ci(est_time=20, stage="base-b", runner_config="1-gpu-small")
register_amd_ci(est_time=20, suite="stage-b-test-1-gpu-small-amd")

import unittest
from array import array
from unittest import mock

import torch

import sglang.srt.mem_cache.kv_cache_builder as kvb
from sglang.srt.configs.model_config import ModelImpl
from sglang.srt.managers.schedule_batch import Req
from sglang.srt.mem_cache.allocator import TokenToKVPoolAllocator
from sglang.srt.mem_cache.allocator.swa import SWATokenToKVPoolAllocator
from sglang.srt.mem_cache.base_prefix_cache import (
    DecLockRefParams,
    MatchPrefixParams,
)
from sglang.srt.mem_cache.cache_init_params import CacheInitParams
from sglang.srt.mem_cache.memory_pool import ReqToTokenPool
from sglang.srt.mem_cache.radix_cache import RadixKey
from sglang.srt.mem_cache.swa_memory_pool import SWAKVPool
from sglang.srt.mem_cache.unified_cache_components.tree_component import ComponentType
from sglang.srt.mem_cache.unified_radix_cache import (
    BASE_COMPONENT_TYPE,
    UnifiedRadixCache,
)
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.srt.server_args import ServerArgs, set_global_server_args_for_scheduler
from sglang.srt.utils import get_device
from sglang.test.test_utils import CustomTestCase

# ---------------------------------------------------------------------------
# Part A: build_kv_cache startup guard
# ---------------------------------------------------------------------------


class _SWATailCapableAllocator:
    """Fake allocator that exposes the SWA-tail-prealloc capability."""

    def __init__(self, page_size: int = 256):
        self.page_size = page_size

    def alloc_extend_swa_tail(self, *args, **kwargs):  # pragma: no cover - marker only
        raise NotImplementedError


class _NoSWATailAllocator:
    """Fake allocator without ``alloc_extend_swa_tail`` (e.g. plain paged pool)."""

    def __init__(self, page_size: int = 256):
        self.page_size = page_size


_TREE_CACHE_SENTINEL = object()


class TestBuildKVCacheDecodeRadixSWAGuard(CustomTestCase):
    """The startup guard should gate on SWA-tail-prealloc capability, not on
    ``is_hybrid_swa`` alone."""

    def _call_build_kv_cache(
        self,
        *,
        is_hybrid_swa: bool,
        is_hybrid_ssm: bool,
        allocator,
        decode_radix_enabled: bool = True,
        disaggregation_mode: str = "decode",
        page_size: int = 256,
    ):
        tp_worker = mock.MagicMock()
        tp_worker.is_hybrid_swa = is_hybrid_swa
        model_runner = tp_worker.model_runner
        model_runner.linear_attn_model_spec = None
        model_runner.hybrid_gdn_config = None
        model_runner.mamba2_config = object() if is_hybrid_ssm else None
        model_runner.kimi_linear_config = None
        model_runner.hybrid_lightning_config = None
        tp_worker.sliding_window_size = 4096
        tp_worker.get_tokens_per_layer_info.return_value = (128, 64)
        tp_worker.get_memory_pool.return_value = (mock.MagicMock(), allocator)

        server_args = mock.MagicMock()
        server_args.disable_radix_cache = False
        server_args.disaggregation_decode_enable_radix_cache = decode_radix_enabled
        server_args.disaggregation_mode = disaggregation_mode
        server_args.enable_dp_attention = False

        model_config = mock.MagicMock()
        model_config.is_multimodal = False

        parallel = mock.MagicMock()
        parallel.dcp_enabled = False

        with mock.patch.object(
            kvb, "get_resolved_model_impl", return_value=ModelImpl.AUTO
        ), mock.patch.object(
            kvb, "get_parallel", return_value=parallel
        ), mock.patch.object(
            kvb, "create_tree_cache", return_value=_TREE_CACHE_SENTINEL
        ), mock.patch.object(
            kvb, "init_mm_embedding_cache"
        ):
            return kvb.build_kv_cache(
                server_args=server_args,
                model_config=model_config,
                tp_worker=tp_worker,
                page_size=page_size,
                spec_algorithm=mock.MagicMock(),
                attn_tp_cpu_group=mock.MagicMock(),
                tp_cpu_group=mock.MagicMock(),
                attn_cp_cpu_group=mock.MagicMock(),
                enable_metrics=False,
                enable_kv_cache_events=False,
                ps=mock.MagicMock(),
                tp_group=mock.MagicMock(),
                pp_group=mock.MagicMock(),
                enable_hierarchical_cache=False,
            )

    def test_swa_with_tail_prealloc_allocator_allows_decode_radix(self):
        """SWA + allocator that can prealloc the SWA tail -> guard passes."""
        result = self._call_build_kv_cache(
            is_hybrid_swa=True,
            is_hybrid_ssm=False,
            allocator=_SWATailCapableAllocator(page_size=256),
        )
        self.assertIs(result.tree_cache, _TREE_CACHE_SENTINEL)
        self.assertTrue(result.is_hybrid_swa)

    def test_swa_without_tail_prealloc_allocator_is_rejected(self):
        """SWA + allocator lacking ``alloc_extend_swa_tail`` -> ValueError."""
        with self.assertRaises(ValueError) as ctx:
            self._call_build_kv_cache(
                is_hybrid_swa=True,
                is_hybrid_ssm=False,
                allocator=_NoSWATailAllocator(page_size=256),
            )
        self.assertIn("sliding window attention", str(ctx.exception))

    def test_swa_tail_capable_but_page_size_one_is_rejected(self):
        """SWA-tail capability requires ``page_size > 1``; page_size==1 rejects."""
        with self.assertRaises(ValueError) as ctx:
            self._call_build_kv_cache(
                is_hybrid_swa=True,
                is_hybrid_ssm=False,
                allocator=_SWATailCapableAllocator(page_size=1),
                page_size=1,
            )
        self.assertIn("sliding window attention", str(ctx.exception))

    def test_hybrid_ssm_is_rejected_even_with_tail_prealloc(self):
        """Mamba/SSM stays unsupported regardless of the SWA-tail capability."""
        with self.assertRaises(ValueError) as ctx:
            self._call_build_kv_cache(
                is_hybrid_swa=False,
                is_hybrid_ssm=True,
                allocator=_SWATailCapableAllocator(page_size=256),
            )
        self.assertIn("Mamba/SSM", str(ctx.exception))

    def test_guard_skipped_when_not_decode_disaggregation_mode(self):
        """The guard only runs for the decode server; prefill/null is unaffected."""
        result = self._call_build_kv_cache(
            is_hybrid_swa=True,
            is_hybrid_ssm=False,
            allocator=_NoSWATailAllocator(page_size=256),
            disaggregation_mode="prefill",
        )
        self.assertIs(result.tree_cache, _TREE_CACHE_SENTINEL)

    def test_guard_skipped_when_decode_radix_disabled(self):
        """Without ``--disaggregation-decode-enable-radix-cache`` the guard is a
        no-op even for a non-capable SWA allocator."""
        result = self._call_build_kv_cache(
            is_hybrid_swa=True,
            is_hybrid_ssm=False,
            allocator=_NoSWATailAllocator(page_size=256),
            decode_radix_enabled=False,
        )
        self.assertIs(result.tree_cache, _TREE_CACHE_SENTINEL)

    def test_real_allocator_capability_matches_guard_premise(self):
        """The guard's ``hasattr(alloc, 'alloc_extend_swa_tail')`` premise must
        agree with the real allocator classes it is meant to classify."""
        self.assertTrue(hasattr(SWATokenToKVPoolAllocator, "alloc_extend_swa_tail"))
        self.assertFalse(hasattr(TokenToKVPoolAllocator, "alloc_extend_swa_tail"))
        # DeepSeek-V4 uses a HiSparse allocator that also exposes the capability.
        from sglang.srt.mem_cache.allocator.hisparse import (
            DeepSeekV4HiSparseTokenToKVPoolAllocator,
        )

        self.assertTrue(
            hasattr(DeepSeekV4HiSparseTokenToKVPoolAllocator, "alloc_extend_swa_tail")
        )


# ---------------------------------------------------------------------------
# Part B: UnifiedRadixCache.cache_unfinished_req full-component walk
# ---------------------------------------------------------------------------


class _SWADecodeRadixScenarios:
    """Hybrid-SWA (FULL + SWA) ``cache_unfinished_req`` scenarios.

    Concrete subclasses set ``page_size`` / ``sliding_window_size`` etc.
    """

    page_size: int
    sliding_window_size: int
    kv_size: int = 1024
    max_context_len: int = 1024
    num_layers: int = 24
    full_attention_layer_ids: tuple = (3, 7, 11, 15, 19, 23)
    num_prefix_pages: int = 8

    _rid: int = 0

    # ---- fixture ----

    def _build(self):
        server_args = ServerArgs(model_path="dummy", page_size=self.page_size)
        set_global_server_args_for_scheduler(server_args)
        device = get_device()

        swa_layer_ids = [
            i for i in range(self.num_layers) if i not in self.full_attention_layer_ids
        ]
        req_to_token_pool = ReqToTokenPool(
            size=10,
            max_context_len=self.max_context_len,
            device=device,
            enable_memory_saver=False,
        )
        kv_pool = SWAKVPool(
            size=self.kv_size,
            size_swa=self.kv_size,
            page_size=self.page_size,
            dtype=torch.bfloat16,
            head_num=2,
            head_dim=64,
            swa_attention_layer_ids=swa_layer_ids,
            full_attention_layer_ids=list(self.full_attention_layer_ids),
            device=device,
        )
        allocator = SWATokenToKVPoolAllocator(
            size=self.kv_size,
            size_swa=self.kv_size,
            page_size=self.page_size,
            dtype=torch.bfloat16,
            device=device,
            kvcache=kv_pool,
            need_sort=False,
        )
        params = CacheInitParams(
            req_to_token_pool=req_to_token_pool,
            token_to_kv_pool_allocator=allocator,
            page_size=self.page_size,
            disable=False,
            sliding_window_size=self.sliding_window_size,
            tree_components=(ComponentType.FULL, ComponentType.SWA),
        )
        cache = UnifiedRadixCache(params=params)
        return cache, allocator, req_to_token_pool

    # ---- helpers ----

    def _make_req(self, req_to_token_pool):
        sp = SamplingParams(temperature=0, max_new_tokens=1)
        req = Req(
            rid=self._rid,
            origin_input_text="",
            origin_input_ids=array("q"),
            sampling_params=sp,
        )
        self._rid += 1
        req_to_token_pool.alloc([req])
        return req

    def _seq(self, start: int) -> list:
        return list(range(start, start + self.num_prefix_pages * self.page_size))

    def _alloc(self, allocator, need_size):
        # SWATokenToKVPoolAllocator.alloc() asserts page_size == 1, and
        # alloc_extend() requires batch tensors unsuitable for unit tests, so
        # for paged SWA replicate alloc_extend's core (mirrors the existing
        # UnifiedRadixCache suite helper).
        if self.page_size == 1:
            return allocator.alloc(need_size)
        ps = self.page_size
        aligned = ((need_size + ps - 1) // ps) * ps
        self.assertLessEqual(aligned, allocator.full_attn_allocator.available_size())
        self.assertLessEqual(aligned, allocator.swa_attn_allocator.available_size())
        full_indices = allocator.full_attn_allocator.alloc(aligned)
        swa_indices = allocator.swa_attn_allocator.alloc(aligned)
        self.assertIsNotNone(full_indices)
        self.assertIsNotNone(swa_indices)
        allocator.full_to_swa_index_mapping[full_indices] = swa_indices
        return full_indices[:need_size]

    # ---- tests ----

    def test_swa_collapse_recovers_full_indices(self):
        """When the whole prefix is out of the sliding window (SWA tombstoned),
        ``match_prefix`` collapses to 0 device indices, but the request must be
        re-pointed onto the full-attention indices for the entire prefix."""
        cache, allocator, req_to_token_pool = self._build()
        tokens = self._seq(1)
        total = len(tokens)

        req = self._make_req(req_to_token_pool)
        req.origin_input_ids = array("q", tokens)
        req.output_ids = array("q")
        req.full_untruncated_fill_ids = array("q", tokens)
        req.set_extend_range(0, total)
        kv_indices = self._alloc(allocator, total)
        req_to_token_pool.write((req.req_pool_idx, slice(0, total)), kv_indices)
        req.kv_committed_len = total
        req.last_node = cache.root_node
        req.cache_protected_len = 0
        req.swa_uuid_for_lock = None
        req.extra_key = None
        # Entire prefix sits outside the sliding window -> SWA head is tombstoned.
        req.swa_evicted_seqlen = total

        cache.cache_unfinished_req(req)

        # match_prefix is SWA-gated and collapses at the out-of-window tombstone.
        matched = cache.match_prefix(
            MatchPrefixParams(key=RadixKey(array("q", tokens)))
        )
        self.assertEqual(len(matched.device_indices), 0)

        # The fix re-points the request onto the ungated FULL-component indices.
        self.assertEqual(len(req.prefix_indices), total)
        self.assertEqual(req.cache_protected_len, total)
        self.assertIsNot(req.last_node, cache.root_node)
        # The recovered indices are exactly the deepest full node's data.
        self.assertIsNotNone(req.last_node.component_data[BASE_COMPONENT_TYPE].value)

        cache.dec_lock_ref(
            req.last_node,
            DecLockRefParams(swa_uuid_for_lock=getattr(req, "swa_uuid_for_lock", None)),
        )
        cache.sanity_check()

    def test_shared_out_of_window_prefix_does_not_assert(self):
        """Regression: a still-generating request that shares a fully
        out-of-window prefix must not trip the ``new_prefix_len <=
        len(new_indices)`` assert (the fatal crash the PR fixes)."""
        cache, allocator, req_to_token_pool = self._build()
        tokens = self._seq(1)
        total = len(tokens)

        # Request 1 caches the whole prefix; its SWA is fully out of window so
        # the tree holds FULL for the whole prefix but SWA is tombstoned.
        req1 = self._make_req(req_to_token_pool)
        req1.origin_input_ids = array("q", tokens)
        req1.output_ids = array("q")
        req1.full_untruncated_fill_ids = array("q", tokens)
        req1.set_extend_range(0, total)
        kv1 = self._alloc(allocator, total)
        req_to_token_pool.write((req1.req_pool_idx, slice(0, total)), kv1)
        req1.kv_committed_len = total
        req1.kv_allocated_len = total
        req1.last_node = cache.root_node
        req1.cache_protected_len = 0
        req1.swa_uuid_for_lock = None
        req1.extra_key = None
        req1.swa_evicted_seqlen = total
        cache.cache_finished_req(req1, is_insert=True)

        # Precondition for the bug: the full prefix is resident but match_prefix
        # collapses its device indices to 0 because of the SWA tombstone.
        matched = cache.match_prefix(
            MatchPrefixParams(key=RadixKey(array("q", tokens)))
        )
        self.assertEqual(len(matched.device_indices), 0)

        # Request 2 shares the whole prefix and is still generating. Without the
        # fix this raises AssertionError (new_prefix_len == total, new_indices == 0).
        req2 = self._make_req(req_to_token_pool)
        req2.origin_input_ids = array("q", tokens)
        req2.output_ids = array("q")
        req2.full_untruncated_fill_ids = array("q", tokens)
        req2.set_extend_range(0, total)
        kv2 = self._alloc(allocator, total)
        req_to_token_pool.write((req2.req_pool_idx, slice(0, total)), kv2)
        req2.kv_committed_len = total
        req2.last_node = cache.root_node
        req2.cache_protected_len = 0
        req2.swa_uuid_for_lock = None
        req2.extra_key = None
        req2.swa_evicted_seqlen = total

        cache.cache_unfinished_req(req2)  # must not raise

        self.assertEqual(len(req2.prefix_indices), total)
        self.assertEqual(req2.cache_protected_len, total)
        self.assertIsNot(req2.last_node, cache.root_node)

        cache.dec_lock_ref(
            req2.last_node,
            DecLockRefParams(
                swa_uuid_for_lock=getattr(req2, "swa_uuid_for_lock", None)
            ),
        )
        cache.sanity_check()

    def test_in_window_prefix_baseline_unaffected(self):
        """When nothing is evicted, the full-component walk agrees with the
        normal (ungated) match: the fix must not regress the common path."""
        cache, allocator, req_to_token_pool = self._build()
        tokens = self._seq(1)
        total = len(tokens)

        req = self._make_req(req_to_token_pool)
        req.origin_input_ids = array("q", tokens)
        req.output_ids = array("q")
        req.full_untruncated_fill_ids = array("q", tokens)
        req.set_extend_range(0, total)
        kv_indices = self._alloc(allocator, total)
        req_to_token_pool.write((req.req_pool_idx, slice(0, total)), kv_indices)
        req.kv_committed_len = total
        req.last_node = cache.root_node
        req.cache_protected_len = 0
        req.swa_uuid_for_lock = None
        req.extra_key = None
        req.swa_evicted_seqlen = 0  # everything still in window

        cache.cache_unfinished_req(req)

        matched = cache.match_prefix(
            MatchPrefixParams(key=RadixKey(array("q", tokens)))
        )
        self.assertEqual(len(matched.device_indices), total)
        self.assertEqual(len(req.prefix_indices), total)
        self.assertEqual(req.cache_protected_len, total)
        self.assertIsNot(req.last_node, cache.root_node)

        cache.dec_lock_ref(
            req.last_node,
            DecLockRefParams(swa_uuid_for_lock=getattr(req, "swa_uuid_for_lock", None)),
        )
        cache.sanity_check()


class TestSWADecodeRadixPageSize1(_SWADecodeRadixScenarios, CustomTestCase):
    page_size = 1
    sliding_window_size = 4


class TestSWADecodeRadixPaged(_SWADecodeRadixScenarios, CustomTestCase):
    page_size = 4
    sliding_window_size = 4


class TestSWADecodeRadixLargePage(_SWADecodeRadixScenarios, CustomTestCase):
    page_size = 64
    sliding_window_size = 64
    kv_size = 4096
    max_context_len = 4096


if __name__ == "__main__":
    unittest.main()
