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

import inspect
import unittest
from array import array
from unittest import mock

import torch

import sglang.srt.mem_cache.kv_cache_builder as kvb
from sglang.srt.configs.model_config import ModelImpl
from sglang.srt.managers.schedule_batch import Req, ReqKvInfo
from sglang.srt.mem_cache.allocator import TokenToKVPoolAllocator
from sglang.srt.mem_cache.allocator.swa import SWATokenToKVPoolAllocator
from sglang.srt.mem_cache.base_prefix_cache import (
    DecLockRefParams,
    EvictParams,
    MatchPrefixParams,
)
from sglang.srt.mem_cache.cache_init_params import CacheInitParams
from sglang.srt.mem_cache.memory_pool import ReqToTokenPool
from sglang.srt.mem_cache.radix_cache import RadixKey
from sglang.srt.mem_cache.swa_memory_pool import SWAKVPool
from sglang.srt.mem_cache.unified_cache.components.tree_component import ComponentType
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


class _DSV4NPULikeAllocator(_SWATailCapableAllocator):
    """Shaped like ``DSV4NPUTokenToKVPoolAllocator``: SWA-tail capable, paged,
    but carrying the c4 compressed-attention sub-allocator."""

    def __init__(self, page_size: int = 256):
        super().__init__(page_size=page_size)
        self.c4_attn_allocator = object()


_TREE_CACHE_SENTINEL = object()


def _release_params(req) -> DecLockRefParams:
    """The release token a request owes for the lock it holds on last_node."""
    return DecLockRefParams(
        swa_uuid_for_lock=req.swa_uuid_for_lock,
        skip_lock_node_ids=req.skip_lock_node_ids,
    )


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

        # ``build_kv_cache`` derives is_hybrid_ssm from the arch-config helpers,
        # which would all return a truthy MagicMock for a mocked model_config.
        with mock.patch.object(
            kvb, "get_resolved_model_impl", return_value=ModelImpl.AUTO
        ), mock.patch.object(
            kvb, "get_parallel", return_value=parallel
        ), mock.patch.object(
            kvb, "create_tree_cache", return_value=_TREE_CACHE_SENTINEL
        ), mock.patch.object(
            kvb, "init_mm_embedding_cache"
        ), mock.patch.object(
            kvb, "linear_attn_model_spec", return_value=None
        ), mock.patch.object(
            kvb, "hybrid_gdn_config", return_value=None
        ), mock.patch.object(
            kvb,
            "mamba2_config",
            return_value=object() if is_hybrid_ssm else None,
        ), mock.patch.object(
            kvb, "kimi_linear_config", return_value=None
        ), mock.patch.object(
            kvb, "hybrid_lightning_config", return_value=None
        ), mock.patch.object(
            kvb, "is_deepseek_dsa", return_value=False
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

    def test_dsv4_npu_allocator_is_still_rejected(self):
        """The DSV4 NPU allocator inherits ``alloc_extend_swa_tail`` and runs at
        page_size 256, so the capability probe alone would admit it -- but
        ``_pre_alloc`` raises on any non-zero decode-side prefix there. The guard
        must keep rejecting it at startup rather than let that surface as a
        mid-serving scheduler crash."""
        with self.assertRaises(ValueError) as ctx:
            self._call_build_kv_cache(
                is_hybrid_swa=True,
                is_hybrid_ssm=False,
                allocator=_DSV4NPULikeAllocator(page_size=256),
            )
        self.assertIn("sliding window attention", str(ctx.exception))

    def test_real_npu_allocator_matches_the_c4_premise(self):
        """The exclusion keys on ``c4_attn_allocator``; pin that the real NPU
        allocator sets it and that the ROCm/CUDA SWA allocator does not."""
        from sglang.srt.hardware_backend.npu.dsv4.dsv4_allocator import (
            DSV4NPUTokenToKVPoolAllocator,
        )

        self.assertTrue(
            issubclass(DSV4NPUTokenToKVPoolAllocator, SWATokenToKVPoolAllocator)
        )
        self.assertIn(
            "c4_attn_allocator",
            inspect.getsource(DSV4NPUTokenToKVPoolAllocator.__init__),
        )
        self.assertNotIn(
            "c4_attn_allocator",
            inspect.getsource(SWATokenToKVPoolAllocator.__init__),
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

    def _alloc_swa_tail(self, allocator, seq_len, swa_tail_len):
        """Call the production bs=1 paged SWA-tail allocator."""
        device = allocator.device
        return allocator.alloc_extend_swa_tail(
            prefix_lens=torch.tensor([0], dtype=torch.int64, device=device),
            prefix_lens_cpu=torch.tensor([0], dtype=torch.int64),
            seq_lens=torch.tensor([seq_len], dtype=torch.int64, device=device),
            seq_lens_cpu=torch.tensor([seq_len], dtype=torch.int64),
            last_loc=torch.tensor([-1], dtype=torch.int64, device=device),
            extend_num_tokens=seq_len,
            swa_tail_len=swa_tail_len,
        )

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
        req.last_node = cache.root_node_handle()
        req.cache_protected_len = 0
        req.swa_uuid_for_lock = None
        req.extra_key = None
        # Entire prefix sits outside the sliding window -> SWA head is tombstoned.
        req.kv = ReqKvInfo(kv_allocated_len=total, swa_evicted_seqlen=total)

        cache.cache_unfinished_req(req)

        # match_prefix is SWA-gated and collapses at the out-of-window tombstone.
        matched = cache.match_prefix(
            MatchPrefixParams(key=RadixKey(array("q", tokens)))
        )
        self.assertEqual(len(matched.device_indices), 0)

        # The fix re-points the request onto the ungated FULL-component indices.
        self.assertEqual(len(req.prefix_indices), total)
        self.assertEqual(req.cache_protected_len, total)
        self.assertNotEqual(req.last_node, cache.root_node_handle())
        # The recovered indices are exactly the deepest full node's data.
        last_node = cache.resolve_node_handle(req.last_node)
        self.assertIsNotNone(last_node.component_data[BASE_COMPONENT_TYPE].value)

        cache.dec_lock_ref(req.last_node, _release_params(req))
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
        req1.last_node = cache.root_node_handle()
        req1.cache_protected_len = 0
        req1.swa_uuid_for_lock = None
        req1.extra_key = None
        req1.kv = ReqKvInfo(kv_allocated_len=total, swa_evicted_seqlen=total)
        cache.cache_finished_req(req1, is_insert=True, kv_len_to_handle=total)

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
        req2.last_node = cache.root_node_handle()
        req2.cache_protected_len = 0
        req2.swa_uuid_for_lock = None
        req2.extra_key = None
        req2.kv = ReqKvInfo(kv_allocated_len=total, swa_evicted_seqlen=total)

        cache.cache_unfinished_req(req2)  # must not raise

        self.assertEqual(len(req2.prefix_indices), total)
        self.assertEqual(req2.cache_protected_len, total)
        self.assertNotEqual(req2.last_node, cache.root_node_handle())

        cache.dec_lock_ref(req2.last_node, _release_params(req2))
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
        req.last_node = cache.root_node_handle()
        req.cache_protected_len = 0
        req.swa_uuid_for_lock = None
        req.extra_key = None
        req.kv = ReqKvInfo(
            kv_allocated_len=total, swa_evicted_seqlen=0  # everything still in window
        )

        cache.cache_unfinished_req(req)

        matched = cache.match_prefix(
            MatchPrefixParams(key=RadixKey(array("q", tokens)))
        )
        self.assertEqual(len(matched.device_indices), total)
        self.assertEqual(len(req.prefix_indices), total)
        self.assertEqual(req.cache_protected_len, total)
        self.assertNotEqual(req.last_node, cache.root_node_handle())

        cache.dec_lock_ref(req.last_node, _release_params(req))
        cache.sanity_check()

    def test_empty_bigram_key_does_not_raise(self):
        """`match_full_device_prefix` walks the tree itself, so it needs the same
        empty-key guard `match_prefix` has: `RadixKey.child_key` indexes into the
        token array and raises IndexError on an empty bigram (EAGLE) key. A short
        request under EAGLE spec decoding page-aligns to a zero-length key."""
        cache, _, _ = self._build()
        empty = RadixKey(array("q"), None, is_bigram=True)
        self.assertEqual(len(empty), 0)

        indices, node_id = cache.tree_core.match_full_device_prefix(empty)

        self.assertEqual(len(indices), 0)
        self.assertEqual(node_id, cache.root_node_handle())

    def test_real_swa_tail_allocation_caches_and_frees_cleanly(self):
        """Exercise the real tail-only allocator, including zero head mappings."""
        if self.page_size == 1:
            self.skipTest("SWA-tail preallocation requires page_size > 1")

        cache, allocator, req_to_token_pool = self._build()
        tokens = self._seq(1)
        total = len(tokens)
        # Mirror DecodePreallocQueue._swa_tail_len: the window is page-aligned
        # *downwards*, so the tail is the window rounded out to the page it
        # starts in -- equal to the window only when window % page_size == 0.
        # Passing the raw window instead would hand the allocator a length it
        # then rounds up on its own (num_swa_pages = ceil(tail / page_size)),
        # and the debit would no longer equal what was asked for.
        window_start = ((total - self.sliding_window_size) // self.page_size) * (
            self.page_size
        )
        swa_tail_len = total - window_start
        self.assertEqual(swa_tail_len % self.page_size, 0)
        self.assertLess(swa_tail_len, total)

        full_before = allocator.full_available_size()
        swa_before = allocator.swa_available_size()
        kv_indices = self._alloc_swa_tail(allocator, total, swa_tail_len)
        self.assertIsNotNone(kv_indices)
        self.assertEqual(len(kv_indices), total)
        self.assertEqual(full_before - allocator.full_available_size(), total)
        self.assertEqual(swa_before - allocator.swa_available_size(), swa_tail_len)

        mapping = allocator.full_to_swa_index_mapping[kv_indices.to(torch.int64)]
        self.assertTrue(torch.all(mapping[:-swa_tail_len] == 0).item())
        self.assertTrue(torch.all(mapping[-swa_tail_len:] > 0).item())

        req = self._make_req(req_to_token_pool)
        req.origin_input_ids = array("q", tokens)
        req.output_ids = array("q")
        req.full_untruncated_fill_ids = array("q", tokens)
        req.set_extend_range(0, total)
        req_to_token_pool.write((req.req_pool_idx, slice(0, total)), kv_indices)
        req.kv_committed_len = total
        req.last_node = cache.root_node_handle()
        req.cache_protected_len = 0
        req.extra_key = None
        req.kv = ReqKvInfo(
            kv_allocated_len=total, swa_evicted_seqlen=total - swa_tail_len
        )

        cache.cache_unfinished_req(req)
        self.assertEqual(len(req.prefix_indices), total)
        self.assertEqual(req.cache_protected_len, total)
        cache.cache_finished_req(req, is_insert=True, kv_len_to_handle=total)
        cache.sanity_check()

        cache.evict(EvictParams(num_tokens=total, swa_num_tokens=swa_tail_len))
        self.assertEqual(allocator.full_available_size(), full_before)
        self.assertEqual(allocator.swa_available_size(), swa_before)
        cache.sanity_check()

    def test_releasing_full_only_lock_does_not_steal_restored_swa_lock(self):
        """A tombstone skipped by A may later be restored and locked by B."""
        if self.page_size == 1:
            self.skipTest("SWA-tail preallocation requires page_size > 1")

        cache, allocator, req_to_token_pool = self._build()
        tokens = self._seq(1)
        total = len(tokens)

        # A inserts a Full-only path and locks it while every SWA component on
        # that path is a tombstone. Its ownership token must remember the skip.
        req_a = self._make_req(req_to_token_pool)
        req_a.origin_input_ids = array("q", tokens)
        req_a.output_ids = array("q")
        req_a.full_untruncated_fill_ids = array("q", tokens)
        req_a.set_extend_range(0, total)
        kv_a = self._alloc(allocator, total)
        req_to_token_pool.write((req_a.req_pool_idx, slice(0, total)), kv_a)
        req_a.kv_committed_len = total
        req_a.last_node = cache.root_node_handle()
        req_a.cache_protected_len = 0
        req_a.extra_key = None
        req_a.kv = ReqKvInfo(kv_allocated_len=total, swa_evicted_seqlen=total)
        cache.cache_unfinished_req(req_a)

        a_params = _release_params(req_a)
        self.assertIn(ComponentType.SWA, a_params.skip_lock_node_ids)
        self.assertIn(req_a.last_node, a_params.skip_lock_node_ids[ComponentType.SWA])

        # B restores the final SWA page under the same Full path and locks it.
        req_b = self._make_req(req_to_token_pool)
        req_b.origin_input_ids = array("q", tokens)
        req_b.output_ids = array("q")
        req_b.full_untruncated_fill_ids = array("q", tokens)
        req_b.set_extend_range(0, total)
        kv_b = self._alloc(allocator, total)
        req_to_token_pool.write((req_b.req_pool_idx, slice(0, total)), kv_b)
        req_b.kv_committed_len = total
        req_b.last_node = cache.root_node_handle()
        req_b.cache_protected_len = 0
        req_b.extra_key = None
        req_b.kv = ReqKvInfo(
            kv_allocated_len=total, swa_evicted_seqlen=total - self.page_size
        )
        cache.cache_unfinished_req(req_b)

        b_swa = cache.resolve_node_handle(req_b.last_node).component_data[
            ComponentType.SWA
        ]
        self.assertIsNotNone(b_swa.value)
        self.assertEqual(b_swa.lock_ref, 1)

        # A never owned B's restored SWA lock. Releasing A must replay its skip
        # set and leave B's ref intact and non-evictable.
        cache.dec_lock_ref(req_a.last_node, a_params)
        self.assertEqual(b_swa.lock_ref, 1)
        evicted = cache.evict(EvictParams(swa_num_tokens=self.page_size))
        self.assertEqual(evicted.swa_num_tokens_evicted, 0)
        self.assertIsNotNone(b_swa.value)

        cache.dec_lock_ref(req_b.last_node, _release_params(req_b))
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


class TestSWADecodeRadixProductionPage(_SWADecodeRadixScenarios, CustomTestCase):
    page_size = 256
    sliding_window_size = 256
    kv_size = 8192
    max_context_len = 8192


class TestSWADecodeRadixWindowSmallerThanPage(_SWADecodeRadixScenarios, CustomTestCase):
    """The geometry DeepSeek-V4-Pro actually serves: window 128 < page 256.

    Every other parameterization has ``sliding_window_size >= page_size``, which
    hides a whole regime of ``_swa_tail_len``: page-aligning ``seq_len - window``
    downwards lands at a different distance from ``seq_len`` once the window is
    shorter than a page, so the tail spans window..window+page rather than
    page..2*page. That length is the threshold in ``_takes_swa_tail_path``, i.e.
    it decides tail-only versus the full ``alloc_extend`` fallback -- so the
    untested regime is the one where the decision boundary actually sits.
    """

    page_size = 256
    sliding_window_size = 128
    kv_size = 8192
    max_context_len = 8192


if __name__ == "__main__":
    unittest.main()
