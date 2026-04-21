"""Unit tests for UnifiedRadixCache"""

import unittest
from dataclasses import dataclass
from typing import Optional

import torch

from sglang.srt.configs.mamba_utils import Mamba2CacheParams, Mamba2StateShape
from sglang.srt.environ import envs
from sglang.srt.managers.schedule_batch import Req
from sglang.srt.mem_cache.allocator import TokenToKVPoolAllocator
from sglang.srt.mem_cache.base_prefix_cache import (
    DecLockRefParams,
    EvictParams,
    EvictResult,
    InsertParams,
    MatchPrefixParams,
)
from sglang.srt.mem_cache.cache_init_params import CacheInitParams
from sglang.srt.mem_cache.common import available_and_evictable_str
from sglang.srt.mem_cache.memory_pool import (
    HybridLinearKVPool,
    HybridReqToTokenPool,
    MHATokenToKVPool,
    ReqToTokenPool,
)
from sglang.srt.mem_cache.radix_cache import RadixKey
from sglang.srt.mem_cache.swa_memory_pool import SWAKVPool, SWATokenToKVPoolAllocator
from sglang.srt.mem_cache.unified_cache_components.tree_component import ComponentType
from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.srt.server_args import (
    ServerArgs,
    get_global_server_args,
    set_global_server_args_for_scheduler,
)
from sglang.srt.utils import get_device
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=30, suite="stage-b-test-1-gpu-small")


@dataclass(frozen=True)
class CacheConfig:
    # Tree
    page_size: int = 1
    components: tuple[ComponentType, ...] = (ComponentType.FULL,)

    # Layer split (only matters for SWA/Mamba)
    num_layers: int = 24
    full_attention_layer_ids: tuple[int, ...] = (3, 7, 11, 15, 19, 23)

    # SWA
    sliding_window_size: Optional[int] = None

    # Mamba
    enable_mamba_extra_buffer: bool = False
    mamba_cache_size: int = 20
    mamba_intermediate_size: int = 256
    mamba_n_groups: int = 1
    mamba_num_heads: int = 2
    mamba_head_dim: int = 16
    mamba_state_size: int = 16
    mamba_conv_kernel: int = 4

    # Model / pool
    kv_size: int = 256
    max_num_reqs: int = 10
    max_context_len: int = 512
    head_num: int = 2
    head_dim: int = 64
    dtype: torch.dtype = torch.bfloat16

    @property
    def has_mamba(self) -> bool:
        return ComponentType.MAMBA in self.components

    @property
    def has_swa(self) -> bool:
        return ComponentType.SWA in self.components

    @property
    def non_full_layer_ids(self) -> list[int]:
        full = set(self.full_attention_layer_ids)
        return [i for i in range(self.num_layers) if i not in full]

    @property
    def label(self) -> str:
        comp = "_".join(c.name for c in self.components)
        parts = [f"{comp}_ps{self.page_size}"]
        if self.sliding_window_size is not None:
            parts.append(f"sw{self.sliding_window_size}")
        defaults = self.__dataclass_fields__
        if (
            self.head_num != defaults["head_num"].default
            or self.num_layers != defaults["num_layers"].default
        ):
            parts.append(f"h{self.head_num}l{self.num_layers}")
        return "_".join(parts)


def build_fixture(cfg: CacheConfig):
    """Create (tree, allocator, req_to_token_pool) from a CacheConfig."""
    set_global_server_args_for_scheduler(
        ServerArgs(model_path="dummy", page_size=cfg.page_size)
    )
    device = get_device()

    mamba2_cache_params = None
    if cfg.has_mamba:
        with envs.SGLANG_MAMBA_SSM_DTYPE.override("bfloat16"):
            shape = Mamba2StateShape.create(
                tp_world_size=1,
                intermediate_size=cfg.mamba_intermediate_size,
                n_groups=cfg.mamba_n_groups,
                num_heads=cfg.mamba_num_heads,
                head_dim=cfg.mamba_head_dim,
                state_size=cfg.mamba_state_size,
                conv_kernel=cfg.mamba_conv_kernel,
            )
            mamba2_cache_params = Mamba2CacheParams(
                shape=shape, layers=cfg.non_full_layer_ids
            )
        req_to_token_pool = HybridReqToTokenPool(
            size=cfg.max_num_reqs,
            mamba_size=cfg.mamba_cache_size,
            mamba_spec_state_size=cfg.max_num_reqs,
            max_context_len=cfg.max_context_len,
            device=device,
            enable_memory_saver=False,
            cache_params=mamba2_cache_params,
            mamba_layer_ids=cfg.non_full_layer_ids,
            enable_mamba_extra_buffer=cfg.enable_mamba_extra_buffer,
            speculative_num_draft_tokens=3,
        )
    else:
        req_to_token_pool = ReqToTokenPool(
            size=cfg.max_num_reqs,
            max_context_len=cfg.max_context_len,
            device=device,
            enable_memory_saver=False,
        )

    if cfg.has_swa:
        kv_pool = SWAKVPool(
            size=cfg.kv_size,
            size_swa=cfg.kv_size,
            page_size=cfg.page_size,
            dtype=cfg.dtype,
            head_num=cfg.head_num,
            head_dim=cfg.head_dim,
            swa_attention_layer_ids=cfg.non_full_layer_ids,
            full_attention_layer_ids=cfg.full_attention_layer_ids,
            enable_kvcache_transpose=False,
            device=device,
        )
        allocator = SWATokenToKVPoolAllocator(
            size=cfg.kv_size,
            size_swa=cfg.kv_size,
            page_size=cfg.page_size,
            dtype=cfg.dtype,
            device=device,
            kvcache=kv_pool,
            need_sort=False,
        )
    elif cfg.has_mamba:
        kv_pool = HybridLinearKVPool(
            size=cfg.kv_size,
            dtype=cfg.dtype,
            page_size=cfg.page_size,
            head_num=cfg.head_num,
            head_dim=cfg.head_dim,
            full_attention_layer_ids=cfg.full_attention_layer_ids,
            enable_kvcache_transpose=False,
            device=device,
            enable_memory_saver=False,
            mamba_pool=req_to_token_pool.mamba_pool,
        )
        allocator = TokenToKVPoolAllocator(
            size=cfg.kv_size,
            dtype=cfg.dtype,
            device=device,
            kvcache=kv_pool,
            need_sort=False,
        )
    else:
        kv_pool = MHATokenToKVPool(
            size=cfg.kv_size,
            page_size=cfg.page_size,
            dtype=cfg.dtype,
            head_num=cfg.head_num,
            head_dim=cfg.head_dim,
            layer_num=cfg.num_layers,
            device=device,
            enable_memory_saver=False,
        )
        allocator = TokenToKVPoolAllocator(
            size=cfg.kv_size,
            dtype=cfg.dtype,
            device=device,
            kvcache=kv_pool,
            need_sort=False,
        )

    tree = UnifiedRadixCache(
        params=CacheInitParams(
            req_to_token_pool=req_to_token_pool,
            token_to_kv_pool_allocator=allocator,
            page_size=cfg.page_size,
            disable=False,
            sliding_window_size=cfg.sliding_window_size,
            tree_components=cfg.components,
            enable_mamba_extra_buffer=cfg.enable_mamba_extra_buffer,
        ),
    )

    return tree, allocator, req_to_token_pool


class UnifiedRadixCacheSuite:

    cfg: CacheConfig
    _rid: int = 0

    def _make_req(self, req_to_token_pool):
        sp = SamplingParams(temperature=0, max_new_tokens=1)
        req = Req(
            rid=self._rid,
            origin_input_text="",
            origin_input_ids=[],
            sampling_params=sp,
        )
        self._rid += 1
        req_to_token_pool.alloc([req])
        return req

    def _make_seq(self, start: int, num_pages: int) -> list[int]:
        """Page-aligned token sequence of num_pages pages."""
        page_size = self.cfg.page_size
        return list(range(start, start + num_pages * page_size))

    def _alloc(self, allocator, need_size):
        if not (self.cfg.has_swa and self.cfg.page_size > 1):
            return allocator.alloc(need_size)

        # SWATokenToKVPoolAllocator.alloc() asserts page_size == 1, and
        # alloc_extend() requires batch tensors unsuitable for unit tests.
        # Replicate alloc_extend's core logic here.
        ps = self.cfg.page_size
        aligned = ((need_size + ps - 1) // ps) * ps
        if aligned > allocator.full_attn_allocator.available_size():
            return None
        if aligned > allocator.swa_attn_allocator.available_size():
            return None
        full_indices = allocator.full_attn_allocator.alloc(aligned)
        swa_indices = allocator.swa_attn_allocator.alloc(aligned)
        assert full_indices is not None and swa_indices is not None
        allocator.full_to_swa_index_mapping[full_indices] = swa_indices
        return full_indices[:need_size]

    def _insert(self, tree, allocator, req_to_token_pool, tokens):
        """Insert tokens, attaching mamba data when the config has mamba."""
        key = RadixKey(tokens)
        value = self._alloc(allocator, len(tokens))
        params = InsertParams(key=key, value=value[: len(key)])
        if self.cfg.has_mamba:
            req = self._make_req(req_to_token_pool)
            params.mamba_value = req.mamba_pool_idx.unsqueeze(0)
        return tree.insert(params)

    def test_insert_and_match_basic(self):
        tree, allocator, req_to_token_pool = build_fixture(self.cfg)

        seq_a = self._make_seq(1, 2)
        seq_b = seq_a + self._make_seq(1000, 1)

        self._insert(tree, allocator, req_to_token_pool, seq_a)
        result = self._insert(tree, allocator, req_to_token_pool, seq_b)
        self.assertEqual(result.prefix_len, len(seq_a))

        m = tree.match_prefix(MatchPrefixParams(key=RadixKey(seq_b)))
        self.assertEqual(len(m.device_indices), len(seq_b))

        m = tree.match_prefix(
            MatchPrefixParams(key=RadixKey(seq_a + self._make_seq(9000, 1)))
        )
        self.assertEqual(len(m.device_indices), len(seq_a))

        m = tree.match_prefix(MatchPrefixParams(key=RadixKey(self._make_seq(5000, 2))))
        self.assertEqual(len(m.device_indices), 0)

        tree.sanity_check()

    def test_shared_prefix_split(self):
        tree, allocator, req_to_token_pool = build_fixture(self.cfg)
        base = self._make_seq(1, 2)
        self._insert(tree, allocator, req_to_token_pool, base)

        branch_a = base + self._make_seq(100, 2)
        branch_b = base + self._make_seq(200, 2)

        result_a = self._insert(tree, allocator, req_to_token_pool, branch_a)
        self.assertEqual(result_a.prefix_len, len(base))
        result_b = self._insert(tree, allocator, req_to_token_pool, branch_b)
        self.assertEqual(result_b.prefix_len, len(base))

        for seq in (branch_a, branch_b):
            m = tree.match_prefix(MatchPrefixParams(key=RadixKey(seq)))
            self.assertEqual(len(m.device_indices), len(seq))

        m = tree.match_prefix(
            MatchPrefixParams(key=RadixKey(base + self._make_seq(999, 1)))
        )
        self.assertEqual(len(m.device_indices), len(base))
        tree.sanity_check()

    def test_evict_basic(self):
        tree, allocator, req_to_token_pool = build_fixture(self.cfg)
        seq_a = self._make_seq(1, 2)
        seq_b = self._make_seq(500, 2)

        self._insert(tree, allocator, req_to_token_pool, seq_a)
        self._insert(tree, allocator, req_to_token_pool, seq_b)
        total = len(seq_a) + len(seq_b)
        self.assertEqual(tree.full_evictable_size(), total)

        result = tree.evict(EvictParams(num_tokens=len(seq_a)))
        self.assertIsInstance(result, EvictResult)
        self.assertGreaterEqual(result.num_tokens_evicted, len(seq_a))
        self.assertTrue(tree.full_evictable_size() <= len(seq_b))
        tree.sanity_check()

    def test_evict_respects_lock_ref(self):
        """Lock protects from eviction; unlock allows re-eviction."""
        tree, allocator, req_to_token_pool = build_fixture(self.cfg)
        seq_a = self._make_seq(1, 2)
        seq_b = self._make_seq(500, 2)
        self._insert(tree, allocator, req_to_token_pool, seq_a)
        self._insert(tree, allocator, req_to_token_pool, seq_b)

        m = tree.match_prefix(MatchPrefixParams(key=RadixKey(seq_a)))
        lock_result = tree.inc_lock_ref(m.last_device_node)

        result = tree.evict(EvictParams(num_tokens=len(seq_a) + len(seq_b)))
        self.assertGreaterEqual(result.num_tokens_evicted, len(seq_b))

        m = tree.match_prefix(MatchPrefixParams(key=RadixKey(seq_a)))
        self.assertEqual(len(m.device_indices), len(seq_a))

        # Unlock -> should now be evictable
        tree.dec_lock_ref(
            m.last_device_node,
            DecLockRefParams(
                swa_uuid_for_lock=getattr(lock_result, "swa_uuid_for_lock", None)
            ),
        )
        result = tree.evict(EvictParams(num_tokens=len(seq_a)))
        self.assertGreaterEqual(result.num_tokens_evicted, len(seq_a))
        tree.sanity_check()

    def test_evict_empty_tree(self):
        tree, _, _ = build_fixture(self.cfg)
        evict_params = EvictParams(num_tokens=10)
        if self.cfg.has_mamba:
            evict_params.mamba_num = 5
        result = tree.evict(evict_params)
        self.assertEqual(result.num_tokens_evicted, 0)
        if self.cfg.has_mamba:
            self.assertEqual(result.mamba_num_evicted, 0)
        tree.sanity_check()

    def test_evict_until_empty(self):
        tree, allocator, req_to_token_pool = build_fixture(self.cfg)
        seqs = [self._make_seq(i * 100, 2) for i in range(5)]
        for s in seqs:
            self._insert(tree, allocator, req_to_token_pool, s)
        total = sum(len(s) for s in seqs)
        self.assertEqual(tree.full_evictable_size(), total)

        result = tree.evict(EvictParams(num_tokens=total * 2))
        self.assertGreaterEqual(result.num_tokens_evicted, total)
        self.assertEqual(tree.full_evictable_size(), 0)
        if self.cfg.has_mamba:
            self.assertEqual(tree.mamba_evictable_size(), 0)

        m = tree.match_prefix(MatchPrefixParams(key=RadixKey(seqs[0])))
        self.assertEqual(len(m.device_indices), 0)
        tree.sanity_check()

    def test_prev_prefix_len(self):
        """Three-step test: free overlap, free partial, no free."""
        tree, allocator, req_to_token_pool = build_fixture(self.cfg)
        initial_avail = allocator.available_size()

        seq_1p = self._make_seq(1, 1)  # 1 page
        seq_2p = self._make_seq(1, 2)  # 2 pages (extends seq_1p)
        seq_3p = self._make_seq(1, 3)  # 3 pages (extends seq_2p)

        # Step 1: insert 1 page
        self._insert(tree, allocator, req_to_token_pool, seq_1p)
        self.assertEqual(allocator.available_size(), initial_avail - len(seq_1p))

        # Step 2: insert 2 pages with prev_prefix_len=0 → frees overlap of 1 page
        key_2p = RadixKey(seq_2p)
        value_2p = self._alloc(allocator, len(seq_2p))
        params = InsertParams(
            key=key_2p,
            value=value_2p[: len(key_2p)],
            prev_prefix_len=0,
        )
        if self.cfg.has_mamba:
            req = self._make_req(req_to_token_pool)
            params.mamba_value = req.mamba_pool_idx.unsqueeze(0)
        result = tree.insert(params)
        self.assertEqual(result.prefix_len, len(seq_1p))
        self.assertEqual(
            allocator.available_size(),
            initial_avail - len(seq_1p) - (len(seq_2p) - len(seq_1p)),
        )

        # Step 3: insert 3 pages with prev_prefix_len=len(seq_2p) → nothing freed
        avail_before = allocator.available_size()
        key_3p = RadixKey(seq_3p)
        value_3p = self._alloc(allocator, len(seq_3p))
        params = InsertParams(
            key=key_3p,
            value=value_3p[: len(key_3p)],
            prev_prefix_len=len(seq_2p),
        )
        if self.cfg.has_mamba:
            req = self._make_req(req_to_token_pool)
            params.mamba_value = req.mamba_pool_idx.unsqueeze(0)
        result = tree.insert(params)
        self.assertEqual(result.prefix_len, len(seq_2p))
        # alloc(3p), freed 0 (prev_prefix_len covers entire overlap), stored 1p new → net -3p
        self.assertEqual(allocator.available_size(), avail_before - len(seq_3p))
        tree.sanity_check()

    def test_node_split_at_boundary(self):
        tree, allocator, req_to_token_pool = build_fixture(self.cfg)
        base = self._make_seq(1, 3)
        self._insert(tree, allocator, req_to_token_pool, base)

        fork_a = base + self._make_seq(100, 1)
        fork_b = base + self._make_seq(200, 1)

        self._insert(tree, allocator, req_to_token_pool, fork_a)
        result = self._insert(tree, allocator, req_to_token_pool, fork_b)
        self.assertEqual(result.prefix_len, len(base))

        for seq in (fork_a, fork_b):
            m = tree.match_prefix(MatchPrefixParams(key=RadixKey(seq)))
            self.assertEqual(len(m.device_indices), len(seq))

        m = tree.match_prefix(
            MatchPrefixParams(key=RadixKey(base + self._make_seq(999, 1)))
        )
        self.assertEqual(len(m.device_indices), len(base))
        tree.sanity_check()

    def test_cache_finished_req_insert(self):
        tree, allocator, req_to_token_pool = build_fixture(self.cfg)
        ps = self.cfg.page_size

        req = self._make_req(req_to_token_pool)
        input_ids = self._make_seq(1, 3)
        output_ids = self._make_seq(2000, 1)
        req.origin_input_ids = input_ids
        req.output_ids = output_ids
        kv_len = len(input_ids) + len(output_ids)
        kv_indices = self._alloc(allocator, kv_len)
        req_to_token_pool.write((req.req_pool_idx, slice(0, kv_len)), kv_indices)
        req.kv_committed_len = kv_len
        req.last_node = tree.root_node
        req.cache_protected_len = 0
        req.swa_uuid_for_lock = None
        req.extra_key = None
        req.fill_ids = input_ids + output_ids
        if self.cfg.has_mamba:
            req.mamba_last_track_seqlen = kv_len

        tree.cache_finished_req(req, is_insert=True)

        all_ids = input_ids + output_ids
        aligned_len = (len(all_ids) // ps) * ps
        m = tree.match_prefix(MatchPrefixParams(key=RadixKey(all_ids[:aligned_len])))
        self.assertEqual(len(m.device_indices), aligned_len)
        tree.sanity_check()

    def test_cache_finished_req_strips_thinking(self):
        tree, allocator, req_to_token_pool = build_fixture(self.cfg)
        ps = self.cfg.page_size

        req = self._make_req(req_to_token_pool)
        prompt_ids = self._make_seq(1, 3)
        output_ids = self._make_seq(2000, 7)
        req.origin_input_ids = prompt_ids
        req.output_ids = output_ids
        req.fill_ids = prompt_ids + output_ids
        kv_len = len(req.fill_ids)
        kv_indices = self._alloc(allocator, kv_len)
        req_to_token_pool.write((req.req_pool_idx, slice(0, kv_len)), kv_indices)
        req.kv_committed_len = kv_len
        req.kv_allocated_len = kv_len
        req.last_node = tree.root_node
        req.cache_protected_len = 0
        req.swa_uuid_for_lock = None
        req.extra_key = None
        if self.cfg.has_mamba:
            req.mamba_last_track_seqlen = kv_len
        req.reasoning_tokens = 1

        get_global_server_args().strip_thinking_cache = True
        try:
            avail_before = allocator.available_size()
            tree.cache_finished_req(req, is_insert=True)
            start_p, end_p = req.pop_overallocated_kv_cache()
        finally:
            get_global_server_args().strip_thinking_cache = False
        if ps > 1:
            start_p = ((start_p + ps - 1) // ps) * ps
        if start_p < end_p:
            allocator.free(
                req_to_token_pool.req_to_token[req.req_pool_idx][start_p:end_p]
            )

        prompt_aligned = (len(prompt_ids) // ps) * ps
        # Thinking+answer must not be reachable past the prompt.
        m = tree.match_prefix(MatchPrefixParams(key=RadixKey(prompt_ids + output_ids)))
        self.assertEqual(len(m.device_indices), prompt_aligned)
        # Only prompt-aligned pages remain owned by the tree.
        self.assertEqual(
            allocator.available_size(), avail_before + kv_len - prompt_aligned
        )
        tree.sanity_check()

    def test_cache_finished_req_no_insert(self):
        tree, allocator, req_to_token_pool = build_fixture(self.cfg)
        req = self._make_req(req_to_token_pool)
        tokens = self._make_seq(1, 2)
        req.origin_input_ids = tokens
        req.output_ids = []
        kv_len = len(tokens)
        kv_indices = self._alloc(allocator, kv_len)
        req_to_token_pool.write((req.req_pool_idx, slice(0, kv_len)), kv_indices)
        req.kv_committed_len = kv_len
        req.last_node = tree.root_node
        req.cache_protected_len = 0
        req.swa_uuid_for_lock = None
        req.extra_key = None
        req.fill_ids = tokens

        avail_before = allocator.available_size()
        tree.cache_finished_req(req, is_insert=False)

        self.assertEqual(allocator.available_size(), avail_before + kv_len)
        m = tree.match_prefix(MatchPrefixParams(key=RadixKey(tokens)))
        self.assertEqual(len(m.device_indices), 0)
        tree.sanity_check()

    def test_cache_unfinished_req(self):
        tree, allocator, req_to_token_pool = build_fixture(self.cfg)

        req = self._make_req(req_to_token_pool)
        tokens = self._make_seq(1, 3)
        req.origin_input_ids = tokens
        req.output_ids = []
        req.fill_ids = tokens[:]
        kv_len = len(tokens)
        kv_indices = self._alloc(allocator, kv_len)
        req_to_token_pool.write((req.req_pool_idx, slice(0, kv_len)), kv_indices)
        req.kv_committed_len = kv_len
        req.last_node = tree.root_node
        req.cache_protected_len = 0
        req.swa_uuid_for_lock = None
        req.extra_key = None
        if self.cfg.has_mamba:
            req.mamba_last_track_seqlen = kv_len

        tree.cache_unfinished_req(req)

        self.assertGreater(len(req.prefix_indices), 0)
        self.assertEqual(req.cache_protected_len, len(req.prefix_indices))
        self.assertIsNotNone(req.last_node)

        tree.dec_lock_ref(
            req.last_node,
            DecLockRefParams(swa_uuid_for_lock=getattr(req, "swa_uuid_for_lock", None)),
        )
        tree.sanity_check()

    def test_diagnostics(self):
        tree, allocator, req_to_token_pool = build_fixture(self.cfg)
        self._insert(tree, allocator, req_to_token_pool, self._make_seq(1, 2))

        diag = tree.available_and_evictable_str()
        self.assertIn("Available full tokens", diag)
        if self.cfg.has_mamba:
            self.assertIn("mamba", diag.lower())
        if self.cfg.has_swa:
            self.assertIn("swa", diag.lower())

        diag2 = available_and_evictable_str(tree)
        self.assertIn("Available full tokens", diag2)
        tree.pretty_print()
        tree.sanity_check()

    def test_multi_branch_tree(self):
        tree, allocator, req_to_token_pool = build_fixture(self.cfg)
        base = self._make_seq(1, 2)
        self._insert(tree, allocator, req_to_token_pool, base)

        for suffix_start in [100, 200, 300]:
            seq = base + self._make_seq(suffix_start, 2)
            self._insert(tree, allocator, req_to_token_pool, seq)

        for suffix_start in [100, 200, 300]:
            seq = base + self._make_seq(suffix_start, 2)
            m = tree.match_prefix(MatchPrefixParams(key=RadixKey(seq)))
            self.assertEqual(len(m.device_indices), len(seq))

        m = tree.match_prefix(
            MatchPrefixParams(key=RadixKey(base + self._make_seq(999, 1)))
        )
        self.assertEqual(len(m.device_indices), len(base))
        tree.sanity_check()

    def test_paged_child_key_is_tuple(self):
        if self.cfg.page_size == 1:
            self.skipTest("page_size > 1 only")
        tree, _, _ = build_fixture(self.cfg)
        key = RadixKey(self._make_seq(1, 1))
        child_key = tree.get_child_key_fn(key)
        self.assertIsInstance(child_key, tuple)

    def test_paged_match_truncates_unaligned_key(self):
        """match_prefix internally aligns keys to page boundary."""
        if self.cfg.page_size == 1:
            self.skipTest("page_size > 1 only")
        ps = self.cfg.page_size
        tree, allocator, req_to_token_pool = build_fixture(self.cfg)
        seq = self._make_seq(1, 2)
        self._insert(tree, allocator, req_to_token_pool, seq)

        # Tree truncates unaligned tail internally, so it matches the seq prefix.
        unaligned = seq + list(range(9000, 9000 + ps - 1))
        m = tree.match_prefix(MatchPrefixParams(key=RadixKey(unaligned)))
        self.assertEqual(len(m.device_indices), len(seq))

        # Below-page-size key aligns to 0 -> no match.
        m = tree.match_prefix(MatchPrefixParams(key=RadixKey(seq[: ps - 1])))
        self.assertEqual(len(m.device_indices), 0)

        tree.sanity_check()

    def test_paged_page_boundary_mismatch(self):
        if self.cfg.page_size == 1:
            self.skipTest("page_size > 1 only")
        ps = self.cfg.page_size
        tree, allocator, req_to_token_pool = build_fixture(self.cfg)
        first_page = self._make_seq(1, 1)
        seq = self._make_seq(1, 2)
        # Insert first page so it retains component data after the split
        # triggered by the partial-page match below.
        self._insert(tree, allocator, req_to_token_pool, first_page)
        self._insert(tree, allocator, req_to_token_pool, seq)

        # Mismatch in second page → only first page matches
        bad_page2 = seq[:ps] + [9999] * ps
        m = tree.match_prefix(MatchPrefixParams(key=RadixKey(bad_page2)))
        self.assertEqual(len(m.device_indices), ps)

        # Mismatch in first page → 0 match
        bad_page1 = [9999] + seq[1:]
        m = tree.match_prefix(MatchPrefixParams(key=RadixKey(bad_page1)))
        self.assertEqual(len(m.device_indices), 0)
        tree.sanity_check()

    def test_paged_cache_finished_unaligned_tail_freed(self):
        if self.cfg.page_size == 1:
            self.skipTest("page_size > 1 only")
        if self.cfg.has_swa:
            self.skipTest("SWA paged allocator accounts in pages, not tokens")
        ps = self.cfg.page_size
        tree, allocator, req_to_token_pool = build_fixture(self.cfg)

        tail_extra = ps // 2
        input_ids = self._make_seq(1, 1) + list(range(8000, 8000 + tail_extra))
        req = self._make_req(req_to_token_pool)
        req.origin_input_ids = input_ids
        req.output_ids = []
        kv_len = len(input_ids)
        kv_indices = self._alloc(allocator, kv_len)
        req_to_token_pool.write((req.req_pool_idx, slice(0, kv_len)), kv_indices)
        req.kv_committed_len = kv_len
        req.last_node = tree.root_node
        req.cache_protected_len = 0
        req.swa_uuid_for_lock = None
        req.extra_key = None
        req.fill_ids = input_ids
        if self.cfg.has_mamba:
            req.mamba_last_track_seqlen = kv_len

        avail_before = allocator.available_size()
        tree.cache_finished_req(req, is_insert=True)

        self.assertEqual(allocator.available_size(), avail_before + tail_extra)
        aligned = input_ids[: (len(input_ids) // ps) * ps]
        m = tree.match_prefix(MatchPrefixParams(key=RadixKey(aligned)))
        self.assertEqual(len(m.device_indices), len(aligned))
        tree.sanity_check()

    def test_mamba_evict_only(self):
        if not self.cfg.has_mamba:
            self.skipTest("requires Mamba component")
        tree, allocator, req_to_token_pool = build_fixture(self.cfg)
        seq_short = self._make_seq(1, 2)
        seq_long = seq_short + self._make_seq(500, 2)
        self._insert(tree, allocator, req_to_token_pool, seq_short)
        self._insert(tree, allocator, req_to_token_pool, seq_long)
        self.assertEqual(tree.mamba_evictable_size(), 2)

        result = tree.evict(EvictParams(num_tokens=0, mamba_num=1))
        self.assertGreaterEqual(result.mamba_num_evicted, 1)
        self.assertGreaterEqual(tree.full_evictable_size(), 0)
        tree.sanity_check()

    def test_mamba_evict_breaks_match(self):
        if not self.cfg.has_mamba:
            self.skipTest("requires Mamba component")
        tree, allocator, req_to_token_pool = build_fixture(self.cfg)
        seq_short = self._make_seq(1, 2)
        seq_long = seq_short + self._make_seq(500, 1)
        self._insert(tree, allocator, req_to_token_pool, seq_short)
        self._insert(tree, allocator, req_to_token_pool, seq_long)

        tree.evict(EvictParams(num_tokens=0, mamba_num=10))
        self.assertEqual(tree.mamba_evictable_size(), 0)

        m = tree.match_prefix(MatchPrefixParams(key=RadixKey(seq_long)))
        self.assertEqual(len(m.device_indices), 0)
        tree.sanity_check()

    def test_mamba_evict_result_accounting(self):
        if not self.cfg.has_mamba:
            self.skipTest("requires Mamba component")
        tree, allocator, req_to_token_pool = build_fixture(self.cfg)
        seq = self._make_seq(1, 3)
        self._insert(tree, allocator, req_to_token_pool, seq)

        result = tree.evict(EvictParams(num_tokens=len(seq)))
        self.assertGreaterEqual(result.num_tokens_evicted, len(seq))
        self.assertGreaterEqual(result.mamba_num_evicted, 1)
        tree.sanity_check()

    def test_mamba_evict_cascades_on_full_leaf(self):
        if not self.cfg.has_mamba:
            self.skipTest("requires Mamba component")
        tree, allocator, req_to_token_pool = build_fixture(self.cfg)
        seq = self._make_seq(1, 2)
        self._insert(tree, allocator, req_to_token_pool, seq)

        result = tree.evict(EvictParams(num_tokens=len(seq)))
        self.assertGreaterEqual(result.num_tokens_evicted, len(seq))
        self.assertGreaterEqual(result.mamba_num_evicted, 1)
        tree.sanity_check()

    def test_mamba_cow_on_match(self):
        if not self.cfg.has_mamba:
            self.skipTest("requires Mamba component")
        tree, allocator, req_to_token_pool = build_fixture(self.cfg)
        mamba_pool = req_to_token_pool.mamba_pool

        seq = self._make_seq(1, 2)
        self._insert(tree, allocator, req_to_token_pool, seq)

        req2 = self._make_req(req_to_token_pool)
        m = tree.match_prefix(
            MatchPrefixParams(key=RadixKey(seq), cow_mamba=True, req=req2)
        )
        self.assertEqual(len(m.device_indices), len(seq))
        self.assertIsNotNone(req2.mamba_pool_idx)

        src_value = m.last_device_node.component_data[ComponentType.MAMBA].value
        self.assertTrue(
            torch.all(
                mamba_pool.mamba_cache.conv[0][:, req2.mamba_pool_idx]
                == mamba_pool.mamba_cache.conv[0][:, src_value]
            )
        )
        tree.sanity_check()

    def test_swa_insert_and_match(self):
        if not self.cfg.has_swa:
            self.skipTest("requires SWA component")
        tree, allocator, req_to_token_pool = build_fixture(self.cfg)
        seq = self._make_seq(1, 3)
        self._insert(tree, allocator, req_to_token_pool, seq)

        m = tree.match_prefix(MatchPrefixParams(key=RadixKey(seq)))
        self.assertEqual(len(m.device_indices), len(seq))
        tree.sanity_check()

    def test_swa_evict_cascades(self):
        """Evict SWA tokens via swa_num_tokens — cascades to lower-priority components."""
        if not self.cfg.has_swa:
            self.skipTest("requires SWA component")
        tree, allocator, req_to_token_pool = build_fixture(self.cfg)
        seq_short = self._make_seq(1, 2)
        seq_long = seq_short + self._make_seq(500, 2)
        self._insert(tree, allocator, req_to_token_pool, seq_short)
        self._insert(tree, allocator, req_to_token_pool, seq_long)

        result = tree.evict(EvictParams(num_tokens=0, swa_num_tokens=len(seq_short)))
        self.assertGreater(result.swa_num_tokens_evicted, 0)
        tree.sanity_check()

    def test_swa_evict_cascades_mamba(self):
        """SWA eviction on an internal node cascades to Mamba."""
        if not self.cfg.has_swa or not self.cfg.has_mamba:
            self.skipTest("requires SWA and Mamba components")
        tree, allocator, req_to_token_pool = build_fixture(self.cfg)
        seq_short = self._make_seq(1, 3)
        seq_long = seq_short + self._make_seq(500, 4)
        self._insert(tree, allocator, req_to_token_pool, seq_short)
        self._insert(tree, allocator, req_to_token_pool, seq_long)

        result = tree.evict(EvictParams(num_tokens=0, swa_num_tokens=len(seq_short)))
        self.assertGreaterEqual(result.swa_num_tokens_evicted, 0)
        tree.sanity_check()

    def test_swa_evict_full_leaf_cascades_all(self):
        if not self.cfg.has_swa:
            self.skipTest("requires SWA component")
        tree, allocator, req_to_token_pool = build_fixture(self.cfg)
        seq_a = self._make_seq(1, 2)
        seq_b = self._make_seq(500, 2)
        self._insert(tree, allocator, req_to_token_pool, seq_a)
        self._insert(tree, allocator, req_to_token_pool, seq_b)

        result = tree.evict(EvictParams(num_tokens=len(seq_a)))
        self.assertGreaterEqual(result.num_tokens_evicted, len(seq_a))
        self.assertGreater(result.swa_num_tokens_evicted, 0)
        if self.cfg.has_mamba:
            self.assertGreaterEqual(result.mamba_num_evicted, 1)
        tree.sanity_check()

    def test_swa_lock_protects_from_eviction(self):
        if not self.cfg.has_swa:
            self.skipTest("requires SWA component")
        tree, allocator, req_to_token_pool = build_fixture(self.cfg)
        seq_a = self._make_seq(1, 2)
        seq_b = self._make_seq(500, 2)
        self._insert(tree, allocator, req_to_token_pool, seq_a)
        self._insert(tree, allocator, req_to_token_pool, seq_b)

        m = tree.match_prefix(MatchPrefixParams(key=RadixKey(seq_a)))
        lock_result = tree.inc_lock_ref(m.last_device_node)

        result = tree.evict(EvictParams(num_tokens=len(seq_a) + len(seq_b)))
        self.assertGreaterEqual(result.num_tokens_evicted, len(seq_b))

        m = tree.match_prefix(MatchPrefixParams(key=RadixKey(seq_a)))
        self.assertEqual(len(m.device_indices), len(seq_a))

        tree.dec_lock_ref(
            m.last_device_node,
            DecLockRefParams(swa_uuid_for_lock=lock_result.swa_uuid_for_lock),
        )
        tree.sanity_check()

    def test_internal_readonly_does_not_modify_tree(self):
        """Verify readonly match does not modify tree structure (no split)."""
        if self.cfg.page_size > 1 or self.cfg.has_mamba or self.cfg.has_swa:
            self.skipTest("Full-only page_size=1 only")
        tree, allocator, req_to_token_pool = build_fixture(self.cfg)

        self._insert(tree, allocator, req_to_token_pool, [1, 2, 3, 4, 5])

        def count_nodes(node):
            count = 1
            for child in node.children.values():
                count += count_nodes(child)
            return count

        node_count_before = count_nodes(tree.root_node)
        self.assertEqual(node_count_before, 2)

        tree._match_prefix_helper(RadixKey([1, 2]))
        value, best_node, best_value_len = tree._match_prefix_helper(
            RadixKey([1, 2, 3, 4])
        )
        self.assertEqual(best_value_len, 2)
        self.assertEqual(best_node.key.token_ids, [3, 4])
        node_count_after_regular = count_nodes(tree.root_node)
        self.assertEqual(node_count_after_regular, node_count_before + 2)

        value, best_node, best_value_len = tree._match_prefix_helper_readonly(
            RadixKey([1, 2, 3])
        )
        self.assertEqual(best_value_len, 1)
        self.assertEqual(best_node.key.token_ids, [1, 2])
        node_count_after_readonly = count_nodes(tree.root_node)
        self.assertEqual(node_count_after_readonly, node_count_after_regular)

        tree.sanity_check()


_CONFIGS: list[CacheConfig] = [
    CacheConfig(page_size=1, components=(ComponentType.FULL,)),
    CacheConfig(page_size=4, components=(ComponentType.FULL,)),
    CacheConfig(page_size=16, components=(ComponentType.FULL,)),
    CacheConfig(
        page_size=64,
        components=(ComponentType.FULL,),
        kv_size=1024,
        max_context_len=1024,
    ),
    CacheConfig(
        page_size=128,
        components=(ComponentType.FULL,),
        kv_size=2048,
        max_context_len=2048,
    ),
    CacheConfig(
        page_size=1,
        components=(ComponentType.FULL, ComponentType.MAMBA),
    ),
    CacheConfig(
        page_size=4,
        components=(ComponentType.FULL, ComponentType.MAMBA),
        enable_mamba_extra_buffer=True,  # Mamba page_size > 1 requires enable_mamba_extra_buffer=True
        mamba_cache_size=60,
    ),
    CacheConfig(
        page_size=64,
        components=(ComponentType.FULL, ComponentType.MAMBA),
        enable_mamba_extra_buffer=True,
        mamba_cache_size=60,
        kv_size=1024,
        max_context_len=1024,
    ),
    CacheConfig(
        page_size=128,
        components=(ComponentType.FULL, ComponentType.MAMBA),
        enable_mamba_extra_buffer=True,
        mamba_cache_size=60,
        kv_size=2048,
        max_context_len=2048,
    ),
    CacheConfig(
        page_size=1,
        components=(ComponentType.FULL, ComponentType.SWA),
        sliding_window_size=4,
    ),
    CacheConfig(
        page_size=4,
        components=(ComponentType.FULL, ComponentType.SWA),
        sliding_window_size=4,
    ),
    CacheConfig(
        page_size=4,
        components=(ComponentType.FULL, ComponentType.SWA),
        sliding_window_size=2,  # window < page_size edge case
    ),
    CacheConfig(
        page_size=16,
        components=(ComponentType.FULL, ComponentType.SWA),
        sliding_window_size=16,
        kv_size=512,
    ),
    CacheConfig(
        page_size=64,
        components=(ComponentType.FULL, ComponentType.SWA),
        sliding_window_size=64,
        kv_size=4096,
        max_context_len=4096,
    ),
    CacheConfig(
        page_size=128,
        components=(ComponentType.FULL, ComponentType.SWA),
        sliding_window_size=128,
        kv_size=8192,
        max_context_len=8192,
    ),
    CacheConfig(
        page_size=128,
        components=(ComponentType.FULL, ComponentType.SWA),
        sliding_window_size=4,
        kv_size=8192,
        max_context_len=8192,
    ),
    CacheConfig(
        page_size=1,
        components=(ComponentType.FULL, ComponentType.SWA),
        sliding_window_size=128,
        kv_size=1024,
        max_context_len=1024,
    ),
    CacheConfig(
        page_size=1,
        components=(ComponentType.FULL, ComponentType.SWA, ComponentType.MAMBA),
        sliding_window_size=128,
        head_num=8,
        num_layers=32,
        full_attention_layer_ids=(7, 15, 23, 31),
        kv_size=1024,
        max_context_len=1024,
    ),
    CacheConfig(
        page_size=64,
        components=(ComponentType.FULL, ComponentType.SWA, ComponentType.MAMBA),
        sliding_window_size=64,
        enable_mamba_extra_buffer=True,
        mamba_cache_size=60,
        kv_size=4096,
        max_context_len=4096,
    ),
]


for _cfg in _CONFIGS:
    _name = f"Test_{_cfg.label}"
    globals()[_name] = type(
        _name, (UnifiedRadixCacheSuite, CustomTestCase), {"cfg": _cfg}
    )
    globals()[_name].__module__ = __name__
del _cfg, _name


if __name__ == "__main__":
    unittest.main()
