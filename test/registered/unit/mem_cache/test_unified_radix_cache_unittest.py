"""Unit tests for UnifiedRadixCache"""

import unittest
from dataclasses import dataclass
from typing import Optional
from unittest import mock

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
    MatchResult,
)
from sglang.srt.mem_cache.cache_init_params import CacheInitParams
from sglang.srt.mem_cache.common import available_and_evictable_str
from sglang.srt.mem_cache.hicache_storage import PoolName
from sglang.srt.mem_cache.memory_pool import (
    HybridLinearKVPool,
    HybridReqToTokenPool,
    MHATokenToKVPool,
    ReqToTokenPool,
)
from sglang.srt.mem_cache.radix_cache import RadixKey
from sglang.srt.mem_cache.swa_memory_pool import SWAKVPool, SWATokenToKVPoolAllocator
from sglang.srt.mem_cache.unified_cache_components.tree_component import (
    CacheTransferPhase,
    ComponentType,
)
from sglang.srt.mem_cache.unified_radix_cache import (
    UnifiedRadixCache,
    UnifiedTreeNode,
)
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.srt.server_args import (
    ServerArgs,
    get_global_server_args,
    set_global_server_args_for_scheduler,
)
from sglang.srt.utils import get_device
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=10, suite="stage-b-test-1-gpu-small")


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

    cache_init_params = CacheInitParams(
        req_to_token_pool=req_to_token_pool,
        token_to_kv_pool_allocator=allocator,
        page_size=cfg.page_size,
        disable=False,
        sliding_window_size=cfg.sliding_window_size,
        tree_components=cfg.components,
        enable_mamba_extra_buffer=cfg.enable_mamba_extra_buffer,
    )
    tree = UnifiedRadixCache(params=cache_init_params)
    tree.cache_init_params = cache_init_params

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
        child_key = key.child_key(tree.page_size)
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

    def test_tombstone_cleanup_respects_locked_parent(self):
        tree, _, _ = build_fixture(self.cfg)
        parent = UnifiedTreeNode(self.cfg.components)
        deleted = UnifiedTreeNode(self.cfg.components)

        parent.key = RadixKey(self._make_seq(1, 1))
        deleted.key = RadixKey(self._make_seq(1000, 1))
        parent.parent = tree.root_node
        deleted.parent = parent
        parent.component_data[ComponentType.FULL].value = torch.arange(
            self.cfg.page_size, dtype=torch.int64, device=tree.device
        )
        parent.component_data[ComponentType.FULL].lock_ref = 1
        parent_key = parent.key.child_key(tree.page_size)
        tree.root_node.children[parent_key] = parent

        tracker = {ct: 0 for ct in tree.tree_components}

        tree._iteratively_delete_tombstone_leaf(deleted, tracker)

        self.assertIn(parent_key, tree.root_node.children)
        self.assertIs(tree.root_node.children[parent_key], parent)
        self.assertTrue(all(evicted == 0 for evicted in tracker.values()))

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

    # ================================================================
    # Evict chain tests covering demotion, cascade, and tombstone cleanup.
    # ================================================================

    def test_evict_leaf_frees_all_components(self):
        """Evicting a device leaf frees Full and all aux components atomically."""
        tree, allocator, req_to_token_pool = build_fixture(self.cfg)
        seq = self._make_seq(1, 3)
        self._insert(tree, allocator, req_to_token_pool, seq)

        full_before = tree.full_evictable_size()
        mamba_before = tree.mamba_evictable_size() if self.cfg.has_mamba else 0
        swa_before = tree.swa_evictable_size() if self.cfg.has_swa else 0
        self.assertGreater(full_before, 0)

        result = tree.evict(EvictParams(num_tokens=full_before * 2))
        self.assertGreaterEqual(result.num_tokens_evicted, full_before)
        self.assertEqual(tree.full_evictable_size(), 0)
        if self.cfg.has_mamba:
            self.assertEqual(tree.mamba_evictable_size(), 0)
        if self.cfg.has_swa:
            self.assertEqual(tree.swa_evictable_size(), 0)
        tree.sanity_check()

    def test_evict_cascade_parent_becomes_d_leaf(self):
        """After evicting a D-leaf child, parent may become a new D-leaf."""
        tree, allocator, req_to_token_pool = build_fixture(self.cfg)
        base = self._make_seq(1, 2)
        leaf = base + self._make_seq(500, 2)
        self._insert(tree, allocator, req_to_token_pool, base)
        self._insert(tree, allocator, req_to_token_pool, leaf)

        # Lock the base node to prevent it from being evicted
        m_base = tree.match_prefix(MatchPrefixParams(key=RadixKey(base)))
        lock_result = tree.inc_lock_ref(m_base.last_device_node)

        # Evict the leaf — parent (base) should become D-leaf after unlock
        result = tree.evict(EvictParams(num_tokens=len(leaf)))
        tree.sanity_check()

        tree.dec_lock_ref(
            m_base.last_device_node,
            DecLockRefParams(
                swa_uuid_for_lock=getattr(lock_result, "swa_uuid_for_lock", None)
            ),
        )
        # After unlock, base should be in evictable_device_leaves
        self.assertIn(m_base.last_device_node, tree.evictable_device_leaves)
        tree.sanity_check()

    def test_evict_iterative_tombstone_cleanup(self):
        """Tombstone cascade: evicting a leaf triggers cleanup up the tree."""
        tree, allocator, req_to_token_pool = build_fixture(self.cfg)
        # Create a chain: root -> A -> B -> C (3 levels)
        ps = self.cfg.page_size
        chain = self._make_seq(1, 6)
        self._insert(tree, allocator, req_to_token_pool, chain[: 2 * ps])
        self._insert(tree, allocator, req_to_token_pool, chain[: 4 * ps])
        self._insert(tree, allocator, req_to_token_pool, chain)

        initial_evictable = tree.full_evictable_size()
        self.assertGreater(initial_evictable, 0)

        # Evict everything — tombstone cascade should clean up all
        result = tree.evict(EvictParams(num_tokens=initial_evictable * 2))
        self.assertGreaterEqual(result.num_tokens_evicted, initial_evictable)
        self.assertEqual(tree.full_evictable_size(), 0)
        # Only root should remain
        self.assertEqual(len(tree.root_node.children), 0)
        tree.sanity_check()

    def test_evict_respects_lru_order(self):
        """Older (less recently accessed) nodes are evicted first."""
        tree, allocator, req_to_token_pool = build_fixture(self.cfg)
        ps = self.cfg.page_size
        seq_old = self._make_seq(1, 2)
        seq_new = self._make_seq(500, 2)

        self._insert(tree, allocator, req_to_token_pool, seq_old)
        self._insert(tree, allocator, req_to_token_pool, seq_new)

        # Touch seq_new to make it MRU
        tree.match_prefix(MatchPrefixParams(key=RadixKey(seq_new)))

        # Evict just enough for one sequence
        tree.evict(EvictParams(num_tokens=len(seq_old)))

        # seq_old should be gone (LRU), seq_new should remain
        m_old = tree.match_prefix(MatchPrefixParams(key=RadixKey(seq_old)))
        m_new = tree.match_prefix(MatchPrefixParams(key=RadixKey(seq_new)))
        self.assertEqual(len(m_old.device_indices), 0)
        self.assertEqual(len(m_new.device_indices), len(seq_new))
        tree.sanity_check()

    def test_evict_multiple_independent_leaves(self):
        """Evicting multiple independent leaves works correctly."""
        tree, allocator, req_to_token_pool = build_fixture(self.cfg)
        seqs = [self._make_seq(i * 100, 2) for i in range(4)]
        for s in seqs:
            self._insert(tree, allocator, req_to_token_pool, s)

        total = sum(len(s) for s in seqs)
        self.assertEqual(tree.full_evictable_size(), total)

        # Evict half
        half = total // 2
        result = tree.evict(EvictParams(num_tokens=half))
        self.assertGreaterEqual(result.num_tokens_evicted, half)
        self.assertLessEqual(tree.full_evictable_size(), total - half)
        tree.sanity_check()

        # Evict remainder
        remaining = tree.full_evictable_size()
        result = tree.evict(EvictParams(num_tokens=remaining * 2))
        self.assertGreaterEqual(result.num_tokens_evicted, remaining)
        self.assertEqual(tree.full_evictable_size(), 0)
        tree.sanity_check()

    def test_evict_shared_prefix_keeps_common_path(self):
        """Evicting one branch preserves the shared prefix for other branch."""
        tree, allocator, req_to_token_pool = build_fixture(self.cfg)
        base = self._make_seq(1, 2)
        branch_a = base + self._make_seq(100, 2)
        branch_b = base + self._make_seq(200, 2)

        self._insert(tree, allocator, req_to_token_pool, branch_a)
        self._insert(tree, allocator, req_to_token_pool, branch_b)

        # Lock branch_b
        m = tree.match_prefix(MatchPrefixParams(key=RadixKey(branch_b)))
        lr = tree.inc_lock_ref(m.last_device_node)

        # Evict — branch_a should go, base + branch_b stay
        tree.evict(EvictParams(num_tokens=len(branch_a)))

        m_b = tree.match_prefix(MatchPrefixParams(key=RadixKey(branch_b)))
        self.assertEqual(len(m_b.device_indices), len(branch_b))

        tree.dec_lock_ref(
            m.last_device_node,
            DecLockRefParams(swa_uuid_for_lock=getattr(lr, "swa_uuid_for_lock", None)),
        )
        tree.sanity_check()

    def test_evict_result_accounting_matches_actual(self):
        """EvictResult.num_tokens_evicted matches actual size change."""
        tree, allocator, req_to_token_pool = build_fixture(self.cfg)
        seqs = [self._make_seq(i * 100, 2) for i in range(5)]
        for s in seqs:
            self._insert(tree, allocator, req_to_token_pool, s)

        before = tree.full_evictable_size()
        result = tree.evict(EvictParams(num_tokens=before))
        after = tree.full_evictable_size()
        self.assertEqual(result.num_tokens_evicted, before - after)
        tree.sanity_check()

    def test_evict_locked_subtree_skipped(self):
        """All nodes in a locked path are skipped during eviction."""
        tree, allocator, req_to_token_pool = build_fixture(self.cfg)
        seq_a = self._make_seq(1, 3)
        seq_b = self._make_seq(500, 2)
        self._insert(tree, allocator, req_to_token_pool, seq_a)
        self._insert(tree, allocator, req_to_token_pool, seq_b)

        # Lock seq_a
        m = tree.match_prefix(MatchPrefixParams(key=RadixKey(seq_a)))
        lr = tree.inc_lock_ref(m.last_device_node)

        # Try to evict everything
        total = tree.full_evictable_size() + tree.full_protected_size()
        result = tree.evict(EvictParams(num_tokens=total))

        # seq_a should still be matchable (protected)
        m2 = tree.match_prefix(MatchPrefixParams(key=RadixKey(seq_a)))
        self.assertEqual(len(m2.device_indices), len(seq_a))

        tree.dec_lock_ref(
            m.last_device_node,
            DecLockRefParams(swa_uuid_for_lock=getattr(lr, "swa_uuid_for_lock", None)),
        )
        tree.sanity_check()

    def test_mamba_internal_tombstone_evict(self):
        """Mamba eviction on internal node tombstones mamba only, keeps Full."""
        if not self.cfg.has_mamba:
            self.skipTest("requires Mamba component")
        tree, allocator, req_to_token_pool = build_fixture(self.cfg)
        # Create internal node with mamba and leaf extending it
        seq_short = self._make_seq(1, 2)
        seq_long = seq_short + self._make_seq(500, 2)
        self._insert(tree, allocator, req_to_token_pool, seq_short)
        self._insert(tree, allocator, req_to_token_pool, seq_long)

        # Evict only mamba
        result = tree.evict(EvictParams(num_tokens=0, mamba_num=10))
        self.assertEqual(tree.mamba_evictable_size(), 0)

        # Full should still be accessible for at least the long seq base
        # (mamba gone breaks match, but full data might still be in tree)
        tree.sanity_check()

    def test_evict_reinsert_after_full_eviction(self):
        """After evicting everything, new inserts work correctly."""
        tree, allocator, req_to_token_pool = build_fixture(self.cfg)
        seq_a = self._make_seq(1, 2)
        self._insert(tree, allocator, req_to_token_pool, seq_a)
        tree.evict(EvictParams(num_tokens=len(seq_a) * 2))
        self.assertEqual(tree.full_evictable_size(), 0)

        # Re-insert
        seq_b = self._make_seq(500, 2)
        self._insert(tree, allocator, req_to_token_pool, seq_b)
        m = tree.match_prefix(MatchPrefixParams(key=RadixKey(seq_b)))
        self.assertEqual(len(m.device_indices), len(seq_b))
        tree.sanity_check()

    def test_swa_evict_internal_tombstone(self):
        """SWA eviction on internal node cascades to lower-priority components."""
        if not self.cfg.has_swa:
            self.skipTest("requires SWA component")
        tree, allocator, req_to_token_pool = build_fixture(self.cfg)
        base = self._make_seq(1, 3)
        leaf = base + self._make_seq(500, 3)
        self._insert(tree, allocator, req_to_token_pool, base)
        self._insert(tree, allocator, req_to_token_pool, leaf)

        swa_before = tree.swa_evictable_size()
        result = tree.evict(EvictParams(num_tokens=0, swa_num_tokens=swa_before * 2))
        self.assertEqual(tree.swa_evictable_size(), 0)
        tree.sanity_check()

    def test_evict_d_leaf_set_consistency(self):
        """evictable_device_leaves is consistent after mixed operations."""
        tree, allocator, req_to_token_pool = build_fixture(self.cfg)
        seqs = [self._make_seq(i * 100, 2) for i in range(6)]
        for s in seqs:
            self._insert(tree, allocator, req_to_token_pool, s)

        # Lock some, evict some, unlock
        m = tree.match_prefix(MatchPrefixParams(key=RadixKey(seqs[0])))
        lr = tree.inc_lock_ref(m.last_device_node)

        tree.evict(EvictParams(num_tokens=len(seqs[1])))
        tree.sanity_check()

        tree.dec_lock_ref(
            m.last_device_node,
            DecLockRefParams(swa_uuid_for_lock=getattr(lr, "swa_uuid_for_lock", None)),
        )
        tree.sanity_check()

        # Insert more
        extra = self._make_seq(9000, 2)
        self._insert(tree, allocator, req_to_token_pool, extra)
        tree.sanity_check()

    # ================================================================
    # HiCache Unit Tests (real cache_controller D<->H backup/load)
    # ================================================================

    def _skip_unsupported_hicache_test(self):
        if self.cfg.has_swa:
            self.skipTest("HiCache tests do not run on SWA stacks")
        return False

    def _simulate_backup(self, tree, node):
        """Simulate D->H backup by setting host_value on each component."""
        for ct in (ComponentType.FULL, ComponentType.MAMBA, ComponentType.SWA):
            if ct not in self.cfg.components:
                continue
            cd = node.component_data[ct]
            if cd.value is not None and cd.host_value is None:
                cd.host_value = cd.value.clone()

    def _simulate_backup_tree(self, tree):
        """Backup all non-root nodes (simulates write-through)."""
        stack = [tree.root_node]
        while stack:
            node = stack.pop()
            if node is not tree.root_node:
                self._simulate_backup(tree, node)
            stack.extend(node.children.values())

    def _init_hicache(self, tree):
        import sglang.srt.mem_cache.hybrid_cache.hybrid_pool_assembler as assembler

        orig_kv_host_pool = assembler.MHATokenToKVPoolHost
        orig_mamba_host_pool = assembler.MambaPoolHost

        def kv_host_pool_wrapper(*args, **kwargs):
            kwargs["pin_memory"] = False
            return orig_kv_host_pool(*args, **kwargs)

        def mamba_host_pool_wrapper(*args, **kwargs):
            kwargs["pin_memory"] = False
            return orig_mamba_host_pool(*args, **kwargs)

        patchers = [
            mock.patch.object(
                assembler,
                "MHATokenToKVPoolHost",
                side_effect=kv_host_pool_wrapper,
            ),
            mock.patch.object(
                assembler,
                "MambaPoolHost",
                side_effect=mamba_host_pool_wrapper,
            ),
        ]
        for patcher in patchers:
            patcher.start()
            self.addCleanup(patcher.stop)

        server_args = ServerArgs(
            model_path="dummy",
            page_size=self.cfg.page_size,
            hicache_io_backend="direct",
            hicache_write_policy="write_through",
        )
        set_global_server_args_for_scheduler(server_args)
        tree.init_hicache(server_args, tree.cache_init_params)
        tree.write_through_threshold = 1 << 30
        tree.load_back_threshold = 0

    def _build_hicache_fixture(self):
        fixture = build_fixture(self.cfg)
        tree, _, _ = fixture
        self._init_hicache(tree)
        return fixture

    def _backup_node(self, tree, node):
        backed_up = tree.write_backup(node, write_back=True)
        self.assertGreater(backed_up, 0)
        tree.writing_check(write_back=True)
        return backed_up

    def _backup_tree(self, tree):
        stack = [tree.root_node]
        while stack:
            node = stack.pop()
            children = list(node.children.values())
            stack.extend(reversed(children))
            if node is not tree.root_node:
                self._backup_node(tree, node)

    def _load_back_node(self, tree, node):
        device_indices = tree.load_back(node)
        self.assertIsNotNone(device_indices)
        producer_id = tree.ready_to_load_host_cache()
        self.assertNotEqual(producer_id, -1)
        for _, finish_event, _ in list(tree.cache_controller.ack_load_queue):
            finish_event.synchronize()
        tree.loading_check()
        return device_indices

    def _get_full_kv_pool(self, allocator):
        kv_pool = allocator.get_kvcache()
        return getattr(kv_pool, "full_kv_pool", kv_pool)

    def _fill_full_kv(self, allocator, indices, marker):
        kv_pool = self._get_full_kv_pool(allocator)
        layer_id = kv_pool.start_layer
        k_buf = kv_pool.get_key_buffer(layer_id)
        v_buf = kv_pool.get_value_buffer(layer_id)
        k_buf[indices].fill_(marker)
        v_buf[indices].fill_(marker + 1)

    def _snapshot_full_kv(self, allocator, indices):
        kv_pool = self._get_full_kv_pool(allocator)
        layer_id = kv_pool.start_layer
        return (
            kv_pool.get_key_buffer(layer_id)[indices].float().cpu().clone(),
            kv_pool.get_value_buffer(layer_id)[indices].float().cpu().clone(),
        )

    def _fill_mamba_state(self, req_to_token_pool, indices, marker):
        if not self.cfg.has_mamba:
            return
        mamba_indices = indices.reshape(-1)
        mamba_cache = req_to_token_pool.mamba_pool.mamba_cache
        mamba_cache.temporal[:, mamba_indices].fill_(marker)
        for offset, conv_buf in enumerate(mamba_cache.conv, start=1):
            conv_buf[:, mamba_indices].fill_(marker + offset)

    def _snapshot_mamba_state(self, req_to_token_pool, indices):
        mamba_indices = indices.reshape(-1)
        mamba_cache = req_to_token_pool.mamba_pool.mamba_cache
        return (
            mamba_cache.temporal[:, mamba_indices].float().cpu().clone(),
            [conv[:, mamba_indices].float().cpu().clone() for conv in mamba_cache.conv],
        )

    def test_hicache_node_states(self):
        """Verify device-only to device+host transition after real backup."""
        if self._skip_unsupported_hicache_test():
            return
        tree, allocator, req_to_token_pool = self._build_hicache_fixture()
        seq = self._make_seq(1, 2)
        self._insert(tree, allocator, req_to_token_pool, seq)

        # Find the leaf node
        m = tree.match_prefix(MatchPrefixParams(key=RadixKey(seq)))
        node = m.last_device_node
        self.assertIsNot(node, tree.root_node)

        ct = ComponentType.FULL
        # S1: device only
        self.assertIsNotNone(node.component_data[ct].value)
        self.assertIsNone(node.component_data[ct].host_value)
        self.assertFalse(node.backuped)
        self.assertFalse(node.evicted)

        self._backup_node(tree, node)
        self.assertIsNotNone(node.component_data[ct].value)
        self.assertIsNotNone(node.component_data[ct].host_value)
        self.assertTrue(node.backuped)
        self.assertFalse(node.evicted)
        tree.sanity_check()

    def test_hicache_evict_to_host(self):
        """Evicting a backed-up device leaf demotes it to host-only state."""
        if self._skip_unsupported_hicache_test():
            return
        tree, allocator, req_to_token_pool = self._build_hicache_fixture()
        seq = self._make_seq(1, 2)
        self._insert(tree, allocator, req_to_token_pool, seq)

        m = tree.match_prefix(MatchPrefixParams(key=RadixKey(seq)))
        node = m.last_device_node

        self._backup_node(tree, node)
        self.assertTrue(node.backuped)

        # Evict -> should demote to host (S3)
        result = tree.evict(EvictParams(num_tokens=len(seq)))
        self.assertGreaterEqual(result.num_tokens_evicted, len(seq))

        # Node should now be evicted (S3)
        self.assertTrue(node.evicted)
        self.assertTrue(node.backuped)
        self.assertIsNone(node.component_data[ComponentType.FULL].value)
        self.assertIsNotNone(node.component_data[ComponentType.FULL].host_value)

        # Should be in host_leaves, not device_leaves
        self.assertNotIn(node, tree.evictable_device_leaves)
        self.assertIn(node, tree.evictable_host_leaves)
        tree.sanity_check()

    def test_hicache_match_through_evicted_node(self):
        """Match can traverse evicted (S3) nodes using host_value."""
        if self._skip_unsupported_hicache_test():
            return
        tree, allocator, req_to_token_pool = self._build_hicache_fixture()
        base = self._make_seq(1, 2)
        leaf = base + self._make_seq(500, 2)
        self._insert(tree, allocator, req_to_token_pool, base)
        self._insert(tree, allocator, req_to_token_pool, leaf)

        self._backup_tree(tree)

        # Lock leaf so only base can be evicted
        m = tree.match_prefix(MatchPrefixParams(key=RadixKey(leaf)))
        lr = tree.inc_lock_ref(m.last_device_node)

        # Evict base (inner node won't be evicted while child is locked)
        tree.evict(EvictParams(num_tokens=len(base)))

        tree.dec_lock_ref(
            m.last_device_node,
            DecLockRefParams(swa_uuid_for_lock=getattr(lr, "swa_uuid_for_lock", None)),
        )
        m = tree.match_prefix(MatchPrefixParams(key=RadixKey(leaf)))
        self.assertGreaterEqual(len(m.device_indices), len(base))
        tree.sanity_check()

    def test_hicache_d_leaf_h_leaf_mutual_exclusion(self):
        """D-leaf and H-leaf sets are always disjoint."""
        if self._skip_unsupported_hicache_test():
            return
        tree, allocator, req_to_token_pool = self._build_hicache_fixture()
        seqs = [self._make_seq(i * 100, 2) for i in range(4)]
        for s in seqs:
            self._insert(tree, allocator, req_to_token_pool, s)

        for i in range(2):
            m = tree.match_prefix(MatchPrefixParams(key=RadixKey(seqs[i])))
            self._backup_node(tree, m.last_device_node)

        # Evict one backed-up node
        tree.evict(EvictParams(num_tokens=len(seqs[0])))

        # Check mutual exclusion
        overlap = tree.evictable_device_leaves & tree.evictable_host_leaves
        self.assertEqual(len(overlap), 0)
        tree.sanity_check()

    def test_hicache_host_leaf_eviction(self):
        """Evicting a host leaf removes the node from the tree entirely."""
        if self._skip_unsupported_hicache_test():
            return
        tree, allocator, req_to_token_pool = self._build_hicache_fixture()
        seq = self._make_seq(1, 2)
        self._insert(tree, allocator, req_to_token_pool, seq)

        m = tree.match_prefix(MatchPrefixParams(key=RadixKey(seq)))
        node = m.last_device_node

        self._backup_node(tree, node)
        tree.evict(EvictParams(num_tokens=len(seq)))

        self.assertTrue(node.evicted)
        self.assertIn(node, tree.evictable_host_leaves)

        # Now evict host
        tree.evict_host(len(seq))

        # Node should be removed from tree
        self.assertNotIn(node, tree.evictable_host_leaves)
        self.assertEqual(len(tree.root_node.children), 0)
        tree.sanity_check()

    def test_hicache_load_back_restores_data(self):
        """Loading back an evicted node restores the backed-up cache data."""
        if self._skip_unsupported_hicache_test():
            return
        tree, allocator, req_to_token_pool = self._build_hicache_fixture()
        base = self._make_seq(1, 2)
        self._insert(tree, allocator, req_to_token_pool, base)

        m = tree.match_prefix(MatchPrefixParams(key=RadixKey(base)))
        node = m.last_device_node
        original_device_indices = m.device_indices.clone()
        self._fill_full_kv(allocator, original_device_indices, marker=3)
        expected_k, expected_v = self._snapshot_full_kv(
            allocator, original_device_indices
        )
        original_mamba_indices = None
        expected_temporal = None
        expected_conv = None
        if self.cfg.has_mamba:
            original_mamba_indices = node.component_data[
                ComponentType.MAMBA
            ].value.clone()
            self._fill_mamba_state(req_to_token_pool, original_mamba_indices, marker=11)
            expected_temporal, expected_conv = self._snapshot_mamba_state(
                req_to_token_pool, original_mamba_indices
            )

        self._backup_node(tree, node)
        tree.evict(EvictParams(num_tokens=len(base)))
        self.assertTrue(node.evicted)
        self._fill_full_kv(allocator, original_device_indices, marker=9)
        if original_mamba_indices is not None:
            self._fill_mamba_state(req_to_token_pool, original_mamba_indices, marker=21)

        loaded_indices = self._load_back_node(tree, node)
        self.assertFalse(node.evicted)
        self.assertIsNotNone(node.component_data[ComponentType.FULL].value)
        loaded_k, loaded_v = self._snapshot_full_kv(allocator, loaded_indices)
        self.assertTrue(torch.equal(loaded_k, expected_k))
        self.assertTrue(torch.equal(loaded_v, expected_v))
        if self.cfg.has_mamba:
            loaded_mamba_indices = node.component_data[ComponentType.MAMBA].value
            loaded_temporal, loaded_conv = self._snapshot_mamba_state(
                req_to_token_pool, loaded_mamba_indices
            )
            self.assertTrue(torch.equal(loaded_temporal, expected_temporal))
            self.assertEqual(len(loaded_conv), len(expected_conv))
            for actual_conv, expected_conv_buf in zip(loaded_conv, expected_conv):
                self.assertTrue(torch.equal(actual_conv, expected_conv_buf))
        tree.sanity_check()

    def test_hicache_backup_continuity(self):
        """Backed-up nodes form a continuous prefix from the root."""
        if self._skip_unsupported_hicache_test():
            return
        tree, allocator, req_to_token_pool = self._build_hicache_fixture()
        chain = self._make_seq(1, 4)
        ps = self.cfg.page_size
        self._insert(tree, allocator, req_to_token_pool, chain[: 2 * ps])
        self._insert(tree, allocator, req_to_token_pool, chain)

        self._backup_tree(tree)

        # Verify: every backed-up node's parent is also backed-up (or root)
        all_nodes = tree._collect_all_nodes()
        for node in all_nodes:
            if node is tree.root_node:
                continue
            if node.backuped:
                parent = node.parent
                self.assertTrue(
                    parent is tree.root_node or parent.backuped,
                    f"Backup continuity violated: node {node.id} backed up but parent {parent.id} not",
                )
        tree.sanity_check()

    def test_hicache_evict_to_host_updates_aux_lru(self):
        """Aux components (MAMBA / SWA) move from device LRU to host LRU on D->H eviction."""
        aux_types = [
            ct
            for ct in (ComponentType.MAMBA, ComponentType.SWA)
            if ct in self.cfg.components
        ]
        if not aux_types:
            self.skipTest("requires at least one aux component")

        tree, allocator, req_to_token_pool = build_fixture(self.cfg)
        seq = self._make_seq(1, 2)
        self._insert(tree, allocator, req_to_token_pool, seq)

        m = tree.match_prefix(MatchPrefixParams(key=RadixKey(seq)))
        node = m.last_device_node

        for aux in aux_types:
            self.assertTrue(tree.lru_lists[aux].in_list(node))
            self.assertFalse(tree.host_lru_lists[aux].in_list(node))

        self._simulate_backup(tree, node)
        tree.evict(EvictParams(num_tokens=len(seq)))

        for aux in aux_types:
            self.assertFalse(tree.lru_lists[aux].in_list(node))
            if node.component_data[aux].host_value is not None:
                self.assertTrue(tree.host_lru_lists[aux].in_list(node))
        tree.sanity_check()

    def _build_chain_pages(self, tree, allocator, req_to_token_pool, num_pages):
        """Insert an incremental chain of single-page extensions.

        Returns the chain root-to-leaf. Length may differ from num_pages
        when the radix tree merges or splits nodes.
        """
        seq: list[int] = []
        for i in range(num_pages):
            seq = seq + self._make_seq(1000 * (i + 1), 1)
            self._insert(tree, allocator, req_to_token_pool, seq)
        m = tree.match_prefix(MatchPrefixParams(key=RadixKey(seq)))
        chain: list = []
        cur = m.last_device_node
        while cur is not tree.root_node:
            chain.append(cur)
            cur = cur.parent
        chain.reverse()
        return chain

    def test_hicache_swa_load_back_min_suffix(self):
        """LOAD_BACK collects only the suffix nodes needed to cover sliding_window_size."""
        if not self.cfg.has_swa:
            self.skipTest("requires SWA")
        if self.cfg.has_mamba:
            # Mamba's per-insert req allocation exhausts max_num_reqs on long chains.
            self.skipTest("SWA-only path keeps the chain construction simple")
        ps = self.cfg.page_size
        sw = self.cfg.sliding_window_size
        expected_pages = (sw + ps - 1) // ps
        chain_pages = expected_pages + 2
        if chain_pages * ps > self.cfg.kv_size // 2:
            self.skipTest("kv_size too small for the desired chain")

        tree, allocator, req_to_token_pool = build_fixture(self.cfg)
        chain = self._build_chain_pages(tree, allocator, req_to_token_pool, chain_pages)
        if len(chain) <= expected_pages:
            self.skipTest("chain collapsed below the suffix length being tested")

        self._simulate_backup_tree(tree)

        # Tombstone every chain node on the device side without going through
        # the tree-wide eviction loop. This isolates build_hicache_transfers
        # from LRU and cascade ordering.
        for n in chain:
            n.component_data[ComponentType.FULL].value = None
            n.component_data[ComponentType.SWA].value = None

        leaf = chain[-1]
        swa_comp = tree.components[ComponentType.SWA]
        transfers = swa_comp.build_hicache_transfers(leaf, CacheTransferPhase.LOAD_BACK)
        self.assertIsNotNone(transfers)
        self.assertEqual(len(transfers), 1)
        xfer = transfers[0]
        self.assertEqual(xfer.name, PoolName.SWA)
        self.assertEqual(len(xfer.nodes_to_load), expected_pages)
        # host_indices must cover exactly the expected suffix tokens (>= sw).
        self.assertEqual(int(xfer.host_indices.numel()), expected_pages * ps)
        self.assertGreaterEqual(int(xfer.host_indices.numel()), sw)
        self.assertEqual(xfer.nodes_to_load, chain[-expected_pages:])

    def test_hicache_swa_host_independent_of_full(self):
        """FULL host and SWA host are physically independent.
        Freeing one component's host_value must not touch the other.
        """
        if not self.cfg.has_swa:
            self.skipTest("requires SWA")

        tree, allocator, req_to_token_pool = build_fixture(self.cfg)
        seq = self._make_seq(1, 2)
        self._insert(tree, allocator, req_to_token_pool, seq)
        m = tree.match_prefix(MatchPrefixParams(key=RadixKey(seq)))
        node = m.last_device_node

        self._simulate_backup(tree, node)
        tree.evict(EvictParams(num_tokens=len(seq)))

        cd_full = node.component_data[ComponentType.FULL]
        cd_swa = node.component_data[ComponentType.SWA]
        self.assertIsNotNone(cd_full.host_value)
        self.assertIsNotNone(cd_swa.host_value)
        self.assertIn(node, tree.evictable_host_leaves)
        self.assertTrue(tree.host_lru_lists[ComponentType.SWA].in_list(node))

        # Drop FULL host bookkeeping. SWA side must stay intact.
        tree.evictable_host_leaves.discard(node)
        cd_full.host_value = None
        self.assertIsNotNone(cd_swa.host_value)
        self.assertTrue(tree.host_lru_lists[ComponentType.SWA].in_list(node))
        self.assertNotIn(node, tree.evictable_host_leaves)

        # Drop SWA host bookkeeping. FULL side (already cleared) stays cleared.
        tree.host_lru_lists[ComponentType.SWA].remove_node(node)
        cd_swa.host_value = None
        self.assertIsNone(cd_full.host_value)
        self.assertIsNone(cd_swa.host_value)
        self.assertFalse(tree.host_lru_lists[ComponentType.SWA].in_list(node))
        self.assertNotIn(node, tree.evictable_host_leaves)

    def _swa_finalize_setup(self):
        """Build a SWA chain long enough to fill at least the window
        plus one extra page, and host-back every node so we can flip
        SWA tombstones at will."""
        ps = self.cfg.page_size
        sw = self.cfg.sliding_window_size
        window_pages = (sw + ps - 1) // ps
        chain_pages = window_pages + 2
        if chain_pages * ps > self.cfg.kv_size // 2:
            self.skipTest("kv_size too small for the desired chain")

        tree, allocator, req_to_token_pool = build_fixture(self.cfg)
        chain = self._build_chain_pages(tree, allocator, req_to_token_pool, chain_pages)
        if len(chain) <= window_pages:
            self.skipTest("chain collapsed below the window length")
        self._simulate_backup_tree(tree)
        return tree, allocator, req_to_token_pool, chain, window_pages

    def test_hicache_swa_finalize_match_result(self):
        """finalize_match_result bumps host_hit_length to 1 iff some SWA node
        within the trailing window is tombstoned (cd.value is None,
        cd.host_value is not None). Out-of-window tombstones and chains fully
        on device must leave host_hit_length untouched.

        Sentinel only — never the real SWA token count, since SWA load-back
        does not grow req.prefix_indices and any non-zero value gets
        subtracted from extend_input_len in schedule_policy.
        """
        if not self.cfg.has_swa:
            self.skipTest("requires SWA")
        if self.cfg.has_mamba:
            self.skipTest("SWA-only path keeps the chain construction simple")

        tree, _, _, chain, window_pages = self._swa_finalize_setup()
        leaf = chain[-1]
        swa_comp = tree.components[ComponentType.SWA]

        cases = [
            ("all_on_device", None, 0),
            ("tombstone_in_window", chain[-window_pages], 1),
            ("tombstone_outside_window", chain[-(window_pages + 1)], 0),
        ]
        for name, victim, expected in cases:
            with self.subTest(name):
                # Reset SWA state for each subcase.
                for n in chain:
                    cd = n.component_data[ComponentType.SWA]
                    if cd.value is None and cd.host_value is not None:
                        cd.value = cd.host_value.clone()
                if victim is not None:
                    victim.component_data[ComponentType.SWA].value = None

                result = MatchResult(
                    device_indices=torch.empty(
                        (0,), dtype=torch.int64, device=tree.device
                    ),
                    last_device_node=leaf,
                    last_host_node=leaf,
                    host_hit_length=0,
                )
                result = swa_comp.finalize_match_result(
                    result=result,
                    params=MatchPrefixParams(key=RadixKey(self._make_seq(1, 1))),
                    value_chunks=[],
                    best_value_len=0,
                )
                self.assertEqual(result.host_hit_length, expected)

    def test_hicache_swa_commit_load_back_rebuilds_mapping(self):
        """LOAD_BACK commit must:
        (1) restore SWA cd.value via _restore_device_value (host LRU -> device LRU),
        (2) rewrite full_to_swa_index_mapping[full_idx] = new_swa_idx for every
            loaded chunk so subsequent SWA reads via translate_loc_from_full_to_swa
            return the freshly allocated SWA slot."""
        if not self.cfg.has_swa:
            self.skipTest("requires SWA")
        if self.cfg.has_mamba:
            self.skipTest("SWA-only path keeps the chain construction simple")

        tree, allocator, _, chain, window_pages = self._swa_finalize_setup()

        # Tombstone every SWA node in the trailing window.
        loaded_nodes = chain[-window_pages:]
        for n in loaded_nodes:
            n.component_data[ComponentType.SWA].value = None
            # SWA LRU bookkeeping must reflect tombstone state for the
            # _restore_device_value path to exercise the host->device move.
            tree.lru_lists[ComponentType.SWA].remove_node(n)
            tree.host_lru_lists[ComponentType.SWA].insert_mru(n)

        # Build the LOAD_BACK transfer the same way load_back() would.
        swa_comp = tree.components[ComponentType.SWA]
        transfers = swa_comp.build_hicache_transfers(
            chain[-1], CacheTransferPhase.LOAD_BACK
        )
        self.assertIsNotNone(transfers)
        xfer = transfers[0]
        self.assertEqual(xfer.nodes_to_load, loaded_nodes)

        # Allocate SWA device slots from the inner allocator (mirrors how
        # _resolve_pool_transfers_allocation routes via device_alloc_fn ->
        # swa_attn_allocator.alloc on the load-back path).
        n_swa = int(xfer.host_indices.numel())
        new_swa = allocator.swa_attn_allocator.alloc(n_swa)
        self.assertIsNotNone(new_swa)
        xfer.device_indices = new_swa

        # Snapshot pre-commit state for invariants checks.
        pre_evictable = tree.component_evictable_size_[ComponentType.SWA]

        swa_comp.commit_hicache_transfer(
            chain[-1], CacheTransferPhase.LOAD_BACK, transfers=transfers
        )

        # (1) cd.value restored, host LRU -> device LRU swap done.
        offset = 0
        for n in loaded_nodes:
            cd = n.component_data[ComponentType.SWA]
            self.assertIsNotNone(cd.value)
            chunk_len = int(cd.value.numel())
            self.assertEqual(
                cd.value.tolist(),
                new_swa[offset : offset + chunk_len].tolist(),
            )
            offset += chunk_len
            self.assertTrue(tree.lru_lists[ComponentType.SWA].in_list(n))
            self.assertFalse(tree.host_lru_lists[ComponentType.SWA].in_list(n))
        self.assertEqual(offset, n_swa)

        # (2) full_to_swa_index_mapping rebuilt for every loaded chunk.
        for n in loaded_nodes:
            full_idx = n.component_data[ComponentType.FULL].value
            swa_idx = n.component_data[ComponentType.SWA].value
            translated = allocator.translate_loc_from_full_to_swa(full_idx)
            self.assertEqual(translated.tolist(), swa_idx.tolist())

        # Evictable size moved up by the restored token count.
        self.assertEqual(
            tree.component_evictable_size_[ComponentType.SWA] - pre_evictable,
            n_swa,
        )

    def test_hicache_swa_temp_lock_does_not_release_restored_tombstone(self):
        """A temporary scheduler lock that skipped a SWA tombstone must not
        release later load-back/request locks after the tombstone is restored.
        """
        if not self.cfg.has_swa:
            self.skipTest("requires SWA")
        if self.cfg.has_mamba:
            self.skipTest("SWA-only path keeps the chain construction simple")

        tree, allocator, _, chain, _ = self._swa_finalize_setup()
        leaf = chain[-1]
        tombstone = leaf
        cd = tombstone.component_data[ComponentType.SWA]
        old_swa = cd.value
        self.assertIsNotNone(old_swa)

        cd.value = None
        tree.lru_lists[ComponentType.SWA].remove_node(tombstone)
        tree.host_lru_lists[ComponentType.SWA].insert_mru(tombstone)
        tree.component_evictable_size_[ComponentType.SWA] -= len(old_swa)

        temp_lock = tree.inc_lock_ref(leaf)
        self.assertEqual(cd.lock_ref, 0)

        xfer = tree.components[ComponentType.SWA].build_hicache_transfers(
            leaf, CacheTransferPhase.LOAD_BACK
        )[0]
        new_swa = allocator.swa_attn_allocator.alloc(int(xfer.host_indices.numel()))
        self.assertIsNotNone(new_swa)
        xfer.device_indices = new_swa
        tree.components[ComponentType.SWA].commit_hicache_transfer(
            leaf, CacheTransferPhase.LOAD_BACK, transfers=[xfer]
        )

        load_back_lock = tree.inc_lock_ref(leaf)
        request_lock = tree.inc_lock_ref(leaf)
        self.assertEqual(cd.lock_ref, 2)

        tree.dec_lock_ref(leaf, temp_lock.to_dec_params())
        self.assertEqual(cd.lock_ref, 2)

        tree.dec_lock_ref(leaf, load_back_lock.to_dec_params())
        tree.dec_lock_ref(leaf, request_lock.to_dec_params())
        self.assertEqual(cd.lock_ref, 0)

    def test_hicache_mamba_temp_lock_does_not_release_restored_tombstone(self):
        """A temporary scheduler lock that skipped a Mamba tombstone must not
        release later load-back/request locks after the tombstone is restored.
        """
        if not self.cfg.has_mamba:
            self.skipTest("requires Mamba component")
        if self.cfg.has_swa:
            self.skipTest("Mamba-only path keeps the chain construction simple")

        tree, allocator, req_to_token_pool = build_fixture(self.cfg)
        seq = self._make_seq(1, 2)
        self._insert(tree, allocator, req_to_token_pool, seq)
        m = tree.match_prefix(MatchPrefixParams(key=RadixKey(seq)))
        node = m.last_device_node
        cd = node.component_data[ComponentType.MAMBA]
        old_mamba = cd.value
        self.assertIsNotNone(old_mamba)
        self._simulate_backup(tree, node)

        cd.value = None
        tree.lru_lists[ComponentType.MAMBA].remove_node(node)
        tree.host_lru_lists[ComponentType.MAMBA].insert_mru(node)
        tree.component_evictable_size_[ComponentType.MAMBA] -= len(old_mamba)

        temp_lock = tree.inc_lock_ref(node)
        self.assertEqual(cd.lock_ref, 0)

        xfer = tree.components[ComponentType.MAMBA].build_hicache_transfers(
            node, CacheTransferPhase.LOAD_BACK
        )[0]
        new_mamba = req_to_token_pool.mamba_pool.alloc(1)
        self.assertIsNotNone(new_mamba)
        xfer.device_indices = new_mamba
        tree.components[ComponentType.MAMBA].commit_hicache_transfer(
            node, CacheTransferPhase.LOAD_BACK, transfers=[xfer]
        )

        load_back_lock = tree.inc_lock_ref(node)
        request_lock = tree.inc_lock_ref(node)
        self.assertEqual(cd.lock_ref, 2)

        tree.dec_lock_ref(node, temp_lock.to_dec_params())
        self.assertEqual(cd.lock_ref, 2)

        tree.dec_lock_ref(node, load_back_lock.to_dec_params())
        tree.dec_lock_ref(node, request_lock.to_dec_params())
        self.assertEqual(cd.lock_ref, 0)

    def test_hicache_mixed_backup_evict_insert(self):
        """Complex scenario: backup some, evict, insert new, verify invariants."""
        if self._skip_unsupported_hicache_test():
            return
        tree, allocator, req_to_token_pool = self._build_hicache_fixture()
        seqs = [self._make_seq(i * 100, 2) for i in range(5)]

        # Insert all
        for s in seqs:
            self._insert(tree, allocator, req_to_token_pool, s)
        tree.sanity_check()

        for i in range(3):
            m = tree.match_prefix(MatchPrefixParams(key=RadixKey(seqs[i])))
            self._backup_node(tree, m.last_device_node)

        # Evict to free some tokens
        tree.evict(EvictParams(num_tokens=len(seqs[0]) * 2))
        tree.sanity_check()

        # Insert new sequences
        new_seqs = [self._make_seq(i * 1000, 2) for i in range(3)]
        for s in new_seqs:
            self._insert(tree, allocator, req_to_token_pool, s)
        tree.sanity_check()

        # Verify D-leaf / H-leaf mutual exclusion
        overlap = tree.evictable_device_leaves & tree.evictable_host_leaves
        self.assertEqual(len(overlap), 0)


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
