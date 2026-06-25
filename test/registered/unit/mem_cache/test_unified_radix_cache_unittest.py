"""Unit tests for UnifiedRadixCache"""

import json
import shutil
import tempfile
import time
import unittest
from array import array
from dataclasses import dataclass, replace
from typing import Optional
from unittest import mock

import torch

from sglang.srt.configs.mamba_utils import Mamba2CacheParams, Mamba2StateShape
from sglang.srt.disaggregation.kv_events import (
    BlockRemoved,
    BlockStored,
    StorageMedium,
)
from sglang.srt.environ import envs
from sglang.srt.layers.attention.fla.chunk_delta_h import CHUNK_SIZE as FLA_CHUNK_SIZE
from sglang.srt.managers.schedule_batch import Req
from sglang.srt.mem_cache.allocator import TokenToKVPoolAllocator
from sglang.srt.mem_cache.allocator.swa import SWATokenToKVPoolAllocator
from sglang.srt.mem_cache.base_prefix_cache import (
    DecLockRefParams,
    EvictParams,
    EvictResult,
    IncLockRefResult,
    InitLoadBackParams,
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
from sglang.srt.mem_cache.swa_memory_pool import SWAKVPool
from sglang.srt.mem_cache.unified_cache_components.tree_component import (
    CacheTransferPhase,
    ComponentType,
    EvictLayer,
    TreeComponent,
)
from sglang.srt.mem_cache.unified_radix_cache import (
    COMPONENT_REGISTRY,
    UnifiedLRUList,
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
from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=10, stage="base-b", runner_config="1-gpu-small")
register_amd_ci(est_time=10, suite="stage-b-test-1-gpu-small-amd")


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
    eviction_policy: str = "lru"
    is_eagle: bool = False

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
        if self.is_eagle:
            parts.append("eagle")
        return "_".join(parts)


class _FakeFullComponent(TreeComponent):
    component_type = ComponentType.FULL

    def create_match_validator(self, match_device_only: bool = False):
        return lambda node: True

    def redistribute_on_node_split(self, new_parent, child):
        return None

    def evict_component(
        self, node, target: EvictLayer = EvictLayer.DEVICE
    ) -> tuple[int, int]:
        return 0, 0

    def drive_eviction(self, params: EvictParams, tracker: dict[ComponentType, int]):
        return None

    def acquire_component_lock(self, node, result):
        return result

    def release_component_lock(self, node, params):
        return None


class TestUnifiedRadixComponentRegistryOverride(CustomTestCase):
    def test_component_registry_override_is_instance_local(self):
        params = CacheInitParams(
            req_to_token_pool=ReqToTokenPool(
                size=2,
                max_context_len=8,
                device="cpu",
                enable_memory_saver=False,
            ),
            token_to_kv_pool_allocator=None,
            page_size=1,
            disable=True,
            tree_components=(ComponentType.FULL,),
            component_registry_override={ComponentType.FULL: _FakeFullComponent},
        )

        tree = UnifiedRadixCache(params=params)

        self.assertIsInstance(tree.components[ComponentType.FULL], _FakeFullComponent)
        self.assertIsNot(COMPONENT_REGISTRY[ComponentType.FULL], _FakeFullComponent)


class TestUnifiedTreeNodeGetPrefixHashValues(CustomTestCase):
    def test_get_prefix_hash_values_not_shared_across_calls(self):
        """Regression guard for cached mutable prefix hash lists (#26177)."""

        def make_node():
            return UnifiedTreeNode(tree_components=(ComponentType.FULL,))

        root = make_node()
        n1 = make_node()
        n1.parent = root
        n1.hash_value = ["h1"]
        n2 = make_node()
        n2.parent = n1
        n2.hash_value = ["h2"]
        n3 = make_node()
        n3.parent = n2
        n3.hash_value = ["h3"]

        first = n3.get_prefix_hash_values(n2)
        self.assertEqual(first, ["h1", "h2"])

        # Mimic downstream storage code that extends `prefix_keys` in place.
        first += ["h3"]

        second = n3.get_prefix_hash_values(n2)
        self.assertEqual(second, ["h1", "h2"])
        self.assertIsNot(second, first)

        n4 = make_node()
        n4.parent = n3
        n4.hash_value = ["h4"]
        self.assertEqual(n4.get_prefix_hash_values(n3), ["h1", "h2", "h3"])


def build_fixture(cfg: CacheConfig, *, enable_kv_cache_events: bool = False):
    """Create (tree, allocator, req_to_token_pool) from a CacheConfig."""
    server_args = ServerArgs(model_path="dummy", page_size=cfg.page_size)
    # MambaRadixCache reads mamba_cache_chunk_size, whose property otherwise
    # loads the HF config for self.model_path — impossible for the dummy model.
    # Mirror the property's default for a dummy HF config: FLA_CHUNK_SIZE.
    server_args._mamba_cache_chunk_size = max(FLA_CHUNK_SIZE, cfg.page_size)
    set_global_server_args_for_scheduler(server_args)
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
        enable_kv_cache_events=enable_kv_cache_events,
        eviction_policy=cfg.eviction_policy,
        is_eagle=cfg.is_eagle,
    )
    tree = UnifiedRadixCache(params=cache_init_params)
    tree.cache_init_params = cache_init_params

    return tree, allocator, req_to_token_pool


class TestUnifiedRadixCacheEagleHiCacheStorageKey(CustomTestCase):
    cfg = CacheConfig(
        page_size=4,
        components=(ComponentType.FULL,),
        is_eagle=True,
        kv_size=64,
        max_context_len=64,
    )

    def test_l3_prefetch_uses_bigram_radix_key(self):
        from sglang.srt.mem_cache.utils import get_hash_str

        tree, allocator, _ = build_fixture(self.cfg)
        tree.enable_storage = True
        tree.prefetch_threshold = 1
        tokens = array("q", [1, 2, 3, 4, 5, 6, 7, 8, 9])

        value = allocator.alloc(len(tokens) - 1)
        self.assertIsNotNone(value)
        tree.insert(InsertParams(key=RadixKey(tokens), value=value))
        match = tree.match_prefix(MatchPrefixParams(key=RadixKey(tokens)))
        leaf = match.last_device_node
        self.assertTrue(leaf.key.is_bigram)
        self.assertEqual(len(leaf.hash_value), 2)

        class FakeHostPool:
            def alloc(self, num_tokens):
                return torch.arange(num_tokens, dtype=torch.int64)

        class FakeCacheController:
            def __init__(self):
                self.mem_pool_host = FakeHostPool()
                self.prefetch_tokens_occupied = 0
                self.prefetch_args = None

            def prefetch_rate_limited(self):
                return False

            def prefetch(
                self,
                request_id,
                host_indices,
                new_input_tokens,
                last_hash=None,
                prefix_keys=None,
                extra_pools=None,
            ):
                self.prefetch_args = (
                    request_id,
                    host_indices,
                    new_input_tokens,
                    last_hash,
                    prefix_keys,
                    extra_pools,
                )
                return mock.Mock()

        controller = FakeCacheController()
        tree.cache_controller = controller
        tree.prefetch_from_storage("req", tree.root_node, tokens)

        _, _, storage_key, _, _, _ = controller.prefetch_args
        self.assertIsInstance(storage_key, RadixKey)
        self.assertTrue(storage_key.is_bigram)
        self.assertEqual(len(storage_key), len(tokens) - 1)

        queried_hashes = []
        running_hash = None
        for start in range(0, len(storage_key), tree.page_size):
            running_hash = get_hash_str(
                storage_key[start : start + tree.page_size], running_hash
            )
            queried_hashes.append(running_hash)
        self.assertEqual(queried_hashes, leaf.hash_value)

        canonical_hashes = []
        running_hash = None
        for start in range(0, len(tokens) - 1, tree.page_size):
            running_hash = get_hash_str(
                tokens[start : start + tree.page_size], running_hash
            )
            canonical_hashes.append(running_hash)
        self.assertNotEqual(canonical_hashes, leaf.hash_value)


class TestUnifiedRadixCacheKVEvents(CustomTestCase):
    cfg = CacheConfig(page_size=2, kv_size=64, max_context_len=64)

    def _insert(self, tree, allocator, tokens):
        key = RadixKey(array("q", tokens))
        value = allocator.alloc(len(tokens))
        self.assertIsNotNone(value)
        return tree.insert(InsertParams(key=key, value=value[: len(key)]))

    def _stored_events(self, tree, medium=None):
        events = [e for e in tree.take_events() if isinstance(e, BlockStored)]
        if medium is not None:
            events = [e for e in events if e.medium == medium]
        return events

    def _removed_events(self, tree, medium=None):
        events = [e for e in tree.take_events() if isinstance(e, BlockRemoved)]
        if medium is not None:
            events = [e for e in events if e.medium == medium]
        return events

    def _leaf_for(self, tree, tokens):
        match = tree.match_prefix(MatchPrefixParams(key=RadixKey(array("q", tokens))))
        self.assertIsNot(match.last_device_node, tree.root_node)
        return match.last_device_node

    def _init_hicache(self, tree, *, write_policy: str = "write_through"):
        import sglang.srt.mem_cache.hybrid_cache.hybrid_pool_assembler as assembler

        # Wrap the host-pool factory (not MHATokenToKVPoolHost directly)
        # because the assembler picks between MHATokenToKVPoolHost and
        # AsymmetricMHATokenToKVPoolHost via get_mha_host_pool_cls(device_pool).
        orig_get_mha_host_pool_cls = assembler.get_mha_host_pool_cls

        def get_mha_host_pool_cls_wrapper(device_pool):
            host_pool_cls = orig_get_mha_host_pool_cls(device_pool)

            def kv_host_pool_wrapper(*args, **kwargs):
                kwargs["pin_memory"] = False
                return host_pool_cls(*args, **kwargs)

            return kv_host_pool_wrapper

        patcher = mock.patch.object(
            assembler,
            "get_mha_host_pool_cls",
            side_effect=get_mha_host_pool_cls_wrapper,
        )
        patcher.start()
        self.addCleanup(patcher.stop)

        server_args = ServerArgs(
            model_path="dummy",
            page_size=self.cfg.page_size,
            hicache_io_backend="direct",
            hicache_write_policy=write_policy,
        )
        set_global_server_args_for_scheduler(server_args)
        tree.init_hicache(server_args, tree.cache_init_params)
        tree.write_through_threshold = 1 << 30
        tree.load_back_threshold = 0

    def _backup_node(self, tree, node):
        backed_up = tree.write_backup(node, write_back=True)
        self.assertGreater(backed_up, 0)
        tree.writing_check(write_back=True)

    def _load_back_node(self, tree, node):
        loaded = tree.load_back(node)
        self.assertTrue(loaded)
        producer_id = tree.ready_to_load_host_cache()
        self.assertNotEqual(producer_id, -1)
        for _, finish_event, _ in list(tree.cache_controller.ack_load_queue):
            finish_event.synchronize()
        tree.loading_check()

    def test_kv_events_store_and_remove_full_blocks(self):
        tree, allocator, _ = build_fixture(self.cfg, enable_kv_cache_events=True)
        tree.take_events()  # Clear the reset event.

        seq = [1, 2, 3, 4]
        self._insert(tree, allocator, seq)
        stored = self._stored_events(tree, StorageMedium.GPU)
        self.assertEqual(len(stored), 2)
        self.assertEqual([list(e.token_ids) for e in stored], [[1, 2], [3, 4]])
        stored_hashes = [e.block_hashes[0] for e in stored]

        result = tree.evict(EvictParams(num_tokens=len(seq)))
        self.assertGreaterEqual(result.num_tokens_evicted, len(seq))
        removed = self._removed_events(tree, StorageMedium.GPU)
        self.assertCountEqual([e.block_hashes[0] for e in removed], stored_hashes)

    def test_kv_events_split_preserves_block_hash_parentage(self):
        tree, allocator, _ = build_fixture(self.cfg, enable_kv_cache_events=True)
        tree.take_events()  # Clear the reset event.

        self._insert(tree, allocator, [1, 2, 3, 4])
        first_insert = self._stored_events(tree, StorageMedium.GPU)
        self.assertEqual(len(first_insert), 2)
        split_parent_hash = first_insert[0].block_hashes[0]

        self._insert(tree, allocator, [1, 2, 5, 6])
        second_insert = self._stored_events(tree, StorageMedium.GPU)
        self.assertEqual(len(second_insert), 1)
        self.assertEqual(list(second_insert[0].token_ids), [5, 6])
        self.assertEqual(second_insert[0].parent_block_hash, split_parent_hash)

        split_parent = next(iter(tree.root_node.children.values()))
        split_child = split_parent.children.get((3, 4))
        self.assertIsNotNone(split_child)
        self.assertEqual(len(split_parent.hash_value), 1)
        self.assertIsNotNone(split_child.hash_value)
        self.assertEqual(len(split_child.hash_value), 1)

    def test_hicache_kv_events_track_gpu_cpu_transitions(self):
        tree, allocator, _ = build_fixture(self.cfg, enable_kv_cache_events=True)
        self._init_hicache(tree)
        tree.take_events()  # Clear reset / init events.

        seq = [1, 2, 3, 4]
        self._insert(tree, allocator, seq)
        stored_gpu = self._stored_events(tree, StorageMedium.GPU)
        self.assertEqual(len(stored_gpu), 2)
        stored_hashes = [e.block_hashes[0] for e in stored_gpu]

        node = self._leaf_for(tree, seq)
        self._backup_node(tree, node)
        stored_cpu = self._stored_events(tree, StorageMedium.CPU)
        self.assertCountEqual([e.block_hashes[0] for e in stored_cpu], stored_hashes)

        tree.evict(EvictParams(num_tokens=len(seq)))
        removed_gpu = self._removed_events(tree, StorageMedium.GPU)
        self.assertCountEqual([e.block_hashes[0] for e in removed_gpu], stored_hashes)

        self._load_back_node(tree, node)
        restored_gpu = self._stored_events(tree, StorageMedium.GPU)
        self.assertCountEqual([e.block_hashes[0] for e in restored_gpu], stored_hashes)

        tree.evict(EvictParams(num_tokens=len(seq)))
        self._removed_events(tree, StorageMedium.GPU)
        tree.evict_host(len(seq))
        removed_cpu = self._removed_events(tree, StorageMedium.CPU)
        self.assertCountEqual([e.block_hashes[0] for e in removed_cpu], stored_hashes)

    def test_hicache_split_pending_write_through_publishes_fragments(self):
        tree, allocator, _ = build_fixture(self.cfg, enable_kv_cache_events=True)
        self._init_hicache(tree)
        tree.take_events()

        self._insert(tree, allocator, [1, 2, 3, 4])
        node = self._leaf_for(tree, [1, 2, 3, 4])
        backed_up = tree.write_backup(node, write_back=True)
        self.assertGreater(backed_up, 0)

        # Split the node while its write-through DMA is still pending.
        self._insert(tree, allocator, [1, 2, 5, 6])
        self.assertEqual(self._stored_events(tree, StorageMedium.CPU), [])

        # Each fragment must also be persisted to L3 on ack: lock_node only
        # holds the suffix after the split.
        tree.enable_storage = True
        with mock.patch.object(tree, "write_backup_storage") as backup_storage:
            tree.writing_check(write_back=True)
        self.assertEqual(
            [
                list(call.args[0].key.token_ids)
                for call in backup_storage.call_args_list
            ],
            [[1, 2], [3, 4]],
        )

        # Both split fragments must be published, with intact parentage.
        stored_cpu = self._stored_events(tree, StorageMedium.CPU)
        self.assertEqual(
            [list(e.token_ids) for e in stored_cpu],
            [[1, 2], [3, 4]],
        )
        self.assertIsNone(stored_cpu[0].parent_block_hash)
        self.assertEqual(stored_cpu[1].parent_block_hash, stored_cpu[0].block_hashes[0])

    def test_hicache_reinsert_evicted_node_emits_gpu_store(self):
        tree, allocator, _ = build_fixture(self.cfg, enable_kv_cache_events=True)
        self._init_hicache(tree)
        tree.take_events()  # Clear reset / init events.

        seq = [1, 2, 3, 4]
        self._insert(tree, allocator, seq)
        stored_gpu = self._stored_events(tree, StorageMedium.GPU)
        self.assertEqual(len(stored_gpu), 2)
        stored_hashes = [e.block_hashes[0] for e in stored_gpu]

        node = self._leaf_for(tree, seq)
        self._backup_node(tree, node)
        self._stored_events(tree, StorageMedium.CPU)

        tree.evict(EvictParams(num_tokens=len(seq)))
        self._removed_events(tree, StorageMedium.GPU)
        self.assertTrue(node.evicted)
        self.assertTrue(node.backuped)

        self._insert(tree, allocator, seq)
        restored_gpu = self._stored_events(tree, StorageMedium.GPU)
        self.assertFalse(node.evicted)
        self.assertCountEqual([e.block_hashes[0] for e in restored_gpu], stored_hashes)


class UnifiedRadixCacheSuite:

    cfg: CacheConfig
    _rid: int = 0

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

    def _apply_match_to_req(self, req, match):
        req.prefix_indices = match.device_indices
        req.last_node = match.last_device_node
        req.last_host_node = match.last_host_node
        req.best_match_node = match.best_match_node
        req.host_hit_length = match.host_hit_length
        req.swa_host_hit_length = match.swa_host_hit_length
        req.mamba_host_hit_length = match.mamba_host_hit_length

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

    def _insert(self, tree, allocator, req_to_token_pool, tokens, priority=0):
        """Insert tokens, attaching mamba data when the config has mamba."""
        key = RadixKey(array("q", tokens))
        value = self._alloc(allocator, len(tokens))
        params = InsertParams(key=key, value=value[: len(key)], priority=priority)
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

        m = tree.match_prefix(MatchPrefixParams(key=RadixKey(array("q", seq_b))))
        self.assertEqual(len(m.device_indices), len(seq_b))

        m = tree.match_prefix(
            MatchPrefixParams(key=RadixKey(array("q", seq_a + self._make_seq(9000, 1))))
        )
        self.assertEqual(len(m.device_indices), len(seq_a))

        m = tree.match_prefix(
            MatchPrefixParams(key=RadixKey(array("q", self._make_seq(5000, 2))))
        )
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
            m = tree.match_prefix(MatchPrefixParams(key=RadixKey(array("q", seq))))
            self.assertEqual(len(m.device_indices), len(seq))

        m = tree.match_prefix(
            MatchPrefixParams(key=RadixKey(array("q", base + self._make_seq(999, 1))))
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

        m = tree.match_prefix(MatchPrefixParams(key=RadixKey(array("q", seq_a))))
        lock_result = tree.inc_lock_ref(m.last_device_node)

        result = tree.evict(EvictParams(num_tokens=len(seq_a) + len(seq_b)))
        self.assertGreaterEqual(result.num_tokens_evicted, len(seq_b))

        m = tree.match_prefix(MatchPrefixParams(key=RadixKey(array("q", seq_a))))
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

        m = tree.match_prefix(MatchPrefixParams(key=RadixKey(array("q", seqs[0]))))
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
        key_2p = RadixKey(array("q", seq_2p))
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
        key_3p = RadixKey(array("q", seq_3p))
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
            m = tree.match_prefix(MatchPrefixParams(key=RadixKey(array("q", seq))))
            self.assertEqual(len(m.device_indices), len(seq))

        m = tree.match_prefix(
            MatchPrefixParams(key=RadixKey(array("q", base + self._make_seq(999, 1))))
        )
        self.assertEqual(len(m.device_indices), len(base))
        tree.sanity_check()

    def test_cache_finished_req_insert(self):
        tree, allocator, req_to_token_pool = build_fixture(self.cfg)
        ps = self.cfg.page_size

        req = self._make_req(req_to_token_pool)
        input_ids = self._make_seq(1, 3)
        output_ids = self._make_seq(2000, 1)
        req.origin_input_ids = array("q", input_ids)
        req.output_ids = array("q", output_ids)
        kv_len = len(input_ids) + len(output_ids)
        kv_indices = self._alloc(allocator, kv_len)
        req_to_token_pool.write((req.req_pool_idx, slice(0, kv_len)), kv_indices)
        req.kv_committed_len = kv_len
        req.last_node = tree.root_node
        req.cache_protected_len = 0
        req.swa_uuid_for_lock = None
        req.extra_key = None
        req.full_untruncated_fill_ids = array("q", input_ids + output_ids)
        req.set_extend_range(
            len(req.prefix_indices), len(req.full_untruncated_fill_ids)
        )
        if self.cfg.has_mamba:
            req.mamba_last_track_seqlen = kv_len

        tree.cache_finished_req(req, is_insert=True)

        all_ids = input_ids + output_ids
        aligned_len = (len(all_ids) // ps) * ps
        m = tree.match_prefix(
            MatchPrefixParams(key=RadixKey(array("q", all_ids[:aligned_len])))
        )
        self.assertEqual(len(m.device_indices), aligned_len)
        tree.sanity_check()

    def test_cache_finished_req_strips_thinking(self):
        tree, allocator, req_to_token_pool = build_fixture(self.cfg)
        ps = self.cfg.page_size

        req = self._make_req(req_to_token_pool)
        prompt_ids = self._make_seq(1, 3)
        output_ids = self._make_seq(2000, 7)
        req.origin_input_ids = array("q", prompt_ids)
        req.output_ids = array("q", output_ids)
        req.full_untruncated_fill_ids = array("q", prompt_ids + output_ids)
        req.set_extend_range(
            len(req.prefix_indices), len(req.full_untruncated_fill_ids)
        )
        kv_len = req.extend_range.end
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
        m = tree.match_prefix(
            MatchPrefixParams(key=RadixKey(array("q", prompt_ids + output_ids)))
        )
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
        req.origin_input_ids = array("q", tokens)
        req.output_ids = array("q")
        kv_len = len(tokens)
        kv_indices = self._alloc(allocator, kv_len)
        req_to_token_pool.write((req.req_pool_idx, slice(0, kv_len)), kv_indices)
        req.kv_committed_len = kv_len
        req.last_node = tree.root_node
        req.cache_protected_len = 0
        req.swa_uuid_for_lock = None
        req.extra_key = None
        req.full_untruncated_fill_ids = array("q", tokens)
        req.set_extend_range(
            len(req.prefix_indices), len(req.full_untruncated_fill_ids)
        )

        avail_before = allocator.available_size()
        tree.cache_finished_req(req, is_insert=False)

        self.assertEqual(allocator.available_size(), avail_before + kv_len)
        m = tree.match_prefix(MatchPrefixParams(key=RadixKey(array("q", tokens))))
        self.assertEqual(len(m.device_indices), 0)
        tree.sanity_check()

    def test_cache_unfinished_req(self):
        tree, allocator, req_to_token_pool = build_fixture(self.cfg)

        req = self._make_req(req_to_token_pool)
        tokens = self._make_seq(1, 3)
        req.origin_input_ids = array("q", tokens)
        req.output_ids = array("q")
        req.full_untruncated_fill_ids = array("q", tokens)
        req.set_extend_range(
            len(req.prefix_indices), len(req.full_untruncated_fill_ids)
        )
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

    def test_swa_unfinished_req_preserves_existing_eviction_boundary(self):
        if not self.cfg.has_swa or self.cfg.has_mamba:
            self.skipTest("requires SWA without Mamba")
        if self.cfg.page_size != 1 or self.cfg.sliding_window_size != 4:
            self.skipTest("requires page_size=1, sliding_window_size=4")
        tree, allocator, req_to_token_pool = build_fixture(self.cfg)

        req = self._make_req(req_to_token_pool)
        tokens = self._make_seq(1, 8)
        evicted_len = 4
        req.origin_input_ids = array("q", tokens)
        req.output_ids = []
        req.full_untruncated_fill_ids = array("q", tokens)
        req.set_extend_range(0, len(req.full_untruncated_fill_ids))
        kv_indices = self._alloc(allocator, len(tokens))
        req_to_token_pool.write((req.req_pool_idx, slice(0, len(tokens))), kv_indices)
        req.kv_committed_len = len(tokens)
        req.last_node = tree.root_node
        req.cache_protected_len = 0
        req.swa_uuid_for_lock = None
        req.extra_key = None
        req.swa_evicted_seqlen = evicted_len

        tree.cache_unfinished_req(req)

        first = next(iter(tree.root_node.children.values()))
        self.assertEqual(len(first.key), evicted_len)
        self.assertIsNone(first.component_data[ComponentType.SWA].value)
        live = next(iter(first.children.values()))
        self.assertEqual(len(live.key), len(tokens) - evicted_len)
        self.assertIsNotNone(live.component_data[ComponentType.SWA].value)

        m = tree.match_prefix(MatchPrefixParams(key=RadixKey(array("q", tokens))))
        self.assertEqual(len(m.device_indices), len(tokens))

        tree.dec_lock_ref(
            req.last_node,
            DecLockRefParams(swa_uuid_for_lock=getattr(req, "swa_uuid_for_lock", None)),
        )
        tree.sanity_check()

    def test_swa_insert_keeps_full_leaf_when_entire_span_is_outside_window(self):
        if not self.cfg.has_swa or self.cfg.has_mamba:
            self.skipTest("requires SWA without Mamba")
        if self.cfg.page_size != 1 or self.cfg.sliding_window_size != 4:
            self.skipTest("requires page_size=1, sliding_window_size=4")
        tree, allocator, _ = build_fixture(self.cfg)

        tokens = self._make_seq(1, 4)
        value = self._alloc(allocator, len(tokens))
        full_available_before = allocator.full_attn_allocator.available_size()

        tree.insert(
            InsertParams(
                key=RadixKey(array("q", tokens)),
                value=value,
                prev_prefix_len=0,
                swa_evicted_seqlen=len(tokens),
            )
        )

        self.assertEqual(
            allocator.full_attn_allocator.available_size(), full_available_before
        )
        node = next(iter(tree.root_node.children.values()))
        self.assertTrue(
            torch.equal(node.component_data[ComponentType.FULL].value, value)
        )
        self.assertIsNone(node.component_data[ComponentType.SWA].value)
        tree.sanity_check()

    def test_swa_overlap_recovery_preserves_locked_full_value(self):
        if not self.cfg.has_swa or self.cfg.has_mamba:
            self.skipTest("requires SWA without Mamba")
        if self.cfg.page_size != 1 or self.cfg.sliding_window_size != 4:
            self.skipTest("requires page_size=1, sliding_window_size=4")
        tree, allocator, req_to_token_pool = build_fixture(self.cfg)

        tokens = self._make_seq(1, 4)
        self._insert(tree, allocator, req_to_token_pool, tokens)
        self._insert(tree, allocator, req_to_token_pool, tokens + self._make_seq(100, 1))

        node = tree.match_prefix(
            MatchPrefixParams(key=RadixKey(array("q", tokens)))
        ).last_device_node
        old_full_value = node.component_data[ComponentType.FULL].value.clone()
        swa_component = tree.components[ComponentType.SWA]
        tracker = {ct: 0 for ct in tree.tree_components}
        tree._evict_component_and_detach_lru(node, swa_component, tracker=tracker)
        self.assertIsNone(node.component_data[ComponentType.SWA].value)

        lock_result = tree.inc_lock_ref(node)
        fresh_value = self._alloc(allocator, len(tokens))
        full_available_before_insert = allocator.full_attn_allocator.available_size()

        tree.insert(
            InsertParams(
                key=RadixKey(array("q", tokens)),
                value=fresh_value,
                prev_prefix_len=0,
                swa_evicted_seqlen=0,
            )
        )

        self.assertEqual(
            allocator.full_attn_allocator.available_size(),
            full_available_before_insert + len(tokens),
        )
        self.assertTrue(
            torch.equal(
                node.component_data[ComponentType.FULL].value,
                old_full_value,
            )
        )
        self.assertIsNone(node.component_data[ComponentType.SWA].value)

        tree.dec_lock_ref(node, lock_result.to_dec_params())
        tree.sanity_check()

    def test_swa_unfinished_req_preserves_fresh_overlap_when_locked_tombstone_rejects_match(
        self,
    ):
        if not self.cfg.has_swa or self.cfg.has_mamba:
            self.skipTest("requires SWA without Mamba")
        if self.cfg.page_size != 1 or self.cfg.sliding_window_size != 4:
            self.skipTest("requires page_size=1, sliding_window_size=4")
        tree, allocator, req_to_token_pool = build_fixture(self.cfg)

        tokens = self._make_seq(1, 4)
        self._insert(tree, allocator, req_to_token_pool, tokens)
        self._insert(tree, allocator, req_to_token_pool, tokens + self._make_seq(100, 1))

        node = tree.match_prefix(
            MatchPrefixParams(key=RadixKey(array("q", tokens)))
        ).last_device_node
        old_full_value = node.component_data[ComponentType.FULL].value.clone()
        swa_component = tree.components[ComponentType.SWA]
        tracker = {ct: 0 for ct in tree.tree_components}
        tree._evict_component_and_detach_lru(node, swa_component, tracker=tracker)
        lock_result = tree.inc_lock_ref(node)

        req = self._make_req(req_to_token_pool)
        req.origin_input_ids = array("q", tokens)
        req.output_ids = []
        req.full_untruncated_fill_ids = array("q", tokens)
        req.set_extend_range(0, len(req.full_untruncated_fill_ids))
        fresh_value = self._alloc(allocator, len(tokens))
        req_to_token_pool.write((req.req_pool_idx, slice(0, len(tokens))), fresh_value)
        req.kv_committed_len = len(tokens)
        req.last_node = tree.root_node
        req.cache_protected_len = 0
        req.swa_uuid_for_lock = None
        req.extra_key = None
        req.swa_evicted_seqlen = 0
        full_available_before = allocator.full_attn_allocator.available_size()

        tree.cache_unfinished_req(req)

        self.assertEqual(
            allocator.full_attn_allocator.available_size(), full_available_before
        )
        self.assertEqual(req.cache_protected_len, 0)
        self.assertEqual(req.prefix_indices.tolist(), fresh_value.tolist())
        self.assertTrue(
            torch.equal(
                node.component_data[ComponentType.FULL].value,
                old_full_value,
            )
        )
        self.assertIsNone(node.component_data[ComponentType.SWA].value)

        tree.dec_lock_ref(node, lock_result.to_dec_params())
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
            m = tree.match_prefix(MatchPrefixParams(key=RadixKey(array("q", seq))))
            self.assertEqual(len(m.device_indices), len(seq))

        m = tree.match_prefix(
            MatchPrefixParams(key=RadixKey(array("q", base + self._make_seq(999, 1))))
        )
        self.assertEqual(len(m.device_indices), len(base))
        tree.sanity_check()

    def test_paged_child_key_is_tuple(self):
        if self.cfg.page_size == 1:
            self.skipTest("page_size > 1 only")
        tree, _, _ = build_fixture(self.cfg)
        key = RadixKey(array("q", self._make_seq(1, 1)))
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
        m = tree.match_prefix(MatchPrefixParams(key=RadixKey(array("q", unaligned))))
        self.assertEqual(len(m.device_indices), len(seq))

        # Below-page-size key aligns to 0 -> no match.
        m = tree.match_prefix(
            MatchPrefixParams(key=RadixKey(array("q", seq[: ps - 1])))
        )
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
        m = tree.match_prefix(MatchPrefixParams(key=RadixKey(array("q", bad_page2))))
        self.assertEqual(len(m.device_indices), ps)

        # Mismatch in first page → 0 match
        bad_page1 = [9999] + seq[1:]
        m = tree.match_prefix(MatchPrefixParams(key=RadixKey(array("q", bad_page1))))
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
        req.origin_input_ids = array("q", input_ids)
        req.output_ids = array("q")
        kv_len = len(input_ids)
        kv_indices = self._alloc(allocator, kv_len)
        req_to_token_pool.write((req.req_pool_idx, slice(0, kv_len)), kv_indices)
        req.kv_committed_len = kv_len
        req.last_node = tree.root_node
        req.cache_protected_len = 0
        req.swa_uuid_for_lock = None
        req.extra_key = None
        req.full_untruncated_fill_ids = array("q", input_ids)
        req.set_extend_range(
            len(req.prefix_indices), len(req.full_untruncated_fill_ids)
        )
        if self.cfg.has_mamba:
            req.mamba_last_track_seqlen = kv_len

        avail_before = allocator.available_size()
        tree.cache_finished_req(req, is_insert=True)

        self.assertEqual(allocator.available_size(), avail_before + tail_extra)
        aligned = input_ids[: (len(input_ids) // ps) * ps]
        m = tree.match_prefix(MatchPrefixParams(key=RadixKey(array("q", aligned))))
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

        m = tree.match_prefix(MatchPrefixParams(key=RadixKey(array("q", seq_long))))
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
            MatchPrefixParams(key=RadixKey(array("q", seq)), cow_mamba=True, req=req2)
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

        m = tree.match_prefix(MatchPrefixParams(key=RadixKey(array("q", seq))))
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

    def test_leaf_transition_swa_evict_spares_locked_full(self):
        if not self.cfg.has_swa or not self.cfg.has_mamba:
            self.skipTest("requires SWA and Mamba components")
        tree, allocator, req_to_token_pool = build_fixture(self.cfg)

        n_short = (self.cfg.sliding_window_size // self.cfg.page_size) + 4
        seq_a = self._make_seq(1, n_short)
        seq_ab = seq_a + self._make_seq(7000, 2)
        self._insert(tree, allocator, req_to_token_pool, seq_a)
        self._insert(tree, allocator, req_to_token_pool, seq_ab)

        m = tree.match_prefix(MatchPrefixParams(key=RadixKey(array("q", seq_a))))
        node_a = m.last_device_node
        self.assertGreater(len(node_a.children), 0, "A must be internal")

        swa_cd = node_a.component_data[ComponentType.SWA]
        mamba_cd = node_a.component_data[ComponentType.MAMBA]
        full_cd = node_a.component_data[ComponentType.FULL]

        # A request locks A, then decodes past the window → early-release the SWA
        # portion. On this internal node, dec_swa_lock_only also drops the
        # strictly-lower-tier Mamba lock (the co-located Mamba is useless once SWA
        # is gone), leaving only the Full path-lock held. This is what guarantees
        # the later SWA-eviction cascade never meets a legitimately-locked Mamba.
        lock_result = tree.inc_lock_ref(node_a)
        self.assertGreaterEqual(mamba_cd.lock_ref, 1, "Mamba locked before release")
        tree.dec_swa_lock_only(node_a, lock_result.swa_uuid_for_lock)
        self.assertEqual(swa_cd.lock_ref, 0)
        self.assertEqual(
            mamba_cd.lock_ref, 0, "dec_swa_lock_only drops the lower-tier Mamba lock"
        )
        self.assertGreaterEqual(full_cd.lock_ref, 1)
        self.assertTrue(tree.lru_lists[ComponentType.SWA].in_list(node_a))

        # Evict the child branch (Full/device eviction only) → A becomes a
        # Full-locked leaf with its now-unlocked SWA still in the SWA LRU. We do
        # NOT tombstone aux at the leaf-transition; the held Full pins the node.
        tree.evict(EvictParams(num_tokens=len(seq_ab)))

        self.assertEqual(len(node_a.children), 0, "A should now be a leaf")
        self.assertGreaterEqual(full_cd.lock_ref, 1, "Full must stay locked")
        self.assertTrue(
            tree.lru_lists[ComponentType.SWA].in_list(node_a),
            "A's unlocked SWA stays in the LRU (not tombstoned at transition)",
        )

        # SWA eviction now selects A (it is in the SWA LRU). A's Full lock is a
        # higher-or-equal internal tier, so the cascade skips it and spares the
        # Full KV. The unlocked lower-tier Mamba is cascaded as part of the atomic
        # leaf teardown. This used to assert `cd.lock_ref == 0` on the locked Full.
        tree.evict(EvictParams(num_tokens=0, swa_num_tokens=len(seq_a)))
        self.assertGreaterEqual(full_cd.lock_ref, 1, "Full must remain locked")
        self.assertIsNotNone(full_cd.value, "Full KV must survive the SWA cascade")
        self.assertIsNone(swa_cd.value, "A's SWA was freed by its own eviction")
        tree.sanity_check()

        tree.dec_lock_ref(node_a, DecLockRefParams(swa_uuid_for_lock=None))
        tree.sanity_check()

    def test_swa_early_release_drops_co_located_mamba_lock(self):
        if not self.cfg.has_swa or not self.cfg.has_mamba:
            self.skipTest("requires SWA and Mamba components")
        tree, allocator, req_to_token_pool = build_fixture(self.cfg)

        n_short = (self.cfg.sliding_window_size // self.cfg.page_size) + 4
        seq_a = self._make_seq(1, n_short)
        self._insert(tree, allocator, req_to_token_pool, seq_a)
        node_a = tree.match_prefix(
            MatchPrefixParams(key=RadixKey(array("q", seq_a)))
        ).last_device_node
        self.assertEqual(len(node_a.children), 0, "A must be a leaf")

        swa_cd = node_a.component_data[ComponentType.SWA]
        mamba_cd = node_a.component_data[ComponentType.MAMBA]
        full_cd = node_a.component_data[ComponentType.FULL]
        self.assertIsNotNone(mamba_cd.value, "A must hold a Mamba checkpoint")

        # Natural lock acquisition — records inc_lock_ref in the lock trace.
        lock_result = tree.inc_lock_ref(node_a)
        self.assertGreaterEqual(swa_cd.lock_ref, 1, "SWA locked")
        self.assertGreaterEqual(mamba_cd.lock_ref, 1, "Mamba locked")
        self.assertGreaterEqual(full_cd.lock_ref, 1, "Full locked")

        # Early SWA release (decode advanced past the window), via the public
        # path the scheduler calls. The leaf's SWA is tombstoned and the
        # co-located lower-tier Mamba lock must drop in the same release.
        tree.dec_swa_lock_only(node_a, lock_result.swa_uuid_for_lock)
        self.assertEqual(swa_cd.lock_ref, 0, "SWA early-released")
        self.assertEqual(
            mamba_cd.lock_ref,
            0,
            "Mamba lock must drop on early SWA release",
        )
        self.assertGreaterEqual(full_cd.lock_ref, 1, "Full stays locked")

    def test_cascade_evict_asserts_on_locked_internal_mamba(self):
        if not self.cfg.has_swa or not self.cfg.has_mamba:
            self.skipTest("requires SWA and Mamba components")
        tree, allocator, req_to_token_pool = build_fixture(self.cfg)

        n_short = (self.cfg.sliding_window_size // self.cfg.page_size) + 4
        seq_a = self._make_seq(1, n_short)
        seq_ab = seq_a + self._make_seq(7000, 2)
        self._insert(tree, allocator, req_to_token_pool, seq_a)
        self._insert(tree, allocator, req_to_token_pool, seq_ab)

        node_a = tree.match_prefix(
            MatchPrefixParams(key=RadixKey(array("q", seq_a)))
        ).last_device_node
        self.assertGreater(len(node_a.children), 0, "A must be internal")

        mamba_cd = node_a.component_data[ComponentType.MAMBA]
        full_cd = node_a.component_data[ComponentType.FULL]
        self.assertIsNotNone(mamba_cd.value, "A must hold a Mamba checkpoint")

        # Lock ONLY Mamba — a stranded lower-priority lock that no supported path
        # produces. The cascade must surface it rather than silently skip.
        tree.components[ComponentType.MAMBA].acquire_component_lock(
            node_a, IncLockRefResult()
        )
        self.assertGreaterEqual(mamba_cd.lock_ref, 1, "Mamba locked")
        self.assertEqual(full_cd.lock_ref, 0, "Full unlocked")

        tracker = {ct: 0 for ct in tree.tree_components}
        tree._evict_component_and_detach_lru(
            node_a,
            tree.components[ComponentType.SWA],
            target=EvictLayer.DEVICE,
            tracker=tracker,
        )
        # No higher-or-equal tier pins the node, so even with early-release on
        # the stranded Mamba lock must trip the hard-invariant assert.
        with self.assertRaises(AssertionError):
            tree._cascade_evict(node_a, tree.components[ComponentType.SWA], tracker)

        # Clean up the forced lock so teardown/sanity is consistent.
        tree.components[ComponentType.MAMBA].release_component_lock(
            node_a, DecLockRefParams(swa_uuid_for_lock=None)
        )

    def test_cascade_evict_asserts_on_locked_leaf_mamba(self):
        if not self.cfg.has_swa or not self.cfg.has_mamba:
            self.skipTest("requires SWA and Mamba components")
        tree, allocator, req_to_token_pool = build_fixture(self.cfg)

        n_short = (self.cfg.sliding_window_size // self.cfg.page_size) + 4
        seq_a = self._make_seq(1, n_short)
        self._insert(tree, allocator, req_to_token_pool, seq_a)

        node_a = tree.match_prefix(
            MatchPrefixParams(key=RadixKey(array("q", seq_a)))
        ).last_device_node
        self.assertEqual(len(node_a.children), 0, "A must be a leaf")

        mamba_cd = node_a.component_data[ComponentType.MAMBA]
        full_cd = node_a.component_data[ComponentType.FULL]
        self.assertIsNotNone(mamba_cd.value, "A must hold a Mamba checkpoint")

        # Lock ONLY Mamba (Full stays unlocked) — a stranded lower-tier lock.
        tree.components[ComponentType.MAMBA].acquire_component_lock(
            node_a, IncLockRefResult()
        )
        self.assertGreaterEqual(mamba_cd.lock_ref, 1, "Mamba locked")
        self.assertEqual(full_cd.lock_ref, 0, "Full unlocked")

        tracker = {ct: 0 for ct in tree.tree_components}
        tree._evict_component_and_detach_lru(
            node_a,
            tree.components[ComponentType.SWA],
            target=EvictLayer.DEVICE,
            tracker=tracker,
        )
        # No higher-or-equal tier pins the node, so even with early-release on
        # the stranded Mamba lock must trip the hard-invariant assert.
        with self.assertRaises(AssertionError):
            tree._cascade_evict(node_a, tree.components[ComponentType.SWA], tracker)

        # Clean up the forced lock so teardown/sanity is consistent.
        tree.components[ComponentType.MAMBA].release_component_lock(
            node_a, DecLockRefParams(swa_uuid_for_lock=None)
        )

    def test_dec_swa_lock_only_hicache_child_on_host_treated_as_device_leaf(self):
        if not self.cfg.has_swa or not self.cfg.has_mamba:
            self.skipTest("requires SWA and Mamba components")

        tree, allocator, req_to_token_pool = build_fixture(self.cfg)

        n_short = (self.cfg.sliding_window_size // self.cfg.page_size) + 4
        seq_a = self._make_seq(1, n_short)
        seq_ab = seq_a + self._make_seq(7000, 2)
        self._insert(tree, allocator, req_to_token_pool, seq_a)
        self._insert(tree, allocator, req_to_token_pool, seq_ab)

        node_a = tree.match_prefix(
            MatchPrefixParams(key=RadixKey(array("q", seq_a)))
        ).last_device_node
        self.assertGreater(len(node_a.children), 0, "A must have children")

        self.assertFalse(
            tree._is_device_leaf(node_a),
            "A is not a device-leaf while child holds Full on device",
        )

        self._simulate_backup(tree, node_a)
        self.assertTrue(node_a.backuped, "node_a must be backuped (invariant a)")

        def _collect_descendants(node):
            out = []
            for c in list(node.children.values()):
                out.extend(_collect_descendants(c))
                out.append(c)
            return out

        descendants = _collect_descendants(node_a)
        self.assertGreater(len(descendants), 0)
        for desc in descendants:
            self._simulate_backup(tree, desc)
            self.assertTrue(desc.backuped, "desc must be backuped before demote")
            tracker = {ct: 0 for ct in tree.tree_components}
            tree._evict_to_host(desc, tracker)
            self.assertTrue(desc.evicted, "desc should be D->H demoted")
            self.assertIsNone(desc.component_data[ComponentType.FULL].value)

        self.assertTrue(
            tree._is_device_leaf(node_a),
            "A is a HiCache device-leaf (no child with Full on device)",
        )
        self.assertGreater(len(node_a.children), 0, "A still has tree-children")
        self.assertIn(node_a, tree.evictable_device_leaves)
        tree.sanity_check()

        lock_result = tree.inc_lock_ref(node_a)
        swa_cd = node_a.component_data[ComponentType.SWA]
        mamba_cd = node_a.component_data[ComponentType.MAMBA]
        full_cd = node_a.component_data[ComponentType.FULL]
        self.assertGreaterEqual(swa_cd.lock_ref, 1)
        self.assertGreaterEqual(mamba_cd.lock_ref, 1)
        self.assertGreaterEqual(full_cd.lock_ref, 1)

        tree.dec_swa_lock_only(node_a, lock_result.swa_uuid_for_lock)
        self.assertEqual(swa_cd.lock_ref, 0, "SWA released")
        self.assertEqual(mamba_cd.lock_ref, 0, "Mamba dropped by dec_swa_lock_only")
        self.assertGreaterEqual(full_cd.lock_ref, 1, "Full kept by contract")
        self.assertIsNotNone(
            swa_cd.value,
            "SWA slot stays under contract (lazy reclaim by drive_eviction)",
        )
        self.assertTrue(
            tree.lru_lists[ComponentType.SWA].in_list(node_a),
            "SWA stays in LRU for drive_eviction to pick later",
        )

        tree.dec_lock_ref(
            node_a, DecLockRefParams(swa_uuid_for_lock=None), skip_swa=True
        )
        self.assertTrue(tree._is_device_leaf(node_a))
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

        m = tree.match_prefix(MatchPrefixParams(key=RadixKey(array("q", seq_a))))
        lock_result = tree.inc_lock_ref(m.last_device_node)

        result = tree.evict(EvictParams(num_tokens=len(seq_a) + len(seq_b)))
        self.assertGreaterEqual(result.num_tokens_evicted, len(seq_b))

        m = tree.match_prefix(MatchPrefixParams(key=RadixKey(array("q", seq_a))))
        self.assertEqual(len(m.device_indices), len(seq_a))

        tree.dec_lock_ref(
            m.last_device_node,
            DecLockRefParams(swa_uuid_for_lock=lock_result.swa_uuid_for_lock),
        )
        tree.sanity_check()

    def test_swa_leaf_capped_to_window_on_insert(self):
        """A long SWA leaf is split so locking it protects one window of SWA
        while full attention still protects the whole sequence."""
        if not self.cfg.has_swa:
            self.skipTest("requires SWA component")

        ps = self.cfg.page_size
        window = self.cfg.sliding_window_size
        tail_size = ((window + ps - 1) // ps) * ps
        tail_pages = tail_size // ps

        for case in ("long_splits", "short_keeps"):
            with self.subTest(case=case):
                tree, allocator, req_to_token_pool = build_fixture(self.cfg)
                num_pages = tail_pages + 2 if case == "long_splits" else tail_pages
                seq = self._make_seq(1, num_pages)
                self._insert(tree, allocator, req_to_token_pool, seq)
                tree.sanity_check()

                leaf = tree.match_prefix(
                    MatchPrefixParams(key=RadixKey(array("q", seq)))
                ).last_device_node
                swa_val = leaf.component_data[ComponentType.SWA].value
                self.assertIsNotNone(swa_val)

                if case == "long_splits":
                    # Capped to one page-aligned window; prefix is a real ancestor.
                    self.assertEqual(len(swa_val), tail_size)
                    self.assertIsNot(leaf.parent, tree.root_node)
                else:
                    # Already within one window — no split.
                    self.assertEqual(len(swa_val), len(seq))
                    self.assertIs(leaf.parent, tree.root_node)

                lock_result = tree.inc_lock_ref(leaf)
                # SWA pins one window; full attention pins everything.
                self.assertEqual(tree.swa_protected_size(), len(swa_val))
                self.assertEqual(tree.full_protected_size(), len(seq))
                tree.sanity_check()
                tree.dec_lock_ref(
                    leaf,
                    DecLockRefParams(swa_uuid_for_lock=lock_result.swa_uuid_for_lock),
                )
                tree.sanity_check()

    def _swa_lru_order(self, tree):
        lru = tree.lru_lists[ComponentType.SWA]
        pt = lru._pt
        nodes: list = []
        cur = lru.head.lru_next[pt]
        while cur is not lru.tail:
            nodes.append(cur)
            cur = cur.lru_next[pt]
        return nodes

    def _swa_pinning_cfg_supported(self) -> bool:
        if not self.cfg.has_swa or self.cfg.has_mamba:
            return False
        cushion = self.cfg.sliding_window_size + self.cfg.page_size
        pages_per_node = 8
        side_pages = 5
        if pages_per_node * self.cfg.page_size < cushion:
            return False
        chain_inserts_pages = 4 * pages_per_node + side_pages
        if self.cfg.kv_size < chain_inserts_pages * self.cfg.page_size:
            return False
        return True

    def test_swa_lru_walk_down_does_not_refresh_ancestors_during_insert(self):
        if not self._swa_pinning_cfg_supported():
            self.skipTest("requires SWA-only config with node size >= cushion")
        tree, allocator, req_to_token_pool = build_fixture(self.cfg)

        seq_a = self._make_seq(1, 8)
        seq_ab = seq_a + self._make_seq(100, 8)
        seq_abc = seq_ab + self._make_seq(200, 8)
        self._insert(tree, allocator, req_to_token_pool, seq_a)
        self._insert(tree, allocator, req_to_token_pool, seq_ab)
        self._insert(tree, allocator, req_to_token_pool, seq_abc)

        seq_side = self._make_seq(900, 5)
        self._insert(tree, allocator, req_to_token_pool, seq_side)

        pre = self._swa_lru_order(tree)
        # Each 8-page segment is cap-split into [prefix, tail]; the tail leads
        # the pair in MRU order, so segment tails sit at even indices.
        self.assertEqual(len(pre), 8)
        side_node, c_node, b_node, a_node = pre[0], pre[2], pre[4], pre[6]

        seq_abcd = seq_abc + self._make_seq(300, 8)
        self._insert(tree, allocator, req_to_token_pool, seq_abcd)

        post = self._swa_lru_order(tree)
        # New segment E adds two nodes (prefix + tail).
        self.assertEqual(len(post), 10)
        # side branch must still appear BEFORE B and A in MRU->LRU order:
        # walk-down on the new segment must not refresh old ancestors.
        side_pos = post.index(side_node)
        self.assertLess(
            side_pos,
            post.index(b_node),
            f"side branch must remain ahead of B (no walk-down refresh); "
            f"post={[n.id for n in post]}, side={side_node.id}, B={b_node.id}",
        )
        self.assertLess(
            side_pos,
            post.index(a_node),
            f"side branch must remain ahead of A (no walk-down refresh); "
            f"post={[n.id for n in post]}, side={side_node.id}, A={a_node.id}",
        )
        tree.sanity_check()

    def test_swa_lru_match_only_refreshes_window_cushion(self):
        if not self._swa_pinning_cfg_supported():
            self.skipTest("requires SWA-only config with node size >= cushion")
        tree, allocator, req_to_token_pool = build_fixture(self.cfg)

        seq_a = self._make_seq(1, 8)
        seq_ab = seq_a + self._make_seq(100, 8)
        seq_abc = seq_ab + self._make_seq(200, 8)
        self._insert(tree, allocator, req_to_token_pool, seq_a)
        self._insert(tree, allocator, req_to_token_pool, seq_ab)
        self._insert(tree, allocator, req_to_token_pool, seq_abc)

        seq_side = self._make_seq(900, 5)
        self._insert(tree, allocator, req_to_token_pool, seq_side)

        pre = self._swa_lru_order(tree)
        # Each 8-page segment is cap-split into [prefix, tail]; tails lead in
        # MRU order, so segment tails sit at even indices.
        self.assertEqual(len(pre), 8)
        side_node, c_node, b_node, a_node = pre[0], pre[2], pre[4], pre[6]

        m = tree.match_prefix(MatchPrefixParams(key=RadixKey(array("q", seq_abc))))
        self.assertEqual(len(m.device_indices), len(seq_abc))

        post = self._swa_lru_order(tree)
        # Matching seq_abc refreshes only the window cushion (C's capped nodes)
        # to the MRU side; out-of-cushion ancestors B and A keep their order.
        self.assertIn(
            c_node, post[:2], f"C must be refreshed to MRU; post={[n.id for n in post]}"
        )
        self.assertLess(
            post.index(side_node),
            post.index(b_node),
            "Side branch must NOT be pushed below ancestors after deep match",
        )
        self.assertLess(post.index(b_node), post.index(a_node), "B must stay above A")
        self.assertGreaterEqual(
            post.index(a_node),
            len(post) - 2,
            "Oldest out-of-cushion ancestor A must stay at the LRU-tail end",
        )
        tree.sanity_check()

    def test_swa_lru_old_ancestors_evict_first_under_pressure(self):
        if not self._swa_pinning_cfg_supported():
            self.skipTest("requires SWA-only config with node size >= cushion")
        tree, allocator, req_to_token_pool = build_fixture(self.cfg)

        seq_a = self._make_seq(1, 8)
        seq_ab = seq_a + self._make_seq(100, 8)
        seq_abc = seq_ab + self._make_seq(200, 8)
        self._insert(tree, allocator, req_to_token_pool, seq_a)
        self._insert(tree, allocator, req_to_token_pool, seq_ab)
        self._insert(tree, allocator, req_to_token_pool, seq_abc)

        seq_side = self._make_seq(900, 5)
        self._insert(tree, allocator, req_to_token_pool, seq_side)

        m = tree.match_prefix(MatchPrefixParams(key=RadixKey(array("q", seq_abc))))
        self.assertEqual(len(m.device_indices), len(seq_abc))

        m_side_before = tree.match_prefix(
            MatchPrefixParams(key=RadixKey(array("q", seq_side)))
        )
        self.assertEqual(len(m_side_before.device_indices), len(seq_side))

        tree.evict(EvictParams(num_tokens=0, swa_num_tokens=self.cfg.page_size))

        m_side_after = tree.match_prefix(
            MatchPrefixParams(key=RadixKey(array("q", seq_side)))
        )
        self.assertEqual(
            len(m_side_after.device_indices),
            len(seq_side),
            "Side branch SWA must survive eviction; oldest ancestors (A) "
            "should be evicted first under bounded SWA LRU refresh.",
        )
        tree.sanity_check()

    def test_swa_lru_cushion_bound_is_sliding_window_plus_page_size(self):
        if not self._swa_pinning_cfg_supported():
            self.skipTest("requires SWA-only config with node size >= cushion")
        tree, allocator, req_to_token_pool = build_fixture(self.cfg)

        seq_a = self._make_seq(1, 8)
        seq_ab = seq_a + self._make_seq(100, 8)
        seq_abc = seq_ab + self._make_seq(200, 8)
        self._insert(tree, allocator, req_to_token_pool, seq_a)
        self._insert(tree, allocator, req_to_token_pool, seq_ab)
        self._insert(tree, allocator, req_to_token_pool, seq_abc)

        seq_side = self._make_seq(900, 5)
        self._insert(tree, allocator, req_to_token_pool, seq_side)

        pre = self._swa_lru_order(tree)
        self.assertEqual(len(pre), 8)
        side_node, c_node, b_node, a_node = pre[0], pre[2], pre[4], pre[6]
        c_prefix = pre[3]  # C's prefix pairs with its tail (c_node) at pre[2:4]

        m = tree.match_prefix(MatchPrefixParams(key=RadixKey(array("q", seq_abc))))
        self.assertEqual(len(m.device_indices), len(seq_abc))
        post = self._swa_lru_order(tree)

        cushion = self.cfg.sliding_window_size + self.cfg.page_size
        # Under leaf-cap no single node exceeds the cushion; it spans C's capped
        # tail plus its prefix, so both of C's nodes are refreshed to the MRU
        # side while B and A keep their relative order below.
        self.assertLess(len(c_node.key), cushion)
        self.assertIn(c_node, post[:2])
        self.assertIn(c_prefix, post[:2])
        side_pos = post.index(side_node)
        b_pos = post.index(b_node)
        a_pos = post.index(a_node)
        self.assertLess(side_pos, b_pos, "B was below side in pre, must stay below")
        self.assertLess(b_pos, a_pos, "A was below B in pre, must stay below")
        tree.sanity_check()

    def test_swa_eager_eviction_on_unfinished_req(self):
        if not self.cfg.has_swa or self.cfg.has_mamba:
            self.skipTest(
                "requires SWA without Mamba (mamba alters effective_cache_len)"
            )
        if self.cfg.page_size != 1 or self.cfg.sliding_window_size != 4:
            self.skipTest("requires page_size=1, sliding_window_size=4")
        tree, allocator, req_to_token_pool = build_fixture(self.cfg)

        pre_len = 20
        req = self._make_req(req_to_token_pool)
        tokens = self._make_seq(1, pre_len)
        req.origin_input_ids = tokens
        req.output_ids = []
        req.full_untruncated_fill_ids = array("q", tokens)
        req.set_extend_range(0, len(req.full_untruncated_fill_ids))
        kv_indices = self._alloc(allocator, pre_len)
        req_to_token_pool.write((req.req_pool_idx, slice(0, pre_len)), kv_indices)
        req.kv_committed_len = pre_len
        req.last_node = tree.root_node
        req.cache_protected_len = 0
        req.swa_uuid_for_lock = None
        req.extra_key = None
        req.swa_evicted_seqlen = 0

        swa_avail_before = allocator.swa_attn_allocator.available_size()

        with envs.SGLANG_OPT_UNIFIED_CACHE_FREE_OUT_OF_WINDOW_SLOTS.override(True):
            tree.cache_unfinished_req(req)

        cushion = self.cfg.sliding_window_size + self.cfg.page_size
        expected_evicted = (pre_len - 1) - cushion
        self.assertEqual(
            req.swa_evicted_seqlen,
            expected_evicted,
            f"swa_evicted_seqlen should advance to (pre_len-1) - cushion = "
            f"{expected_evicted}, got {req.swa_evicted_seqlen}",
        )

        swa_avail_after = allocator.swa_attn_allocator.available_size()
        self.assertGreaterEqual(
            swa_avail_after - swa_avail_before,
            expected_evicted,
            f"SWA pool should have freed at least {expected_evicted} slots; "
            f"before={swa_avail_before}, after={swa_avail_after}",
        )

        tree.dec_lock_ref(
            req.last_node,
            DecLockRefParams(swa_uuid_for_lock=getattr(req, "swa_uuid_for_lock", None)),
        )
        tree.sanity_check()

    def test_swa_eager_eviction_noop_when_within_window(self):
        if not self.cfg.has_swa or self.cfg.has_mamba:
            self.skipTest("requires SWA without Mamba")
        if self.cfg.page_size != 1 or self.cfg.sliding_window_size != 4:
            self.skipTest("requires page_size=1, sliding_window_size=4")
        tree, allocator, req_to_token_pool = build_fixture(self.cfg)

        cushion = self.cfg.sliding_window_size + self.cfg.page_size  # = 5
        pre_len = cushion  # exactly at the boundary, nothing slid out
        req = self._make_req(req_to_token_pool)
        tokens = self._make_seq(1, pre_len)
        req.origin_input_ids = tokens
        req.output_ids = []
        req.full_untruncated_fill_ids = array("q", tokens)
        req.set_extend_range(0, len(req.full_untruncated_fill_ids))
        kv_indices = self._alloc(allocator, pre_len)
        req_to_token_pool.write((req.req_pool_idx, slice(0, pre_len)), kv_indices)
        req.kv_committed_len = pre_len
        req.last_node = tree.root_node
        req.cache_protected_len = 0
        req.swa_uuid_for_lock = None
        req.extra_key = None
        req.swa_evicted_seqlen = 0

        with envs.SGLANG_OPT_UNIFIED_CACHE_FREE_OUT_OF_WINDOW_SLOTS.override(True):
            tree.cache_unfinished_req(req)

        self.assertEqual(
            req.swa_evicted_seqlen,
            0,
            "Nothing should be evicted when prefill fits inside the cushion",
        )

        tree.dec_lock_ref(
            req.last_node,
            DecLockRefParams(swa_uuid_for_lock=getattr(req, "swa_uuid_for_lock", None)),
        )
        tree.sanity_check()

    def test_swa_sanity_check_passes_after_deep_match(self):
        if not self._swa_pinning_cfg_supported():
            self.skipTest("requires SWA-only config with node size >= cushion")
        tree, allocator, req_to_token_pool = build_fixture(self.cfg)

        seq_a = self._make_seq(1, 8)
        seq_ab = seq_a + self._make_seq(100, 8)
        seq_abc = seq_ab + self._make_seq(200, 8)
        self._insert(tree, allocator, req_to_token_pool, seq_a)
        self._insert(tree, allocator, req_to_token_pool, seq_ab)
        self._insert(tree, allocator, req_to_token_pool, seq_abc)
        self._insert(tree, allocator, req_to_token_pool, self._make_seq(900, 5))

        for _ in range(3):
            m = tree.match_prefix(MatchPrefixParams(key=RadixKey(array("q", seq_abc))))
            self.assertEqual(len(m.device_indices), len(seq_abc))
            tree.sanity_check()

    def test_tombstone_cleanup_respects_locked_parent(self):
        tree, _, _ = build_fixture(self.cfg)
        parent = UnifiedTreeNode(self.cfg.components)
        deleted = UnifiedTreeNode(self.cfg.components)

        parent.key = RadixKey(array("q", self._make_seq(1, 1)))
        deleted.key = RadixKey(array("q", self._make_seq(1000, 1)))
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
        if not hasattr(UnifiedRadixCache, "_match_prefix_helper_readonly"):
            self.skipTest("_match_prefix_helper_readonly is not available")
        tree, allocator, req_to_token_pool = build_fixture(self.cfg)

        self._insert(tree, allocator, req_to_token_pool, [1, 2, 3, 4, 5])

        def count_nodes(node):
            count = 1
            for child in node.children.values():
                count += count_nodes(child)
            return count

        node_count_before = count_nodes(tree.root_node)
        self.assertEqual(node_count_before, 2)

        tree._match_prefix_helper(RadixKey(array("q", [1, 2])))
        (
            value,
            best_match_node,
            best_match_device_node,
            best_value_len,
        ) = tree._match_prefix_helper(RadixKey(array("q", [1, 2, 3, 4])))
        self.assertEqual(best_value_len, 2)
        self.assertEqual(list(best_match_node.key.token_ids), [3, 4])
        self.assertIs(best_match_device_node, best_match_node)
        node_count_after_regular = count_nodes(tree.root_node)
        self.assertEqual(node_count_after_regular, node_count_before + 2)

        (
            value,
            best_match_node,
            best_match_device_node,
            best_value_len,
        ) = tree._match_prefix_helper_readonly(RadixKey(array("q", [1, 2, 3])))
        self.assertEqual(best_value_len, 1)
        self.assertEqual(list(best_match_node.key.token_ids), [1, 2])
        self.assertIs(best_match_device_node, best_match_node)
        node_count_after_readonly = count_nodes(tree.root_node)
        self.assertEqual(node_count_after_readonly, node_count_after_regular)

        tree.sanity_check()

    # ================================================================
    # Evict chain tests covering demotion, cascade, and tombstone cleanup.
    # ================================================================

    def test_aux_evict_full_locked_leaf_tombstones_aux_only(self):
        aux_types = [
            ct
            for ct in (ComponentType.SWA, ComponentType.MAMBA)
            if ct in self.cfg.components
        ]
        if not aux_types:
            self.skipTest("requires an auxiliary component")
        if len(aux_types) > 1:
            self.skipTest("single-aux case keeps cascade expectations precise")
        aux = aux_types[0]

        tree, allocator, req_to_token_pool = build_fixture(self.cfg)
        # One page stays within a single SWA window (no leaf-cap split).
        seq = self._make_seq(1, 1)
        self._insert(tree, allocator, req_to_token_pool, seq)

        match = tree.match_prefix(MatchPrefixParams(key=RadixKey(array("q", seq))))
        node = match.last_device_node
        full_cd = node.component_data[ComponentType.FULL]
        aux_cd = node.component_data[aux]
        self.assertEqual(len(node.children), 0)
        self.assertIsNotNone(full_cd.value)
        self.assertIsNotNone(aux_cd.value)

        lock_result = tree.inc_lock_ref(node)
        self.assertGreater(full_cd.lock_ref, 0)
        self.assertGreater(aux_cd.lock_ref, 0)

        aux_len = len(aux_cd.value)
        tree.component_protected_size_[aux] -= aux_len
        tree.component_evictable_size_[aux] += aux_len
        aux_cd.lock_ref = 0
        self.assertNotIn(node, tree.evictable_device_leaves)

        evict_params = EvictParams(num_tokens=0)
        if aux == ComponentType.SWA:
            evict_params.swa_num_tokens = aux_len
        else:
            evict_params.mamba_num = aux_len
        result = tree.evict(evict_params)

        self.assertEqual(result.num_tokens_evicted, 0)
        if aux == ComponentType.SWA:
            self.assertEqual(result.swa_num_tokens_evicted, aux_len)
        else:
            self.assertEqual(result.mamba_num_evicted, aux_len)
        self.assertIsNotNone(full_cd.value)
        self.assertIsNone(aux_cd.value)
        self.assertFalse(tree.lru_lists[aux].in_list(node))

        tree.dec_lock_ref(
            node,
            DecLockRefParams(swa_uuid_for_lock=lock_result.swa_uuid_for_lock),
        )
        tree.sanity_check()

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
        m_base = tree.match_prefix(MatchPrefixParams(key=RadixKey(array("q", base))))
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
        tree.match_prefix(MatchPrefixParams(key=RadixKey(array("q", seq_new))))

        # Evict just enough for one sequence
        tree.evict(EvictParams(num_tokens=len(seq_old)))

        # seq_old should be gone (LRU), seq_new should remain
        m_old = tree.match_prefix(MatchPrefixParams(key=RadixKey(array("q", seq_old))))
        m_new = tree.match_prefix(MatchPrefixParams(key=RadixKey(array("q", seq_new))))
        self.assertEqual(len(m_old.device_indices), 0)
        self.assertEqual(len(m_new.device_indices), len(seq_new))
        tree.sanity_check()

    def test_evict_respects_priority_policy(self):
        if self.cfg.components != (ComponentType.FULL,):
            self.skipTest("priority policy ordering is covered on Full-only configs")
        priority_cfg = replace(self.cfg, eviction_policy="priority")
        tree, allocator, req_to_token_pool = build_fixture(priority_cfg)
        seq_high = self._make_seq(1, 2)
        seq_low = self._make_seq(500, 2)

        self._insert(tree, allocator, req_to_token_pool, seq_high, priority=10)
        self._insert(tree, allocator, req_to_token_pool, seq_low, priority=0)

        tree.evict(EvictParams(num_tokens=len(seq_low)))

        m_high = tree.match_prefix(
            MatchPrefixParams(key=RadixKey(array("q", seq_high)))
        )
        m_low = tree.match_prefix(MatchPrefixParams(key=RadixKey(array("q", seq_low))))
        self.assertEqual(len(m_high.device_indices), len(seq_high))
        self.assertEqual(len(m_low.device_indices), 0)
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
        m = tree.match_prefix(MatchPrefixParams(key=RadixKey(array("q", branch_b))))
        lr = tree.inc_lock_ref(m.last_device_node)

        # Evict — branch_a should go, base + branch_b stay
        tree.evict(EvictParams(num_tokens=len(branch_a)))

        m_b = tree.match_prefix(MatchPrefixParams(key=RadixKey(array("q", branch_b))))
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
        m = tree.match_prefix(MatchPrefixParams(key=RadixKey(array("q", seq_a))))
        lr = tree.inc_lock_ref(m.last_device_node)

        # Try to evict everything
        total = tree.full_evictable_size() + tree.full_protected_size()
        result = tree.evict(EvictParams(num_tokens=total))

        # seq_a should still be matchable (protected)
        m2 = tree.match_prefix(MatchPrefixParams(key=RadixKey(array("q", seq_a))))
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
        m = tree.match_prefix(MatchPrefixParams(key=RadixKey(array("q", seq_b))))
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
        m = tree.match_prefix(MatchPrefixParams(key=RadixKey(array("q", seqs[0]))))
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

    # ---------- L3 storage (file backend) helpers ----------

    def _path_chain(self, tree, node):
        """Return root->node node chain (excluding root)."""
        chain = []
        cur = node
        while cur is not tree.root_node:
            chain.append(cur)
            cur = cur.parent
        chain.reverse()
        return chain

    def _write_path_to_l3(self, tree, node):
        """Offload every node on root->node path from host to L3 storage."""
        for n in self._path_chain(tree, node):
            tree.write_backup_storage(n)

    def _flush_l3_backups(self, tree, timeout: float = 10.0):
        """Wait for backup threads to finish, then drain acks (release locks)."""
        deadline = time.time() + timeout
        while tree.ongoing_backup and time.time() < deadline:
            tree.drain_storage_control_queues()
            if tree.ongoing_backup:
                time.sleep(0.01)
        tree.drain_storage_control_queues()
        self.assertFalse(tree.ongoing_backup, "L3 backups did not complete in time")

    def _run_prefetch_to_completion(self, tree, req_id, timeout: float = 10.0):
        deadline = time.time() + timeout
        while time.time() < deadline:
            if tree.check_prefetch_progress(req_id):
                return
            time.sleep(0.01)
        self.fail(f"prefetch {req_id} did not complete in time")

    def _all_page_hashes(self, tree, node):
        hashes = []
        for n in self._path_chain(tree, node):
            hashes.extend(list(n.hash_value))
        return hashes

    def test_hicache_l3_write_storage(self):
        """D->H->L3 offload: every KV page lands in the file storage backend."""
        if self._skip_unsupported_hicache_test():
            return
        if self.cfg.has_mamba:
            self.skipTest("mamba L3 offload is out of scope for this unit fixture")

        storage_dir = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, storage_dir, ignore_errors=True)

        tree, allocator, req_to_token_pool = build_fixture(self.cfg)
        self._init_hicache(
            tree,
            storage_backend="file",
            storage_dir=storage_dir,
            prefetch_threshold=1,
        )

        seq = self._make_seq(1, 4)
        self._insert(tree, allocator, req_to_token_pool, seq)
        m = tree.match_prefix(MatchPrefixParams(key=RadixKey(array("q", seq))))
        leaf = m.last_device_node

        # D->H first, then H->L3.
        self._backup_node(tree, leaf)
        self.assertTrue(leaf.hash_value)
        self._write_path_to_l3(tree, leaf)
        self._flush_l3_backups(tree)

        # Every KV page hash on the path must now exist in storage.
        backend = tree.cache_controller.storage_backend
        page_hashes = self._all_page_hashes(tree, leaf)
        self.assertEqual(len(page_hashes), len(seq) // self.cfg.page_size)
        self.assertEqual(backend.batch_exists(page_hashes), len(page_hashes))
        tree.sanity_check()

    def test_hicache_l3_prefetch(self):
        """L3 round trip: write with one tree, prefetch into a fresh tree.

        Uses two independent trees that share the same file storage dir so the
        prefetch path genuinely reloads from L3 (no host/device residue).
        """
        if self._skip_unsupported_hicache_test():
            return
        if self.cfg.has_mamba:
            self.skipTest("mamba L3 prefetch is out of scope for this unit fixture")

        storage_dir = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, storage_dir, ignore_errors=True)
        num_pages = 4
        if self.cfg.has_swa:
            # SWA L3 prefetch is all-or-nothing over one full sliding window.
            # Keep the generic L3 round trip valid for SWA configs whose window
            # is larger than the old fixed 4-page request.
            sw_pages = (
                self.cfg.sliding_window_size + self.cfg.page_size - 1
            ) // self.cfg.page_size
            num_pages = max(num_pages, sw_pages + 1)
        seq = self._make_seq(1, num_pages)

        # --- Producer tree: fill KV, backup D->H, offload H->L3. ---
        prod, prod_alloc, prod_rtp = build_fixture(self.cfg)
        self._init_hicache(
            prod,
            storage_backend="file",
            storage_dir=storage_dir,
            prefetch_threshold=1,
        )
        self._insert(prod, prod_alloc, prod_rtp, seq)
        mp = prod.match_prefix(MatchPrefixParams(key=RadixKey(array("q", seq))))
        prod_leaf = mp.last_device_node
        self._fill_full_kv(prod_alloc, mp.device_indices, marker=7)
        expected_k, expected_v = self._snapshot_full_kv(prod_alloc, mp.device_indices)
        self._backup_node(prod, prod_leaf)
        self._write_path_to_l3(prod, prod_leaf)
        self._flush_l3_backups(prod)

        # --- Consumer tree: prefetch the same tokens straight from L3. ---
        cons, cons_alloc, cons_rtp = build_fixture(self.cfg)
        self._init_hicache(
            cons,
            storage_backend="file",
            storage_dir=storage_dir,
            prefetch_threshold=1,
        )
        req_id = "l3-prefetch-req"
        cons.prefetch_from_storage(req_id, cons.root_node, array("q", seq), None, None)
        self._run_prefetch_to_completion(cons, req_id)
        cons.drain_storage_control_queues()

        # The full prefix must now be a host hit (loaded from L3).
        mc = cons.match_prefix(MatchPrefixParams(key=RadixKey(array("q", seq))))
        self.assertEqual(mc.host_hit_length, len(seq))
        host_node = mc.last_host_node
        self.assertIsNot(host_node, cons.root_node)
        self.assertTrue(host_node.evicted)

        # Load the reloaded host prefix back to device and verify KV bytes.
        self._load_back_node(cons, host_node)
        loaded_indices = cons.match_prefix(
            MatchPrefixParams(key=RadixKey(array("q", seq)))
        ).device_indices
        self.assertEqual(len(loaded_indices), len(seq))
        loaded_k, loaded_v = self._snapshot_full_kv(cons_alloc, loaded_indices)
        self.assertTrue(torch.equal(loaded_k, expected_k))
        self.assertTrue(torch.equal(loaded_v, expected_v))
        cons.sanity_check()

    # ---------- TP consistency for SWA prefetch (all-or-nothing) ----------

    def _patch_tp_all_reduce(self, tree, drop_swa: bool):
        """Fake all_reduce so check_prefetch_progress runs the tp>1 path."""
        import torch.distributed as dist

        min_sizes = []

        def swa_packed_index():
            # Packed tensor is [completed_tokens, *sidecar_hits]; sidecar order
            # matches comp_xfers stored in ongoing_prefetch (one live entry).
            for info in tree.ongoing_prefetch.values():
                comp_xfers = info[-1]
                names = [t.name for xfers in comp_xfers.values() for t in xfers]
                if PoolName.SWA in names:
                    return 1 + names.index(PoolName.SWA)
            return None

        def fake(tensor, op=None, group=None):
            if op == dist.ReduceOp.MIN:
                min_sizes.append(tensor.numel())
                if drop_swa:
                    idx = swa_packed_index()
                    if idx is not None and idx < tensor.numel():
                        tensor[idx] = 0
            return None

        p = mock.patch.object(dist, "all_reduce", side_effect=fake)
        p.start()
        self.addCleanup(p.stop)
        return min_sizes

    def _swa_host_on_path(self, tree, seq):
        m = tree.match_prefix(MatchPrefixParams(key=RadixKey(array("q", seq))))
        node = m.last_host_node
        while node is not tree.root_node:
            if node.component_data[ComponentType.SWA].host_value is not None:
                return True
            node = node.parent
        return False

    def _l3_produce(self, storage_dir, seq):
        prod, prod_alloc, prod_rtp = build_fixture(self.cfg)
        self._init_hicache(
            prod, storage_backend="file", storage_dir=storage_dir, prefetch_threshold=1
        )
        self._insert(prod, prod_alloc, prod_rtp, seq)
        leaf = prod.match_prefix(
            MatchPrefixParams(key=RadixKey(array("q", seq)))
        ).last_device_node
        self._backup_node(prod, leaf)
        self._write_path_to_l3(prod, leaf)
        self._flush_l3_backups(prod)

    def _l3_consumer(self, storage_dir):
        cons, _, _ = build_fixture(self.cfg)
        self._init_hicache(
            cons, storage_backend="file", storage_dir=storage_dir, prefetch_threshold=1
        )
        return cons

    def _consume_prefetch(self, cons, seq, req_id):
        cons.prefetch_from_storage(req_id, cons.root_node, array("q", seq), None, None)
        self._run_prefetch_to_completion(cons, req_id)
        cons.drain_storage_control_queues()

    def _setup_swa_tp_prefetch(self):
        """Skip non-SWA fixtures; produce one full SWA window+1 page to L3.

        Returns (storage_dir, seq) or None when this fixture cannot exercise
        SWA L3 prefetch (then the caller skips).
        """
        if not self.cfg.has_swa or self.cfg.has_mamba:
            self.skipTest("SWA-only fixture required")
        if self._skip_unsupported_hicache_test():
            return None
        sw_pages = (
            self.cfg.sliding_window_size + self.cfg.page_size - 1
        ) // self.cfg.page_size
        seq = self._make_seq(1, sw_pages + 1)

        storage_dir = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, storage_dir, ignore_errors=True)
        self._l3_produce(storage_dir, seq)

        # Baseline (single rank) must actually adopt SWA, else the TP assertions
        # below would be vacuous -> skip.
        base = self._l3_consumer(storage_dir)
        self._consume_prefetch(base, seq, "base")
        if not self._swa_host_on_path(base, seq):
            self.skipTest("fixture does not exercise SWA L3 prefetch")
        return storage_dir, seq

    def test_tp_swa_prefetch_dropped_when_peer_misses(self):
        """A peer rank missing the SWA window drops the whole prefetch result
        on every rank (TP-consistent all-or-nothing)."""
        setup = self._setup_swa_tp_prefetch()
        if setup is None:
            return
        storage_dir, seq = setup

        cons = self._l3_consumer(storage_dir)
        cons.tp_world_size = 2
        min_sizes = self._patch_tp_all_reduce(cons, drop_swa=True)
        self._consume_prefetch(cons, seq, "drop")

        m = cons.match_prefix(MatchPrefixParams(key=RadixKey(array("q", seq))))
        self.assertEqual(m.host_hit_length, 0)
        self.assertFalse(
            self._swa_host_on_path(cons, seq), "SWA must be dropped when a peer misses"
        )
        # Full + sidecars must be synced through a packed MIN all_reduce. The
        # poll loop may observe more than one completed check, so do not pin the
        # exact number of reductions.
        self.assertIn(2, min_sizes)
        cons.sanity_check()

    def test_tp_swa_prefetch_adopted_when_peer_present(self):
        """When every rank has the full SWA window, SWA is adopted under tp>1
        (the single packed all_reduce path still works)."""
        setup = self._setup_swa_tp_prefetch()
        if setup is None:
            return
        storage_dir, seq = setup

        cons = self._l3_consumer(storage_dir)
        cons.tp_world_size = 2
        min_sizes = self._patch_tp_all_reduce(cons, drop_swa=False)  # peer == local
        self._consume_prefetch(cons, seq, "keep")

        m = cons.match_prefix(MatchPrefixParams(key=RadixKey(array("q", seq))))
        self.assertEqual(m.host_hit_length, len(seq))
        self.assertTrue(
            self._swa_host_on_path(cons, seq),
            "SWA must be adopted when all ranks have it",
        )
        self.assertIn(2, min_sizes)
        cons.sanity_check()

    def _skip_unsupported_hicache_test(self):
        if self.cfg.has_swa and self.cfg.has_mamba:
            self.skipTest("HiCache unit fixture does not support SWA + Mamba stacks")
        return False

    def _simulate_backup(self, tree, node):
        """Simulate D->H backup over the whole root->node path (parent-first)."""
        chain = []
        cur = node
        while cur is not tree.root_node:
            chain.append(cur)
            cur = cur.parent
        for ancestor in reversed(chain):
            for ct in (ComponentType.FULL, ComponentType.MAMBA, ComponentType.SWA):
                if ct not in self.cfg.components:
                    continue
                cd = ancestor.component_data[ct]
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

    def _init_hicache(
        self,
        tree,
        *,
        write_policy: str = "write_through",
        storage_backend: Optional[str] = None,
        storage_dir: Optional[str] = None,
        prefetch_threshold: Optional[int] = None,
        prefetch_policy: str = "wait_complete",
    ):
        import sglang.srt.mem_cache.hybrid_cache.hybrid_pool_assembler as assembler

        # See _init_hicache: wrap the factory rather than MHATokenToKVPoolHost
        # directly so the pin_memory=False override applies to both
        # MHATokenToKVPoolHost and AsymmetricMHATokenToKVPoolHost.
        orig_get_mha_host_pool_cls = assembler.get_mha_host_pool_cls
        orig_mamba_host_pool = assembler.MambaPoolHost

        def get_mha_host_pool_cls_wrapper(device_pool):
            host_pool_cls = orig_get_mha_host_pool_cls(device_pool)

            def kv_host_pool_wrapper(*args, **kwargs):
                kwargs["pin_memory"] = False
                return host_pool_cls(*args, **kwargs)

            return kv_host_pool_wrapper

        def mamba_host_pool_wrapper(*args, **kwargs):
            kwargs["pin_memory"] = False
            return orig_mamba_host_pool(*args, **kwargs)

        patchers = [
            mock.patch.object(
                assembler,
                "get_mha_host_pool_cls",
                side_effect=get_mha_host_pool_cls_wrapper,
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

        storage_extra_config = None
        if storage_backend == "file":
            import sglang.srt.managers.cache_controller as cache_controller

            # The file-backend storage config records TP/PP rank/size.  These unit
            # fixtures run without initializing distributed parallel state, so
            # provide the local single-rank values that the fixture represents.
            tp_rank_patcher = mock.patch.object(
                cache_controller, "get_tensor_model_parallel_rank", return_value=0
            )
            tp_size_patcher = mock.patch.object(
                cache_controller, "get_tensor_model_parallel_world_size", return_value=1
            )
            pp_rank_patcher = mock.patch.object(
                cache_controller, "get_pipeline_model_parallel_rank", return_value=0
            )
            pp_size_patcher = mock.patch.object(
                cache_controller,
                "get_pipeline_model_parallel_world_size",
                return_value=1,
            )
            tp_rank_patcher.start()
            tp_size_patcher.start()
            pp_rank_patcher.start()
            pp_size_patcher.start()
            self.addCleanup(tp_rank_patcher.stop)
            self.addCleanup(tp_size_patcher.stop)
            self.addCleanup(pp_rank_patcher.stop)
            self.addCleanup(pp_size_patcher.stop)

            assert storage_dir is not None, "file backend needs a storage_dir"
            # HiCacheFile reads the directory from this env var.
            cm = envs.SGLANG_HICACHE_FILE_BACKEND_STORAGE_DIR.override(storage_dir)
            cm.__enter__()
            self.addCleanup(cm.__exit__, None, None, None)
            extra = {}
            if prefetch_threshold is not None:
                extra["prefetch_threshold"] = prefetch_threshold
            storage_extra_config = json.dumps(extra) if extra else None

        server_args = ServerArgs(
            model_path="dummy",
            page_size=self.cfg.page_size,
            hicache_io_backend="direct",
            hicache_write_policy=write_policy,
            hicache_storage_backend=storage_backend,
            hicache_storage_backend_extra_config=storage_extra_config,
            hicache_storage_prefetch_policy=prefetch_policy,
        )
        # See build_fixture for why _mamba_cache_chunk_size is preset.
        server_args._mamba_cache_chunk_size = max(FLA_CHUNK_SIZE, self.cfg.page_size)
        set_global_server_args_for_scheduler(server_args)
        tree.init_hicache(server_args, tree.cache_init_params)
        tree.write_through_threshold = 1 << 30
        tree.load_back_threshold = 0
        if storage_backend is not None:
            # Unit fixtures size host/device pools equally, which makes the
            # production prefetch capacity limit (host - device) zero.  Keep the
            # L3 tests focused on storage round trips by allowing one fixture
            # worth of prefetch tokens.
            tree.cache_controller.prefetch_capacity_limit = max(
                tree.cache_controller.prefetch_capacity_limit,
                tree.cache_controller.mem_pool_host.size,
            )
            # Background prefetch/backup threads are daemon; stop them per-test.
            self.addCleanup(tree.cache_controller._stop_storage_threads)

    def _build_hicache_fixture(self):
        fixture = build_fixture(self.cfg)
        tree, _, _ = fixture
        self._init_hicache(tree)
        return fixture

    def _backup_node(self, tree, node):
        # Parent-first backup over the whole path: one insert can span several
        # nodes, so a single-node backup would leave an unbacked ancestor.
        chain = []
        cur = node
        while cur is not tree.root_node:
            chain.append(cur)
            cur = cur.parent
        backed_up = 0
        for ancestor in reversed(chain):
            if ancestor.backuped:
                continue
            backed_up = tree.write_backup(ancestor, write_back=True)
            self.assertGreater(backed_up, 0)
        tree.writing_check(write_back=True)
        self.assertTrue(node.backuped)
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
        loaded = tree.load_back(node)
        self.assertTrue(loaded)
        producer_id = tree.ready_to_load_host_cache()
        self.assertNotEqual(producer_id, -1)
        for _, finish_event, _ in list(tree.cache_controller.ack_load_queue):
            finish_event.synchronize()
        tree.loading_check()
        return node.component_data[ComponentType.FULL].value

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

    def test_hicache_evict_device_leaf_aborts_demote_when_backup_fails(self):
        """when write_backup cannot allocate host pool,
        _evict_device_leaf should not evict it to host."""
        if self._skip_unsupported_hicache_test():
            return
        tree, allocator, req_to_token_pool = build_fixture(self.cfg)
        self._init_hicache(tree, write_policy="write_back")
        ct = ComponentType.FULL

        seq = self._make_seq(1, 2)
        self._insert(tree, allocator, req_to_token_pool, seq)
        m = tree.match_prefix(MatchPrefixParams(key=RadixKey(array("q", seq))))
        node = m.last_device_node
        self.assertIsNot(node, tree.root_node)
        self.assertFalse(node.backuped)
        self.assertFalse(node.evicted)

        tracker = {c: 0 for c in tree.tree_components}
        with mock.patch.object(tree, "write_backup", return_value=0):
            tree._evict_device_leaf(node, tracker)

        self.assertFalse(node.evicted)
        self.assertIsNotNone(node.component_data[ct].value)
        self.assertIsNone(node.component_data[ct].host_value)

        with self.assertRaises(AssertionError):
            tree._evict_to_host(node, {c: 0 for c in tree.tree_components})

        tree.sanity_check()

    def test_hicache_node_states(self):
        """Verify device-only to device+host transition after real backup."""
        if self._skip_unsupported_hicache_test():
            return
        tree, allocator, req_to_token_pool = self._build_hicache_fixture()
        seq = self._make_seq(1, 2)
        self._insert(tree, allocator, req_to_token_pool, seq)

        # Find the leaf node
        m = tree.match_prefix(MatchPrefixParams(key=RadixKey(array("q", seq))))
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

        m = tree.match_prefix(MatchPrefixParams(key=RadixKey(array("q", seq))))
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
        m = tree.match_prefix(MatchPrefixParams(key=RadixKey(array("q", leaf))))
        lr = tree.inc_lock_ref(m.last_device_node)

        # Evict base (inner node won't be evicted while child is locked)
        tree.evict(EvictParams(num_tokens=len(base)))

        tree.dec_lock_ref(
            m.last_device_node,
            DecLockRefParams(swa_uuid_for_lock=getattr(lr, "swa_uuid_for_lock", None)),
        )
        m = tree.match_prefix(MatchPrefixParams(key=RadixKey(array("q", leaf))))
        self.assertGreaterEqual(len(m.device_indices), len(base))
        tree.sanity_check()

    def test_hicache_partial_match_splits_evicted_backed_up_node(self):
        """Partial matches on host-only nodes must keep the host prefix usable."""
        if self._skip_unsupported_hicache_test():
            return
        tree, allocator, req_to_token_pool = self._build_hicache_fixture()
        ps = self.cfg.page_size
        seq = self._make_seq(1, 4)
        expected_prefix = seq[: 2 * ps]
        expected_suffix = seq[len(expected_prefix) :]
        query = expected_prefix + self._make_seq(9000, 1)

        self._insert(tree, allocator, req_to_token_pool, seq)
        m = tree.match_prefix(MatchPrefixParams(key=RadixKey(array("q", seq))))
        node = m.last_device_node
        self._backup_node(tree, node)

        tree.evict(EvictParams(num_tokens=len(seq)))
        self.assertTrue(node.evicted)
        self.assertTrue(node.backuped)

        m = tree.match_prefix(MatchPrefixParams(key=RadixKey(array("q", query))))

        self.assertEqual(len(m.device_indices), 0)
        self.assertIs(m.last_device_node, tree.root_node)

        # Locate the host prefix via last_host_node and rebuild prefix/suffix
        # from path keys (a leaf may span several nodes).
        if self.cfg.has_mamba:
            self.assertEqual(m.host_hit_length, 0)
            self.assertIs(m.last_host_node, tree.root_node)
        else:
            self.assertEqual(m.host_hit_length, len(expected_prefix))
            split_parent = m.last_host_node
            self.assertIsNot(split_parent, tree.root_node)
            self.assertTrue(split_parent.evicted)
            self.assertTrue(split_parent.backuped)
            # root -> split_parent keys reconstruct expected_prefix
            prefix_tokens: list[int] = []
            chain = []
            cur = split_parent
            while cur is not tree.root_node:
                chain.append(cur)
                cur = cur.parent
            for n in reversed(chain):
                prefix_tokens.extend(n.key.token_ids)
            self.assertEqual(prefix_tokens, expected_prefix)
            # the diverged suffix stays as evicted+backuped descendant(s)
            suffix_tokens: list[int] = []
            cur = split_parent
            while cur.children:
                self.assertEqual(len(cur.children), 1)
                cur = next(iter(cur.children.values()))
                suffix_tokens.extend(cur.key.token_ids)
            self.assertEqual(suffix_tokens, expected_suffix)
            self.assertTrue(cur.evicted and cur.backuped)
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
            m = tree.match_prefix(MatchPrefixParams(key=RadixKey(array("q", seqs[i]))))
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

        m = tree.match_prefix(MatchPrefixParams(key=RadixKey(array("q", seq))))
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

        m = tree.match_prefix(MatchPrefixParams(key=RadixKey(array("q", base))))
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

        self._load_back_node(tree, node)
        self.assertFalse(node.evicted)
        self.assertIsNotNone(node.component_data[ComponentType.FULL].value)
        # Gather the whole reloaded prefix via match (a leaf may be split).
        loaded_indices = tree.match_prefix(
            MatchPrefixParams(key=RadixKey(array("q", base)))
        ).device_indices
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

    def test_hicache_write_through_offloads_swa_split_leaf(self):
        """A SWA boundary-split leaf should offload normally under write-through."""
        if not self.cfg.has_swa:
            self.skipTest("requires SWA")
        if self.cfg.has_mamba:
            self.skipTest("SWA-only path keeps the split setup simple")

        ps = self.cfg.page_size
        tree, allocator, _ = build_fixture(self.cfg)
        self._init_hicache(tree)
        tree.write_through_threshold = 1

        seq = self._make_seq(1, 2)
        value = self._alloc(allocator, len(seq))
        result = tree.insert(
            InsertParams(
                key=RadixKey(seq),
                value=value,
                swa_evicted_seqlen=ps,
            )
        )
        self.assertEqual(result.prefix_len, 0)

        self.assertEqual(len(tree.root_node.children), 1)
        split_parent = next(iter(tree.root_node.children.values()))
        self.assertEqual(len(split_parent.children), 1)
        split_leaf = next(iter(split_parent.children.values()))

        tree.writing_check(write_back=True)
        tree.evict(EvictParams(num_tokens=len(seq)))
        self.assertTrue(split_leaf.evicted)
        self.assertTrue(split_leaf.backuped)
        self.assertIn(split_leaf, tree.evictable_host_leaves)
        tree.sanity_check()

    def test_swa_deep_tree_backup_evict_loadback_stress(self):
        """Deep multi-node SWA tree (long leaves, decode-evict tombstones,
        shared-prefix branches) through write-through backup -> evict ->
        loadback, asserting sanity throughout."""
        if not self.cfg.has_swa:
            self.skipTest("requires SWA")
        if self._skip_unsupported_hicache_test():
            return
        if self.cfg.has_mamba:
            self.skipTest("SWA-only keeps the deep-tree topology precise")

        tree, allocator, req_to_token_pool = self._build_hicache_fixture()
        tree.write_through_threshold = 1  # real eager write-through auto-backup
        ps = self.cfg.page_size
        # Window in pages; sizes scale with it so the leaf-cap split fires.
        tail_size = ((self.cfg.sliding_window_size + ps - 1) // ps) * ps
        wp = max(1, tail_size // ps)

        def insert_swa(tokens, swa_ev):
            value = self._alloc(allocator, len(tokens))
            if value is None:
                return False
            tree.insert(
                InsertParams(
                    key=RadixKey(array("q", tokens)),
                    value=value,
                    swa_evicted_seqlen=swa_ev,
                )
            )
            tree.writing_check()
            tree.sanity_check()
            return True

        base = self._make_seq(1, wp + 2)  # long leaf -> cap-split
        if not insert_swa(base, 0):
            self.skipTest("kv pool too small for deep-tree stress")
        # fresh decode-evicted seq: tombstone-prefix + cap-split stacked
        insert_swa(self._make_seq(20000, wp + 2), ps)
        insert_swa(base + self._make_seq(70000, 2), 0)  # depth
        for i in range(2):  # width: branches off the base prefix
            insert_swa(base[: 2 * ps] + self._make_seq(80000 + 1000 * i, 3), 0)

        self.assertGreaterEqual(len(tree._collect_all_nodes()), 5)

        # Stepwise eviction -> demote to host, sanity after each round.
        for _ in range(4):
            full_ev = tree.full_evictable_size()
            if full_ev == 0:
                break
            tree.evict(
                EvictParams(
                    num_tokens=max(ps, full_ev // 2),
                    swa_num_tokens=tree.swa_evictable_size(),
                )
            )
            tree.sanity_check()

        # Load evicted prefixes back from host, sanity after each.
        for tokens in (base, base[: 2 * ps]):
            m = tree.match_prefix(MatchPrefixParams(key=RadixKey(array("q", tokens))))
            anchor = m.best_match_node
            if anchor is not tree.root_node and anchor.evicted:
                if tree.load_back(anchor):
                    self._finish_pending_loads(tree)
                    self._release_ongoing_load_back_locks(tree)
            tree.sanity_check()

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

        m = tree.match_prefix(MatchPrefixParams(key=RadixKey(array("q", seq))))
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
        m = tree.match_prefix(MatchPrefixParams(key=RadixKey(array("q", seq))))
        chain: list = []
        cur = m.last_device_node
        while cur is not tree.root_node:
            chain.append(cur)
            cur = cur.parent
        chain.reverse()
        return chain

    def _release_ongoing_load_back_locks(self, tree):
        for node, lock_params, host_lock_params in list(
            tree.ongoing_load_back.values()
        ):
            tree.dec_lock_ref(node, lock_params)
            tree.dec_host_lock_ref(node, host_lock_params)
        tree.ongoing_load_back.clear()

    def _finish_pending_loads(self, tree):
        producer_id = tree.ready_to_load_host_cache()
        self.assertNotEqual(producer_id, -1)
        for _, finish_event, _ in list(tree.cache_controller.ack_load_queue):
            finish_event.synchronize()
        tree.loading_check()

    def _match_tokens_for_chain(self, chain):
        tokens: list[int] = []
        for node in chain:
            tokens.extend(node.key.token_ids)
        return tokens

    def _set_aux_host_tombstone(self, tree, node, component_type):
        cd = node.component_data[component_type]
        self.assertIsNotNone(cd.value)
        if cd.host_value is None:
            cd.host_value = cd.value.clone()
        old_value = cd.value
        cd.value = None
        if component_type in tree.lru_lists and tree.lru_lists[component_type].in_list(
            node
        ):
            tree.lru_lists[component_type].remove_node(node)
        tree.host_lru_lists[component_type].insert_mru(node)
        tree.component_evictable_size_[component_type] -= len(old_value)

    def test_match_prefix_best_and_device_node_without_hicache(self):
        tree, allocator, req_to_token_pool = build_fixture(self.cfg)
        ps = self.cfg.page_size
        min_tokens = 2 * ps
        if self.cfg.has_swa:
            min_tokens = max(min_tokens, self.cfg.sliding_window_size + ps)
        seq = self._make_seq(1, (min_tokens + ps - 1) // ps)
        self._insert(tree, allocator, req_to_token_pool, seq)

        result = tree.match_prefix(MatchPrefixParams(key=RadixKey(array("q", seq))))

        self.assertEqual(len(result.device_indices), len(seq))
        self.assertIs(result.best_match_node, result.last_device_node)
        self.assertIs(result.last_host_node, result.last_device_node)
        self.assertEqual(result.host_hit_length, 0)

    def test_hicache_mamba_host_best_match_keeps_device_anchor(self):
        if not self.cfg.has_mamba or self.cfg.has_swa or self.cfg.page_size != 1:
            self.skipTest("requires page_size=1 Full+Mamba")
        tree, allocator, req_to_token_pool = self._build_hicache_fixture()
        chain = self._build_chain_pages(tree, allocator, req_to_token_pool, 3)
        if len(chain) < 3:
            self.skipTest("chain too short")
        leaf = chain[-1]
        parent = chain[-2]
        tokens = self._match_tokens_for_chain(chain)

        self._backup_node(tree, leaf)
        tree.evict(EvictParams(num_tokens=len(leaf.key)))
        self.assertTrue(leaf.evicted)

        result = tree.match_prefix(MatchPrefixParams(key=RadixKey(array("q", tokens))))

        self.assertIs(result.best_match_node, leaf)
        self.assertIs(result.last_device_node, parent)
        self.assertEqual(len(result.device_indices), len(tokens) - len(leaf.key))
        self.assertEqual(result.host_hit_length, len(leaf.key))

    def test_hicache_swa_host_best_match_keeps_device_anchor(self):
        if not self.cfg.has_swa or self.cfg.has_mamba or self.cfg.page_size != 1:
            self.skipTest("requires page_size=1 Full+SWA")
        tree, allocator, req_to_token_pool = self._build_hicache_fixture()
        chain = self._build_chain_pages(tree, allocator, req_to_token_pool, 3)
        if len(chain) < 3:
            self.skipTest("chain too short")
        leaf = chain[-1]
        parent = chain[-2]
        tokens = self._match_tokens_for_chain(chain)

        self._backup_node(tree, leaf)
        tree.evict(EvictParams(num_tokens=len(leaf.key)))
        self.assertTrue(leaf.evicted)

        result = tree.match_prefix(MatchPrefixParams(key=RadixKey(array("q", tokens))))

        self.assertIs(result.best_match_node, leaf)
        self.assertIs(result.last_device_node, parent)
        self.assertEqual(len(result.device_indices), len(tokens) - len(leaf.key))
        self.assertEqual(result.host_hit_length, len(leaf.key))
        self.assertEqual(result.swa_host_hit_length, len(leaf.key))

        tree, allocator, req_to_token_pool = self._build_hicache_fixture()
        chain = self._build_chain_pages(tree, allocator, req_to_token_pool, 3)
        if len(chain) < 3:
            self.skipTest("chain too short")
        leaf = chain[-1]
        parent = chain[-2]
        tokens = self._match_tokens_for_chain(chain)

        self._set_aux_host_tombstone(tree, leaf, ComponentType.SWA)

        result = tree.match_prefix(MatchPrefixParams(key=RadixKey(array("q", tokens))))

        self.assertIs(result.best_match_node, leaf)
        self.assertIs(result.last_device_node, parent)
        self.assertEqual(len(result.device_indices), len(tokens) - len(leaf.key))
        self.assertEqual(result.host_hit_length, 0)
        self.assertEqual(result.swa_host_hit_length, len(leaf.key))

    def test_mamba_branching_seqlen_disabled_under_hicache(self):
        if not self.cfg.has_mamba or self.cfg.has_swa or self.cfg.page_size != 1:
            self.skipTest("requires page_size=1 Full+Mamba")
        tree, allocator, req_to_token_pool = build_fixture(self.cfg)
        chunk_size = get_global_server_args().mamba_cache_chunk_size
        tokens = self._make_seq(1, chunk_size + 1)
        self._insert(tree, allocator, req_to_token_pool, tokens)
        leaf = tree.match_prefix(
            MatchPrefixParams(key=RadixKey(array("q", tokens)))
        ).last_device_node

        mamba_cd = leaf.component_data[ComponentType.MAMBA]
        mamba_cd.value = None
        no_hicache = tree.match_prefix(
            MatchPrefixParams(key=RadixKey(array("q", tokens)))
        )
        self.assertIs(no_hicache.best_match_node, tree.root_node)
        self.assertIs(no_hicache.last_device_node, tree.root_node)
        self.assertEqual(no_hicache.mamba_branching_seqlen, chunk_size)

        tree_h, allocator_h, req_to_token_pool_h = self._build_hicache_fixture()
        self._insert(tree_h, allocator_h, req_to_token_pool_h, tokens)
        leaf_h = tree_h.match_prefix(
            MatchPrefixParams(key=RadixKey(array("q", tokens)))
        ).last_device_node
        self._backup_node(tree_h, leaf_h)
        tree_h.evict(EvictParams(num_tokens=len(tokens)))
        with_hicache = tree_h.match_prefix(
            MatchPrefixParams(key=RadixKey(array("q", tokens)))
        )
        self.assertIs(with_hicache.best_match_node, leaf_h)
        self.assertIs(with_hicache.last_device_node, tree_h.root_node)
        self.assertIsNone(with_hicache.mamba_branching_seqlen)

    def test_scheduler_hicache_full_mamba_init_load_back_appends_new_indices(self):
        if not self.cfg.has_mamba or self.cfg.has_swa or self.cfg.page_size != 1:
            self.skipTest("requires page_size=1 Full+Mamba")
        tree, allocator, req_to_token_pool = self._build_hicache_fixture()
        chain = self._build_chain_pages(tree, allocator, req_to_token_pool, 3)
        if len(chain) < 3:
            self.skipTest("chain too short")
        leaf = chain[-1]
        tokens = self._match_tokens_for_chain(chain)

        self._backup_node(tree, leaf)
        tree.evict(EvictParams(num_tokens=len(leaf.key)))
        self.assertTrue(leaf.evicted)

        req = self._make_req(req_to_token_pool)
        match = tree.match_prefix(
            MatchPrefixParams(key=RadixKey(array("q", tokens)), req=req)
        )
        self._apply_match_to_req(req, match)

        new_indices, new_node = tree.init_load_back(
            InitLoadBackParams(
                best_match_node=req.best_match_node,
                host_hit_length=req.host_hit_length,
                req=req,
            )
        )

        self.assertIs(new_node, leaf)
        self.assertEqual(len(torch.cat([req.prefix_indices, new_indices])), len(tokens))
        self.assertIsNotNone(leaf.component_data[ComponentType.MAMBA].value)
        self._finish_pending_loads(tree)
        self._release_ongoing_load_back_locks(tree)

    def test_scheduler_hicache_aux_only_load_back_appends_full_device_indices(self):
        if self.cfg.page_size != 1:
            self.skipTest("page_size=1 keeps the expected suffix precise")
        aux = None
        if self.cfg.has_swa and not self.cfg.has_mamba:
            aux = ComponentType.SWA
        elif self.cfg.has_mamba and not self.cfg.has_swa:
            aux = ComponentType.MAMBA
        if aux is None:
            self.skipTest("requires exactly one aux component")

        tree, allocator, req_to_token_pool = self._build_hicache_fixture()
        chain = self._build_chain_pages(tree, allocator, req_to_token_pool, 3)
        if len(chain) < 3:
            self.skipTest("chain too short")
        leaf = chain[-1]
        tokens = self._match_tokens_for_chain(chain)
        leaf_full = leaf.component_data[ComponentType.FULL].value.clone()
        self._backup_node(tree, leaf)
        self._set_aux_host_tombstone(tree, leaf, aux)

        req = self._make_req(req_to_token_pool)
        match = tree.match_prefix(
            MatchPrefixParams(key=RadixKey(array("q", tokens)), req=req)
        )
        self._apply_match_to_req(req, match)

        new_indices, new_node = tree.init_load_back(
            InitLoadBackParams(
                best_match_node=req.best_match_node,
                host_hit_length=req.host_hit_length,
                req=req,
            )
        )

        self.assertIs(new_node, leaf)
        self.assertEqual(new_indices.tolist(), leaf_full.tolist())
        self.assertEqual(len(torch.cat([req.prefix_indices, new_indices])), len(tokens))
        self.assertEqual(
            leaf.component_data[ComponentType.FULL].value.tolist(),
            leaf_full.tolist(),
        )
        self.assertIsNotNone(leaf.component_data[aux].value)
        self._finish_pending_loads(tree)
        self._release_ongoing_load_back_locks(tree)

    def test_scheduler_hicache_load_back_fallback_keeps_old_anchor(self):
        if not self.cfg.has_mamba or self.cfg.has_swa or self.cfg.page_size != 1:
            self.skipTest("requires page_size=1 Full+Mamba")
        tree, allocator, req_to_token_pool = self._build_hicache_fixture()
        chain = self._build_chain_pages(tree, allocator, req_to_token_pool, 3)
        if len(chain) < 3:
            self.skipTest("chain too short")
        leaf = chain[-1]
        tokens = self._match_tokens_for_chain(chain)

        self._backup_node(tree, leaf)
        tree.evict(EvictParams(num_tokens=len(leaf.key)))

        req = self._make_req(req_to_token_pool)
        match = tree.match_prefix(
            MatchPrefixParams(key=RadixKey(array("q", tokens)), req=req)
        )
        self._apply_match_to_req(req, match)

        new_indices, new_node = tree.init_load_back(
            InitLoadBackParams(
                best_match_node=req.best_match_node,
                host_hit_length=req.host_hit_length,
                req=req,
                mem_quota=-1_000_000,
            )
        )

        self.assertEqual(len(new_indices), 0)
        self.assertIs(new_node, match.last_device_node)
        self.assertIsNone(leaf.component_data[ComponentType.FULL].value)
        self.assertIsNone(leaf.component_data[ComponentType.MAMBA].value)

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
        m = tree.match_prefix(MatchPrefixParams(key=RadixKey(array("q", seq))))
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
        """finalize_match_result accumulates host_value lengths of SWA tombstones
        within the trailing sliding window into ``swa_host_hit_length``. Out-of-window
        tombstones and chains fully on device must leave ``swa_host_hit_length`` at 0.
        ``host_hit_length`` is Full-KV only and is never written by SWA.
        """
        if not self.cfg.has_swa:
            self.skipTest("requires SWA")
        if self.cfg.has_mamba:
            self.skipTest("SWA-only path keeps the chain construction simple")

        tree, _, _, chain, window_pages = self._swa_finalize_setup()
        leaf = chain[-1]
        ps = self.cfg.page_size
        swa_comp = tree.components[ComponentType.SWA]

        cases = [
            ("all_on_device", None, 0),
            ("tombstone_in_window", chain[-window_pages], ps),
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
                    best_match_node=leaf,
                    host_hit_length=0,
                )
                result = swa_comp.finalize_match_result(
                    result=result,
                    params=MatchPrefixParams(
                        key=RadixKey(array("q", self._make_seq(1, 1)))
                    ),
                    value_chunks=[],
                    best_value_len=0,
                )
                self.assertEqual(result.host_hit_length, 0)
                self.assertEqual(result.swa_host_hit_length, expected)

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

    def _swa_anchor_chain_tokens(self, num_pages: int) -> list[int]:
        """Reproduce the token sequence used by _build_chain_pages."""
        tokens: list[int] = []
        for i in range(num_pages):
            tokens.extend(self._make_seq(1000 * (i + 1), 1))
        return tokens

    def _swa_anchor_setup(self):
        """Chain layout (root-to-leaf):

            ... top padding ...
            chain[-(window_pages + 1)] = N : SWA tombstone
            chain[-(window_pages)]     = Y : SWA host-only, FULL host-backed
            chain[-(window_pages - 1)..-2] : SWA device, FULL.host=None
            chain[-1]                  = X : SWA device, FULL.host=None

        Anchor=X has window_pages of device pages below N, so the SWA
        load_back walker accumulates n_swa >= sliding_window_size and
        exits before reaching N. Anchor=Y has only 1 page and walks into
        N's tombstone.
        """
        if not self.cfg.has_swa:
            self.skipTest("requires SWA")
        if self.cfg.has_mamba:
            self.skipTest("SWA-only path keeps the chain construction simple")
        ps = self.cfg.page_size
        sw = self.cfg.sliding_window_size
        if ps >= sw:
            self.skipTest("test scenario requires ps < sw")
        window_pages = (sw + ps - 1) // ps
        chain_pages = window_pages + 3
        if chain_pages * ps > self.cfg.kv_size // 2:
            self.skipTest("kv_size too small for the desired chain")

        tree, allocator, req_to_token_pool = self._build_hicache_fixture()
        chain = self._build_chain_pages(tree, allocator, req_to_token_pool, chain_pages)
        if len(chain) < chain_pages:
            self.skipTest("chain too short")
        self._backup_tree(tree)

        x = chain[-1]
        y = chain[-window_pages]
        n = chain[-(window_pages + 1)]

        n.component_data[ComponentType.SWA].value = None
        n.component_data[ComponentType.SWA].host_value = None
        y.component_data[ComponentType.SWA].value = None
        # Strip FULL.host on X + intermediates so last_host_node walks past
        # them to Y. Y.FULL untouched preserves the leaf-up evict invariant.
        for node in chain[-(window_pages - 1) :]:
            node.component_data[ComponentType.FULL].host_value = None

        tokens = self._swa_anchor_chain_tokens(len(chain))
        return tree, chain, n, y, x, tokens

    def test_hicache_swa_load_back_anchored_on_best_match_node(self):
        tree, _, _, y, x, _ = self._swa_anchor_setup()
        ps = self.cfg.page_size
        swa_comp = tree.components[ComponentType.SWA]

        transfers = swa_comp.build_hicache_transfers(x, CacheTransferPhase.LOAD_BACK)
        self.assertEqual(len(transfers), 1)
        xfer = transfers[0]
        self.assertEqual(xfer.name, PoolName.SWA)
        self.assertEqual(xfer.nodes_to_load, [y])
        self.assertEqual(int(xfer.host_indices.numel()), ps)

        with self.assertRaises(AssertionError):
            swa_comp.build_hicache_transfers(y, CacheTransferPhase.LOAD_BACK)

    def test_hicache_swa_finalize_anchored_on_best_match_node(self):
        tree, _, _, y, x, _ = self._swa_anchor_setup()
        swa_comp = tree.components[ComponentType.SWA]
        ps = self.cfg.page_size

        base = MatchResult(
            device_indices=torch.empty((0,), dtype=torch.int64, device=tree.device),
            last_device_node=x,
            last_host_node=y,
            best_match_node=x,
            host_hit_length=0,
        )
        result = swa_comp.finalize_match_result(
            result=base,
            params=MatchPrefixParams(key=RadixKey(array("q", self._make_seq(1, 1)))),
            value_chunks=[],
            best_value_len=0,
        )
        # SWA host hit goes to swa_host_hit_length; host_hit_length stays at 0.
        self.assertEqual(result.host_hit_length, 0)
        self.assertEqual(result.swa_host_hit_length, ps)

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

    def test_hicache_swa_load_back_uses_full_pool_capacity(self):
        """load_back should gate Full KV load on Full pool capacity only."""
        if not self.cfg.has_swa:
            self.skipTest("requires SWA")
        if self.cfg.has_mamba:
            self.skipTest("SWA-only path")
        if self.cfg.page_size > 1:
            self.skipTest("page_size==1 for direct swa_attn_allocator access")

        tree, allocator, req_to_token_pool = self._build_hicache_fixture()

        sw = self.cfg.sliding_window_size
        kv_tokens = sw + 2
        chain = self._build_chain_pages(tree, allocator, req_to_token_pool, kv_tokens)
        if len(chain) < kv_tokens:
            self.skipTest("chain too short")
        leaf = chain[-1]

        self._backup_tree(tree)
        result = tree.evict(EvictParams(num_tokens=kv_tokens))
        self.assertGreaterEqual(result.num_tokens_evicted, kv_tokens)
        self.assertIsNone(leaf.component_data[ComponentType.FULL].value)

        kv_xfer = tree.components[ComponentType.FULL].build_hicache_transfers(
            leaf, CacheTransferPhase.LOAD_BACK
        )[0]
        self.assertEqual(int(kv_xfer.host_indices.numel()), kv_tokens)

        swa_xfer = tree.components[ComponentType.SWA].build_hicache_transfers(
            leaf, CacheTransferPhase.LOAD_BACK
        )[0]
        self.assertEqual(int(swa_xfer.host_indices.numel()), sw)

        # Leave tree-owned SWA available for controller-side SWA eviction.
        unrelated_seq = self._make_seq(100_000, sw)
        self._insert(tree, allocator, req_to_token_pool, unrelated_seq)
        self.assertGreaterEqual(tree.swa_evictable_size(), sw)

        # Make raw SWA availability smaller than both load-back transfers.
        target_swa_avail = sw - 1
        swa_avail = allocator.swa_attn_allocator.available_size()
        self.assertGreaterEqual(swa_avail, target_swa_avail)
        if swa_avail > target_swa_avail:
            external_swa = allocator.swa_attn_allocator.alloc(
                swa_avail - target_swa_avail
            )
            self.assertIsNotNone(external_swa)

        self.assertGreaterEqual(
            allocator.full_attn_allocator.available_size(),
            int(kv_xfer.host_indices.numel()),
        )
        self.assertLess(
            allocator.swa_attn_allocator.available_size(),
            int(kv_xfer.host_indices.numel()),
        )
        self.assertLess(
            allocator.swa_attn_allocator.available_size(),
            int(swa_xfer.host_indices.numel()),
        )

        with mock.patch.object(tree, "evict", wraps=tree.evict) as evict_mock:
            self.assertTrue(tree.load_back(leaf))

        # Full pre-eviction must not be triggered by SWA pool pressure.
        full_pre_evict_calls = [
            call
            for call in evict_mock.call_args_list
            if call.args and call.args[0].num_tokens > 0
        ]
        self.assertEqual(full_pre_evict_calls, [])

        # SWA shortage is handled by the controller through SWA-only eviction.
        self.assertTrue(
            any(
                call.args
                and call.args[0].num_tokens == 0
                and call.args[0].swa_num_tokens > 0
                for call in evict_mock.call_args_list
            )
        )

        self._finish_pending_loads(tree)
        self.assertIsNotNone(leaf.component_data[ComponentType.FULL].value)
        self._release_ongoing_load_back_locks(tree)
        tree.sanity_check()

    def test_hicache_full_temp_lock_skips_evicted_anchor_and_mirrors_on_release(
        self,
    ):
        """Acquire records the evicted anchor in skip_lock_node_ids (phase 1)
        and locks device-on ancestors only (phase 2). After load_back
        restores the anchor, a second acquire covers it; releasing the
        first must mirror the skip so the anchor's lock_ref is not
        decremented twice.
        """
        if self._skip_unsupported_hicache_test():
            return
        ps = self.cfg.page_size
        if 3 * ps > self.cfg.kv_size // 2:
            self.skipTest("kv_size too small")
        tree, allocator, req_to_token_pool = self._build_hicache_fixture()
        chain = self._build_chain_pages(tree, allocator, req_to_token_pool, 3)
        if len(chain) < 3:
            self.skipTest("chain too short")
        a, y, anchor = chain
        self._simulate_backup_tree(tree)

        cd_anchor = anchor.component_data[ComponentType.FULL]
        cd_a = a.component_data[ComponentType.FULL]
        cd_y = y.component_data[ComponentType.FULL]
        anchor_value = cd_anchor.value
        cd_anchor.value = None

        self.assertEqual(cd_anchor.lock_ref, 0)
        self.assertEqual(cd_y.lock_ref, 0)
        self.assertEqual(cd_a.lock_ref, 0)

        temp_lock = tree.inc_lock_ref(anchor)
        self.assertEqual(cd_anchor.lock_ref, 0)
        self.assertEqual(cd_y.lock_ref, 1)
        self.assertEqual(cd_a.lock_ref, 1)
        self.assertIn(ComponentType.FULL, temp_lock.skip_lock_node_ids)
        self.assertIn(anchor.id, temp_lock.skip_lock_node_ids[ComponentType.FULL])

        cd_anchor.value = anchor_value

        second_lock = tree.inc_lock_ref(anchor)
        self.assertEqual(cd_anchor.lock_ref, 1)
        self.assertEqual(cd_y.lock_ref, 2)
        self.assertEqual(cd_a.lock_ref, 2)

        tree.dec_lock_ref(anchor, temp_lock.to_dec_params())
        self.assertEqual(cd_anchor.lock_ref, 1)
        self.assertEqual(cd_y.lock_ref, 1)
        self.assertEqual(cd_a.lock_ref, 1)

        tree.dec_lock_ref(anchor, second_lock.to_dec_params())
        self.assertEqual(cd_anchor.lock_ref, 0)
        self.assertEqual(cd_y.lock_ref, 0)
        self.assertEqual(cd_a.lock_ref, 0)

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
        m = tree.match_prefix(MatchPrefixParams(key=RadixKey(array("q", seq))))
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
        new_mamba = req_to_token_pool.mamba_allocator.alloc(1)
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
            m = tree.match_prefix(MatchPrefixParams(key=RadixKey(array("q", seqs[i]))))
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

    def test_hicache_write_back_leaf_backup(self):
        """write_back: evicting a device leaf backs it up to host"""
        if self._skip_unsupported_hicache_test():
            return
        tree, allocator, req_to_token_pool = build_fixture(self.cfg)
        self._init_hicache(tree, write_policy="write_back")

        base = self._make_seq(1, 2)
        leaf_seq = base + self._make_seq(500, 2)
        self._insert(tree, allocator, req_to_token_pool, base)
        self._insert(tree, allocator, req_to_token_pool, leaf_seq)

        m = tree.match_prefix(MatchPrefixParams(key=RadixKey(array("q", leaf_seq))))
        leaf = m.last_device_node
        parent = leaf.parent
        self.assertIsNot(parent, tree.root_node)

        self.assertFalse(leaf.backuped)
        self.assertFalse(parent.backuped)

        lr = tree.inc_lock_ref(parent)
        try:
            evict_tokens = len(leaf_seq) - len(base)
            tree.evict(EvictParams(num_tokens=evict_tokens))
        finally:
            tree.dec_lock_ref(
                parent,
                DecLockRefParams(
                    swa_uuid_for_lock=getattr(lr, "swa_uuid_for_lock", None)
                ),
            )

        self.assertTrue(leaf.evicted, "leaf should be demoted to host")
        self.assertTrue(leaf.backuped, "write_back must back up the leaf on eviction")
        self.assertFalse(
            parent.backuped, "parent must NOT be backed up under write_back"
        )

        tree.sanity_check()


class UnifiedLRUListBoundedRefreshTest(CustomTestCase):

    components = (ComponentType.FULL, ComponentType.SWA)

    def _make_node(self, key_len: int) -> UnifiedTreeNode:
        n = UnifiedTreeNode(self.components)
        n.key = RadixKey(list(range(key_len)))
        return n

    def _build_chain(self, key_lens: list[int]) -> tuple:
        root = self._make_node(0)
        nodes = []
        parent = root
        for kl in key_lens:
            n = self._make_node(kl)
            n.parent = parent
            nodes.append(n)
            parent = n
        return root, nodes

    def _lru_order(self, lru: UnifiedLRUList) -> list:
        pt = lru._pt
        out = []
        cur = lru.head.lru_next[pt]
        while cur is not lru.tail:
            out.append(cur)
            cur = cur.lru_next[pt]
        return out

    def test_bounded_refresh_stops_after_accumulated_meets_window(self):
        root, [a, b, c, d] = self._build_chain([2, 2, 2, 2])
        lru = UnifiedLRUList(ComponentType.SWA, self.components)
        for n in (a, b, c, d):
            lru.insert_mru(n)
        self.assertEqual(self._lru_order(lru), [d, c, b, a])

        # window=5, page_size=1 implicit; nodes are size 2 each
        # Walking up from D: visit D(acc=2<5) -> visit C(acc=4<5) -> visit
        # B(acc=6>=5, refresh and stop). A is NOT touched.
        lru.reset_node_and_window_ancestors_mru(
            d, root, window_size=5, should_include=lambda _n: True
        )
        # Expected MRU->LRU: D, C, B (refreshed in walk-up order), A (untouched)
        self.assertEqual(self._lru_order(lru), [d, c, b, a])

    def test_bounded_refresh_skips_non_included(self):
        root, [a, b, c, d] = self._build_chain([2, 2, 2, 2])
        lru = UnifiedLRUList(ComponentType.SWA, self.components)
        for n in (a, c, d):  # b excluded from LRU (simulated tombstone)
            lru.insert_mru(n)
        self.assertEqual(self._lru_order(lru), [d, c, a])

        included = {a, c, d}
        lru.reset_node_and_window_ancestors_mru(
            d, root, window_size=5, should_include=lambda n: n in included
        )
        # D, C refreshed; B contributes 2 to acc (now 6 >= 5) but is skipped;
        # A is not visited because the walk stops at B.
        self.assertEqual(self._lru_order(lru), [d, c, a])

    def test_bounded_refresh_visits_only_until_window_filled(self):
        root, [a, b, c, d] = self._build_chain([3, 3, 3, 3])
        lru = UnifiedLRUList(ComponentType.SWA, self.components)
        # MRU->LRU: A, B, C, D (deepest is at the LRU tail, oldest)
        for n in (d, c, b, a):
            lru.insert_mru(n)
        self.assertEqual(self._lru_order(lru), [a, b, c, d])

        # window=5: walking up from D visits D(acc=3<5) and C(acc=6>=5, stop).
        # Order expected: [D, C, A, B]. Why: D and C move to head in that
        # order (D first, then C right after D). A and B keep their relative
        # positions (they were the surviving prefix [A, B] before).
        lru.reset_node_and_window_ancestors_mru(
            d, root, window_size=5, should_include=lambda _n: True
        )
        self.assertEqual(self._lru_order(lru), [d, c, a, b])

    def test_bounded_refresh_stops_at_root(self):
        root, [a, b] = self._build_chain([1, 1])
        lru = UnifiedLRUList(ComponentType.SWA, self.components)
        for n in (a, b):
            lru.insert_mru(n)
        self.assertEqual(self._lru_order(lru), [b, a])

        # Big window — refresh walks A and B both, then hits root and stops.
        lru.reset_node_and_window_ancestors_mru(
            b, root, window_size=1000, should_include=lambda _n: True
        )
        self.assertEqual(self._lru_order(lru), [b, a])

    def test_bounded_refresh_window_zero_is_noop(self):
        root, [a, b, c] = self._build_chain([2, 2, 2])
        lru = UnifiedLRUList(ComponentType.SWA, self.components)
        for n in (a, b, c):
            lru.insert_mru(n)
        before = self._lru_order(lru)
        lru.reset_node_and_window_ancestors_mru(
            c, root, window_size=0, should_include=lambda _n: True
        )
        self.assertEqual(self._lru_order(lru), before)


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
