"""Shared fixtures for the UnifiedRadixCache unit tests."""

import atexit
import json
from array import array
from dataclasses import dataclass
from typing import Optional
from unittest import mock

import torch

from sglang.kernels.ops.attention.fla.chunk_delta_h import CHUNK_SIZE as FLA_CHUNK_SIZE
from sglang.srt.configs.mamba_utils import Mamba2CacheParams, Mamba2StateShape
from sglang.srt.environ import envs
from sglang.srt.managers.schedule_batch import (
    Req,
)
from sglang.srt.mem_cache.allocator import TokenToKVPoolAllocator
from sglang.srt.mem_cache.allocator.swa import SWATokenToKVPoolAllocator
from sglang.srt.mem_cache.base_prefix_cache import (
    InsertParams,
    MatchPrefixParams,
)
from sglang.srt.mem_cache.cache_init_params import CacheInitParams
from sglang.srt.mem_cache.events import KvEventLoraNames
from sglang.srt.mem_cache.memory_pool import (
    HybridLinearKVPool,
    HybridReqToTokenPool,
    MHATokenToKVPool,
    ReqToTokenPool,
)
from sglang.srt.mem_cache.radix_cache import RadixKey
from sglang.srt.mem_cache.swa_memory_pool import SWAKVPool
from sglang.srt.mem_cache.unified_cache.cache_action import (
    BackupKV,
)
from sglang.srt.mem_cache.unified_cache.components.base import (
    ComponentType,
    TreeComponent,
)
from sglang.srt.mem_cache.unified_cache.tree_core_registry import (
    _TREE_CORE_REGISTRY,
    resolve_tree_core_backend,
)
from sglang.srt.mem_cache.unified_radix_cache import (
    UnifiedRadixCache,
)
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.srt.server_args import (
    ServerArgs,
    set_global_server_args_for_scheduler,
)
from sglang.srt.utils import get_device
from sglang.test.kits.unified_radix_cache_unittest.unified_tree_core_inspection_interface import (
    UnifiedTreeCoreInspectionInterface,
)
from sglang.test.kits.unified_radix_cache_unittest.unified_tree_core_inspector import (
    UnifiedTreeCoreInspector,
)

# A dedicated test entry point overrides this without changing the process-wide
# production backend selection. Direct Python-core tests in this module remain
# Python-only; every fixture-backed cache test is shared by both inspectors.
_TREE_CORE_TEST_BACKEND: Optional[str] = "python"


def _selected_tree_core_test_backend() -> str:
    return _TREE_CORE_TEST_BACKEND or envs.SGLANG_UNIFIED_RADIX_TREE_CORE_BACKEND.get()


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
    draft_swa_window: int = 0
    decoder_swa_bounded_replay: bool = False

    # Mamba
    enable_mamba_extra_buffer: bool = False
    mamba_cache_size: int = 20
    mamba_intermediate_size: int = 256
    mamba_n_groups: int = 1
    mamba_num_heads: int = 2
    mamba_head_dim: int = 16
    mamba_state_size: int = 16
    mamba_conv_kernel: int = 4
    enable_int8_mamba_checkpoint: bool = False

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


def _drop_hicache_atexit_pin(cache):
    """init_hicache registers ``cache.shutdown`` with atexit, which pins the
    fixture's device and host pools until process exit; hundreds of HiCache
    fixtures in this suite would otherwise accumulate on the GPU."""
    atexit.unregister(cache.shutdown)


def _write_backup(cache, node, write_back: bool = False) -> int:
    """Back up one node's KV D->H via the tree's build+execute primitives."""
    return cache._execute_and_commit_kv_backup(
        BackupKV(node_ids=cache.tree_core.build_backup_node_ids(node, write_back)),
        write_back,
    )


def _node_key_length(cache, node_id):
    return cache.tree_core.get_node_key_length(node_id)


def _node_token_ids(cache, node_id):
    return cache.tree_core.get_node_token_ids(node_id)


def _node_parent(cache, node_id):
    return cache.tree_core.get_parent_node_id(node_id)


def _node_children(cache, node_id):
    return cache.tree_core.get_child_node_ids(node_id)


def _device_value(cache, node_id, component_type):
    return cache.tree_core.get_component_device_value(node_id, component_type)


def _host_value(cache, node_id, component_type):
    return cache.tree_core.get_component_host_value(node_id, component_type)


def _device_lock_ref(cache, node_id, component_type):
    return cache.tree_core.get_component_device_lock_ref(node_id, component_type)


def _host_lock_ref(cache, node_id, component_type):
    return cache.tree_core.get_component_host_lock_ref(node_id, component_type)


def build_fixture(
    cfg: CacheConfig,
    enable_kv_cache_events: bool = False,
    dynamo_kv_event_format: bool = False,
    enable_session_radix_cache: bool = False,
    tree_page_size: Optional[int] = None,
    mamba_cache_chunk_size: Optional[int] = None,
    component_registry_override: Optional[
        dict[ComponentType, type[TreeComponent]]
    ] = None,
    tree_core_backend: Optional[str] = None,
    kv_event_lora_names: Optional[KvEventLoraNames] = None,
):
    """Create (tree, allocator, req_to_token_pool) from a CacheConfig.

    ``tree_page_size`` stands in for DCP, which widens the tree page past the
    ``page_size`` the rest of the config still sees. It only reaches values
    derived from the tree page: the allocator keeps ``cfg.page_size``, whereas
    DCP sets the two equal, so do not read insert or match behaviour off it.
    """
    server_args = ServerArgs(
        model_path="dummy",
        page_size=cfg.page_size,
        enable_int8_mamba_checkpoint=cfg.enable_int8_mamba_checkpoint,
    )
    # The mamba component reads mamba_cache_chunk_size, whose property otherwise
    # loads the HF config for self.model_path — impossible for the dummy model.
    # Mirror the property's default for a dummy HF config: FLA_CHUNK_SIZE.
    server_args._mamba_cache_chunk_size = (
        max(FLA_CHUNK_SIZE, cfg.page_size)
        if mamba_cache_chunk_size is None
        else mamba_cache_chunk_size
    )
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
            device=device,
        )
        # Tree values reach the allocator as start_pos=0 segments, so only the
        # debug cross-check against torch.unique catches a mis-aligned value.
        with envs.SGLANG_DEBUG_MEMORY_POOL.override(True):
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
        page_size=cfg.page_size if tree_page_size is None else tree_page_size,
        disable=False,
        sliding_window_size=cfg.sliding_window_size,
        draft_swa_window=cfg.draft_swa_window,
        decoder_swa_bounded_replay=cfg.decoder_swa_bounded_replay,
        tree_components=cfg.components,
        enable_mamba_extra_buffer=cfg.enable_mamba_extra_buffer,
        enable_kv_cache_events=enable_kv_cache_events,
        dynamo_kv_event_format=dynamo_kv_event_format,
        kv_event_lora_names=kv_event_lora_names or KvEventLoraNames(),
        enable_session_radix_cache=enable_session_radix_cache,
        eviction_policy=cfg.eviction_policy,
        is_eagle=cfg.is_eagle,
        component_registry_override=component_registry_override,
    )
    requested_backend = tree_core_backend or _selected_tree_core_test_backend()
    selected_backend = resolve_tree_core_backend(requested_backend, cache_init_params)

    def python_inspector_factory(params, components):
        return UnifiedTreeCoreInspector(params, components)

    def rust_inspector_factory(params, _components):
        from sglang.test.kits.unified_radix_cache_unittest.rust_unified_tree_core_inspector import (
            RustUnifiedTreeCoreInspector,
        )

        return RustUnifiedTreeCoreInspector(params)

    with (
        envs.SGLANG_UNIFIED_RADIX_TREE_CORE_BACKEND.override(requested_backend),
        mock.patch.dict(
            _TREE_CORE_REGISTRY,
            {"python": python_inspector_factory, "rust": rust_inspector_factory},
        ),
    ):
        cache = UnifiedRadixCache(params=cache_init_params)
    assert cache._tree_core_backend == selected_backend
    assert isinstance(cache.tree_core, UnifiedTreeCoreInspectionInterface), (
        "The shared unified radix-cache unit suite requires a TreeCore backend "
        "that implements UnifiedTreeCoreInspectionInterface"
    )
    cache.cache_init_params = cache_init_params

    return cache, allocator, req_to_token_pool


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


class UnifiedRadixCacheFixtures:
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
        req.prefix_len = match.device_prefix_len
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

    def _insert(
        self,
        cache,
        allocator,
        req_to_token_pool,
        tokens,
        priority=0,
        *,
        extra_key=None,
        cache_salt=None,
    ):
        """Insert tokens, attaching mamba data when the config has mamba."""
        key = RadixKey(array("q", tokens), extra_key=extra_key, cache_salt=cache_salt)
        value = self._alloc(allocator, len(tokens))
        params = InsertParams(key=key, value=value[: len(key)], priority=priority)
        if self.cfg.has_mamba:
            req = self._make_req(req_to_token_pool)
            params.mamba_value = req.kv.mamba_pool_idx.unsqueeze(0)
        return cache.insert(params)

    def _path_chain(self, cache, node):
        """Return root->node node chain (excluding root)."""
        chain = []
        cur = node
        while cur != cache.root_node_handle():
            chain.append(cur)
            cur = _node_parent(cache, cur)
        chain.reverse()
        return chain

    def _skip_unsupported_hicache_test(self):
        if self.cfg.has_swa and self.cfg.has_mamba:
            self.skipTest("HiCache unit fixture does not support SWA + Mamba stacks")
        return False

    def _simulate_backup(self, cache, node):
        """Simulate D->H backup over the whole root->node path (parent-first)."""
        for ancestor in self._path_chain(cache, node):
            for ct in (ComponentType.FULL, ComponentType.MAMBA, ComponentType.SWA):
                if ct not in self.cfg.components:
                    continue
                value = _device_value(cache, ancestor, ct)
                if value is not None and _host_value(cache, ancestor, ct) is None:
                    cache.tree_core.set_component_host_value_raw(
                        ancestor, ct, value.clone()
                    )
            # A real backup registers duplicate tracking at its ack.
            cache.tree_core.update_duplicate_tracking(ancestor)

    def _simulate_backup_tree(self, cache):
        """Backup all non-root nodes (simulates write-through)."""
        root = cache.root_node_handle()
        stack = [root]
        while stack:
            node = stack.pop()
            if node != root:
                self._simulate_backup(cache, node)
            stack.extend(_node_children(cache, node))

    def _init_hicache(
        self,
        cache,
        *,
        write_policy: str = "write_through",
        storage_backend: Optional[str] = None,
        storage_dir: Optional[str] = None,
        prefetch_threshold: Optional[int] = None,
        prefetch_policy: str = "wait_complete",
        host_memory_mode: str = "cache",
        storage_extra: Optional[dict] = None,
        context_length: Optional[int] = None,
    ):
        storage_extra_config = None
        if storage_backend == "file":
            from sglang.srt.runtime_context import get_parallel

            # The file-backend storage config records TP/PP rank/size. These unit
            # fixtures run without initializing distributed parallel state, so
            # force the local single-rank topology the fixture represents.
            parallel_override = get_parallel().override(
                tp_rank=0, tp_size=1, pp_rank=0, pp_size=1
            )
            parallel_override.__enter__()
            self.addCleanup(parallel_override.__exit__, None, None, None)

            assert storage_dir is not None, "file backend needs a storage_dir"
            # HiCacheFile reads the directory from this env var.
            cm = envs.SGLANG_HICACHE_FILE_BACKEND_STORAGE_DIR.override(storage_dir)
            cm.__enter__()
            self.addCleanup(cm.__exit__, None, None, None)
            extra = {}
            if prefetch_threshold is not None:
                extra["prefetch_threshold"] = prefetch_threshold
            if storage_extra:
                extra.update(storage_extra)
            storage_extra_config = json.dumps(extra) if extra else None

        # See _init_hicache: production kernel IO backend, pinned pools.
        server_args = ServerArgs(
            model_path="dummy",
            page_size=self.cfg.page_size,
            hicache_io_backend="kernel",
            hicache_write_policy=write_policy,
            hicache_storage_backend=storage_backend,
            hicache_storage_backend_extra_config=storage_extra_config,
            hicache_storage_prefetch_policy=prefetch_policy,
            hicache_host_memory_mode=host_memory_mode,
            context_length=context_length,
        )
        # See build_fixture for why _mamba_cache_chunk_size is preset.
        server_args._mamba_cache_chunk_size = max(FLA_CHUNK_SIZE, self.cfg.page_size)
        set_global_server_args_for_scheduler(server_args)
        cache.init_hicache(server_args, cache.cache_init_params)
        self.addCleanup(_drop_hicache_atexit_pin, cache)
        self.addCleanup(cache.release_host_resources)
        cache.write_through_threshold = 1 << 30
        cache.load_back_threshold = 0
        if storage_backend is not None:
            # Unit fixtures size host/device pools equally, which makes the
            # production prefetch capacity limit (host - device) zero.  Keep the
            # L3 tests focused on storage round trips by allowing one fixture
            # worth of prefetch tokens.
            cache.cache_controller.prefetch_capacity_limit = max(
                cache.cache_controller.prefetch_capacity_limit,
                cache.cache_controller.mem_pool_host.size,
            )
            # Background prefetch/backup threads are daemon; stop them per-test.
            self.addCleanup(cache.cache_controller._stop_storage_threads)

    def _build_hicache_fixture(self, tree_core_backend: Optional[str] = None):
        fixture = build_fixture(self.cfg, tree_core_backend=tree_core_backend)
        cache, _, _ = fixture
        self._init_hicache(cache)
        return fixture

    def _backup_node(self, cache, node):
        # Parent-first backup over the whole path: one insert can span several
        # nodes, so a single-node backup would leave an unbacked ancestor.
        backed_up = 0
        for ancestor in self._path_chain(cache, node):
            if cache.tree_core.is_backuped(ancestor):
                continue
            backed_up = _write_backup(cache, ancestor, write_back=True)
            self.assertGreater(backed_up, 0)
        cache.writing_check(write_back=True)
        self.assertTrue(cache.tree_core.is_backuped(node))
        return backed_up

    def _backup_tree(self, cache):
        root = cache.root_node_handle()
        stack = [root]
        while stack:
            node = stack.pop()
            children = _node_children(cache, node)
            stack.extend(reversed(children))
            if node != root:
                self._backup_node(cache, node)

    def _load_back_node(self, cache, node, req=None):
        loaded = cache.load_back(node, req=req)
        self.assertTrue(loaded)
        producer_id = cache.ready_to_load_host_cache()
        self.assertNotEqual(producer_id, -1)
        for ack in list(cache.cache_controller.ack_load_queue):
            ack.finish_event.synchronize()
        cache.loading_check()
        return _device_value(cache, node, ComponentType.FULL)

    def _get_full_kv_pool(self, allocator):
        kv_pool = allocator.get_kvcache()
        return getattr(kv_pool, "full_kv_pool", kv_pool)

    def _fill_full_kv(self, allocator, indices, marker):
        kv_pool = self._get_full_kv_pool(allocator)
        layer_id = kv_pool.start_layer
        # `buf[indices]` is an advanced-index copy — assign, never `fill_()`.
        kv_pool.get_key_buffer(layer_id)[indices] = marker
        kv_pool.get_value_buffer(layer_id)[indices] = marker + 1

    def _snapshot_full_kv(self, allocator, indices):
        kv_pool = self._get_full_kv_pool(allocator)
        layer_id = kv_pool.start_layer
        return (
            kv_pool.get_key_buffer(layer_id)[indices].float().cpu().clone(),
            kv_pool.get_value_buffer(layer_id)[indices].float().cpu().clone(),
        )

    def _build_chain_pages(self, cache, allocator, req_to_token_pool, num_pages):
        """Insert an incremental chain of single-page extensions.

        Returns the chain root-to-leaf. Length may differ from num_pages
        when the radix tree merges or splits nodes.
        """
        seq: list[int] = []
        for i in range(num_pages):
            seq = seq + self._make_seq(1000 * (i + 1), 1)
            self._insert(cache, allocator, req_to_token_pool, seq)
        m = cache.match_prefix(MatchPrefixParams(key=RadixKey(array("q", seq))))
        chain: list = []
        cur = m.last_device_node
        while cur != cache.root_node_handle():
            chain.append(cur)
            cur = _node_parent(cache, cur)
        chain.reverse()
        return chain

    def _release_ongoing_load_back_locks(self, cache):
        for node, lock_params, host_lock_params in list(
            cache.ongoing_load_back.values()
        ):
            cache.dec_lock_ref(node, lock_params)
            cache.dec_host_lock_ref(node, host_lock_params)
        cache.ongoing_load_back.clear()

    def _finish_pending_loads(self, cache):
        producer_id = cache.ready_to_load_host_cache()
        self.assertNotEqual(producer_id, -1)
        for ack in list(cache.cache_controller.ack_load_queue):
            ack.finish_event.synchronize()
        cache.loading_check()

    def _match_tokens_for_chain(self, cache, chain):
        tokens: list[int] = []
        for node in chain:
            tokens.extend(_node_token_ids(cache, node))
        return tokens

    def _set_aux_host_tombstone(self, cache, node, component_type):
        value = _device_value(cache, node, component_type)
        self.assertIsNotNone(value)
        if _host_value(cache, node, component_type) is None:
            cache.tree_core.set_component_host_value_raw(
                node, component_type, value.clone()
            )
        cache.tree_core.set_component_device_value_raw(node, component_type, None)
        if cache.tree_core.is_node_in_device_lru(node, component_type):
            cache.tree_core.remove_node_from_device_lru(node, component_type)
        cache.tree_core.insert_node_into_host_lru(node, component_type)
        cache.tree_core.set_component_evictable_size(
            component_type,
            cache.tree_core.component_evictable_size(component_type) - len(value),
        )

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

        cache, allocator, req_to_token_pool = build_fixture(self.cfg)
        chain = self._build_chain_pages(
            cache, allocator, req_to_token_pool, chain_pages
        )
        if len(chain) <= window_pages:
            self.skipTest("chain collapsed below the window length")
        self._simulate_backup_tree(cache)
        return cache, allocator, req_to_token_pool, chain, window_pages

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

        cache, allocator, req_to_token_pool = self._build_hicache_fixture()
        chain = self._build_chain_pages(
            cache, allocator, req_to_token_pool, chain_pages
        )
        if len(chain) < chain_pages:
            self.skipTest("chain too short")
        self._backup_tree(cache)

        x = chain[-1]
        y = chain[-window_pages]
        n = chain[-(window_pages + 1)]

        cache.tree_core.set_component_device_value_raw(n, ComponentType.SWA, None)
        cache.tree_core.set_component_host_value_raw(n, ComponentType.SWA, None)
        cache.tree_core.set_component_device_value_raw(y, ComponentType.SWA, None)
        # Strip FULL.host on X + intermediates so last_host_node walks past
        # them to Y. Y.FULL untouched preserves the leaf-up evict invariant.
        for node in chain[-(window_pages - 1) :]:
            cache.tree_core.set_component_host_value_raw(node, ComponentType.FULL, None)

        tokens = self._swa_anchor_chain_tokens(len(chain))
        return cache, chain, n, y, x, tokens


class _InsertWalkSuite:
    """Fixture helpers from UnifiedRadixCacheSuite, without inheriting its tests."""

    _rid = 0
    _make_req = UnifiedRadixCacheFixtures._make_req
    _apply_match_to_req = UnifiedRadixCacheFixtures._apply_match_to_req
    _alloc = UnifiedRadixCacheFixtures._alloc
    _insert = UnifiedRadixCacheFixtures._insert
    _init_hicache = UnifiedRadixCacheFixtures._init_hicache
    _build_hicache_fixture = UnifiedRadixCacheFixtures._build_hicache_fixture
    _make_seq = UnifiedRadixCacheFixtures._make_seq
    _skip_unsupported_hicache_test = (
        UnifiedRadixCacheFixtures._skip_unsupported_hicache_test
    )
