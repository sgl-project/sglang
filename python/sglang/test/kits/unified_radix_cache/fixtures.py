"""Builders and node accessors for the UnifiedRadixCache unit tests."""

import atexit
from array import array
from typing import Optional
from unittest import mock

from sglang.kernels.ops.attention.fla.chunk_delta_h import CHUNK_SIZE as FLA_CHUNK_SIZE
from sglang.srt.configs.mamba_utils import Mamba2CacheParams, Mamba2StateShape
from sglang.srt.environ import envs
from sglang.srt.mem_cache.allocator import TokenToKVPoolAllocator
from sglang.srt.mem_cache.allocator.swa import SWATokenToKVPoolAllocator
from sglang.srt.mem_cache.base_prefix_cache import (
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
from sglang.srt.server_args import (
    ServerArgs,
    set_global_server_args_for_scheduler,
)
from sglang.srt.utils import get_device
from sglang.test.kits.unified_radix_cache.config import (
    CacheConfig,
)
from sglang.test.kits.unified_radix_cache.unified_tree_core_inspection_interface import (
    UnifiedTreeCoreInspectionInterface,
)
from sglang.test.kits.unified_radix_cache.unified_tree_core_inspector import (
    UnifiedTreeCoreInspector,
)

# The Rust re-run entry point overrides this without changing the process-wide
# production backend selection. Every cache built by build_fixture follows it;
# tests that construct a Python core directly stay Python-only.
TREE_CORE_TEST_BACKEND: Optional[str] = "python"


def selected_tree_core_test_backend() -> str:
    return TREE_CORE_TEST_BACKEND or envs.SGLANG_UNIFIED_RADIX_TREE_CORE_BACKEND.get()


def drop_hicache_atexit_pin(cache):
    """init_hicache registers ``cache.shutdown`` with atexit, which pins the
    fixture's device and host pools until process exit; hundreds of HiCache
    fixtures in one test process would otherwise accumulate on the GPU."""
    atexit.unregister(cache.shutdown)


def write_backup(cache, node, write_back: bool = False) -> int:
    """Back up one node's KV D->H via the tree's build+execute primitives."""
    return cache._execute_and_commit_kv_backup(
        BackupKV(node_ids=cache.tree_core.build_backup_node_ids(node, write_back)),
        write_back,
    )


def node_key_length(cache, node_id):
    return cache.tree_core.get_node_key_length(node_id)


def node_token_ids(cache, node_id):
    return cache.tree_core.get_node_token_ids(node_id)


def node_parent(cache, node_id):
    return cache.tree_core.get_parent_node_id(node_id)


def node_children(cache, node_id):
    return cache.tree_core.get_child_node_ids(node_id)


def node_device_value(cache, node_id, component_type):
    return cache.tree_core.get_component_device_value(node_id, component_type)


def node_host_value(cache, node_id, component_type):
    return cache.tree_core.get_component_host_value(node_id, component_type)


def node_device_lock_ref(cache, node_id, component_type):
    return cache.tree_core.get_component_device_lock_ref(node_id, component_type)


def node_host_lock_ref(cache, node_id, component_type):
    return cache.tree_core.get_component_host_lock_ref(node_id, component_type)


def match_prefix(cache, tokens):
    return cache.match_prefix(MatchPrefixParams(key=RadixKey(array("q", tokens))))


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
    requested_backend = tree_core_backend or selected_tree_core_test_backend()
    selected_backend = resolve_tree_core_backend(requested_backend, cache_init_params)

    def python_inspector_factory(params, components):
        return UnifiedTreeCoreInspector(params, components)

    def rust_inspector_factory(params, _components):
        from sglang.test.kits.unified_radix_cache.rust_unified_tree_core_inspector import (
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
