"""Shared fixtures for the UnifiedRadixCache unit tests."""

import atexit
import json
import shutil
import tempfile
import time
from array import array
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Optional
from unittest import mock

import torch

from sglang.kernels.ops.attention.fla.chunk_delta_h import CHUNK_SIZE as FLA_CHUNK_SIZE
from sglang.srt.configs.mamba_utils import Mamba2CacheParams, Mamba2StateShape
from sglang.srt.environ import envs
from sglang.srt.managers.schedule_batch import (
    Req,
)
from sglang.srt.managers.schedule_policy import PrefillAdder
from sglang.srt.mem_cache.allocator import TokenToKVPoolAllocator
from sglang.srt.mem_cache.allocator.swa import SWATokenToKVPoolAllocator
from sglang.srt.mem_cache.base_prefix_cache import (
    CacheRequestHandle,
    InsertParams,
    MatchPrefixParams,
)
from sglang.srt.mem_cache.cache_init_params import CacheInitParams
from sglang.srt.mem_cache.events import KvEventLoraNames
from sglang.srt.mem_cache.hicache_storage import (
    PoolHitPolicy,
    PoolName,
    PoolTransfer,
)
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

# The Rust re-run entry point overrides this without changing the process-wide
# production backend selection. Every cache built by build_fixture follows it;
# tests that construct a Python core directly stay Python-only.
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


def _aux_storage_key_transfers(cache, node_id):
    transfers = []
    if ComponentType.SWA in cache.tree_components:
        value = _device_value(cache, node_id, ComponentType.SWA)
        if value is not None:
            num_pages = len(value) // cache.page_size
            if num_pages > 0:
                transfers.append(
                    PoolTransfer(
                        name=PoolName.SWA,
                        keys=cache.tree_core.get_hash_values(node_id)[-num_pages:],
                        hit_policy=PoolHitPolicy.TRAILING_PAGES,
                    )
                )
    return transfers or None


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

    def _swa_lru_order(self, cache):
        return cache.tree_core.get_component_device_lru_node_ids(ComponentType.SWA)

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

    # ---------- L3 storage (file backend) helpers ----------

    def _path_chain(self, cache, node):
        """Return root->node node chain (excluding root)."""
        chain = []
        cur = node
        while cur != cache.root_node_handle():
            chain.append(cur)
            cur = _node_parent(cache, cur)
        chain.reverse()
        return chain

    def _write_path_to_l3(self, cache, node):
        """Offload every node on root->node path from host to L3 storage."""
        for n in self._path_chain(cache, node):
            cache.write_backup_storage(n)

    def _ongoing_l3_backups(self, cache):
        """Storage writes in flight (buffer mode tracks them on the pipeline)."""
        if cache.buffer_pipeline is not None:
            return cache.buffer_pipeline.ongoing_backup
        return cache.ongoing_backup

    def _flush_l3_backups(self, cache, timeout: float = 10.0):
        """Wait for backup threads to finish, then drain acks (release locks)."""
        deadline = time.time() + timeout
        while self._ongoing_l3_backups(cache) and time.time() < deadline:
            cache.drain_storage_control_queues()
            if self._ongoing_l3_backups(cache):
                time.sleep(0.01)
        cache.drain_storage_control_queues()
        self.assertFalse(
            self._ongoing_l3_backups(cache), "L3 backups did not complete in time"
        )

    def _run_prefetch_to_completion(self, cache, req_id, timeout: float = 10.0):
        deadline = time.time() + timeout
        while time.time() < deadline:
            # Host memory is reserved (and IO started) by the scheduler-thread
            # drain once the L3 hit count is known, and the buffer-mode fill
            # commits at its H2D ack in loading_check — pump the full event
            # round like the real scheduler loop does.
            cache.check_hicache_events()
            if cache.check_prefetch_progress(req_id):
                # Buffer mode parks a completed fetch as a staged prefetch;
                # consume it like the PrefillAdder would so callers see the
                # span tree-resident.
                if (
                    cache.buffer_pipeline is not None
                    and cache.buffer_pipeline.has_staged(req_id)
                ):
                    self._consume_staged_prefetch(cache, req_id, timeout=timeout)
                return
            time.sleep(0.01)
        self.fail(f"prefetch {req_id} did not complete in time")

    def _consume_staged_prefetch(
        self,
        cache,
        req_id,
        prefix_len=None,
        timeout: float = 10.0,
        *,
        extra_key=None,
        cache_salt=None,
        last_node=None,
        on_dispatched=None,
        return_req: bool = False,
    ):
        """Simulate the PrefillAdder consuming a staged prefetch at admission:
        init_load_back (buffer dispatch: device alloc + queued H2D), the batch
        start_loading flush, then pump until the ack commit lands. Returns
        the spliced device indices (empty on degrade). ``on_dispatched`` runs
        between the two, where the H2D is still in flight."""
        from sglang.srt.mem_cache.base_prefix_cache import InitLoadBackParams

        f = cache.buffer_pipeline.staged_prefetches[req_id]
        if prefix_len is None:
            prefix_len = f.matched_len
        req = mock.Mock()
        req.rid = req_id.rid
        req.cache_request_handle = req_id
        req.extra_key = extra_key
        req.cache_salt = cache_salt
        req.prefix_len = prefix_len
        req.last_node = cache.root_node_handle() if last_node is None else last_node
        joint = cache.match_prefix(
            MatchPrefixParams(
                key=RadixKey(
                    array("q", f.key_tokens), extra_key=extra_key, cache_salt=cache_salt
                )
            )
        )
        req.prefix_len = joint.device_prefix_len
        joint_covers_span = joint.device_prefix_len >= len(f.key_tokens)
        adder = PrefillAdder.__new__(PrefillAdder)
        adder.tree_cache = cache
        with (
            mock.patch.object(
                cache.tree_core,
                "match_full_device_prefix",
                wraps=cache.tree_core.match_full_device_prefix,
            ) as full_match,
            mock.patch.object(
                cache.tree_core,
                "collect_full_device_indices",
                wraps=cache.tree_core.collect_full_device_indices,
            ) as collect_indices,
        ):
            self.assertTrue(
                cache.buffer_pipeline.prepare_staged_prefetch(req),
                "production admission FULL rematch unexpectedly deferred",
            )
            with adder._lock_node(req.last_node):
                loaded = cache.init_load_back(
                    InitLoadBackParams(
                        best_match_node=None,
                        host_hit_length=req.host_hit_length,
                        req=req,
                    )
                )
            if joint_covers_span:
                # A joint match past the staged span is kept without a rematch.
                full_match.assert_not_called()
            else:
                full_match.assert_called_once()
                # A load-back committing here reads the path again for its
                # slot-ownership check.
                collect_indices.assert_called()
        loaded_len, last_node = (0, req.last_node) if loaded is None else loaded
        path = cache.path_device_indices(last_node)
        new_indices = path[len(path) - loaded_len :]
        if on_dispatched is not None:
            on_dispatched()
        # Batch formation flushes the queued load into the batch's producer.
        cache.ready_to_load_host_cache()
        self._pump_hicache_until(
            cache,
            lambda: not cache.buffer_pipeline.ongoing_buffer_load_back,
            "staged-prefetch consumption did not commit",
            timeout=timeout,
        )
        return (new_indices, req) if return_req else new_indices

    def _all_page_hashes(self, cache, node):
        hashes = []
        for n in self._path_chain(cache, node):
            hashes.extend(cache.tree_core.get_hash_values(n))
        return hashes

    def _init_buffer_hicache(
        self,
        cache,
        storage_dir,
        prefetch_policy: str = "wait_complete",
        storage_extra: Optional[dict] = None,
        context_length: Optional[int] = None,
    ):
        if self.cfg.has_mamba:
            self.skipTest(
                "load-back requests here are Mocks; buffer_only Mamba is "
                "covered by TestUnifiedHybridBufferOnlyBitExact"
            )
        self._init_hicache(
            cache,
            storage_backend="file",
            storage_dir=storage_dir,
            prefetch_threshold=1,
            host_memory_mode="buffer_only",
            prefetch_policy=prefetch_policy,
            storage_extra=storage_extra,
            context_length=context_length,
        )

    def _pump_hicache_until(self, cache, cond, msg, timeout: float = 10.0):
        deadline = time.time() + timeout
        while time.time() < deadline:
            cache.check_hicache_events()
            if cond():
                return
            time.sleep(0.01)
        self.fail(msg)

    def _host_avail_sizes(self, cache):
        group = cache.cache_controller.mem_pool_host
        return {entry.name: entry.host_pool.available_size() for entry in group.entries}

    def _storage_exists_count(self, cache, page_hashes, pool_transfers=None):
        """Ground-truth longest-prefix existence count from the backend,
        folded across pools like the prefetch hit query."""
        from sglang.srt.mem_cache.hicache_storage import HiCacheStorageExtraInfo

        backend = cache.cache_controller.storage_backend
        extra_info = HiCacheStorageExtraInfo(prefix_keys=None)
        if pool_transfers:
            result = backend.batch_exists_v2(page_hashes, pool_transfers, extra_info)
            return min(result.kv_hit_pages, len(page_hashes))
        return backend.batch_exists(page_hashes, extra_info)

    def _buffer_backup_and_wait(self, cache, node):
        # Parent-first over the whole path, mirroring the production trigger
        # (_inc_hit_count fires per matched node on the insert walk): the SWA
        # component may have split the leaf at the window boundary.
        pipeline = cache.buffer_pipeline
        chain = self._path_chain(cache, node)
        _write_backup(cache, node)
        for n in chain:
            self.assertIn(n, pipeline.inflight_backup_node_ids)
        self._pump_hicache_until(
            cache,
            lambda: (
                not pipeline.inflight_backup_node_ids and not pipeline.ongoing_backup
            ),
            "buffer backup pipeline did not drain",
        )

    def _produce_buffer_l3(
        self, storage_dir, seq, marker=None, *, extra_key=None, cache_salt=None
    ):
        """Producer tree in buffer mode: insert seq and push it to L3."""
        prod, prod_alloc, prod_rtp = build_fixture(self.cfg)
        self._init_buffer_hicache(prod, storage_dir)
        self._insert(
            prod,
            prod_alloc,
            prod_rtp,
            seq,
            extra_key=extra_key,
            cache_salt=cache_salt,
        )
        key = RadixKey(array("q", seq), extra_key=extra_key, cache_salt=cache_salt)
        leaf = prod.match_prefix(MatchPrefixParams(key=key)).last_device_node
        expected = None
        if marker is not None:
            m = prod.match_prefix(MatchPrefixParams(key=key))
            self._fill_full_kv(
                prod_alloc, prod.path_device_indices(m.last_device_node), marker=marker
            )
            expected = self._snapshot_full_kv(
                prod_alloc, prod.path_device_indices(m.last_device_node)
            )
        self._buffer_backup_and_wait(prod, leaf)
        return leaf, expected

    def _buffer_swa_seq(self, min_pages=4):
        """Sequence long enough for SWA prefetch (one full window + 1)."""
        num_pages = min_pages
        if self.cfg.has_swa:
            sw_pages = (
                self.cfg.sliding_window_size + self.cfg.page_size - 1
            ) // self.cfg.page_size
            num_pages = max(num_pages, sw_pages + 1)
        return self._make_seq(1, num_pages)

    def _drain_buffer_writes(self, cache):
        pipeline = cache.buffer_pipeline
        self._pump_hicache_until(
            cache,
            lambda: (
                not pipeline.inflight_backup_node_ids and not pipeline.ongoing_backup
            ),
            "buffer backup pipeline did not drain",
        )
        self.assertFalse(pipeline.pending_write_queue)
        self.assertFalse(pipeline._queued_span_refs)
        self.assertEqual(pipeline.write_backlog_tokens_, 0)
        self.assertEqual(pipeline.write_staged_tokens_, 0)

    @staticmethod
    def _writes_by_pool(write_storage, extra):
        """Key tuples of the KV writes and of the ``extra`` pool's writes
        among the storage writes recorded by ``write_storage``."""
        kv_writes, extra_writes = set(), set()
        for call in write_storage.call_args_list:
            host_indices, _, hash_value = call.args[:3]
            transfers = call.kwargs.get("extra_pools") or []
            names = [transfer.name for transfer in transfers]
            if hash_value:
                assert extra not in names
                kv_writes.add(tuple(hash_value))
            elif extra in names:
                # The pool's transfers reach storage as one contiguous span.
                assert host_indices.numel() == 0 and names == [extra]
                extra_writes.add(tuple(transfers[0].keys))
        return kv_writes, extra_writes

    def _assert_masked_full_is_reused(self, storage_dir, seq, live_tail: int):
        cons, cons_alloc, cons_rtp = build_fixture(self.cfg)
        self._init_buffer_hicache(cons, storage_dir)
        cons.enable_storage_metrics = True
        cons.storage_metrics_collector = mock.Mock()

        # Masked state: nodes born with live FULL under SWA tombstones (a
        # sibling insert whose SWA ring had slid past the span). live_tail is
        # how far short of the span end the ring stopped.
        live_from = len(seq) - live_tail
        value = self._alloc(cons_alloc, len(seq))
        cons.insert(
            InsertParams(
                key=RadixKey(array("q", seq)),
                value=value[: len(seq)],
                component_evicted_seqlens={ComponentType.SWA: live_from},
            )
        )
        masked = cons.match_prefix(MatchPrefixParams(key=RadixKey(array("q", seq))))
        self.assertEqual(masked.device_prefix_len, 0, "unified match not masked")
        self.assertEqual(masked.full_kv_hit_length, len(seq), "live FULL not resident")
        masked_full = value[: len(seq)].clone()
        # Model the real SWA-eviction path: tombstoned positions have their
        # allocator mapping cleared, while the live trailing portion remains.
        cons_alloc.free_swa(masked_full[:live_from])
        self._fill_full_kv(cons_alloc, masked_full, marker=7)
        masked_kv = self._snapshot_full_kv(cons_alloc, masked_full)
        tail_full = masked_full[live_from:]
        tail_swa = cons_alloc.full_to_swa_index_mapping[tail_full].clone()
        self.assertTrue(bool((tail_swa > 0).all()), "tail SWA not left resident")

        avail0 = self._host_avail_sizes(cons)
        req_id = CacheRequestHandle(f"masked-overlap-{live_tail}", 0)
        cons.prefetch_from_storage(
            req_id, cons.root_node_handle(), array("q", seq), None, None
        )
        # A parked SWA-only retry has no FULL suffix, but must re-pin its prefix.
        with mock.patch.object(cons.swa_kv_pool_host, "alloc", return_value=None):
            self._pump_hicache_until(
                cons,
                lambda: bool(cons.buffer_pipeline.pending_hit_allocs),
                "SWA-only hit was not parked on the staging shortfall",
            )
            self.assertEqual(
                cons.ongoing_prefetch[req_id].operation.storage_hit_count, 0
            )
            self.assertEqual(
                cons.buffer_pipeline._prefetch_prefix_ctx[req_id][0], list(seq)
            )
            self.assertEqual(cons.buffer_pipeline.anchor_locks, {})
            self.assertEqual(cons.buffer_pipeline.anchor_locked_tokens_, 0)
        self._pump_hicache_until(
            cons,
            lambda: (
                cons.check_prefetch_progress(req_id)
                and cons.buffer_pipeline.has_staged(req_id)
            ),
            "prefetch did not stage",
        )
        self.assertIn(req_id, cons.buffer_pipeline.anchor_locks)
        cons.pop_prefetch_loaded_tokens(req_id)
        staged = cons.buffer_pipeline.staged_prefetches[req_id]
        self.assertEqual(staged.matched_len, len(seq))
        self.assertEqual(staged.num_tokens, 0, "resident FULL must not be re-fetched")
        window_tokens = sum(
            len(t.host_indices)
            for t in staged.aux_xfers
            if t.name == PoolName.SWA and t.host_indices is not None
        )
        self.assertGreater(window_tokens, live_tail, "window not partly tombstoned")
        full_avail0 = cons_alloc.full_available_size()
        mapping_before = cons_alloc.full_to_swa_index_mapping[masked_full].clone()
        dispatched = {}

        def capture_swa_mapping():
            load = cons.cache_controller.load_queue[-1]
            swa_xfer = next(t for t in load.pool_transfers if t.name == PoolName.SWA)
            dispatched["device"] = swa_xfer.device_indices.clone()
            full_window = masked_full[-window_tokens:]
            old_window = mapping_before[-window_tokens:]
            expected = old_window.clone()
            expected[old_window <= 0] = swa_xfer.device_indices
            self.assertTrue(
                torch.equal(
                    cons_alloc.full_to_swa_index_mapping[full_window],
                    expected,
                ),
                "admitted request did not fill the missing SWA mappings",
            )

        spliced, req = self._consume_staged_prefetch(
            cons,
            req_id,
            prefix_len=0,
            on_dispatched=capture_swa_mapping,
            return_req=True,
        )
        self.assertEqual(int(spliced.numel()), 0)
        self.assertEqual(req.prefix_len, len(seq), "FULL rematch was lost")
        self.assertEqual(len(dispatched["device"]), window_tokens - live_tail)
        self.assertEqual(cons_alloc.full_available_size(), full_avail0)
        k, v = self._snapshot_full_kv(cons_alloc, masked_full)
        self.assertTrue(torch.equal(k, masked_kv[0]))
        self.assertTrue(torch.equal(v, masked_kv[1]))
        self.assertTrue(
            torch.equal(cons_alloc.full_to_swa_index_mapping[tail_full], tail_swa),
            "live tail changed while dropping the staged overlap",
        )
        self.assertEqual(cons.buffer_pipeline.ongoing_buffer_load_back, {})
        self.assertEqual(cons.buffer_pipeline.anchor_locks, {})
        self.assertEqual(cons.cache_controller.prefetch_tokens_occupied, 0)
        self.assertEqual(self._host_avail_sizes(cons), avail0)
        after = cons.match_prefix(MatchPrefixParams(key=RadixKey(array("q", seq))))
        self.assertEqual(after.full_kv_hit_length, len(seq))
        self.assertEqual(after.device_prefix_len, len(seq))
        # Only the fetched SWA tail made the resident FULL reusable.
        self.assertEqual((req.storage_hit_start, req.storage_hit_length), (0, len(seq)))
        cons.storage_metrics_collector.log_storage_prefetch_unfulfilled_tokens.assert_not_called()
        cons.sanity_check()

    def _assert_sibling_head_is_trimmed(self, storage_dir, seq, window_sized: bool):
        _, (expected_k, expected_v) = self._produce_buffer_l3(
            storage_dir, seq, marker=9
        )

        cons, cons_alloc, cons_rtp = build_fixture(self.cfg)
        self._init_buffer_hicache(cons, storage_dir)
        avail0 = self._host_avail_sizes(cons)

        req_id = CacheRequestHandle(f"growth-trim-{len(seq)}", 0)
        cons.prefetch_from_storage(
            req_id, cons.root_node_handle(), array("q", seq), None, None
        )
        self._pump_hicache_until(
            cons,
            lambda: (
                cons.check_prefetch_progress(req_id)
                and cons.buffer_pipeline.has_staged(req_id)
            ),
            "prefetch did not stage",
        )
        cons.pop_prefetch_loaded_tokens(req_id)

        # Sibling publishes the first page of the span while the hold parks.
        head = seq[: self.cfg.page_size]
        self._insert(cons, cons_alloc, cons_rtp, head)
        sib = cons.match_prefix(MatchPrefixParams(key=RadixKey(array("q", head))))
        self.assertEqual(sib.device_prefix_len, len(head))
        self._fill_full_kv(
            cons_alloc, cons.path_device_indices(sib.last_device_node), marker=3
        )
        head_k, head_v = self._snapshot_full_kv(
            cons_alloc, cons.path_device_indices(sib.last_device_node)
        )

        if window_sized:
            # FULL shrank below a window; the complete SWA staging must survive.
            staged = cons.buffer_pipeline.staged_prefetches[req_id]
            aux_max = max(
                (
                    t.host_indices.numel()
                    for t in staged.aux_xfers
                    if t.host_indices is not None
                ),
                default=0,
            )
            self.assertGreater(aux_max, len(seq) - len(head))

        # The surfaced host hit is the splice-able tail, not the full span.
        req = SimpleNamespace(
            rid=req_id.rid,
            cache_request_handle=req_id,
            prefix_len=sib.device_prefix_len,
            kv=SimpleNamespace(cache_protected_len=sib.device_prefix_len),
        )
        self.assertTrue(cons.buffer_pipeline.prepare_staged_prefetch(req))
        self.assertEqual(req.host_hit_length, len(seq) - len(head))
        self.assertTrue(cons.buffer_pipeline.has_staged(req_id))

        spliced = self._consume_staged_prefetch(cons, req_id, prefix_len=len(head))
        self.assertEqual(int(spliced.numel()), len(seq) - len(head))

        m = cons.match_prefix(MatchPrefixParams(key=RadixKey(array("q", seq))))
        self.assertEqual(m.device_prefix_len, len(seq))
        self.assertTrue(
            torch.equal(
                cons.path_device_indices(m.last_device_node)[: len(head)],
                cons.path_device_indices(sib.last_device_node),
            )
        )
        k, v = self._snapshot_full_kv(
            cons_alloc, cons.path_device_indices(m.last_device_node)[len(head) :]
        )
        self.assertTrue(torch.equal(k, expected_k[len(head) :]))
        self.assertTrue(torch.equal(v, expected_v[len(head) :]))
        hk, hv = self._snapshot_full_kv(
            cons_alloc, cons.path_device_indices(m.last_device_node)[: len(head)]
        )
        self.assertTrue(torch.equal(hk, head_k))
        self.assertTrue(torch.equal(hv, head_v))

        # The whole bounce (trimmed head included) frees at the ack.
        self.assertEqual(self._host_avail_sizes(cons), avail0)
        self.assertEqual(cons.cache_controller.prefetch_tokens_occupied, 0)
        cons.sanity_check()

    def _patch_tp_prefetch_sync(self, cache, drop_swa: bool):
        """Fake all_reduce so _reduce_prefetch_ack runs the tp>1 path."""
        import torch.distributed as dist

        cc = cache.cache_controller

        # Fake _reduce_prefetch_ack to drop SWA pool_hits when drop_swa is True.
        def fake_reduce(ack):
            if drop_swa and ack.pool_hits is not None:
                if PoolName.SWA.value in ack.pool_hits:
                    ack.pool_hits[PoolName.SWA.value] = 0

        p_reduce = mock.patch.object(
            cc, "_reduce_prefetch_ack", side_effect=fake_reduce
        )
        p_reduce.start()
        self.addCleanup(p_reduce.stop)

        # Make all_reduce no-op.  The real all_reduce fails in unit tests with tp_world_size=2.
        p_dist = mock.patch.object(dist, "all_reduce", return_value=None)
        p_dist.start()
        self.addCleanup(p_dist.stop)

    def _swa_host_on_path(self, cache, seq):
        m = cache.match_prefix(MatchPrefixParams(key=RadixKey(array("q", seq))))
        node = m.last_host_node
        while node != cache.root_node_handle():
            if _host_value(cache, node, ComponentType.SWA) is not None:
                return True
            node = _node_parent(cache, node)
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
        cons.prefetch_from_storage(
            req_id, cons.root_node_handle(), array("q", seq), None, None
        )
        self._run_prefetch_to_completion(cons, req_id)

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
        self._consume_prefetch(base, seq, CacheRequestHandle("base", 0))
        if not self._swa_host_on_path(base, seq):
            self.skipTest("fixture does not exercise SWA L3 prefetch")
        return storage_dir, seq

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

    def _fill_mamba_state(self, req_to_token_pool, indices, marker):
        if not self.cfg.has_mamba:
            return
        mamba_indices = indices.reshape(-1)
        mamba_cache = req_to_token_pool.mamba_pool.mamba_cache
        mamba_cache.temporal[:, mamba_indices] = marker
        for offset, conv_buf in enumerate(mamba_cache.conv, start=1):
            conv_buf[:, mamba_indices] = marker + offset

    def _snapshot_mamba_state(self, req_to_token_pool, indices):
        mamba_indices = indices.reshape(-1)
        mamba_cache = req_to_token_pool.mamba_pool.mamba_cache
        return (
            mamba_cache.temporal[:, mamba_indices].float().cpu().clone(),
            [conv[:, mamba_indices].float().cpu().clone() for conv in mamba_cache.conv],
        )

    def _build_internal_mamba_fixture(self, write_policy):
        """HiCache fixture where seq_a's node is INTERNAL (seq_b splits a
        suffix off it) and holds its own mamba state."""
        cache, allocator, req_to_token_pool = build_fixture(self.cfg)
        self._init_hicache(cache, write_policy=write_policy)
        seq_a = self._make_seq(1, 2)
        seq_b = seq_a + self._make_seq(1000, 1)
        self._insert(cache, allocator, req_to_token_pool, seq_a)
        self._insert(cache, allocator, req_to_token_pool, seq_b)
        return cache, req_to_token_pool, seq_a

    def _build_internal_swa_fixture(self, write_policy):
        """HiCache fixture where seq_a's node is INTERNAL and its child spans
        the sliding window, so the leaf backup walk cannot cover it."""
        cache, allocator, req_to_token_pool = build_fixture(self.cfg)
        self._init_hicache(cache, write_policy=write_policy)
        ps = self.cfg.page_size
        tail_pages = max(1, -(-self.cfg.sliding_window_size // ps))
        seq_a = self._make_seq(1, 2)
        seq_b = seq_a + self._make_seq(1000, tail_pages)
        self._insert(cache, allocator, req_to_token_pool, seq_a)
        self._insert(cache, allocator, req_to_token_pool, seq_b)
        return cache, req_to_token_pool, seq_a, seq_b

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
