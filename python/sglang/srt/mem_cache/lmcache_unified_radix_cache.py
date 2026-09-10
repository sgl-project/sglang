"""Unified radix tree backed by the LMCache multiprocess service.

The class reuses ``UnifiedRadixCache`` only for its device radix tree and
component machinery.  It does not initialize HiCache, does not allocate a
host pool, and does not expose LMCache's internal CPU/remote hierarchy to the
tree.  External hits are loaded into private GPU slots before they are
published into the radix tree.
"""

from __future__ import annotations

import atexit
import hashlib
import logging
from array import array
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Optional

import torch
from lmcache.integration.sglang.unified_lmcache_mp_connector import (
    LMCacheLoadOperation,
    LMCacheLookupOperation,
    LMCacheStoreOperation,
    SGLangKVComponentGroup,
    UnifiedLMCacheMPConnector,
)

from sglang.srt.configs.model_config import AttentionArch, is_deepseek_v4
from sglang.srt.mem_cache.base_prefix_cache import (
    DecLockRefParams,
    EvictParams,
    InitLoadBackParams,
    InsertParams,
    MatchPrefixParams,
    MatchResult,
)
from sglang.srt.mem_cache.radix_cache import RadixKey
from sglang.srt.mem_cache.unified_cache.components import ComponentType
from sglang.srt.mem_cache.unified_radix_cache import NodeId, UnifiedRadixCache

if TYPE_CHECKING:
    from sglang.srt.configs.model_config import ModelConfig
    from sglang.srt.managers.schedule_batch import Req
    from sglang.srt.mem_cache.cache_init_params import CacheInitParams

logger = logging.getLogger(__name__)


@dataclass
class _ExternalFlow:
    key: RadixKey
    lookup: LMCacheLookupOperation
    total_hit: Optional[int] = None
    local_hit_tokens: Optional[int] = None
    load: Optional[LMCacheLoadOperation] = None
    anchor_node: Optional[NodeId] = None
    anchor_lock: Optional[DecLockRefParams] = None
    mamba_value: Optional[torch.Tensor] = None
    allocated_mamba_for_load: bool = False
    request_mamba_value: Optional[torch.Tensor] = None
    allocated_request_mamba_for_load: bool = False
    load_req: Optional[Req] = None
    free_mamba_after_load: bool = False
    loaded_skip_tokens: int = 0
    released_skip_tokens: int = 0
    prefix_published: bool = False
    load_completed: bool = False
    retire_requested: bool = False
    cancelled: bool = False


@dataclass
class _PendingStore:
    operation: LMCacheStoreOperation
    node_id: NodeId
    lock_params: DecLockRefParams


class LMCacheUnifiedRadixCache(UnifiedRadixCache):
    """Unified radix tree with device-direct LMCache MP KV/state I/O."""

    storage_backend_type = "LMCacheMP"

    def __init__(
        self,
        params: CacheInitParams,
        *,
        model_config: ModelConfig,
        tp_size: int,
        tp_rank: int,
        lmcache_config_file: Optional[str],
        forward_stream: Any,
    ) -> None:
        components = tuple(params.tree_components or ())
        if not components or components[0] is not ComponentType.FULL:
            raise NotImplementedError(
                "LMCacheUnifiedRadixCache requires FULL as its base component"
            )
        unsupported = set(components) - {
            ComponentType.FULL,
            ComponentType.SWA,
            ComponentType.MAMBA,
        }
        if unsupported:
            names = ", ".join(sorted(component.name for component in unsupported))
            raise NotImplementedError(
                "LMCacheUnifiedRadixCache does not yet support tree "
                f"components: {names}"
            )
        super().__init__(params)
        self._mamba_component = next(
            (
                component
                for component in self._components_tuple
                if component.component_type is ComponentType.MAMBA
            ),
            None,
        )
        if self._mamba_component is not None:
            if self._mamba_component.int8_ckpt_pool is not None:
                raise NotImplementedError(
                    "LMCache Mamba hybrid support does not yet support int8 "
                    "radix checkpoints"
                )
        kv_groups = self._resolve_registered_groups()
        self._lmcache_component_types = tuple(self.tree_components)
        self.lmcache_connector = UnifiedLMCacheMPConnector(
            config_file=lmcache_config_file,
            model_name=model_config.model_path,
            tp_size=tp_size,
            tp_rank=tp_rank,
            tp_group=params.tp_cache_group,
            pp_size=params.pp_size,
            pp_rank=params.pp_rank,
            pp_group=params.pp_cache_group,
            page_size=self.page_size,
            kv_groups=kv_groups,
            # SGLang classifies DS V4 as MHA for its custom attention backend,
            # but its page-native KV is still replicated across TP like MLA.
            mla_enabled=(
                model_config.attention_arch == AttentionArch.MLA
                or is_deepseek_v4(model_config.hf_config)
            ),
        )
        self._external_flows: dict[str, _ExternalFlow] = {}
        self._forward_stream = forward_stream
        self._pending_stores: list[_PendingStore] = []
        self._pending_store_counts: dict[str, int] = {}
        self._session_finish_requested: set[str] = set()
        self.prefetch_loaded_tokens_by_reqid: dict[str, int] = {}
        self._lmcache_closed = False
        # Kept only for compatibility with the scheduler's existing HiCache
        # prefetch entry point. LMCache does not pass hash-prefix keys.
        self.hicache_storage_pass_prefix_keys = False
        atexit.register(self.shutdown)

    def _lmcache_all_reduce(
        self, tensor: torch.Tensor, op: torch.distributed.ReduceOp
    ) -> None:
        """Reduce across every TP x PP scheduler participating in LMCache.

        UnifiedRadixCache's PP synchronization intentionally lets PP0 decide
        and broadcasts that result to later stages. LMCache is different:
        every PP stage owns distinct layer tensors and performs its own IPC
        transfer, so readiness and failures must include every stage.
        """
        if self.tp_world_size > 1:
            torch.distributed.all_reduce(tensor, op=op, group=self.tp_group)
        if self.pp_size > 1:
            if self.pp_group is None:
                raise RuntimeError("LMCache PP requires a CPU PP process group")
            torch.distributed.all_reduce(tensor, op=op, group=self.pp_group)

    @staticmethod
    def _resolve_pool_tensors(kv_pool) -> tuple[torch.Tensor, ...]:
        kv_buffer = getattr(kv_pool, "kv_buffer", None)
        if kv_buffer is not None:
            if not kv_buffer:
                raise NotImplementedError("LMCache cannot register an empty MLA pool")
            tensors = tuple(
                tensor
                for tensor in kv_buffer
                if tensor.numel() > 0 or not getattr(kv_pool, "use_dsa", False)
            )
            if not tensors:
                raise NotImplementedError("DSA pool has no locally owned KV buffers")
            return tensors

        k_buffer = getattr(kv_pool, "k_buffer", None)
        v_buffer = getattr(kv_pool, "v_buffer", None)
        if not k_buffer or not v_buffer or len(k_buffer) != len(v_buffer):
            raise NotImplementedError(
                f"Unsupported SGLang KV pool for LMCache MP: {type(kv_pool).__name__}"
            )
        if getattr(kv_pool, "kv_cache_layout", "nhd") != "nhd":
            raise NotImplementedError(
                "LMCacheUnifiedRadixCache currently supports NHD MHA pools only"
            )
        return tuple([*k_buffer, *v_buffer])

    @staticmethod
    def _resolve_dsa_indexer_tensors(kv_pool) -> tuple[torch.Tensor, ...]:
        """Return non-empty page-native DSA indexer buffers.

        DSA indexers are FULL-KV sidecars: their block IDs come from the FULL
        page allocator, but each source row already contains a whole page's
        quantized values and scales.
        """
        if not getattr(kv_pool, "use_dsa", False):
            return ()
        buffers = getattr(kv_pool, "index_k_with_scale_buffer", None)
        if buffers is None:
            raise NotImplementedError(
                f"DSA pool {type(kv_pool).__name__} has no indexer buffers"
            )
        tensors = tuple(tensor for tensor in buffers if tensor.numel() > 0)
        if not tensors:
            raise NotImplementedError("DSA pool has no locally owned indexer buffers")
        if any(tensor.dim() != 2 for tensor in tensors):
            shapes = [tuple(tensor.shape) for tensor in tensors]
            raise NotImplementedError(
                f"LMCache MP expects page-native 2-D DSA indexer buffers, got {shapes}"
            )
        return tensors

    @staticmethod
    def _validate_page_native_tensors(
        pool_name: str, tensors: tuple[torch.Tensor, ...]
    ) -> tuple[torch.Tensor, ...]:
        """Validate tensors whose leading row is one complete SGLang page."""
        if not tensors:
            return tensors
        shapes = [tuple(tensor.shape) for tensor in tensors]
        if any(tensor.dim() != 2 for tensor in tensors):
            raise NotImplementedError(
                f"LMCache MP expects 2-D page-native {pool_name} buffers, got {shapes}"
            )
        if any(not tensor.is_contiguous() for tensor in tensors):
            raise NotImplementedError(
                f"LMCache MP requires contiguous page-native {pool_name} buffers"
            )
        block_counts = {tensor.shape[0] for tensor in tensors}
        if len(block_counts) != 1:
            raise ValueError(
                f"DeepSeek V4 {pool_name} buffers expose different page counts: "
                f"{sorted(block_counts)}"
            )
        return tensors

    @classmethod
    def _resolve_dsv4_full_page_tensors(cls, kv_pool) -> tuple[torch.Tensor, ...]:
        """Resolve DS V4 C4/C128 data tied to the FULL page-id space.

        DS V4 has no dense FULL KV tensor.  Its persistent prefix consists of
        compressed C4 KV, the C4 indexer, and compressed C128 KV.  Every source
        buffer row already contains all compressed slots belonging to one
        logical 256-token FULL page, so all of them reuse the FULL block IDs.
        """
        if getattr(kv_pool, "_unified_kv", False):
            raise NotImplementedError(
                "LMCache DeepSeek V4 does not yet support ROCm unified_kv_triton; "
                "its request-scoped SWA ring has no content-stable block-id space"
            )

        c4_pool = getattr(kv_pool, "c4_kv_pool", None)
        if c4_pool is not None and hasattr(
            c4_pool, "full_to_hisparse_device_index_mapping"
        ):
            raise NotImplementedError(
                "LMCache DeepSeek V4 does not yet support HiSparse C4 remapping"
            )

        tensors: list[torch.Tensor] = []
        c4_buffers = getattr(c4_pool, "kv_buffer", None)
        if c4_buffers:
            tensors.extend(tensor for tensor in c4_buffers if tensor.numel() > 0)

        indexer_pool = getattr(kv_pool, "c4_indexer_kv_pool", None)
        if indexer_pool is not None:
            if getattr(indexer_pool, "uses_aiter_fp4_layout", False):
                # AITER stores FP4 payload and scales in separate page-native
                # tensors. Expose each physical buffer as one opaque byte row
                # per page so LMCache can register both without a copy.
                split_buffers = (
                    getattr(indexer_pool, "index_k_payload_buffer", None),
                    getattr(indexer_pool, "index_k_scale_buffer", None),
                )
                if any(not buffers for buffers in split_buffers):
                    raise NotImplementedError(
                        "DeepSeek V4 AITER FP4 indexer is missing payload or scale "
                        "buffers"
                    )
                for buffers in split_buffers:
                    tensors.extend(
                        tensor.view(torch.uint8).reshape(tensor.shape[0], -1)
                        for tensor in buffers
                        if tensor.numel() > 0
                    )
            else:
                buffers = getattr(indexer_pool, "index_k_with_scale_buffer", None)
                if buffers:
                    tensors.extend(tensor for tensor in buffers if tensor.numel() > 0)

        c128_pool = getattr(kv_pool, "c128_kv_pool", None)
        c128_buffers = getattr(c128_pool, "kv_buffer", None)
        if c128_buffers:
            tensors.extend(tensor for tensor in c128_buffers if tensor.numel() > 0)

        resolved = cls._validate_page_native_tensors("FULL sidecar", tuple(tensors))
        if not resolved:
            raise NotImplementedError(
                "DeepSeek V4 pool has no locally owned C4/C128/indexer buffers"
            )
        return resolved

    @classmethod
    def _resolve_dsv4_swa_page_tensors(cls, kv_pool) -> tuple[torch.Tensor, ...]:
        """Resolve DS V4 SWA KV and C4 compressor states by SWA page.

        HiCache treats both C4 state pools as trailing-page sidecars of SWA.
        Re-view each flat state ring as one opaque row per SWA page so it can
        reuse exactly the same SWA block IDs.  C128 state is intentionally not
        included: with the fixed 256-token page it is complete at a cached page
        boundary, matching ``build_deepseek_v4_hicache_stack``.
        """
        swa_pool = getattr(kv_pool, "swa_kv_pool", None)
        if swa_pool is None:
            raise NotImplementedError(
                "LMCache DeepSeek V4 requires the standard page-addressed SWA pool"
            )

        swa_tensors = tuple(
            tensor
            for tensor in getattr(swa_pool, "kv_buffer", ())
            if tensor.numel() > 0
        )
        resolved_swa = cls._validate_page_native_tensors("SWA", swa_tensors)
        if not resolved_swa:
            raise NotImplementedError(
                "DeepSeek V4 pool has no locally owned SWA buffers"
            )
        swa_page_count = resolved_swa[0].shape[0]
        tensors: list[torch.Tensor] = list(resolved_swa)
        for state_pools in (
            getattr(kv_pool, "compress_state_pools", ()),
            getattr(kv_pool, "indexer_compress_state_pools", ()),
        ):
            for state_pool in state_pools:
                if state_pool is None or state_pool.ratio != 4:
                    continue
                state = state_pool.kv_score_buffer.kv_score
                if not state.is_contiguous():
                    raise NotImplementedError(
                        "LMCache DeepSeek V4 requires contiguous C4 state buffers"
                    )
                ring_size = int(state_pool.ring_size)
                if ring_size <= 0:
                    raise ValueError(
                        f"DeepSeek V4 C4 state has invalid ring size {ring_size}"
                    )
                usable_rows = state.shape[0] // ring_size * ring_size
                state_bytes = state.view(torch.uint8).reshape(state.shape[0], -1)
                state_pages = state_bytes[:usable_rows].reshape(
                    usable_rows // ring_size, -1
                )
                if state_pages.shape[0] < swa_page_count:
                    raise ValueError(
                        "DeepSeek V4 C4 state exposes fewer pages than its SWA "
                        f"pool: {state_pages.shape[0]} < {swa_page_count}"
                    )
                # State backing may include extra capacity, but SWA block IDs can
                # address only the page range exposed by the SWA KV pool.
                tensors.append(state_pages[:swa_page_count])

        resolved = cls._validate_page_native_tensors("SWA/state", tuple(tensors))
        if not resolved:
            raise NotImplementedError(
                "DeepSeek V4 pool has no locally owned SWA/state buffers"
            )
        return resolved

    def _resolve_dsv4_registered_groups(self, kv_pool) -> list[SGLangKVComponentGroup]:
        if tuple(self.tree_components) != (ComponentType.FULL, ComponentType.SWA):
            names = [component.name for component in self.tree_components]
            raise NotImplementedError(
                f"LMCache DeepSeek V4 expects FULL/SWA tree components, got {names}"
            )

        full_tensors = self._resolve_dsv4_full_page_tensors(kv_pool)
        swa_tensors = self._resolve_dsv4_swa_page_tensors(kv_pool)
        return [
            SGLangKVComponentGroup(
                name="full",
                kv_tensors=full_tensors,
                sliding_window_size=-1,
                tokens_per_block=self.page_size,
                slots_per_block=self.page_size,
                tensor_rows_per_block=(1,) * len(full_tensors),
            ),
            SGLangKVComponentGroup(
                name="swa",
                kv_tensors=swa_tensors,
                sliding_window_size=self._aligned_swa_window_size(),
                tokens_per_block=self.page_size,
                slots_per_block=self.page_size,
                tensor_rows_per_block=(1,) * len(swa_tensors),
            ),
        ]

    @staticmethod
    def _resolve_mamba_pool_tensors(mamba_pool) -> tuple[torch.Tensor, ...]:
        """Expose Mamba state as zero-copy tensors for LMCache registration.

        Keep this connector-specific adaptation local instead of adding an
        LMCache-facing API to the shared Mamba pool classes.
        """
        unified_buffer = getattr(mamba_pool, "_unified_buffer", None)
        sub_pool_name = getattr(mamba_pool, "_sub_pool_name", None)
        if unified_buffer is not None and sub_pool_name is not None:
            if unified_buffer.anchor_bytes(sub_pool_name) != 0:
                raise NotImplementedError(
                    "LMCache requires the unified Mamba pool to start at its "
                    "backing buffer base"
                )
            entry_bytes = unified_buffer.mamba_spec(sub_pool_name).entry_bytes()
            num_slots = mamba_pool._max_size + 1
            raw = unified_buffer._raw[: num_slots * entry_bytes]
            return (raw.view(num_slots, 1, entry_bytes),)

        tensors: list[torch.Tensor] = []
        for (
            field,
            state_tensor,
            slice_axis,
        ) in mamba_pool._iter_transfer_state_tensors():
            if slice_axis != 0:
                raise NotImplementedError(
                    f"LMCache MP does not support {field} state with slot "
                    f"slice_axis={slice_axis}"
                )
            for layer_idx in range(mamba_pool.num_mamba_layers):
                layer_tensor = state_tensor[layer_idx]
                if not layer_tensor.is_contiguous():
                    raise NotImplementedError(
                        f"LMCache MP requires contiguous {field} state for "
                        f"Mamba layer {layer_idx}"
                    )
                tensors.append(layer_tensor)
        return tuple(tensors)

    @staticmethod
    def _reset_mamba_checkpoint_metadata(mamba_pool, indices: torch.Tensor) -> None:
        """Reset ReplaySSM cursors that are not persisted by LMCache."""
        for name in (
            "replayssm_write_pos",
            "replayssm_cache_base",
            "replayssm_is_flush",
        ):
            value = getattr(mamba_pool, name, None)
            if value is not None:
                value[indices] = 0

    def _resolve_registered_groups(self) -> list[SGLangKVComponentGroup]:
        """Map Unified tree components to LMCache engine KV groups."""
        kv_pool = self.token_to_kv_pool_allocator.get_kvcache()
        from sglang.srt.mem_cache.deepseek_v4_memory_pool import (
            DeepSeekV4TokenToKVPool,
        )

        if isinstance(kv_pool, DeepSeekV4TokenToKVPool):
            return self._resolve_dsv4_registered_groups(kv_pool)

        groups: list[SGLangKVComponentGroup] = []
        for component_type in self.tree_components:
            if component_type is ComponentType.FULL:
                component_pool = getattr(kv_pool, "full_kv_pool", kv_pool)
                tensors = self._resolve_pool_tensors(component_pool)
                tensor_rows_per_block = (self.page_size,) * len(tensors)
                dsa_tensors = self._resolve_dsa_indexer_tensors(component_pool)
                if dsa_tensors:
                    tensors = (*tensors, *dsa_tensors)
                    tensor_rows_per_block += (1,) * len(dsa_tensors)
                groups.append(
                    SGLangKVComponentGroup(
                        name=component_type.name.lower(),
                        kv_tensors=tensors,
                        sliding_window_size=-1,
                        tokens_per_block=self.page_size,
                        slots_per_block=self.page_size,
                        tensor_rows_per_block=tensor_rows_per_block,
                        recurrent_state=False,
                    )
                )
            elif component_type is ComponentType.SWA:
                component_pool = getattr(kv_pool, "swa_kv_pool", None)
                if component_pool is None:
                    raise NotImplementedError(
                        f"SWA component requires an SWA KV sub-pool, got "
                        f"{type(kv_pool).__name__}"
                    )
                tensors = self._resolve_pool_tensors(component_pool)
                groups.append(
                    SGLangKVComponentGroup(
                        name=component_type.name.lower(),
                        kv_tensors=tensors,
                        sliding_window_size=self._aligned_swa_window_size(),
                        tokens_per_block=self.page_size,
                        slots_per_block=self.page_size,
                        tensor_rows_per_block=(self.page_size,) * len(tensors),
                        recurrent_state=False,
                    )
                )
            elif component_type is ComponentType.MAMBA:
                mamba_pool = self.req_to_token_pool.mamba_pool
                checkpoint_grid = self._mamba_component.mamba_checkpoint_grid
                groups.append(
                    SGLangKVComponentGroup(
                        name=component_type.name.lower(),
                        kv_tensors=tuple(
                            tensor.view(tensor.shape[0], 1, -1)
                            for tensor in self._resolve_mamba_pool_tensors(mamba_pool)
                        ),
                        # One state slot is a complete recurrent-state page. It
                        # semantically represents the state after the final
                        # token of this checkpoint interval; it is not one
                        # token-sized fragment of an attention page.
                        sliding_window_size=checkpoint_grid,
                        tokens_per_block=checkpoint_grid,
                        slots_per_block=1,
                        recurrent_state=True,
                    )
                )
            else:
                raise AssertionError(f"Unexpected LMCache component {component_type}")
        return groups

    def _aligned_swa_window_size(self) -> int:
        assert self._sliding_window_size is not None
        return (
            (self._sliding_window_size + self.page_size - 1)
            // self.page_size
            * self.page_size
        )

    def _device_indices_by_group(
        self,
        full_indices: torch.Tensor,
        *,
        mamba_value: Optional[torch.Tensor] = None,
        mamba_transfer_tokens: Optional[int] = None,
    ) -> list[torch.Tensor]:
        """Translate tree FULL ids into each component's physical address space."""
        allocator = self.token_to_kv_pool_allocator
        result: list[torch.Tensor] = []
        for component_type in self._lmcache_component_types:
            if component_type is ComponentType.FULL:
                result.append(allocator.translate_kv_indices_for_transfer(full_indices))
            elif component_type is ComponentType.SWA:
                result.append(allocator.translate_loc_from_full_to_swa(full_indices))
            elif component_type is ComponentType.MAMBA:
                checkpoint_grid = self._mamba_component.mamba_checkpoint_grid
                logical_tokens = (
                    len(full_indices)
                    if mamba_transfer_tokens is None
                    else mamba_transfer_tokens
                )
                if logical_tokens % checkpoint_grid:
                    raise ValueError(
                        "LMCache Mamba transfer range must align to checkpoint "
                        f"grid {checkpoint_grid}, got {logical_tokens} tokens"
                    )
                checkpoint_slots = torch.zeros(
                    logical_tokens // checkpoint_grid,
                    dtype=torch.int64,
                    device=full_indices.device,
                )
                if mamba_value is not None and checkpoint_slots.numel():
                    physical = self.req_to_token_pool.translate_mamba_indices(
                        mamba_value.view(-1)
                    )
                    if physical.numel() != 1:
                        raise ValueError(
                            "LMCache Mamba transfer expects exactly one state slot"
                        )
                    checkpoint_slots[-1] = physical[0]
                result.append(checkpoint_slots)
            else:
                raise AssertionError(f"Unexpected LMCache component {component_type}")
        return result

    def _allocate_external_mamba_slot(self) -> Optional[torch.Tensor]:
        if self._mamba_component is None:
            return None
        allocator = self.req_to_token_pool.mamba_allocator
        slot = allocator.alloc(1)
        if slot is None:
            self.evict(EvictParams(num_tokens=0, mamba_num=1))
            slot = allocator.alloc(1)
        if slot is not None:
            physical = self.req_to_token_pool.translate_mamba_indices(slot)
            self._reset_mamba_checkpoint_metadata(
                self.req_to_token_pool.mamba_pool, physical
            )
        return slot

    def _arm_external_mamba_cow(self, flow: _ExternalFlow, req: Req) -> None:
        """Give a request a mutable state slot and COW from the loaded checkpoint."""
        checkpoint = flow.mamba_value
        if checkpoint is None:
            return
        if req.kv.mamba_pool_idx is None:
            active = self._allocate_external_mamba_slot()
            assert active is not None, "Cannot allocate Mamba request state for LMCache"
            req.kv.mamba_pool_idx = active[0]
            flow.request_mamba_value = active
            flow.allocated_request_mamba_for_load = True
        elif flow.request_mamba_value is None:
            flow.request_mamba_value = req.kv.mamba_pool_idx.reshape(1)
            flow.allocated_request_mamba_for_load = False
        flow.load_req = req
        req.kv.mamba_cow_src_index = checkpoint
        req.kv.mamba_needs_clear = False

    # ------------------------------------------------------------------
    # Lookup + asynchronous retrieve
    # ------------------------------------------------------------------

    @staticmethod
    def _external_cache_salt(
        cache_salt: Optional[str], extra_key: Optional[str]
    ) -> str:
        """Namespace remote entries by both SGLang key dimensions.

        LMCache has one cache-salt field while SGLang's radix key keeps the
        user salt and multimodal/adapter extra key separately. Hashing both
        avoids external hits crossing those namespaces.
        """
        if not extra_key:
            return cache_salt or ""
        payload = f"{cache_salt or ''}\0{extra_key}".encode()
        return hashlib.sha256(payload).hexdigest()

    def is_backuped(self, node_id: NodeId) -> bool:
        # LMCache is external to the tree. Any resident L1 boundary is a valid
        # lookup anchor because lookup keys are rebuilt from the complete token
        # sequence rather than from a host-tree hash chain.
        return True

    def match_prefix(self, params: MatchPrefixParams) -> MatchResult:
        """Expose a completed LMCache lookup as a request-scoped host hit.

        LMCache owns its CPU staging memory, so there is no SGLang host tree to
        match.  Before admission, report the externally ready suffix through
        the existing host-hit fields.  Once ``init_load_back`` has assigned
        private GPU slots, expose those slots directly if admission retries.
        """
        requested_key_len = len(params.key)
        req = params.req
        flow = self._external_flows.get(req.rid) if req is not None else None
        if (
            flow is not None
            and flow.total_hit is not None
            and flow.local_hit_tokens is not None
            and flow.load is None
        ):
            # Every rank must take the same host-hit/init_load_back branch.
            # Restrict the second match to the shortest L1 prefix observed
            # across TP x PP when lookup completed.
            params = MatchPrefixParams(
                key=params.key[: flow.local_hit_tokens],
                cow_mamba=params.cow_mamba,
                req=req,
            )
        elif flow is not None and flow.load is not None and params.cow_mamba:
            # Admission may retry after returning a request-owned active slot.
            # Re-arm COW from LMCache's immutable checkpoint instead of using
            # an unrelated tree checkpoint.
            self._arm_external_mamba_cow(flow, req)
            params = MatchPrefixParams(key=params.key, cow_mamba=False, req=req)
        result = super().match_prefix(params)
        if req is None:
            return result
        if flow is None or flow.total_hit is None or flow.cancelled:
            return result

        total_hit = min(flow.total_hit, requested_key_len, len(flow.key))
        local_hit = len(result.device_indices)
        skip = 0
        if flow.load is not None and not flow.prefix_published:
            # Another request may publish part or all of this prefix while our
            # retrieve is in flight. Record the shadowed private slots before
            # the fully-local early return below. Once this flow starts its own
            # insert, however, the same local prefix is tree-owned and must not
            # be released as if another request had shadowed it.
            skip = max(
                min(local_hit, total_hit) - flow.load.local_hit_tokens,
                0,
            )
            flow.loaded_skip_tokens = max(flow.loaded_skip_tokens, skip)
            if flow.load.result:
                self._release_unused_loaded_slots(flow)
        if total_hit <= local_hit:
            return result

        if flow.load is None:
            external_hit = total_hit - local_hit
            return result._replace(
                last_host_node=result.last_device_node,
                best_match_node=result.last_device_node,
                host_hit_length=external_hit,
                swa_host_hit_length=(
                    min(external_hit, self._aligned_swa_window_size())
                    if self.is_swa_enabled
                    else 0
                ),
                mamba_host_hit_length=(1 if self._mamba_component is not None else 0),
                mamba_branching_seqlen=None,
                full_kv_hit_length=max(result.full_kv_hit_length, total_hit),
            )

        # ``init_load_back`` may have submitted H2D for a request that did not
        # make the final batch. Preserve its private slots across the next
        # admission attempt; the forward stream already waits on this load.
        suffix = flow.load.device_indices[skip : skip + max(total_hit - local_hit, 0)]
        return result._replace(
            device_indices=torch.cat([result.device_indices, suffix]),
            last_host_node=result.last_device_node,
            best_match_node=result.last_device_node,
            host_hit_length=0,
            swa_host_hit_length=0,
            mamba_host_hit_length=0,
            mamba_branching_seqlen=None,
            full_kv_hit_length=max(result.full_kv_hit_length, total_hit),
        )

    def prefetch_from_storage(
        self,
        req_id: str,
        last_host_node_id: NodeId,
        new_input_tokens: list[int],
        last_hash: Optional[str] = None,
        prefix_keys: Optional[list[str]] = None,
        matched_prefix_tokens: Optional[list[int]] = None,
        extra_key: Optional[str] = None,
        cache_salt: Optional[str] = None,
    ) -> None:
        del last_hash, prefix_keys
        if req_id in self._external_flows:
            return
        local_tokens = list(matched_prefix_tokens or [])
        token_ids = local_tokens + list(new_input_tokens)
        anchor_extra_key, anchor_cache_salt = self.tree_core.prefetch_anchor_info(
            last_host_node_id
        )
        extra_key = extra_key or anchor_extra_key
        cache_salt = cache_salt or anchor_cache_salt
        key = RadixKey(
            array("q", token_ids),
            extra_key=extra_key,
            is_bigram=self.tree_core.is_eagle,
            cache_salt=cache_salt,
        ).page_aligned(self.page_size)
        if len(key) == 0:
            return
        token_ids = key.raw_token_ids()[: len(key)]
        lookup = self.lmcache_connector.submit_lookup(
            req_id,
            token_ids,
            local_hit_tokens=min(len(local_tokens), len(key)),
            cache_salt=self._external_cache_salt(cache_salt, extra_key),
        )
        self._external_flows[req_id] = _ExternalFlow(
            key=key,
            lookup=lookup,
        )

    def _allocate_external_slots(self, num_tokens: int) -> Optional[torch.Tensor]:
        allocator = self.token_to_kv_pool_allocator
        if not self.is_swa_enabled:
            if allocator.available_size() < num_tokens:
                self.evict(EvictParams(num_tokens=num_tokens))
            return allocator.alloc(num_tokens)

        # Only the final SWA window is consumed by attention.  Allocate FULL
        # slots for the complete external suffix, but SWA slots only for its
        # page-aligned tail; older SWA block IDs are routed to dummy page 0.
        assert self._sliding_window_size is not None
        swa_tail_tokens = min(
            num_tokens,
            self._aligned_swa_window_size(),
        )
        full_allocator = allocator.full_attn_allocator
        swa_allocator = allocator.swa_attn_allocator
        if full_allocator.available_size() < num_tokens:
            self.evict(EvictParams(num_tokens=num_tokens))
        if swa_allocator.available_size() < swa_tail_tokens:
            self.evict(EvictParams(swa_num_tokens=swa_tail_tokens))

        full_indices = full_allocator.alloc(num_tokens)
        if full_indices is None:
            return None
        if swa_tail_tokens == 0:
            return full_indices

        tail_full_indices = full_indices[-swa_tail_tokens:]
        if hasattr(swa_allocator, "alloc_with_virtual"):
            # Unified-memory SWA uses the same virtual page ids on both sides.
            virtual_pages = torch.unique(tail_full_indices // self.page_size)
            try:
                swa_allocator.alloc_with_virtual(virtual_pages)
            except Exception:
                full_allocator.free(full_indices)
                logger.exception("Failed to allocate unified SWA slots for LMCache")
                return None
        else:
            swa_indices = swa_allocator.alloc(swa_tail_tokens)
            if swa_indices is None:
                full_allocator.free(full_indices)
                return None
            allocator.set_full_to_swa_mapping(tail_full_indices, swa_indices)
        return full_indices

    def _start_external_load(
        self, flow: _ExternalFlow, req: Req
    ) -> Optional[torch.Tensor]:
        assert flow.total_hit is not None
        assert flow.local_hit_tokens is not None
        local_hit = flow.local_hit_tokens
        latest = super().match_prefix(MatchPrefixParams(key=flow.key[:local_hit]))
        total_hit = min(flow.total_hit, len(flow.key))
        if total_hit <= local_hit:
            return self.tree_core.empty_match_result.device_indices

        # Pin the common L1 boundary before allocation because allocator
        # eviction may run below. Every rank intentionally loads from this
        # same boundary even if a particular local tree had matched deeper.
        flow.anchor_node = latest.last_device_node
        flow.anchor_lock = self.inc_lock_ref(flow.anchor_node).to_dec_params()
        num_tokens = total_hit - local_hit
        device_indices = self._allocate_external_slots(num_tokens)
        mamba_value = None
        allocated_mamba_for_load = False
        request_mamba_value = None
        allocated_request_mamba_for_load = False
        if device_indices is not None and self._mamba_component is not None:
            # The externally restored checkpoint is immutable tree data.  Keep
            # it separate from the request's active recurrent state, which the
            # model mutates during the following prefill/decode.
            mamba_value = self._allocate_external_mamba_slot()
            allocated_mamba_for_load = mamba_value is not None
            if req.kv.mamba_pool_idx is None:
                request_mamba_value = self._allocate_external_mamba_slot()
                allocated_request_mamba_for_load = request_mamba_value is not None
            else:
                request_mamba_value = req.kv.mamba_pool_idx.reshape(1)
        allocation_ok = torch.tensor(
            [
                int(
                    device_indices is not None
                    and (
                        self._mamba_component is None
                        or (mamba_value is not None and request_mamba_value is not None)
                    )
                )
            ],
            dtype=torch.int32,
            device="cpu",
        )
        self._lmcache_all_reduce(allocation_ok, torch.distributed.ReduceOp.MIN)
        if not allocation_ok.item():
            if device_indices is not None:
                self.token_to_kv_pool_allocator.free(device_indices)
            if allocated_mamba_for_load and mamba_value is not None:
                self.req_to_token_pool.mamba_allocator.free(mamba_value)
            if allocated_request_mamba_for_load and request_mamba_value is not None:
                self.req_to_token_pool.mamba_allocator.free(request_mamba_value)
            self._release_flow_anchor(flow)
            logger.debug(
                "LMCache retrieve declined for %s: a parallel rank cannot allocate "
                "%d GPU slots",
                flow.lookup.request_id,
                num_tokens,
            )
            return None
        assert device_indices is not None
        flow.mamba_value = mamba_value
        flow.allocated_mamba_for_load = allocated_mamba_for_load
        flow.request_mamba_value = request_mamba_value
        flow.allocated_request_mamba_for_load = allocated_request_mamba_for_load
        flow.load_req = req
        flow.free_mamba_after_load = allocated_mamba_for_load
        flow.loaded_skip_tokens = 0
        if allocated_request_mamba_for_load:
            assert request_mamba_value is not None
            req.kv.mamba_pool_idx = request_mamba_value[0]
            req.kv.mamba_needs_clear = False

        try:
            load_start = local_hit // self.lmcache_connector.chunk_size
            load_start *= self.lmcache_connector.chunk_size
            # Submit while admission can still fall back to ordinary prefill.
            # The stream wait is asynchronous and does not block the scheduler CPU.
            flow.load = self.lmcache_connector.submit_load(
                flow.lookup,
                self._device_indices_by_group(
                    device_indices,
                    mamba_value=mamba_value,
                    mamba_transfer_tokens=total_hit - load_start,
                ),
                local_hit_tokens=local_hit,
                owned_device_indices=device_indices,
                producer_stream=self._forward_stream,
            )
            if not self.lmcache_connector.prepare_load_on_stream(
                flow.load, self._forward_stream
            ):
                raise RuntimeError("LMCache server rejected the retrieve request")
            if mamba_value is not None:
                # prepare_load_on_stream() first makes the forward stream wait
                # for LMCache's H2D completion.  The regular deferred Mamba COW
                # therefore runs afterwards on that same forward stream.
                self._arm_external_mamba_cow(flow, req)
        except Exception:
            self.token_to_kv_pool_allocator.free(device_indices)
            if allocated_mamba_for_load and mamba_value is not None:
                self.req_to_token_pool.mamba_allocator.free(mamba_value)
            if allocated_request_mamba_for_load and request_mamba_value is not None:
                self.req_to_token_pool.mamba_allocator.free(request_mamba_value)
                req.kv.mamba_pool_idx = None
            if req.kv.mamba_cow_src_index is mamba_value:
                req.kv.mamba_cow_src_index = None
            flow.mamba_value = None
            flow.allocated_mamba_for_load = False
            flow.request_mamba_value = None
            flow.allocated_request_mamba_for_load = False
            flow.load_req = None
            flow.free_mamba_after_load = False
            self.dec_lock_ref(flow.anchor_node, flow.anchor_lock)
            flow.anchor_node = None
            flow.anchor_lock = None
            logger.exception(
                "LMCache retrieve submission failed for %s", flow.lookup.request_id
            )
            flow.load = None
            return None
        return device_indices[flow.loaded_skip_tokens :]

    def check_prefetch_progress(self, req_id: str) -> bool:
        flow = self._external_flows.get(req_id)
        if flow is None:
            return True
        if flow.cancelled:
            return False

        if flow.total_hit is None:
            total_hit = self.lmcache_connector.poll_lookup(flow.lookup)
            if total_hit is None:
                return False
            latest = super().match_prefix(MatchPrefixParams(key=flow.key))
            local_hit = torch.tensor(
                [len(latest.device_indices)], dtype=torch.int64, device="cpu"
            )
            self._lmcache_all_reduce(local_hit, torch.distributed.ReduceOp.MIN)
            total_hit = min(total_hit, len(flow.key))
            local_hit_tokens = int(local_hit.item())
            release_end = min(
                total_hit,
                local_hit_tokens
                // self.lmcache_connector.chunk_size
                * self.lmcache_connector.chunk_size,
            )
            if release_end > 0:
                # Every rank advances the same local lookup state, while only
                # the lookup leader sends FREE_LOOKUP_LOCKS to LMCache.
                self.lmcache_connector.free_lookup_locks(
                    req_id, start=0, end=release_end
                )
            if total_hit <= local_hit_tokens:
                self._external_flows.pop(req_id, None)
                self.prefetch_loaded_tokens_by_reqid[req_id] = 0
                self.prefetch_loaded_storage_start_by_reqid.pop(req_id, None)
                return True
            flow.total_hit = total_hit
            flow.local_hit_tokens = local_hit_tokens
            self.prefetch_loaded_tokens_by_reqid[req_id] = total_hit - local_hit_tokens
            self.prefetch_loaded_storage_start_by_reqid[req_id] = local_hit_tokens

        return True

    def _release_flow_anchor(self, flow: _ExternalFlow) -> None:
        if flow.anchor_node is not None and flow.anchor_lock is not None:
            self.dec_lock_ref(flow.anchor_node, flow.anchor_lock)
        flow.anchor_node = None
        flow.anchor_lock = None

    def _release_unused_loaded_slots(self, flow: _ExternalFlow) -> None:
        """Free retrieved slots shadowed by a longer rank-local L1 prefix."""
        assert flow.load is not None
        release_end = min(flow.loaded_skip_tokens, len(flow.load.device_indices))
        if release_end <= flow.released_skip_tokens:
            return
        self.token_to_kv_pool_allocator.free(
            flow.load.device_indices[flow.released_skip_tokens : release_end]
        )
        flow.released_skip_tokens = release_end

    def _finish_failed_load(self, flow: _ExternalFlow) -> None:
        assert flow.load is not None
        if not flow.prefix_published:
            self.token_to_kv_pool_allocator.free(
                flow.load.device_indices[flow.released_skip_tokens :]
            )
        if flow.free_mamba_after_load and flow.mamba_value is not None:
            self.req_to_token_pool.mamba_allocator.free(flow.mamba_value)
            if (
                flow.load_req is not None
                and flow.load_req.kv.mamba_cow_src_index is flow.mamba_value
            ):
                flow.load_req.kv.mamba_cow_src_index = None
            flow.mamba_value = None
        if (
            not flow.cancelled
            and flow.allocated_request_mamba_for_load
            and flow.request_mamba_value is not None
            and flow.load_req is not None
            and flow.load_req.kv.mamba_pool_idx is not None
            and int(flow.load_req.kv.mamba_pool_idx.item())
            == int(flow.request_mamba_value[0].item())
        ):
            self.req_to_token_pool.mamba_allocator.free(flow.request_mamba_value)
            flow.load_req.kv.mamba_pool_idx = None
        flow.request_mamba_value = None
        flow.allocated_request_mamba_for_load = False
        flow.load_req = None
        flow.free_mamba_after_load = False
        self._release_flow_anchor(flow)
        rid = flow.lookup.request_id
        self._external_flows.pop(rid, None)
        if flow.cancelled:
            self.prefetch_loaded_tokens_by_reqid.pop(rid, None)
        else:
            self.prefetch_loaded_tokens_by_reqid[rid] = 0
        self.prefetch_loaded_storage_start_by_reqid.pop(rid, None)

    def _finish_successful_load(self, flow: _ExternalFlow) -> None:
        """Finish local bookkeeping after LMCache has completed the retrieve."""
        assert flow.load is not None and flow.load_completed and flow.load.result
        self._release_unused_loaded_slots(flow)
        # Keep the immutable Mamba checkpoint until the normal request-cache
        # callback publishes the loaded boundary into the radix tree.

    def _finalize_retired_flow(self, flow: _ExternalFlow) -> None:
        """Release a successfully loaded flow after its cache callback retired it."""
        assert flow.load is not None and flow.load_completed and flow.load.result
        self._finish_successful_load(flow)
        if flow.mamba_value is not None:
            # Successful loads normally transfer checkpoint ownership to the
            # tree in _publish_external_loaded_prefix(). Keep this fallback for
            # any request path that retires without inserting.
            if (
                flow.load_req is not None
                and flow.load_req.kv.mamba_cow_src_index is flow.mamba_value
            ):
                flow.load_req.kv.mamba_cow_src_index = None
            self.req_to_token_pool.mamba_allocator.free(flow.mamba_value)
            flow.mamba_value = None
            flow.allocated_mamba_for_load = False
            flow.free_mamba_after_load = False
        flow.request_mamba_value = None
        flow.allocated_request_mamba_for_load = False
        flow.load_req = None
        self._release_flow_anchor(flow)
        rid = flow.lookup.request_id
        self._external_flows.pop(rid, None)

    def _retire_loaded_flow(self, rid: str) -> None:
        """Request retirement without entering a readiness-gated collective."""
        flow = self._external_flows.get(rid)
        if flow is None:
            return
        if flow.load is None:
            if flow.total_hit is None:
                return
            self.lmcache_connector.free_lookup_locks(
                rid,
                start=flow.lookup.lock_start,
                end=flow.total_hit,
            )
            self._external_flows.pop(rid, None)
            return

        flow.retire_requested = True
        if not flow.load_completed:
            return
        if flow.load.result:
            self._finalize_retired_flow(flow)
        else:
            cancelled = flow.cancelled
            self._finish_failed_load(flow)
            if not cancelled:
                logger.warning(
                    "LMCache retrieve failed after admission for request %s", rid
                )

    def _prepare_external_slots_for_insert(self, req: Req) -> None:
        """Restore tree ownership and describe LMCache's sparse SWA suffix.

        Admission temporarily includes loaded slots in ``cache_protected_len``
        so they remain attached to the request.  Unified cache insertion uses
        that field as an ownership boundary, however, and would otherwise
        neither insert nor free an out-of-window SWA page loaded by LMCache.

        LMCache allocates SWA slots only for the trailing window of an external
        hit.  Mark its older FULL-only range as SWA-evicted so insertion creates
        tombstones there even when generic out-of-window freeing is disabled.
        """
        flow = self._external_flows.get(req.rid)
        if flow is None or flow.load is None:
            return
        tree_owned_len = flow.load.local_hit_tokens + flow.loaded_skip_tokens
        req.kv.cache_protected_len = min(req.kv.cache_protected_len, tree_owned_len)
        if self.is_swa_enabled:
            external_tokens = len(flow.load.device_indices)
            swa_missing_end = flow.load.local_hit_tokens + max(
                external_tokens - self._aligned_swa_window_size(), 0
            )
            req.kv.swa_evicted_seqlen = max(req.kv.swa_evicted_seqlen, swa_missing_end)

    def _publish_external_loaded_prefix(self, req: Req, *, token_ids_len: int) -> None:
        """Publish an LMCache-restored unified prefix into the device tree.

        Full/SWA slots loaded by LMCache remain request-owned until this normal
        cache callback.  At that point insert adopts those same slots and the
        independent Mamba checkpoint together.  The request active Mamba slot
        is never inserted because forward has already COW'd the checkpoint into
        it and may have mutated it.
        """
        flow = self._external_flows.get(req.rid)
        # Always restore the real tree-owned boundary first. For Full/SWA-only
        # flows, the regular UnifiedRadixCache callback below will adopt the
        # loaded suffix; Mamba flows publish their exact checkpoint here.
        self._prepare_external_slots_for_insert(req)
        if flow is None or flow.load is None or flow.total_hit is None:
            return
        if flow.mamba_value is None:
            # The regular UnifiedRadixCache callback immediately below adopts
            # the Full/SWA slots. Its internal rematch must not classify those
            # same slots as private copies shadowed by another request.
            flow.prefix_published = True
            return

        total_hit = min(flow.total_hit, len(flow.key))
        if total_hit > token_ids_len:
            raise RuntimeError(
                "LMCache Mamba checkpoint lies beyond the request KV boundary: "
                f"{total_hit}>{token_ids_len} for request {req.rid}"
            )
        if total_hit <= 0:
            return

        prev_prefix_len = min(req.kv.cache_protected_len, total_hit)
        kv_indices = self.req_to_token_pool.req_to_token[
            req.kv.req_pool_idx, :token_ids_len
        ]
        key = flow.key[:total_hit]
        checkpoint = flow.mamba_value
        result = self.insert(
            InsertParams(
                key=key,
                value=kv_indices[:total_hit].to(dtype=torch.int64, copy=True),
                mamba_value=checkpoint,
                prev_prefix_len=prev_prefix_len,
                swa_evicted_seqlen=req.kv.swa_evicted_seqlen,
                chunked=True,
                priority=getattr(req, "priority", 0) or 0,
            )
        )

        # A concurrent request may already have published this exact boundary.
        # In that case insert keeps the canonical checkpoint and ours remains
        # caller-owned, so return it exactly once.
        if result.mamba_exist:
            self.req_to_token_pool.mamba_allocator.free(checkpoint)
        flow.mamba_value = None
        flow.allocated_mamba_for_load = False
        flow.free_mamba_after_load = False

        matched = super().match_prefix(
            MatchPrefixParams(key=key, req=req, cow_mamba=False)
        )
        if len(matched.device_indices) < total_hit:
            raise RuntimeError(
                "LMCache loaded prefix was inserted but is not reusable across "
                f"all UnifiedRadixCache components: {len(matched.device_indices)}"
                f"/{total_hit} tokens for request {req.rid}"
            )
        canonical = matched.device_indices[:total_hit]
        self.req_to_token_pool.write(
            (req.kv.req_pool_idx, slice(prev_prefix_len, total_hit)),
            canonical[prev_prefix_len:],
        )

        # Hand the request lock from its old local boundary to the newly
        # published boundary.  The flow's independent anchor lock is released
        # later by _retire_loaded_flow().
        if req.last_node is not None:
            self._dec_req_lock(req)
        lock_result = self.inc_lock_ref(matched.last_device_node)
        if total_hit < token_ids_len:
            req.prefix_indices = torch.cat(
                [canonical, kv_indices[total_hit:].to(dtype=torch.int64, copy=True)]
            )
        else:
            req.prefix_indices = canonical
        req.kv.cache_protected_len = total_hit
        req.last_node = matched.last_device_node
        req.swa_uuid_for_lock = lock_result.swa_uuid_for_lock
        req.skip_lock_node_ids = lock_result.skip_lock_node_ids
        req.swa_prefix_lock_released = False
        flow.request_mamba_value = None
        flow.allocated_request_mamba_for_load = False
        flow.load_req = None
        flow.prefix_published = True

    def pop_prefetch_loaded_tokens(self, req_id: str) -> int:
        self.prefetch_loaded_storage_start_by_reqid.pop(req_id, None)
        return self.prefetch_loaded_tokens_by_reqid.pop(req_id, 0)

    # ------------------------------------------------------------------
    # Asynchronous store
    # ------------------------------------------------------------------

    def _submit_store(self, req: Req, token_ids: list[int]) -> None:
        key = RadixKey(
            token_ids,
            req.extra_key,
            is_bigram=self.tree_core.is_eagle,
            cache_salt=req.cache_salt,
        ).page_aligned(self.page_size)
        aligned_len = (
            len(key)
            // self.lmcache_connector.chunk_size
            * self.lmcache_connector.chunk_size
        )
        key = key[:aligned_len]
        if len(key) == 0:
            return
        matched = super().match_prefix(MatchPrefixParams(key=key))
        # SWA branching-point caching may publish only the repaired branch in
        # this callback. Store that safe prefix now; a later callback extends it.
        resident_len = torch.tensor(
            [min(len(matched.device_indices), len(key))],
            dtype=torch.int64,
            device="cpu",
        )
        self._lmcache_all_reduce(resident_len, torch.distributed.ReduceOp.MIN)
        resident_len = (
            int(resident_len.item())
            // self.lmcache_connector.chunk_size
            * self.lmcache_connector.chunk_size
        )
        if resident_len == 0:
            logger.debug(
                "LMCache store skipped for %s: radix prefix has no complete "
                "LMCache chunk (%d/%d tokens)",
                req.rid,
                len(matched.device_indices),
                len(key),
            )
            return
        if resident_len < len(key):
            logger.debug(
                "LMCache store for %s is limited to the common resident prefix: "
                "%d/%d tokens",
                req.rid,
                resident_len,
                len(key),
            )
            key = key[:resident_len]
            # The original match may end at a deeper node. Re-match the shortened
            # key so the store lock and any Mamba checkpoint use this exact boundary.
            matched = super().match_prefix(MatchPrefixParams(key=key))
        lock_params = self.inc_lock_ref(matched.last_device_node).to_dec_params()
        mamba_value = (
            self.tree_core.get_component_device_value(
                matched.best_match_node, ComponentType.MAMBA
            )
            if self._mamba_component is not None
            else None
        )
        mamba_is_resident = torch.tensor(
            [int(self._mamba_component is None or mamba_value is not None)],
            dtype=torch.int32,
            device="cpu",
        )
        self._lmcache_all_reduce(mamba_is_resident, torch.distributed.ReduceOp.MIN)
        if not mamba_is_resident.item():
            self.dec_lock_ref(matched.last_device_node, lock_params)
            logger.debug(
                "LMCache store skipped for %s: no Mamba checkpoint at token %d",
                req.rid,
                len(key),
            )
            return
        try:
            operation = self.lmcache_connector.submit_store(
                req.rid,
                key.raw_token_ids()[: len(key)],
                self._device_indices_by_group(
                    matched.device_indices[: len(key)], mamba_value=mamba_value
                ),
                cache_salt=self._external_cache_salt(req.cache_salt, req.extra_key),
            )
        except Exception:
            self.dec_lock_ref(matched.last_device_node, lock_params)
            logger.exception("LMCache store submission failed for %s", req.rid)
            return
        if operation is None:
            self.dec_lock_ref(matched.last_device_node, lock_params)
            return
        self._pending_stores.append(
            _PendingStore(operation, matched.last_device_node, lock_params)
        )
        self._pending_store_counts[req.rid] = (
            self._pending_store_counts.get(req.rid, 0) + 1
        )

    def _request_session_finish(self, rid: str) -> None:
        """End the LMCache session after this request's stores have completed."""
        self._session_finish_requested.add(rid)
        self._finish_session_if_store_idle(rid)

    def _finish_session_if_store_idle(self, rid: str) -> None:
        if (
            rid not in self._session_finish_requested
            or self._pending_store_counts.get(rid, 0) > 0
        ):
            return
        self._session_finish_requested.remove(rid)
        self.lmcache_connector.finish_request(rid)

    def cache_unfinished_req(self, req: Req, chunked: bool = False, **kwargs) -> None:
        self._publish_external_loaded_prefix(req, token_ids_len=len(req.get_fill_ids()))
        super().cache_unfinished_req(req, chunked=chunked, **kwargs)
        self._retire_loaded_flow(req.rid)
        self._submit_store(req, req.get_fill_ids())

    def cache_finished_req(
        self, req: Req, is_insert: bool = True, *, kv_len_to_handle: int, **kwargs
    ) -> None:
        if not is_insert:
            self.release_aborted_request(req.rid)
        else:
            self._publish_external_loaded_prefix(req, token_ids_len=kv_len_to_handle)
        super().cache_finished_req(
            req,
            is_insert=is_insert,
            kv_len_to_handle=kv_len_to_handle,
            **kwargs,
        )
        self._retire_loaded_flow(req.rid)
        if is_insert:
            token_ids = (req.origin_input_ids + req.output_ids)[:kv_len_to_handle]
            self._submit_store(req, token_ids)
            self._request_session_finish(req.rid)

    # ------------------------------------------------------------------
    # Future polling and lifecycle
    # ------------------------------------------------------------------

    def _ready_prefix_count(self, operations) -> int:
        count = 0
        for operation in operations:
            if not operation.query():
                break
            count += 1
        tensor = torch.tensor([count], dtype=torch.int64, device="cpu")
        self._lmcache_all_reduce(tensor, torch.distributed.ReduceOp.MIN)
        return int(tensor.item())

    def check_hicache_events(self) -> None:
        """Poll LMCache retrieve/store futures at the scheduler safe point."""
        load_flows = [
            flow
            for flow in self._external_flows.values()
            if flow.load is not None and not flow.load_completed
        ]
        ready_loads = self._ready_prefix_count([f.load for f in load_flows])
        for flow in load_flows[:ready_loads]:
            assert flow.load is not None
            success = self.lmcache_connector.complete_load(flow.load)
            flow.load_completed = True
            cancelled = flow.cancelled
            if cancelled or not success:
                self._finish_failed_load(flow)
                if not cancelled:
                    logger.warning(
                        "LMCache retrieve failed after admission for request %s",
                        flow.lookup.request_id,
                    )
            elif flow.retire_requested:
                self._finalize_retired_flow(flow)
            else:
                self._finish_successful_load(flow)

        ready_stores = self._ready_prefix_count(
            [pending.operation for pending in self._pending_stores]
        )
        for _ in range(ready_stores):
            pending = self._pending_stores.pop(0)
            rid = pending.operation.request_id
            try:
                self.lmcache_connector.complete_store(pending.operation)
            finally:
                # STORE failure is recoverable, but its source tree node must
                # never remain protected after the operation has completed.
                self.dec_lock_ref(pending.node_id, pending.lock_params)
                remaining = self._pending_store_counts[rid] - 1
                if remaining > 0:
                    self._pending_store_counts[rid] = remaining
                else:
                    self._pending_store_counts.pop(rid)
                self._finish_session_if_store_idle(rid)

    def has_pending_cache_operations(self) -> bool:
        return bool(self._external_flows or self._pending_stores)

    def release_aborted_request(self, rid: str) -> None:
        self.prefetch_loaded_tokens_by_reqid.pop(rid, None)
        self.prefetch_loaded_storage_start_by_reqid.pop(rid, None)
        flow = self._external_flows.get(rid)
        if flow is None:
            self._request_session_finish(rid)
            return
        flow.cancelled = True
        # LMCache writes only the independent checkpoint slot. Generic request
        # cleanup may therefore release the active slot normally; the flow
        # keeps the checkpoint alive until its H2D completion is observed.
        if flow.load is not None and flow.mamba_value is not None:
            flow.free_mamba_after_load = True
            if (
                flow.load_req is not None
                and flow.load_req.kv.mamba_cow_src_index is flow.mamba_value
            ):
                flow.load_req.kv.mamba_cow_src_index = None
        if flow.load is None:
            if flow.total_hit is not None:
                # LOOKUP completed but no RETRIEVE was submitted, so no H2D
                # completion callback will consume its remaining read locks.
                self._retire_loaded_flow(rid)
            else:
                # END_SESSION, issued below, is ordered after the in-flight
                # LOOKUP and releases any locks it may eventually acquire.
                self._external_flows.pop(rid, None)
        elif flow.load_completed:
            self._finish_failed_load(flow)
        else:
            # Completion, including its TP/PP collective, is driven only by
            # check_hicache_events().
            flow.retire_requested = True
        self._request_session_finish(rid)

    def init_load_back(self, params: InitLoadBackParams) -> tuple[torch.Tensor, NodeId]:
        req = params.req
        if req is None:
            return (
                self.tree_core.empty_match_result.device_indices,
                params.best_match_node,
            )
        flow = self._external_flows.get(req.rid)
        if flow is None or flow.total_hit is None or flow.load is not None:
            return (
                self.tree_core.empty_match_result.device_indices,
                params.best_match_node,
            )

        device_indices = self._start_external_load(flow, req)
        if device_indices is None:
            req.storage_hit_length = 0
            req.host_hit_length = 0
            req.swa_host_hit_length = 0
            req.mamba_host_hit_length = 0
            self.prefetch_loaded_tokens_by_reqid.pop(req.rid, None)
            self.prefetch_loaded_storage_start_by_reqid.pop(req.rid, None)
            # No retrieve will consume the remaining lookup locks. Retire the
            # unloaded flow through the normal range-aware release path.
            self._retire_loaded_flow(req.rid)
            return (
                self.tree_core.empty_match_result.device_indices,
                params.best_match_node,
            )
        return device_indices, params.best_match_node

    def ready_to_load_host_cache(self) -> int:
        # the H2D operation has been submitted in `init_load_back`,
        # because if submit retrieve failed, we can fall back to
        # the normal prefill, so there is nothing to do here.
        return -1

    def supports_retraction_backup(self) -> bool:
        # TODO(chunxiaozheng): implement retraction backup
        return False

    def clear_storage_backend(self) -> bool:
        return self.lmcache_connector.clear()

    def reset(self) -> None:
        # ``UnifiedRadixCache.__init__`` calls virtual reset before the connector
        # exists, hence all LMCache cleanup is presence-gated.
        connector = getattr(self, "lmcache_connector", None)
        if connector is not None:
            for flow in list(getattr(self, "_external_flows", {}).values()):
                if flow.load is not None:
                    try:
                        flow.load.future.result(timeout=connector.operation_timeout)
                    except Exception:
                        pass
                    connector.complete_load(flow.load, synchronize=False)
                    flow.load_completed = True
                    if flow.prefix_published:
                        # The tree owns the published suffix; only a private
                        # prefix shadowed before insertion remains ours to free.
                        self._release_unused_loaded_slots(flow)
                    else:
                        self.token_to_kv_pool_allocator.free(
                            flow.load.device_indices[flow.released_skip_tokens :]
                        )
                    if flow.mamba_value is not None:
                        self.req_to_token_pool.mamba_allocator.free(flow.mamba_value)
                        flow.mamba_value = None
                    self._release_flow_anchor(flow)
            for pending in list(getattr(self, "_pending_stores", [])):
                try:
                    pending.operation.future.result(timeout=connector.operation_timeout)
                except Exception:
                    pass
                connector.complete_store(pending.operation, synchronize=False)
                self.dec_lock_ref(pending.node_id, pending.lock_params)
            self._external_flows.clear()
            self._pending_stores.clear()
            self._pending_store_counts.clear()
            self._session_finish_requested.clear()
            self.prefetch_loaded_tokens_by_reqid.clear()
            connector.end_all_sessions()
        super().reset()

    def shutdown(self) -> None:
        if getattr(self, "_lmcache_closed", True):
            return
        self.reset()
        self.lmcache_connector.close()
        self._lmcache_closed = True

    def release_host_resources(self) -> None:
        self.shutdown()
