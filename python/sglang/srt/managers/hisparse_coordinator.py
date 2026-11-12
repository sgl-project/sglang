from __future__ import annotations

import logging
import types
import weakref
from typing import (
    TYPE_CHECKING,
    Dict,
    List,
    NamedTuple,
    Optional,
    Tuple,
    Union,
)

import torch

from sglang.srt.utils import get_device_module, is_hip, is_xpu

if is_xpu():
    from sgl_kernel import (
        load_cache_to_device_buffer_dsv4_mla,
        load_cache_to_device_buffer_mla,
    )

    def copy_cache_planned_mla(*args, **kwargs):
        raise RuntimeError(
            "HiSparse shared-index prefetch is unsupported on XPU: "
            "copy_cache_planned_mla has no AOT sgl_kernel implementation."
        )

    def load_blocks_to_device_buffer_mha(*args, **kwargs):
        raise RuntimeError(
            "MiniMax M3 HiSparse block swap-in is unsupported on XPU: "
            "load_blocks_to_device_buffer_mha has no AOT sgl_kernel implementation."
        )

else:
    from sglang.kernels.ops.kvcache.hisparse import (
        HiSparseSpecState,
        copy_cache_planned_mla,
        load_blocks_to_device_buffer_mha,
        load_cache_to_device_buffer_dsv4_mla,
        load_cache_to_device_buffer_mla,
        load_cache_to_device_buffer_spec_mla,
    )

from sglang.srt.configs.model_config import dsa_layer_skips_topk, is_deepseek_dsa
from sglang.srt.environ import envs
from sglang.srt.managers.schedule_batch import Req
from sglang.srt.managers.scheduler_components.request_list import (
    IndexedReqList,
    ListKind,
    ReqList,
    move,
)
from sglang.srt.mem_cache.allocator.hisparse import (
    DeepSeekV4HiSparseTokenToKVPoolAllocator,
    HiSparseTokenToKVPoolAllocator,
)
from sglang.srt.mem_cache.memory_pool import MiniMaxSparseKVPool, ReqToTokenPool
from sglang.srt.mem_cache.memory_pool_host import DeepSeekV4PagedHostPool
from sglang.srt.mem_cache.pool_host.mha import HiSparseMHATokenToKVPoolHost
from sglang.srt.mem_cache.pool_host.mla import MLATokenToKVPoolHost
from sglang.srt.runtime_context import get_parallel

if TYPE_CHECKING:
    from sglang.kernels.ops.kvcache.hisparse import HiSparseSpecState
    from sglang.srt.managers.schedule_batch import ScheduleBatch

device_module = get_device_module()

_is_hip = is_hip()
_is_xpu = is_xpu()

logger = logging.getLogger(__name__)


class HiSparseAct(NamedTuple):
    start_event: device_module.Event
    finish_event: device_module.Event
    req: Req


class HiSparseTokenStats(NamedTuple):
    device_tokens: int
    device_token_usage: float
    host_tokens: int
    host_token_usage: float


def resolve_shared_index_layers(
    *,
    hf_text_config,
) -> Optional[List[bool]]:
    """Per-layer "reuses the previous layer's top-k index" pattern, or None.

    Mirrors DeepseekV2AttentionMLA's skip_topk derivation (index_topk_pattern /
    index_topk_freq / cli_factor); None when the model has no sharing or the
    prefetch cannot run (pipeline parallelism or the kill-switch).
    """
    if not is_deepseek_dsa(hf_text_config):
        return None
    num_layers = hf_text_config.num_hidden_layers
    cli_factor = getattr(hf_text_config, "cli_factor", 1) or 1
    if cli_factor > 1:
        pattern = [i % cli_factor != 0 for i in range(num_layers)]
    else:
        pattern = [dsa_layer_skips_topk(hf_text_config, i) for i in range(num_layers)]
    if not any(pattern):
        return None
    # Speculative decoding is supported by the HiSparse multi-step swap path;
    # only pipeline parallelism still prevents the shared-index prefetch plan.
    if get_parallel().pp_size != 1:
        logger.warning(
            "HiSparse shared-index prefetch is unsupported under pipeline "
            "parallelism; falling back to synchronous swap-in."
        )
        return None
    if envs.SGLANG_DISABLE_HISPARSE_PREFETCH.get():
        logger.info(
            "HiSparse shared-index prefetch disabled via "
            "SGLANG_DISABLE_HISPARSE_PREFETCH; using synchronous swap-in."
        )
        return None
    return pattern


def _build_prefetch_groups(
    is_shared_index_layer: List[bool],
) -> Tuple[Dict[int, List[int]], List[int]]:
    """Group consecutive shared-index (skip) layers under their anchor layer.

    Returns (groups, slot): anchor layer_id -> ordered skip layers, and each
    skip layer's position in its group (indexes the per-slot prefetch events).
    """
    groups: Dict[int, List[int]] = {}
    slot = [0] * len(is_shared_index_layer)
    anchor = None
    for i, is_shared in enumerate(is_shared_index_layer):
        if not is_shared:
            anchor = i  # compute layer; anchors the skip layers after it
            continue
        assert anchor is not None, (
            f"shared-index (skip) layer {i} has no preceding compute layer; "
            "the model's index-topk pattern is invalid"
        )
        group = groups.setdefault(anchor, [])
        slot[i] = len(group)
        group.append(i)
    return groups, slot


class HiSparseSpecSwapManager:
    """Own HiSparse speculative-swap state and request lifecycle."""

    HASH_MULTIPLIER = 2654435761
    MIN_SCRATCH_CAPACITY = 1024
    NUM_METADATA_VALUES_PER_OCCURRENCE = 5
    NUM_COUNTERS_PER_REQUEST = 4

    def __init__(
        self,
        coordinator: HiSparseCoordinator,
        *,
        num_draft_tokens: int,
    ) -> None:
        self._coordinator = weakref.proxy(coordinator)
        self.enabled = num_draft_tokens > 0
        self.num_draft_tokens = num_draft_tokens
        self.scratch_capacity = 0
        self._scratch_reqs: set[int] = set()

        layer_num = coordinator.mem_pool_device.layer_num
        max_num_req_slots = coordinator.req_to_token_pool.req_to_token.shape[0]
        self.req_to_scratch = torch.empty(
            (layer_num, max_num_req_slots, 0),
            dtype=torch.int32,
            device=coordinator.device,
        )
        self.states: tuple[HiSparseSpecState, ...] = ()
        self.top_k_device_locs: Optional[torch.Tensor] = None
        if not self.enabled:
            return

        if _is_xpu:
            raise ValueError("HiSparse speculative swap is unsupported on XPU.")
        if coordinator.is_dsv4_hisparse:
            raise ValueError("HiSparse spec swap currently supports DSA caches only.")
        if not 2 <= num_draft_tokens <= 4 or coordinator.top_k < 1024:
            raise ValueError(
                "HiSparse spec swap requires 2-4 draft tokens and top_k >= 1024."
            )
        workspace_capacity = num_draft_tokens * coordinator.top_k
        if workspace_capacity > 8192:
            raise ValueError("HiSparse spec swap supports at most 8192 occurrences.")
        if max_num_req_slots > coordinator.device_buffer_size:
            raise ValueError(
                "HiSparse spec request capacity must not exceed device_buffer_size."
            )

        self.scratch_capacity = min(
            workspace_capacity,
            max(
                self.MIN_SCRATCH_CAPACITY,
                workspace_capacity - coordinator.device_buffer_size,
            ),
        )
        hash_size = 1 << max(1, (2 * coordinator.device_buffer_size - 1).bit_length())
        self._cache_index = torch.full(
            (layer_num, max_num_req_slots, 2, hash_size),
            -1,
            dtype=torch.int64,
            device=coordinator.device,
        )
        self._cache_policy = torch.zeros(
            (layer_num, max_num_req_slots + 1, coordinator.device_buffer_size),
            dtype=torch.int32,
            device=coordinator.device,
        )
        metadata_width = max(
            self.NUM_COUNTERS_PER_REQUEST * max_num_req_slots,
            self.NUM_METADATA_VALUES_PER_OCCURRENCE * workspace_capacity,
        )
        self._scratch_state = torch.full(
            (max_num_req_slots + 1, metadata_width),
            -1,
            dtype=torch.int32,
            device=coordinator.device,
        )
        self._scratch_state[0].zero_()
        self.req_to_scratch = torch.zeros(
            (layer_num, max_num_req_slots, self.scratch_capacity),
            dtype=torch.int32,
            device=coordinator.device,
        )
        self.states = tuple(
            HiSparseSpecState(
                cache_index=self._cache_index[layer_id],
                cache_policy=self._cache_policy[layer_id],
                scratch_locs=self.req_to_scratch[layer_id],
                scratch_state=self._scratch_state,
            )
            for layer_id in range(layer_num)
        )

        tokens = torch.arange(
            coordinator.device_buffer_size,
            dtype=torch.int64,
            device=coordinator.device,
        )
        self._hash_slots = ((tokens * self.HASH_MULTIPLIER) & (hash_size - 1)).to(
            torch.long
        )
        self._hash_entries = (tokens << 32) | tokens
        self.top_k_device_locs = torch.full(
            (max_num_req_slots, num_draft_tokens, coordinator.top_k),
            -1,
            dtype=torch.int32,
            device=coordinator.device,
        )

    @property
    def miss_plan_capacity(self) -> int:
        return self._coordinator.top_k * max(1, self.num_draft_tokens)

    def invalidate_cache(self, req_pool_idx: int) -> None:
        if self.enabled:
            self._cache_index[:, req_pool_idx].fill_(-1)

    def reset(self, req_pool_idx: int) -> None:
        if not self.enabled:
            return
        self._cache_index[:, req_pool_idx].fill_(-1)
        self._cache_index[:, req_pool_idx, 0, self._hash_slots] = (
            self._hash_entries.view(1, -1)
        )
        self._cache_policy[:, 0, req_pool_idx].zero_()
        self._cache_policy[:, req_pool_idx + 1].zero_()
        max_num_req_slots = self.req_to_scratch.shape[1]
        self._scratch_state[
            0,
            req_pool_idx : self.NUM_COUNTERS_PER_REQUEST
            * max_num_req_slots : max_num_req_slots,
        ] = 0
        self._scratch_state[req_pool_idx + 1].fill_(-1)

    def allocate_scratch(self, req_pool_idx: int) -> None:
        """Reserve one request's graph-stable swap workspace at admission."""
        if not self.enabled or req_pool_idx in self._scratch_reqs:
            return
        allocator = self._coordinator.token_to_kv_pool_allocator
        if (  # HS_SPECRES v2: admission counted resident requests' extra pages as free; demote to free them
            getattr(self._coordinator, "hs_spec_res", False)
            is True  # a bool; SGLang's unit tests pass MagicMocks
            and allocator.hisparse_attn_allocator.available_size()
            < self.scratch_capacity
        ):
            self._coordinator.hs_spec_reclaims[0] += 1
            self._coordinator.reclaim_device(self.scratch_capacity)
        if isinstance(
            getattr(self._coordinator, "hs_el", None), dict
        ):  # HS_ELASTIC: scratch below the band (a dict; tests pass MagicMocks)
            scratch_locs = allocator._hs_el_alloc_below_or_demote(self.scratch_capacity)
        else:
            scratch_locs = allocator.hisparse_attn_allocator.alloc(
                self.scratch_capacity
            )
        if scratch_locs is None:
            raise RuntimeError(
                "HiSparse spec failed to allocate "
                f"{self.scratch_capacity} scratch slots."
            )
        # Physical slots are shared by all KV layers, but every full-index
        # layer rotates those IDs between persistent cache and scratch
        # independently. Keep per-layer location views so one layer cannot
        # overwrite the next layer's ownership state.
        self.req_to_scratch[:, req_pool_idx].copy_(scratch_locs.to(torch.int32))
        self._scratch_reqs.add(req_pool_idx)

    def free_unrotated_scratch(self, req_pool_idx: int) -> None:
        """Free scratch before any swap kernel has changed its ownership."""
        if not self.enabled or req_pool_idx not in self._scratch_reqs:
            return
        self._coordinator.token_to_kv_pool_allocator.free_hisparse_indices(
            self.req_to_scratch[0, req_pool_idx].clone()
        )
        self.clear_scratch(req_pool_idx)

    def clear_scratch(self, req_pool_idx: int) -> None:
        if not self.enabled or req_pool_idx not in self._scratch_reqs:
            return
        self.req_to_scratch[:, req_pool_idx].zero_()
        self._scratch_reqs.remove(req_pool_idx)

    def extend_owned_locs(
        self, req_pool_idx: int, owned_locs: torch.Tensor
    ) -> torch.Tensor:
        if not self.enabled or req_pool_idx not in self._scratch_reqs:
            return owned_locs
        return torch.cat([owned_locs, self.req_to_scratch[:, req_pool_idx].reshape(-1)])

    def prepare_verify(self, batch: ScheduleBatch) -> None:
        """Bind target-verify KV writes to the side buffer's extra page."""
        coordinator = self._coordinator
        req_pool_indices = batch.req_pool_indices
        verify_cache_locs = batch.out_cache_loc
        start_positions = batch.seq_lens

        if not self.enabled:
            raise RuntimeError("HiSparse spec is not initialized.")
        if self.num_draft_tokens > coordinator.page_size - 1:
            raise ValueError(
                f"HiSparse spec needs {self.num_draft_tokens} verify slots, but the "
                f"extra page has only {coordinator.page_size - 1} usable slots."
            )
        expected_slots = req_pool_indices.numel() * self.num_draft_tokens
        if verify_cache_locs.numel() != expected_slots:
            raise ValueError(
                f"HiSparse verify slot mismatch: expected {expected_slots}, "
                f"got {verify_cache_locs.numel()}."
            )
        reqs = batch.reqs
        if coordinator.hs_spec_res:  # HS_SPECRES: resident rows write their verify KV to their own mirrored slots
            res_rows = coordinator._hs_spec_res_prepare(batch)
            if len(res_rows) == len(reqs):
                return
            if res_rows:
                rs = set(res_rows)
                buf_rows = [i for i in range(len(reqs)) if i not in rs]
                idx = coordinator._gpu_rows(buf_rows)
                reqs = [reqs[i] for i in buf_rows]
                req_pool_indices = req_pool_indices[idx]
                start_positions = start_positions[idx]
                verify_cache_locs = verify_cache_locs.reshape(
                    -1, self.num_draft_tokens
                )[idx].reshape(-1)
                expected_slots = len(buf_rows) * self.num_draft_tokens

        # CPU-side logical reservations cover verify even when overlap lags seq_lens.
        for req in reqs:
            req_pool_idx = req.kv.req_pool_idx
            allocated_len = int(
                coordinator.req_to_host_pool_allocated_len[req_pool_idx]
            )
            if req.kv.kv_allocated_len > allocated_len:
                coordinator.mem_pool_host.alloc_paged_token_slots(
                    coordinator.req_to_host_pool,
                    coordinator.req_to_host_pool_allocated_len,
                    req_pool_idx,
                    allocated_len,
                    req.kv.kv_allocated_len - allocated_len,
                )

        extra_start = coordinator.device_buffer_size + 1
        row_indices = torch.repeat_interleave(req_pool_indices, self.num_draft_tokens)
        offsets = (
            torch.arange(
                expected_slots, dtype=torch.int64, device=req_pool_indices.device
            )
            % self.num_draft_tokens
        )
        token_positions = (
            torch.repeat_interleave(
                start_positions.to(torch.int64), self.num_draft_tokens
            )
            + offsets
        )
        columns = extra_start + offsets
        device_locs = coordinator.req_to_device_buffer[row_indices, columns]
        coordinator.req_device_buffer_tokens[:, row_indices, columns] = (
            token_positions.to(torch.int32).unsqueeze(0)
        )
        coordinator.req_device_buffer_token_locs[:, row_indices, columns] = (
            device_locs.to(torch.int32).unsqueeze(0)
        )
        coordinator.token_to_kv_pool_allocator.full_to_hisparse_device_index_mapping[
            verify_cache_locs
        ] = device_locs

    def _backup_device_locs_to_host(
        self, host_locs: torch.Tensor, device_locs: torch.Tensor
    ) -> None:
        if host_locs.numel() == 0:
            return
        coordinator = self._coordinator
        coordinator.wait_for_pending_backup()
        schedule_stream = device_module.current_stream()
        device_locs = device_locs.contiguous()
        with device_module.stream(coordinator.decode_backup_stream):
            coordinator.decode_backup_stream.wait_stream(schedule_stream)
            if coordinator.decode_producer_stream is not None:
                coordinator.decode_backup_stream.wait_stream(
                    coordinator.decode_producer_stream
                )
            coordinator.mem_pool_host.backup_from_device_all_layer(
                coordinator.mem_pool_device,
                host_locs,
                device_locs,
                io_backend="kernel",
            )
            if host_locs.is_cuda:
                host_locs.record_stream(coordinator.decode_backup_stream)
            if device_locs.is_cuda:
                device_locs.record_stream(coordinator.decode_backup_stream)
            coordinator._backup_done_event.record()
        coordinator._has_pending_backup = True
        coordinator.wait_for_pending_backup()

    def commit_accept_tokens(
        self, batch: ScheduleBatch, accept_indices: torch.Tensor
    ) -> None:
        """Persist accepted target-verify KV in host and the persistent buffer."""
        if batch.forward_mode.is_idle():
            return
        coordinator = self._coordinator
        req_pool_indices = batch.req_pool_indices[:, None]
        verify_cache_locs = batch.out_cache_loc

        if verify_cache_locs.numel() == 0:
            return
        batch_size = req_pool_indices.numel()
        if batch_size == 0 or verify_cache_locs.numel() % batch_size != 0:
            raise ValueError("HiSparse verify cache locations are not request-aligned.")

        reqs = batch.reqs
        seq_lens = batch.seq_lens
        row_ids = None  # HS_SPECRES: batch rows taking the stock commit (None = all)
        all_verify_locs = verify_cache_locs
        if coordinator.hs_spec_res:
            res = coordinator._resident
            res_rows = [i for i, r in enumerate(reqs) if r.kv.req_pool_idx in res]
            coordinator._hs_spec_commit_hook(batch, accept_indices, res_rows)
            if len(res_rows) == len(reqs):
                return  # resident rows: accepted KV is already in its own slot; host copy at demotion
            if res_rows:
                rs = set(res_rows)
                buf_rows = [i for i in range(len(reqs)) if i not in rs]
                row_ids = coordinator._gpu_rows(buf_rows)
                reqs = [reqs[i] for i in buf_rows]
                req_pool_indices = req_pool_indices[row_ids]
                seq_lens = seq_lens[row_ids]
                accept_indices = accept_indices[row_ids]
                verify_cache_locs = all_verify_locs.reshape(batch_size, -1)[
                    row_ids
                ].reshape(-1)

        valid = accept_indices >= 0
        counts = valid.sum(dim=1)
        steps = torch.arange(accept_indices.shape[1], device=accept_indices.device)
        positions = seq_lens[:, None].to(torch.int64) + steps
        mapping = (
            coordinator.token_to_kv_pool_allocator.full_to_hisparse_device_index_mapping
        )
        # Keep a fixed width: boolean indexing and GPU-sized repeat_interleave sync.
        verify_width = all_verify_locs.numel() // batch_size
        rows_full = (  # HS_SPECRES: accept indices index the whole batch's verify tokens
            torch.arange(batch_size, device=accept_indices.device)
            if row_ids is None
            else row_ids
        )
        row_offsets = rows_full[:, None] * verify_width
        source_offsets = torch.where(valid, accept_indices, row_offsets + steps)
        source_device_locs = mapping[all_verify_locs[source_offsets]].clone()
        canonical_locs = coordinator.req_to_token_pool.req_to_token[
            req_pool_indices, positions
        ]

        # Keep hot tokens and each request's newest token in persistent slots.
        cache_mask = valid & (
            (positions < coordinator.device_buffer_size)
            | (steps == counts[:, None] - 1)
        )
        cache_positions = torch.where(
            cache_mask,
            positions.clamp(max=coordinator.device_buffer_size),
            coordinator.device_buffer_size + 1 + steps,
        )
        cache_locs = coordinator.req_to_device_buffer[req_pool_indices, cache_positions]
        destination_locs = torch.where(cache_mask, cache_locs, source_device_locs)
        coordinator.mem_pool_device.transfer_values_on_device(
            dst_indices=destination_locs.flatten(),
            src_indices=source_device_locs.flatten(),
        )

        # Rejected padding goes to this request's reserved, not-yet-visible host slots.
        host_locs = coordinator.req_to_host_pool[req_pool_indices, positions]
        self._backup_device_locs_to_host(
            host_locs.flatten(), destination_locs.flatten()
        )

        # Scalar indexed assignment performs a blocking CPU-to-CUDA copy.
        mapping.index_fill_(0, verify_cache_locs, 0)
        mapping[canonical_locs] = torch.where(cache_mask, cache_locs, 0)
        coordinator.req_device_buffer_tokens[:, req_pool_indices, cache_positions] = (
            torch.where(
                cache_mask,
                positions.to(torch.int32),
                coordinator.req_device_buffer_tokens[
                    :, req_pool_indices, cache_positions
                ],
            )
        )
        coordinator.req_device_buffer_token_locs[
            :, req_pool_indices, cache_positions
        ] = torch.where(
            cache_mask,
            cache_locs.to(torch.int32),
            coordinator.req_device_buffer_token_locs[
                :, req_pool_indices, cache_positions
            ],
        )
        for req in reqs:
            coordinator._skip_first_backup[req.kv.req_pool_idx] = True

    def swap_in(
        self,
        req_pool_indices: torch.Tensor,
        compressed_seq_lens: torch.Tensor,
        top_k_result: torch.Tensor,
        layer_id: int,
        **plan,
    ) -> torch.Tensor:
        if not self.enabled:
            raise RuntimeError("HiSparse spec swap is not initialized.")
        num_reqs = req_pool_indices.size(0)
        num_steps, num_top_k = top_k_result.shape[1:]
        if num_steps != self.num_draft_tokens:
            raise ValueError(
                f"HiSparse spec step mismatch: expected {self.num_draft_tokens}, "
                f"got {num_steps}."
            )
        assert self.top_k_device_locs is not None
        top_k_indices = self.top_k_device_locs[:num_reqs, :num_steps, :num_top_k]
        coordinator = self._coordinator
        res_kw = {}
        if (
            coordinator.enable_resident
        ):  # HS_SPECRES: resident rows are translated, not swapped
            res_kw = dict(
                resident=coordinator.req_resident,
                req_to_token=coordinator.req_to_token_pool.req_to_token,
                full_to_device=coordinator.token_to_kv_pool_allocator.full_to_hisparse_device_index_mapping,
            )
        load_cache_to_device_buffer_spec_mla(
            top_k_tokens=top_k_result,
            device_buffer_tokens=coordinator.req_device_buffer_tokens[layer_id],
            host_cache_locs=coordinator.req_to_host_pool,
            device_buffer_locs=coordinator.req_device_buffer_token_locs[layer_id],
            host_cache=coordinator.mem_pool_host.kv_buffer[layer_id],
            device_buffer=coordinator.mem_pool_device.kv_buffer[layer_id],
            top_k_device_locs=top_k_indices,
            req_pool_indices=req_pool_indices,
            seq_lens=compressed_seq_lens,
            state=self.states[layer_id],
            num_real_reqs=coordinator.num_real_reqs,
            **plan,
            **res_kw,
        )
        return top_k_indices

    def output_locs(self, top_k_result: torch.Tensor, num_reqs: int) -> torch.Tensor:
        if not self.enabled or self.top_k_device_locs is None:
            raise RuntimeError("HiSparse spec swap is not initialized.")
        return self.top_k_device_locs[
            :num_reqs, : top_k_result.shape[1], : top_k_result.shape[2]
        ]


_HS_I64_MAX = (
    2**63 - 1
)  # HS_LEAN: hs_lazy_from of a request whose host copy is complete


class HiSparseCoordinator:
    hs_spec_res = (
        False  # HS_SPECRES (class default for coordinators built without __init__)
    )
    hs_el = None  # HS_ELASTIC (class default)
    _hs_anchor_copy_blocks = 0
    _hs_stats_every = 0
    _hs_step = 0
    _hs_map = None
    _hs_map_init = None
    _hs_group_of_layer = types.MappingProxyType({})
    enable_resident = False
    hs_lazy = False
    hs_lazy_from = None
    _resident = types.MappingProxyType({})
    resident_demotions = 0
    _resident_steps = 0
    hs_excl = False  # HS_EXCL (class default)
    _hs_excl_skip = types.MappingProxyType({})
    _hs_excl_base = types.MappingProxyType({})

    def __init__(
        self,
        req_to_token_pool: ReqToTokenPool,
        token_to_kv_pool_allocator: Union[
            HiSparseTokenToKVPoolAllocator,
            DeepSeekV4HiSparseTokenToKVPoolAllocator,
        ],
        top_k: int,
        device_buffer_size: int,
        device: str,
        tp_group,
        host_to_device_ratio: int = 2,
        swap_in_block_size: int = 960,
        shared_index_layers: Optional[List[bool]] = None,
        num_draft_tokens: int = 0,
    ):
        self.req_to_token_pool = req_to_token_pool
        self.token_to_kv_pool_allocator = token_to_kv_pool_allocator
        self.top_k = top_k
        self.device_buffer_size = device_buffer_size
        self.device = device
        self.swap_in_block_size = swap_in_block_size
        # Timing probe: skip the host->device KV bytes to measure the "IO is
        # free" floor. Produces garbage output; benchmarking only.
        self.skip_io = envs.SGLANG_DEBUG_HISPARSE_SKIP_IO.get()
        if _is_xpu and self.skip_io:
            raise ValueError(
                "SGLANG_DEBUG_HISPARSE_SKIP_IO is unsupported on XPU: the AOT swap-in "
                "ops do not accept skip_io, so timings would silently include the KV "
                "copy."
            )
        self.compress_ratio = self.token_to_kv_pool_allocator.compress_ratio

        kvcache = self.token_to_kv_pool_allocator.get_kvcache()
        self.is_dsv4_hisparse = isinstance(
            self.token_to_kv_pool_allocator, DeepSeekV4HiSparseTokenToKVPoolAllocator
        )
        self.is_m3_hisparse = isinstance(kvcache, MiniMaxSparseKVPool)
        self.mem_pool_device = self.token_to_kv_pool_allocator.get_hisparse_kvcache()
        if self.is_dsv4_hisparse:
            page_size = self.mem_pool_device.page_size
            num_host_pages = (
                self.token_to_kv_pool_allocator.size_full // self.compress_ratio
                + page_size
                - 1
            ) // page_size
            self.mem_pool_host = DeepSeekV4PagedHostPool(
                pool_name="dsv4_hisparse_c4",
                device_buffers=self.mem_pool_device.kv_buffer,
                item_bytes=self.mem_pool_device.bytes_per_page_padded,
                num_host_pages=num_host_pages,
                slot_page_size=page_size,
                layout="layer_first",
            )
            self.item_size_bytes = (
                self.mem_pool_device.kv_cache_total_dim
                * self.mem_pool_device.store_dtype.itemsize
            )
        else:
            assert isinstance(
                self.token_to_kv_pool_allocator, HiSparseTokenToKVPoolAllocator
            )
            if self.is_m3_hisparse:
                assert self.mem_pool_device.head_num == 1, (
                    "MiniMax M3 HiSparse requires one KV head per TP rank, "
                    f"got {self.mem_pool_device.head_num}. Increase the "
                    "tensor-parallel size."
                )
                self.mem_pool_host = HiSparseMHATokenToKVPoolHost(
                    buffer_info=self.mem_pool_device.get_device_pool_infos()[
                        0
                    ].buffer_info,
                    device_capacity_tokens=self.mem_pool_device.size
                    if self.mem_pool_device.host_capacity_tokens is None
                    else self.mem_pool_device.host_capacity_tokens,
                    start_layer=self.mem_pool_device.start_layer,
                    end_layer=self.mem_pool_device.end_layer,
                    host_to_device_ratio=host_to_device_ratio,
                    page_size=self.mem_pool_device.page_size,
                )
                self.item_size_bytes = self.mem_pool_device.bytes_per_token_k
            else:
                # HS_EXCL: resident requests keep no host copy, so the host pool may hold fewer tokens than the logical
                # pool (size = device tokens x ratio + one page, the stock rule, with the ratio derived from the tokens)
                hs_host_ratio = host_to_device_ratio
                hs_host_tokens = envs.SGLANG_HISPARSE_HOST_TOKENS.get()
                if hs_host_tokens > 0:
                    if not self._hs_excl_wanted(num_draft_tokens):
                        raise ValueError(
                            "SGLANG_HISPARSE_HOST_TOKENS needs SGLANG_HISPARSE_EXCLUSIVE=1 with residency (and its "
                            "speculative variant under speculative decoding) and the lazy host backup"
                        )
                    dev_cap = (
                        self.mem_pool_device.size
                        if self.mem_pool_device.host_capacity_tokens is None
                        else self.mem_pool_device.host_capacity_tokens
                    )
                    hs_host_ratio = min(
                        float(host_to_device_ratio), hs_host_tokens / dev_cap
                    )
                self.mem_pool_host = MLATokenToKVPoolHost(
                    buffer_info=self.mem_pool_device.get_device_pool_infos()[
                        0
                    ].buffer_info,
                    device_capacity_tokens=self.mem_pool_device.size
                    if self.mem_pool_device.host_capacity_tokens is None
                    else self.mem_pool_device.host_capacity_tokens,
                    start_layer=self.mem_pool_device.start_layer,
                    end_layer=self.mem_pool_device.end_layer,
                    target_layer_num=self.mem_pool_device.layer_num,
                    host_to_device_ratio=hs_host_ratio,
                    host_size=0,
                    page_size=self.mem_pool_device.page_size,
                    layout="layer_first",
                    override_kv_cache_dim=self.mem_pool_device.kv_cache_dim,
                )
                self.item_size_bytes = self.mem_pool_host.token_stride_size
        self.page_size = self.mem_pool_device.page_size

        max_num_req_slots = req_to_token_pool.req_to_token.shape[0]
        max_context_len = req_to_token_pool.max_context_len
        max_compressed_context_len = (
            max_context_len + self.compress_ratio - 1
        ) // self.compress_ratio

        # to have an extra page for new tokens
        self.padded_buffer_size = (
            self.device_buffer_size + self.mem_pool_device.page_size
        )

        self.req_to_device_buffer = torch.zeros(
            (max_num_req_slots, self.padded_buffer_size),
            dtype=torch.int64,
            device=device,
        )
        self.req_device_buffer_size = torch.zeros(
            max_num_req_slots, dtype=torch.int64, device="cpu"
        )
        self.req_to_host_pool = torch.full(
            (max_num_req_slots, max_compressed_context_len + self.page_size),
            -1,
            dtype=torch.int64,
            device=device,
        )
        self.req_to_host_pool_allocated_len = torch.zeros(
            max_num_req_slots, dtype=torch.int64, device="cpu"
        )

        self.write_staging_stream = device_module.Stream()
        self.decode_backup_stream = device_module.Stream()
        self.ack_staging_queue = IndexedReqList(
            ListKind.HISPARSE_STAGING, req_of=lambda x: x.req
        )
        self.decode_producer_stream = None
        self._backup_done_event = device_module.Event()
        self._has_pending_backup = False

        self.tp_group = tp_group
        self.tp_world_size = torch.distributed.get_world_size(group=self.tp_group)

        # initialize data structures for swap-in kernel
        layer_num = self.mem_pool_device.layer_num
        self.req_device_buffer_tokens = torch.full(
            (layer_num, max_num_req_slots, self.padded_buffer_size),
            -1,
            dtype=torch.int32,
            device=device,
        )
        self.req_device_buffer_token_locs = torch.full(
            (layer_num, max_num_req_slots, self.padded_buffer_size),
            -1,
            dtype=torch.int32,
            device=device,
        )
        self._lru_init = torch.arange(
            self.device_buffer_size, dtype=torch.int16, device=device
        )
        self.lru_slots = (
            self._lru_init.view(1, 1, -1)
            .repeat(layer_num, max_num_req_slots, 1)
            .contiguous()
        )
        self.spec_swap = HiSparseSpecSwapManager(
            self,
            num_draft_tokens=num_draft_tokens,
        )
        if isinstance(self.token_to_kv_pool_allocator, HiSparseTokenToKVPoolAllocator):
            self.token_to_kv_pool_allocator.configure_spec_scratch(
                self.spec_swap.scratch_capacity,
                device_buffer_size=self.padded_buffer_size,
            )
        self._device_buffer_arange_i32 = torch.arange(
            self.device_buffer_size, dtype=torch.int32, device=device
        )

        # Pre-allocated output buffer for swap-in (CUDA-graph safe). MiniMax
        # selects blocks, so its flattened token-slot output can occupy any
        # prefix up to the full device working-set size.
        swap_output_width = (
            self.device_buffer_size if self.is_m3_hisparse else self.top_k
        )
        self.top_k_device_locs_buffer = torch.full(
            (max_num_req_slots, swap_output_width),
            -1,
            dtype=torch.int32,
            device=device,
        )
        self.raw_indices_buffer = torch.full(
            (max_num_req_slots, self.top_k), -1, dtype=torch.int32, device=device
        )
        # Scalar tensor: number of real (non-padded) requests in the batch.
        # Updated before each graph replay so padded blocks early-return.
        self.num_real_reqs = torch.zeros(1, dtype=torch.int32, device=device)

        # HS_RESIDENT: a long request keeps its whole KV in the device pool after staging while there is room, and is
        # demoted to the normal LRU buffer on demand (a new prefill or a resident request's growth needs the space).
        # HS_SPECRES: under speculative decoding only with SGLANG_HISPARSE_SPEC_RESIDENT (and the lazy host backup)
        self.hs_spec_res = bool(
            num_draft_tokens > 0
            and envs.SGLANG_HISPARSE_SPEC_RESIDENT.get()
            and envs.SGLANG_HISPARSE_LAZY_BACKUP.get()
        )
        self.enable_resident = bool(
            envs.SGLANG_HISPARSE_RESIDENT.get()
            and not self.is_dsv4_hisparse
            and not self.is_m3_hisparse
            and (num_draft_tokens == 0 or self.hs_spec_res)
            and not envs.SGLANG_HISPARSE_HOST_MAP.get()
        )
        self.hs_spec_res = bool(self.hs_spec_res and self.enable_resident)
        self._hs_dev_len = {}  # HS_SPECRES: req_pool_idx -> positions with device slots (mirrored logical pages)
        self._hs_spec_steps = [0, 0, 0]  # verify steps; resident rows; buffered rows
        self.hs_spec_reclaims = [
            0,
            0,
        ]  # HS_SPECRES v2: reclaims before a scratch / a buffer allocation
        # HS_ELASTIC: the pool's VMM families and limits (None: static pool)
        self.hs_el = getattr(self.token_to_kv_pool_allocator, "_hs_el", None)
        self.hs_el_stats = [
            0,
            0,
            0,
            0,
            0,
            0,
        ]  # band grows, band shrinks, pages relocated, buffer pages moved, demotions, rounds
        if self.hs_el is not None:
            self.hs_el_grow_band(log=True)
            # v3: physical chunks for the whole reservation now, pooled: no cuMemCreate runs after start-up, when
            # PyTorch's caching allocator may hold the rest of HBM (v2 created the last ~0.1 GiB at the end of a full
            # fill and could fail there: cuMemCreate out of memory, c16 descending ladder)
            el = self.hs_el
            kv, ik = el["kv"], el["ik"]
            # the most ever mapped at once: band + keys <= E, except that keys mapped for the whole logical pool round
            # up to whole 2 MiB chunks per layer (up to one chunk per layer above E, with the band at zero)
            total = max(-(-el["E"] // kv.g), ik.stride // ik.g * ik.layers)
            have = kv.handles() + ik.mapped // ik.g * ik.layers + kv.x.pooled(kv.g)
            if total > have:
                kv.x.reserve_pool(kv.dev, kv.g, total - have)
            el["creates0"] = int(kv.x.creates())
            logger.info(
                "HiSparse elastic: %d physical chunks of %d MiB for the %.1f GiB reservation created up front "
                "(%d pooled); cuMemCreate calls %d",
                total,
                kv.g >> 20,
                el["E"] / 2**30,
                kv.x.pooled(kv.g),
                el["creates0"],
            )
        self._hs_arange_page = torch.arange(
            self.page_size, dtype=torch.int64, device=device
        )
        # v4: every variant of the device-to-device copy kernel now, while HBM has room (num_locs_upper is a constexpr:
        # one variant per power of two up to 256; a variant first needed at a larger batch was loaded with HBM full and
        # failed, c13 b240). Identity copies of the padding slot 0, which no request owns.
        if (
            (self.enable_resident or self.hs_el is not None)
            and not self.is_dsv4_hisparse
            and not self.is_m3_hisparse
        ):
            for n in (1, 2, 3, 5, 9, 16, 17, 32, 33, 48, 65, 80, 129, 144):
                z = torch.zeros(n, dtype=torch.int64, device=device)
                self.mem_pool_device.transfer_values_on_device(z, z)
        if envs.SGLANG_HISPARSE_RESIDENT.get() and not self.enable_resident:
            logger.warning(
                "HiSparse: SGLANG_HISPARSE_RESIDENT ignored (DSv4, MiniMax, speculative decoding or host map)"
            )
        self.req_resident = torch.zeros(
            max_num_req_slots, dtype=torch.int8, device=device
        )
        self._resident: dict = {}  # req_pool_idx -> Req
        self.resident_demotions = 0
        self._resident_steps = 0
        # HS_LEAN: lazy host backup of resident requests (+ lean all-resident prep)
        self.hs_lazy = bool(
            self.enable_resident and envs.SGLANG_HISPARSE_LAZY_BACKUP.get()
        )
        self._hs_staged_len = {}  # req_pool_idx -> tokens backed up by staging (the host copy's prefix)
        self.hs_lazy_from = None  # first position without a host copy, per request slot (resident requests)
        if self.hs_lazy:
            self.hs_lazy_from = torch.full(
                (max_num_req_slots,), _HS_I64_MAX, dtype=torch.int64, device=device
            )
            logger.info(
                "HiSparse: lazy host backup of resident requests (at demotion), lean all-resident decode prep"
            )
        self.hs_lazy_backups = 0
        self.hs_lazy_tokens = 0
        # HS_EXCL: a request that will stay resident gets no host copy at staging (no host slots, no copy); a demotion
        # copies its whole KV. Short prompts are staged as before (waiting while the host pool has no room). Demotions,
        # the admission count and the decode check count only free host slots.
        self.hs_excl = bool(self.hs_lazy and envs.SGLANG_HISPARSE_EXCLUSIVE.get())
        self._hs_excl_skip = {}  # req_pool_idx -> prefill length of a request staged without a host copy
        self._hs_excl_base = {}  # req_pool_idx -> device-only prefix of a resident request (validation hooks)
        self._hs_excl_margin = max(0, envs.SGLANG_HISPARSE_EXCL_HOST_MARGIN.get())
        # stagings without a copy; tokens copied by whole-request demotions; demotion picks the host limited (or found
        # none); decode checks failed on host slots; short prompts that waited for host slots
        self.hs_excl_stats = [0, 0, 0, 0, 0]
        self._hs_host_len_np = (
            self.req_to_host_pool_allocated_len.numpy()
        )  # a view: the CPU tensor's own memory
        if self.hs_excl != (
            self._hs_excl_wanted(num_draft_tokens)
            and not self.is_dsv4_hisparse
            and not self.is_m3_hisparse
        ):
            raise RuntimeError(
                "HiSparse exclusive residency: switch and pool sizing disagree"
            )
        if self.hs_excl:
            logger.info(
                "HiSparse exclusive residency: resident requests keep no host copy; host pool %d tokens for %d "
                "logical tokens; admission margin %d host slots",
                self.mem_pool_host.size,
                self.token_to_kv_pool_allocator.size_full,
                self._hs_excl_margin,
            )
        elif envs.SGLANG_HISPARSE_EXCLUSIVE.get():
            logger.warning(
                "HiSparse: SGLANG_HISPARSE_EXCLUSIVE ignored (needs residency and the lazy backup)"
            )
        self._hs_lean = [
            0,
            0,
        ]  # decode steps prepared by the lean all-resident path, by the general path
        if self.hs_spec_res:
            logger.info(
                "HiSparse: resident requests under speculative decoding (mirrored device pages, lazy backup)"
            )
        if self.enable_resident:
            self.token_to_kv_pool_allocator.set_reclaimer(self)
            logger.info(
                "HiSparse: resident requests on (whole KV in the device pool while it fits; demoted on demand to the "
                "%d-token buffer)",
                self.device_buffer_size,
            )

        # CPU flag: True means "skip backup on the next decode step" because
        # staging already backed up all prefill tokens.  Cleared after one step.
        self._skip_first_backup = [False] * max_num_req_slots

        self._init_shared_index_prefetch(
            shared_index_layers=shared_index_layers,
            layer_num=layer_num,
            max_num_req_slots=max_num_req_slots,
        )
        # HS_PATCH: knobs and geometry log
        self._hs_anchor_copy_blocks = envs.SGLANG_HISPARSE_ANCHOR_COPY_BLOCKS.get()
        self._hs_stats_every = envs.SGLANG_HISPARSE_STATS_EVERY.get()
        self._hs_step = 0
        if self._hs_anchor_copy_blocks > 0 and not self.enable_prefetch:
            logger.warning(
                "HS_PATCH: SGLANG_HISPARSE_ANCHOR_COPY_BLOCKS needs the shared-index "
                "prefetch (miss plan buffers); ignored"
            )
            self._hs_anchor_copy_blocks = 0
        try:
            dev_alloc = self.token_to_kv_pool_allocator.hisparse_attn_allocator
            logger.info(
                "HiSparse geometry: device pool %d tokens, logical/host pool %d tokens, "
                "host pool bytes %.1f GiB (item %d B x %d layers), top_k %d, device buffer %d (+%d page), "
                "layers %d, prefetch %s (%d anchors), prefetch blocks %d, anchor copy blocks %d, "
                "swap_in_block_size %d, skip_io %s",
                dev_alloc.size,
                self.token_to_kv_pool_allocator.size_full,
                self.mem_pool_host.size * self.item_size_bytes * layer_num / 2**30,
                self.item_size_bytes,
                layer_num,
                self.top_k,
                self.device_buffer_size,
                self.page_size,
                layer_num,
                self.enable_prefetch,
                len(self._prefetch_groups) if self.enable_prefetch else 0,
                getattr(self, "_prefetch_copy_blocks", 0),
                self._hs_anchor_copy_blocks,
                self.swap_in_block_size,
                self.skip_io,
            )
        except Exception as e:  # logging only
            logger.warning("HiSparse geometry log failed: %r", e)
        # HS_PATCH v4: persistent host-slot -> buffer-slot map per anchor group (needs the shared-index prefetch
        # so that skip layers share the anchor's buffer; DSA MLA pools only)
        self._hs_map = None
        self._hs_map_init = None
        self._hs_group_of_layer = {}
        if (
            envs.SGLANG_HISPARSE_HOST_MAP.get()
            and self.enable_prefetch
            and not self.is_dsv4_hisparse
            and not self.is_m3_hisparse
            and not _is_xpu
        ):
            anchors = [
                l for l in range(layer_num) if not self._is_shared_index_layer[l]
            ]
            self._hs_group_of_layer = {l: i for i, l in enumerate(anchors)}
            n_slots = int(self.mem_pool_host.size) + self.page_size + 1
            self._hs_map = torch.full(
                (len(anchors), n_slots), -1, dtype=torch.int16, device=device
            )
            self._hs_map_init = torch.zeros(
                (len(anchors), max_num_req_slots), dtype=torch.int8, device=device
            )
            logger.info(
                "HiSparse host map: %d anchor groups x %d host slots, %.1f MiB",
                len(anchors),
                n_slots,
                self._hs_map.numel() * 2 / 2**20,
            )

    def _init_shared_index_prefetch(
        self,
        shared_index_layers: Optional[List[bool]],
        layer_num: int,
        max_num_req_slots: int,
    ) -> None:
        """Set up the plan-then-IO prefetch for shared-index (IndexShare) models:
        the anchor's kernel records its miss plan and skip layers replay it on
        `prefetch_stream`, overlapping their IO with the intervening compute."""
        if shared_index_layers is not None and len(shared_index_layers) != layer_num:
            # Attention-layer count differs from num_hidden_layers (e.g. Longcat
            # doubles it): pattern would be misindexed, fall back to synchronous.
            logger.warning(
                "HiSparse shared-index prefetch disabled: pattern length %d != "
                "KV pool layer_num %d; using synchronous swap-in.",
                len(shared_index_layers),
                layer_num,
            )
            shared_index_layers = None
        if shared_index_layers is not None and _is_xpu:
            logger.warning(
                "HiSparse shared-index prefetch disabled on XPU: "
                "copy_cache_planned_mla is unavailable; using synchronous swap-in."
            )
            shared_index_layers = None
        self._is_shared_index_layer = list(shared_index_layers or [False] * layer_num)
        self.enable_prefetch = any(self._is_shared_index_layer)
        self._prefetch_groups, self._prefetch_slot = _build_prefetch_groups(
            self._is_shared_index_layer
        )
        if not self.enable_prefetch:
            return

        # Small fixed grid for the copy-only kernel: low SM footprint so the
        # copies overlap compute with little contention.
        self._prefetch_copy_blocks = max(
            1, envs.SGLANG_HISPARSE_PREFETCH_BLOCKS.get()
        )  # HS_PATCH (upstream: 4)
        max_group_size = max(len(g) for g in self._prefetch_groups.values())
        self.prefetch_stream = device_module.Stream()
        self._prefetch_events = [device_module.Event() for _ in range(max_group_size)]
        # Plan recorded by the current anchor, replayed by its skip layers. One
        # buffer set suffices: the last skip layer's event wait orders the next
        # anchor's writes after this group's copies.
        miss_plan_capacity = self.spec_swap.miss_plan_capacity
        self._miss_src = torch.zeros(
            (max_num_req_slots, miss_plan_capacity),
            dtype=torch.int64,
            device=self.device,
        )
        self._miss_dst = torch.zeros(
            (max_num_req_slots, miss_plan_capacity),
            dtype=torch.int32,
            device=self.device,
        )
        self._miss_count = torch.zeros(
            (max_num_req_slots,), dtype=torch.int32, device=self.device
        )
        logger.info(
            "HiSparse: shared-index prefetch (plan-then-IO) enabled; %d anchor "
            "group(s), %d skip layer(s) of %d total.",
            len(self._prefetch_groups),
            sum(self._is_shared_index_layer),
            layer_num,
        )

    def set_decode_producer_stream(self, stream) -> None:
        self.decode_producer_stream = stream

    def allocate_spec_scratch(self, req_pool_indices: List[int]) -> None:
        for req_pool_idx in req_pool_indices:
            self.spec_swap.allocate_scratch(req_pool_idx)

    def destroy(self) -> None:
        # Drain in-flight transfers so the buffer is idle, then unregister it.
        # See HostKVCache.destroy for why the explicit unregister matters.
        self.write_staging_stream.synchronize()
        self.decode_backup_stream.synchronize()
        if self.enable_prefetch:
            # Skip-layer copies read the pinned host pool on the prefetch stream.
            self.prefetch_stream.synchronize()
        self.mem_pool_host.destroy()

    def get_token_stats(self) -> HiSparseTokenStats:
        device_allocator = self.token_to_kv_pool_allocator.hisparse_attn_allocator
        device_capacity = device_allocator.size
        device_tokens = device_capacity - device_allocator.available_size()
        host_capacity = self.mem_pool_host.size
        host_tokens = host_capacity - self.mem_pool_host.available_size()
        return HiSparseTokenStats(
            device_tokens=device_tokens,
            device_token_usage=(
                device_tokens / device_capacity if device_capacity > 0 else 0.0
            ),
            host_tokens=host_tokens,
            host_token_usage=(
                host_tokens / host_capacity if host_capacity > 0 else 0.0
            ),
        )

    def admit_request_into_staging(self, req: Req) -> None:
        """Start the host copy of a prefilled request and move it into staging."""
        req.hisparse_staging = True
        if self.hs_excl and self._hs_excl_stage(
            req
        ):  # HS_EXCL: no host copy, or waiting for host slots
            return
        move([self._hs_stage_copy(req)], self.ack_staging_queue)

    def _hs_stage_copy(self, req: Req) -> HiSparseAct:
        """Start the prompt's staging copy to host (the stock admit_request_into_staging); return its entry for the
        staging list."""
        full_kv_indices = self.req_to_token_pool.req_to_token[
            req.kv.req_pool_idx, : req.extend_end
        ].to(dtype=torch.int64, copy=True)
        device_indices = (
            self.mem_pool_device.translate_loc_from_full_to_hisparse_device(
                full_kv_indices
            )
        )

        prefill_len = len(device_indices)
        if (
            self.hs_lazy
        ):  # HS_LEAN: the host copy's prefix if the request becomes resident
            self._hs_staged_len[req.kv.req_pool_idx] = prefill_len
        host_indices = self.mem_pool_host.alloc_paged_token_slots(
            self.req_to_host_pool,
            self.req_to_host_pool_allocated_len,
            req.kv.req_pool_idx,
            0,
            prefill_len,
        )

        start_event = device_module.Event()
        finish_event = device_module.Event()
        start_event.record()
        with device_module.stream(self.write_staging_stream):
            start_event.wait(self.write_staging_stream)
            self.mem_pool_host.backup_from_device_all_layer(
                self.mem_pool_device,
                host_indices,
                device_indices,
                io_backend="kernel",
            )
            finish_event.record()
            if host_indices.is_cuda:
                host_indices.record_stream(self.write_staging_stream)
            if device_indices.is_cuda:
                device_indices.record_stream(self.write_staging_stream)

        return HiSparseAct(start_event, finish_event, req)

    # ---- HS_EXCL -----------------------------------------------------------------------------------------------------
    @staticmethod
    def _hs_excl_wanted(num_draft_tokens: int) -> bool:
        """HS_EXCL's switch and what it needs (DSA MLA pools): residency, its speculative variant under speculative
        decoding, the lazy host backup, no host map."""
        return bool(
            envs.SGLANG_HISPARSE_EXCLUSIVE.get()
            and envs.SGLANG_HISPARSE_RESIDENT.get()
            and envs.SGLANG_HISPARSE_LAZY_BACKUP.get()
            and not envs.SGLANG_HISPARSE_HOST_MAP.get()
            and (num_draft_tokens == 0 or envs.SGLANG_HISPARSE_SPEC_RESIDENT.get())
        )

    def _hs_excl_stage(self, req: Req) -> bool:
        """A prompt longer than the hot buffer will stay resident: no host slots and no staging copy (an event that is
        already complete keeps the staging queue's order and the TP count of ready requests). A short prompt is staged
        as before when the host pool has room for it, else it waits, in order: it joins the staging list without events
        (no copy yet) and _hs_excl_drain_wait starts its copy later. True: handled here."""
        rid = req.kv.req_pool_idx
        n = self.host_token_len(int(req.extend_end))
        if n > self.device_buffer_size:
            self._hs_excl_skip[rid] = n
            self._hs_staged_len[rid] = 0
            ev = device_module.Event()
            ev.record()
            move([HiSparseAct(ev, ev, req)], self.ack_staging_queue)
            self.hs_excl_stats[0] += 1
            self._hs_excl_skip_hook(req, n)
            return True
        if self._hs_excl_waiting() or not self._hs_excl_host_fits(n):
            move([HiSparseAct(None, None, req)], self.ack_staging_queue)
            self.hs_excl_stats[4] += 1
            return True
        return False

    def _hs_excl_waiting(self) -> bool:
        """Whether a short prompt waits for host slots (a staging-list entry without events)."""
        return self.hs_excl and any(
            act.finish_event is None for act in self.ack_staging_queue.items()
        )

    def _hs_excl_skip_hook(self, req: Req, n: int) -> None:
        """HS_EXCL: validation anchor (the in-situ hooks hash the prompt rows the stock staging would have copied)."""
        return

    def _hs_excl_drain_wait(self) -> None:
        """Stage the waiting short prompts, in order, while the host pool has room. A started copy's entry moves behind
        the entries already copying, as a fresh admission would join the list; the waiting ones stay last."""
        items = list(self.ack_staging_queue.items())
        copying = [act for act in items if act.finish_event is not None]
        waiting = [act for act in items if act.finish_event is None]
        started = []
        while waiting and self._hs_excl_host_fits(
            self.host_token_len(int(waiting[0].req.extend_end))
        ):
            started.append(self._hs_stage_copy(waiting.pop(0).req))
        if started:
            self.ack_staging_queue.reorder(copying + started + waiting)

    def _hs_excl_reserve(self) -> int:
        """Host slots a demotion leaves free: two pages for the next round of every buffered request and of the
        demoted one, and one page."""
        n_buf = int((self.req_device_buffer_size > 0).sum())
        return (2 * n_buf + 3) * self.page_size

    def _hs_excl_host_fits(self, n: int) -> bool:
        P = self.page_size
        return (
            self.mem_pool_host.available_size()
            >= -(-n // P) * P + self._hs_excl_reserve()
        )

    def _hs_excl_host_need(self, rid: int) -> int:
        """Host slots a demotion of resident request `rid` allocates at most: its device-owned positions
        (page-rounded) beyond the host slots it holds."""
        return max(
            0,
            self._owned_device_slots(self._resident[rid])
            - int(self._hs_host_len_np[rid]),
        )

    def _hs_excl_pick(self) -> Optional[int]:
        """The largest resident request (allocated length; reclaim_device's order and ties) whose demotion the host
        pool can take with the reserve left free; None if there is none."""
        room = self.mem_pool_host.available_size() - self._hs_excl_reserve()
        best, best_len, top = None, -1, -1
        for rid, req in self._resident.items():
            n = int(req.kv.kv_allocated_len)
            top = max(top, n)
            if n > best_len and self._hs_excl_host_need(rid) <= room:
                best, best_len = rid, n
        if best is None or best_len < top:
            self.hs_excl_stats[2] += 1
        return best

    def _hs_excl_reclaimable(self, room: int) -> int:
        """Device slots freed by the demotions a host room of `room` slots can take, largest first."""
        total = 0
        for rid, req in sorted(
            self._resident.items(), key=lambda kv: -int(kv[1].kv.kv_allocated_len)
        ):
            need = self._hs_excl_host_need(rid)
            if need <= room:
                room -= need
                total += max(self._owned_device_slots(req) - self.padded_buffer_size, 0)
        return total

    def hs_excl_decode_fits(
        self, requests, num_tokens: int = 0, device_term: Optional[int] = None
    ) -> bool:
        """The scheduler's decode check (allocator.check_decode_capacity): the next round's host slots of the buffered
        requests (up to one page past each one's allocated length) must be free, and the round's device slots
        (num_tokens) must come from the device term plus the demotions the host can take AFTER those host slots (both
        draw on the host pool in the same round); else the scheduler retracts."""
        P = self.page_size
        hl = self._hs_host_len_np
        res = self._resident
        need = 0
        for req in requests:
            rid = req.kv.req_pool_idx
            if rid is None or rid in res:
                continue
            want = -(-int(req.kv.kv_allocated_len) // P) * P + P
            have = int(hl[rid])
            if want > have:
                need += want - have
        free = self.mem_pool_host.available_size()
        ok = need <= free
        if ok and device_term is not None:
            # plain decode: a short request whose buffer reaches the hot-buffer size also takes the padded page in the
            # same step (_grow_device_buffers: 960 -> 1088 at a 1024 buffer), one page more than the scheduler counts
            B, caps, extra = self.device_buffer_size, self.req_device_buffer_size, 0
            for req in requests:
                rid = req.kv.req_pool_idx
                if rid is None or rid in res:
                    continue
                cap, n = int(caps[rid]), int(req.kv.kv_committed_len) + 1
                if 0 < cap < n <= B:
                    new_cap = min(-(-n // P) * P, B)
                    if new_cap == B:
                        new_cap = self.padded_buffer_size
                    extra += max(0, new_cap - cap - P)
            dev_need = num_tokens + extra
            if dev_need > device_term:
                ok = dev_need <= device_term + self._hs_excl_reclaimable(
                    free - need - self._hs_excl_reserve()
                )
        if ok:
            return True
        self.hs_excl_stats[3] += 1
        if self.hs_excl_stats[3] % 100 == 1:
            logger.info(
                "HiSparse exclusive: decode check short (host %d needed, %d free; device %d tokens + padded "
                "pages, term %s); stats %s",
                need,
                free,
                num_tokens,
                device_term,
                self.hs_excl_stats,
            )
        return False

    def admit_request_direct(self, req: Req) -> None:
        """Direct-to-host path: KV data already resides in host pool via RDMA.

        Skips staging DMA entirely. Only allocates a small device buffer
        (4KB) for decode-time swap-in, then marks the request as ready.
        Host indices were already written to req_to_host_pool.

        Metadata fixups after alloc_device_buffer():
        - alloc_device_buffer() sets device_buffer_tokens = [0, 1, ..., buf_size-1],
          which tells the swap-in kernel that those tokens are cached in the device
          buffer.  In the staging path this is correct (prefill filled the buffer),
          but here the buffer is empty.
        """
        self.alloc_device_buffer(req)

        host_len = self.host_token_len(req.kv.kv_allocated_len)
        if host_len <= self.device_buffer_size:
            # Short sequences (seq_len <= device_buffer_size): the kernel fast path
            # returns device_buffer_locs directly without any host loading, so we
            # must preload all tokens from host pool into the device buffer
            # TODO(hzh0425): Optimize this.
            self._preload_to_device_buffer(req)
        else:
            # Long sequence: reset device_buffer_tokens to -1 so the kernel
            # sees all slots as empty -> every top-k lookup is a miss -> host load.
            self.req_device_buffer_tokens[
                :, req.kv.req_pool_idx, : self.device_buffer_size
            ] = -1
            self.spec_swap.invalidate_cache(req.kv.req_pool_idx)

        req.hisparse_staging = False
        self._skip_first_backup[req.kv.req_pool_idx] = True
        logger.debug("HiSparse: admitting request %s directly", req.rid)

    def host_token_len(self, kv_allocated_len: int) -> int:
        if self.is_dsv4_hisparse:
            return kv_allocated_len // self.compress_ratio
        return kv_allocated_len

    def _preload_to_device_buffer(self, req: Req) -> None:
        """Preload all tokens from host pool into the device buffer."""
        n = self.host_token_len(req.kv.kv_allocated_len)
        host_indices = self.req_to_host_pool[req.kv.req_pool_idx, :n]
        device_locs = self.req_to_device_buffer[req.kv.req_pool_idx, :n]

        layer_ids = (
            range(self.mem_pool_device.start_layer, self.mem_pool_device.end_layer)
            if self.is_m3_hisparse
            else range(self.mem_pool_device.layer_num)
        )
        for layer_id in layer_ids:
            self.mem_pool_host.load_to_device_per_layer(
                self.mem_pool_device,
                host_indices,
                device_locs,
                layer_id,
                io_backend="kernel",
            )

    def alloc_device_buffer(self, req: Req) -> None:
        if self.is_dsv4_hisparse:
            allocated_len = req.extend_end
            alloc_size = self.padded_buffer_size
        else:
            allocated_len = req.kv.kv_allocated_len
            page_size = self.mem_pool_device.page_size
            # Allocate only enough for current tokens (page-aligned).
            # When prefill already fills device_buffer_size, include the reserved page.
            alloc_size = min(
                ((allocated_len + page_size - 1) // page_size) * page_size,
                self.device_buffer_size,
            )
            # Spec verify needs the complete, graph-stable buffer even for short prompts.
            if self.spec_swap.enabled or alloc_size == self.device_buffer_size:
                alloc_size = self.padded_buffer_size

        compressed_logical_indices = (
            self.mem_pool_device.translate_loc_from_full_to_compressed(
                self.req_to_token_pool.req_to_token[req.kv.req_pool_idx, :allocated_len]
            )
        )
        compressed_len = len(compressed_logical_indices)

        buffer_indices = self.token_to_kv_pool_allocator.alloc_device_buffer(
            compressed_logical_indices, alloc_size
        )
        if buffer_indices is None:
            logger.error(
                "HiSparse: alloc_device_buffer failed for req %s "
                "(compressed_len=%d, alloc_size=%d)",
                req.rid,
                compressed_len,
                alloc_size,
            )
            raise RuntimeError("HiSparse alloc_device_buffer returned None")

        buffer_indices = buffer_indices.to(torch.int32)
        self.req_to_device_buffer[req.kv.req_pool_idx, :alloc_size] = buffer_indices
        self.req_device_buffer_size[req.kv.req_pool_idx] = alloc_size

        self.req_device_buffer_tokens[
            :, req.kv.req_pool_idx, : self.device_buffer_size
        ] = self._device_buffer_arange_i32
        self.req_device_buffer_token_locs[:, req.kv.req_pool_idx, :alloc_size] = (
            buffer_indices[:alloc_size]
        )
        self.spec_swap.allocate_scratch(req.kv.req_pool_idx)
        self.spec_swap.reset(req.kv.req_pool_idx)
        if (
            self._hs_map_init is not None
        ):  # HS_PATCH v4: the kernel rebuilds this request's map on first use
            self._hs_map_init[:, req.kv.req_pool_idx] = 0

    def _grow_device_buffers(
        self,
        seq_lens: torch.Tensor,
        req_pool_indices: torch.Tensor,
        seq_lens_cpu: torch.Tensor,
        req_pool_indices_cpu: torch.Tensor,
    ) -> torch.Tensor:
        """Grow device buffers for requests whose sequence length exceeds current capacity."""
        current_caps = self.req_device_buffer_size[req_pool_indices_cpu]
        short_reqs_cpu = seq_lens_cpu <= self.device_buffer_size
        needs_grow_cpu = short_reqs_cpu & (seq_lens_cpu > current_caps)

        if torch.any(needs_grow_cpu):
            page_size = self.mem_pool_device.page_size
            grow_indices = torch.where(needs_grow_cpu)[0]

            # Compute all grow sizes on CPU, then do a single bulk allocation
            req_idxs = []
            old_caps = []
            new_caps = []
            grow_sizes = []
            total_grow = 0
            for i in grow_indices.tolist():
                req_idx = int(req_pool_indices_cpu[i])
                current_cap = int(current_caps[i])
                seq_len = int(seq_lens_cpu[i])

                new_cap = min(
                    ((seq_len + page_size - 1) // page_size) * page_size,
                    self.device_buffer_size,
                )
                if new_cap == self.device_buffer_size:
                    new_cap = self.padded_buffer_size
                grow_size = new_cap - current_cap
                if grow_size <= 0:
                    continue
                req_idxs.append(req_idx)
                old_caps.append(current_cap)
                new_caps.append(new_cap)
                grow_sizes.append(grow_size)
                total_grow += grow_size

            if total_grow > 0:
                if self._resident:  # HS_RESIDENT: demote resident requests to make room
                    self.reclaim_device(total_grow)
                if (
                    self.hs_el is not None
                ):  # HS_ELASTIC v4: buffers live below the band, their growth too
                    all_new_indices = (
                        self.token_to_kv_pool_allocator._hs_el_alloc_below_or_demote(
                            total_grow
                        )
                    )
                else:
                    all_new_indices = (
                        self.token_to_kv_pool_allocator.hisparse_attn_allocator.alloc(
                            total_grow
                        )
                    )
                if all_new_indices is None:
                    logger.error(
                        "HiSparse: _grow_device_buffers bulk alloc failed "
                        "(total_grow=%d)",
                        total_grow,
                    )
                    raise RuntimeError(
                        f"HiSparse _grow_device_buffers failed (total_grow={total_grow})"
                    )

                offset = 0
                for req_idx, current_cap, new_cap, grow_size in zip(
                    req_idxs, old_caps, new_caps, grow_sizes
                ):
                    chunk = all_new_indices[offset : offset + grow_size]
                    offset += grow_size
                    self.req_to_device_buffer[req_idx, current_cap:new_cap] = chunk
                    self.req_device_buffer_token_locs[
                        :, req_idx, current_cap:new_cap
                    ] = chunk
                    self.req_device_buffer_size[req_idx] = new_cap

        reserved_positions = (seq_lens - 1).clamp(max=self.device_buffer_size)
        return self.req_to_device_buffer[req_pool_indices, reserved_positions]

    # ---- HS_RESIDENT ----------------------------------------------------------------------------------------------
    def _gpu_rows(self, rows) -> torch.Tensor:
        # Pinned, non-blocking H2D: a plain torch.tensor(..., device=) would wait for the schedule stream's queued work.
        return torch.tensor(rows, dtype=torch.int64, pin_memory=True).to(
            self.device, non_blocking=True
        )

    def _can_stay_resident(self, req: Req) -> bool:
        # Short requests already live wholly in their buffer (the kernel's short fast path).
        return (
            self.enable_resident
            and self.host_token_len(req.kv.kv_allocated_len) > self.device_buffer_size
        )

    def _make_resident(self, req: Req) -> None:
        rid = req.kv.req_pool_idx
        self._resident[rid] = req
        self.req_resident[rid] = 1
        self.req_device_buffer_size[rid] = 0
        if self.hs_lazy:  # HS_LEAN: decode tokens stay on the device only, until a demotion backs them up
            self.hs_lazy_from[rid] = self._hs_staged_len[rid]
        n_dev = (
            self._hs_excl_skip.pop(rid, None) if self.hs_excl else None
        )  # HS_EXCL: the prefill, device only
        if n_dev is not None:
            self._hs_excl_base[rid] = n_dev
        if self.hs_spec_res:  # HS_SPECRES: positions [0, staged) have their prefill device slots (HS_EXCL: [0, prefill))
            self._hs_dev_len[rid] = self._hs_staged_len[rid] if n_dev is None else n_dev

    def _owned_device_slots(self, req: Req) -> int:
        page = self.mem_pool_device.page_size
        n = req.kv.kv_allocated_len
        if self.hs_spec_res:  # HS_SPECRES v2: only mirrored positions hold device slots; the rest are reserved ahead
            n = self._hs_dev_len.get(req.kv.req_pool_idx, n)
        return -(-n // page) * page

    def reclaimable_slots(self) -> int:
        """Device slots that demoting every resident request would free (the allocator's admission budget)."""
        if self.hs_excl:  # HS_EXCL: only demotions the host pool can take, keeping the reserve and the margin free
            return self._hs_excl_reclaimable(
                self.mem_pool_host.available_size()
                - self._hs_excl_reserve()
                - self._hs_excl_margin
            )
        return sum(
            max(self._owned_device_slots(r) - self.padded_buffer_size, 0)
            for r in self._resident.values()
        )

    def reclaim_device(self, need_slots: int) -> bool:
        """Demote resident requests, largest first, until `need_slots` device slots are free."""
        dev = self.token_to_kv_pool_allocator.hisparse_attn_allocator
        while dev.available_size() < need_slots:
            if not self._resident:
                return False
            rid = max(
                self._resident, key=lambda r: self._resident[r].kv.kv_allocated_len
            )
            if (
                self.hs_excl
            ):  # HS_EXCL: the largest resident request the host pool can take
                rid = self._hs_excl_pick()
                if rid is None:
                    return False
            self._demote(rid)
        return True

    # ---- HS_ELASTIC ------------------------------------------------------------------------------------------------
    def _hs_el_budget(self):
        el = self.hs_el
        kv, ik = el["kv"], el["ik"]
        band = (kv.mapped - kv.perm) * kv.layers
        return el["E"] - band - ik.mapped * ik.layers

    def _hs_el_sync(self) -> None:
        # nothing may touch memory being unmapped or moved: the in-flight batch, the staging and backup copies
        if self.decode_producer_stream is not None:
            device_module.current_stream().wait_stream(self.decode_producer_stream)
        self.wait_for_pending_backup()
        torch.cuda.synchronize()

    def hs_el_demote_one(self) -> bool:
        if not self._resident:
            return False
        rid = max(self._resident, key=lambda r: self._resident[r].kv.kv_allocated_len)
        if self.hs_excl:  # HS_EXCL: the largest resident request the host pool can take
            rid = self._hs_excl_pick()
            if rid is None:
                return False
        self._demote(rid)
        self.hs_el_stats[4] += 1
        return True

    def _hs_el_log_high(self) -> int:
        """Highest allocated logical page (0 when none): the logical allocator's free list is sorted below its limit."""
        el = self.hs_el
        alloc = self.token_to_kv_pool_allocator.logical_attn_allocator
        alloc.merge_and_sort_free()
        lim = el["log_max_page"]
        fp = alloc.free_pages
        n = int(fp.numel())
        if n == 0:
            return lim
        # free pages form [1..lim] minus used; the top run of free pages ends at lim
        tail = fp[-min(n, lim) :]
        expect = torch.arange(lim - tail.numel() + 1, lim + 1, device=fp.device)
        same = (tail == expect).flip(0).to(torch.int64).cumprod(0).sum()
        return lim - int(same)

    def hs_el_grow_band(self, log: bool = False, keep_extra: int = 0) -> None:
        """Lend index-K mapped above the logical high-water + headroom (+ keep_extra pages a pending logical
        allocation will take) to the band."""
        el = self.hs_el
        alloc = self.token_to_kv_pool_allocator
        kv, ik, pb, row, P = (
            el["kv"],
            el["ik"],
            el["ik_page_bytes"],
            el["row"],
            self.page_size,
        )
        keep_pages = self._hs_el_log_high() + el["headroom"] + 2 + keep_extra
        keep = max(0, min(ik.stride, (keep_pages + 1) * pb))
        g = ik.g
        if -(-keep // g) * g < ik.mapped:
            self._hs_el_sync()
            new_max = min(
                alloc.logical_attn_allocator.num_pages, -(-keep // g) * g // pb - 1
            )
            alloc.hs_el_set_limit("log", new_max)
            ik.shrink_to(keep)
        # map band chunks while the budget allows
        added = 0
        while (
            self._hs_el_budget() >= kv.chunk_bytes() and kv.mapped + kv.g <= kv.stride
        ):
            kv.grow_to(kv.mapped + kv.g)
            added += 1
        if added:
            alloc.hs_el_set_limit("dev", kv.mapped // row // P - 1)
            self.hs_el_stats[0] += 1
        if added or log:
            logger.info(
                "HiSparse elastic: band %d slots (%.1f GiB), index-K mapped for %d logical pages; budget left %.2f "
                "GiB; stats %s; runtime cuMemCreate %d",
                kv.mapped // row - el["k_min"] - P,
                (kv.mapped - kv.perm) * kv.layers / 2**30,
                ik.mapped // pb,
                self._hs_el_budget() / 2**30,
                self.hs_el_stats,
                int(kv.x.creates()) - el.get("creates0", 0),
            )

    def hs_el_ensure_logical(self, n_pages: int) -> None:
        """Index-K for n_pages more logical pages (+ headroom); takes the band back when the budget is short."""
        el = self.hs_el
        alloc = self.token_to_kv_pool_allocator
        la = alloc.logical_attn_allocator
        if la.available_size() // self.page_size >= n_pages + 1:
            return
        kv, ik, pb, row, P = (
            el["kv"],
            el["ik"],
            el["ik_page_bytes"],
            el["row"],
            self.page_size,
        )
        if el["log_max_page"] >= la.num_pages:
            return  # all logical pages mapped already
        want = min(ik.stride, (el["log_max_page"] + 2 + n_pages + el["headroom"]) * pb)
        need = (-(-want // ik.g) * ik.g - ik.mapped) * ik.layers
        if need > self._hs_el_budget():
            short = need - self._hs_el_budget()
            chunks = -(-short // kv.chunk_bytes())
            self._hs_el_shrink_band(max(kv.perm, kv.mapped - chunks * kv.g))
        ik.grow_to(want)
        alloc.hs_el_set_limit("log", min(la.num_pages, ik.mapped // pb - 1))

    def _hs_el_shrink_band(self, new_bytes: int) -> None:
        """Take band rows at and above new_bytes back: mapped pages there move below (fresh pages), demoting resident
        requests when there is no room; then the chunks are unmapped."""
        el = self.hs_el
        alloc = self.token_to_kv_pool_allocator
        dev = alloc.hisparse_attn_allocator
        kv, row, P = el["kv"], el["row"], self.page_size
        new_rows = new_bytes // row
        new_max_page = new_rows // P - 1
        self._hs_el_sync()
        alloc.hs_el_set_limit("dev", new_max_page)  # free pages above are dropped
        lo = (new_max_page + 1) * P
        mapping = alloc.full_to_hisparse_device_index_mapping
        while True:
            sel = torch.nonzero(mapping[:-1] >= lo).squeeze(1)
            if sel.numel() == 0:
                break
            slots = mapping[sel]
            pages = torch.unique(slots // P)
            new = dev.alloc(len(pages) * P)
            if new is None:
                if not self.hs_el_demote_one():
                    raise RuntimeError(
                        "HiSparse elastic: no room to take the band back"
                    )
                self._hs_el_sync()
                continue
            new_pages = new.view(-1, P)[:, 0] // P
            src = (pages[:, None] * P + torch.arange(P, device=self.device)).reshape(-1)
            dst = (
                new_pages[:, None] * P + torch.arange(P, device=self.device)
            ).reshape(-1)
            alloc.hs_copy_rows(src, dst)
            idx = torch.searchsorted(pages, slots // P)
            mapping[sel] = new_pages[idx] * P + slots % P
            self.hs_el_stats[2] += len(pages)
            break
        # nothing else may hold pages above the limit: buffers and scratch live below the band. v4: rows of buffered
        # requests only (plain decode writes a resident row's newest-token slot, its own and just moved, into the
        # reserved column of req_device_buffer_token_locs; nothing reads it there)
        sw = self.spec_swap
        buf_rows = (
            torch.nonzero(self.req_device_buffer_size > 0).squeeze(1).to(self.device)
        )
        held = int((self.req_to_device_buffer[buf_rows] >= lo).sum()) + int(
            (self.req_device_buffer_token_locs[:, buf_rows] >= lo).sum()
        )
        if getattr(sw, "enabled", False):
            held += int((sw.req_to_scratch >= lo).sum())
        if held:
            raise RuntimeError(
                f"HiSparse elastic: {held} buffer/scratch slots above the band limit {lo}"
            )
        # v2: no free page above the limit either (demotions above may have freed some; _release_page_ids drops them)
        fp = torch.cat((dev.free_pages, *getattr(dev, "staged_pages", [])))
        if fp.numel() and int(fp.max()) > new_max_page:
            raise RuntimeError(
                f"HiSparse elastic: free device page {int(fp.max())} above the band limit {new_max_page}"
            )
        torch.cuda.synchronize()
        kv.shrink_to(new_bytes)
        self.hs_el_stats[1] += 1
        logger.info(
            "HiSparse elastic: band taken back to %d slots; stats %s; runtime cuMemCreate %d",
            new_rows - el["k_min"] - P,
            self.hs_el_stats,
            int(kv.x.creates()) - el.get("creates0", 0),
        )

    def _demote(self, rid: int) -> None:
        """Turn resident request `rid` into a buffered one: its host copy is complete up to the newest token with KV,
        which moves to the reserved slot (where the next step backs it up from, as for any buffered request)."""
        req = self._resident.pop(rid)
        # Ordered after the in-flight batch (it may still read these slots) and after the pending host backup.
        if self.decode_producer_stream is not None:
            device_module.current_stream().wait_stream(self.decode_producer_stream)
        self.wait_for_pending_backup()
        allocated_len = req.kv.kv_allocated_len
        logical = self.req_to_token_pool.req_to_token[rid, :allocated_len].to(
            torch.int64
        )
        mapping = self.token_to_kv_pool_allocator.full_to_hisparse_device_index_mapping
        device_locs = mapping[logical]
        if self.hs_lazy:  # HS_LEAN: back up the tokens after the staged prefix first
            self._hs_lazy_backup(rid, device_locs)
        # HS_SPECRES: under speculative decoding the mapped positions run past the committed length (logical pages
        # reserved ahead), so there is no newest token to place: the reserved slot stays empty (read from host on demand)
        newest_src = (
            None if self.hs_spec_res else device_locs[device_locs > 0][-1:].clone()
        )
        if newest_src is not None and self.hs_el is not None:
            # HS_ELASTIC v4: into the reserved position's slot before anything is freed or moved (alloc_device_buffer
            # may move band buffer pages into freed pages, the newest token's included; a page move carries this row)
            self.mem_pool_device.transfer_values_on_device(
                device_locs[self.device_buffer_size : self.device_buffer_size + 1],
                newest_src,
            )
            newest_src = None
        # Keeps the first padded slots (tokens 0..padded-1 hold their own KV), frees the rest, unmaps every token.
        buffer_locs = self.token_to_kv_pool_allocator.alloc_device_buffer(
            logical, self.padded_buffer_size
        )
        reserved = buffer_locs[self.device_buffer_size : self.device_buffer_size + 1]
        if newest_src is not None:
            self.mem_pool_device.transfer_values_on_device(reserved, newest_src)
        buf32 = buffer_locs.to(torch.int32)
        self.req_to_device_buffer[rid, : self.padded_buffer_size] = buf32
        self.req_device_buffer_size[rid] = self.padded_buffer_size
        self.req_device_buffer_tokens[:, rid, : self.device_buffer_size] = (
            self._device_buffer_arange_i32
        )
        self.req_device_buffer_tokens[:, rid, self.device_buffer_size :] = -1
        self.req_device_buffer_token_locs[:, rid, : self.padded_buffer_size] = buf32
        self.lru_slots[:, rid, :].copy_(self._lru_init)
        if self.hs_spec_res:  # HS_SPECRES: the speculative swap state of a freshly staged buffered request
            self.spec_swap.allocate_scratch(rid)
            self.spec_swap.reset(rid)
            self._hs_dev_len.pop(rid, None)
        self.req_resident[rid] = 0
        self.resident_demotions += 1
        logger.info(
            "HiSparse: demoted request %s (%d tokens) to the buffer; %d resident, %d demotions so far",
            req.rid,
            allocated_len,
            len(self._resident),
            self.resident_demotions,
        )

    def _hs_lazy_backup(self, rid: int, device_locs: torch.Tensor) -> None:
        """HS_LEAN: copy resident request `rid`'s tokens [staged, m) to its host slots (m = tokens with a device slot,
        which for a resident request are a prefix). On the current stream: after the in-flight batch and the pending
        backup (the caller waited for both), before the demotion frees any slot."""
        staged = self._hs_staged_len[rid]
        mapped = device_locs > 0
        m = int(mapped.sum())  # host sync; demotions are rare
        if m < staged or not bool(mapped[:m].all()):
            raise RuntimeError(
                f"HiSparse lazy backup: device slots of request slot {rid} are not a prefix (m={m}, staged={staged})"
            )
        n = m - staged
        host_locs = None
        if n > 0:
            host_locs = self.mem_pool_host.alloc_paged_token_slots(
                self.req_to_host_pool,
                self.req_to_host_pool_allocated_len,
                rid,
                staged,
                n,
            )
            self.mem_pool_host.backup_from_device_all_layer(
                self.mem_pool_device,
                host_locs,
                device_locs[staged:m],
                io_backend="kernel",
            )
        self._hs_lazy_done(rid, staged, m, host_locs, device_locs)
        self.hs_lazy_from[rid] = _HS_I64_MAX
        self.hs_lazy_backups += 1
        self.hs_lazy_tokens += n
        if self.hs_excl and staged == 0:  # HS_EXCL: a whole-request copy
            self.hs_excl_stats[1] += n
        self._hs_excl_base.pop(rid, None)

    def _hs_spec_res_prepare(self, batch) -> List[int]:
        """HS_SPECRES: give every resident request of this verify batch the device pages of the logical pages it holds
        (kv_allocated_len, reserved ahead by spec decode), mirrored page for page: first the rest of the device page that
        holds position L - 1 (L = positions already mapped), then whole new pages. Demotes resident requests (largest
        first) when the free pages do not cover it. Returns the batch rows that are resident afterwards."""
        P = self.page_size
        if (
            self.hs_el is not None
        ):  # HS_ELASTIC: lend index-K slack to the band now and then
            self.hs_el_stats[5] += 1
            if self.hs_el_stats[5] % 200 == 0:
                self.hs_el_grow_band()
        res = self._resident
        dev = self.token_to_kv_pool_allocator.hisparse_attn_allocator

        def plan(rows):
            out = []
            for i in rows:
                req = batch.reqs[i]
                rid = req.kv.req_pool_idx
                L, E = self._hs_dev_len[rid], int(req.kv.kv_allocated_len)
                if E > L:
                    out.append((rid, L, min(E, -(-L // P) * P), E))
            return out

        rows = [i for i, req in enumerate(batch.reqs) if req.kv.req_pool_idx in res]
        todo = plan(rows)
        need = sum(E - pe for _, _, pe, E in todo)
        if need > dev.available_size():
            self.reclaim_device(
                need
            )  # may demote rows of this batch: they take the buffered path below
            rows = [i for i in rows if batch.reqs[i].kv.req_pool_idx in res]
            todo = plan(rows)
            need = sum(E - pe for _, _, pe, E in todo)
            if need > dev.available_size():
                raise RuntimeError(
                    f"HiSparse spec residency: {need} device slots needed after reclaim"
                )
        if todo:
            full = dev.alloc(need) if need else None
            if need and full is None:
                raise RuntimeError(
                    f"HiSparse spec residency: device allocation of {need} slots failed"
                )
            r2t = self.req_to_token_pool.req_to_token
            mapping = (
                self.token_to_kv_pool_allocator.full_to_hisparse_device_index_mapping
            )
            log_parts, dev_parts, off = [], [], 0
            for rid, L, pe, E in todo:
                if pe > L:  # the rest of the device page that holds position L - 1
                    base = mapping[r2t[rid, L - 1].to(torch.int64)]
                    log_parts.append(r2t[rid, L:pe])
                    dev_parts.append(base + 1 + self._hs_arange_page[: pe - L])
                if E > pe:
                    log_parts.append(r2t[rid, pe:E])
                    dev_parts.append(full[off : off + E - pe].to(torch.int64))
                    off += E - pe
                self._hs_dev_len[rid] = E
            mapping[torch.cat(log_parts).to(torch.int64)] = torch.cat(dev_parts)
        st = self._hs_spec_steps
        st[0] += 1
        st[1] += len(rows)
        st[2] += len(batch.reqs) - len(rows)
        if st[0] % 1000 == 0:
            logger.info(
                "HiSparse spec resident: %d verify steps, %.1f resident / %.1f buffered rows per step; %d resident now, "
                "%d demotions so far; lazy backups %d (%d tokens); reclaims before scratch %d, before buffer %d; "
                "exclusive %s, host free %d",
                st[0],
                st[1] / st[0],
                st[2] / st[0],
                len(res),
                self.resident_demotions,
                self.hs_lazy_backups,
                self.hs_lazy_tokens,
                self.hs_spec_reclaims[0],
                self.hs_spec_reclaims[1],
                self.hs_excl_stats if self.hs_excl else "off",
                self.mem_pool_host.available_size(),
            )
        return rows

    def _hs_spec_commit_hook(self, batch, accept_indices, res_rows) -> None:
        """HS_SPECRES: validation anchor (the in-situ hooks shadow the per-step backup the resident rows skip)."""
        return

    def _hs_lazy_done(self, rid, staged, m, host_locs, device_locs) -> None:
        """HS_LEAN: validation anchor (the in-situ hooks compare the lazy copy with the per-step backup)."""
        return

    def _hs_lean_decode(
        self,
        seq_lens,
        out_cache_loc,
        req_pool_indices,
        seq_lens_cpu,
        req_pool_indices_cpu,
    ) -> bool:
        """HS_LEAN: decode prep of a batch whose requests are all resident: what _resident_decode_locs does for such a
        batch (each new token's paged device slot next to the previous token's, or a fresh page), mapped from its
        logical slot. Returns False, having done nothing, for any other batch or when the free device pages do not
        cover the step (the general path reclaims)."""
        rpi = req_pool_indices_cpu.tolist()
        res = self._resident
        if not rpi or any(r not in res for r in rpi):
            self._hs_lean[1] += 1
            return False
        page = self.mem_pool_device.page_size
        dev = self.token_to_kv_pool_allocator.hisparse_attn_allocator
        need_pages = sum(1 for s in seq_lens_cpu.tolist() if (s - 1) % page == 0)
        if need_pages * page > dev.available_size():
            self._hs_lean[1] += 1
            return False
        self._hs_lean[0] += 1
        self._resident_steps += 1
        if self._resident_steps % 1000 == 0:
            logger.info(
                "HiSparse resident: %d of %d running requests resident, %d demotions so far; lean prep %d steps, "
                "general %d; lazy backups %d (%d tokens); exclusive %s, host free %d",
                len(rpi),
                len(rpi),
                self.resident_demotions,
                self._hs_lean[0],
                self._hs_lean[1],
                self.hs_lazy_backups,
                self.hs_lazy_tokens,
                self.hs_excl_stats if self.hs_excl else "off",
                self.mem_pool_host.available_size(),
            )
        mapping = self.token_to_kv_pool_allocator.full_to_hisparse_device_index_mapping
        prev_logical = self.req_to_token_pool.req_to_token[
            req_pool_indices, seq_lens - 2
        ].to(torch.int64)
        new_locs = dev.alloc_decode(seq_lens, seq_lens_cpu, mapping[prev_logical])
        if new_locs is None:
            raise RuntimeError("HiSparse lean decode allocation failed")
        compressed_locs = self.token_to_kv_pool_allocator.get_last_loc_compressed(
            out_cache_loc
        )
        self.mem_pool_device.full_to_hisparse_device_index_mapping[compressed_locs] = (
            new_locs
        )
        return True

    def _resident_decode_locs(
        self,
        reserved_buffer_loc: torch.Tensor,
        seq_lens: torch.Tensor,
        req_pool_indices: torch.Tensor,
        seq_lens_cpu: torch.Tensor,
        req_pool_indices_cpu: torch.Tensor,
    ) -> torch.Tensor:
        """Device slots for this step's new tokens of resident requests (paged, next to the previous token)."""
        rpi_cpu = req_pool_indices_cpu.tolist()
        rows = [i for i, r in enumerate(rpi_cpu) if r in self._resident]
        self._resident_steps += 1
        if self._resident_steps % 1000 == 0:
            logger.info(
                "HiSparse resident: %d of %d running requests resident, %d demotions so far",
                len(rows),
                len(rpi_cpu),
                self.resident_demotions,
            )
        if not rows:
            return reserved_buffer_loc
        page = self.mem_pool_device.page_size
        dev = self.token_to_kv_pool_allocator.hisparse_attn_allocator
        need_pages = sum(1 for i in rows if (int(seq_lens_cpu[i]) - 1) % page == 0)
        if need_pages * page > dev.available_size():
            self.reclaim_device(need_pages * page)
            demoted = [i for i in rows if rpi_cpu[i] not in self._resident]
            if demoted:  # these take the buffered path this step: the reserved slot of their new buffer
                d = self._gpu_rows(demoted)
                reserved_buffer_loc[d] = self.req_to_device_buffer[
                    req_pool_indices[d],
                    (seq_lens[d] - 1).clamp(max=self.device_buffer_size),
                ]
            rows = [i for i in rows if rpi_cpu[i] in self._resident]
            if not rows:
                return reserved_buffer_loc
        idx = self._gpu_rows(rows)
        rr = req_pool_indices[idx]
        sl = seq_lens[idx]
        prev_logical = self.req_to_token_pool.req_to_token[rr, sl - 2].to(torch.int64)
        prev_dev = (
            self.token_to_kv_pool_allocator.full_to_hisparse_device_index_mapping[
                prev_logical
            ]
        )
        new_locs = dev.alloc_decode(sl, seq_lens_cpu[rows], prev_dev)
        if new_locs is None:
            raise RuntimeError(
                "HiSparse resident decode allocation failed after reclaim"
            )
        reserved_buffer_loc[idx] = new_locs.to(reserved_buffer_loc.dtype)
        return reserved_buffer_loc

    def request_lists(self) -> Dict[ListKind, ReqList]:
        return {ListKind.HISPARSE_STAGING: self.ack_staging_queue}

    def move_ready_reqs(self, dst: ReqList) -> List[Req]:
        """Move the staged requests whose host copy has landed into `dst`, and
        return them."""
        ready_reqs: List[Req] = []
        if self._hs_excl_waiting():  # HS_EXCL: short prompts that waited for host slots
            self._hs_excl_drain_wait()
        if len(self.ack_staging_queue) == 0:
            return ready_reqs

        # HS_EXCL: entries still waiting for host slots have no copy (no events) and are not ready
        staged = [
            act
            for act in self.ack_staging_queue.items()
            if act.finish_event is not None
        ]
        finish_count = 0
        for _, finish_event, _ in staged:
            if not finish_event.query():
                break
            finish_count += 1
        queue_size = torch.tensor(finish_count, dtype=torch.int, device="cpu")
        if self.tp_world_size > 1:
            # synchronize TP workers to make sure the same update to scheduler
            torch.distributed.all_reduce(
                queue_size,
                op=torch.distributed.ReduceOp.MIN,
                group=self.tp_group,
            )
        finish_count = int(queue_size.item())
        move([req for _, _, req in staged[:finish_count]], dst)
        for _, _, req in staged[:finish_count]:
            # prepare device buffer and update req (HS_RESIDENT: or keep the whole prefill resident)
            if req.kv.req_pool_idx in self._hs_excl_skip or self._can_stay_resident(
                req
            ):  # HS_EXCL: no host copy
                self._make_resident(req)
            else:
                if (
                    self._resident
                ):  # a prompt just under the buffer size takes one extra page
                    self.reclaim_device(self.mem_pool_device.page_size)
                self.alloc_device_buffer(req)
            self._skip_first_backup[req.kv.req_pool_idx] = True
            req.hisparse_staging = False
            ready_reqs.append(req)
        return ready_reqs

    def prepare_spec_verify(self, batch: ScheduleBatch) -> None:
        self.spec_swap.prepare_verify(batch)

    def commit_spec_accept_tokens(
        self, batch: ScheduleBatch, accept_indices: torch.Tensor
    ) -> None:
        self.spec_swap.commit_accept_tokens(batch, accept_indices)

    def map_last_loc_to_buffer(
        self,
        seq_lens: torch.Tensor,
        out_cache_loc: torch.Tensor,
        req_pool_indices: torch.Tensor,
        seq_lens_cpu: torch.Tensor,
        req_pool_indices_cpu: torch.Tensor,
    ) -> None:
        self._hs_maybe_log_stats(seq_lens_cpu)  # HS_PATCH
        self._eager_backup_previous_token(
            seq_lens, req_pool_indices, seq_lens_cpu, req_pool_indices_cpu
        )

        if not self.is_dsv4_hisparse:
            if self.hs_lazy and self._hs_lean_decode(  # HS_LEAN
                seq_lens,
                out_cache_loc,
                req_pool_indices,
                seq_lens_cpu,
                req_pool_indices_cpu,
            ):
                return
            # Grow device buffers if needed and resolve the latest-token slot.
            reserved_buffer_loc = self._grow_device_buffers(
                seq_lens, req_pool_indices, seq_lens_cpu, req_pool_indices_cpu
            )
            if (
                self._resident
            ):  # HS_RESIDENT: resident requests' new tokens get their own device slots
                reserved_buffer_loc = self._resident_decode_locs(
                    reserved_buffer_loc,
                    seq_lens,
                    req_pool_indices,
                    seq_lens_cpu,
                    req_pool_indices_cpu,
                )
            self.req_device_buffer_token_locs[
                :, req_pool_indices, self.device_buffer_size
            ] = reserved_buffer_loc.to(torch.int32)

            compressed_locs = self.token_to_kv_pool_allocator.get_last_loc_compressed(
                out_cache_loc
            )
            # Page-size-one allocation creates a temporary slot before remapping
            # the new token into the request's reserved device-buffer slot.
            if _is_hip or self.mem_pool_device.page_size == 1:
                previous_locs = self.mem_pool_device._translate_loc_to_hisparse_device(
                    compressed_locs
                )
                stale_locs = previous_locs[
                    (previous_locs > 0) & (previous_locs != reserved_buffer_loc)
                ]
                if stale_locs.numel() > 0:
                    self.token_to_kv_pool_allocator.free_hisparse_indices(stale_locs)

            self.mem_pool_device.full_to_hisparse_device_index_mapping[
                compressed_locs
            ] = reserved_buffer_loc
            return

        active_reqs = seq_lens % self.compress_ratio == 0
        if not torch.any(active_reqs):
            return

        active_seq_lens = seq_lens[active_reqs]
        active_out_cache_loc = out_cache_loc[active_reqs]
        active_req_pool_indices = req_pool_indices[active_reqs]

        compressed_seq_lens = active_seq_lens // self.compress_ratio
        reserved_positions = (compressed_seq_lens - 1).clamp(
            max=self.device_buffer_size
        )
        reserved_buffer_loc = self.req_to_device_buffer[
            active_req_pool_indices, reserved_positions
        ]

        self.req_device_buffer_token_locs[
            :, active_req_pool_indices, self.device_buffer_size
        ] = reserved_buffer_loc.to(torch.int32)

        compressed_locs = self.token_to_kv_pool_allocator.get_last_loc_compressed(
            active_out_cache_loc
        )
        self.mem_pool_device.full_to_hisparse_device_index_mapping[compressed_locs] = (
            reserved_buffer_loc
        )

    def _eager_backup_previous_token(
        self,
        seq_lens: torch.Tensor,
        req_pool_indices: torch.Tensor,
        seq_lens_cpu: torch.Tensor,
        req_pool_indices_cpu: torch.Tensor,
    ) -> None:
        """Back up the previous compressed token to host memory.

        Each newly produced compressed token (one per `compress_ratio` decode
        steps) must be backed up to host so the swap-in kernel can later
        recover it.

        Two cases are skipped:
        - The first decode step right after staging: all prefill tokens were
          already backed up during staging, so there is nothing new to save.
        - Steps where `(seq_len - 1) % compress_ratio != 0`: no new compressed
          token was produced this step.
        """
        # Build the list of batch positions that need a host backup.
        # Skip the first decode step after staging (prefill already backed up),
        # and skip non-aligned steps that did not produce a new compressed token.
        backup_indices = []
        lazy_rows = []  # HS_LEAN: resident rows due for a backup; their tokens stay on the device until a demotion
        lazy = self._resident if self.hs_lazy else None
        for i in range(len(seq_lens_cpu)):
            req_idx = int(req_pool_indices_cpu[i])
            if self._skip_first_backup[req_idx]:
                self._skip_first_backup[req_idx] = False
                continue
            if (int(seq_lens_cpu[i]) - 1) % self.compress_ratio == 0:
                if lazy and req_idx in lazy:
                    lazy_rows.append(i)
                else:
                    backup_indices.append(i)
        self._hs_lazy_rows_hook(
            lazy_rows, seq_lens, req_pool_indices, seq_lens_cpu, req_pool_indices_cpu
        )

        if not backup_indices:
            return

        # Use a pinned, non-blocking H2D copy to avoid synchronizing the host
        # with pending work on the schedule stream.
        backup_indices_gpu = torch.tensor(
            backup_indices, dtype=torch.int64, device="cpu", pin_memory=True
        ).to(self.device, non_blocking=True)
        backup_req_indices = req_pool_indices[backup_indices_gpu]

        # The previous compressed token's position and its device buffer slot:
        #  compressed_pos = (seq_len - 1) // compress_ratio - 1
        #  - short: slot = compressed_pos          (within the regular buffer)
        #  - long:  slot = device_buffer_size      (the reserved slot)
        prev_seq_lens = seq_lens[backup_indices_gpu] - 1
        compressed_prev_seq_lens = prev_seq_lens // self.compress_ratio
        actual_compressed_pos = compressed_prev_seq_lens - 1

        buffer_slot = actual_compressed_pos.clamp(max=self.device_buffer_size)

        device_locs = self.req_to_device_buffer[backup_req_indices, buffer_slot]
        if self._resident:  # HS_RESIDENT: a resident request's previous token sits in its own device slot
            res_rows = [
                j
                for j, i in enumerate(backup_indices)
                if int(req_pool_indices_cpu[i]) in self._resident
            ]
            if res_rows:
                res_mask = torch.zeros(
                    len(backup_indices), dtype=torch.bool, pin_memory=True
                )
                res_mask[res_rows] = True
                res_mask = res_mask.to(self.device, non_blocking=True)
                own_locs = self.token_to_kv_pool_allocator.full_to_hisparse_device_index_mapping[
                    self.req_to_token_pool.req_to_token[
                        backup_req_indices, actual_compressed_pos
                    ].to(torch.int64)
                ]
                device_locs = torch.where(res_mask, own_locs, device_locs)

        host_locs_list = []
        for i in backup_indices:
            req_idx = int(req_pool_indices_cpu[i])
            start_pos = (int(seq_lens_cpu[i]) - 1) // self.compress_ratio - 1
            host_locs = self.mem_pool_host.alloc_paged_token_slots(
                self.req_to_host_pool,
                self.req_to_host_pool_allocated_len,
                req_idx,
                start_pos,
                1,
            )
            host_locs_list.append(host_locs)
        host_locs = torch.cat(host_locs_list)

        self.wait_for_pending_backup()
        schedule_stream = device_module.current_stream()
        with device_module.stream(self.decode_backup_stream):
            self.decode_backup_stream.wait_stream(schedule_stream)
            if self.decode_producer_stream is not None:
                self.decode_backup_stream.wait_stream(self.decode_producer_stream)
            if self.is_dsv4_hisparse:
                # One scattered token per request: a multiple of the page size
                # must not be mistaken for whole pages.
                self.mem_pool_host.backup_tokens_from_device_all_layer(
                    host_locs, device_locs
                )
            else:
                self.mem_pool_host.backup_from_device_all_layer(
                    self.mem_pool_device,
                    host_locs,
                    device_locs,
                    io_backend="kernel",
                )
            self._backup_done_event.record()
            if host_locs.is_cuda:
                host_locs.record_stream(self.decode_backup_stream)
            if backup_req_indices.is_cuda:
                backup_req_indices.record_stream(self.decode_backup_stream)
            if actual_compressed_pos.is_cuda:
                actual_compressed_pos.record_stream(self.decode_backup_stream)
            if device_locs.is_cuda:
                device_locs.record_stream(self.decode_backup_stream)
        self._has_pending_backup = True

    def _hs_lazy_rows_hook(
        self, lazy_rows, seq_lens, req_pool_indices, seq_lens_cpu, req_pool_indices_cpu
    ) -> None:
        """HS_LEAN: validation anchor (the in-situ hooks shadow the per-step backup the lazy rows skipped)."""
        return

    def wait_for_pending_backup(self) -> None:
        if not self._has_pending_backup:
            return
        self._backup_done_event.wait(device_module.current_stream())
        self._has_pending_backup = False

    def naive_load_topk(
        self,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        top_k_tokens: torch.Tensor,
        layer_id: int,
    ) -> torch.Tensor:
        """Load top-k selected tokens into device memory and return their device indices.

        This is a naive per-request loop implementation for debugging/validation.
        Production code uses swap_in_selected_pages (JIT CUDA kernel) instead.

        Note: dsv4 hisparse is not supported — DeepSeekV4SingleKVPoolHost has no
        load_to_device_per_layer and indices live in compressed space. Currently
        only used as a kernel oracle in test_hisparse_unit.py (non-dsv4 path).

        Args:
            req_pool_indices: Pool indices for each request.  Shape: (num_reqs,)
            seq_lens: Sequence lengths for each request.  Shape: (num_reqs,)
            top_k_tokens: Selected token positions per request.  Shape: (num_reqs, top_k)
            layer_id: The layer to load KV cache for.

        Returns:
            Device KV cache indices for the selected tokens.  Shape: (num_reqs, top_k)
        """
        assert not self.is_dsv4_hisparse, (
            "naive_load_topk is not implemented for dsv4 hisparse"
        )
        num_reqs = req_pool_indices.size(0)
        top_k_indices = torch.full(
            (num_reqs, self.top_k), -1, dtype=torch.int32, device=self.device
        )

        for i in range(num_reqs):
            seq_len = int(seq_lens[i].item())
            top_n = min(seq_len, self.top_k)
            if top_n == 0:
                continue

            req_idx = int(req_pool_indices[i].item())
            selected_tokens = top_k_tokens[i, :top_n].to(dtype=torch.int64)

            assert torch.all(selected_tokens >= 0), (
                f"Req {req_idx}: selected tokens contain negative positions"
            )
            assert torch.all(selected_tokens < seq_len), (
                f"Req {req_idx}: selected tokens {selected_tokens.tolist()} "
                f"out of range for seq_len={seq_len}"
            )

            if seq_len <= self.device_buffer_size:
                device_indices = self.req_to_device_buffer[req_idx, selected_tokens]
            else:
                device_indices = torch.empty(
                    top_n, dtype=torch.int64, device=self.device
                )

                is_latest_token = selected_tokens == (seq_len - 1)
                needs_host_load = ~is_latest_token

                device_indices[is_latest_token] = self.req_to_device_buffer[
                    req_idx, self.device_buffer_size
                ]

                num_to_load = int(needs_host_load.sum().item())
                if num_to_load > 0:
                    tokens_to_load = selected_tokens[needs_host_load]
                    host_locs = self.req_to_host_pool[req_idx, tokens_to_load]

                    invalid_mask = host_locs < 0
                    if torch.any(invalid_mask):
                        bad_positions = tokens_to_load[invalid_mask].tolist()
                        raise AssertionError(
                            f"Req {req_idx} (seq_len={seq_len}, layer={layer_id}): "
                            f"missing host backup at token positions {bad_positions}"
                        )

                    buffer_locs = self.req_to_device_buffer[req_idx, :num_to_load]
                    device_indices[needs_host_load] = buffer_locs

                    self.mem_pool_host.load_to_device_per_layer(
                        self.mem_pool_device,
                        host_locs,
                        buffer_locs,
                        layer_id,
                        io_backend="kernel",
                    )

            top_k_indices[i, :top_n] = device_indices.to(torch.int32)

        return top_k_indices

    def _release_staging(self, req: Req) -> None:
        """Free a staged request's host + device resources."""
        if self.hs_excl:  # HS_EXCL: a long prompt staged without a host copy
            self._hs_excl_skip.pop(req.kv.req_pool_idx, None)
        # Wait for any in-flight staging DMA to complete before freeing
        self.write_staging_stream.synchronize()

        prefill_len = req.extend_end
        allocated_locs = self.req_to_token_pool.req_to_token[
            req.kv.req_pool_idx, :prefill_len
        ]
        self.token_to_kv_pool_allocator.free_hisparse(allocated_locs)

        # Free host memory that was allocated during admit_request_into_staging
        host_indices = self.mem_pool_host.allocated_host_indices(
            self.req_to_host_pool,
            req.kv.req_pool_idx,
            self.req_to_host_pool_allocated_len[req.kv.req_pool_idx],
        )
        if self._hs_map is not None and host_indices.numel() > 0:  # HS_PATCH v4
            self._hs_map[:, host_indices] = -1
        if host_indices.numel() > 0:
            self.mem_pool_host.free(host_indices)
        self.req_to_host_pool[req.kv.req_pool_idx, :] = -1
        self.req_to_host_pool_allocated_len[req.kv.req_pool_idx] = 0
        self._skip_first_backup[req.kv.req_pool_idx] = False
        self.spec_swap.free_unrotated_scratch(req.kv.req_pool_idx)
        req.hisparse_staging = False

    def release_request(self, req: Req):
        """Free what a request holds here, whether finished, aborted or retracted."""
        if req.hisparse_staging:
            self._release_staging(req)
            return

        # release resources only after the execution of a potential overlapped batch
        if self.decode_producer_stream is not None:
            device_module.current_stream().wait_stream(self.decode_producer_stream)
        self.wait_for_pending_backup()
        if self.hs_excl:  # HS_EXCL
            self._hs_excl_skip.pop(req.kv.req_pool_idx, None)
            self._hs_excl_base.pop(req.kv.req_pool_idx, None)
        if self._resident and self._resident.pop(req.kv.req_pool_idx, None) is not None:
            # HS_RESIDENT: its device slots stay mapped; release_kv_cache frees them as it does a prefill's.
            self.req_resident[req.kv.req_pool_idx] = 0
            if self.hs_lazy:  # HS_LEAN
                self.hs_lazy_from[req.kv.req_pool_idx] = _HS_I64_MAX
            self._hs_dev_len.pop(req.kv.req_pool_idx, None)  # HS_SPECRES

        # Use kv_allocated_len (not seqlen): under speculative decoding the
        # allocator can over-allocate beyond the committed seqlen, and those
        # extra slots may carry stale mapping entries pointing at buffer slots
        # we just freed via free_hisparse_indices(all_hi). If left set, the
        # subsequent release_kv_cache -> allocator.free -> free_hisparse path
        # re-frees them (double-free into the page allocator's free list).
        allocated_len = req.kv.kv_allocated_len
        req_pool_idx = req.kv.req_pool_idx
        allocated_locs = self.req_to_token_pool.req_to_token[
            req_pool_idx, :allocated_len
        ]

        # The spec commit kernel rotates physical locations between each
        # layer's persistent cache and scratch.  Their current union is the
        # request's allocation ownership; the original side-buffer row is no
        # longer authoritative after the first rotation.
        current_cap = int(self.req_device_buffer_size[req_pool_idx])
        if current_cap > 0:
            owned_locs = self.req_device_buffer_token_locs[
                :, req_pool_idx, :current_cap
            ].reshape(-1)
            owned_locs = self.spec_swap.extend_owned_locs(req_pool_idx, owned_locs)
            all_hi = torch.unique(owned_locs[owned_locs > 0])
            if all_hi.numel() > 0:
                self.token_to_kv_pool_allocator.free_hisparse_indices(all_hi)

            compressed_locs = (
                self.mem_pool_device.translate_loc_from_full_to_compressed(
                    allocated_locs
                )
            )
            self.mem_pool_device.full_to_hisparse_device_index_mapping[
                compressed_locs
            ] = 0
        else:
            # The request can finish or be aborted during prefill, before a
            # device side buffer exists.  Its ordinary prefill mapping is still
            # owned by release_kv_cache; only release our separate workspace.
            self.spec_swap.free_unrotated_scratch(req_pool_idx)

        host_indices = self.mem_pool_host.allocated_host_indices(
            self.req_to_host_pool,
            req.kv.req_pool_idx,
            self.req_to_host_pool_allocated_len[req.kv.req_pool_idx],
        )
        if (
            self._hs_map is not None and host_indices.numel() > 0
        ):  # HS_PATCH v4: its host slots will be reused
            self._hs_map[:, host_indices] = -1
        if self._hs_map_init is not None:
            self._hs_map_init[:, req_pool_idx] = 0
        if host_indices.numel() > 0:
            self.mem_pool_host.free(host_indices)

        # clear req info
        self.req_device_buffer_tokens[:, req.kv.req_pool_idx, :] = -1
        self.req_device_buffer_token_locs[:, req.kv.req_pool_idx, :] = -1
        self.req_to_device_buffer[req.kv.req_pool_idx, :] = 0
        self.req_device_buffer_size[req.kv.req_pool_idx] = 0
        self.req_to_host_pool[req.kv.req_pool_idx, :] = -1
        self.req_to_host_pool_allocated_len[req.kv.req_pool_idx] = 0
        self.lru_slots[:, req.kv.req_pool_idx, :].copy_(self._lru_init)
        self.spec_swap.clear_scratch(req_pool_idx)
        self._skip_first_backup[req.kv.req_pool_idx] = False

    def _hs_maybe_log_stats(self, seq_lens_cpu) -> None:
        """HS_PATCH: every SGLANG_HISPARSE_STATS_EVERY decode steps, log the misses per request of the last
        anchor layer that ran (its plan is in _miss_count; one sync per sample)."""
        if self._hs_stats_every <= 0 or not self.enable_prefetch:
            return
        self._hs_step += 1
        if self._hs_step % self._hs_stats_every:
            return
        bs = int(seq_lens_cpu.numel())
        if bs == 0:
            return
        long_mask = seq_lens_cpu > self.device_buffer_size
        n_long = int(long_mask.sum())
        mc = self._miss_count[:bs].to("cpu")
        if n_long:
            mcl = mc[long_mask].float()
            logger.info(
                "HiSparse stats: step %d, bs %d (%d long), misses/req of top_k %d: mean %.1f, min %d, max %d, "
                "hit rate %.3f, miss bytes/layer %.2f MB",
                self._hs_step,
                bs,
                n_long,
                self.top_k,
                mcl.mean().item(),
                int(mcl.min()),
                int(mcl.max()),
                1.0 - mcl.mean().item() / self.top_k,
                mcl.sum().item() * self.item_size_bytes / 1e6,
            )
        else:
            logger.info(
                "HiSparse stats: step %d, bs %d, no long request yet", self._hs_step, bs
            )

    def _run_swap_in_kernel(
        self,
        req_pool_indices: torch.Tensor,
        compressed_seq_lens: torch.Tensor,
        top_k_result: torch.Tensor,
        layer_id: int,
        record_plan: bool = False,
        plan_only: bool = False,
    ) -> torch.Tensor:
        """Run the full plan+IO swap-in kernel for one layer; return its slot table.

        record_plan (set on the anchor of a shared-index group) also records the
        miss plan into self._miss_{src,dst,count} for the skip layers to replay.
        """
        num_reqs = req_pool_indices.size(0)
        plan = (
            dict(
                miss_src=self._miss_src[:num_reqs],
                miss_dst=self._miss_dst[:num_reqs],
                miss_count=self._miss_count[:num_reqs],
            )
            if record_plan
            else {}
        )

        if top_k_result.ndim == 3:
            return self.spec_swap.swap_in(
                req_pool_indices,
                compressed_seq_lens,
                top_k_result,
                layer_id,
                **plan,
            )
        if top_k_result.ndim != 2:
            raise ValueError("HiSparse top-k must be two- or three-dimensional.")

        top_k_indices = self.top_k_device_locs_buffer[:num_reqs, : self.top_k]
        swap_in_fn = (
            load_cache_to_device_buffer_dsv4_mla
            if self.is_dsv4_hisparse
            else load_cache_to_device_buffer_mla
        )
        skip_io_kwargs = (
            {} if _is_xpu else dict(skip_io=self.skip_io or plan_only)
        )  # HS_PATCH
        if (
            self._hs_map is not None and layer_id in self._hs_group_of_layer
        ):  # HS_PATCH v4
            g = self._hs_group_of_layer[layer_id]
            skip_io_kwargs.update(
                host_map=self._hs_map[g], map_init=self._hs_map_init[g]
            )
        if self.enable_resident:  # HS_RESIDENT
            skip_io_kwargs.update(
                resident=self.req_resident,
                req_to_token=self.req_to_token_pool.req_to_token,
                full_to_device=self.token_to_kv_pool_allocator.full_to_hisparse_device_index_mapping,
            )
        swap_in_fn(
            top_k_tokens=top_k_result,
            device_buffer_tokens=self.req_device_buffer_tokens[layer_id],
            host_cache_locs=self.req_to_host_pool,
            device_buffer_locs=self.req_device_buffer_token_locs[layer_id],
            host_cache=self.mem_pool_host.kv_buffer[layer_id],
            device_buffer=self.mem_pool_device.kv_buffer[layer_id],
            top_k_device_locs=top_k_indices,
            req_pool_indices=req_pool_indices,
            seq_lens=compressed_seq_lens,
            lru_slots=self.lru_slots[layer_id],
            item_size_bytes=self.item_size_bytes,
            num_top_k=self.top_k,
            hot_buffer_size=self.device_buffer_size,
            page_size=1,
            block_size=self.swap_in_block_size,
            num_real_reqs=self.num_real_reqs,
            **skip_io_kwargs,
            **plan,
        )
        return top_k_indices

    def swap_in_selected_blocks(
        self,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        top_k_blocks: torch.Tensor,
        layer_id: int,
        sparse_block_size: int,
    ) -> torch.Tensor:
        assert self.is_m3_hisparse
        num_reqs = req_pool_indices.size(0)
        num_selected_tokens = top_k_blocks.size(1) * sparse_block_size
        assert num_selected_tokens <= self.device_buffer_size, (
            f"MiniMax M3 selected {num_selected_tokens} tokens, but the "
            f"HiSparse device buffer holds only {self.device_buffer_size}."
        )
        top_k_indices = self.top_k_device_locs_buffer[:num_reqs, :num_selected_tokens]
        host_layer = layer_id - self.mem_pool_device.start_layer
        load_blocks_to_device_buffer_mha(
            top_k_blocks=top_k_blocks,
            device_buffer_tokens=self.req_device_buffer_tokens[host_layer],
            host_cache_locs=self.req_to_host_pool,
            device_buffer_locs=self.req_device_buffer_token_locs[host_layer],
            host_cache_k=self.mem_pool_host.k_buffer[host_layer],
            host_cache_v=self.mem_pool_host.v_buffer[host_layer],
            device_buffer_k=self.mem_pool_device.layer_router.read(layer_id).key(),
            device_buffer_v=self.mem_pool_device.layer_router.read(layer_id).value(),
            top_k_device_locs=top_k_indices,
            req_pool_indices=req_pool_indices,
            seq_lens=seq_lens,
            lru_slots=self.lru_slots[host_layer],
            item_size_bytes=self.item_size_bytes,
            hot_buffer_size=self.device_buffer_size,
            sparse_block_size=sparse_block_size,
            page_size=1,
            block_size=self.swap_in_block_size,
            num_real_reqs=self.num_real_reqs,
            skip_io=self.skip_io,
        )
        return top_k_indices

    def _run_copy_only_kernel(self, num_reqs: int, skip_layer: int) -> None:
        """Replay the anchor's recorded miss plan into a skip layer's buffers
        (IO-only; the anchor's slot table stays valid -- lockstep layout)."""
        copy_cache_planned_mla(
            miss_src=self._miss_src[:num_reqs],
            miss_dst=self._miss_dst[:num_reqs],
            miss_count=self._miss_count[:num_reqs],
            num_real_reqs=self.num_real_reqs,
            host_cache=self.mem_pool_host.kv_buffer[skip_layer],
            device_buffer=self.mem_pool_device.kv_buffer[skip_layer],
            item_size_bytes=self.item_size_bytes,
            num_blocks=self._prefetch_copy_blocks,
            is_dsv4_layout=self.is_dsv4_hisparse,
            skip_io=self.skip_io,
        )

    def swap_in_selected_pages(
        self,
        req_pool_indices: torch.Tensor,
        compressed_seq_lens: torch.Tensor,
        top_k_result: torch.Tensor,
        layer_id: int,
    ) -> torch.Tensor:
        """Swap selected top-k tokens into device memory and return their indices.

        With prefetch enabled, anchors swap in synchronously (recording the miss
        plan) and prefetch their skip layers' copies; skip layers just wait.
        """
        if not self.enable_prefetch:
            return self._run_swap_in_kernel(
                req_pool_indices,
                compressed_seq_lens,
                top_k_result,
                layer_id,
            )

        num_reqs = req_pool_indices.size(0)
        if self._is_shared_index_layer[layer_id]:
            # Skip layer: wait for its prefetched copy; the anchor's slot table
            # applies (shared index + lockstep buffers).
            slot = self._prefetch_slot[layer_id]
            self._prefetch_events[slot].wait(device_module.current_stream())
            if top_k_result.ndim == 3:
                return self.spec_swap.output_locs(top_k_result, num_reqs)
            return self.top_k_device_locs_buffer[:num_reqs, : self.top_k]

        # Anchor: swap in synchronously (recording the plan), then prefetch the
        # skip layers' copies on the side stream.
        group = self._prefetch_groups.get(layer_id)
        # HS_PATCH: plan-only anchor + wide planned copy on the compute stream
        wide = self._hs_anchor_copy_blocks > 0 and top_k_result.ndim == 2
        anchor_locs = self._run_swap_in_kernel(
            req_pool_indices,
            compressed_seq_lens,
            top_k_result,
            layer_id,
            record_plan=(group is not None) or wide,
            plan_only=wide,
        )
        if wide:
            copy_cache_planned_mla(
                miss_src=self._miss_src[:num_reqs],
                miss_dst=self._miss_dst[:num_reqs],
                miss_count=self._miss_count[:num_reqs],
                num_real_reqs=self.num_real_reqs,
                host_cache=self.mem_pool_host.kv_buffer[layer_id],
                device_buffer=self.mem_pool_device.kv_buffer[layer_id],
                item_size_bytes=self.item_size_bytes,
                num_blocks=self._hs_anchor_copy_blocks,
                is_dsv4_layout=self.is_dsv4_hisparse,
                skip_io=self.skip_io,
            )
        if group:
            # Fork: the prefetch stream must observe the anchor's plan (produced
            # on the current stream) before replaying it.
            self.prefetch_stream.wait_stream(device_module.current_stream())
            with device_module.stream(self.prefetch_stream):
                for skip_layer in group:
                    self._run_copy_only_kernel(num_reqs, skip_layer)
                    self._prefetch_events[self._prefetch_slot[skip_layer]].record(
                        self.prefetch_stream
                    )
        return anchor_locs
