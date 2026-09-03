# to be combined with the sparse coordinator class and sparse algorithm family

import logging
import weakref
from typing import TYPE_CHECKING, Dict, List, NamedTuple, Optional, Tuple, Union

import torch

from sglang.kernels.ops.kvcache.hisparse import (
    HiSparseSpecState,
    copy_cache_planned_mla,
    load_cache_to_device_buffer_dsv4_mla,
    load_cache_to_device_buffer_mla,
    load_cache_to_device_buffer_spec_mla,
)
from sglang.srt.configs.model_config import dsa_layer_skips_topk, is_deepseek_dsa
from sglang.srt.environ import envs
from sglang.srt.managers.schedule_batch import Req
from sglang.srt.mem_cache.allocator.hisparse import (
    DeepSeekV4HiSparseTokenToKVPoolAllocator,
    HiSparseTokenToKVPoolAllocator,
)
from sglang.srt.mem_cache.hisparse_memory_pool import (
    HiSparseDSATokenToKVPool,
)
from sglang.srt.mem_cache.memory_pool import ReqToTokenPool
from sglang.srt.mem_cache.memory_pool_host import DeepSeekV4PagedHostPool
from sglang.srt.mem_cache.pool_host.mla import MLATokenToKVPoolHost
from sglang.srt.utils import get_device_module, is_hip

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import ScheduleBatch

device_module = get_device_module()

_is_hip = is_hip()

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
    pp_size: int,
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
    if pp_size != 1:
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
        coordinator: "HiSparseCoordinator",
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

    def ensure_scratch(self, req_pool_indices_cpu: torch.Tensor) -> None:
        if not self.enabled:
            return
        req_indices = [int(idx) for idx in req_pool_indices_cpu.tolist()]
        missing = [idx for idx in req_indices if idx not in self._scratch_reqs]
        if not missing:
            return

        total_slots = len(missing) * self.scratch_capacity
        allocator = self._coordinator.token_to_kv_pool_allocator
        scratch_locs = allocator.hisparse_attn_allocator.alloc(total_slots)
        if scratch_locs is None:
            raise RuntimeError(
                f"HiSparse spec failed to allocate {total_slots} scratch slots."
            )
        scratch_locs = scratch_locs.to(torch.int32).view(
            len(missing), self.scratch_capacity
        )
        for row, req_pool_idx in enumerate(missing):
            # Physical slots are shared by all KV layers, but every full-index
            # layer rotates those IDs between persistent cache and scratch
            # independently. Keep per-layer location views so one layer cannot
            # overwrite the next layer's ownership state.
            self.req_to_scratch[:, req_pool_idx].copy_(scratch_locs[row])
            self._scratch_reqs.add(req_pool_idx)

    def free_scratch(self, req_pool_idx: int) -> None:
        if not self.enabled or req_pool_idx not in self._scratch_reqs:
            return
        scratch_locs = torch.unique(self.req_to_scratch[:, req_pool_idx].reshape(-1))
        self._coordinator.token_to_kv_pool_allocator.free_hisparse_indices(scratch_locs)
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

    def release_decode_reserve(
        self,
        batch: "ScheduleBatch",
        current_kv_lens_cpu: torch.Tensor,
        next_kv_lens_cpu: torch.Tensor,
    ) -> None:
        """Release physical pages hidden behind spec decode's logical reserve."""
        coordinator = self._coordinator
        req_pool_indices = batch.req_pool_indices
        logical_locs = [
            coordinator.req_to_token_pool.req_to_token[
                req_pool_idx, current_len:next_len
            ]
            for req_pool_idx, current_len, next_len in zip(
                req_pool_indices,
                current_kv_lens_cpu.tolist(),
                next_kv_lens_cpu.tolist(),
                strict=True,
            )
            if current_len < next_len
        ]
        if not logical_locs:
            return
        logical_locs = torch.cat(logical_locs)
        allocator = coordinator.token_to_kv_pool_allocator
        mapping = allocator.full_to_hisparse_device_index_mapping
        mapped_device_locs = mapping[logical_locs]
        mapped_device_locs = mapped_device_locs[mapped_device_locs > 0]
        if mapped_device_locs.numel() > 0:
            allocator.free_hisparse_indices(mapped_device_locs)
        mapping[logical_locs] = 0

    def prepare_verify(self, batch: "ScheduleBatch") -> None:
        """Bind target-verify KV writes to the side buffer's extra page."""
        coordinator = self._coordinator
        req_pool_indices = batch.req_pool_indices
        req_pool_indices_cpu = batch.req_pool_indices_cpu
        if req_pool_indices_cpu is None:
            req_pool_indices_cpu = req_pool_indices.cpu()
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

        coordinator._grow_device_buffers_to(
            req_pool_indices_cpu,
            torch.full_like(req_pool_indices_cpu, coordinator.padded_buffer_size),
        )
        self.ensure_scratch(req_pool_indices_cpu)

        extra_start = coordinator.device_buffer_size + 1
        total_slots = req_pool_indices.numel() * self.num_draft_tokens
        row_indices = torch.repeat_interleave(req_pool_indices, self.num_draft_tokens)
        offsets = (
            torch.arange(total_slots, dtype=torch.int64, device=req_pool_indices.device)
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
        self,
        host_locs: torch.Tensor,
        device_locs: torch.Tensor,
        *,
        wait: bool,
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
        if wait:
            coordinator.wait_for_pending_backup()

    def commit_accept_tokens(
        self, batch: "ScheduleBatch", accept_indices: torch.Tensor
    ) -> None:
        """Persist accepted target-verify KV in host and the persistent buffer."""
        if batch.forward_mode.is_idle():
            return
        coordinator = self._coordinator
        req_pool_indices = batch.req_pool_indices
        req_pool_indices_cpu = batch.req_pool_indices_cpu
        if req_pool_indices_cpu is None:
            req_pool_indices_cpu = req_pool_indices.cpu()
        seq_lens = batch.seq_lens
        seq_lens_cpu = batch.seq_lens_cpu
        if seq_lens_cpu is None:
            seq_lens_cpu = seq_lens.cpu()
        verify_cache_locs = batch.out_cache_loc

        if verify_cache_locs.numel() == 0:
            return
        batch_size = req_pool_indices.numel()
        if batch_size == 0 or verify_cache_locs.numel() % batch_size != 0:
            raise ValueError("HiSparse verify cache locations are not request-aligned.")

        counts = (accept_indices >= 0).sum(dim=1).to(torch.int64)
        counts_cpu = counts.cpu()
        num_accept_tokens = int(counts_cpu.sum().item())
        mapping = (
            coordinator.token_to_kv_pool_allocator.full_to_hisparse_device_index_mapping
        )
        if num_accept_tokens == 0:
            mapping[verify_cache_locs] = 0
            return
        accept_offsets = accept_indices[accept_indices >= 0].to(torch.int64)
        if torch.any(accept_offsets >= verify_cache_locs.numel()):
            raise ValueError("HiSparse accept_indices point outside verify_cache_locs.")

        accept_source_locs = verify_cache_locs[accept_offsets]
        source_device_locs = mapping[accept_source_locs].clone()
        accept_req_indices = torch.repeat_interleave(req_pool_indices, counts)
        segment_offsets = torch.cat(
            [
                torch.zeros(1, dtype=torch.int64, device=counts.device),
                counts.cumsum(dim=0),
            ]
        )
        positions_in_req = torch.arange(
            num_accept_tokens, dtype=torch.int64, device=counts.device
        ) - torch.repeat_interleave(segment_offsets[:-1], counts)
        accept_positions = (
            torch.repeat_interleave(seq_lens.to(torch.int64), counts) + positions_in_req
        )
        canonical_locs = coordinator.req_to_token_pool.req_to_token[
            accept_req_indices, accept_positions
        ]

        destination_locs = source_device_locs.clone()
        hot_mask = accept_positions < coordinator.device_buffer_size
        if torch.any(hot_mask):
            destination_locs[hot_mask] = coordinator.req_to_device_buffer[
                accept_req_indices[hot_mask], accept_positions[hot_mask]
            ]
        last_offsets = segment_offsets[1:] - 1
        last_positions = accept_positions[last_offsets]
        newest_columns = last_positions.clamp(max=coordinator.device_buffer_size)
        newest_locs = coordinator.req_to_device_buffer[req_pool_indices, newest_columns]
        destination_locs[last_offsets] = newest_locs

        needs_copy = destination_locs != source_device_locs
        if torch.any(needs_copy):
            coordinator.mem_pool_device.transfer_values_on_device(
                dst_indices=destination_locs[needs_copy],
                src_indices=source_device_locs[needs_copy],
            )

        host_locs = []
        for req_pool_idx, seq_len, count in zip(
            req_pool_indices_cpu.tolist(),
            seq_lens_cpu.tolist(),
            counts_cpu.tolist(),
            strict=True,
        ):
            count = int(count)
            if count > 0:
                host_locs.append(
                    coordinator.mem_pool_host.alloc_paged_token_slots(
                        coordinator.req_to_host_pool,
                        coordinator.req_to_host_pool_allocated_len,
                        int(req_pool_idx),
                        int(seq_len),
                        count,
                    )
                )
        self._backup_device_locs_to_host(
            torch.cat(host_locs), destination_locs, wait=True
        )

        mapping[verify_cache_locs] = 0
        if torch.any(hot_mask):
            mapping[canonical_locs[hot_mask]] = destination_locs[hot_mask]
            coordinator.req_device_buffer_tokens[
                :, accept_req_indices[hot_mask], accept_positions[hot_mask]
            ] = (accept_positions[hot_mask].to(torch.int32).unsqueeze(0))
            coordinator.req_device_buffer_token_locs[
                :, accept_req_indices[hot_mask], accept_positions[hot_mask]
            ] = (destination_locs[hot_mask].to(torch.int32).unsqueeze(0))

        mapping[canonical_locs[last_offsets]] = newest_locs
        coordinator.req_device_buffer_tokens[:, req_pool_indices, newest_columns] = (
            last_positions.to(torch.int32).unsqueeze(0)
        )
        coordinator.req_device_buffer_token_locs[
            :, req_pool_indices, newest_columns
        ] = newest_locs.to(torch.int32).unsqueeze(0)
        for req_pool_idx in req_pool_indices_cpu.tolist():
            coordinator._skip_first_backup[int(req_pool_idx)] = True

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
        )
        return top_k_indices

    def output_locs(self, top_k_result: torch.Tensor, num_reqs: int) -> torch.Tensor:
        if not self.enabled or self.top_k_device_locs is None:
            raise RuntimeError("HiSparse spec swap is not initialized.")
        return self.top_k_device_locs[
            :num_reqs, : top_k_result.shape[1], : top_k_result.shape[2]
        ]


class HiSparseCoordinator:
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
        self.compress_ratio = self.token_to_kv_pool_allocator.compress_ratio

        self.is_dsv4_hisparse = isinstance(
            self.token_to_kv_pool_allocator, DeepSeekV4HiSparseTokenToKVPoolAllocator
        )
        if self.is_dsv4_hisparse:
            self.mem_pool_device = self.token_to_kv_pool_allocator.hisparse_kvcache
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
            self.mem_pool_device: HiSparseDSATokenToKVPool = (
                self.token_to_kv_pool_allocator.get_kvcache()
            )
            self.mem_pool_host = MLATokenToKVPoolHost(
                device_pool=self.mem_pool_device,
                host_to_device_ratio=host_to_device_ratio,
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
        self.ack_staging_queue: List[HiSparseAct] = []
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
        self._device_buffer_arange_i32 = torch.arange(
            self.device_buffer_size, dtype=torch.int32, device=device
        )

        # Pre-allocated output buffer for swap_in_selected_pages (CUDA-graph safe)
        self.top_k_device_locs_buffer = torch.full(
            (max_num_req_slots, self.top_k), -1, dtype=torch.int32, device=device
        )
        self.raw_indices_buffer = torch.full(
            (max_num_req_slots, self.top_k), -1, dtype=torch.int32, device=device
        )
        # Scalar tensor: number of real (non-padded) requests in the batch.
        # Updated before each graph replay so padded blocks early-return.
        self.num_real_reqs = torch.zeros(1, dtype=torch.int32, device=device)

        # CPU flag: True means "skip backup on the next decode step" because
        # staging already backed up all prefill tokens.  Cleared after one step.
        self._skip_first_backup = [False] * max_num_req_slots

        self._init_shared_index_prefetch(
            shared_index_layers=shared_index_layers,
            layer_num=layer_num,
            max_num_req_slots=max_num_req_slots,
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
        self._is_shared_index_layer = list(shared_index_layers or [False] * layer_num)
        self.enable_prefetch = any(self._is_shared_index_layer)
        self._prefetch_groups, self._prefetch_slot = _build_prefetch_groups(
            self._is_shared_index_layer
        )
        if not self.enable_prefetch:
            return

        # Small fixed grid for the copy-only kernel: low SM footprint so the
        # copies overlap compute with little contention.
        self._prefetch_copy_blocks = 4
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
        req.hisparse_staging = True

        full_kv_indices = self.req_to_token_pool.req_to_token[
            req.kv.req_pool_idx, : req.extend_range.end
        ].to(dtype=torch.int64, copy=True)
        device_indices = (
            self.mem_pool_device.translate_loc_from_full_to_hisparse_device(
                full_kv_indices
            )
        )

        prefill_len = len(device_indices)
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

        self.ack_staging_queue.append(HiSparseAct(start_event, finish_event, req))

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

        for layer_id in range(self.mem_pool_device.layer_num):
            self.mem_pool_host.load_to_device_per_layer(
                self.mem_pool_device,
                host_indices,
                device_locs,
                layer_id,
                io_backend="kernel",
            )

    def alloc_device_buffer(self, req: Req) -> None:
        if self.is_dsv4_hisparse:
            allocated_len = req.extend_range.end
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
            if alloc_size == self.device_buffer_size:
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
        self.spec_swap.reset(req.kv.req_pool_idx)

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

    def _grow_device_buffers_to(
        self, req_pool_indices_cpu: torch.Tensor, target_caps: torch.Tensor
    ) -> None:
        """Grow selected request buffers to explicit capacities."""
        grow_reqs = []
        for req_pool_idx, target_cap in zip(
            req_pool_indices_cpu.tolist(), target_caps.tolist(), strict=True
        ):
            old_cap = int(self.req_device_buffer_size[req_pool_idx])
            if old_cap < target_cap:
                grow_reqs.append((int(req_pool_idx), old_cap, int(target_cap)))
        total_grow = sum(new - old for _, old, new in grow_reqs)
        if total_grow == 0:
            return

        new_locs = self.token_to_kv_pool_allocator.hisparse_attn_allocator.alloc(
            total_grow
        )
        if new_locs is None:
            raise RuntimeError(f"HiSparse failed to allocate {total_grow} KV slots.")
        offset = 0
        for req_pool_idx, old_cap, new_cap in grow_reqs:
            chunk = new_locs[offset : offset + new_cap - old_cap]
            offset += new_cap - old_cap
            self.req_to_device_buffer[req_pool_idx, old_cap:new_cap] = chunk
            self.req_device_buffer_token_locs[:, req_pool_idx, old_cap:new_cap] = chunk
            self.req_device_buffer_size[req_pool_idx] = new_cap

    def has_ongoing_staging(self) -> bool:
        return len(self.ack_staging_queue) > 0

    def collect_ready_reqs(self) -> List[Req]:
        ready_reqs: List[Req] = []
        if len(self.ack_staging_queue) == 0:
            return ready_reqs

        finish_count = 0
        for _, finish_event, _ in self.ack_staging_queue:
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
        while finish_count > 0:
            _, _, req = self.ack_staging_queue.pop(0)
            # prepare device buffer and update req
            self.alloc_device_buffer(req)
            self._skip_first_backup[req.kv.req_pool_idx] = True
            req.hisparse_staging = False
            finish_count -= 1
            ready_reqs.append(req)
        return ready_reqs

    def release_spec_decode_reserve(
        self,
        batch: "ScheduleBatch",
        current_kv_lens_cpu: torch.Tensor,
        next_kv_lens_cpu: torch.Tensor,
    ) -> None:
        self.spec_swap.release_decode_reserve(
            batch, current_kv_lens_cpu, next_kv_lens_cpu
        )

    def prepare_spec_verify(self, batch: "ScheduleBatch") -> None:
        self.spec_swap.prepare_verify(batch)

    def commit_spec_accept_tokens(
        self, batch: "ScheduleBatch", accept_indices: torch.Tensor
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
        self._eager_backup_previous_token(
            seq_lens, req_pool_indices, seq_lens_cpu, req_pool_indices_cpu
        )

        if not self.is_dsv4_hisparse:
            # Grow device buffers if needed and resolve the latest-token slot.
            reserved_buffer_loc = self._grow_device_buffers(
                seq_lens, req_pool_indices, seq_lens_cpu, req_pool_indices_cpu
            )
            self.req_device_buffer_token_locs[
                :, req_pool_indices, self.device_buffer_size
            ] = reserved_buffer_loc.to(torch.int32)

            compressed_locs = self.token_to_kv_pool_allocator.get_last_loc_compressed(
                out_cache_loc
            )
            # ROCm: the decode remap creates a temporary hisparse device slot per
            # new token (via the page_size==1 allocator path). Free the stale
            # slot before pointing the mapping at the reserved device-buffer slot,
            # otherwise the temporary slots leak and corrupt later swap-in lookups.
            # CUDA keeps the original behavior: the swap-in kernel consumes only
            # top_k_device_locs, so stale mapping entries are harmless there.
            if _is_hip:
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
        for i in range(len(seq_lens_cpu)):
            req_idx = int(req_pool_indices_cpu[i])
            if self._skip_first_backup[req_idx]:
                self._skip_first_backup[req_idx] = False
                continue
            if (int(seq_lens_cpu[i]) - 1) % self.compress_ratio == 0:
                backup_indices.append(i)

        if not backup_indices:
            return

        backup_indices_gpu = torch.tensor(
            backup_indices, dtype=torch.int64, device=self.device
        )
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
        assert (
            not self.is_dsv4_hisparse
        ), "naive_load_topk is not implemented for dsv4 hisparse"
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

            assert torch.all(
                selected_tokens >= 0
            ), f"Req {req_idx}: selected tokens contain negative positions"
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

    def abort_staging_request(self, req: Req) -> None:
        """Remove a request from the staging queue and free its host + device resources.

        Must be called when aborting a request that has been admitted into staging
        but has not yet completed (i.e. req.hisparse_staging is True).
        """
        # Remove from staging queue
        self.ack_staging_queue = [
            act for act in self.ack_staging_queue if act.req is not req
        ]
        # Wait for any in-flight staging DMA to complete before freeing
        self.write_staging_stream.synchronize()

        prefill_len = req.extend_range.end
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
        if host_indices.numel() > 0:
            self.mem_pool_host.free(host_indices)
        self.req_to_host_pool[req.kv.req_pool_idx, :] = -1
        self.req_to_host_pool_allocated_len[req.kv.req_pool_idx] = 0
        self._skip_first_backup[req.kv.req_pool_idx] = False
        req.hisparse_staging = False

    def retract_req(self, req: Req) -> None:
        if req.hisparse_staging:
            self.abort_staging_request(req)
        else:
            self.request_finished(req)

    def request_finished(self, req: Req):
        # release resources only after the execution of a potential overlapped batch
        if self.decode_producer_stream is not None:
            device_module.current_stream().wait_stream(self.decode_producer_stream)
        self.wait_for_pending_backup()

        # Use kv_allocated_len (not seqlen): under speculative decoding the
        # allocator can over-allocate beyond the committed seqlen, and those
        # extra slots may carry stale mapping entries pointing at buffer slots
        # we just freed via free_hisparse_indices(all_hi). If left set, the
        # subsequent release_kv_cache -> allocator.free -> free_hisparse path
        # re-frees them (double-free into the page allocator's free list).
        allocated_len = req.kv.kv_allocated_len
        req_pool_idx = req.kv.req_pool_idx

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

        allocated_locs = self.req_to_token_pool.req_to_token[
            req_pool_idx, :allocated_len
        ]
        compressed_locs = self.mem_pool_device.translate_loc_from_full_to_compressed(
            allocated_locs
        )
        self.mem_pool_device.full_to_hisparse_device_index_mapping[compressed_locs] = 0

        host_indices = self.mem_pool_host.allocated_host_indices(
            self.req_to_host_pool,
            req_pool_idx,
            self.req_to_host_pool_allocated_len[req_pool_idx],
        )
        if host_indices.numel() > 0:
            self.mem_pool_host.free(host_indices)

        # clear req info
        self.req_device_buffer_tokens[:, req_pool_idx, :] = -1
        self.req_device_buffer_token_locs[:, req_pool_idx, :] = -1
        self.req_to_device_buffer[req_pool_idx, :] = 0
        self.req_device_buffer_size[req_pool_idx] = 0
        self.req_to_host_pool[req_pool_idx, :] = -1
        self.req_to_host_pool_allocated_len[req_pool_idx] = 0
        self.lru_slots[:, req_pool_idx, :].copy_(self._lru_init)
        self.spec_swap.clear_scratch(req_pool_idx)
        self._skip_first_backup[req_pool_idx] = False

    def _run_swap_in_kernel(
        self,
        req_pool_indices: torch.Tensor,
        compressed_seq_lens: torch.Tensor,
        top_k_result: torch.Tensor,
        layer_id: int,
        record_plan: bool = False,
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

        top_k_indices = self.top_k_device_locs_buffer[:num_reqs]
        swap_in_fn = (
            load_cache_to_device_buffer_dsv4_mla
            if self.is_dsv4_hisparse
            else load_cache_to_device_buffer_mla
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
            skip_io=self.skip_io,
            **plan,
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
                req_pool_indices, compressed_seq_lens, top_k_result, layer_id
            )

        num_reqs = req_pool_indices.size(0)
        if self._is_shared_index_layer[layer_id]:
            # Skip layer: wait for its prefetched copy; the anchor's slot table
            # applies (shared index + lockstep buffers).
            slot = self._prefetch_slot[layer_id]
            self._prefetch_events[slot].wait(device_module.current_stream())
            if top_k_result.ndim == 3:
                return self.spec_swap.output_locs(top_k_result, num_reqs)
            return self.top_k_device_locs_buffer[:num_reqs]

        # Anchor: swap in synchronously (recording the plan), then prefetch the
        # skip layers' copies on the side stream.
        group = self._prefetch_groups.get(layer_id)
        anchor_locs = self._run_swap_in_kernel(
            req_pool_indices,
            compressed_seq_lens,
            top_k_result,
            layer_id,
            record_plan=group is not None,
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
