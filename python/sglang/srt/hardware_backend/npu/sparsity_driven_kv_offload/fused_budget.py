"""Plan persistent allocations for native LIM and CopySFA offload."""

from __future__ import annotations

from copy import copy
from math import prod
from typing import TYPE_CHECKING

import msgspec
import torch

from sglang.srt.utils.common import ceil_div

if TYPE_CHECKING:
    from sglang.srt.configs.model_config import ModelConfig

# Fixed native LIM ABI dimensions, independent of model geometry.
MISS_CAPACITY = 32768
MAX_QUERY_WIDTH = 14

_SLOT_MAP_DTYPE = torch.int32
_SLOT_MAP_EXTRA_ROWS = 1


class BufferSpec(msgspec.Struct, frozen=True):
    """Shape, dtype and multiplicity of a persistent tensor allocation."""

    shape: tuple[int, ...]
    dtype: torch.dtype
    count: int = 1

    @property
    def nbytes(self) -> int:
        return prod(self.shape) * self.dtype.itemsize * self.count

    def allocate(self, device, fill_value=None) -> torch.Tensor:
        """Allocate one tensor using this spec, optionally initializing it."""
        if fill_value is not None:
            return torch.full(self.shape, fill_value, dtype=self.dtype, device=device)
        return torch.empty(self.shape, dtype=self.dtype, device=device)


class SparseKVHBMSizing(msgspec.Struct, frozen=True, kw_only=True):
    """Final HBM sizing for the startup allocation log."""

    available_bytes: int
    planned_bytes: int
    index_k_and_draft_bytes: int
    full_context_bytes_per_request: int


class BudgetPlan:
    """Persistent sparse KV buffer geometry tied to one Indexer K pool."""

    def __init__(
        self,
        *,
        request_rows: int,
        context_len: int,
        num_layers: int,
        top_k: int,
        kv_dtype: torch.dtype,
        copy_sfa_query_heads: int,
        max_query_width: int,
        cache_tokens: int,
        page_size: int,
        kv_head_num: int,
        kv_lora_rank: int,
        qk_rope_head_dim: int,
        index_head_dim: int,
        prefill_workspace_tokens: int = 0,
        resident_rows: int | None = None,
    ) -> None:
        resident_rows = request_rows if resident_rows is None else resident_rows
        if not 1 <= resident_rows <= request_rows:
            raise ValueError("Resident rows must be between 1 and request rows.")
        self.request_rows = request_rows
        self.resident_rows = resident_rows
        self.context_len = context_len
        self.num_layers = num_layers
        self.top_k = top_k
        self.kv_dtype = kv_dtype
        self.copy_sfa_query_heads = copy_sfa_query_heads
        self.max_query_width = max_query_width
        self.cache_tokens = cache_tokens
        self.page_size = page_size
        self.kv_head_num = kv_head_num
        self.index_head_dim = index_head_dim
        self.kv_cache_dim = kv_lora_rank + qk_rope_head_dim
        self.prefill_workspace_tokens = prefill_workspace_tokens
        metadata_dtype = torch.int32
        self.resident_buffers = {
            "device_k_buffer": BufferSpec(
                (resident_rows, cache_tokens, kv_head_num, kv_lora_rank),
                kv_dtype,
                num_layers,
            ),
            "device_v_buffer": BufferSpec(
                (resident_rows, cache_tokens, kv_head_num, qk_rope_head_dim),
                kv_dtype,
                num_layers,
            ),
            "device_slot_map": BufferSpec(
                (resident_rows + _SLOT_MAP_EXTRA_ROWS, context_len),
                _SLOT_MAP_DTYPE,
                num_layers,
            ),
            "short_resident_end": BufferSpec(
                (num_layers, request_rows), metadata_dtype
            ),
            "last_prefix": BufferSpec((resident_rows + 1,), metadata_dtype),
        }
        if resident_rows < request_rows:
            self.resident_buffers["request_to_resident"] = BufferSpec(
                (request_rows,), metadata_dtype
            )
        self.manager_buffers = {
            "planner_hbm_block_table": BufferSpec(
                (resident_rows, cache_tokens // page_size), metadata_dtype
            ),
        }
        if prefill_workspace_tokens:
            self.manager_buffers.update(
                prefill_nope=BufferSpec(
                    (prefill_workspace_tokens, kv_head_num, kv_lora_rank), kv_dtype
                ),
                prefill_rope=BufferSpec(
                    (prefill_workspace_tokens, kv_head_num, qk_rope_head_dim), kv_dtype
                ),
            )
        self.adapter_buffers = {
            "source_ids": BufferSpec(
                (resident_rows * max_query_width, top_k), metadata_dtype
            ),
            "device_slots": BufferSpec(
                (resident_rows * max_query_width, top_k), metadata_dtype
            ),
            "topk_misses": BufferSpec(
                (resident_rows * max_query_width,), metadata_dtype
            ),
            "miss_sources": BufferSpec((resident_rows, MISS_CAPACITY), metadata_dtype),
            "miss_slots": BufferSpec((resident_rows, MISS_CAPACITY), metadata_dtype),
            "miss_counts": BufferSpec((resident_rows,), metadata_dtype),
            "query_ends": BufferSpec((max_query_width, resident_rows), metadata_dtype),
            "block_table": BufferSpec(
                (resident_rows, context_len // page_size), metadata_dtype
            ),
        }
        for name in (
            "request_safe_slots",
            "resident_safe_slots",
            "_pre_lengths",
            "_pre_states",
            "_pre_next_prefix",
            "_pre_cache_sizes",
            "actual_seq_lengths_kv",
        ):
            self.adapter_buffers[name] = BufferSpec((resident_rows,), metadata_dtype)
        for name in ("_pre_sparse", "_pre_valid"):
            self.adapter_buffers[name] = BufferSpec((resident_rows,), torch.bool)
        self.device_kv_bytes = sum(
            self.resident_buffers[n].nbytes
            for n in ("device_k_buffer", "device_v_buffer")
        )
        self.slot_map_bytes = self.resident_buffers["device_slot_map"].nbytes
        self.prefill_workspace_bytes = sum(
            s.nbytes
            for n, s in self.manager_buffers.items()
            if n.startswith("prefill_")
        )
        self.adapter_bytes = sum(s.nbytes for s in self.adapter_buffers.values())
        self.host_token_capacity = None
        self.host_kv_spec = None

    def validate_batch_size(self, batch_size: int, mode: str) -> None:
        if batch_size > self.resident_rows:
            raise ValueError(
                f"Fused sparse KV {mode} batch size {batch_size} exceeds "
                f"allocated resident_rows={self.resident_rows}."
            )

    @property
    def managed_hbm_bytes(self) -> int:
        """Reserve HBM for resident KV, slot maps, LIM metadata, and prefill."""
        return (
            self.device_kv_bytes
            + self.slot_map_bytes
            + self.prefill_workspace_bytes
            + self.adapter_bytes
            + self.resident_buffers["last_prefix"].nbytes
        )

    def full_context_bytes_per_request(
        self, index_k_and_draft_bytes_per_token: int
    ) -> int:
        return get_sparse_kv_full_context_bytes_per_request(
            cache_tokens=self.cache_tokens,
            context_len=self.context_len,
            resident_kv_bytes_per_token=(
                self.num_layers
                * self.kv_head_num
                * self.kv_cache_dim
                * self.kv_dtype.itemsize
            ),
            index_k_and_draft_bytes_per_token=index_k_and_draft_bytes_per_token,
            slot_map_bytes_per_request=_slot_map_row_bytes(
                self.context_len, self.num_layers
            ),
        )

    @property
    def host_kv_bytes(self) -> int:
        """Report the physical Host allocation after binding the Indexer K pool."""
        if self.host_kv_spec is None:
            raise RuntimeError("Bind the Indexer K pool before reading Host KV bytes.")
        return self.host_kv_spec.nbytes

    def without_prefill_workspace(self) -> BudgetPlan:
        """Return an unbound plan without the optional workspace."""
        result = copy(self)
        result.manager_buffers = {
            name: spec
            for name, spec in self.manager_buffers.items()
            if name not in ("prefill_nope", "prefill_rope")
        }
        result.prefill_workspace_tokens = 0
        result.prefill_workspace_bytes = 0
        return result

    def bind_index_k_pool(self, index_k_pool) -> BudgetPlan:
        """Bind the Host KV layout to the physical Indexer K pool."""
        indexer_layer_ids = getattr(
            index_k_pool, "indexer_layer_ids", (index_k_pool.start_layer,)
        )
        if not indexer_layer_ids:
            raise ValueError("Native LIM requires at least one indexer K layer.")
        key = index_k_pool.get_index_k_buffer(indexer_layer_ids[0])
        if (
            key.dtype != torch.bfloat16
            or key.ndim != 4
            or tuple(key.shape[1:])
            != (self.page_size, self.kv_head_num, self.index_head_dim)
        ):
            raise ValueError(
                "Indexer K pool layout does not match the model budget plan."
            )
        # The page count already includes the Indexer K allocator's dummy page.
        host_token_capacity = key.shape[0] * self.page_size
        self.host_token_capacity = host_token_capacity
        self.host_kv_spec = BufferSpec(
            (self.host_token_capacity, self.kv_head_num, self.kv_cache_dim),
            self.kv_dtype,
            self.num_layers,
        )
        return self


def resolve_sparse_kv_budget(
    *,
    model_config: ModelConfig,
    page_size: int,
    request_rows: int,
    context_len: int,
    num_layers: int,
    top_k: int,
    kv_bytes_per_token: int,
    copy_sfa_query_heads: int,
    kv_dtype: torch.dtype = torch.bfloat16,
    cache_tokens: int = 0,
    max_query_width: int = 1,
    prefill_workspace_tokens: int = 0,
    resident_rows: int | None = None,
) -> BudgetPlan:
    """Resolve and validate native persistent allocations before pool sizing."""
    kv_head_num = model_config.mla_kv_head_num
    kv_lora_rank = model_config.kv_lora_rank
    qk_rope_head_dim = model_config.qk_rope_head_dim
    index_head_dim = model_config.index_head_dim
    if context_len > 2**21:
        raise ValueError("Native LIM requires context length <= 2**21.")
    if not (1 <= copy_sfa_query_heads <= 64 or copy_sfa_query_heads == 128):
        raise ValueError("Invalid native LIM query geometry.")
    if (
        kv_dtype != torch.bfloat16
        or kv_bytes_per_token
        != kv_head_num * (kv_lora_rank + qk_rope_head_dim) * kv_dtype.itemsize
    ):
        raise ValueError("Native LIM requires BF16 MLA KV storage.")
    cache_tokens = get_sparse_kv_cache_tokens(cache_tokens, max_query_width, top_k)
    if (
        cache_tokens < max_query_width * top_k
        or cache_tokens + page_size > MISS_CAPACITY
        or cache_tokens % page_size
    ):
        raise ValueError(
            f"Native LIM cache capacity must cover query width * TopK, be <= {MISS_CAPACITY - page_size} and aligned to {page_size}."
        )
    return BudgetPlan(
        request_rows=request_rows,
        resident_rows=resident_rows,
        context_len=ceil_div(context_len, page_size) * page_size,
        num_layers=num_layers,
        top_k=top_k,
        kv_dtype=kv_dtype,
        copy_sfa_query_heads=copy_sfa_query_heads,
        max_query_width=max_query_width,
        cache_tokens=cache_tokens,
        page_size=page_size,
        kv_head_num=kv_head_num,
        kv_lora_rank=kv_lora_rank,
        qk_rope_head_dim=qk_rope_head_dim,
        index_head_dim=index_head_dim,
        prefill_workspace_tokens=prefill_workspace_tokens,
    )


def get_sparse_kv_cache_tokens(
    configured_cache_tokens: int, max_query_width: int, top_k: int
) -> int:
    """Return the configured resident capacity or the default query union size."""
    if not 1 <= max_query_width <= MAX_QUERY_WIDTH:
        raise ValueError("Invalid native LIM query geometry.")
    return configured_cache_tokens or max_query_width * top_k


def _slot_map_row_bytes(context_len: int, num_layers: int) -> int:
    return context_len * num_layers * _SLOT_MAP_DTYPE.itemsize


def get_sparse_kv_full_context_bytes_per_request(
    *,
    cache_tokens: int,
    context_len: int,
    resident_kv_bytes_per_token: int,
    index_k_and_draft_bytes_per_token: int,
    slot_map_bytes_per_request: int,
) -> int:
    """Estimate resident KV, full-context Indexer K and draft KV, and the slot map."""
    return (
        cache_tokens * resident_kv_bytes_per_token
        + context_len * index_k_and_draft_bytes_per_token
        + slot_map_bytes_per_request
    )


def fit_sparse_running_requests(
    available_bytes: int,
    *,
    max_total_tokens: int | None,
    cache_tokens: int,
    context_len: int,
    num_layers: int,
    kv_bytes_per_token: int,
    index_k_and_draft_bytes_per_token: int,
) -> int:
    """Fit full-context requests under the HBM and token limits."""
    slot_map_bytes_per_request = _slot_map_row_bytes(context_len, num_layers)
    bytes_per_request = get_sparse_kv_full_context_bytes_per_request(
        cache_tokens=cache_tokens,
        context_len=context_len,
        resident_kv_bytes_per_token=num_layers * kv_bytes_per_token,
        index_k_and_draft_bytes_per_token=index_k_and_draft_bytes_per_token,
        slot_map_bytes_per_request=slot_map_bytes_per_request,
    )
    # The map includes the dummy resident row and one additional row.
    fixed_slot_map_bytes = (1 + _SLOT_MAP_EXTRA_ROWS) * slot_map_bytes_per_request
    fitting_rows = (available_bytes - fixed_slot_map_bytes) // bytes_per_request
    if max_total_tokens is not None:
        fitting_rows = min(fitting_rows, max_total_tokens // context_len)
    if fitting_rows < 1:
        raise RuntimeError(
            "Cannot fit one sparse KV resident row: "
            f"need {bytes_per_request + fixed_slot_map_bytes} bytes including fixed "
            f"slot-map rows, have {available_bytes} bytes (C={cache_tokens}). "
            f"One full sparse KV context requires context_len={context_len} tokens "
            f"after page alignment (max_total_tokens={max_total_tokens})."
        )
    return fitting_rows
