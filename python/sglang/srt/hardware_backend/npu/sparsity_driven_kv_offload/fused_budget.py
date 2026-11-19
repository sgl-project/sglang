"""Plan persistent allocations for native LIM and CopySFA offload."""

from __future__ import annotations

import logging
from copy import copy
from dataclasses import dataclass
from math import prod
from typing import TYPE_CHECKING

import torch

from sglang.srt.utils.common import ceil_div

if TYPE_CHECKING:
    from sglang.srt.configs.model_config import ModelConfig

# Fixed native LIM ABI dimensions, independent of model geometry.
logger = logging.getLogger(__name__)

MISS_CAPACITY = 32768
MAX_QUERY_WIDTH = 14
C8_ROW_BYTES = 656


@dataclass(frozen=True)
class BufferSpec:
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


class BudgetPlan:
    """Persistent buffer geometry and accounting for one native KV pool."""

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
        batch_capacity: int,
        cache_tokens: int,
        page_size: int,
        kv_lora_rank: int,
        qk_rope_head_dim: int,
        index_head_dim: int,
        prefill_workspace_tokens: int = 0,
        resident_rows: int | None = None,
        packed_c8: bool = False,
    ) -> None:
        resident_rows = request_rows if resident_rows is None else resident_rows
        if not 1 <= resident_rows <= request_rows:
            raise ValueError("Resident rows must be between 1 and request rows.")
        self.request_rows = request_rows
        self.resident_rows = resident_rows
        self.packed_c8 = packed_c8
        self.context_len = context_len
        self.num_layers = num_layers
        self.top_k = top_k
        self.kv_dtype = kv_dtype
        self.copy_sfa_query_heads = copy_sfa_query_heads
        self.max_query_width = max_query_width
        self.batch_capacity = batch_capacity
        self.cache_tokens = cache_tokens
        self.page_size = page_size
        self.index_head_dim = index_head_dim
        self.kv_cache_dim = kv_lora_rank + qk_rope_head_dim
        self.prefill_workspace_tokens = prefill_workspace_tokens
        metadata_dtype = torch.int32
        self.manager_buffers = {
            "device_kv_buffer": BufferSpec(
                (resident_rows, cache_tokens, 1, kv_lora_rank),
                kv_dtype,
                num_layers,
            ),
            "device_rope_buffer": BufferSpec(
                (resident_rows, cache_tokens, 1, qk_rope_head_dim),
                kv_dtype,
                num_layers,
            ),
            "device_slot_map": BufferSpec(
                (resident_rows + 1, context_len), metadata_dtype, num_layers
            ),
            "resident_begin": BufferSpec((num_layers, request_rows), metadata_dtype),
            "resident_end": BufferSpec((num_layers, request_rows), metadata_dtype),
            "planner_hbm_block_table": BufferSpec(
                (batch_capacity, cache_tokens // page_size), metadata_dtype
            ),
        }
        if packed_c8:
            if resident_rows <= 1:
                raise ValueError("A5 C8 requires at least one live resident row.")
            self.manager_buffers["device_kv_buffer"] = BufferSpec(
                (resident_rows - 1, cache_tokens, 1, C8_ROW_BYTES),
                torch.int8,
                num_layers,
            )
            del self.manager_buffers["device_rope_buffer"]
            # The final column is a write sink for inactive planner lanes.
            self.manager_buffers["device_slot_map"] = BufferSpec(
                (resident_rows + 1, context_len + 1), metadata_dtype, num_layers
            )
            self.manager_buffers["slot_to_source"] = BufferSpec(
                (resident_rows + 1, cache_tokens + 1), metadata_dtype, num_layers
            )
        if resident_rows < request_rows:
            self.manager_buffers["request_to_resident"] = BufferSpec(
                (request_rows,), metadata_dtype
            )
        if prefill_workspace_tokens:
            self.manager_buffers.update(
                prefill_nope=BufferSpec(
                    (prefill_workspace_tokens, 1, kv_lora_rank), kv_dtype
                ),
                prefill_rope=BufferSpec(
                    (prefill_workspace_tokens, 1, qk_rope_head_dim), kv_dtype
                ),
            )
        self.adapter_buffers = {
            "source_ids": BufferSpec(
                (batch_capacity * max_query_width, top_k), metadata_dtype
            ),
            "device_slots": BufferSpec(
                (batch_capacity * max_query_width, top_k), metadata_dtype
            ),
            "topk_misses": BufferSpec(
                (batch_capacity * max_query_width,), metadata_dtype
            ),
            "miss_sources": BufferSpec((batch_capacity, MISS_CAPACITY), metadata_dtype),
            "miss_slots": BufferSpec((batch_capacity, MISS_CAPACITY), metadata_dtype),
            "miss_counts": BufferSpec((batch_capacity,), metadata_dtype),
            "query_ends": BufferSpec((max_query_width, batch_capacity), metadata_dtype),
            "block_table": BufferSpec(
                (batch_capacity, context_len // page_size), metadata_dtype
            ),
            "last_prefix": BufferSpec((num_layers, resident_rows + 1), metadata_dtype),
        }
        # Per batch row: six int32 metadata arrays and two boolean masks.
        for name in (
            "request_safe_slots",
            "resident_safe_slots",
            "_pre_lengths",
            "_pre_states",
            "_pre_next_prefix",
            "_pre_cache_sizes",
        ):
            self.adapter_buffers[name] = BufferSpec((batch_capacity,), metadata_dtype)
        for name in ("_pre_sparse", "_pre_valid"):
            self.adapter_buffers[name] = BufferSpec((batch_capacity,), torch.bool)
        self.device_kv_bytes = sum(
            spec.nbytes
            for name, spec in self.manager_buffers.items()
            if name in ("device_kv_buffer", "device_rope_buffer")
        )
        self.slot_map_bytes = self.manager_buffers["device_slot_map"].nbytes
        # Reserve one maximal forward output without allocating it persistently.
        self.output_scratch_bytes = BufferSpec(
            (batch_capacity * max_query_width, copy_sfa_query_heads, kv_lora_rank),
            torch.bfloat16 if packed_c8 else kv_dtype,
        ).nbytes
        self.copy_sfa_bytes = sum(
            s.nbytes
            for n, s in self.manager_buffers.items()
            if n not in ("device_kv_buffer", "device_rope_buffer", "device_slot_map")
            and not n.startswith("prefill_")
        )
        self.prefill_workspace_bytes = sum(
            s.nbytes
            for n, s in self.manager_buffers.items()
            if n.startswith("prefill_")
        )
        self.adapter_bytes = sum(s.nbytes for s in self.adapter_buffers.values())
        self.host_token_capacity = None
        self.host_kv_spec = None

    @property
    def managed_hbm_bytes(self) -> int:
        return (
            self.device_kv_bytes
            + self.slot_map_bytes
            + self.copy_sfa_bytes
            + self.output_scratch_bytes
            + self.prefill_workspace_bytes
            + self.adapter_bytes
        )

    @property
    def host_kv_bytes(self) -> int:
        """Report the physical Host allocation only after native pool binding."""
        if self.host_kv_spec is None:
            raise RuntimeError("Bind the native KV pool before reading Host KV bytes.")
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

    def bind_native_pool(self, pool) -> BudgetPlan:
        """Bind the physical layout once after native pool creation."""
        indexer_layer_ids = getattr(pool, "indexer_layer_ids", (pool.start_layer,))
        if not indexer_layer_ids:
            raise ValueError("Native LIM requires at least one indexer K layer.")
        key = pool.get_index_k_buffer(indexer_layer_ids[0])
        if (
            key.dtype != (torch.float8_e4m3fn if self.packed_c8 else torch.bfloat16)
            or key.ndim != 4
            or tuple(key.shape[1:]) != (self.page_size, 1, self.index_head_dim)
        ):
            raise ValueError("Native index keys do not match the model budget plan.")
        # get_index_k_buffer returns one layer: [pages, page_size, 1, head_dim].
        # Its pages already include the native allocator's dummy page.
        host_token_capacity = key.shape[0] * self.page_size
        self.host_token_capacity = host_token_capacity
        self.host_kv_spec = BufferSpec(
            (
                1,
                self.host_token_capacity,
                1,
                C8_ROW_BYTES if self.packed_c8 else self.kv_cache_dim,
            ),
            torch.int8 if self.packed_c8 else self.kv_dtype,
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
    hbm_budget_bytes: int = 0,
    max_query_width: int = 1,
    graph_batch_capacity: int = 0,
    prefill_workspace_tokens: int = 0,
    resident_rows: int | None = None,
) -> BudgetPlan:
    """Resolve and validate native persistent allocations before pool sizing."""
    kv_lora_rank = model_config.kv_lora_rank
    qk_rope_head_dim = model_config.qk_rope_head_dim
    index_head_dim = model_config.index_head_dim
    packed_c8 = kv_dtype == torch.float8_e4m3fn
    if context_len > 2**21:
        raise ValueError("Native LIM requires context length <= 2**21.")
    if not 1 <= max_query_width <= MAX_QUERY_WIDTH or not (
        1 <= copy_sfa_query_heads <= 64 or copy_sfa_query_heads == 128
    ):
        raise ValueError("Invalid native LIM query geometry.")
    if packed_c8:
        if (
            (kv_lora_rank, qk_rope_head_dim, index_head_dim, page_size, top_k)
            != (512, 64, 128, 128, 2048)
            or kv_bytes_per_token != C8_ROW_BYTES
            or context_len > 262144
            or copy_sfa_query_heads not in (8, 16, 32, 64, 128)
        ):
            raise ValueError("Invalid A5 packed C8 geometry.")
        prefill_workspace_tokens = 0
    elif (
        kv_dtype != torch.bfloat16
        or kv_bytes_per_token != (kv_lora_rank + qk_rope_head_dim) * kv_dtype.itemsize
    ):
        raise ValueError("Native LIM requires BF16 MLA KV storage.")
    capacity = cache_tokens or max_query_width * top_k
    if (
        capacity < max_query_width * top_k
        or capacity + page_size > MISS_CAPACITY
        or capacity % page_size
    ):
        raise ValueError(
            "Native LIM cache capacity must cover query width * TopK, be <= 32640 and aligned to 128."
        )
    resident_rows = request_rows if resident_rows is None else resident_rows
    plan_options = dict(
        request_rows=request_rows,
        resident_rows=resident_rows,
        context_len=ceil_div(context_len, page_size) * page_size,
        num_layers=num_layers,
        top_k=top_k,
        kv_dtype=kv_dtype,
        copy_sfa_query_heads=copy_sfa_query_heads,
        max_query_width=max_query_width,
        batch_capacity=max(resident_rows, graph_batch_capacity),
        cache_tokens=capacity,
        page_size=page_size,
        kv_lora_rank=kv_lora_rank,
        qk_rope_head_dim=qk_rope_head_dim,
        index_head_dim=index_head_dim,
        packed_c8=packed_c8,
    )
    base = BudgetPlan(**plan_options)
    workspace_bytes = sum(
        BufferSpec((prefill_workspace_tokens, 1, width), kv_dtype).nbytes
        for width in (kv_lora_rank, qk_rope_head_dim)
    )
    planned_bytes = base.managed_hbm_bytes + workspace_bytes
    if (
        prefill_workspace_tokens
        and hbm_budget_bytes
        and planned_bytes > hbm_budget_bytes
    ):
        logger.warning(
            "Disabling sparse prefill workspace (%d tokens): planned HBM %d bytes "
            "exceeds budget %d bytes.",
            prefill_workspace_tokens,
            planned_bytes,
            hbm_budget_bytes,
        )
        prefill_workspace_tokens = 0
    result = (
        BudgetPlan(**plan_options, prefill_workspace_tokens=prefill_workspace_tokens)
        if prefill_workspace_tokens
        else base
    )
    if hbm_budget_bytes and result.managed_hbm_bytes > hbm_budget_bytes:
        raise ValueError(
            f"Sparse KV manager needs {result.managed_hbm_bytes} bytes, exceeding SGLANG_NPU_SPARSE_KV_HBM_BUDGET_MB ({hbm_budget_bytes} bytes)."
        )
    return result
