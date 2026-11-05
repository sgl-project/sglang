"""Plan persistent allocations for native LIM and CopySFA offload."""

from dataclasses import dataclass
from math import prod

import torch

PAGE_SIZE = 128
INDEXER_HEAD_DIM = 128
INDEX_HEAD_CAPACITY = 64
MISS_CAPACITY = 32768
COPY_SFA_MAX_QUERY_HEADS = 128
MAX_CACHE_TOKENS = MISS_CAPACITY - PAGE_SIZE
MLA_KV_LORA_RANK = 512
MLA_ROPE_DIM = 64
TOP_K = 2048
MAX_QUERY_WIDTH = 14


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

    def validate(self, value, name: str) -> None:
        values = value if isinstance(value, list) else [value]
        if len(values) != self.count or any(
            tuple(t.shape) != self.shape or t.dtype != self.dtype for t in values
        ):
            raise ValueError(f"Sparse KV buffer {name} does not match {self}.")


class BudgetPlan:
    """Persistent buffer geometry and accounting for one native KV pool."""

    scale_dtype = torch.float32
    scale_element_size = torch.float32.itemsize

    def __init__(
        self,
        request_rows: int,
        context_len: int,
        num_layers: int,
        top_k: int,
        kv_dtype: torch.dtype,
        copy_sfa_query_heads: int,
        max_query_width: int,
        batch_capacity: int,
        cache_tokens: int,
    ) -> None:
        self.request_rows = request_rows
        self.context_len = context_len
        self.num_layers = num_layers
        self.top_k = top_k
        self.kv_dtype = kv_dtype
        self.copy_sfa_query_heads = copy_sfa_query_heads
        self.max_query_width = max_query_width
        self.batch_capacity = batch_capacity
        self.cache_tokens = cache_tokens
        index_dtype = torch.int32
        self.manager_buffers = {
            "device_kv_buffer": BufferSpec(
                (request_rows, cache_tokens, 1, MLA_KV_LORA_RANK + MLA_ROPE_DIM),
                kv_dtype,
                num_layers,
            ),
            "device_slot_map": BufferSpec(
                (request_rows + 1, context_len), index_dtype, num_layers
            ),
            "resident_begin": BufferSpec((num_layers, request_rows), index_dtype),
            "resident_end": BufferSpec((num_layers, request_rows), index_dtype),
            "planner_actual_seq_lengths_kv": BufferSpec((batch_capacity,), index_dtype),
            "planner_num_cache_tokens": BufferSpec((batch_capacity,), index_dtype),
            "planner_hbm_block_table": BufferSpec(
                (batch_capacity, cache_tokens // PAGE_SIZE), index_dtype
            ),
            "planner_dram_block_table": BufferSpec(
                (batch_capacity, context_len // PAGE_SIZE), index_dtype
            ),
            "fused_attn_out": BufferSpec(
                (
                    batch_capacity * max_query_width,
                    copy_sfa_query_heads,
                    MLA_KV_LORA_RANK,
                ),
                kv_dtype,
            ),
        }
        self.adapter_buffers = {
            "source_ids": BufferSpec(
                (batch_capacity * max_query_width, 1, top_k), index_dtype
            ),
            "device_slots": BufferSpec(
                (batch_capacity * max_query_width, 1, top_k), index_dtype
            ),
            "topk_misses": BufferSpec((batch_capacity * max_query_width,), index_dtype),
            "miss_sources": BufferSpec((batch_capacity, MISS_CAPACITY), index_dtype),
            "miss_slots": BufferSpec((batch_capacity, MISS_CAPACITY), index_dtype),
            "miss_counts": BufferSpec((batch_capacity,), index_dtype),
            "query_scale": BufferSpec(
                (batch_capacity * max_query_width, INDEX_HEAD_CAPACITY),
                self.scale_dtype,
            ),
            "query_ends": BufferSpec((max_query_width, batch_capacity), index_dtype),
            "block_table": BufferSpec(
                (batch_capacity, context_len // PAGE_SIZE), index_dtype
            ),
            "last_prefix": BufferSpec((num_layers, request_rows + 1), index_dtype),
            "_metadata_prefix_scratch": BufferSpec((request_rows + 1,), index_dtype),
        }
        # Per batch row: six int32 metadata arrays and two boolean masks (26 bytes).
        for name in (
            "_pre_safe_slots",
            "_pre_lengths",
            "_pre_prefixes",
            "_pre_states",
            "_pre_next_prefix",
            "_pre_cache_sizes",
        ):
            self.adapter_buffers[name] = BufferSpec((batch_capacity,), index_dtype)
        for name in ("_pre_sparse", "_pre_valid"):
            self.adapter_buffers[name] = BufferSpec((batch_capacity,), torch.bool)
        self.device_kv_bytes = self.manager_buffers["device_kv_buffer"].nbytes
        self.slot_map_bytes = self.manager_buffers["device_slot_map"].nbytes
        self.copy_sfa_bytes = sum(
            s.nbytes
            for n, s in self.manager_buffers.items()
            if n not in ("device_kv_buffer", "device_slot_map")
        )
        self.adapter_bytes = sum(s.nbytes for s in self.adapter_buffers.values())
        self.host_kv_bytes = (
            request_rows
            * num_layers
            * context_len
            * (MLA_KV_LORA_RANK + MLA_ROPE_DIM)
            * kv_dtype.itemsize
        )
        self.host_token_capacity = None
        self.host_kv_spec = None
        self.key_scale_spec = None

    @property
    def managed_hbm_bytes(self) -> int:
        return (
            self.device_kv_bytes
            + self.slot_map_bytes
            + self.copy_sfa_bytes
            + self.adapter_bytes
        )

    def bind_native_pool(self, pool) -> "BudgetPlan":
        """Bind the native physical layout once, checking subsequent bindings."""
        if (
            pool.page_size != PAGE_SIZE
            or pool.layer_num != self.num_layers
            or pool.store_dtype != self.kv_dtype
            or (pool.kv_lora_rank, pool.qk_rope_head_dim)
            != (MLA_KV_LORA_RANK, MLA_ROPE_DIM)
        ):
            raise ValueError("Native KV pool does not match the sparse budget plan.")
        indexer_layer_ids = getattr(pool, "indexer_layer_ids", (pool.start_layer,))
        if not indexer_layer_ids:
            raise ValueError("Native LIM requires at least one indexer K layer.")
        key = pool.get_index_k_buffer(indexer_layer_ids[0])
        if (
            key.dtype != torch.bfloat16
            or key.ndim != 4
            or tuple(key.shape[1:]) != (PAGE_SIZE, 1, INDEXER_HEAD_DIM)
        ):
            raise ValueError("Native LIM requires BF16 index keys [pages,128,1,128].")
        host_token_capacity = (
            (int(pool.size) + int(pool.page_size) + PAGE_SIZE - 1) // PAGE_SIZE
        ) * PAGE_SIZE
        if key.shape[0] != host_token_capacity // PAGE_SIZE:
            raise ValueError(
                "Native index key pages do not match KV capacity and dummy page."
            )
        key_scale_spec = BufferSpec(tuple(key.shape[:3]), self.scale_dtype)
        if self.host_token_capacity is not None:
            if (
                self.host_token_capacity != host_token_capacity
                or self.key_scale_spec != key_scale_spec
            ):
                raise ValueError(
                    "Native KV pool differs from the already bound layout."
                )
            return self
        self.key_scale_spec = key_scale_spec
        self.host_token_capacity = host_token_capacity
        self.host_kv_spec = BufferSpec(
            (1, self.host_token_capacity, 1, MLA_KV_LORA_RANK + MLA_ROPE_DIM),
            self.kv_dtype,
            self.num_layers,
        )
        self.host_kv_bytes = (
            self.host_token_capacity
            * self.num_layers
            * (MLA_KV_LORA_RANK + MLA_ROPE_DIM)
            * self.kv_dtype.itemsize
        )
        return self

    def validate_buffers(self, owner, specs: dict[str, BufferSpec]) -> None:
        for name, spec in specs.items():
            spec.validate(getattr(owner, name), name)


def resolve_sparse_kv_budget(
    *,
    request_rows: int,
    context_len: int,
    num_layers: int,
    top_k: int,
    kv_bytes_per_token: int,
    kv_dtype: torch.dtype = torch.bfloat16,
    cache_tokens: int = 0,
    hbm_budget_bytes: int = 0,
    copy_sfa_query_heads: int = 128,
    max_query_width: int = 1,
    graph_batch_capacity: int = 0,
) -> BudgetPlan:
    """Resolve and validate native persistent allocations before pool sizing."""
    for name, value in (
        ("request_rows", request_rows),
        ("context_len", context_len),
        ("num_layers", num_layers),
        ("kv_bytes_per_token", kv_bytes_per_token),
        ("top_k", top_k),
        ("max_query_width", max_query_width),
        ("copy_sfa_query_heads", copy_sfa_query_heads),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{name} must be a positive integer.")
    for name, value in (
        ("cache_tokens", cache_tokens),
        ("hbm_budget_bytes", hbm_budget_bytes),
        ("graph_batch_capacity", graph_batch_capacity),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"{name} must be a nonnegative integer.")
    if top_k != TOP_K or context_len > 2**21:
        raise ValueError("Native LIM requires K=2048 and context length <= 2**21.")
    if (
        not 1 <= max_query_width <= MAX_QUERY_WIDTH
        or not 1 <= copy_sfa_query_heads <= COPY_SFA_MAX_QUERY_HEADS
    ):
        raise ValueError("Invalid native LIM query geometry.")
    if (
        kv_dtype != torch.bfloat16
        or kv_bytes_per_token != (MLA_KV_LORA_RANK + MLA_ROPE_DIM) * kv_dtype.itemsize
    ):
        raise ValueError("Native LIM requires BF16 MLA KV storage.")
    capacity = cache_tokens or max_query_width * top_k
    if (
        capacity < max_query_width * top_k
        or capacity > MAX_CACHE_TOKENS
        or capacity % PAGE_SIZE
    ):
        raise ValueError(
            "Native LIM cache capacity must cover query width * TopK, be <= 32640 and aligned to 128."
        )
    result = BudgetPlan(
        request_rows,
        (context_len + PAGE_SIZE - 1) // PAGE_SIZE * PAGE_SIZE,
        num_layers,
        top_k,
        kv_dtype,
        copy_sfa_query_heads,
        max_query_width,
        max(request_rows, graph_batch_capacity),
        capacity,
    )
    if hbm_budget_bytes and result.managed_hbm_bytes > hbm_budget_bytes:
        raise ValueError(
            f"Sparse KV manager needs {result.managed_hbm_bytes} bytes, exceeding SGLANG_NPU_SPARSE_KV_HBM_BUDGET_MB ({hbm_budget_bytes} bytes)."
        )
    return result
