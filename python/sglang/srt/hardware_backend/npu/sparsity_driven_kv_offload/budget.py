"""Persistent allocations for the native LIM and CopySFA offload path."""

from dataclasses import dataclass

MAX_QUERY_WIDTH = 14


def sparse_batch_capacity(request_rows: int, graph_batch_capacity: int = 0) -> int:
    if request_rows <= 0 or graph_batch_capacity < 0:
        raise ValueError("Sparse KV capacities must be positive/nonnegative.")
    return max(request_rows, graph_batch_capacity)


@dataclass(frozen=True)
class SparseKVBudget:
    cache_tokens: int
    device_kv_bytes: int
    slot_map_bytes: int
    policy_bytes: int
    selected_kv_bytes: int
    host_kv_bytes: int
    adapter_bytes: int = 0
    copy_sfa_bytes: int = 0

    @property
    def managed_hbm_bytes(self) -> int:
        return (
            self.device_kv_bytes
            + self.slot_map_bytes
            + self.adapter_bytes
            + self.copy_sfa_bytes
        )


def resolve_sparse_kv_budget(
    *,
    request_rows: int,
    context_len: int,
    num_layers: int,
    top_k: int,
    kv_bytes_per_token: int,
    cache_tokens: int = 0,
    hbm_budget_bytes: int = 0,
    copy_sfa_query_heads: int = 128,
    max_query_width: int = 1,
    graph_batch_capacity: int = 0,
    device_tail_tokens: int = 0,
) -> SparseKVBudget:
    """Reserve the same native buffers allocated by the manager and adapter."""
    for name, value in (
        ("request_rows", request_rows),
        ("context_len", context_len),
        ("num_layers", num_layers),
        ("kv_bytes_per_token", kv_bytes_per_token),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{name} must be a positive integer.")
    if top_k != 2048 or context_len > 2**21:
        raise ValueError("Native LIM requires K=2048 and context length <= 2**21.")
    if not 1 <= max_query_width <= MAX_QUERY_WIDTH:
        raise ValueError(f"Native LIM query width must be in [1,{MAX_QUERY_WIDTH}].")
    if not 1 <= copy_sfa_query_heads <= 128:
        raise ValueError("CopySFA padded query heads must be in [1,128].")
    if device_tail_tokens not in (0, 256):
        raise ValueError("A causal HBM tail requires 256 slots.")
    if cache_tokens < 0 or hbm_budget_bytes < 0:
        raise ValueError("Cache size and HBM budget must be nonnegative.")
    context_len = (context_len + 127) // 128 * 128
    minimum = max_query_width * top_k
    capacity = cache_tokens or minimum
    if capacity < minimum or capacity > 32640 or capacity % 128:
        raise ValueError(
            f"Native LIM requires C >= query width * TopK ({minimum}), "
            "C <= 32640, aligned to 128."
        )
    batch = sparse_batch_capacity(request_rows, graph_batch_capacity)
    tokens = batch * max_query_width
    adapter = tokens * (2 * top_k * 4 + 4 + 64 * 4)
    adapter += batch * (
        2 * 32768 * 4 + 4 + context_len // 128 * 4 + max_query_width * 4 + 26
    )
    adapter += (num_layers + 1) * (request_rows + 1) * 4
    slot_map = num_layers * (request_rows + 1) * context_len * 4
    stride = capacity + device_tail_tokens
    copy_bytes = batch * stride * 17 + tokens * 9
    copy_bytes += num_layers * request_rows * 8 + request_rows * 12
    copy_bytes += tokens * copy_sfa_query_heads * 512 * (kv_bytes_per_token // 576)
    copy_bytes += batch * (2 + stride // 128 + context_len // 128) * 4
    result = SparseKVBudget(
        cache_tokens=capacity,
        device_kv_bytes=request_rows * num_layers * stride * kv_bytes_per_token,
        slot_map_bytes=slot_map,
        policy_bytes=0,
        selected_kv_bytes=0,
        host_kv_bytes=request_rows * num_layers * context_len * kv_bytes_per_token,
        adapter_bytes=adapter,
        copy_sfa_bytes=copy_bytes,
    )
    if hbm_budget_bytes and result.managed_hbm_bytes > hbm_budget_bytes:
        raise ValueError(
            f"Sparse KV manager needs {result.managed_hbm_bytes} bytes, exceeding "
            f"SGLANG_NPU_SPARSE_KV_HBM_BUDGET_MB ({hbm_budget_bytes} bytes)."
        )
    return result
