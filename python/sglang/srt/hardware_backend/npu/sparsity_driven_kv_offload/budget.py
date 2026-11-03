"""Allocation accounting for the Ascend sparse KV manager (no device imports)."""

from dataclasses import dataclass


def sparse_batch_capacity(request_rows: int, graph_batch_capacity: int = 0) -> int:
    """Size batch workspaces without widening the valid request-ID range."""
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
            + self.policy_bytes
            + self.selected_kv_bytes
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
    indexer_impl: str = "legacy",
    copy_sfa: bool = False,
    copy_sfa_query_heads: int = 128,
    mtp_full_visible: bool = False,
    fused_copy_sfa_metadata: bool = False,
    fused_pre_metadata: bool = False,
    graph_batch_capacity: int = 0,
) -> SparseKVBudget:
    """Size persistent manager allocations, including padding and one workspace.

    A zero token setting defaults to K, or derives C from a nonzero byte budget.
    The budget excludes model/indexer KV, allocator overhead, operator temporaries
    and graph pools; those still need headroom in --mem-fraction-static.
    """
    for name, value in (
        ("request_rows", request_rows),
        ("context_len", context_len),
        ("num_layers", num_layers),
        ("top_k", top_k),
        ("kv_bytes_per_token", kv_bytes_per_token),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{name} must be a positive integer, got {value!r}.")
    for name, value in (
        ("cache_tokens", cache_tokens),
        ("hbm_budget_bytes", hbm_budget_bytes),
        ("graph_batch_capacity", graph_batch_capacity),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"{name} must be a nonnegative integer, got {value!r}.")

    if indexer_impl not in ("legacy", "li_reference", "lim"):
        raise ValueError(f"Unknown sparse indexer implementation: {indexer_impl}.")
    if copy_sfa:
        if top_k != 2048 or (
            cache_tokens != 0
            and (cache_tokens < 2048 or cache_tokens > 32640 or cache_tokens % 128)
        ):
            raise ValueError(
                "Copy-SFA requires K=2048 and C in [2048,32640], aligned to 128."
            )
        if cache_tokens > top_k and indexer_impl != "lim":
            raise ValueError("Copy-SFA with C>K requires the native LIM indexer.")
        if copy_sfa_query_heads <= 0 or copy_sfa_query_heads > 128:
            raise ValueError("Copy-SFA padded query heads must be in [1,128].")
        context_len = (context_len + 127) // 128 * 128
        cache_tokens = cache_tokens or top_k
    if mtp_full_visible and (not copy_sfa or indexer_impl != "lim"):
        raise ValueError("Experimental MTP requires native LIM and Copy-SFA.")
    if mtp_full_visible and cache_tokens < 4 * top_k:
        raise ValueError("Experimental Q4 MTP requires capacity >= 4 * TopK.")
    experimental = indexer_impl != "legacy"
    native = indexer_impl == "lim"
    batch_rows = sparse_batch_capacity(request_rows, graph_batch_capacity)
    map_width = (
        ((context_len + 127) // 128 * 128) if native else (context_len // 8 + 1) * 8
    )
    slot_map_bytes = num_layers * (request_rows + 1) * map_width * 4
    selected_kv_bytes = batch_rows * top_k * kv_bytes_per_token
    slot_ids_bytes = top_k * 8
    adapter_bytes = 0
    if experimental:
        if top_k != 2048 or context_len > 2**21:
            raise ValueError(
                "LIM/reference requires TopK=2048 and context length <= 2**21."
            )
        adapter_bytes = (
            batch_rows
            * (
                (0 if copy_sfa else (top_k + 128) * (kv_bytes_per_token + 4))
                + 2 * top_k * 4
                + 2 * 32768 * 4
                + 64 * 4
                + 3 * 4
                + ((context_len + 127) // 128) * 4
            )
            + num_layers * (request_rows + 1) * 4
            + 128 * 4
        )
    if native and copy_sfa:
        if fused_copy_sfa_metadata:
            adapter_bytes += (request_rows + 1 + batch_rows) * 4
        if fused_pre_metadata:
            adapter_bytes += batch_rows * (6 * 4 + 2)
    mtp_output_bytes = 0
    if mtp_full_visible:
        # Q4 source/slot rows, route counts, query scales and Q endpoints.
        adapter_bytes += batch_rows * 4 * (2 * top_k * 4 + 4 + 64 * 4)
        adapter_bytes += batch_rows * 4
        mtp_output_bytes = (
            batch_rows * 4 * copy_sfa_query_heads * 512 * (kv_bytes_per_token // 576)
        )
    auto_capacity = cache_tokens == 0
    if auto_capacity and hbm_budget_bytes:
        # Conservative LRU accounting: slot owners (int32), slot order (int64),
        # and the shared int64 initial-order template.
        per_slot = request_rows * num_layers * kv_bytes_per_token + (
            0 if native else (request_rows + 1) * num_layers * 12 + 8
        )
        cache_tokens = max(
            top_k,
            min(
                max(context_len, top_k),
                (
                    hbm_budget_bytes
                    - slot_map_bytes
                    - selected_kv_bytes
                    - slot_ids_bytes
                    - adapter_bytes
                    - mtp_output_bytes
                )
                // per_slot,
            ),
        )
    elif cache_tokens == 0:
        cache_tokens = top_k
    if experimental:
        if auto_capacity:
            cache_tokens = min(32640, cache_tokens // 128 * 128)
        if cache_tokens > 32640 or cache_tokens % 128:
            raise ValueError(
                "LIM/reference cache capacity must be a multiple of 128 and <= 32640."
            )
    if cache_tokens < top_k:
        raise ValueError(
            f"Sparse KV cache capacity ({cache_tokens}) must be >= top_k ({top_k}); "
            "increase SGLANG_NPU_SPARSE_KV_HBM_BUDGET_MB or "
            "SGLANG_NPU_SPARSE_KV_CACHE_SIZE."
        )
    policy_bytes = (
        ((request_rows + 1) * num_layers * 12 + 8) * cache_tokens
        if cache_tokens > top_k and not native
        else 0
    ) + slot_ids_bytes
    copy_sfa_bytes = 0
    if copy_sfa:
        copy_sfa_bytes = mtp_output_bytes + (
            num_layers * (request_rows + 1) * cache_tokens * 4
            + batch_rows * (2 * top_k + 2 * cache_tokens + 5) * 4
            + (batch_rows + 1) * ((cache_tokens + context_len) // 128) * 4
            + batch_rows * copy_sfa_query_heads * 512 * (kv_bytes_per_token // 576)
        )
    result = SparseKVBudget(
        cache_tokens=cache_tokens,
        device_kv_bytes=request_rows * num_layers * cache_tokens * kv_bytes_per_token,
        slot_map_bytes=slot_map_bytes,
        policy_bytes=policy_bytes,
        selected_kv_bytes=selected_kv_bytes,
        host_kv_bytes=request_rows * num_layers * context_len * kv_bytes_per_token,
        adapter_bytes=adapter_bytes,
        copy_sfa_bytes=copy_sfa_bytes,
    )
    if hbm_budget_bytes and result.managed_hbm_bytes > hbm_budget_bytes:
        raise ValueError(
            f"Sparse KV manager needs {result.managed_hbm_bytes} bytes, exceeding "
            f"SGLANG_NPU_SPARSE_KV_HBM_BUDGET_MB ({hbm_budget_bytes} bytes)."
        )
    return result
