"""Shared request-map geometry and a CPU-only joint KV/request budget solver."""
from typing import Callable


def request_mapping_shape(size: int, max_context_len: int) -> tuple[int, int]:
    if size < 0 or max_context_len <= 0:
        raise ValueError("Invalid request mapping capacity")
    # Slot zero is the actual pool's padded-request row.
    return size + 1, max_context_len


def request_mapping_bytes(size: int, max_context_len: int) -> int:
    rows, columns = request_mapping_shape(size, max_context_len)
    return rows * columns * 4  # req_to_token is torch.int32 in both pool classes.


def solve_request_mapping_budget(
    *,
    budget_bytes: int,
    token_ceiling: int,
    page_size: int,
    mapping_bytes_for_tokens: Callable[[int], int],
    kv_capacity_for_budget: Callable[[int], int],
) -> int:
    """Largest aligned capacity whose KV and request map fit the byte budget.

    Request count/map cost is nondecreasing with token capacity; the existing
    KV calculator is nondecreasing with its byte budget. Thus feasibility is a
    prefix of page counts. Callbacks must be local CPU calculations, never
    distributed collectives. A zero KV capacity means that budget cannot fit
    one page; other calculator failures must propagate.
    """
    if budget_bytes < 0 or token_ceiling <= 0 or page_size <= 0:
        raise ValueError("Invalid initial KV/request budget")
    lower, upper = 0, token_ceiling // page_size
    while lower < upper:
        candidate = (lower + upper + 1) // 2
        tokens = candidate * page_size
        mapping_bytes = mapping_bytes_for_tokens(tokens)
        if mapping_bytes < 0:
            raise ValueError("Negative request mapping cost")
        remaining = budget_bytes - mapping_bytes
        if remaining >= 0 and kv_capacity_for_budget(remaining) >= tokens:
            lower = candidate
        else:
            upper = candidate - 1
    if lower == 0:
        raise RuntimeError("GPU budget cannot fit a request mapping and one KV page")
    return lower * page_size
