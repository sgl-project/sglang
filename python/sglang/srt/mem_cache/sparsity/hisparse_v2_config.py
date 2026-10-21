from __future__ import annotations

import json
import math

import msgspec

from sglang.srt.runtime_context import get_memory, get_schedule


class HiSparseV2Config(msgspec.Struct, frozen=True, kw_only=True):
    top_k: int
    buffer_tokens: int
    expansion_ratio: float
    block_size: int


def resolve_hisparse_v2_query_steps(
    *,
    algorithm: str | None,
    num_draft_tokens: int | None,
    eagle_topk: int | None,
    adaptive: bool,
    multi_layer: bool,
) -> int:
    if algorithm is None or algorithm.upper() == "NONE":
        return 1
    if algorithm.upper() != "EAGLE" or multi_layer:
        raise ValueError("HiSparse V2 MTP requires a single EAGLE draft model")
    if (eagle_topk or 1) != 1:
        raise ValueError("HiSparse V2 MTP requires --speculative-eagle-topk 1")
    if adaptive:
        raise ValueError("HiSparse V2 MTP does not support speculative-adaptive")
    if type(num_draft_tokens) is not int or not 2 <= num_draft_tokens <= 4:
        raise ValueError("HiSparse V2 MTP requires 2-4 speculative draft tokens")
    return num_draft_tokens


def hisparse_v2_dsv4_host_tokens(full_size: int, page_size: int) -> int:
    """Host tokens HiCache gives a DeepSeek-V4 pool: the device FULL pages times
    --hicache-ratio, as _deepseek_v4_num_host_pages sizes it."""
    device_pages = (full_size + page_size - 1) // page_size
    return int(device_pages * get_memory().hicache_ratio) * page_size


def parse_hisparse_v2_config(*, model_top_k: int = 2048) -> HiSparseV2Config:
    memory = get_memory()
    return resolve_hisparse_v2_config(
        config=memory.hisparse_config,
        model_top_k=model_top_k,
        hicache_ratio=memory.hicache_ratio,
        page_size=get_schedule().page_size,
    )


def resolve_hisparse_v2_config(
    *, config: str | None, model_top_k: int, hicache_ratio: float, page_size: int
) -> HiSparseV2Config:
    raw = json.loads(config or "{}")
    if not isinstance(raw, dict):
        raise ValueError("hisparse_config must be a JSON object")
    top_k = raw.get("top_k", model_top_k)
    if top_k != model_top_k:
        raise ValueError("HiSparse V2 top_k must match the model's index_topk")
    buffer_tokens = raw.get("device_buffer_size", top_k)
    block_size = raw.get("swap_in_block_size", 960)
    for name, value in (
        ("top_k", top_k),
        ("device_buffer_size", buffer_tokens),
        ("swap_in_block_size", block_size),
    ):
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise ValueError(f"HiSparse V2 {name} must be a positive integer")
    if buffer_tokens < top_k or buffer_tokens > 32767:
        raise ValueError("HiSparse V2 device_buffer_size must be in [top_k, 32767]")
    if buffer_tokens % page_size:
        raise ValueError("HiSparse V2 device_buffer_size must be page-aligned")
    if block_size > 1024 or block_size % 32:
        raise ValueError(
            "HiSparse V2 swap_in_block_size must be a warp multiple <= 1024"
        )
    expansion = raw.get("expansion_ratio", hicache_ratio)
    if isinstance(expansion, bool) or not isinstance(expansion, (int, float)):
        raise ValueError("HiSparse V2 expansion_ratio must be numeric")
    if not math.isfinite(expansion) or expansion <= 0:
        raise ValueError("HiSparse V2 expansion_ratio must be finite and positive")
    return HiSparseV2Config(
        top_k=top_k,
        buffer_tokens=buffer_tokens,
        expansion_ratio=float(expansion),
        block_size=block_size,
    )
