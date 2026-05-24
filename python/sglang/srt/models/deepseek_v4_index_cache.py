from dataclasses import dataclass
from typing import Optional

import torch


@dataclass(frozen=True)
class DSV4IndexCache:
    raw_indices: torch.Tensor


def make_index_cache_from_metadata(core_metadata) -> DSV4IndexCache:
    cache = DSV4IndexCache(raw_indices=core_metadata.c4_sparse_raw_indices)
    _check_cache_shape(cache)
    return cache


def assign_index_cache_to_metadata(
    index_cache: DSV4IndexCache, core_metadata, indexer_metadata
) -> Optional[torch.Tensor]:
    _check_cache_shape(index_cache)
    assert (
        core_metadata.c4_sparse_raw_indices is not None
    ), "raw index cache cannot be reused when c4_sparse_raw_indices is unavailable"
    assert core_metadata.c4_sparse_raw_indices.shape == index_cache.raw_indices.shape, (
        f"raw index cache shape mismatch: {core_metadata.c4_sparse_raw_indices.shape=} "
        f"{index_cache.raw_indices.shape=}"
    )
    assert (
        core_metadata.c4_sparse_page_indices.shape == index_cache.raw_indices.shape
    ), (
        f"physical index target shape mismatch: "
        f"{core_metadata.c4_sparse_page_indices.shape=} "
        f"{index_cache.raw_indices.shape=}"
    )
    core_metadata.c4_sparse_raw_indices = index_cache.raw_indices
    _translate_raw_indices_to_page_indices(
        index_cache.raw_indices,
        indexer_metadata.c4_seq_lens,
        core_metadata.page_table,
        core_metadata.c4_sparse_page_indices,
        indexer_metadata.c4_page_size,
    )
    return core_metadata.c4_sparse_raw_indices


def _check_cache_shape(index_cache: DSV4IndexCache) -> None:
    assert index_cache.raw_indices is not None, "raw index cache is required"
    assert (
        index_cache.raw_indices.dim() == 2
    ), f"expected raw index cache to be 2D, got {index_cache.raw_indices.shape=}"


def _translate_raw_indices_to_page_indices(
    raw_indices: torch.Tensor,
    seq_lens: torch.Tensor,
    page_table: torch.Tensor,
    out_page_indices: torch.Tensor,
    page_size: int,
) -> None:
    page_bits = (page_size - 1).bit_length() if page_size > 1 else 0
    page_mask = page_size - 1
    valid = raw_indices >= 0
    if seq_lens.dim() > 1:
        seq_lens = seq_lens.squeeze(-1)
    valid = valid & (raw_indices < seq_lens.unsqueeze(1))
    page_idx = torch.clamp(raw_indices >> page_bits, min=0, max=page_table.shape[1] - 1)
    offset_in_page = raw_indices & page_mask
    physical_pages = torch.gather(page_table, dim=1, index=page_idx.long())
    page_indices = ((physical_pages << page_bits) | offset_in_page).to(torch.int32)
    out_page_indices.copy_(
        torch.where(valid, page_indices, torch.full_like(page_indices, -1))
    )


def should_reuse_index_cache(
    skip_topk: Optional[bool],
    prev_index_cache,
    hisparse_coordinator,
) -> bool:
    return (
        skip_topk is True
        and prev_index_cache is not None
        and hisparse_coordinator is None
    )


def should_return_index_cache(
    next_skip_topk: Optional[bool],
    hisparse_coordinator,
) -> bool:
    return next_skip_topk is True and hisparse_coordinator is None


def index_cache_enabled_for_seq_lens(
    seq_lens: torch.Tensor,
    min_seq_len: int,
    compress_ratio: int = 4,
) -> bool:
    if min_seq_len <= 0:
        return True
    if seq_lens.numel() == 0:
        return False
    return int(seq_lens.max().item()) * compress_ratio >= min_seq_len


def should_disable_cuda_graph_for_index_cache_gate(
    config,
    seq_lens_cpu: Optional[torch.Tensor],
) -> bool:
    min_seq_len = getattr(config, "index_topk_min_seq_len", 0)
    if min_seq_len <= 0:
        return False
    if (
        getattr(config, "index_topk_pattern", None) is None
        and getattr(config, "index_topk_freq", 1) <= 1
    ):
        return False
    if seq_lens_cpu is None or seq_lens_cpu.numel() == 0:
        return True
    return int(seq_lens_cpu.max().item()) * 4 < min_seq_len


def get_c4_layer_ids(config) -> list[int]:
    return [
        idx
        for idx, ratio in enumerate(getattr(config, "compress_ratios", []))
        if ratio == 4
    ]


def _validate_index_topk_pattern(index_topk_pattern: str) -> None:
    invalid = sorted(set(index_topk_pattern) - {"F", "S"})
    if invalid:
        raise ValueError(
            f"index_topk_pattern contains unsupported entries {invalid}; "
            'expected only "F" and "S"'
        )


def _validate_first_c4_layer_is_full(
    index_topk_pattern: str, c4_layer_ids: list[int]
) -> None:
    if len(index_topk_pattern) == len(c4_layer_ids):
        first_c4_entry = index_topk_pattern[0]
    else:
        first_c4_entry = index_topk_pattern[c4_layer_ids[0]]
    if first_c4_entry != "F":
        raise ValueError("index_topk_pattern must keep the first C4 layer as F")


def get_index_cache_policy(
    config,
    layer_id: int,
    compress_ratio: int,
    is_nextn: bool = False,
) -> tuple[Optional[bool], Optional[bool]]:
    if is_nextn or compress_ratio != 4:
        return None, None

    c4_layer_ids = get_c4_layer_ids(config)
    try:
        c4_index = c4_layer_ids.index(layer_id)
    except ValueError:
        return None, None
    next_c4_layer_id = (
        c4_layer_ids[c4_index + 1] if c4_index < len(c4_layer_ids) - 1 else None
    )

    index_topk_pattern = getattr(config, "index_topk_pattern", None)
    index_topk_freq = getattr(config, "index_topk_freq", 1)
    if index_topk_pattern is None and index_topk_freq <= 1:
        return None, None

    if index_topk_pattern is not None:
        _validate_index_topk_pattern(index_topk_pattern)
        if len(index_topk_pattern) == len(c4_layer_ids):
            _validate_first_c4_layer_is_full(index_topk_pattern, c4_layer_ids)
            skip_topk = index_topk_pattern[c4_index] == "S"
            next_skip_topk = (
                c4_index < len(index_topk_pattern) - 1
                and index_topk_pattern[c4_index + 1] == "S"
            )
            return skip_topk, next_skip_topk

        if c4_layer_ids[0] >= len(index_topk_pattern):
            raise ValueError(
                f"index_topk_pattern length {len(index_topk_pattern)} "
                f"does not cover first C4 layer {c4_layer_ids[0]}"
            )
        if layer_id >= len(index_topk_pattern):
            raise ValueError(
                f"index_topk_pattern length {len(index_topk_pattern)} "
                f"does not cover C4 layer {layer_id}"
            )
        if next_c4_layer_id is not None and next_c4_layer_id >= len(index_topk_pattern):
            raise ValueError(
                f"index_topk_pattern length {len(index_topk_pattern)} "
                f"does not cover next C4 layer {next_c4_layer_id}"
            )
        _validate_first_c4_layer_is_full(index_topk_pattern, c4_layer_ids)
        skip_topk = index_topk_pattern[layer_id] == "S"
        next_skip_topk = (
            next_c4_layer_id is not None and index_topk_pattern[next_c4_layer_id] == "S"
        )
        return skip_topk, next_skip_topk

    skip_topk = c4_index % index_topk_freq != 0
    next_skip_topk = (
        c4_index < len(c4_layer_ids) - 1 and (c4_index + 1) % index_topk_freq != 0
    )
    return skip_topk, next_skip_topk
