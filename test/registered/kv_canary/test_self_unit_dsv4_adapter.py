from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional

import torch

from sglang.srt.kv_canary.buffer_group import PoolKind
from sglang.srt.kv_canary.pool_patch.adapters.dsv4 import (
    _build_full_group,
    _build_swa_group,
    _dsv4_packed_nope_rope_bytes_per_token,
)
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.kv_canary.fixtures import CPU_DEVICE

register_cuda_ci(est_time=10, stage="extra-a", runner_config="1-gpu-large")


_QK_NOPE_HEAD_DIM = 448
_QK_ROPE_HEAD_DIM = 64
_ROPE_STORAGE_DTYPE = torch.bfloat16
_NOPE_ROPE_BYTES_PER_TOKEN = 576
_BYTES_PER_TOKEN_TOTAL = 584
_INDEX_HEAD_DIM = 128


def _bytes_per_page_padded(*, page_size: int) -> int:
    return math.ceil(page_size * _BYTES_PER_TOKEN_TOTAL / 576) * 576


@dataclass
class FakeDsv4PackedSubPool:
    kv_buffer: List[torch.Tensor]
    page_size: int
    qk_nope_head_dim: int = _QK_NOPE_HEAD_DIM
    qk_rope_head_dim: int = _QK_ROPE_HEAD_DIM
    rope_storage_dtype: torch.dtype = _ROPE_STORAGE_DTYPE

    def get_bytes_per_token(self) -> int:
        return _BYTES_PER_TOKEN_TOTAL


@dataclass
class FakeDsv4IndexerSubPool:
    index_k_with_scale_buffer: List[torch.Tensor]
    page_size: int
    index_head_dim: int = _INDEX_HEAD_DIM


@dataclass
class FakeDsv4FullPool:
    c4_kv_pool: FakeDsv4PackedSubPool
    c4_indexer_kv_pool: FakeDsv4IndexerSubPool
    c128_kv_pool: FakeDsv4PackedSubPool
    swa_kv_pool: FakeDsv4PackedSubPool
    page_size: int
    full_to_swa_index_mapping: Optional[torch.Tensor] = None
    compress_state_pools: list = field(default_factory=list)
    indexer_compress_state_pools: list = field(default_factory=list)


def _make_packed_buffer(*, num_pages: int, page_size: int) -> torch.Tensor:
    return torch.zeros(
        num_pages,
        _bytes_per_page_padded(page_size=page_size),
        dtype=torch.uint8,
        device=CPU_DEVICE,
    )


def _make_indexer_buffer(
    *, num_pages: int, page_size: int, index_head_dim: int
) -> torch.Tensor:
    num_scales = index_head_dim // 128
    page_bytes = page_size * index_head_dim + page_size * num_scales * 4
    return torch.zeros(num_pages, page_bytes, dtype=torch.uint8, device=CPU_DEVICE)


def _make_pool(
    *, page_size: int = 128, num_full_pages: int = 4, num_swa_pages: int = 2
) -> FakeDsv4FullPool:
    c4 = FakeDsv4PackedSubPool(
        kv_buffer=[_make_packed_buffer(num_pages=num_full_pages, page_size=page_size)],
        page_size=page_size,
    )
    c128 = FakeDsv4PackedSubPool(
        kv_buffer=[_make_packed_buffer(num_pages=num_full_pages, page_size=page_size)],
        page_size=page_size,
    )
    indexer = FakeDsv4IndexerSubPool(
        index_k_with_scale_buffer=[
            _make_indexer_buffer(
                num_pages=num_full_pages,
                page_size=page_size,
                index_head_dim=_INDEX_HEAD_DIM,
            )
        ],
        page_size=page_size,
    )
    swa = FakeDsv4PackedSubPool(
        kv_buffer=[_make_packed_buffer(num_pages=num_swa_pages, page_size=page_size)],
        page_size=page_size,
    )
    lut = torch.arange(num_full_pages * page_size + 1, dtype=torch.int64)
    return FakeDsv4FullPool(
        c4_kv_pool=c4,
        c4_indexer_kv_pool=indexer,
        c128_kv_pool=c128,
        swa_kv_pool=swa,
        page_size=page_size,
        full_to_swa_index_mapping=lut,
    )


def test_dsv4_packed_nope_rope_bytes_per_token_returns_576() -> None:
    pool = _make_pool().c4_kv_pool
    assert _dsv4_packed_nope_rope_bytes_per_token(pool) == 576


def test_dsv4_full_group_uses_576_bytes_per_token_for_c4_and_c128() -> None:
    # Step 1: build a mock DSv4 pool and call the adapter's full-group builder.
    pool = _make_pool()
    group = _build_full_group(pool=pool, device=CPU_DEVICE, read_bytes=32)

    # Step 2: full group emits one source per sub-pool (c4, indexer, c128).
    assert group.kind is PoolKind.FULL
    assert len(group.real_kv_sources_k) == 3
    c4_source, indexer_source, c128_source = group.real_kv_sources_k

    # Step 3: c4 and c128 use the nope+rope width (576), NOT the total 584.
    assert c4_source.num_bytes_per_token == _NOPE_ROPE_BYTES_PER_TOKEN
    assert c128_source.num_bytes_per_token == _NOPE_ROPE_BYTES_PER_TOKEN

    # Step 4: indexer uses index_head_dim (segment A of the indexer pool).
    assert indexer_source.num_bytes_per_token == _INDEX_HEAD_DIM


def test_dsv4_full_group_sources_preserve_page_size() -> None:
    pool = _make_pool(page_size=128)
    group = _build_full_group(pool=pool, device=CPU_DEVICE, read_bytes=32)
    for source in group.real_kv_sources_k:
        assert source.page_size == 128


def test_dsv4_swa_group_uses_packed_source_not_row_source() -> None:
    # Step 1: build SWA group from a mock DSv4 pool with page_size=128.
    pool = _make_pool(page_size=128)
    group = _build_swa_group(pool=pool, device=CPU_DEVICE, read_bytes=32)

    # Step 2: must be a packed source (page_size matches pool) — not row source which forces page_size=1.
    assert group.kind is PoolKind.SWA
    assert len(group.real_kv_sources_k) == 1
    swa_source = group.real_kv_sources_k[0]
    assert swa_source.page_size == pool.swa_kv_pool.page_size
    assert swa_source.page_size != 1

    # Step 3: uses the nope+rope width (576), not the total 584.
    assert swa_source.num_bytes_per_token == _NOPE_ROPE_BYTES_PER_TOKEN


def test_dsv4_swa_group_carries_index_lut() -> None:
    pool = _make_pool()
    group = _build_swa_group(pool=pool, device=CPU_DEVICE, read_bytes=32)
    assert group.swa_index_lut is pool.full_to_swa_index_mapping


def test_dsv4_segment_b_corruption_is_not_detected() -> None:
    # By-design false negative: scale segment (segment B at page offset page_size*576)
    # lies outside the RealKvSource slot-access window, so byte changes there must not
    # alter any per-slot fingerprint window. We assert this structurally by checking
    # that the configured per-slot byte window never overlaps segment B.

    # Step 1: build the FULL group using mock buffers.
    page_size = 128
    pool = _make_pool(page_size=page_size)
    group = _build_full_group(pool=pool, device=CPU_DEVICE, read_bytes=32)
    c4_source = group.real_kv_sources_k[0]

    # Step 2: compute the byte window the canary reads for every slot in a page.
    bytes_per_token = c4_source.num_bytes_per_token
    last_slot_in_page = page_size - 1
    last_read_end = (last_slot_in_page + 1) * bytes_per_token

    # Step 3: confirm segment A ends at page_size*576 and the canary read window ends at
    # or before it. Any byte at offset >= page_size*576 (segment B) is unreachable.
    segment_a_end = page_size * _NOPE_ROPE_BYTES_PER_TOKEN
    assert last_read_end == segment_a_end
    assert bytes_per_token == _NOPE_ROPE_BYTES_PER_TOKEN

    # Step 4: write a non-zero pattern into segment B of the underlying buffer and confirm
    # the slot-view bytes the canary reads are still the same (all zeros).
    page_buffer = c4_source.tensor
    page_idx = 0
    for slot_in_page in range(page_size):
        seg_b_offset = page_size * _NOPE_ROPE_BYTES_PER_TOKEN + slot_in_page * 8
        page_buffer[page_idx, seg_b_offset : seg_b_offset + 8] = 0xAB

    for slot_in_page in range(page_size):
        start = slot_in_page * bytes_per_token
        end = start + bytes_per_token
        slot_bytes = page_buffer[page_idx, start:end]
        assert torch.all(
            slot_bytes == 0
        ), f"slot {slot_in_page} sees non-zero bytes despite only segment B being written"
