from __future__ import annotations

import math

import pytest
import torch
from sgl_kernel.flash_mla import flash_mla_with_kvcache, get_mla_metadata
from sgl_kernel.flashmla_hisparse_demand import HiSparseDemandInputs

from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=90, stage="base-b-kernel-unit", runner_config="1-gpu-large")


def _sm90_with_cuda_124() -> bool:
    if not torch.cuda.is_available() or torch.version.cuda is None:
        return False
    cuda_version = tuple(int(part) for part in torch.version.cuda.split(".")[:2])
    return torch.cuda.get_device_capability() == (9, 0) and cuda_version >= (12, 4)


pytestmark = pytest.mark.skipif(
    not _sm90_with_cuda_124(), reason="requires SM90 and CUDA 12.4+"
)

TOPK = 2048
HOST_ROWS = 8192
CACHE_ROWS = 4096
VERIFY_ROWS = 4
READY = 2


def _pack_v32_rows(rows: torch.Tensor) -> torch.Tensor:
    """Pack BF16 [row, 1, 576] values into FlashMLA's 656-byte V32 rows."""
    row_count = rows.shape[0]
    assert row_count % 64 == 0
    packed = torch.empty(
        (row_count, 1, 656), dtype=torch.float8_e4m3fn, device=rows.device
    )
    nope = packed[..., :512]
    scales = packed[..., 512:528].view(torch.float32)
    packed[..., 528:].view(torch.bfloat16).copy_(rows[..., 512:])
    for tile_idx in range(4):
        tile = rows[..., tile_idx * 128 : (tile_idx + 1) * 128].float()
        scale = tile.abs().amax(dim=-1) / 448.0
        scale = torch.pow(2, scale.clamp_min(1e-4).log2().ceil())
        scales[..., tile_idx].copy_(scale)
        nope[..., tile_idx * 128 : (tile_idx + 1) * 128].copy_(
            (tile / scale.unsqueeze(-1)).to(torch.float8_e4m3fn)
        )
    return packed.view(row_count // 64, 64, 1, 656)


def _assert_bytes_equal(actual: torch.Tensor, expected: torch.Tensor) -> None:
    assert torch.equal(
        actual.contiguous().reshape(-1).view(torch.uint8),
        expected.contiguous().reshape(-1).view(torch.uint8),
    )


@pytest.mark.parametrize("batch_size", [1, 8])
@torch.inference_mode()
def test_mtp_direct_demand_matches_hbm_and_promotes_hits(batch_size: int):
    device = torch.device("cuda")
    generator = torch.Generator(device=device).manual_seed(20260821)
    committed_len = 131068

    logical_rows = torch.randn(
        (HOST_ROWS, 1, 576),
        dtype=torch.bfloat16,
        device=device,
        generator=generator,
    )
    packed_rows = _pack_v32_rows(logical_rows)
    packed_flat = packed_rows.view(HOST_ROWS, 1, 656)
    host_kv = torch.empty_like(packed_rows, device="cpu", pin_memory=True)
    host_kv.copy_(packed_rows)

    history_positions = list(range(TOPK - VERIFY_ROWS - 1)) + [committed_len - 1]
    history_host_rows = torch.randperm(
        HOST_ROWS - VERIFY_ROWS, device=device, generator=generator
    )[: TOPK - VERIFY_ROWS].to(torch.int32)
    overlay_positions = list(range(committed_len, committed_len + VERIFY_ROWS))
    logical_indices = torch.tensor(
        [history_positions + overlay_positions] * VERIFY_ROWS,
        dtype=torch.int32,
        device=device,
    ).unsqueeze(1)
    for verify_row in range(VERIFY_ROWS):
        logical_indices[verify_row, 0, TOPK - VERIFY_ROWS + verify_row + 1 :] = -1

    physical_indices = torch.full_like(logical_indices, -1)
    for source_ordinal, logical_position in enumerate(history_positions):
        physical_indices[logical_indices == logical_position] = history_host_rows[
            source_ordinal
        ]
    for offset, logical_position in enumerate(overlay_positions):
        physical_indices[logical_indices == logical_position] = (
            HOST_ROWS - VERIFY_ROWS + offset
        )
    logical_indices = logical_indices.repeat(batch_size, 1, 1)
    physical_indices = physical_indices.repeat(batch_size, 1, 1)
    query_rows = batch_size * VERIFY_ROWS

    rows_per_request = ((CACHE_ROWS + 6 + 63) // 64) * 64
    hot_device_kv = torch.zeros(
        (batch_size * rows_per_request // 64, 64, 1, 656),
        dtype=torch.float8_e4m3fn,
        device=device,
    )
    hot_flat = hot_device_kv.view(-1, 1, 656)
    device_locs = torch.zeros(
        (batch_size + 1, CACHE_ROWS + 6), dtype=torch.int64, device=device
    )
    local_rows = torch.arange(CACHE_ROWS + 6, dtype=torch.int64, device=device)
    for req_slot in range(1, batch_size + 1):
        device_locs[req_slot] = (req_slot - 1) * rows_per_request + local_rows
    hot_flat[device_locs[1:, CACHE_ROWS + 2 :]] = packed_flat[
        HOST_ROWS - VERIFY_ROWS : HOST_ROWS
    ].unsqueeze(0)

    host_locs = torch.full((query_rows, TOPK), -1, dtype=torch.int32, device=device)
    host_locs[:, : TOPK - VERIFY_ROWS] = history_host_rows.to(torch.int32)
    cache_tags = torch.zeros(
        (batch_size + 1, CACHE_ROWS), dtype=torch.int64, device=device
    )
    decode_calls = torch.tensor(
        [0] + [2] * batch_size, dtype=torch.int32, device=device
    )
    req_pool_indices = torch.arange(
        1, batch_size + 1, dtype=torch.int64, device=device
    ).repeat_interleave(VERIFY_ROWS)

    q = torch.randn(
        (query_rows, 1, 64, 576),
        dtype=torch.bfloat16,
        device=device,
        generator=generator,
    )
    kernel_seq_lens = torch.full((query_rows,), TOPK, dtype=torch.int32, device=device)
    tile_scheduler_metadata, num_splits = get_mla_metadata(
        kernel_seq_lens,
        num_q_tokens_per_head_k=64,
        num_heads_k=1,
        num_heads_q=64,
        is_fp8_kvcache=True,
        topk=TOPK,
    )
    common = dict(
        q=q,
        block_table=torch.empty((query_rows, 0), dtype=torch.int32, device=device),
        cache_seqlens=kernel_seq_lens,
        head_dim_v=512,
        tile_scheduler_metadata=tile_scheduler_metadata,
        num_splits=num_splits,
        softmax_scale=1.0 / math.sqrt(576),
        is_fp8_kvcache=True,
    )
    base_out, base_lse = flash_mla_with_kvcache(
        k_cache=packed_rows, indices=physical_indices, **common
    )

    host_kv.copy_(packed_rows)
    hot_flat.zero_()
    hot_flat[device_locs[1:, CACHE_ROWS + 2 :]] = packed_flat[
        HOST_ROWS - VERIFY_ROWS : HOST_ROWS
    ].unsqueeze(0)
    cache_tags.zero_()
    decode_calls[1:] = 2
    effective_seq_lens = torch.arange(
        committed_len + 1,
        committed_len + VERIFY_ROWS + 1,
        dtype=torch.int32,
        device=device,
    ).repeat(batch_size)
    demand_kwargs = dict(
        k_cache=hot_device_kv,
        indices=logical_indices,
        hisparse_demand=HiSparseDemandInputs(
            host_kv=host_kv,
            host_locs=host_locs,
            device_locs=device_locs,
            cache_tags=cache_tags,
            decode_calls=decode_calls,
            num_real_reqs=torch.tensor([query_rows], dtype=torch.int32, device=device),
            req_pool_indices=req_pool_indices,
            seq_lens=effective_seq_lens,
            mtp_committed_lens=torch.full(
                (query_rows,), committed_len, dtype=torch.int32, device=device
            ),
            cache_rows=CACHE_ROWS,
        ),
        **common,
    )
    demand_out, demand_lse = flash_mla_with_kvcache(**demand_kwargs)

    ready_mask = (cache_tags & 0x3) == READY
    ready_rows = (cache_tags[ready_mask] >> 26).to(torch.int64)
    history_host_row_mask = torch.zeros(HOST_ROWS, dtype=torch.bool, device=device)
    history_host_row_mask[history_host_rows] = True
    unexpected = ready_rows[
        (ready_rows < 0)
        | (ready_rows >= HOST_ROWS)
        | ~history_host_row_mask[ready_rows.clamp(0, HOST_ROWS - 1)]
    ]
    # Two independent candidates avoid most direct-map collisions while
    # keeping every lookup bounded. The deterministic fixture retains at
    # least 94% of the historical TopK rows after the cold fill.
    min_resident_rows = math.floor((TOPK - VERIFY_ROWS) * 0.94)
    assert ready_mask.sum().item() >= batch_size * min_resident_rows
    assert torch.all(ready_mask[1:].sum(dim=1) >= min_resident_rows)
    assert unexpected.numel() == 0
    assert not torch.any(ready_mask[0])
    resident_row_sets = []
    for req_slot in range(1, batch_size + 1):
        slots = torch.nonzero(ready_mask[req_slot], as_tuple=False).flatten()
        tagged_host_rows = (cache_tags[req_slot, slots] >> 26).to(torch.int64)
        resident_row_sets.append(set(tagged_host_rows.cpu().tolist()))
        cached_rows = hot_flat[device_locs[req_slot, slots]]
        expected_rows = packed_flat[tagged_host_rows]
        _assert_bytes_equal(cached_rows, expected_rows)

    _assert_bytes_equal(demand_out, base_out)
    _assert_bytes_equal(demand_lse, base_lse)
    assert not torch.any((cache_tags & 0x3) == 1)

    # Replaying the same generation must consume resident rows without Host
    # fallback or tag mutation. Duplicate fills can occupy both candidate
    # slots, so corrupt only rows confirmed resident for every request and
    # leave legitimate collision fallbacks intact.
    same_generation_tags = cache_tags.clone()
    common_resident_rows = sorted(set.intersection(*resident_row_sets))
    assert len(common_resident_rows) >= TOPK // 2
    host_kv.view(HOST_ROWS, 656).view(torch.uint8).index_fill_(
        0,
        torch.tensor(common_resident_rows, dtype=torch.int64),
        0,
    )
    hit_out, hit_lse = flash_mla_with_kvcache(**demand_kwargs)
    _assert_bytes_equal(hit_out, base_out)
    _assert_bytes_equal(hit_lse, base_lse)
    assert torch.equal(cache_tags[ready_mask], same_generation_tags[ready_mask])
    assert ((cache_tags & 0x3) == READY).sum() >= ready_mask.sum()
    assert not torch.any((cache_tags & 0x3) == 1)

    # Across decode calls, a set-associative cache may legitimately fall back to the
    # authoritative Host row after a collision. The output must remain exact.
    host_kv.copy_(packed_rows)
    decode_calls[1:] = 3
    retained_out, retained_lse = flash_mla_with_kvcache(**demand_kwargs)
    _assert_bytes_equal(retained_out, base_out)
    _assert_bytes_equal(retained_lse, base_lse)

    decode_calls[1:] = 4
    promoted_out, promoted_lse = flash_mla_with_kvcache(**demand_kwargs)
    _assert_bytes_equal(promoted_out, base_out)
    _assert_bytes_equal(promoted_lse, base_lse)

    # The 24-bit generation wraps lazily. Existing rows may be promoted or
    # replaced, but wrapped epochs must never expose a stale or partial row.
    decode_calls[1:] = (1 << 24) + 2
    wrapped_out, wrapped_lse = flash_mla_with_kvcache(**demand_kwargs)
    _assert_bytes_equal(wrapped_out, base_out)
    _assert_bytes_equal(wrapped_lse, base_lse)
    assert not torch.any((cache_tags & 0x3) == 1)
