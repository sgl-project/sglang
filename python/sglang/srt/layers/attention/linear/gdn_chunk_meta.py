from dataclasses import dataclass

import torch

# Budget for cumsum block size selection on Ascend NPU, tuned against
# shared-memory / kernel constraints.
_GDN_CUMSUM_BLOCK_BUDGET = 2**18

# Large block size used by solve_tril Triton kernel
_GDN_LARGE_BLOCK_SIZE = 608 * 2


@dataclass(frozen=True)
class GDNChunkedPrefillMetadata:
    block_indices_cumsum: torch.Tensor
    chunk_indices_chunk64: torch.Tensor
    chunk_offsets_chunk64: torch.Tensor
    chunk_indices_large_block: torch.Tensor


def _next_power_of_2(x: int) -> int:
    return 1 if x <= 1 else 1 << (x - 1).bit_length()


def _chunk_counts(seq_lens: torch.Tensor, chunk_size: int) -> torch.Tensor:
    return torch.div(seq_lens + chunk_size - 1, chunk_size, rounding_mode="floor")


def _prepare_chunk_indices(
    seq_lens: torch.Tensor, chunk_size: int, device: torch.device
) -> torch.Tensor:
    chunk_counts = _chunk_counts(seq_lens, chunk_size)
    total_chunks = int(chunk_counts.sum().item())
    if total_chunks == 0:
        return torch.empty((0, 2), dtype=torch.int32, device=device)

    seq_indices = torch.repeat_interleave(
        torch.arange(chunk_counts.numel(), dtype=torch.int32), chunk_counts
    )
    # Vectorized: local_indices[i] = i - cumulative_offset_for_its_sequence.
    # Avoids O(batch) Python-level arange calls + cat.
    chunk_offsets_i64 = torch.zeros(chunk_counts.numel() + 1, dtype=torch.int64)
    chunk_offsets_i64[1:] = torch.cumsum(chunk_counts.to(torch.int64), dim=0)
    local_indices = (
        torch.arange(total_chunks, dtype=torch.int64)
        - chunk_offsets_i64[:-1].repeat_interleave(chunk_counts)
    ).to(torch.int32)

    result = torch.stack([seq_indices, local_indices], dim=1)
    # Use pinned memory for faster non-blocking H2D transfer.
    device_obj = torch.device(device) if isinstance(device, str) else device
    if device_obj.type != "cpu":
        result = result.pin_memory()
    return result.to(device=device, non_blocking=True)


def _prepare_chunk_offsets(
    seq_lens: torch.Tensor, chunk_size: int, device: torch.device
) -> torch.Tensor:
    chunk_counts = _chunk_counts(seq_lens, chunk_size)
    offsets = torch.empty(chunk_counts.numel() + 1, dtype=torch.int32, pin_memory=True)
    offsets[0] = 0
    offsets[1:] = torch.cumsum(chunk_counts, dim=0)
    return offsets.to(device=device, non_blocking=True)


def build_gdn_chunked_prefill_meta(
    *,
    cu_seqlens_cpu: torch.Tensor,
    num_heads: int,
    device: torch.device,
    chunk_size: int = 64,
    large_block_size: int = _GDN_LARGE_BLOCK_SIZE,
) -> GDNChunkedPrefillMetadata:
    cu_seqlens_cpu = cu_seqlens_cpu.to(device="cpu", dtype=torch.int32)
    seq_lens = cu_seqlens_cpu[1:].to(torch.int64) - cu_seqlens_cpu[:-1].to(torch.int64)

    cumsum_block_size = _next_power_of_2(
        _GDN_CUMSUM_BLOCK_BUDGET // (num_heads * chunk_size)
    )
    return GDNChunkedPrefillMetadata(
        block_indices_cumsum=_prepare_chunk_indices(
            seq_lens, cumsum_block_size, device
        ),
        chunk_indices_chunk64=_prepare_chunk_indices(seq_lens, chunk_size, device),
        chunk_offsets_chunk64=_prepare_chunk_offsets(seq_lens, chunk_size, device),
        chunk_indices_large_block=_prepare_chunk_indices(
            seq_lens, large_block_size, device
        ),
    )
