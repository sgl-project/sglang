from dataclasses import dataclass

import torch


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
    local_indices = torch.cat(
        [torch.arange(int(n), dtype=torch.int32) for n in chunk_counts.tolist()]
    )
    return torch.stack([seq_indices, local_indices], dim=1).to(device=device)


def _prepare_chunk_offsets(
    seq_lens: torch.Tensor, chunk_size: int, device: torch.device
) -> torch.Tensor:
    chunk_counts = _chunk_counts(seq_lens, chunk_size)
    offsets = torch.empty(chunk_counts.numel() + 1, dtype=torch.int32)
    offsets[0] = 0
    offsets[1:] = torch.cumsum(chunk_counts, dim=0)
    return offsets.to(device=device)


def build_gdn_chunked_prefill_meta(
    *,
    cu_seqlens_cpu: torch.Tensor,
    num_heads: int,
    device: torch.device,
    chunk_size: int = 64,
    large_block_size: int = 608 * 2,
) -> GDNChunkedPrefillMetadata:
    cu_seqlens_cpu = cu_seqlens_cpu.to(device="cpu", dtype=torch.int32)
    seq_lens = cu_seqlens_cpu[1:].to(torch.int64) - cu_seqlens_cpu[:-1].to(torch.int64)

    cumsum_block_size = _next_power_of_2((2**18) // (num_heads * chunk_size))
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
