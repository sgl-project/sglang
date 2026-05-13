from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import deep_gemm
import torch

from sglang.srt.environ import envs


@dataclass
class _PagedMqaLogitsMetadataChunk:
    start: int
    end: int
    schedule_meta: torch.Tensor


@dataclass
class _PagedMqaLogitsMetadata:
    chunks: List[_PagedMqaLogitsMetadataChunk]

    def copy_(self, other: "_PagedMqaLogitsMetadata"):
        raise Exception("Not expect to be copied")


def _sequence_chunk_ranges(
    seq_lens: torch.Tensor,
    batch_size: int,
    chunk_size: int,
) -> List[Tuple[int, int]]:
    lens = seq_lens.detach().flatten().to("cpu")
    boundary_indices = torch.nonzero(lens[1:] < lens[:-1], as_tuple=False).flatten()
    starts = [0, *boundary_indices.add(1).tolist(), batch_size]

    ranges: List[Tuple[int, int]] = []
    current_start = starts[0]
    current_end = starts[1]
    for seq_start, seq_end in zip(starts[1:-1], starts[2:]):
        if current_end - current_start + seq_end - seq_start <= chunk_size:
            current_end = seq_end
        else:
            ranges.append((current_start, current_end))
            current_start, current_end = seq_start, seq_end
    ranges.append((current_start, current_end))
    return ranges


def chunk_ranges_from_seq_lens(
    seq_lens: Optional[List[int]],
    chunk_size: int,
) -> Optional[List[Tuple[int, int]]]:
    if seq_lens is None:
        return None
    ranges: List[Tuple[int, int]] = []
    current_start = 0
    current_end = 0
    for seq_len in seq_lens:
        seq_start, seq_end = current_end, current_end + int(seq_len)
        if current_end > current_start and seq_end - current_start > chunk_size:
            ranges.append((current_start, current_end))
            current_start = seq_start
        current_end = seq_end
    if current_end > current_start:
        ranges.append((current_start, current_end))
    return ranges


def get_paged_mqa_logits_metadata_chunked(
    context_lens: torch.Tensor,
    block_kv: int,
    num_sms: int,
    chunk_ranges: Optional[List[Tuple[int, int]]] = None,
) -> Union[_PagedMqaLogitsMetadata, torch.Tensor]:
    chunk_size = envs.SGLANG_OPT_DG_PAGED_MQA_LOGITS_CHUNK_SIZE.get()
    if chunk_size == -1:
        chunk_size = envs.SGLANG_DSV4_PREFILL_METADATA_CHUNK_SIZE.get()
    batch_size = context_lens.shape[0]

    if chunk_size <= 0 or batch_size <= chunk_size:
        return deep_gemm.get_paged_mqa_logits_metadata(
            context_lens.unsqueeze(-1) if context_lens.dim() == 1 else context_lens,
            block_kv,
            num_sms,
        )

    chunk_ranges = chunk_ranges or _sequence_chunk_ranges(
        context_lens, batch_size, chunk_size
    )
    if chunk_ranges == [(0, batch_size)]:
        return deep_gemm.get_paged_mqa_logits_metadata(
            context_lens.unsqueeze(-1) if context_lens.dim() == 1 else context_lens,
            block_kv,
            num_sms,
        )

    chunks: List[_PagedMqaLogitsMetadataChunk] = []
    for start, end in chunk_ranges:
        schedule_meta = deep_gemm.get_paged_mqa_logits_metadata(
            (
                (context_lens[start:end]).unsqueeze(-1)
                if context_lens.dim() == 1
                else context_lens[start:end]
            ),
            block_kv,
            num_sms,
        )
        chunks.append(
            _PagedMqaLogitsMetadataChunk(
                start=start, end=end, schedule_meta=schedule_meta
            )
        )

    return _PagedMqaLogitsMetadata(chunks=chunks)


def fp8_paged_mqa_logits_chunked(
    q: torch.Tensor,
    kv_cache: torch.Tensor,
    weights: torch.Tensor,
    context_lens: torch.Tensor,
    block_table: torch.Tensor,
    schedule_meta: Union[_PagedMqaLogitsMetadata, torch.Tensor],
    max_context_len: int,
    clean_logits: bool,
) -> torch.Tensor:
    if not isinstance(schedule_meta, _PagedMqaLogitsMetadata):
        return deep_gemm.fp8_paged_mqa_logits(
            q,
            kv_cache,
            weights,
            context_lens,
            block_table,
            schedule_meta,
            max_context_len,
            clean_logits,
        )

    all_logits = []
    for chunk_meta in schedule_meta.chunks:
        chunk_logits = deep_gemm.fp8_paged_mqa_logits(
            q[chunk_meta.start : chunk_meta.end],
            kv_cache,
            weights[chunk_meta.start : chunk_meta.end],
            context_lens[chunk_meta.start : chunk_meta.end],
            block_table[chunk_meta.start : chunk_meta.end],
            chunk_meta.schedule_meta,
            max_context_len,
            clean_logits,
        )
        all_logits.append(chunk_logits)

    return torch.cat(all_logits, dim=0)
