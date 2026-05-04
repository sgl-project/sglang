from dataclasses import dataclass
from typing import List, Union

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


def get_paged_mqa_logits_metadata_chunked(
    context_lens: torch.Tensor,
    block_kv: int,
    num_sms: int,
) -> Union[_PagedMqaLogitsMetadata, torch.Tensor]:
    chunk_size = envs.SGLANG_OPT_DG_PAGED_MQA_LOGITS_CHUNK_SIZE.get()
    batch_size = context_lens.shape[0]

    if batch_size <= chunk_size:
        return deep_gemm.get_paged_mqa_logits_metadata(context_lens, block_kv, num_sms)

    chunks: List[_PagedMqaLogitsMetadataChunk] = []
    for start in range(0, batch_size, chunk_size):
        end = min(start + chunk_size, batch_size)
        schedule_meta = deep_gemm.get_paged_mqa_logits_metadata(
            context_lens[start:end],
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
