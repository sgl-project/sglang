"""Pack the content channels in the compressed DSv4.1 cache and retain BF16 RoPE."""

import torch

from sglang.srt.hardware_backend.npu.dsv4.dsv41_int8 import (
    decode_int8,
    encode_int8,
    metadata_bytes,
    physical_page_map,
)

NOPE_DIM = 448
ROPE_DIM = 64
HEAD_DIM = NOPE_DIM + ROPE_DIM
GROUP_SIZE = 224
ROPE_BYTES = ROPE_DIM * torch.bfloat16.itemsize
SCALE_OFFSET = ROPE_BYTES + NOPE_DIM
_METADATA_END = SCALE_OFFSET + metadata_bytes(NOPE_DIM, GROUP_SIZE)
RECORD_BYTES = ((_METADATA_END + 31) // 32) * 32


def pack_kv(kv: torch.Tensor) -> torch.Tensor:
    if kv.dtype != torch.bfloat16 or kv.shape[-1] != HEAD_DIM:
        raise ValueError("DeepSeek-V4.1 INT8 KV requires BF16 rows of width 512")
    values, metadata = encode_int8(kv[..., :NOPE_DIM], GROUP_SIZE)
    record = kv.new_zeros((*kv.shape[:-1], RECORD_BYTES), dtype=torch.uint8)
    record[..., :ROPE_BYTES].view(torch.bfloat16).copy_(kv[..., NOPE_DIM:])
    record[..., ROPE_BYTES:SCALE_OFFSET].view(torch.int8).copy_(values)
    record[..., SCALE_OFFSET:_METADATA_END].copy_(metadata)
    return record


def unpack_kv(record: torch.Tensor) -> torch.Tensor:
    if record.dtype != torch.uint8 or record.shape[-1] != RECORD_BYTES:
        raise ValueError(
            f"Expected an unsigned byte buffer of {RECORD_BYTES}-byte KV records"
        )
    values = record[..., ROPE_BYTES:SCALE_OFFSET].view(torch.int8)
    nope = decode_int8(values, record[..., SCALE_OFFSET:_METADATA_END], GROUP_SIZE)
    return torch.cat((nope, record[..., :ROPE_BYTES].view(torch.bfloat16)), dim=-1)


def materialize_kv_pages(
    cache: torch.Tensor,
    block_table: torch.Tensor,
    lengths: torch.Tensor,
    *,
    fixed_shape=False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Decode only the batch's referenced pages for CANN 9.1 BF16 sparse MLA.

    Remapping physical page IDs preserves logical sparse indices. Page zero
    holds padding; unused page-table tails can contain old physical page IDs.
    """
    if cache.dtype != torch.uint8:
        return cache, block_table
    physical, remapped = physical_page_map(
        block_table, cache.shape[1], lengths, fixed_shape=fixed_shape
    )
    return unpack_kv(cache.index_select(0, physical)), remapped
