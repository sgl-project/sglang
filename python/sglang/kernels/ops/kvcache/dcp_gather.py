"""Unpack rank-major DCP KV directly into attention's destination buffers."""

import torch
import triton
import triton.language as tl


@triton.jit
def _unpack_dcp_kv_kernel(
    gathered,
    metadata,
    out_k,
    out_pe,
    rank_rows: tl.constexpr,
    world_size: tl.constexpr,
    k_dim: tl.constexpr,
    pe_dim: tl.constexpr,
    k_stride: tl.constexpr,
    pe_stride: tl.constexpr,
    BLOCK: tl.constexpr,
    padded_start=0,
    start=0,
    length=0,
    output_start=0,
):
    if metadata is not None:
        req = tl.program_id(1)
        padded_start = tl.load(metadata + req * 4)
        start = tl.load(metadata + req * 4 + 1)
        length = tl.load(metadata + req * 4 + 2)
        output_start = tl.load(metadata + req * 4 + 3)
    dim = k_dim + pe_dim
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    token, feature = offsets // dim, offsets % dim
    position = start + token
    source_row = (
        position % world_size * rank_rows + padded_start + position // world_size
    )
    value = tl.load(
        gathered + source_row.to(tl.int64) * dim + feature,
        mask=token < length,
        other=0.0,
    )
    output_row = (output_start + token).to(tl.int64)
    tl.store(
        out_k + output_row * k_stride + feature,
        value,
        mask=(token < length) & (feature < k_dim),
    )
    tl.store(
        out_pe + output_row * pe_stride + feature - k_dim,
        value,
        mask=(token < length) & (feature >= k_dim),
    )


def unpack_dcp_kv(
    gathered: torch.Tensor,
    metadata: torch.Tensor | tuple[int, int, int, int],
    out_k: torch.Tensor,
    out_pe: torch.Tensor,
    world_size: int,
    max_prefix_len: int,
) -> None:
    """Write prefixes, leaving suffix slots untouched.

    Each metadata row is (padded_start, start % world_size, prefix_len,
    output_start); a single request can pass that tuple directly. Outputs may
    be separate tensors or strided latent/rope views of a combined MLA buffer.
    """
    if max_prefix_len == 0:
        return
    scalar_metadata = (0, 0, 0, 0)
    num_requests = 1 if isinstance(metadata, tuple) else metadata.shape[0]
    if isinstance(metadata, tuple):
        scalar_metadata, metadata = metadata, None
    elif metadata.device.type == "cpu" and num_requests == 1:
        # A single request needs no metadata upload or device allocation.
        scalar_metadata = metadata[0].tolist()
        metadata = None
    else:
        metadata = metadata.to(device=gathered.device, non_blocking=True)
    dim = out_k.shape[-1] + out_pe.shape[-1]
    _unpack_dcp_kv_kernel[(triton.cdiv(max_prefix_len * dim, 2048), num_requests)](
        gathered,
        metadata,
        out_k,
        out_pe,
        gathered.shape[0] // world_size,
        world_size,
        out_k.shape[-1],
        out_pe.shape[-1],
        out_k.stride(0),
        out_pe.stride(0),
        BLOCK=2048,
        padded_start=scalar_metadata[0],
        start=scalar_metadata[1],
        length=scalar_metadata[2],
        output_start=scalar_metadata[3],
    )
