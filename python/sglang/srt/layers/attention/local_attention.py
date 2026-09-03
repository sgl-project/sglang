"""Chunked (local) attention for Llama4 iRoPE layers on the FA3/FA4 backend.

Each sequence is split into ``attention_chunk_size`` blocks that the kernel sees
as independent "virtual" batch items.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable, Dict, Optional

import numpy as np
import torch

if TYPE_CHECKING:
    from sglang.srt.layers.attention.flashattention_backend import (
        FlashAttentionMetadata,
    )
    from sglang.srt.layers.radix_attention import RadixAttention


@dataclass
class LocalAttentionMetadata:
    local_query_start_loc: torch.Tensor = None  # cu_seqlens_q for local attention
    local_seqused_k: torch.Tensor = None  # sequence lengths for local attention
    local_block_table: torch.Tensor = None  # block table for local attention
    local_max_query_len: int = 0  # max query length for local attention
    local_max_seq_len: int = 0  # max sequence length for local attention


class LocalAttentionMetadataBuilder:
    """The backend owns the CUDA-graph buffers; ``alloc_cuda_graph_buffers``
    returns them and the capture/replay methods take them back in."""

    def __init__(
        self,
        *,
        attention_chunk_size: int,
        page_size: int,
        max_context_len: int,
        device: torch.device,
        swa_translate: Optional[Callable[[torch.Tensor], torch.Tensor]],
    ):
        self.attention_chunk_size = attention_chunk_size
        self.page_size = page_size
        self.max_context_len = max_context_len
        self.device = device
        self.swa_translate = swa_translate

    def applies(self, layer: RadixAttention, metadata: FlashAttentionMetadata) -> bool:
        return metadata.local_attn_metadata is not None and layer.use_irope

    def alloc_cuda_graph_buffers(self, max_bs: int) -> Dict[str, torch.Tensor]:
        page_size = self.page_size or 1
        attn_chunk_size = self.attention_chunk_size
        max_virtual_batches = max_bs * (
            (self.max_context_len + attn_chunk_size - 1) // attn_chunk_size
        )
        max_pages_per_block = (attn_chunk_size + page_size - 1) // page_size
        return {
            "local_query_start_loc": torch.zeros(
                max_virtual_batches + 1, dtype=torch.int32, device=self.device
            ),
            "local_seqused_k": torch.zeros(
                max_virtual_batches, dtype=torch.int32, device=self.device
            ),
            "local_block_table": torch.zeros(
                max_virtual_batches,
                max_pages_per_block,
                dtype=torch.int32,
                device=self.device,
            ),
        }

    def _kernel_page_table(self, page_table: torch.Tensor) -> torch.Tensor:
        if self.swa_translate is not None:
            return self.swa_translate(page_table).to(torch.int32)
        return page_table

    def build(
        self,
        *,
        cu_seqlens_q: Optional[torch.Tensor],
        cache_seqlens_int32: Optional[torch.Tensor],
        page_table: Optional[torch.Tensor],
        device: torch.device,
    ) -> Optional[LocalAttentionMetadata]:
        if cu_seqlens_q is None or cache_seqlens_int32 is None or page_table is None:
            return None
        page_table = self._kernel_page_table(page_table)
        if self.page_size > 1:
            # Convert the eager token table to physical page indices.
            page_table = page_table[:, :: self.page_size] // self.page_size

        (
            seqlens_q_local_np,
            cu_seqlens_q_local_np,
            seqlens_k_local_np,
            block_table_local,
        ) = make_local_attention_virtual_batches(
            self.attention_chunk_size,
            cu_seqlens_q.cpu().numpy(),
            cache_seqlens_int32.cpu().numpy(),
            page_table,
            self.page_size,
            preserve_attn_chunk_size=True,
        )
        return LocalAttentionMetadata(
            local_query_start_loc=torch.from_numpy(cu_seqlens_q_local_np).to(device),
            local_seqused_k=torch.from_numpy(seqlens_k_local_np).to(device),
            local_block_table=block_table_local.to(device),
            local_max_query_len=int(seqlens_q_local_np.max()),
            local_max_seq_len=int(seqlens_k_local_np.max()),
        )

    def build_for_capture(
        self,
        metadata: FlashAttentionMetadata,
        bs: int,
        *,
        buffers: Dict[str, torch.Tensor],
    ) -> LocalAttentionMetadata:
        """Views of the preallocated buffers sized to what this capture needs."""
        seq_lens_capture = metadata.cache_seqlens_int32
        max_seq_len = int(seq_lens_capture.max().item())

        (
            seqlens_q_local_np,
            cu_seqlens_q_local_np,
            seqlens_k_local_np,
            block_table_local_np,
        ) = make_local_attention_virtual_batches(
            self.attention_chunk_size,
            metadata.cu_seqlens_q.cpu().numpy(),
            seq_lens_capture.cpu().numpy(),
            metadata.page_table,
            self.page_size,
            preserve_attn_chunk_size=True,
        )
        q_len = len(cu_seqlens_q_local_np)
        k_len = len(seqlens_k_local_np)
        b0 = block_table_local_np.shape[0] if block_table_local_np.shape[0] > 0 else bs
        b1 = block_table_local_np.shape[1] if block_table_local_np.shape[1] > 0 else 1
        return LocalAttentionMetadata(
            local_query_start_loc=buffers["local_query_start_loc"][:q_len],
            local_seqused_k=buffers["local_seqused_k"][:k_len],
            local_block_table=buffers["local_block_table"][:b0, :b1],
            local_max_query_len=1,
            local_max_seq_len=max_seq_len,
        )

    def update_for_replay(
        self,
        metadata: FlashAttentionMetadata,
        bs: int,
        *,
        buffers: Dict[str, torch.Tensor],
    ) -> None:
        """Refill the preallocated buffers in place before a CUDA-graph replay."""
        local_q_buf = buffers["local_query_start_loc"]
        local_k_buf = buffers["local_seqused_k"]
        local_block_buf = buffers["local_block_table"]

        # Decode: one query token per request.
        cu_seqlens_q = torch.arange(
            bs + 1, device=local_q_buf.device, dtype=local_q_buf.dtype
        )
        seqlens = metadata.cache_seqlens_int32[:bs]
        # Slice to bs and the real max seq len: rows past it hold zeros or stale
        # ids that would corrupt the replay.
        max_seq_len = int(seqlens.max().item())
        sliced_page_table = self._kernel_page_table(
            metadata.page_table[:bs, :max_seq_len]
        )

        (
            seqlens_q_local_np,
            cu_seqlens_q_local_np,
            seqlens_k_local_np,
            block_table_local,
        ) = make_local_attention_virtual_batches(
            self.attention_chunk_size,
            cu_seqlens_q.cpu().numpy(),
            seqlens.cpu().numpy(),
            sliced_page_table,
            self.page_size,
            preserve_attn_chunk_size=True,
        )

        device = local_q_buf.device
        cu_seqlens_q_local = torch.from_numpy(cu_seqlens_q_local_np).to(device)
        seqlens_k_local = torch.from_numpy(seqlens_k_local_np).to(device)
        block_table_local = block_table_local.to(device)
        q_len = cu_seqlens_q_local.shape[0]
        k_len = seqlens_k_local.shape[0]
        b0, b1 = block_table_local.shape

        local_q_buf[:q_len].copy_(cu_seqlens_q_local)
        local_q_buf[q_len:].fill_(0)
        local_k_buf[:k_len].copy_(seqlens_k_local)
        local_k_buf[k_len:].fill_(0)
        local_block_buf[:b0, :b1].copy_(block_table_local)
        local_block_buf[b0:, :].fill_(0)
        local_block_buf[:b0, b1:].fill_(0)

        if metadata.local_attn_metadata is not None:
            lam = metadata.local_attn_metadata
            lam.local_max_query_len = int(seqlens_q_local_np.max())
            lam.local_max_seq_len = int(seqlens_k_local_np.max())


def make_local_attention_virtual_batches(
    attn_chunk_size: int,
    query_start_loc_np: np.ndarray,
    seq_lens_np: np.ndarray,
    block_table: torch.Tensor,
    page_size: int = 0,
    preserve_attn_chunk_size: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, torch.Tensor]:
    """
    Take in `query_start_loc_np` and `seq_lens_np` and break the sequences into
    local attention blocks, where each block is passed to the attention kernel
    as an independent local ("virtual") batch item.

    Args:
        attn_chunk_size: Size of local attention chunks
        query_start_loc_np: Cumulative sum of query lengths (numpy array)
        seq_lens_np: Sequence lengths (numpy array)
        block_table: Block table for KV cache
        page_size: Size of each page in the KV cache
        preserve_attn_chunk_size: Skip sequence-length-based chunk normalization.

    Returns:
        seqlens_q_local: Query sequence lengths for local attention
        cu_seqlens_q_local: Cumulative sum of query sequence lengths for local attention
        seqlens_k_local: Key sequence lengths for local attention
        block_table_local: Block table for local attention
    """
    if not preserve_attn_chunk_size:
        max_seq_len = seq_lens_np.max()
        effective_chunk_size = min(attn_chunk_size, max_seq_len)
        effective_chunk_size = (effective_chunk_size // page_size) * page_size
        if effective_chunk_size < page_size:
            effective_chunk_size = page_size
        attn_chunk_size = effective_chunk_size

    q_seqlens = query_start_loc_np[1:] - query_start_loc_np[:-1]
    actual_batch_size = seq_lens_np.shape[0]

    # Handle if we are starting in the middle of a local attention block,
    #  we assume q_seqlens > 0 (for all elements), for each batch idx we compute
    #  the number of tokens that are not in the first local attention block and
    #  then we can simply use a cdiv for the rest.
    # For example if we have:
    #   attn_chunk_size = 4
    #   q_seqlens = [4, 10, 5]
    #   k_seqlens = [6, 17, 9]
    # Then we would get:
    #   new_tokens_in_first_block = [2, 1, 4]
    #   local_blocks = [2, 4, 2]
    q_tokens_in_first_block = np.minimum(
        attn_chunk_size - ((seq_lens_np - q_seqlens) % attn_chunk_size), q_seqlens
    ).astype(np.int32)
    tokens_in_last_block = attn_chunk_size + (seq_lens_np % -attn_chunk_size)
    local_blocks = 1 + cdiv(q_seqlens - q_tokens_in_first_block, attn_chunk_size)

    # Once we know the number of local blocks we can compute the request spans
    #  for each batch idx, we can figure out the number of "virtual" requests we
    #  have to make,
    # For the above example we would get:
    #   seqlens_q_local = [2, 2, 1, 4, 4, 1, 4, 1]
    #
    # First Get batched arange. (E.g., [2, 4, 2] -> [0, 1, 0, 1, 2, 3, 0, 1])
    #   (TODO: max a utility to share this code with _prepare_inputs)
    # arange step 1. [2, 4, 2] -> [2, 6, 8]
    cu_num_blocks = np.cumsum(local_blocks)
    virtual_batches = cu_num_blocks[-1]
    # arange step 2. [2, 6, 8] -> [0, 0, 2, 2, 2, 2, 6, 6]
    block_offsets = np.repeat(cu_num_blocks - local_blocks, local_blocks)
    # arange step 3. [0, 1, 0, 1, 2, 3, 0, 1]
    arange = np.arange(virtual_batches, dtype=np.int32) - block_offsets
    # also compute reverse arange (i.e. [1, 0, 3, 2, 1, 0, 1, 0])
    rarange = np.repeat(local_blocks, local_blocks) - arange - 1
    # Then we can compute the seqlens_q_local, handling the fact that the
    #  first and last blocks could be partial
    seqlens_q_local = np.repeat(q_seqlens - q_tokens_in_first_block, local_blocks)
    # set the first block since this may be a partial block
    seqlens_q_local[arange == 0] = q_tokens_in_first_block
    # set the remaining blocks
    seqlens_q_local[arange > 0] = np.minimum(
        seqlens_q_local - attn_chunk_size * (arange - 1), attn_chunk_size
    )[arange > 0]

    # convert from q_seqlens to cu_seqlens_q
    cu_seqlens_q_local = np.pad(np.cumsum(seqlens_q_local), (1, 0)).astype(np.int32)

    # compute the seqlens_k_local,
    #  basically a full local attention block for all but the last block in each
    #  batch
    # For our example this will be:
    #   seqlens_k_local = [4, 2, 4, 4, 4, 1, 4, 1]
    seqlens_k_local = np.full(cu_num_blocks[-1], attn_chunk_size, dtype=np.int32)
    seqlens_k_local[cu_num_blocks - 1] = tokens_in_last_block

    k_seqstarts_absolute = np.repeat(seq_lens_np, local_blocks) - (
        rarange * attn_chunk_size + np.repeat(tokens_in_last_block, local_blocks)
    )
    # For the example the local attention blocks start at:
    #                           _b0_  _____b1_____  _b2_
    #   k_seqstarts_absolute = [0, 4, 4, 8, 12, 16, 4, 8]
    block_starts = k_seqstarts_absolute // page_size

    assert attn_chunk_size % page_size == 0, (
        f"attn_chunk_size {attn_chunk_size} is not divisible by page_size {page_size}"
    )
    pages_per_local_batch = attn_chunk_size // page_size

    # Create a block_table for the local attention blocks
    # For out example if we have a block-table like (assuming page_size=2):
    #   block_table = [
    #     [ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9],  < batch 0
    #     [10, 11, 12, 13, 14, 15, 16, 17, 18, 19],  < batch 1
    #     [20, 21, 22, 23, 24, 25, 26, 27, 28, 29],  < batch 2
    #   ]
    # Then for the local batches we would want a block-table like
    #   block_table_local = [
    #     [  0,  1 ], < local-batch 0, (batch 0, starting from k[0])
    #     [  2,  3 ], < local-batch 1, (batch 0, starting from k[4])
    #     [ 12, 13 ], < local-batch 2, (batch 1, starting from k[4])
    #     [ 14, 15 ], < local-batch 3, (batch 1, starting from k[8])
    #     [ 16, 17 ], < local-batch 4, (batch 1, starting from k[12])
    #     [ 18, 19 ], < local-batch 5, (batch 1, starting from k[16])
    #     [ 22, 23 ], < local-batch 6, (batch 2, starting from k[4])
    #     [ 24, 25 ], < local-batch 7, (batch 2, starting from k[8])
    #   ]
    block_indices = np.broadcast_to(
        np.arange(pages_per_local_batch, dtype=np.int32),
        (virtual_batches, pages_per_local_batch),
    ) + np.expand_dims(block_starts, axis=1)
    # Ensure block_indices doesn't exceed block_table dimensions
    # This is a critical safety check that prevents index out of bounds errors
    # when dealing with large sequences (>8192 tokens) or when the block_table
    # dimensions are smaller than what would be needed for the full attention chunk size.
    block_indices = block_indices.flatten().clip(max=block_table.shape[1] - 1)
    batch_indices = np.repeat(
        np.arange(actual_batch_size, dtype=np.int32),
        local_blocks * pages_per_local_batch,
    )

    # NOTE: https://github.com/pytorch/pytorch/pull/160256 causes performance
    # regression when using numpy arrays (batch and block indices) to index into
    # torch tensor (block_table). As a workaround, convert numpy arrays to torch
    # tensor first, which recovers perf.
    batch_indices_torch = torch.from_numpy(batch_indices)
    block_indices_torch = torch.from_numpy(block_indices)
    block_table_local = block_table[batch_indices_torch, block_indices_torch].view(
        virtual_batches, -1
    )

    return seqlens_q_local, cu_seqlens_q_local, seqlens_k_local, block_table_local


def cdiv(a: int, b: int) -> int:
    """Ceiling division."""
    return -(a // -b)
