from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional, Union

import numpy as np
import torch

from sglang.srt.configs.model_config import AttentionArch
from sglang.srt.layers.attention.base_attn_backend import AttentionBackend
from sglang.srt.managers.schedule_batch import global_server_args_dict
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode
from sglang.srt.speculative.eagle_utils import EagleDraftInput, EagleVerifyInput

if TYPE_CHECKING:
    from sglang.srt.layers.radix_attention import RadixAttention
    from sglang.srt.model_executor.model_runner import ModelRunner

from sgl_kernel import merge_state_v2
from sgl_kernel.flash_attn import flash_attn_varlen_func, flash_attn_with_kvcache


@dataclass
class FlashAttentionMetadata:
    """Metadata to be init once in the model forward pass,
    each layer's forward pass can reuse the metadata.

    For each init metadata function, we will try set up them in below order
    """

    # Sequence lengths for the forward batch
    cache_seqlens_int32: torch.Tensor = None
    # Maximum sequence length for query
    max_seq_len_q: int = 1
    # Maximum sequence length for key
    max_seq_len_k: int = 0
    # Cumulative sequence lengths for query
    cu_seqlens_q: torch.Tensor = None
    # Cumulative sequence lengths for key
    cu_seqlens_k: torch.Tensor = None
    # Window size (typically used by Gemma)
    window_size: tuple = (-1, -1)
    # Page table, the index of KV Cache Tables/Blocks
    page_table: torch.Tensor = None

    # Encoder metadata
    # Cumulative sequence lengths for encoder key
    encoder_cu_seqlens_k: torch.Tensor = None
    # Maximum sequence length for encoder key
    encoder_max_seq_len_k: int = 0
    # Sequence lengths for the forward batch
    encoder_lens_int32: torch.Tensor = None
    # Page table for the encoder
    encoder_page_table: torch.Tensor = None

    @dataclass
    class LocalAttentionMetadata:
        local_query_start_loc: torch.Tensor = None  # cu_seqlens_q for local attention
        local_seqused_k: torch.Tensor = None  # sequence lengths for local attention
        local_block_table: torch.Tensor = None  # block table for local attention
        local_max_query_len: int = 0  # max query length for local attention
        local_max_seq_len: int = 0  # max sequence length for local attention

    local_attn_metadata: Optional[LocalAttentionMetadata] = None


# Copied from:
# https://github.com/houseroad/vllm/blob/4e45bfcaf928bdb9bd952b4ac922a3c205589ae8/vllm/v1/attention/backends/flash_attn.py
#
# Take in `query_start_loc_np` and `seq_lens_np` and break the sequences into
# local attention blocks, where each block is passed to the attention kernel
# as an independent local ("virtual") batch item.
#
# For example, if are performing a chunked prefill a batch of 3 sequences:
#   q_seqlens  = [4, 10, 5]
#   kv_seqlens = [6, 17, 9]
# Then normally for regular attention we would compute with an attention mask
#  for batch idx 0 (q_seqlens = 4, kv_seqlens = 6) like:
#   batch idx: 0 (q_seqlens = 4, kv_seqlens = 6)
#        k_toks >   0 1 2 3 4 5
#        q_toks v  _____________
#               0 | 1 1 1
#               1 | 1 1 1 1
#               2 | 1 1 1 1 1
#               3 | 1 1 1 1 1 1
#
# for local attention (with attn_chunk_size = 4) we would compute with an
#  attention mask like:
#   batch idx: 0  (q_seqlens = 4, kv_seqlens = 6, attn_chunk_size = 4)
#        k_toks >   0 1 2 3 4 5
#        q_toks v  _____________
#               0 | 1 1 1
#               1 | 1 1 1 1
#               2 |         1
#               3 |         1 1
#
# We can simulate this mask using standard flash-attention by breaking the
#  sequences into local ("virtual") batches, where each local batch item is a
#  local attention block, so in this case batch idx 0 would be broken up into:
#
#   local-batch idx: 0 (q_seqlens = 2, kv_seqlens = 4)  (batch 0)
#        k_toks >   0 1 2 3
#        q_toks v  _____________
#               0 | 1 1 1
#               1 | 1 1 1 1
#   local-batch idx: 1 (q_seqlens = 2, kv_seqlens = 2) (batch 0)
#        k_toks >   4 5
#        q_toks v  _____________
#               2 | 1
#               3 | 1 1
#
# e.g. if we have:
#   attn_chunk_size = 4
#   query_start_loc_np = [0, 4, 14, 19] (q_seqlens = [4, 10, 5])
# Then this function would return:
#                           __b0__  ______b1______  __b2__ < orig batch indices
#   q_seqlens_local    = [   2,  2,  1,  4,  4,  1,  4,  1]
#   cu_seqlens_q_local = [0, 4,  6, 10, 14, 18, 19, 23, 24]
#   seqlens_k_local    = [   4,  2,  4,  4,  4,  1,  4,  1]
#   block_table_local  : shape[local_virtual_batches, pages_per_local_batch]
def make_local_attention_virtual_batches(
    attn_chunk_size: int,
    query_start_loc_np: np.ndarray,
    seq_lens_np: np.ndarray,
    block_table: torch.Tensor,
    page_size: int = 0,
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

    Returns:
        seqlens_q_local: Query sequence lengths for local attention
        cu_seqlens_q_local: Cumulative sum of query sequence lengths for local attention
        seqlens_k_local: Key sequence lengths for local attention
        block_table_local: Block table for local attention
    """
    # Adjust attention_chunk_size based on the actual sequence length
    # to avoid index out of bounds errors
    max_seq_len = seq_lens_np.max()
    effective_chunk_size = min(attn_chunk_size, max_seq_len)
    # Make sure effective_chunk_size is divisible by page_size
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
        f"attn_chunk_size {attn_chunk_size} is not "
        f"divisible by page_size {page_size}"
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
    block_table_local = block_table[batch_indices, block_indices].view(
        virtual_batches, -1
    )

    return seqlens_q_local, cu_seqlens_q_local, seqlens_k_local, block_table_local


def cdiv(a: int, b: int) -> int:
    """Ceiling division."""
    return -(a // -b)


# TODO(hebiao064): remove this once we have a better way to handle the merge_state_v2 torch.compile issue
@torch._dynamo.disable()
def merge_state_v2_wrapper(o, s_a, o_exp, s_b):
    return merge_state_v2(o, s_a, o_exp, s_b)


class FlashAttentionBackend(AttentionBackend):
    """FlashAttention backend implementation.

    Note about the init:
    - If no spec decoding
        - FlashAttentionBackend will be init once when the server starts.
    - If spec decoding
        - FlashAttentionBackend will be init once for the target worker
        - FlashAttentionMultiStepBackend will be once for the draft worker
            - It will spawn num_steps FlashAttentionBackend for the draft worker

    Note about CUDA Graph:
    - We only support CUDA Graph for Decode (Normal Decode and Draft Decode) and Target Verify.
    - We don't support CUDA Graph for Extend and Draft Extend.
    - When server init, init_cuda_graph_state will be called first and then init_cuda_graph_capture will be called.
    - For each forward batch, init_replay_cuda_graph will be called first and then replay the graph.
    """

    def __init__(
        self,
        model_runner: ModelRunner,
        skip_prefill: bool = False,
        speculative_step_id=0,
        topk=0,
        speculative_num_steps=0,
    ):
        super().__init__()

        assert not (
            model_runner.sliding_window_size is not None
            and model_runner.model_config.is_encoder_decoder
        ), "Sliding window and cross attention are not supported together"

        self.forward_metadata: FlashAttentionMetadata = None
        # extra metdata for handling speculative decoding topk > 1, extended draft decode and verify
        self.forward_metadata_spec_decode_expand: FlashAttentionMetadata = None
        self.max_context_len = model_runner.model_config.context_len
        self.device = model_runner.device
        self.decode_cuda_graph_metadata = {}
        self.target_verify_metadata = {}
        self.req_to_token = model_runner.req_to_token_pool.req_to_token
        self.kv_cache_dtype = model_runner.kv_cache_dtype
        self.kv_cache_dtype_str = model_runner.server_args.kv_cache_dtype
        self.page_size = model_runner.page_size
        self.use_mla = model_runner.model_config.attention_arch == AttentionArch.MLA
        self.skip_prefill = skip_prefill
        self.topk = model_runner.server_args.speculative_eagle_topk or 0
        self.speculative_num_steps = speculative_num_steps
        self.speculative_num_draft_tokens = (
            model_runner.server_args.speculative_num_draft_tokens
        )
        self.speculative_step_id = speculative_step_id

        # Local attention settings
        self.attention_chunk_size = (
            model_runner.attention_chunk_size
            if hasattr(model_runner, "attention_chunk_size")
            else None
        )

    def init_forward_metadata(self, forward_batch: ForwardBatch):
        """Initialize forward metadata hence all layers in the forward pass can reuse it."""
        metadata = FlashAttentionMetadata()
        seqlens_in_batch = forward_batch.seq_lens
        batch_size = len(seqlens_in_batch)
        device = seqlens_in_batch.device

        if forward_batch.forward_mode.is_decode_or_idle():
            # Draft Decode
            if forward_batch.spec_info is not None:
                if self.topk <= 1:
                    metadata.cache_seqlens_int32 = (
                        seqlens_in_batch + (self.speculative_step_id + 1)
                    ).to(torch.int32)
                    metadata.max_seq_len_k = forward_batch.seq_lens_cpu.max().item() + (
                        self.speculative_step_id + 1
                    )
                    metadata.cu_seqlens_q = torch.arange(
                        0, batch_size + 1, dtype=torch.int32, device=device
                    )
                    metadata.cu_seqlens_k = torch.nn.functional.pad(
                        torch.cumsum(
                            metadata.cache_seqlens_int32, dim=0, dtype=torch.int32
                        ),
                        (1, 0),
                    )
                    metadata.page_table = forward_batch.req_to_token_pool.req_to_token[
                        forward_batch.req_pool_indices, : metadata.max_seq_len_k
                    ]
                else:
                    metadata.cache_seqlens_int32 = (seqlens_in_batch).to(torch.int32)
                    metadata.max_seq_len_q = self.topk
                    metadata.max_seq_len_k = forward_batch.seq_lens_cpu.max().item()
                    metadata.cu_seqlens_q = torch.arange(
                        0,
                        batch_size * self.topk + 1,
                        step=self.topk,
                        dtype=torch.int32,
                        device=device,
                    )
                    metadata.cu_seqlens_k = torch.nn.functional.pad(
                        torch.cumsum(
                            metadata.cache_seqlens_int32, dim=0, dtype=torch.int32
                        ),
                        (1, 0),
                    )
                    metadata.page_table = forward_batch.req_to_token_pool.req_to_token[
                        forward_batch.req_pool_indices, : metadata.max_seq_len_k
                    ]

                    metadata_expand = FlashAttentionMetadata()
                    decode_length = self.speculative_step_id + 1
                    metadata_expand.cache_seqlens_int32 = torch.full(
                        (seqlens_in_batch.numel() * self.topk,),
                        decode_length,
                        device=device,
                        dtype=torch.int32,
                    )
                    metadata_expand.max_seq_len_q = 1
                    metadata_expand.max_seq_len_k = self.speculative_step_id + 1
                    metadata_expand.cu_seqlens_q = torch.arange(
                        0,
                        metadata_expand.cache_seqlens_int32.numel() + 1,
                        dtype=torch.int32,
                        device=device,
                    )
                    metadata_expand.cu_seqlens_k = torch.arange(
                        0,
                        metadata_expand.cache_seqlens_int32.numel() * decode_length + 1,
                        step=decode_length,
                        dtype=torch.int32,
                        device=device,
                    )
                    cache_loc = forward_batch.out_cache_loc.view(
                        self.speculative_num_steps, -1
                    ).T.contiguous()
                    metadata_expand.page_table = (
                        cache_loc[:, :decode_length].contiguous().to(torch.int32)
                    )
                    self.forward_metadata_spec_decode_expand = metadata_expand
            else:
                # Normal Decode
                metadata.cache_seqlens_int32 = seqlens_in_batch.to(torch.int32)
                metadata.max_seq_len_k = forward_batch.seq_lens_cpu.max().item()
                metadata.cu_seqlens_q = torch.arange(
                    0, batch_size + 1, dtype=torch.int32, device=device
                )
                metadata.cu_seqlens_k = torch.nn.functional.pad(
                    torch.cumsum(seqlens_in_batch, dim=0, dtype=torch.int32), (1, 0)
                )
                metadata.page_table = forward_batch.req_to_token_pool.req_to_token[
                    forward_batch.req_pool_indices, : metadata.max_seq_len_k
                ]
            # TODO: we need to test this part for llama 4 eagle case
            self._init_local_attn_metadata(metadata, device)
        elif forward_batch.forward_mode.is_target_verify():
            if self.topk <= 1:
                metadata.cache_seqlens_int32 = (
                    forward_batch.seq_lens + self.speculative_num_draft_tokens
                ).to(torch.int32)
                metadata.max_seq_len_q = self.speculative_num_draft_tokens
                metadata.max_seq_len_k = (
                    forward_batch.seq_lens_cpu.max().item()
                    + self.speculative_num_draft_tokens
                )
                metadata.cu_seqlens_q = torch.arange(
                    0,
                    batch_size * self.speculative_num_draft_tokens + 1,
                    self.speculative_num_draft_tokens,
                    dtype=torch.int32,
                    device=device,
                )
                metadata.cu_seqlens_k = torch.nn.functional.pad(
                    torch.cumsum(
                        metadata.cache_seqlens_int32, dim=0, dtype=torch.int32
                    ),
                    (1, 0),
                )
                metadata.page_table = forward_batch.req_to_token_pool.req_to_token[
                    forward_batch.req_pool_indices, : metadata.max_seq_len_k
                ]

                self._init_local_attn_metadata(metadata, device)
            else:
                metadata.cache_seqlens_int32 = forward_batch.seq_lens.to(torch.int32)
                metadata.max_seq_len_q = self.speculative_num_draft_tokens
                metadata.max_seq_len_k = forward_batch.seq_lens_cpu.max().item()
                metadata.cu_seqlens_q = torch.arange(
                    0,
                    batch_size * self.speculative_num_draft_tokens + 1,
                    step=self.speculative_num_draft_tokens,
                    dtype=torch.int32,
                    device=device,
                )
                metadata.cu_seqlens_k = torch.nn.functional.pad(
                    torch.cumsum(
                        metadata.cache_seqlens_int32, dim=0, dtype=torch.int32
                    ),
                    (1, 0),
                )
                metadata.page_table = forward_batch.req_to_token_pool.req_to_token[
                    forward_batch.req_pool_indices, : metadata.max_seq_len_k
                ]

                metadata_expand = FlashAttentionMetadata()

                metadata_expand.max_seq_len_q = 1
                metadata_expand.cu_seqlens_q = torch.arange(
                    0,
                    forward_batch.seq_lens.numel() * self.speculative_num_draft_tokens
                    + 1,
                    dtype=torch.int32,
                    device=device,
                )

                # create expand page table
                offsets = torch.arange(
                    self.speculative_num_draft_tokens, device=device
                ).unsqueeze(
                    0
                )  # shape: (1, self.speculative_num_draft_tokens)
                cols = offsets.expand(
                    forward_batch.seq_lens.numel(), -1
                ) + forward_batch.seq_lens.unsqueeze(1)
                cum_len = torch.nn.functional.pad(
                    torch.cumsum(
                        (
                            forward_batch.seq_lens + self.speculative_num_draft_tokens
                        ).repeat_interleave(self.speculative_num_draft_tokens),
                        dim=0,
                    ),
                    (1, 0),
                )[:-1]
                mask_extraction_indices = (
                    cols.repeat_interleave(self.speculative_num_draft_tokens, dim=0)
                    + cum_len[:, None]
                ).view(1, -1)
                mask = forward_batch.spec_info.custom_mask[
                    mask_extraction_indices
                ].view(
                    -1, self.speculative_num_draft_tokens
                )  # (bsz * draft_num, draft_num)

                # shift table indices to avoid padding
                # non_masked_page_table [[8, 9, 10],   mask (display with int format) [[1, 0, 0],
                #                        [8, 9, 10],                                   [1, 1, 0],
                #                        [8, 9, 10]]                                   [1, 0, 1]]
                # if masked with padding [[8, 0, 0],   our mask without padding       [[8, 9, 10],
                #                        [8, 9, 0],                                    [8, 9, 10],
                #                        [8, 0, 10]]                                   [8, 10, 9]]
                # note here cache_seqlens_int32 is [1, 2, 2] so extra page indices will be ignored in each row
                col_indices = offsets.expand(
                    mask.shape[0], self.speculative_num_draft_tokens
                )
                # Build keys: if an entry is valid (mask==True), keep its original index;
                # if not, add self.speculative_num_draft_tokens so that it sorts after all valid entries.
                keys = torch.where(
                    mask, col_indices, col_indices + self.speculative_num_draft_tokens
                )
                _, sort_order = torch.sort(keys, dim=1)
                non_masked_page_table = (
                    forward_batch.req_to_token_pool.req_to_token[
                        forward_batch.req_pool_indices, :
                    ]
                    .gather(1, cols)
                    .repeat_interleave(self.speculative_num_draft_tokens, dim=0)
                )  # (bsz, draft_num)
                metadata_expand.page_table = non_masked_page_table.gather(1, sort_order)
                metadata_expand.cache_seqlens_int32 = mask.sum(dim=1).to(torch.int32)
                metadata_expand.cu_seqlens_k = torch.nn.functional.pad(
                    torch.cumsum(
                        metadata_expand.cache_seqlens_int32, dim=0, dtype=torch.int32
                    ),
                    (1, 0),
                )
                metadata_expand.max_seq_len_k = (
                    metadata_expand.cache_seqlens_int32.max().item()
                )
                self.forward_metadata_spec_decode_expand = metadata_expand
        elif forward_batch.forward_mode.is_extend_or_draft_extend_or_mixed():
            metadata.cache_seqlens_int32 = seqlens_in_batch.to(torch.int32)
            metadata.max_seq_len_k = forward_batch.seq_lens_cpu.max().item()
            metadata.cu_seqlens_k = torch.nn.functional.pad(
                torch.cumsum(seqlens_in_batch, dim=0, dtype=torch.int32), (1, 0)
            )
            metadata.page_table = forward_batch.req_to_token_pool.req_to_token[
                forward_batch.req_pool_indices, : metadata.max_seq_len_k
            ]

            if (
                any(forward_batch.extend_prefix_lens_cpu)
                or forward_batch.forward_mode == ForwardMode.DRAFT_EXTEND
            ):
                extend_seq_lens = forward_batch.extend_seq_lens
                metadata.max_seq_len_q = max(forward_batch.extend_seq_lens_cpu)
                metadata.cu_seqlens_q = torch.nn.functional.pad(
                    torch.cumsum(extend_seq_lens, dim=0, dtype=torch.int32), (1, 0)
                )
            else:
                metadata.max_seq_len_q = metadata.max_seq_len_k
                metadata.cu_seqlens_q = metadata.cu_seqlens_k

            # Setup local attention if enabled
            if forward_batch.forward_mode == ForwardMode.EXTEND:
                self._init_local_attn_metadata(metadata, device)

        # Encoder metadata for cross attention
        if forward_batch.encoder_lens is not None:
            assert (
                forward_batch.encoder_lens.numel() == 1
            ), "Only encoder size 1 is supported for now"

            metadata.encoder_lens_int32 = forward_batch.encoder_lens.to(torch.int32)
            metadata.encoder_cu_seqlens_k = torch.nn.functional.pad(
                torch.cumsum(metadata.encoder_lens_int32, dim=0, dtype=torch.int32),
                (1, 0),
            )
            metadata.encoder_max_seq_len_k = metadata.encoder_lens_int32.max().item()
            metadata.encoder_page_table = forward_batch.req_to_token_pool.req_to_token[
                forward_batch.req_pool_indices, : metadata.encoder_max_seq_len_k
            ]

            # Currently only support forward_batch.encoder_lens.numel() == 1
            metadata.page_table = forward_batch.req_to_token_pool.req_to_token[
                forward_batch.req_pool_indices,
                metadata.encoder_max_seq_len_k : (
                    metadata.encoder_max_seq_len_k + metadata.max_seq_len_k
                ),
            ]

        # Convert the page table to a strided format which is needed by FA3 API
        if self.page_size > 1:
            self.strided_indices = torch.arange(
                0, metadata.page_table.shape[1], self.page_size, device=self.device
            )
            metadata.page_table = (
                metadata.page_table[:, self.strided_indices] // self.page_size
            )

        self.forward_metadata = metadata

    def forward_extend(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        save_kv_cache=True,
        # For multi-head latent attention
        q_rope: Optional[torch.Tensor] = None,
        k_rope: Optional[torch.Tensor] = None,
    ):
        if k is not None:
            assert v is not None
            if save_kv_cache:
                cache_loc = (
                    forward_batch.out_cache_loc
                    if not layer.is_cross_attention
                    else forward_batch.encoder_out_cache_loc
                )
                if not self.use_mla:
                    forward_batch.token_to_kv_pool.set_kv_buffer(
                        layer, cache_loc, k, v, layer.k_scale, layer.v_scale
                    )
                else:
                    forward_batch.token_to_kv_pool.set_mla_kv_buffer(
                        layer,
                        cache_loc,
                        k,
                        k_rope,
                    )

        # Use precomputed metadata across all layers
        metadata = self.forward_metadata

        # Calculate window size (can be moved to metadata if layer properties don't change)
        # we don't do layer.sliding_window_size - 1 since in model.get_attention_sliding_window_size() we already - 1
        # here is two side inclusive
        window_size = (
            (layer.sliding_window_size, 0)
            if layer.sliding_window_size is not None and layer.sliding_window_size > -1
            else (-1, -1)
        )
        k_descale, v_descale = None, None
        # only use kv scaling if: 1) fp8 kv is explicitly enabled, 2) RadixAttention
        # has corresponding quantization method so that layer.k_scale is not None
        if self.kv_cache_dtype_str != "auto" and layer.k_scale is not None:
            descale_shape = (forward_batch.batch_size, layer.tp_k_head_num)
            k_descale = layer.k_scale.expand(descale_shape)
            v_descale = layer.v_scale.expand(descale_shape)
            q = q.to(self.kv_cache_dtype)
        causal = not layer.is_cross_attention

        # Check if we should use local attention
        use_local_attn = (
            self.attention_chunk_size is not None
            and metadata.local_attn_metadata is not None
            and (hasattr(layer, "use_irope") and layer.use_irope)
        )

        # We do cascade attention for Target Verify with topk > 1
        use_cascade_attn = (
            forward_batch.forward_mode.is_target_verify() and self.topk > 1
        )

        # Get the appropriate page table based on whether we're using local attention
        if use_local_attn:
            local_metadata = metadata.local_attn_metadata
            page_table = local_metadata.local_block_table
            cu_seqlens_q = local_metadata.local_query_start_loc
            cache_seqlens = local_metadata.local_seqused_k
            max_seqlen_q = local_metadata.local_max_query_len
            max_seqlen_k = local_metadata.local_max_seq_len
        else:
            page_table = metadata.page_table
            cu_seqlens_q = metadata.cu_seqlens_q
            cache_seqlens = metadata.cache_seqlens_int32
            max_seqlen_q = metadata.max_seq_len_q
            max_seqlen_k = metadata.max_seq_len_k
            cu_seqlens_k = metadata.cu_seqlens_k

        # Use Flash Attention for prefill
        if not self.use_mla:
            # Do multi-head attention
            key_cache, value_cache = forward_batch.token_to_kv_pool.get_kv_buffer(
                layer.layer_id
            )
            key_cache = key_cache.view(
                -1, self.page_size, layer.tp_k_head_num, layer.head_dim
            )
            value_cache = value_cache.view(
                -1, self.page_size, layer.tp_v_head_num, layer.head_dim
            )
            if layer.is_cross_attention:
                page_table = metadata.encoder_page_table
                cache_seqlens = metadata.encoder_lens_int32
                cu_seqlens_k = metadata.encoder_cu_seqlens_k
                window_size = (-1, -1)

            result = flash_attn_with_kvcache(
                q=q.contiguous().view(-1, layer.tp_q_head_num, layer.head_dim),
                k_cache=key_cache,
                v_cache=value_cache,
                page_table=page_table,
                cache_seqlens=cache_seqlens,
                cu_seqlens_q=cu_seqlens_q,
                cu_seqlens_k_new=cu_seqlens_k if not use_local_attn else None,
                max_seqlen_q=max_seqlen_q,
                softmax_scale=layer.scaling,
                causal=False if use_cascade_attn else causal,
                window_size=window_size,
                softcap=layer.logit_cap,
                k_descale=k_descale,
                v_descale=v_descale,
                return_softmax_lse=use_cascade_attn,
            )

            if use_cascade_attn:
                o, softmax_lse, *rest = result
                o_expand, softmax_lse_expand, *rest_expand = flash_attn_with_kvcache(
                    q=q.contiguous().view(-1, layer.tp_q_head_num, layer.head_dim),
                    k_cache=key_cache,
                    v_cache=value_cache,
                    page_table=self.forward_metadata_spec_decode_expand.page_table,
                    cache_seqlens=self.forward_metadata_spec_decode_expand.cache_seqlens_int32,
                    cu_seqlens_q=self.forward_metadata_spec_decode_expand.cu_seqlens_q,
                    cu_seqlens_k_new=self.forward_metadata_spec_decode_expand.cu_seqlens_k,
                    max_seqlen_q=self.forward_metadata_spec_decode_expand.max_seq_len_q,
                    softmax_scale=layer.scaling,
                    causal=False,
                    window_size=window_size,
                    softcap=layer.logit_cap,
                    k_descale=k_descale,
                    v_descale=v_descale,
                    return_softmax_lse=True,
                )
                o, _ = merge_state_v2_wrapper(
                    o,
                    softmax_lse.T.contiguous(),
                    o_expand,
                    softmax_lse_expand.T.contiguous(),
                )
            else:
                o = result
        else:
            if (
                not global_server_args_dict["disable_chunked_prefix_cache"]
                and forward_batch.attn_attend_prefix_cache is not None
                and not forward_batch.forward_mode.is_target_verify()
                and not forward_batch.forward_mode.is_draft_extend()
            ):
                # Do multi-head attention with chunked prefix cache

                if forward_batch.attn_attend_prefix_cache:
                    # MHA for chunked prefix kv cache when running model with MLA
                    assert forward_batch.prefix_chunk_idx is not None
                    assert forward_batch.prefix_chunk_cu_seq_lens is not None
                    assert forward_batch.prefix_chunk_max_seq_lens is not None

                    chunk_idx = forward_batch.prefix_chunk_idx
                    assert chunk_idx >= 0

                    output, lse, *rest = flash_attn_varlen_func(
                        q=q.view(-1, layer.tp_q_head_num, layer.head_dim),
                        k=k.view(-1, layer.tp_k_head_num, layer.head_dim),
                        v=v.view(-1, layer.tp_k_head_num, layer.v_head_dim),
                        cu_seqlens_q=metadata.cu_seqlens_q,
                        cu_seqlens_k=forward_batch.prefix_chunk_cu_seq_lens[chunk_idx],
                        max_seqlen_q=metadata.max_seq_len_q,
                        max_seqlen_k=forward_batch.prefix_chunk_max_seq_lens[chunk_idx],
                        softmax_scale=layer.scaling,
                        causal=False,
                        return_softmax_lse=True,
                    )
                else:
                    # MHA for extend part of sequence without attending prefix kv cache
                    output, lse, *rest = flash_attn_varlen_func(
                        q=q.view(-1, layer.tp_q_head_num, layer.head_dim),
                        k=k.view(-1, layer.tp_k_head_num, layer.head_dim),
                        v=v.view(-1, layer.tp_k_head_num, layer.v_head_dim),
                        cu_seqlens_q=metadata.cu_seqlens_q,
                        cu_seqlens_k=metadata.cu_seqlens_q,
                        max_seqlen_q=metadata.max_seq_len_q,
                        max_seqlen_k=metadata.max_seq_len_q,
                        softmax_scale=layer.scaling,
                        causal=True,
                        return_softmax_lse=True,
                    )
                return output, lse
            else:
                # Do absorbed multi-latent attention
                kv_cache = forward_batch.token_to_kv_pool.get_key_buffer(layer.layer_id)
                k_rope = kv_cache[:, :, layer.v_head_dim :]
                c_kv = kv_cache[:, :, : layer.v_head_dim]
                k_rope_cache = k_rope.view(
                    -1,
                    self.page_size,
                    layer.tp_k_head_num,
                    layer.head_dim - layer.v_head_dim,
                )
                c_kv_cache = c_kv.view(
                    -1, self.page_size, layer.tp_v_head_num, layer.v_head_dim
                )
                if q_rope is not None:
                    q_nope = q.view(-1, layer.tp_q_head_num, layer.v_head_dim)
                    q_rope = q_rope.view(
                        -1, layer.tp_q_head_num, layer.head_dim - layer.v_head_dim
                    )
                else:
                    q_all = q.contiguous().view(-1, layer.tp_q_head_num, layer.head_dim)
                    q_nope = q_all[:, :, : layer.v_head_dim]
                    q_rope = q_all[:, :, layer.v_head_dim :]

                result = flash_attn_with_kvcache(
                    q=q_rope,
                    k_cache=k_rope_cache,
                    v_cache=c_kv_cache,
                    qv=q_nope,
                    page_table=page_table,
                    cache_seqlens=cache_seqlens,
                    cu_seqlens_q=cu_seqlens_q,
                    cu_seqlens_k_new=cu_seqlens_k if not use_local_attn else None,
                    max_seqlen_q=max_seqlen_q,
                    softmax_scale=layer.scaling,
                    causal=False if use_cascade_attn else causal,
                    softcap=layer.logit_cap,
                    k_descale=k_descale,
                    v_descale=v_descale,
                    return_softmax_lse=use_cascade_attn,
                )
                if use_cascade_attn:
                    o, softmax_lse, *rest = result
                    o_expand, softmax_lse_expand, *rest_expand = (
                        flash_attn_with_kvcache(
                            q=q_rope,
                            k_cache=k_rope_cache,
                            v_cache=c_kv_cache,
                            qv=q_nope,
                            page_table=self.forward_metadata_spec_decode_expand.page_table,
                            cache_seqlens=self.forward_metadata_spec_decode_expand.cache_seqlens_int32,
                            cu_seqlens_q=self.forward_metadata_spec_decode_expand.cu_seqlens_q,
                            cu_seqlens_k_new=self.forward_metadata_spec_decode_expand.cu_seqlens_k,
                            max_seqlen_q=self.forward_metadata_spec_decode_expand.max_seq_len_q,
                            softmax_scale=layer.scaling,
                            causal=False,
                            window_size=window_size,
                            softcap=layer.logit_cap,
                            k_descale=k_descale,
                            v_descale=v_descale,
                            return_softmax_lse=True,
                        )
                    )
                    o, _ = merge_state_v2_wrapper(
                        o,
                        softmax_lse.T.contiguous(),
                        o_expand,
                        softmax_lse_expand.T.contiguous(),
                    )
                else:
                    o = result

        return o.view(-1, layer.tp_q_head_num * layer.v_head_dim)

    def forward_decode(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        save_kv_cache=True,
        # For multi-head latent attention
        q_rope: Optional[torch.Tensor] = None,
        k_rope: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if k is not None:
            assert v is not None
            if save_kv_cache:
                cache_loc = (
                    forward_batch.out_cache_loc
                    if not layer.is_cross_attention
                    else forward_batch.encoder_out_cache_loc
                )
                if not self.use_mla:
                    forward_batch.token_to_kv_pool.set_kv_buffer(
                        layer, cache_loc, k, v, layer.k_scale, layer.v_scale
                    )
                else:
                    forward_batch.token_to_kv_pool.set_mla_kv_buffer(
                        layer,
                        cache_loc,
                        k,
                        k_rope,
                    )

        # Use precomputed metadata across all layers
        metadata = self.forward_metadata
        local_attn_metadata = getattr(metadata, "local_attn_metadata", None)
        use_local_attention = (
            self.attention_chunk_size is not None and local_attn_metadata is not None
        )
        # We do cascade attention for Draft Decode with topk > 1
        use_cascade_attn = self.topk > 1

        # Calculate window size (can be moved to metadata if layer properties don't change)
        # we don't do layer.sliding_window_size - 1 since in model.get_attention_sliding_window_size() we already - 1
        # here is two side inclusive
        window_size = (
            (layer.sliding_window_size, 0)
            if layer.sliding_window_size is not None and layer.sliding_window_size > -1
            else (-1, -1)
        )
        causal = not layer.is_cross_attention

        k_descale, v_descale = None, None
        # only use kv scaling if: 1) fp8 kv is explicitly enabled, 2) RadixAttention
        # has corresponding quantization method so that layer.k_scale is not None
        if self.kv_cache_dtype_str != "auto":
            if layer.k_scale is not None:
                descale_shape = (forward_batch.batch_size, layer.tp_k_head_num)
                k_descale = layer.k_scale.expand(descale_shape)
                v_descale = layer.v_scale.expand(descale_shape)
            q = q.to(self.kv_cache_dtype)

        if not self.use_mla:
            # Do multi-head attention

            key_cache, value_cache = forward_batch.token_to_kv_pool.get_kv_buffer(
                layer.layer_id
            )
            key_cache = key_cache.view(
                -1, self.page_size, layer.tp_k_head_num, layer.head_dim
            )
            value_cache = value_cache.view(
                -1, self.page_size, layer.tp_v_head_num, layer.head_dim
            )

            if layer.is_cross_attention:
                # Always use non-chunked logic for cross-attention
                o = flash_attn_with_kvcache(
                    q=q.contiguous().view(-1, layer.tp_q_head_num, layer.head_dim),
                    k_cache=key_cache,
                    v_cache=value_cache,
                    page_table=metadata.encoder_page_table,
                    cache_seqlens=metadata.encoder_lens_int32,
                    cu_seqlens_q=metadata.cu_seqlens_q,
                    cu_seqlens_k_new=metadata.encoder_cu_seqlens_k,
                    max_seqlen_q=1,
                    softmax_scale=layer.scaling,
                    causal=False,
                    window_size=(-1, -1),
                    softcap=layer.logit_cap,
                    k_descale=k_descale,
                    v_descale=v_descale,
                )
            elif use_local_attention:
                # Use chunked (local) attention batching for self-attention
                o = flash_attn_with_kvcache(
                    q=q.contiguous().view(-1, layer.tp_q_head_num, layer.head_dim),
                    k_cache=key_cache,
                    v_cache=value_cache,
                    page_table=local_attn_metadata.local_block_table,
                    cache_seqlens=local_attn_metadata.local_seqused_k,
                    cu_seqlens_q=local_attn_metadata.local_query_start_loc,
                    cu_seqlens_k_new=metadata.cu_seqlens_k,
                    max_seqlen_q=local_attn_metadata.local_max_query_len,
                    softmax_scale=layer.scaling,
                    causal=True,
                    window_size=(-1, -1),
                    softcap=layer.logit_cap,
                    k_descale=k_descale,
                    v_descale=v_descale,
                )
            else:
                page_table = metadata.page_table
                cache_seqlens = metadata.cache_seqlens_int32
                cu_seqlens_k = metadata.cu_seqlens_k
                max_seqlen_q = metadata.max_seq_len_q
                q_reshaped = q.contiguous().view(
                    -1, layer.tp_q_head_num, layer.head_dim
                )

                # Default: single-token self-attention
                result = flash_attn_with_kvcache(
                    q=q_reshaped,
                    k_cache=key_cache,
                    v_cache=value_cache,
                    page_table=page_table,
                    cache_seqlens=cache_seqlens,
                    cu_seqlens_q=metadata.cu_seqlens_q,
                    cu_seqlens_k_new=cu_seqlens_k,
                    max_seqlen_q=max_seqlen_q,
                    softmax_scale=layer.scaling,
                    causal=False if use_cascade_attn else causal,
                    window_size=window_size,
                    softcap=layer.logit_cap,
                    k_descale=k_descale,
                    v_descale=v_descale,
                    return_softmax_lse=use_cascade_attn,
                )
                if use_cascade_attn:
                    o, softmax_lse, *rest = result
                    o_expand, softmax_lse_expand, *rest_expand = (
                        flash_attn_with_kvcache(
                            q=q_reshaped,
                            k_cache=key_cache,
                            v_cache=value_cache,
                            page_table=self.forward_metadata_spec_decode_expand.page_table,
                            cache_seqlens=self.forward_metadata_spec_decode_expand.cache_seqlens_int32,
                            cu_seqlens_q=self.forward_metadata_spec_decode_expand.cu_seqlens_q,
                            cu_seqlens_k_new=self.forward_metadata_spec_decode_expand.cu_seqlens_k,
                            max_seqlen_q=self.forward_metadata_spec_decode_expand.max_seq_len_q,
                            softmax_scale=layer.scaling,
                            causal=False,
                            window_size=window_size,
                            softcap=layer.logit_cap,
                            k_descale=k_descale,
                            v_descale=v_descale,
                            return_softmax_lse=True,
                        )
                    )
                    o, _ = merge_state_v2(
                        o,
                        softmax_lse.T.contiguous(),
                        o_expand,
                        softmax_lse_expand.T.contiguous(),
                    )
                else:
                    o = result
        else:
            # Do absorbed multi-latent attention
            kv_cache = forward_batch.token_to_kv_pool.get_key_buffer(layer.layer_id)
            k_rope = kv_cache[:, :, layer.v_head_dim :]
            c_kv = kv_cache[:, :, : layer.v_head_dim]
            k_rope_cache = k_rope.view(
                -1,
                self.page_size,
                layer.tp_k_head_num,
                layer.head_dim - layer.v_head_dim,
            )
            c_kv_cache = c_kv.view(
                -1, self.page_size, layer.tp_v_head_num, layer.v_head_dim
            )

            if q_rope is not None:
                q_nope = q.view(-1, layer.tp_q_head_num, layer.v_head_dim)
                q_rope = q_rope.view(
                    -1, layer.tp_q_head_num, layer.head_dim - layer.v_head_dim
                )
            else:
                q_all = q.contiguous().view(-1, layer.tp_q_head_num, layer.head_dim)
                q_nope = q_all[:, :, : layer.v_head_dim]
                q_rope = q_all[:, :, layer.v_head_dim :]
            max_seqlen_q = metadata.max_seq_len_q

            result = flash_attn_with_kvcache(
                q=q_rope,
                k_cache=k_rope_cache,
                v_cache=c_kv_cache,
                qv=q_nope,
                page_table=metadata.page_table,
                cache_seqlens=metadata.cache_seqlens_int32,
                cu_seqlens_q=metadata.cu_seqlens_q,
                cu_seqlens_k_new=metadata.cu_seqlens_k,
                max_seqlen_q=max_seqlen_q,
                softmax_scale=layer.scaling,
                causal=False if use_cascade_attn else causal,
                softcap=layer.logit_cap,
                k_descale=k_descale,
                v_descale=v_descale,
                return_softmax_lse=use_cascade_attn,  # softmax_lse is needed for merge states
            )
            if use_cascade_attn:
                o, softmax_lse, *rest = result
                o_expand, softmax_lse_expand, *rest_expand = flash_attn_with_kvcache(
                    q=q_rope,
                    k_cache=k_rope_cache,
                    v_cache=c_kv_cache,
                    qv=q_nope,
                    page_table=self.forward_metadata_spec_decode_expand.page_table,
                    cache_seqlens=self.forward_metadata_spec_decode_expand.cache_seqlens_int32,
                    cu_seqlens_q=self.forward_metadata_spec_decode_expand.cu_seqlens_q,
                    cu_seqlens_k_new=self.forward_metadata_spec_decode_expand.cu_seqlens_k,
                    max_seqlen_q=self.forward_metadata_spec_decode_expand.max_seq_len_q,
                    softmax_scale=layer.scaling,
                    causal=False,
                    window_size=window_size,
                    softcap=layer.logit_cap,
                    k_descale=k_descale,
                    v_descale=v_descale,
                    return_softmax_lse=True,
                )
                o, _ = merge_state_v2(
                    o,
                    softmax_lse.T.contiguous(),
                    o_expand,
                    softmax_lse_expand.T.contiguous(),
                )
            else:
                o = result

        return o.view(-1, layer.tp_q_head_num * layer.v_head_dim)

    def init_cuda_graph_state(self, max_bs: int):
        """Initialize CUDA graph state for the attention backend.

        Args:
            max_bs (int): Maximum batch size to support in CUDA graphs

        This creates fixed-size tensors that will be reused during CUDA graph replay
        to avoid memory allocations.
        """

        # This is being used by normal decode and draft decode when topk == 1
        self.decode_cuda_graph_metadata = {
            "cache_seqlens": torch.zeros(max_bs, dtype=torch.int32, device=self.device),
            "cu_seqlens_q": torch.arange(
                0, max_bs + 1, dtype=torch.int32, device=self.device
            ),
            "cu_seqlens_k": torch.zeros(
                max_bs + 1, dtype=torch.int32, device=self.device
            ),
            "page_table": torch.zeros(
                max_bs,
                (self.max_context_len + self.page_size - 1) // self.page_size,
                dtype=torch.int32,
                device=self.device,
            ),
            "page_table_draft_decode": torch.zeros(
                max_bs,
                (self.max_context_len + self.page_size - 1) // self.page_size,
                dtype=torch.int32,
                device=self.device,
            ),
            "strided_indices": torch.arange(
                0, self.max_context_len, self.page_size, device=self.device
            ),
        }

        # This is used by draft decode's first half of metadata when topk > 1
        if self.topk > 1:
            self.draft_decode_metadata_topk_normal = {
                "cache_seqlens": torch.zeros(
                    max_bs, dtype=torch.int32, device=self.device
                ),
                "cu_seqlens_q": torch.arange(
                    0,
                    max_bs * self.topk + 1,
                    step=self.topk,
                    dtype=torch.int32,
                    device=self.device,
                ),
                "cu_seqlens_k": torch.zeros(
                    max_bs + 1, dtype=torch.int32, device=self.device
                ),
                "page_table": torch.zeros(
                    max_bs,
                    self.max_context_len,
                    dtype=torch.int32,
                    device=self.device,
                ),
            }

            # This is used by draft decode's second half of metadata when topk > 1
            decode_length = self.speculative_step_id + 1
            self.draft_decode_metadata_topk_expand = {
                "cache_seqlens": torch.full(
                    (max_bs * self.topk,),
                    decode_length,
                    device=self.device,
                    dtype=torch.int32,
                ),
                "cu_seqlens_q": torch.arange(
                    0,
                    max_bs * self.topk + 1,
                    dtype=torch.int32,
                    device=self.device,
                ),
                "cu_seqlens_k": torch.arange(
                    0,
                    max_bs * self.topk * decode_length + 1,
                    step=decode_length,
                    dtype=torch.int32,
                    device=self.device,
                ),
                "page_table": torch.zeros(
                    max_bs * self.topk,
                    decode_length,
                    dtype=torch.int32,
                    device=self.device,
                ),
            }

        if (
            self.speculative_num_draft_tokens is not None
            and self.speculative_num_draft_tokens > 0
        ):
            self.target_verify_metadata = {
                "cache_seqlens": torch.zeros(
                    max_bs, dtype=torch.int32, device=self.device
                ),
                "cu_seqlens_q": torch.arange(
                    0,
                    max_bs * self.speculative_num_draft_tokens + 1,
                    step=self.speculative_num_draft_tokens,
                    dtype=torch.int32,
                    device=self.device,
                ),
                "cu_seqlens_k": torch.zeros(
                    max_bs + 1, dtype=torch.int32, device=self.device
                ),
                "page_table": torch.zeros(
                    max_bs,
                    (self.max_context_len + self.page_size - 1) // self.page_size,
                    dtype=torch.int32,
                    device=self.device,
                ),
                "strided_indices": torch.arange(
                    0, self.max_context_len, self.page_size, device=self.device
                ),
            }

        if self.topk > 1:
            self.target_verify_metadata_topk_normal = {
                "cache_seqlens": torch.zeros(
                    max_bs, dtype=torch.int32, device=self.device
                ),
                "cu_seqlens_q": torch.arange(
                    0,
                    max_bs * self.speculative_num_draft_tokens + 1,
                    step=self.speculative_num_draft_tokens,
                    dtype=torch.int32,
                    device=self.device,
                ),
                "cu_seqlens_k": torch.zeros(
                    max_bs + 1, dtype=torch.int32, device=self.device
                ),
                "page_table": torch.zeros(
                    max_bs,
                    self.max_context_len,
                    dtype=torch.int32,
                    device=self.device,
                ),
            }

            self.target_verify_metadata_topk_expand = {
                "cache_seqlens": torch.zeros(
                    max_bs * self.speculative_num_draft_tokens,
                    dtype=torch.int32,
                    device=self.device,
                ),
                "cu_seqlens_k": torch.zeros(
                    max_bs * self.speculative_num_draft_tokens + 1,
                    dtype=torch.int32,
                    device=self.device,
                ),
                "cu_seqlens_q": torch.arange(
                    0,
                    max_bs * self.speculative_num_draft_tokens + 1,
                    dtype=torch.int32,
                    device=self.device,
                ),
                "page_table": torch.zeros(
                    max_bs * self.speculative_num_draft_tokens,
                    self.speculative_num_draft_tokens,
                    dtype=torch.int32,
                    device=self.device,
                ),
            }

        self.encoder_metadata = {
            "encoder_page_table": torch.zeros(
                max_bs,
                self.max_context_len,
                dtype=torch.int32,
                device=self.device,
            ),
            "encoder_lens_int32": torch.zeros(
                max_bs, dtype=torch.int32, device=self.device
            ),
            "encoder_cu_seqlens_k": torch.zeros(
                max_bs + 1, dtype=torch.int32, device=self.device
            ),
        }

    def init_forward_metadata_capture_cuda_graph(
        self,
        bs: int,
        num_tokens: int,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        encoder_lens: Optional[torch.Tensor],
        forward_mode: ForwardMode,
        spec_info: Optional[Union[EagleDraftInput, EagleVerifyInput]],
    ):
        """Initialize forward metadata for capturing CUDA graph."""
        metadata = FlashAttentionMetadata()

        # metadata_expand is needed for Spec Decoding when top k > 1
        metadata_expand = FlashAttentionMetadata()

        device = seq_lens.device
        if forward_mode.is_decode_or_idle():
            if spec_info is not None:
                # Draft Decode
                if self.topk <= 1:
                    # When topk = 1, we use the normal decode metadata
                    metadata.cache_seqlens_int32 = self.decode_cuda_graph_metadata[
                        "cache_seqlens"
                    ][:bs]
                    metadata.max_seq_len_k = seq_lens.max().item() + (
                        self.speculative_step_id + 1
                    )
                    metadata.cu_seqlens_q = self.decode_cuda_graph_metadata[
                        "cu_seqlens_q"
                    ][: bs + 1]
                    metadata.cu_seqlens_k = torch.nn.functional.pad(
                        torch.cumsum(
                            metadata.cache_seqlens_int32, dim=0, dtype=torch.int32
                        ),
                        (1, 0),
                    )
                    metadata.page_table = self.decode_cuda_graph_metadata[
                        "page_table_draft_decode"
                    ][req_pool_indices, :]
                    self.decode_cuda_graph_metadata[bs] = metadata
                else:
                    # When top k > 1, we need two specific draft decode metadata, and then merge states
                    # 1. The first half of metadata for prefix tokens
                    metadata.cache_seqlens_int32 = (
                        self.draft_decode_metadata_topk_normal["cache_seqlens"][:bs]
                    )
                    metadata.max_seq_len_q = self.topk
                    metadata.max_seq_len_k = seq_lens.max().item()
                    metadata.cu_seqlens_q = self.draft_decode_metadata_topk_normal[
                        "cu_seqlens_q"
                    ][: bs + 1]
                    metadata.cu_seqlens_k = self.draft_decode_metadata_topk_normal[
                        "cu_seqlens_k"
                    ][: bs + 1]
                    metadata.page_table = self.draft_decode_metadata_topk_normal[
                        "page_table"
                    ][req_pool_indices, :]

                    # 2. The second half of metadata for draft tokens (per_batch_num_tokens = topk)
                    metadata_expand.cache_seqlens_int32 = (
                        self.draft_decode_metadata_topk_expand["cache_seqlens"][
                            : bs * self.topk
                        ]
                    )
                    metadata_expand.max_seq_len_q = 1
                    metadata_expand.max_seq_len_k = (
                        self.speculative_step_id + 1
                    )  # , do this in replay
                    metadata_expand.cu_seqlens_q = (
                        self.draft_decode_metadata_topk_expand["cu_seqlens_q"][
                            : bs * self.topk + 1
                        ]
                    )
                    metadata_expand.cu_seqlens_k = (
                        self.draft_decode_metadata_topk_expand["cu_seqlens_k"][
                            : bs * self.topk + 1
                        ]
                    )
                    metadata_expand.page_table = self.draft_decode_metadata_topk_expand[
                        "page_table"
                    ][: bs * self.topk]
                    self.draft_decode_metadata_topk_normal[bs] = metadata
                    self.draft_decode_metadata_topk_expand[bs] = metadata_expand
            else:
                # Normal Decode
                # Get sequence information
                metadata.cache_seqlens_int32 = seq_lens.to(torch.int32)
                batch_size = len(seq_lens)
                device = seq_lens.device
                metadata.cu_seqlens_k = torch.nn.functional.pad(
                    torch.cumsum(seq_lens, dim=0, dtype=torch.int32), (1, 0)
                )
                # Precompute maximum sequence length
                metadata.max_seq_len_k = seq_lens.max().item()
                # Precompute page table
                metadata.page_table = self.decode_cuda_graph_metadata["page_table"][
                    req_pool_indices, :
                ]
                # Precompute cumulative sequence lengths
                metadata.cu_seqlens_q = torch.arange(
                    0, batch_size + 1, dtype=torch.int32, device=device
                )
                self.decode_cuda_graph_metadata[bs] = metadata

        elif forward_mode.is_target_verify():
            if self.topk <= 1:
                metadata.cache_seqlens_int32 = self.target_verify_metadata[
                    "cache_seqlens"
                ][:bs]
                metadata.cache_seqlens_int32.copy_(
                    (seq_lens + self.speculative_num_draft_tokens).to(torch.int32)
                )

                metadata.max_seq_len_q = self.speculative_num_draft_tokens
                metadata.max_seq_len_k = (
                    seq_lens.max().item() + self.speculative_num_draft_tokens
                )

                metadata.cu_seqlens_q = torch.arange(
                    0,
                    bs * self.speculative_num_draft_tokens + 1,
                    self.speculative_num_draft_tokens,
                    dtype=torch.int32,
                    device=device,
                )

                metadata.cu_seqlens_k = self.target_verify_metadata["cu_seqlens_k"][
                    : (bs + 1)
                ]

                metadata.page_table = self.target_verify_metadata["page_table"][
                    req_pool_indices, :
                ]

                self.target_verify_metadata[bs] = metadata
            else:
                # When topk > 1, we need two specific target verify metadata, and then merge states
                # 1. The first half of metadata for prefix tokens
                metadata.cache_seqlens_int32 = self.target_verify_metadata_topk_normal[
                    "cache_seqlens"
                ][:bs]
                metadata.max_seq_len_q = self.speculative_num_draft_tokens
                # metadata.max_seq_len_k = forward_batch.seq_lens_cpu.max().item(), do this in replay
                metadata.cu_seqlens_q = self.target_verify_metadata_topk_normal[
                    "cu_seqlens_q"
                ][: bs + 1]
                metadata.cu_seqlens_k = self.target_verify_metadata_topk_normal[
                    "cu_seqlens_k"
                ][: bs + 1]
                metadata.page_table = self.target_verify_metadata_topk_normal[
                    "page_table"
                ][req_pool_indices, :]

                # 2. The second half of metadata for draft tokens (per_batch_num_tokens = topk)
                metadata_expand.cache_seqlens_int32 = (
                    self.target_verify_metadata_topk_expand["cache_seqlens"][
                        : bs * self.speculative_num_draft_tokens
                    ]
                )
                metadata_expand.max_seq_len_q = 1
                metadata_expand.cu_seqlens_q = self.target_verify_metadata_topk_expand[
                    "cu_seqlens_q"
                ][: bs * self.speculative_num_draft_tokens + 1]
                metadata_expand.cu_seqlens_k = self.target_verify_metadata_topk_expand[
                    "cu_seqlens_k"
                ][: bs * self.speculative_num_draft_tokens + 1]

                metadata_expand.page_table = self.target_verify_metadata_topk_expand[
                    "page_table"
                ][: bs * self.speculative_num_draft_tokens]

                self.target_verify_metadata_topk_normal[bs] = metadata
                self.target_verify_metadata_topk_expand[bs] = metadata_expand

        if encoder_lens is not None:
            encoder_bs = encoder_lens.numel()
            metadata.encoder_lens_int32 = self.encoder_metadata["encoder_lens_int32"][
                :encoder_bs
            ]
            metadata.encoder_cu_seqlens_k = self.encoder_metadata[
                "encoder_cu_seqlens_k"
            ][: (encoder_bs + 1)]

            metadata.encoder_page_table = self.encoder_metadata["encoder_page_table"][
                req_pool_indices, :
            ]

        self.forward_metadata = metadata
        self.forward_metadata_spec_decode_expand = metadata_expand

    def init_forward_metadata_replay_cuda_graph(
        self,
        bs: int,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        seq_lens_sum: int,
        encoder_lens: Optional[torch.Tensor],
        forward_mode: ForwardMode,
        spec_info: Optional[Union[EagleDraftInput, EagleVerifyInput]],
        seq_lens_cpu: Optional[torch.Tensor],
        out_cache_loc: torch.Tensor = None,
    ):
        """Initialize forward metadata for replaying CUDA graph."""
        seq_lens = seq_lens[:bs]
        seq_lens_cpu = seq_lens_cpu[:bs]
        req_pool_indices = req_pool_indices[:bs]
        device = seq_lens.device
        metadata = None
        metadata_expand = None

        if forward_mode.is_decode_or_idle():

            if spec_info is not None:
                # Draft Decode
                if self.topk <= 1:
                    metadata = self.decode_cuda_graph_metadata[bs]
                    # When topk = 1, we use the normal decode metadata
                    metadata.cache_seqlens_int32.copy_(
                        (seq_lens + (self.speculative_step_id + 1)).to(torch.int32)
                    )

                    metadata.max_seq_len_k = seq_lens_cpu.max().item() + (
                        self.speculative_step_id + 1
                    )
                    metadata.cu_seqlens_k.copy_(
                        torch.nn.functional.pad(
                            torch.cumsum(
                                metadata.cache_seqlens_int32, dim=0, dtype=torch.int32
                            ),
                            (1, 0),
                        )
                    )

                    max_seq_pages = (
                        metadata.max_seq_len_k + self.page_size - 1
                    ) // self.page_size
                    page_indices = self.req_to_token[
                        req_pool_indices[:, None],
                        self.decode_cuda_graph_metadata["strided_indices"][
                            :max_seq_pages
                        ],
                    ]

                    page_indices //= self.page_size
                    metadata.page_table[:, :max_seq_pages].copy_(page_indices)
                else:
                    # When top k > 1, we need two specific draft decode metadata, and then merge states
                    # 1. The first half of metadata for prefix tokens
                    metadata = self.draft_decode_metadata_topk_normal[bs]
                    metadata.cache_seqlens_int32.copy_(seq_lens.to(torch.int32))
                    # metadata.max_seq_len_q = self.topk, already set in capture
                    metadata.max_seq_len_k = seq_lens_cpu.max().item()
                    # metadata.cu_seqlens_q already set in capture
                    metadata.cu_seqlens_k.copy_(
                        torch.nn.functional.pad(
                            torch.cumsum(
                                metadata.cache_seqlens_int32, dim=0, dtype=torch.int32
                            ),
                            (1, 0),
                        )
                    )

                    page_table = self.req_to_token[
                        req_pool_indices, : metadata.max_seq_len_k
                    ]

                    metadata.page_table[:, : metadata.max_seq_len_k].copy_(page_table)

                    # 2. The second half of metadata for draft tokens (per_batch_num_tokens = topk)
                    metadata_expand = self.draft_decode_metadata_topk_expand[bs]
                    decode_length = self.speculative_step_id + 1
                    cache_loc = out_cache_loc.view(
                        self.speculative_num_steps, -1
                    ).T.contiguous()
                    metadata_expand.page_table[: cache_loc.shape[0]].copy_(
                        cache_loc[:, :decode_length].contiguous().to(torch.int32)
                    )
                # TODO: we need to test this part for llama 4 eagle case
                self._init_local_attn_metadata(metadata, device)
            else:
                metadata = self.decode_cuda_graph_metadata[bs]
                # Normal Decode
                max_len = seq_lens_cpu.max().item()
                metadata.max_seq_len_k = max_len

                metadata.cache_seqlens_int32 = seq_lens.to(torch.int32)
                metadata.cu_seqlens_k = torch.nn.functional.pad(
                    torch.cumsum(seq_lens, dim=0, dtype=torch.int32), (1, 0)
                )

                max_seq_pages = (
                    metadata.max_seq_len_k + self.page_size - 1
                ) // self.page_size
                page_indices = self.req_to_token[
                    req_pool_indices[:, None],
                    self.decode_cuda_graph_metadata["strided_indices"][:max_seq_pages][
                        None, :
                    ],
                ]
                page_indices //= self.page_size
                metadata.page_table[:, :max_seq_pages].copy_(page_indices)
                metadata.page_table[:, max_seq_pages:].fill_(0)

                self._init_local_attn_metadata(metadata, device)
        elif forward_mode.is_target_verify():
            if self.topk <= 1:
                metadata = self.target_verify_metadata[bs]
                metadata.cache_seqlens_int32.copy_(
                    (seq_lens + self.speculative_num_draft_tokens).to(torch.int32)
                )

                metadata.max_seq_len_k = (
                    seq_lens_cpu.max().item() + self.speculative_num_draft_tokens
                )
                metadata.cu_seqlens_k.copy_(
                    torch.nn.functional.pad(
                        torch.cumsum(
                            metadata.cache_seqlens_int32, dim=0, dtype=torch.int32
                        ),
                        (1, 0),
                    )
                )
                max_seq_pages = (
                    metadata.max_seq_len_k + self.page_size - 1
                ) // self.page_size
                page_indices = self.req_to_token[
                    req_pool_indices[:, None],
                    self.decode_cuda_graph_metadata["strided_indices"][:max_seq_pages],
                ]
                page_indices //= self.page_size
                metadata.page_table[:, :max_seq_pages].copy_(page_indices)
            else:
                # When topk > 1, we need two specific target verify metadata, and then merge states
                # 1. The first half of metadata for prefix tokens
                metadata = self.target_verify_metadata_topk_normal[bs]
                metadata.cache_seqlens_int32.copy_(seq_lens.to(torch.int32))
                # metadata.max_seq_len_q = self.speculative_num_draft_tokens, already set in capture
                metadata.max_seq_len_k = seq_lens_cpu.max().item()
                # metadata.cu_seqlens_q already set in capture
                metadata.cu_seqlens_k.copy_(
                    torch.nn.functional.pad(
                        torch.cumsum(
                            metadata.cache_seqlens_int32, dim=0, dtype=torch.int32
                        ),
                        (1, 0),
                    )
                )
                page_table = self.req_to_token[
                    req_pool_indices, : metadata.max_seq_len_k
                ]
                metadata.page_table[:, : metadata.max_seq_len_k].copy_(page_table)

                # 2. The second half of metadata for draft tokens (per_batch_num_tokens = topk)
                metadata_expand = self.target_verify_metadata_topk_expand[bs]
                # metadata_expand.max_seq_len_q = 1, already set in capture
                # metadata_expand.cu_seqlens_q already set in capture

                offsets = torch.arange(
                    self.speculative_num_draft_tokens, device=device
                ).unsqueeze(
                    0
                )  # shape: (1, self.speculative_num_draft_tokens)
                cols = offsets.expand(seq_lens.numel(), -1) + seq_lens.unsqueeze(1)
                cum_len = torch.nn.functional.pad(
                    torch.cumsum(
                        (
                            seq_lens + self.speculative_num_draft_tokens
                        ).repeat_interleave(self.speculative_num_draft_tokens),
                        dim=0,
                    ),
                    (1, 0),
                )[:-1]
                mask_extraction_indices = (
                    cols.repeat_interleave(self.speculative_num_draft_tokens, dim=0)
                    + cum_len[:, None]
                ).view(1, -1)
                # avoid extracting padded seq indices which will be out of boundary
                mask_extraction_indices[
                    :, spec_info.positions.numel() * self.speculative_num_draft_tokens :
                ].fill_(0)

                mask = spec_info.custom_mask[mask_extraction_indices].view(
                    -1, self.speculative_num_draft_tokens
                )  # (bsz * draft_num, draft_num)
                col_indices = offsets.expand(
                    mask.shape[0], self.speculative_num_draft_tokens
                )
                keys = torch.where(
                    mask, col_indices, col_indices + self.speculative_num_draft_tokens
                )
                _, sort_order = torch.sort(keys, dim=1)

                non_masked_page_table = (
                    self.req_to_token[req_pool_indices, :]
                    .gather(1, cols)
                    .repeat_interleave(self.speculative_num_draft_tokens, dim=0)
                )  # (bsz, draft_num)
                metadata_expand.page_table.copy_(
                    non_masked_page_table.gather(1, sort_order)
                )
                metadata_expand.cache_seqlens_int32.copy_(
                    mask.sum(dim=1).to(torch.int32)
                )
                metadata_expand.cu_seqlens_k.copy_(
                    torch.nn.functional.pad(
                        torch.cumsum(
                            metadata_expand.cache_seqlens_int32,
                            dim=0,
                            dtype=torch.int32,
                        ),
                        (1, 0),
                    )
                )
                metadata_expand.max_seq_len_k = (
                    metadata_expand.cache_seqlens_int32.max().item()
                )

        if encoder_lens is not None:
            # Only support encoder size 1 for now
            metadata.encoder_max_seq_len_k = encoder_lens[0]
            metadata.encoder_lens_int32.copy_(encoder_lens[:1])
            metadata.encoder_cu_seqlens_k.copy_(
                torch.nn.functional.pad(
                    torch.cumsum(metadata.encoder_lens_int32, dim=0, dtype=torch.int32),
                    (1, 0),
                )
            )

            metadata.encoder_page_table[:, : metadata.encoder_max_seq_len_k].copy_(
                self.req_to_token[req_pool_indices, : metadata.encoder_max_seq_len_k]
            )

            # Update the regular page table
            page_table = self.req_to_token[
                req_pool_indices,
                metadata.encoder_max_seq_len_k : (
                    metadata.encoder_max_seq_len_k + metadata.max_seq_len_k
                ),
            ]
            metadata.page_table[:, : metadata.max_seq_len_k].copy_(page_table)

        self.forward_metadata = metadata
        self.forward_metadata_spec_decode_expand = metadata_expand

    def get_cuda_graph_seq_len_fill_value(self):
        """Get the fill value for sequence length in CUDA graph."""
        return 0

    def _init_local_attn_metadata(self, metadata: FlashAttentionMetadata, device):
        """Centralized utility to initialize local_attn_metadata if chunked attention is enabled."""
        if self.attention_chunk_size is None:
            metadata.local_attn_metadata = None
            return

        cu_seqlens_q = metadata.cu_seqlens_q
        cache_seqlens_int32 = metadata.cache_seqlens_int32
        page_table = metadata.page_table
        if cu_seqlens_q is None or cache_seqlens_int32 is None or page_table is None:
            metadata.local_attn_metadata = None
            return

        cu_seqlens_q_np = cu_seqlens_q.cpu().numpy()
        seq_lens_np = cache_seqlens_int32.cpu().numpy()
        (
            seqlens_q_local_np,
            cu_seqlens_q_local_np,
            seqlens_k_local_np,
            block_table_local,
        ) = make_local_attention_virtual_batches(
            self.attention_chunk_size,
            cu_seqlens_q_np,
            seq_lens_np,
            page_table,
            self.page_size,
        )
        local_metadata = FlashAttentionMetadata.LocalAttentionMetadata(
            local_query_start_loc=torch.from_numpy(cu_seqlens_q_local_np).to(device),
            local_seqused_k=torch.from_numpy(seqlens_k_local_np).to(device),
            local_block_table=block_table_local.to(device),
            local_max_query_len=int(seqlens_q_local_np.max()),
            local_max_seq_len=int(seqlens_k_local_np.max()),
        )
        metadata.local_attn_metadata = local_metadata


class FlashAttentionMultiStepBackend:

    def __init__(
        self, model_runner: ModelRunner, topk: int, speculative_num_steps: int
    ):
        self.model_runner = model_runner
        self.topk = topk
        self.speculative_num_steps = speculative_num_steps
        self.attn_backends = []
        for i in range(self.speculative_num_steps):
            self.attn_backends.append(
                FlashAttentionBackend(
                    model_runner,
                    speculative_step_id=i,
                    topk=self.topk,
                    speculative_num_steps=self.speculative_num_steps,
                )
            )

    def init_forward_metadata(self, forward_batch: ForwardBatch):
        for i in range(self.speculative_num_steps - 1):
            self.attn_backends[i].init_forward_metadata(forward_batch)

    def init_cuda_graph_state(self, max_bs: int):
        for i in range(self.speculative_num_steps):
            self.attn_backends[i].init_cuda_graph_state(max_bs)

    def init_forward_metadata_capture_cuda_graph(
        self,
        forward_batch: ForwardBatch,
    ):
        assert forward_batch.spec_info is not None
        assert isinstance(forward_batch.spec_info, EagleDraftInput)

        for i in range(self.speculative_num_steps - 1):
            self.attn_backends[i].init_forward_metadata_capture_cuda_graph(
                forward_batch.batch_size,
                forward_batch.batch_size * self.topk,
                forward_batch.req_pool_indices,
                forward_batch.seq_lens,
                encoder_lens=forward_batch.encoder_lens,
                forward_mode=ForwardMode.DECODE,
                spec_info=forward_batch.spec_info,
            )

    def init_forward_metadata_replay_cuda_graph(
        self, forward_batch: ForwardBatch, bs: int
    ):
        assert forward_batch.spec_info is not None
        assert isinstance(forward_batch.spec_info, EagleDraftInput)

        for i in range(self.speculative_num_steps - 1):
            self.attn_backends[i].init_forward_metadata_replay_cuda_graph(
                bs,
                forward_batch.req_pool_indices,
                forward_batch.seq_lens,
                forward_batch.seq_lens_sum,
                encoder_lens=forward_batch.encoder_lens,
                forward_mode=ForwardMode.DECODE,
                spec_info=forward_batch.spec_info,
                seq_lens_cpu=forward_batch.seq_lens_cpu,
                out_cache_loc=forward_batch.out_cache_loc,
            )
