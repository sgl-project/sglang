# Mixin class for metadata management of Deepseek MHA forward (chunked prefix cache)
# More details can be found in python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mha.py

from typing import List, Optional

import torch
import triton
import triton.language as tl

from sglang.srt.layers.attention.utils import create_flashinfer_kv_indices_triton


class ForwardBatchDeepSeekMHAMixin:
    # For MLA chunked prefix cache used in chunked prefill
    # Tell attention backend whether the kv cache needs to be attended in current pass
    attn_attend_prefix_cache: Optional[bool] = None
    # Number of prefix cache chunks
    num_prefix_chunks: Optional[int] = None
    # Index of current chunk, used by attention backend
    prefix_chunk_idx: Optional[int] = None
    # Maximum number of tokens in each chunk per sequence. Computed from maximum chunk capacity
    prefix_chunk_len: Optional[int] = None
    # Start positions of prefix cache for each chunk, (num_prefix_chunks, batch_size)
    prefix_chunk_starts: Optional[torch.Tensor] = None
    # Lengths of prefix cache for each chunk, (num_prefix_chunks, batch_size)
    prefix_chunk_seq_lens: Optional[torch.Tensor] = None
    # Accumulated lengths of prefix cache for each chunk, (num_prefix_chunks, batch_size + 1)
    prefix_chunk_cu_seq_lens: Optional[torch.Tensor] = None
    # Max lengths of prefix cache for each chunk, (num_prefix_chunks,)
    prefix_chunk_max_seq_lens: Optional[List[int]] = None
    # Number of tokens in each prefix cache chunk, (num_prefix_chunks,)
    prefix_chunk_num_tokens: Optional[List[int]] = None
    # KV Indices for each chunk
    prefix_chunk_kv_indices: Optional[List[torch.Tensor]] = None
    # For MLA chunked prefix cache used in chunked prefill
    # Tell attention backend whether lse needs to be returned
    mha_return_lse: Optional[bool] = None
    # Whether to apply MHA_ONE_SHOT forward method
    mha_one_shot: Optional[bool] = None
    # KV Indices for MHA_ONE_SHOT forward method
    mha_one_shot_kv_indices: Optional[torch.Tensor] = None

    def get_max_chunk_capacity(self):
        # Maximum number of tokens in each chunk
        # TODO: Should be changed to a better value, maybe passed through server args
        return 128 * 1024

    def set_prefix_chunk_idx(self, idx: int):
        self.prefix_chunk_idx = idx

    def set_attn_attend_prefix_cache(self, attn_attend_prefix_cache: bool):
        self.attn_attend_prefix_cache = attn_attend_prefix_cache

    def prepare_chunked_kv_indices(self, device: torch.device):
        self.prefix_chunk_kv_indices = []
        for idx in range(self.num_prefix_chunks):
            chunk_starts = self.prefix_chunk_starts[idx]
            chunk_seq_lens = self.prefix_chunk_seq_lens[idx]
            chunk_cu_seq_lens = self.prefix_chunk_cu_seq_lens[idx]
            num_chunk_tokens = self.prefix_chunk_num_tokens[idx]

            chunk_kv_indices = torch.empty(
                num_chunk_tokens, dtype=torch.int32, device=device
            )

            create_chunked_prefix_cache_kv_indices[(self.batch_size,)](
                self.req_to_token_pool.req_to_token,
                self.req_pool_indices,
                chunk_starts,
                chunk_seq_lens,
                chunk_cu_seq_lens,
                chunk_kv_indices,
                self.req_to_token_pool.req_to_token.shape[1],
            )
            self.prefix_chunk_kv_indices.append(chunk_kv_indices)

    # Here we suppose the length of each chunk is equal
    # For example, if we have 4 sequences with prefix length [256, 512, 768, 1024], prefix_chunk_len = 256
    # num_prefix_chunks = cdiv(1024, 256) = 4
    # prefix_chunk_starts = [[0, 0, 0, 0], [256, 256, 256, 256], [512, 512, 512, 512], [768, 768, 768, 768]]
    # prefix_chunk_ends = [[256, 256, 256, 256], [256, 512, 512, 512], [256, 512, 768, 768], [256, 512, 768, 1024]]
    # prefix_chunk_seq_lens = [[256, 256, 256, 256], [0, 256, 256, 256], [0, 0, 256, 256], [0, 0, 0, 256]]
    # TODO: Implement a better way to allocate chunk lengths that uses memory spaces more efficiently.
    def get_prefix_chunk_seq_lens(
        self, prefix_lens: torch.Tensor, num_prefix_chunks: int, prefix_chunk_len: int
    ):
        device = prefix_lens.device
        prefix_chunk_starts = (
            torch.arange(num_prefix_chunks, device=device, dtype=torch.int32)
            .unsqueeze(1)
            .expand(-1, self.batch_size)
            * prefix_chunk_len
        )
        prefix_chunk_ends = torch.min(
            prefix_lens.unsqueeze(0),
            prefix_chunk_starts + prefix_chunk_len,
        ).to(torch.int32)

        prefix_chunk_seq_lens = (
            (prefix_chunk_ends - prefix_chunk_starts).clamp(min=0).to(torch.int32)
        )

        return prefix_chunk_starts, prefix_chunk_seq_lens

    # Called before each attention module if using chunked kv cache for prefill
    # Some of the codes are adapted from https://github.com/vllm-project/vllm/blob/main/vllm/v1/attention/backends/mla/common.py
    def prepare_chunked_prefix_cache_info(self, device: torch.device):

        from sglang.srt.mem_cache.memory_pool import MLATokenToKVPool

        assert isinstance(
            self.token_to_kv_pool, MLATokenToKVPool
        ), "Currently chunked prefix cache can only be used by Deepseek models"

        if not any(self.extend_prefix_lens_cpu):
            self.num_prefix_chunks = 0
            return

        if self.prefix_chunk_len is not None:
            # Chunked kv cache info already prepared by prior modules
            return

        self.prefix_chunk_idx = -1

        # chunk_capacity is the maximum number of tokens in each chunk
        chunk_capacity = self.get_max_chunk_capacity()
        self.prefix_chunk_len = chunk_capacity // self.batch_size

        self.num_prefix_chunks = (
            max(self.extend_prefix_lens_cpu) + self.prefix_chunk_len - 1
        ) // self.prefix_chunk_len

        # Here we compute chunk lens twice to avoid stream sync, once on gpu and once on cpu.
        prefix_chunk_starts_cuda, prefix_chunk_seq_lens_cuda = (
            self.get_prefix_chunk_seq_lens(
                self.extend_prefix_lens,
                self.num_prefix_chunks,
                self.prefix_chunk_len,
            )
        )
        _, prefix_chunk_seq_lens_cpu = self.get_prefix_chunk_seq_lens(
            torch.tensor(self.extend_prefix_lens_cpu),
            self.num_prefix_chunks,
            self.prefix_chunk_len,
        )
        self.prefix_chunk_starts = prefix_chunk_starts_cuda
        self.prefix_chunk_seq_lens = prefix_chunk_seq_lens_cuda

        # Metadata for attention backend
        self.prefix_chunk_cu_seq_lens = torch.zeros(
            self.num_prefix_chunks,
            self.batch_size + 1,
            device=device,
            dtype=torch.int32,
        )
        self.prefix_chunk_cu_seq_lens[:, 1:] = prefix_chunk_seq_lens_cuda.cumsum(
            dim=1
        ).to(torch.int32)
        self.prefix_chunk_max_seq_lens = prefix_chunk_seq_lens_cpu.max(
            dim=1
        ).values.tolist()

        self.prefix_chunk_num_tokens = prefix_chunk_seq_lens_cpu.sum(dim=1).tolist()
        assert max(self.prefix_chunk_num_tokens) <= self.get_max_chunk_capacity()

        # Precompute the kv indices for each chunk
        self.prepare_chunked_kv_indices(device)

    def fetch_mha_one_shot_kv_indices(self):
        if self.mha_one_shot_kv_indices is not None:
            return self.mha_one_shot_kv_indices
        batch_size = self.batch_size
        paged_kernel_lens_sum = sum(self.seq_lens_cpu)
        kv_indices = torch.empty(
            paged_kernel_lens_sum,
            dtype=torch.int32,
            device=self.req_pool_indices.device,
        )
        kv_indptr = torch.zeros(
            batch_size + 1,
            dtype=torch.int32,
            device=self.req_pool_indices.device,
        )
        kv_indptr[1:] = torch.cumsum(self.seq_lens, dim=0)
        create_flashinfer_kv_indices_triton[(self.batch_size,)](
            self.req_to_token_pool.req_to_token,
            self.req_pool_indices,
            self.seq_lens,
            kv_indptr,
            None,
            kv_indices,
            self.req_to_token_pool.req_to_token.shape[1],
        )
        self.mha_one_shot_kv_indices = kv_indices
        return kv_indices


@triton.jit
def create_chunked_prefix_cache_kv_indices(
    req_to_token_ptr,  # (max_batch, max_context_len,)
    req_pool_indices_ptr,  # (batch_size,)
    chunk_start_idx_ptr,  # (batch_size,)
    chunk_seq_lens_ptr,  # (batch_size,)
    chunk_cu_seq_lens_ptr,  # (batch_size + 1,)
    chunk_kv_indices_ptr,  # (num_chunk_tokens,)
    req_to_token_ptr_stride: tl.constexpr,
):
    BLOCK_SIZE: tl.constexpr = 512
    pid = tl.program_id(axis=0)

    # find the req pool idx, this is for batch to token
    req_pool_index = tl.load(req_pool_indices_ptr + pid)
    chunk_kv_indices_offset = tl.load(chunk_cu_seq_lens_ptr + pid)

    # get the token positions of current chunk
    chunk_start_pos = tl.load(chunk_start_idx_ptr + pid).to(tl.int32)
    chunk_seq_len = tl.load(chunk_seq_lens_ptr + pid).to(tl.int32)

    num_loop = tl.cdiv(chunk_seq_len, BLOCK_SIZE)
    for i in range(num_loop):
        offset = tl.arange(0, BLOCK_SIZE) + i * BLOCK_SIZE
        mask = offset < chunk_seq_len
        data = tl.load(
            req_to_token_ptr
            + req_pool_index * req_to_token_ptr_stride
            + chunk_start_pos
            + offset,
            mask=mask,
        )
        tl.store(
            chunk_kv_indices_ptr + chunk_kv_indices_offset + offset, data, mask=mask
        )
