from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import TYPE_CHECKING, Optional

import torch
import torch.nn.functional as F

# FlashInfer wrapper imports for CUDA graph support
from flashinfer import BatchDecodeWithPagedKVCacheWrapper

from sglang.kernels.ops.attention.metadata import normal_decode_set_metadata
from sglang.srt.configs.minicpm import MiniCPMHybridConfig
from sglang.srt.layers.attention.base_attn_backend import AttentionBackend
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode
from sglang.srt.runtime_context import get_parallel
from sglang.srt.server_args import get_global_server_args
from sglang.srt.speculative.spec_info import SpecInput
from sglang.srt.utils import is_flashinfer_available

if TYPE_CHECKING:
    from sglang.srt.layers.radix_attention import RadixAttention
    from sglang.srt.model_executor.model_runner import ModelRunner


import tilelang
import tilelang.math

from sglang.jit_kernel.minicpm_sala import (
    get_block_table_v2,
    get_block_table_v3,
)
from sglang.srt.layers.attention.minicpm.attention_kernels import (
    AttentionParams,
    create_attention_kernel,
)
from sglang.srt.layers.attention.minicpm.fuse_kernel import (
    _bucket_size,
    fused_attn_pooling_online_topk_decode,
    fused_attn_pooling_online_topk_prefill,
)
from sglang.srt.layers.attention.minicpm.sparse_utils import (
    CompressionLevelMetadata,
    SparseBatchAnalyzer,
    SparseConfig,
    SparseMetadataBuilder,
    allocate_and_compress_keys,
    compressed_attention,
    compressed_attention_tilelang,
    get_compress_k_v2,
    get_compress_k_v2_padded,
)


@dataclass
class MiniCPMBackendMetadata:
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
    # Page table for Sliding Window Attention
    swa_page_table: torch.Tensor = None
    total_q: int = -1  # use for max_pooling_1d_varlen

    # Flashinfer-specific metadata (pre-converted to avoid graph capture issues)
    flashinfer_kv_indptr: torch.Tensor = None
    flashinfer_kv_indices: torch.Tensor = None
    flashinfer_kv_last_page_len: torch.Tensor = None
    decode_wrapper: Optional[BatchDecodeWithPagedKVCacheWrapper] = None

    # Stage1 optimization metadata
    cu_seqlens_q_adjusted: Optional[torch.Tensor] = None
    max_seqlen_q_adjusted: Optional[int] = None
    cache_seqlens_int32_stage1: torch.Tensor = None

    # MiniCPM sparse attention metadata
    sparse_batch_size: int = 0
    sparse_idx: Optional[list[int]] = None


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
def cdiv(a: int, b: int) -> int:
    """Ceiling division."""
    return -(a // -b)


class MiniCPMSparseBackend(AttentionBackend):
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
        fa_impl_ver=3,
    ):

        self.forward_metadata: MiniCPMBackendMetadata = None
        self.max_context_len = model_runner.model_config.context_len
        self.device = model_runner.device
        self.enable_cuda_graph = not get_global_server_args().disable_cuda_graph
        self.decode_cuda_graph_metadata = {}
        self.req_to_token_pool = model_runner.req_to_token_pool
        self.token_to_kv_pool = model_runner.token_to_kv_pool
        self.req_to_token = model_runner.req_to_token_pool.req_to_token
        self.req_to_sparse_k1_token = (
            model_runner.req_to_token_pool.req_to_sparse_k1_token
        )
        self.req_to_sparse_k2_token = (
            model_runner.req_to_token_pool.req_to_sparse_k2_token
        )
        self.kv_cache_dtype = model_runner.kv_cache_dtype
        self.kv_cache_dtype_str = model_runner.server_args.kv_cache_dtype
        self.page_size = model_runner.page_size
        # MiniCPM does not support local attention
        self.use_mla = False
        self.attention_chunk_size = None
        self.is_encoder_decoder = False
        self.skip_prefill = skip_prefill
        tp_size = get_parallel().attn_tp_size
        self.num_kv_heads = model_runner.model_config.num_key_value_heads // tp_size

        self.fa_impl_ver = fa_impl_ver

        # For each layer, the sliding_window_size can be different. This is only used for preparing SWA metadata.
        # We use `layer.sliding_window_size` to decide whether to use SWA for each layer.
        self.sliding_window_size = model_runner.sliding_window_size
        self.has_swa = (
            self.sliding_window_size is not None and self.sliding_window_size > -1
        )

        # If num_splits == 0, we use a heuristic to automatically determine the number of splits.
        # We set nums splits to 1 if deterministic inference is enabled.
        # See https://thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference/ for more details.
        self.num_splits = (
            1 if model_runner.server_args.enable_deterministic_inference else 0
        )

        # Sparse attention configuration (required for MiniCPM)
        hf_config = model_runner.model_config.hf_config

        # MiniCPM must have sparse attention enabled
        if not isinstance(hf_config, MiniCPMHybridConfig) or not (
            hf_config.has_minicpm_sparse_attention
        ):
            raise ValueError(
                "MiniCPM model must have sparse attention enabled. "
                "Please ensure the model config has MiniCPM sparse attention enabled."
            )
        self.has_minicpm_sparse_attention = True

        self.kernel_size = hf_config.sparse_kernel_size
        self.kernel_stride = hf_config.sparse_kernel_stride
        self.init_blocks = hf_config.sparse_init_blocks
        self.block_size = hf_config.sparse_block_size
        self.window_size = hf_config.sparse_window_size
        self.minicpm_dense_as_sparse = model_runner.server_args.minicpm_dense_as_sparse
        self.dense_len = (
            0 if self.minicpm_dense_as_sparse else hf_config.sparse_dense_len
        )
        self.config_dense_len = hf_config.sparse_dense_len
        topk = hf_config.sparse_topk
        self.use_nope = hf_config.sparse_use_nope
        self.local_blocks = self.window_size // self.block_size  # local_blocks
        self.sparse_topk = topk + (self.window_size // self.block_size)
        self.num_sparse_topk_tokens = self.block_size * self.sparse_topk

        # Head group number derived from model configuration
        self.head_dim = model_runner.model_config.head_dim
        self.head_group_num = model_runner.model_config.num_key_value_heads
        self.heads_per_group = (
            model_runner.model_config.num_attention_heads // self.head_group_num
        )
        self.k1_kernel_size = self.kernel_size
        self.k1_kernel_stride = self.kernel_stride
        self.k2_kernel_size = self.kernel_size * 4
        self.k2_kernel_stride = self.kernel_stride * 4

        self.minicpm_fuse_topk = model_runner.server_args.minicpm_fuse_topk
        self.minicpm_split_stage1 = model_runner.server_args.minicpm_split_stage1

        max_cache_len = self.max_context_len
        pooled_k_len = (max_cache_len + self.block_size - 1) // self.block_size

        output_topk = min(self.sparse_topk, pooled_k_len)

        # For the kernel, we need power of 2 topk
        topk_power2 = tilelang.math.next_power_of_2(output_topk)
        kernel_topk = min(topk_power2, pooled_k_len)
        # Make sure it's still power of 2
        if kernel_topk != tilelang.math.next_power_of_2(kernel_topk):
            kernel_topk = tilelang.math.next_power_of_2(kernel_topk) // 2
        kernel_topk = max(8, kernel_topk)
        # FIXME: Read from model config
        dtype_str = "bfloat16"
        self.decode_fused_kernels = {}
        self.prefill_fused_kernels = {}
        bucketed_pooled_k_len = _bucket_size(pooled_k_len)

        pooling_block_stride = self.block_size // self.kernel_stride  # = 64 // 16 = 4
        pooling_pad_len = (
            self.kernel_size // self.kernel_stride - 1
        )  # = 32 // 16 - 1 = 1
        pooling_num_offs = (
            self.kernel_size // self.kernel_stride
            + self.block_size // self.kernel_stride
            - 1
        )

        if model_runner.server_args.minicpm_fuse_topk:
            bucketed_actual_max_seqlen_q = _bucket_size(
                model_runner.server_args.chunked_prefill_size
            )
            bucketed_actual_max_seqlen_k = _bucket_size(
                self.max_context_len // self.kernel_stride
            )
            for bs in range(1, model_runner.server_args.max_running_requests + 1):
                decode_kernel = fused_attn_pooling_online_topk_decode(
                    batch_size=bs,
                    groups=self.heads_per_group,
                    heads=model_runner.model_config.num_attention_heads,
                    dim=self.head_dim,
                    topk=kernel_topk,
                    pooled_k_len=bucketed_pooled_k_len,
                    m_block_dim=16,
                    block_stride=pooling_block_stride,
                    pad_len=pooling_pad_len,
                    num_offs=pooling_num_offs,
                    block_size=self.block_size,
                    init_blocks=self.init_blocks,
                    local_blocks=self.local_blocks,
                    dtype_str=dtype_str,
                )
                self.decode_fused_kernels[bs] = decode_kernel
                prefill_kernel = fused_attn_pooling_online_topk_prefill(
                    batch_size=bs,
                    groups=self.heads_per_group,
                    heads=model_runner.model_config.num_attention_heads,
                    dim=self.head_dim,
                    topk=kernel_topk,
                    max_seqlen_q_grid=model_runner.server_args.chunked_prefill_size,  # Bucketed for grid
                    pooled_k_len=bucketed_pooled_k_len,
                    actual_max_seqlen_q=bucketed_actual_max_seqlen_q,  # Bucketed for causal mask
                    actual_max_seqlen_k=bucketed_actual_max_seqlen_k,  # Bucketed for causal mask
                    m_block_dim=16,
                    block_stride=pooling_block_stride,
                    pad_len=pooling_pad_len,
                    num_offs=pooling_num_offs,
                    block_size=self.block_size,
                    init_blocks=self.init_blocks,
                    local_blocks=self.local_blocks,
                    dtype_str=dtype_str,
                )
                self.prefill_fused_kernels[bs] = prefill_kernel

        # Initialize attention kernel
        # Parse kernel type from attention_backend
        attention_backend = model_runner.server_args.attention_backend

        # Validate and map attention_backend to kernel type
        if attention_backend == "minicpm_flashattn":
            attention_kernel = "flash_attn"
        elif attention_backend == "minicpm_flashinfer":
            attention_kernel = "flashinfer"
        else:
            raise ValueError(
                f"Invalid attention_backend '{attention_backend}' for MiniCPM model. "
                f"Expected 'minicpm_flashattn' or 'minicpm_flashinfer'."
            )

        # Validate kernel type
        if attention_kernel == "flashinfer" and not is_flashinfer_available():
            raise ValueError(
                "FlashInfer kernel requested but flashinfer is not available. "
                "Please install flashinfer or use attention_backend='minicpm' or 'minicpm_flashattn'"
            )
        self.attention_kernel = create_attention_kernel(attention_kernel, model_runner)
        self.attention_kernel_type = attention_kernel

        # Initialize sparse attention helpers (required for MiniCPM)
        sparse_config = SparseConfig.from_model_config(
            hf_config, model_runner.model_config
        )
        self.sparse_batch_analyzer = SparseBatchAnalyzer(sparse_config)
        self.sparse_metadata_builder = SparseMetadataBuilder(
            sparse_config,
            num_kv_heads=self.num_kv_heads,
            max_context_len=self.max_context_len,
        )

    def update_batch_for_sparse(
        self, forward_batch: ForwardBatch, metadata: MiniCPMBackendMetadata
    ):
        cu_seqlens_q = metadata.cu_seqlens_q

        compression_metadata = (
            self.sparse_metadata_builder.build_k1_k2_compression_metadata(
                forward_batch=forward_batch,
                base_metadata=metadata,
                req_to_sparse_k1_token=self.req_to_sparse_k1_token,
                req_to_sparse_k2_token=self.req_to_sparse_k2_token,
                k1_kernel_size=self.k1_kernel_size,
                k1_kernel_stride=self.k1_kernel_stride,
                k2_kernel_size=self.k2_kernel_size,
                k2_kernel_stride=self.k2_kernel_stride,
                cu_seqlens_q=cu_seqlens_q,
            )
        )

        # Map k1/k2 compression metadata objects
        metadata.k1 = compression_metadata["k1"]
        metadata.k2 = compression_metadata["k2"]

        if forward_batch.forward_mode.is_extend_or_draft_extend_or_mixed():
            metadata.sparse_bs_list = (
                self.sparse_batch_analyzer.identify_sparse_batches(
                    forward_batch, self.minicpm_dense_as_sparse
                )
            )

            seqlen_q_sparse_bs, metadata.seqlen_k_sparse_bs_tensor = (
                self.sparse_metadata_builder.build_sequence_lengths(
                    cu_seqlens_q,
                    forward_batch.extend_prefix_lens,
                    metadata.sparse_bs_list,
                )
            )

            cu_seqlens_q_sparse_bs = torch.tensor(
                [0] + seqlen_q_sparse_bs, dtype=torch.int32, device=cu_seqlens_q.device
            ).cumsum(dtype=torch.int32, dim=0)

            extend_prefix_lens_sparse = torch.tensor(
                [
                    forward_batch.extend_prefix_lens_cpu[bs]
                    for bs in metadata.sparse_bs_list
                ],
                dtype=torch.long,
                device="cpu",
            )

            metadata.token_to_bs, metadata.token_pos_in_bs = (
                self.sparse_metadata_builder.build_token_mappings(
                    cu_seqlens_q_sparse_bs,
                    extend_prefix_lens_sparse,
                    seqlen_q_sparse_bs,
                )
            )
            metadata.token_to_bs = metadata.token_to_bs.to(
                device=metadata.cu_seqlens_q.device
            )
            metadata.token_pos_in_bs = metadata.token_pos_in_bs.to(
                device=metadata.cu_seqlens_q.device
            )

            prefill_metadata = (
                self.sparse_metadata_builder.build_sparse_prefill_metadata(
                    forward_batch=forward_batch,
                    base_metadata=metadata,
                    sparse_bs_list=metadata.sparse_bs_list,
                    head_group_num=self.head_group_num,
                    dense_len=self.dense_len,
                    sparse_topk=self.sparse_topk,
                    block_size=self.block_size,
                    cu_seqlens_q=cu_seqlens_q,
                    sparse_page_table_dtype=metadata.page_table.dtype,
                    sparse_page_table_device=metadata.page_table.device,
                )
            )

            metadata.sparse_page_table = prefill_metadata["sparse_page_table"]
            metadata.sparse_cu_seqlens_q_cpu = prefill_metadata[
                "sparse_cu_seqlens_q_cpu"
            ]
            metadata.sparse_cu_seqlens_q = prefill_metadata["sparse_cu_seqlens_q"]
            metadata.old_bs_to_new_bs_range = prefill_metadata["old_bs_to_new_bs_range"]
            metadata.sparse_max_seq_len_q = prefill_metadata["sparse_max_seq_len_q"]

            metadata.sparse_batch_size = len(metadata.sparse_bs_list)
            metadata.sparse_idx = prefill_metadata["sparse_idx"]

            # Stage1 optimization metadata for prefill mode
            metadata.cache_seqlens_int32_stage1 = metadata.cache_seqlens_int32 - 1
            seqlens_q_sparse_list = []
            for i in range(forward_batch.batch_size):
                if forward_batch.seq_lens_cpu[i] >= self.dense_len:
                    seqlens_q_sparse_list.append(forward_batch.extend_seq_lens_cpu[i])

            if len(seqlens_q_sparse_list) > 0:
                seqlen_q_sparse_tensor = torch.tensor(
                    seqlens_q_sparse_list,
                    dtype=torch.int32,
                    device=metadata.cu_seqlens_q.device,
                )
                cu_seqlen_q_sparse_tensor = F.pad(
                    torch.cumsum(seqlen_q_sparse_tensor, dim=0, dtype=torch.int32),
                    (1, 0),
                )
                metadata.cu_seqlens_q_adjusted = (
                    cu_seqlen_q_sparse_tensor * self.heads_per_group
                )
                metadata.max_seqlen_q_adjusted = (
                    seqlen_q_sparse_tensor.max().item() * self.heads_per_group
                )
            else:
                metadata.cu_seqlens_q_adjusted = (
                    metadata.cu_seqlens_q * self.heads_per_group
                )
                metadata.max_seqlen_q_adjusted = (
                    metadata.max_seq_len_q * self.heads_per_group
                )
        else:
            decode_metadata = self.sparse_metadata_builder.build_sparse_decode_metadata(
                forward_batch=forward_batch,
                base_metadata=metadata,
                head_group_num=self.head_group_num,
                dense_len=self.dense_len,
                sparse_topk=self.sparse_topk,
                block_size=self.block_size,
            )

            metadata.sparse_cache_seqlens_int32 = decode_metadata[
                "sparse_cache_seqlens_int32"
            ]
            metadata.sparse_cu_seqlens_k = decode_metadata["sparse_cu_seqlens_k"]
            metadata.sparse_cu_seqlens_q = decode_metadata["sparse_cu_seqlens_q"]
            metadata.sparse_page_table = decode_metadata["sparse_page_table"]
            metadata.token_to_bs = decode_metadata["token_to_bs"]

            # Stage1 optimization metadata for decode mode
            metadata.cache_seqlens_int32_stage1 = metadata.cache_seqlens_int32 - 1
            metadata.cu_seqlens_q_adjusted = (
                metadata.cu_seqlens_q * self.heads_per_group
            )
            metadata.max_seqlen_q_adjusted = (
                metadata.max_seq_len_q * self.heads_per_group
            )

    def init_forward_metadata(self, forward_batch: ForwardBatch):
        """Initialize forward metadata hence all layers in the forward pass can reuse it."""
        if forward_batch.forward_mode.is_target_verify():
            raise NotImplementedError(
                "MiniCPM backend does not support speculative decoding (target verify)"
            )
        if forward_batch.forward_mode.is_draft_extend_v2():
            raise NotImplementedError(
                "MiniCPM backend does not support speculative decoding (draft extend)"
            )

        metadata = MiniCPMBackendMetadata()
        seqlens_in_batch = forward_batch.seq_lens
        batch_size = forward_batch.batch_size
        device = seqlens_in_batch.device

        if forward_batch.forward_mode.is_decode_or_idle():
            metadata.cache_seqlens_int32 = seqlens_in_batch.to(torch.int32)
            metadata.max_seq_len_q = 1
            metadata.max_seq_len_k = forward_batch.seq_lens_cpu.max().item()
            metadata.cu_seqlens_q = torch.arange(
                0, batch_size + 1, dtype=torch.int32, device=device
            )
            metadata.cu_seqlens_k = torch.nn.functional.pad(
                torch.cumsum(seqlens_in_batch, dim=0, dtype=torch.int32), (1, 0)
            )
            metadata.page_table = self.req_to_token_pool.req_to_token[
                forward_batch.req_pool_indices, : metadata.max_seq_len_k
            ]
        elif forward_batch.forward_mode.is_extend_or_draft_extend_or_mixed(
            include_draft_extend_v2=True
        ):
            metadata.cache_seqlens_int32 = seqlens_in_batch.to(torch.int32)
            metadata.max_seq_len_k = forward_batch.seq_lens_cpu.max().item()
            metadata.cu_seqlens_k = torch.nn.functional.pad(
                torch.cumsum(seqlens_in_batch, dim=0, dtype=torch.int32), (1, 0)
            )
            metadata.page_table = self.req_to_token_pool.req_to_token[
                forward_batch.req_pool_indices, : metadata.max_seq_len_k
            ]

            if any(forward_batch.extend_prefix_lens_cpu):
                extend_seq_lens = forward_batch.extend_seq_lens
                metadata.max_seq_len_q = max(forward_batch.extend_seq_lens_cpu)
                metadata.cu_seqlens_q = torch.nn.functional.pad(
                    torch.cumsum(extend_seq_lens, dim=0, dtype=torch.int32), (1, 0)
                )
            else:
                metadata.max_seq_len_q = metadata.max_seq_len_k
                metadata.cu_seqlens_q = metadata.cu_seqlens_k

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
            metadata.encoder_page_table = self.req_to_token_pool.req_to_token[
                forward_batch.req_pool_indices, : metadata.encoder_max_seq_len_k
            ]

            # Currently only support forward_batch.encoder_lens.numel() == 1
            metadata.page_table = self.req_to_token_pool.req_to_token[
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

        self.update_batch_for_sparse(forward_batch, metadata)

        self.forward_metadata = metadata

    def get_topk_for_sparse(
        self,
        query_states,
        key_states,
        value_states,
        query_length,
        layer,
        forward_batch,
        is_prefill=True,
        dropout=0.0,
        softmax_scale=None,
        no_rope_param=None,
        past_key_value=None,
        decode_batch_id=0,
    ):
        if is_prefill:

            all_sparse = (
                self.forward_metadata.sparse_batch_size == forward_batch.batch_size
            )
            if all_sparse:
                # all batch is sparse
                metadata = self.forward_metadata
                compressed_k = torch.full(
                    (
                        forward_batch.batch_size
                        * self.max_context_len
                        // self.k1_kernel_stride,
                        self.head_group_num,
                        self.head_dim,
                    ),
                    dtype=torch.bfloat16,
                    device=self.device,
                    fill_value=float("-inf"),
                )
                compressed_k2 = torch.full(
                    (
                        forward_batch.batch_size
                        * self.max_context_len
                        // self.k2_kernel_stride,
                        self.head_group_num,
                        self.head_dim,
                    ),
                    dtype=torch.bfloat16,
                    device=self.device,
                    fill_value=float("-inf"),
                )

                get_compress_k_v2(
                    layer=layer,
                    forward_batch=forward_batch,
                    metadata=metadata,
                    full_compressed_k1=compressed_k,  # output
                    full_compressed_k2=compressed_k2,  # output
                    max_context_length=self.max_context_len,
                )

                cu_seqlens_k = metadata.cu_seqlens_k
                max_seqlen_in_batch_k = metadata.max_seq_len_k
                cu_seqlens_q = metadata.cu_seqlens_q
                max_seqlen_in_batch_q = metadata.max_seq_len_q

                ret = self.sparse_get_topk_impl(
                    query_states,
                    cu_seqlens_q,
                    cu_seqlens_k,
                    max_seqlen_in_batch_q,
                    max_seqlen_in_batch_k,
                    no_rope_param=no_rope_param,
                    compressed_k=compressed_k,
                    compressed_cu_seqlens=metadata.k1.cu_seqlens,
                    compressed_k2=compressed_k2,
                    compressed_cu_seqlens2=metadata.k2.cu_seqlens,
                    fused_kernel=(
                        self.prefill_fused_kernels[forward_batch.batch_size]
                        if self.minicpm_fuse_topk
                        else None
                    ),
                )
                return ret

            topk_metadata = self.sparse_metadata_builder.build_prefill_topk_metadata(
                forward_batch=forward_batch,
                base_metadata=self.forward_metadata,
                key_states=key_states,
                query_states=query_states,
                tp_q_head_num=layer.tp_q_head_num,
                head_dim=layer.head_dim,
                compress_k1_kernel_size=self.k1_kernel_size,
                compress_k1_kernel_stride=self.k1_kernel_stride,
                compress_k2_kernel_size=self.k2_kernel_size,
                compress_k2_kernel_stride=self.k2_kernel_stride,
                dense_len=self.dense_len,
            )

            sparse_bs = topk_metadata["sparse_bs"]
            topk_metadata["seqlens_q_sparse_bs"]
            topk_metadata["seqlens_k_sparse_bs"]
            k1_lens = topk_metadata["k1_lens"]
            k2_lens = topk_metadata["k2_lens"]

            full_compressed_k1, full_compressed_k2 = allocate_and_compress_keys(
                layer=layer,
                forward_batch=forward_batch,
                metadata=self.forward_metadata,
                k1_token_nums=sum(k1_lens),
                k2_token_nums=sum(k2_lens),
                dtype=key_states.dtype,
                device=key_states.device,
                max_context_length=self.max_context_len,
                minicpm_split_stage1=self.minicpm_split_stage1,
            )

            pt_k1, pt_k2 = 0, 0
            compressed_k = torch.zeros(
                (sum(k1_lens[sparse_bs]), layer.tp_k_head_num, layer.head_dim),
                dtype=key_states.dtype,
                device=key_states.device,
            )
            compressed_k2 = torch.zeros(
                (sum(k2_lens[sparse_bs]), layer.tp_k_head_num, layer.head_dim),
                dtype=key_states.dtype,
                device=key_states.device,
            )

            compressed_cu_seqlens, compressed_cu_seqlens2 = [0], [0]

            for sparse_bs_idx in sparse_bs:
                start = self.forward_metadata.k1.cu_seqlens[sparse_bs_idx]
                end = self.forward_metadata.k1.cu_seqlens[sparse_bs_idx + 1]
                compressed_k[pt_k1 : pt_k1 + (end - start), :, :] = full_compressed_k1[
                    start:end, :, :
                ]

                start2 = self.forward_metadata.k2.cu_seqlens[sparse_bs_idx]
                end2 = self.forward_metadata.k2.cu_seqlens[sparse_bs_idx + 1]
                compressed_k2[pt_k2 : pt_k2 + (end2 - start2), :, :] = (
                    full_compressed_k2[start2:end2, :, :]
                )

                pt_k1 += k1_lens[sparse_bs_idx]
                pt_k2 += k2_lens[sparse_bs_idx]
                compressed_cu_seqlens.append(
                    compressed_cu_seqlens[-1] + k1_lens[sparse_bs_idx]
                )
                compressed_cu_seqlens2.append(
                    compressed_cu_seqlens2[-1] + k2_lens[sparse_bs_idx]
                )

            compressed_cu_seqlens = torch.tensor(
                compressed_cu_seqlens, dtype=torch.int32, device=key_states.device
            )
            compressed_cu_seqlens2 = torch.tensor(
                compressed_cu_seqlens2, dtype=torch.int32, device=key_states.device
            )

            cu_seqlens_q = topk_metadata["cu_seqlens_q"]
            cu_seqlens_k = topk_metadata["cu_seqlens_k"]
            max_seqlen_in_batch_q = topk_metadata["max_seqlen_q"]
            max_seqlen_in_batch_k = topk_metadata["max_seqlen_k"]
            query_states = topk_metadata["query_states"]

            ret = self.sparse_get_topk_impl(
                query_states,
                cu_seqlens_q,
                cu_seqlens_k,
                max_seqlen_in_batch_q,
                max_seqlen_in_batch_k,
                no_rope_param=no_rope_param,
                compressed_k=compressed_k,
                compressed_cu_seqlens=compressed_cu_seqlens,
                compressed_k2=compressed_k2,
                compressed_cu_seqlens2=compressed_cu_seqlens2,
                fused_kernel=(
                    self.prefill_fused_kernels[forward_batch.batch_size]
                    if self.minicpm_fuse_topk
                    else None
                ),
            )
            return ret
        else:
            metadata = self.forward_metadata

            if self.enable_cuda_graph:
                if self.minicpm_split_stage1:
                    get_compress_k_v2_padded(
                        layer=layer,
                        forward_batch=forward_batch,
                        metadata=metadata,
                        full_compressed_k1=self.decode_cuda_graph_metadata[
                            "compress_k1"
                        ][
                            : forward_batch.batch_size
                            * self.max_context_len
                            // self.k1_kernel_stride,
                            :,
                            :,
                        ],
                        full_compressed_k2=self.decode_cuda_graph_metadata[
                            "compress_k2"
                        ][
                            : forward_batch.batch_size
                            * self.max_context_len
                            // self.k2_kernel_stride,
                            :,
                            :,
                        ],
                        max_context_length=self.max_context_len,
                    )
                else:
                    get_compress_k_v2(
                        layer=layer,
                        forward_batch=forward_batch,
                        metadata=metadata,
                        full_compressed_k1=self.decode_cuda_graph_metadata[
                            "compress_k1"
                        ][
                            : forward_batch.batch_size
                            * self.max_context_len
                            // self.k1_kernel_stride,
                            :,
                            :,
                        ],
                        full_compressed_k2=self.decode_cuda_graph_metadata[
                            "compress_k2"
                        ][
                            : forward_batch.batch_size
                            * self.max_context_len
                            // self.k2_kernel_stride,
                            :,
                            :,
                        ],
                        max_context_length=self.max_context_len,
                    )
            else:
                compressed_k, compressed_k2 = allocate_and_compress_keys(
                    layer=layer,
                    forward_batch=forward_batch,
                    metadata=metadata,
                    k1_token_nums=forward_batch.batch_size
                    * self.max_context_len
                    // self.k1_kernel_stride,
                    k2_token_nums=forward_batch.batch_size
                    * self.max_context_len
                    // self.k2_kernel_stride,
                    dtype=torch.bfloat16,
                    device=self.device,
                    max_context_length=self.max_context_len,
                    minicpm_split_stage1=self.minicpm_split_stage1,
                )

            topk_metadata = self.sparse_metadata_builder.build_decode_topk_metadata(
                forward_batch=forward_batch,
                base_metadata=metadata,
                query_states=query_states,
            )

            cu_seqlens_q = topk_metadata["cu_seqlens_q"]
            cu_seqlens_k = topk_metadata["cu_seqlens_k"]
            max_seqlen_in_batch_q = topk_metadata["max_seqlen_q"]
            max_seqlen_in_batch_k = topk_metadata["max_seqlen_k"]
            query_states = topk_metadata["query_states"]

            if self.enable_cuda_graph:
                ret = self.sparse_get_topk_impl(
                    query_states,
                    cu_seqlens_q,
                    cu_seqlens_k,
                    max_seqlen_in_batch_q,
                    max_seqlen_in_batch_k,
                    no_rope_param=no_rope_param,
                    compressed_k=self.decode_cuda_graph_metadata["compress_k1"][
                        : forward_batch.batch_size
                        * self.max_context_len
                        // self.k1_kernel_stride,
                        :,
                        :,
                    ],
                    compressed_cu_seqlens=metadata.k1.cu_seqlens,
                    compressed_k2=self.decode_cuda_graph_metadata["compress_k2"][
                        : forward_batch.batch_size
                        * self.max_context_len
                        // self.k2_kernel_stride,
                        :,
                        :,
                    ],
                    compressed_cu_seqlens2=metadata.k2.cu_seqlens,
                    fused_kernel=(
                        self.decode_fused_kernels[forward_batch.batch_size]
                        if self.minicpm_fuse_topk
                        else None
                    ),
                )

            else:
                ret = self.sparse_get_topk_impl(
                    query_states,
                    cu_seqlens_q,
                    cu_seqlens_k,
                    max_seqlen_in_batch_q,
                    max_seqlen_in_batch_k,
                    no_rope_param=no_rope_param,
                    compressed_k=compressed_k,
                    compressed_cu_seqlens=metadata.k1.cu_seqlens,
                    compressed_k2=compressed_k2,
                    compressed_cu_seqlens2=metadata.k2.cu_seqlens,
                    fused_kernel=(
                        self.decode_fused_kernels[forward_batch.batch_size]
                        if self.minicpm_fuse_topk
                        else None
                    ),
                )

        return ret

    def sparse_get_topk_impl(
        self,
        query_layer,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_in_batch_q,
        max_seqlen_in_batch_k,
        #    max_seqlen_k1,
        no_rope_param=None,
        compressed_k=None,
        compressed_cu_seqlens=None,
        compressed_k2=None,
        compressed_cu_seqlens2=None,
        fused_kernel=None,
    ):
        cache_lens = None
        if max_seqlen_in_batch_k > max_seqlen_in_batch_q:
            if max_seqlen_in_batch_q == 1:
                cache_lens = self.forward_metadata.cache_seqlens_int32_stage1
            else:
                seq_lens_k = cu_seqlens_k[1:] - cu_seqlens_k[:-1]
                seq_lens_q = cu_seqlens_q[1:] - cu_seqlens_q[:-1]
                cache_lens = seq_lens_k - seq_lens_q
        else:
            batch_size = cu_seqlens_q.shape[0] - 1
            cache_lens = torch.zeros(
                batch_size, dtype=torch.int32, device=cu_seqlens_q.device
            )

        if not self.minicpm_fuse_topk:
            topk_idx = compressed_attention(
                (
                    query_layer
                    if no_rope_param is None
                    else no_rope_param["query_states_no_rope"]
                ),
                compressed_k,
                compressed_k2,
                self.kernel_size,
                self.kernel_stride,
                self.block_size,
                self.sparse_topk,
                cu_seqlens_q,
                compressed_cu_seqlens,
                compressed_cu_seqlens2,
                max_seqlen_in_batch_q,
                # self.max_context_len // self.kernel_size,
                self.max_context_len,
                None,
                init_blocks=self.init_blocks,
                local_blocks=self.local_blocks,
                cache_lens=cache_lens,
                cu_seqlens_q_adjusted=self.forward_metadata.cu_seqlens_q_adjusted,
                max_seqlen_q_adjusted=self.forward_metadata.max_seqlen_q_adjusted,
                # block_score_buffer=self.forward_metadata.block_score_buffer
                minicpm_split_stage1=self.minicpm_split_stage1,
            )
        else:
            topk_idx = compressed_attention_tilelang(
                (
                    query_layer
                    if no_rope_param is None
                    else no_rope_param["query_states_no_rope"]
                ),
                compressed_k,
                compressed_k2,
                self.kernel_size,
                self.kernel_stride,
                self.block_size,
                self.sparse_topk,
                cu_seqlens_q,
                compressed_cu_seqlens,
                compressed_cu_seqlens2,
                max_seqlen_in_batch_q,
                self.forward_metadata.k1.max_seq_len,
                None,
                init_blocks=self.init_blocks,
                local_blocks=self.local_blocks,
                cache_lens=cache_lens,
                fused_kernel=fused_kernel,
                max_cache_len=self.max_context_len,
            )

        return topk_idx

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
        sinks: Optional[torch.Tensor] = None,
    ):
        if layer.is_cross_attention:
            raise NotImplementedError(
                "MiniCPM backend does not support cross attention"
            )
        if forward_batch.forward_mode.is_draft_extend_v2():
            raise NotImplementedError(
                "MiniCPM backend does not support draft extend mode"
            )

        if k is not None:
            assert v is not None
            if save_kv_cache:
                cache_loc = forward_batch.out_cache_loc
                self.token_to_kv_pool.set_kv_buffer(
                    layer, cache_loc, k, v, layer.k_scale, layer.v_scale
                )

        # Use precomputed metadata across all layers
        metadata = self.forward_metadata

        # Calculate window size (can be moved to metadata if layer properties don't change)
        # we don't do layer.sliding_window_size - 1 since in model.get_attention_sliding_window_size() we already - 1
        # here is two side inclusive
        is_swa_layer = (
            layer.sliding_window_size is not None and layer.sliding_window_size > -1
        )
        window_size = (layer.sliding_window_size, 0) if is_swa_layer else (-1, -1)
        k_descale, v_descale = None, None
        # only use kv scaling if: 1) fp8 kv is explicitly enabled, 2) RadixAttention
        # has corresponding quantization method so that layer.k_scale is not None,
        # 3) layer.head_dim <= 256 since fa3 kernel require fp16 and bf16 data type in this case,
        # 4) fa_impl_ver != 4 since fa4 does not currently support fp8 queries and keys.
        if (
            self.kv_cache_dtype_str != "auto"
            and layer.head_dim <= 256
            and self.fa_impl_ver != 4
        ):
            if layer.k_scale is not None:
                descale_shape = (forward_batch.batch_size, layer.tp_k_head_num)
                k_descale = layer.k_scale.expand(descale_shape)
                v_descale = layer.v_scale.expand(descale_shape)
            q = q.to(self.kv_cache_dtype)
            q_rope = q_rope.to(self.kv_cache_dtype) if q_rope is not None else None
            k_rope = k_rope.to(self.kv_cache_dtype) if k_rope is not None else None
        # MiniCPM backend does not support cross attention or encoder-only attention
        causal = True

        # For fa3 interface version compatibility, we put new fields into conditional keyword args
        kwargs = {}
        if self.fa_impl_ver != 3:
            kwargs["ver"] = self.fa_impl_ver
        if sinks is not None:
            kwargs["sinks"] = sinks

        # Get the appropriate page table (without local attention or SWA support)
        page_table = metadata.page_table
        cache_seqlens = metadata.cache_seqlens_int32
        cu_seqlens_k = metadata.cu_seqlens_k

        bs = forward_batch.batch_size
        if max(forward_batch.seq_lens_cpu) >= self.dense_len:
            q_reshaped = q.contiguous().view(-1, layer.tp_q_head_num, layer.head_dim)
            topk_idx = self.get_topk_for_sparse(
                q_reshaped, k, v, q.shape[0], layer, forward_batch
            )

            sparse_page_table_sparse_bs = get_block_table_v2(
                topk_idx,
                page_table,
                metadata.token_to_bs,
                metadata.token_pos_in_bs,
                metadata.seqlen_k_sparse_bs_tensor,
            ).reshape(-1, self.num_sparse_topk_tokens)

            # copy page table for sparse bs
            metadata.sparse_page_table[
                metadata.sparse_idx, : self.num_sparse_topk_tokens
            ] = sparse_page_table_sparse_bs
        else:
            total_k1 = self.forward_metadata.k1.cu_total_compress_token_nums[-1].item()
            total_k2 = self.forward_metadata.k2.cu_total_compress_token_nums[-1].item()

            full_compressed_k1_ext, full_compressed_k2_ext = allocate_and_compress_keys(
                layer=layer,
                forward_batch=forward_batch,
                metadata=self.forward_metadata,
                k1_token_nums=total_k1,
                k2_token_nums=total_k2,
                dtype=k.dtype,
                device=k.device,
                max_context_length=self.max_context_len,
                minicpm_split_stage1=self.minicpm_split_stage1,
            )

        q_reshaped = q.contiguous().view(-1, layer.tp_q_head_num // 2, layer.head_dim)
        if metadata.sparse_batch_size < bs:
            # copy dense page table for dense bs
            metadata.sparse_page_table.shape[1]
            dense_bs_list = [i for i in range(bs) if i not in metadata.sparse_bs_list]
            for dense_bs in dense_bs_list:
                kv_len = forward_batch.seq_lens_cpu[dense_bs]
                sparse_page_table_idx_start = metadata.old_bs_to_new_bs_range[dense_bs]
                sparse_page_table_idx_end = metadata.old_bs_to_new_bs_range[
                    dense_bs + 1
                ]
                assert (
                    sparse_page_table_idx_end - sparse_page_table_idx_start == 2
                ), "dense bs should have 2 head_group, but get {}".format(
                    sparse_page_table_idx_end - sparse_page_table_idx_start
                )

                ps = metadata.sparse_cu_seqlens_q_cpu[sparse_page_table_idx_start]
                len_ = (
                    metadata.sparse_cu_seqlens_q_cpu[sparse_page_table_idx_start + 1]
                    - ps
                )
                assert (
                    len_ == forward_batch.extend_seq_lens_cpu[dense_bs]
                ), "dense bs seqlen mismatch {} vs {}".format(
                    len_, forward_batch.extend_seq_lens_cpu[dense_bs]
                )
                t = q_reshaped[ps : ps + 2 * len_, :, :].clone()
                q_reshaped[ps : ps + len_, :, :] = t[0::2, :, :]
                q_reshaped[ps + len_ : ps + 2 * len_, :, :] = t[1::2, :, :]

                metadata.sparse_page_table[sparse_page_table_idx_start, :kv_len] = (
                    page_table[dense_bs, :kv_len] * 2
                )
                metadata.sparse_page_table[sparse_page_table_idx_start + 1, :kv_len] = (
                    page_table[dense_bs, :kv_len] * 2 + 1
                )

        metadata.sparse_cache_seqlens_int32 = (
            (metadata.sparse_page_table != 0)
            .sum(dim=1)
            .to(dtype=cache_seqlens.dtype, device=cache_seqlens.device)
        )

        # this seem not necessary to update perlayer
        metadata.sparse_cu_seqlens_k = F.pad(
            torch.cumsum(
                metadata.sparse_cache_seqlens_int32, dim=0, dtype=cu_seqlens_k.dtype
            ),
            (1, 0),
        )

        key_cache, value_cache = self.token_to_kv_pool.get_kv_buffer(layer.layer_id)

        key_cache = key_cache.view(
            -1, self.page_size, layer.tp_k_head_num // 2, layer.head_dim
        )
        value_cache = value_cache.view(
            -1, self.page_size, layer.tp_v_head_num // 2, layer.head_dim
        )

        # Prepare attention parameters
        attn_params = AttentionParams(
            q=q.contiguous().view(-1, layer.tp_q_head_num // 2, layer.head_dim),
            k_cache=key_cache,
            v_cache=value_cache,
            page_table=metadata.sparse_page_table,
            cache_seqlens=metadata.sparse_cache_seqlens_int32,
            cu_seqlens_q=metadata.sparse_cu_seqlens_q,
            cu_seqlens_k_new=metadata.sparse_cu_seqlens_k,
            max_seqlen_q=metadata.sparse_max_seq_len_q,
            softmax_scale=layer.scaling,
            causal=causal,
            window_size=window_size,
            softcap=layer.logit_cap,
            k_descale=k_descale,
            v_descale=v_descale,
            num_splits=self.num_splits,
            fa_impl_ver=self.fa_impl_ver,
        )

        # Use the attention kernel abstraction
        result = self.attention_kernel.forward(attn_params)

        if metadata.sparse_batch_size < bs:
            metadata.sparse_page_table.shape[1]
            dense_bs_list = [i for i in range(bs) if i not in metadata.sparse_bs_list]
            for dense_bs in dense_bs_list:
                sparse_page_table_idx_start = metadata.old_bs_to_new_bs_range[dense_bs]
                sparse_page_table_idx_end = metadata.old_bs_to_new_bs_range[
                    dense_bs + 1
                ]
                assert (
                    sparse_page_table_idx_end - sparse_page_table_idx_start == 2
                ), "dense bs should have 2 head_group"

                ps = metadata.sparse_cu_seqlens_q_cpu[sparse_page_table_idx_start]
                len_ = (
                    metadata.sparse_cu_seqlens_q_cpu[sparse_page_table_idx_start + 1]
                    - ps
                )
                assert (
                    len_ == forward_batch.extend_seq_lens_cpu[dense_bs]
                ), "dense bs seqlen mismatch {} vs {}".format(
                    len_, forward_batch.extend_seq_lens_cpu[dense_bs]
                )
                t = result[ps : ps + 2 * len_, :, :].clone()
                result[ps : ps + 2 * len_ : 2, :, :] = t[0:len_, :, :]
                result[ps + 1 : ps + 2 * len_ : 2, :, :] = t[len_ : 2 * len_, :, :]

        return result.view(-1, layer.tp_q_head_num * layer.head_dim)

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
        sinks: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        assert self.fa_impl_ver in [3], "Only FA3 support decoding"

        # Check for unsupported features
        if layer.is_cross_attention:
            raise NotImplementedError(
                "MiniCPM backend does not support cross attention"
            )
        # MiniCPM does not support local attention

        bs = forward_batch.batch_size
        if k is not None:
            assert v is not None
            if save_kv_cache:
                cache_loc = forward_batch.out_cache_loc
                self.token_to_kv_pool.set_kv_buffer(
                    layer, cache_loc, k, v, layer.k_scale, layer.v_scale
                )

        # Use precomputed metadata across all layers
        metadata = self.forward_metadata

        # Calculate window size (can be moved to metadata if layer properties don't change)
        # we don't do layer.sliding_window_size - 1 since in model.get_attention_sliding_window_size() we already - 1
        # here is two side inclusive
        is_swa_layer = (
            layer.sliding_window_size is not None and layer.sliding_window_size > -1
        )
        window_size = (layer.sliding_window_size, 0) if is_swa_layer else (-1, -1)

        # MiniCPM backend does not support cross attention or encoder-only attention
        causal = True

        # For fa3 interface version compatibility, we put new fields into conditional keyword args
        kwargs = {}
        if self.fa_impl_ver != 3:
            kwargs["ver"] = self.fa_impl_ver
        if sinks is not None:
            kwargs["sinks"] = sinks

        k_descale, v_descale = None, None
        # only use kv scaling if: 1) fp8 kv is explicitly enabled, 2) RadixAttention
        # has corresponding quantization method so that layer.k_scale is not None,
        # 3) layer.head_dim <= 256 since fa3 kernel require fp16 and bf16 data type in this case.
        if self.kv_cache_dtype_str != "auto" and layer.head_dim <= 256:
            if layer.k_scale is not None:
                descale_shape = (forward_batch.batch_size, layer.tp_k_head_num)
                k_descale = layer.k_scale.expand(descale_shape)
                v_descale = layer.v_scale.expand(descale_shape)
            q = q.to(self.kv_cache_dtype)
            q_rope = q_rope.to(self.kv_cache_dtype) if q_rope is not None else None
            k_rope = k_rope.to(self.kv_cache_dtype) if k_rope is not None else None
        # Do multi-head attention (without cross-attention or local attention support)

        key_cache, value_cache = self.token_to_kv_pool.get_kv_buffer(layer.layer_id)
        key_cache = key_cache.view(
            -1, self.page_size, layer.tp_k_head_num, layer.head_dim
        )
        value_cache = value_cache.view(
            -1, self.page_size, layer.tp_v_head_num, layer.v_head_dim
        )

        page_table = metadata.page_table
        cache_seqlens = metadata.cache_seqlens_int32
        max_seqlen_q = metadata.max_seq_len_q
        q_reshaped = q.contiguous().view(-1, layer.tp_q_head_num, layer.head_dim)

        topk_idx = self.get_topk_for_sparse(
            q_reshaped.unsqueeze(0),
            k.unsqueeze(0),
            v.unsqueeze(0),
            1,
            layer,
            forward_batch,
            False,
        )
        sparse_page_table = get_block_table_v3(
            topk_idx,
            page_table,
            metadata.token_to_bs,
            cache_seqlens,
            cache_seqlens,
        ).reshape(-1, self.num_sparse_topk_tokens)

        metadata.sparse_page_table[: 2 * bs, : self.num_sparse_topk_tokens] = (
            sparse_page_table[:, : self.num_sparse_topk_tokens]
        )

        q_reshaped_by_head_group = q_reshaped.reshape(
            -1, layer.tp_q_head_num // 2, layer.head_dim
        )
        assert self.page_size == 1
        key_cache_by_head_group = key_cache.reshape(
            -1, self.page_size, layer.tp_k_head_num // 2, layer.head_dim
        )
        value_cache_by_head_group = value_cache.reshape(
            -1, self.page_size, layer.tp_v_head_num // 2, layer.head_dim
        )

        # prepare seqlen_k and it's presum
        sparse_cache_seqlens = metadata.sparse_cache_seqlens_int32
        sparse_cu_seqlens_k = metadata.sparse_cu_seqlens_k
        sparse_cu_seqlens_q = metadata.sparse_cu_seqlens_q

        # Prepare attention parameters
        # For CUDA graph mode, use decode_wrapper. Otherwise pass pre-converted metadata.
        decode_wrapper = metadata.decode_wrapper
        if decode_wrapper is not None:
            flashinfer_kv_indptr = None
            flashinfer_kv_indices = None
            flashinfer_kv_last_page_len = None
        else:
            flashinfer_kv_indptr = metadata.flashinfer_kv_indptr
            flashinfer_kv_indices = metadata.flashinfer_kv_indices
            flashinfer_kv_last_page_len = metadata.flashinfer_kv_last_page_len

        attn_params = AttentionParams(
            q=q_reshaped_by_head_group,
            k_cache=key_cache_by_head_group,
            v_cache=value_cache_by_head_group,
            page_table=metadata.sparse_page_table,
            cache_seqlens=sparse_cache_seqlens,
            cu_seqlens_q=sparse_cu_seqlens_q,
            cu_seqlens_k_new=sparse_cu_seqlens_k,
            max_seqlen_q=max_seqlen_q,
            softmax_scale=layer.scaling,
            causal=causal,
            window_size=window_size,
            softcap=layer.logit_cap,
            k_descale=k_descale,
            v_descale=v_descale,
            num_splits=self.num_splits,
            fa_impl_ver=self.fa_impl_ver,
            # Flashinfer metadata or wrapper (mutually exclusive for CUDA graph compatibility)
            decode_wrapper=decode_wrapper,
            flashinfer_kv_indptr=flashinfer_kv_indptr,
            flashinfer_kv_indices=flashinfer_kv_indices,
            flashinfer_kv_last_page_len=flashinfer_kv_last_page_len,
        )

        # Use the attention kernel abstraction
        result = self.attention_kernel.forward(attn_params)

        o = result

        return o.view(-1, layer.tp_q_head_num * layer.v_head_dim)

    def init_cuda_graph_state(self, max_bs: int, max_num_tokens: int):
        """Initialize CUDA graph state for the attention backend.

        Args:
            max_bs (int): Maximum batch size to support in CUDA graphs

        This creates fixed-size tensors that will be reused during CUDA graph replay
        to avoid memory allocations.
        """
        max_num_pages = (self.max_context_len + self.page_size - 1) // self.page_size
        max_k1_num_pages = (
            (self.max_context_len - self.k1_kernel_size) // self.k1_kernel_stride
            + 1
            + self.page_size
            - 1
        ) // self.page_size
        max_k2_num_pages = (
            (self.max_context_len - self.k2_kernel_size) // self.k2_kernel_stride
            + 1
            + self.page_size
            - 1
        ) // self.page_size
        sparse_max_num_pages = (
            self.num_sparse_topk_tokens + self.page_size - 1
        ) // self.page_size

        # Precompute kv_indptr for sparse mode (cache_seqlens is fixed)
        # For sparse mode, kv_indptr[i] = i * num_sparse_topk_tokens
        precomputed_kv_indptr = (
            torch.arange(0, max_bs * 2 + 1, dtype=torch.int32, device=self.device)
            * self.num_sparse_topk_tokens
        )

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
                max_num_pages,
                dtype=torch.int32,
                device=self.device,
            ),
            "strided_indices": torch.arange(
                0, self.max_context_len, self.page_size, device=self.device
            ),
            **(
                {
                    # sparse attention related metadata
                    # For sparse attention, cache_seqlens is fixed to num_sparse_topk_tokens
                    "sparse_cache_seqlens": torch.full(
                        (max_bs * 2,),
                        self.num_sparse_topk_tokens,
                        dtype=torch.int32,
                        device=self.device,
                    ),
                    "sparse_cu_seqlens_q": torch.arange(
                        0, max_bs * 2 + 1, dtype=torch.int32, device=self.device
                    ),
                    # For sparse mode, cu_seqlens_k[i] = i * num_sparse_topk_tokens
                    "sparse_cu_seqlens_k": torch.arange(
                        0,
                        (max_bs * 2 + 1) * self.num_sparse_topk_tokens,
                        self.num_sparse_topk_tokens,
                        dtype=torch.int32,
                        device=self.device,
                    ),
                    "token_to_bs": torch.arange(
                        0, max_bs, dtype=torch.int32, device=self.device
                    ),
                    "token_pos_in_bs": torch.ones(
                        max_bs, dtype=torch.int32, device=self.device
                    ),
                    "sparse_page_table": torch.zeros(
                        max_bs * 2,
                        sparse_max_num_pages,
                        dtype=torch.int32,
                        device=self.device,
                    ),
                    # TODO more precisely, it is max(0, (max_context_length - kernel_size) // kernel_stride + 1)
                    "compress_k1": torch.zeros(
                        (
                            max_bs * self.max_context_len // self.k1_kernel_stride,
                            self.head_group_num,
                            self.head_dim,
                        ),
                        dtype=torch.bfloat16,
                        device=self.device,
                    ),
                    "compress_k2": torch.zeros(
                        (
                            max_bs * self.max_context_len // self.k2_kernel_stride,
                            self.head_group_num,
                            self.head_dim,
                        ),
                        dtype=torch.bfloat16,
                        device=self.device,
                    ),
                    "k1.cu_seqlens": torch.zeros(
                        max_bs + 1, dtype=torch.int32, device=self.device
                    ),
                    "k2.cu_seqlens": torch.zeros(
                        max_bs + 1, dtype=torch.int32, device=self.device
                    ),
                    # too many arrays are stored, in order to support cuda graph, they are temporarily added,
                    # the parameters required for compress_k_core_new are reduced
                    # k1
                    "k1.table": torch.zeros(
                        max_bs,
                        max_k1_num_pages,
                        dtype=torch.int32,
                        device=self.device,
                    ),
                    "k1.history_compress_token_nums": torch.zeros(
                        max_bs, dtype=torch.int32, device=self.device
                    ),
                    "k1.new_token_nums": torch.zeros(
                        max_bs, dtype=torch.int32, device=self.device
                    ),
                    "k1.new_compress_token_nums": torch.zeros(
                        max_bs, dtype=torch.int32, device=self.device
                    ),
                    "k1.total_compress_token_nums": torch.zeros(
                        max_bs, dtype=torch.int32, device=self.device
                    ),
                    "k1.cu_new_token_nums": torch.zeros(
                        max_bs + 1, dtype=torch.int32, device=self.device
                    ),
                    "k1.cu_new_compress_token_nums": torch.zeros(
                        max_bs + 1, dtype=torch.int32, device=self.device
                    ),
                    "k1.cu_total_compress_token_nums": torch.zeros(
                        max_bs + 1, dtype=torch.int32, device=self.device
                    ),
                    # k2
                    "k2.table": torch.zeros(
                        max_bs,
                        max_k2_num_pages,
                        dtype=torch.int32,
                        device=self.device,
                    ),
                    "k2.history_compress_token_nums": torch.zeros(
                        max_bs, dtype=torch.int32, device=self.device
                    ),
                    "k2.new_token_nums": torch.zeros(
                        max_bs, dtype=torch.int32, device=self.device
                    ),
                    "k2.new_compress_token_nums": torch.zeros(
                        max_bs, dtype=torch.int32, device=self.device
                    ),
                    "k2.total_compress_token_nums": torch.zeros(
                        max_bs, dtype=torch.int32, device=self.device
                    ),
                    "k2.cu_new_token_nums": torch.zeros(
                        max_bs + 1, dtype=torch.int32, device=self.device
                    ),
                    "k2.cu_new_compress_token_nums": torch.zeros(
                        max_bs + 1, dtype=torch.int32, device=self.device
                    ),
                    "k2.cu_total_compress_token_nums": torch.zeros(
                        max_bs + 1, dtype=torch.int32, device=self.device
                    ),
                    # Flashinfer-specific tensors (pre-allocated for CUDA graph)
                    # For sparse attention (tokens > 8192):
                    # - batch_size is doubled (2 head groups)
                    # - kv_indptr is PRECOMPUTED: [0, topk, 2*topk, ...]
                    # - Buffers are updated each replay by flattening sparse_page_table
                    "flashinfer_kv_indptr": precomputed_kv_indptr,
                    "flashinfer_kv_indices": torch.zeros(
                        max_bs * 2 * self.num_sparse_topk_tokens,
                        dtype=torch.int32,
                        device=self.device,
                    ),
                    "flashinfer_kv_last_page_len": torch.ones(
                        max_bs * 2, dtype=torch.int32, device=self.device
                    ),
                    # Stage1 optimization metadata
                    "cu_seqlens_q_adjusted": torch.arange(
                        0, max_bs + 1, dtype=torch.int32, device=self.device
                    )
                    * self.heads_per_group,
                    "cache_seqlens_int32_stage1": torch.zeros(
                        max_bs, dtype=torch.int32, device=self.device
                    ),
                }
            ),
        }

    def init_forward_metadata_out_graph(
        self,
        forward_batch: ForwardBatch,
        in_capture: bool = False,
    ):
        """New-ABC CUDA graph metadata entry (runs outside ``graph.capture()``).

        Adapts the legacy capture/replay split to main's single
        ``init_forward_metadata_out_graph(fb, in_capture)`` contract:

          * ``in_capture=True``  (capture prep): build slice-view metadata /
            FlashInfer wrapper, store it under ``decode_cuda_graph_metadata[bs]``.
          * ``in_capture=False`` (replay prep): refresh the underlying buffers
            and re-plan the wrapper before ``graph.replay()``.

        ``init_forward_metadata_in_graph`` stays the ABC no-op (nothing here is
        graph-recordable).
        """
        bs = forward_batch.batch_size
        req_pool_indices = forward_batch.req_pool_indices
        seq_lens = forward_batch.seq_lens
        forward_mode = forward_batch.forward_mode
        spec_info = forward_batch.spec_info
        encoder_lens = forward_batch.encoder_lens

        if in_capture:
            num_tokens = forward_batch.positions.numel()
            self._capture_cuda_graph_metadata(
                bs,
                num_tokens,
                req_pool_indices,
                seq_lens,
                encoder_lens,
                forward_mode,
                spec_info,
            )
        else:
            self._replay_cuda_graph_metadata(
                bs,
                req_pool_indices,
                seq_lens,
                forward_batch.seq_lens_sum,
                encoder_lens,
                forward_mode,
                spec_info,
                forward_batch.seq_lens_cpu,
                forward_batch.out_cache_loc,
                forward_batch,
            )

    def _build_sparse_decode_replay_metadata(
        self,
        forward_batch: ForwardBatch,
        metadata: MiniCPMBackendMetadata,
    ):
        decode_metadata = self.sparse_metadata_builder.build_sparse_decode_metadata(
            forward_batch=forward_batch,
            base_metadata=metadata,
            head_group_num=self.head_group_num,
            dense_len=self.dense_len,
            sparse_topk=self.sparse_topk,
            block_size=self.block_size,
        )
        compression_metadata = (
            self.sparse_metadata_builder.build_k1_k2_compression_metadata(
                forward_batch=forward_batch,
                base_metadata=metadata,
                req_to_sparse_k1_token=self.req_to_sparse_k1_token,
                req_to_sparse_k2_token=self.req_to_sparse_k2_token,
                k1_kernel_size=self.k1_kernel_size,
                k1_kernel_stride=self.k1_kernel_stride,
                k2_kernel_size=self.k2_kernel_size,
                k2_kernel_stride=self.k2_kernel_stride,
                cu_seqlens_q=metadata.cu_seqlens_q,
            )
        )
        return decode_metadata, compression_metadata

    def _capture_cuda_graph_metadata(
        self,
        bs: int,
        num_tokens: int,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        encoder_lens: Optional[torch.Tensor],
        forward_mode: ForwardMode,
        spec_info: Optional[SpecInput],
    ):
        """Initialize forward metadata for capturing CUDA graph.

        For FlashInfer sparse mode (tokens > 8192), uses wrapper state persistence pattern:

        Key Challenge:
        - sparse_page_table is computed DYNAMICALLY during forward() based on query vectors
        - FlashInfer requires kv_indptr/kv_indices converted BEFORE CUDA graph replay
        - Direct conversion fails because converted data becomes stale by replay time

        Solution: Wrapper State Persistence Pattern
        1. Create wrapper with BUFFER VIEWS (slices of pre-allocated tensors)
        2. Store wrapper references (addresses) - NOT the tensor data
        3. Before each replay: UPDATE UNDERLYING STORAGE at those addresses
        4. Call fast_decode_plan() to sync wrapper's cached pointers/metadata
        5. Wrapper accesses updated storage through same addresses

        Example:
        - Capture: wrapper = BatchDecodeWrapper(paged_kv_indptr_buffer=full[:bs+1], ...)
        - Replay (before graph): full[:bs+1].copy_(new_data)  # Update storage
        - Replay (before graph): fast_decode_plan(wrapper, ...)  # Sync pointers
        - Graph execution: wrapper uses updated data via same addresses

        For sparse mode:
        - kv_indptr is PRECOMPUTED and static: [0, K, 2K, ..., bs*K]
        - Only kv_indices needs updating per request
        - 128x memory reduction: bs*max_num_pages -> bs*num_sparse_topk_tokens
        """
        metadata = MiniCPMBackendMetadata()

        device = seq_lens.device
        if forward_mode.is_decode_or_idle():
            metadata.cache_seqlens_int32 = seq_lens.to(torch.int32)
            batch_size = len(seq_lens)
            device = seq_lens.device
            metadata.cu_seqlens_k = torch.nn.functional.pad(
                torch.cumsum(seq_lens, dim=0, dtype=torch.int32), (1, 0)
            )
            metadata.max_seq_len_k = seq_lens.max().item()
            metadata.page_table = self.decode_cuda_graph_metadata["page_table"][:bs, :]

            metadata.cu_seqlens_q = torch.arange(
                0, batch_size + 1, dtype=torch.int32, device=device
            )
            metadata.sparse_cache_seqlens_int32 = self.decode_cuda_graph_metadata[
                "sparse_cache_seqlens"
            ][: batch_size * 2]
            metadata.sparse_cu_seqlens_q = self.decode_cuda_graph_metadata[
                "sparse_cu_seqlens_q"
            ][: batch_size * 2 + 1]
            metadata.sparse_cu_seqlens_k = self.decode_cuda_graph_metadata[
                "sparse_cu_seqlens_k"
            ][: batch_size * 2 + 1]
            metadata.token_to_bs = self.decode_cuda_graph_metadata["token_to_bs"][
                :batch_size
            ]
            metadata.token_pos_in_bs = self.decode_cuda_graph_metadata[
                "token_pos_in_bs"
            ][:batch_size]
            metadata.sparse_page_table = self.decode_cuda_graph_metadata[
                "sparse_page_table"
            ][: batch_size * 2, :]

            metadata.k1 = CompressionLevelMetadata()
            metadata.k2 = CompressionLevelMetadata()

            metadata.k1.cu_seqlens = self.decode_cuda_graph_metadata["k1.cu_seqlens"][
                : batch_size + 1
            ]
            metadata.k2.cu_seqlens = self.decode_cuda_graph_metadata["k2.cu_seqlens"][
                : batch_size + 1
            ]
            assume_kv_len = self.config_dense_len
            assume_k1_len = (
                assume_kv_len - self.k1_kernel_size
            ) // self.k1_kernel_stride + 1
            assume_k2_len = (
                assume_kv_len - self.k2_kernel_size
            ) // self.k2_kernel_stride + 1
            for i in range(bs):
                metadata.cu_seqlens_k[i + 1] = metadata.cu_seqlens_k[i] + assume_kv_len
                metadata.k1.cu_seqlens[i + 1] = (
                    metadata.k1.cu_seqlens[i] + assume_k1_len
                )
                metadata.k2.cu_seqlens[i + 1] = (
                    metadata.k2.cu_seqlens[i] + assume_k2_len
                )

            metadata.max_seq_len_k = assume_kv_len
            metadata.k1.max_seq_len = assume_k1_len
            metadata.k2.max_seq_len = assume_k2_len

            metadata.total_q = bs

            # compress k1

            metadata.k1.history_compress_token_nums = self.decode_cuda_graph_metadata[
                "k1.history_compress_token_nums"
            ][:batch_size]
            metadata.k1.new_token_nums = self.decode_cuda_graph_metadata[
                "k1.new_token_nums"
            ][:batch_size]
            metadata.k1.new_compress_token_nums = self.decode_cuda_graph_metadata[
                "k1.new_compress_token_nums"
            ][:batch_size]
            metadata.k1.total_compress_token_nums = self.decode_cuda_graph_metadata[
                "k1.total_compress_token_nums"
            ][:batch_size]
            metadata.k1.cu_new_token_nums = self.decode_cuda_graph_metadata[
                "k1.cu_new_token_nums"
            ][: batch_size + 1]
            metadata.k1.cu_new_compress_token_nums = self.decode_cuda_graph_metadata[
                "k1.cu_new_compress_token_nums"
            ][: batch_size + 1]
            metadata.k1.cu_total_compress_token_nums = self.decode_cuda_graph_metadata[
                "k1.cu_total_compress_token_nums"
            ][: batch_size + 1]
            # compress k2
            metadata.k2.history_compress_token_nums = self.decode_cuda_graph_metadata[
                "k2.history_compress_token_nums"
            ][:batch_size]
            metadata.k2.new_token_nums = self.decode_cuda_graph_metadata[
                "k2.new_token_nums"
            ][:batch_size]
            metadata.k2.new_compress_token_nums = self.decode_cuda_graph_metadata[
                "k2.new_compress_token_nums"
            ][:batch_size]
            metadata.k2.total_compress_token_nums = self.decode_cuda_graph_metadata[
                "k2.total_compress_token_nums"
            ][:batch_size]
            metadata.k2.cu_new_token_nums = self.decode_cuda_graph_metadata[
                "k2.cu_new_token_nums"
            ][: batch_size + 1]
            metadata.k2.cu_new_compress_token_nums = self.decode_cuda_graph_metadata[
                "k2.cu_new_compress_token_nums"
            ][: batch_size + 1]
            metadata.k2.cu_total_compress_token_nums = self.decode_cuda_graph_metadata[
                "k2.cu_total_compress_token_nums"
            ][: batch_size + 1]

            metadata.k1.table = self.decode_cuda_graph_metadata["k1.table"][:bs, :]
            metadata.k2.table = self.decode_cuda_graph_metadata["k2.table"][:bs, :]

            # Stage1 optimization metadata
            metadata.cu_seqlens_q_adjusted = self.decode_cuda_graph_metadata[
                "cu_seqlens_q_adjusted"
            ][: batch_size + 1]
            metadata.cache_seqlens_int32_stage1 = self.decode_cuda_graph_metadata[
                "cache_seqlens_int32_stage1"
            ][:batch_size]
            # For decode mode, adjusted max_seqlen_q is fixed to heads_per_group
            metadata.max_seqlen_q_adjusted = (
                metadata.max_seq_len_q * self.heads_per_group
            )

            # For flashinfer with CUDA graph:
            # During capture, we need to set up the wrapper with slice views.
            # The actual data will be updated during replay (outside graph).
            if self.attention_kernel_type == "flashinfer":
                sparse_bs = bs * 2

                # Get batch-sized SLICE VIEWS of pre-allocated buffers
                kv_indptr_view = self.decode_cuda_graph_metadata[
                    "flashinfer_kv_indptr"
                ][: sparse_bs + 1]
                kv_indices_view = self.decode_cuda_graph_metadata[
                    "flashinfer_kv_indices"
                ][: sparse_bs * self.num_sparse_topk_tokens]
                kv_last_page_len_view = self.decode_cuda_graph_metadata[
                    "flashinfer_kv_last_page_len"
                ][:sparse_bs]

                # Create FlashInfer wrapper with SLICE VIEWS
                # FlashInfer captures the ADDRESSES of these slice views
                # NOTE: use_tensor_cores=True is required for num_kv_heads=1 to avoid plan_info=None bug
                flashinfer_wrapper = BatchDecodeWithPagedKVCacheWrapper(
                    self.attention_kernel.decode_workspace,
                    "NHD",
                    use_cuda_graph=True,
                    use_tensor_cores=True,
                    paged_kv_indptr_buffer=kv_indptr_view,  # Slice address captured
                    paged_kv_indices_buffer=kv_indices_view,  # Slice address captured
                    paged_kv_last_page_len_buffer=kv_last_page_len_view,  # Slice address
                )

                # Call begin_forward to initialize wrapper during capture
                flashinfer_wrapper.begin_forward(
                    kv_indptr_view,
                    kv_indices_view,
                    kv_last_page_len_view,
                    self.attention_kernel.num_qo_heads
                    // 2,  # Query heads (for each head group)
                    self.attention_kernel.num_kv_heads
                    // 2,  # KV heads (for each head group)
                    self.head_dim,
                    self.page_size,
                    q_data_type=self.attention_kernel.q_data_type,
                    kv_data_type=self.attention_kernel.data_type,
                    non_blocking=True,
                )

                # Store wrapper for replay and forward pass
                metadata.decode_wrapper = flashinfer_wrapper
                metadata.flashinfer_kv_indptr = kv_indptr_view
                metadata.flashinfer_kv_indices = kv_indices_view
                metadata.flashinfer_kv_last_page_len = kv_last_page_len_view

            self.decode_cuda_graph_metadata[bs] = metadata

            if encoder_lens is not None:
                encoder_bs = encoder_lens.numel()
                metadata.encoder_lens_int32 = self.encoder_metadata[
                    "encoder_lens_int32"
                ][:encoder_bs]
                metadata.encoder_cu_seqlens_k = self.encoder_metadata[
                    "encoder_cu_seqlens_k"
                ][: (encoder_bs + 1)]

                metadata.encoder_page_table = self.encoder_metadata[
                    "encoder_page_table"
                ][:bs, :]

        else:
            raise NotImplementedError(
                "MiniCPM backend CUDA graph capture only supports decode/idle mode, "
                f"got {forward_mode}"
            )

        self.forward_metadata = metadata

    def _replay_cuda_graph_metadata(
        self,
        bs: int,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        seq_lens_sum: int,
        encoder_lens: Optional[torch.Tensor],
        forward_mode: ForwardMode,
        spec_info: Optional[SpecInput],
        seq_lens_cpu: Optional[torch.Tensor],
        out_cache_loc: Optional[torch.Tensor] = None,
        forward_batch: Optional[ForwardBatch] = None,
    ):
        """Initialize forward metadata for replaying CUDA graph.

        For FlashInfer sparse mode, implements the update phase of wrapper state persistence:

        Update Process (executed OUTSIDE CUDA graph):
        1. Retrieve wrapper stored during capture
        2. Update underlying storage: flatten sparse_page_table into kv_indices
        3. Call begin_forward() to sync wrapper's cached pointers/metadata
        4. Synchronize GPU to ensure updates complete before graph replay

        The wrapper holds addresses to pre-allocated buffers. By updating the underlying
        tensor data at those addresses, the wrapper sees the fresh data without reallocation.
        """
        seq_lens = seq_lens[:bs]
        seq_lens_cpu = seq_lens_cpu[:bs] if seq_lens_cpu is not None else seq_lens
        req_pool_indices = req_pool_indices[:bs]
        metadata = None

        if forward_mode.is_decode_or_idle():
            # Normal Decode
            metadata = self.decode_cuda_graph_metadata[bs]
            max_len = seq_lens_cpu.max().item()
            max_seq_pages = (max_len + self.page_size - 1) // self.page_size
            metadata.max_seq_len_k = max_len

            normal_decode_set_metadata(
                metadata.cache_seqlens_int32,
                metadata.cu_seqlens_k,
                metadata.page_table,
                self.req_to_token,
                req_pool_indices,
                self.decode_cuda_graph_metadata["strided_indices"],
                max_seq_pages,
                seq_lens,
                0,
                self.page_size,
                metadata.swa_page_table,
                None,
            )

            real_bs = bs - forward_batch.num_padding
            sparse_forward_batch = SimpleNamespace(
                batch_size=real_bs,
                req_pool_indices=req_pool_indices[:real_bs],
                seq_lens_cpu=seq_lens_cpu[:real_bs],
            )
            decode_metadata, compression_metadata = (
                self._build_sparse_decode_replay_metadata(
                    sparse_forward_batch, metadata
                )
            )

            metadata.sparse_cache_seqlens_int32[: 2 * real_bs].copy_(
                decode_metadata["sparse_cache_seqlens_int32"]
            )
            metadata.sparse_cu_seqlens_k[: 2 * real_bs + 1].copy_(
                decode_metadata["sparse_cu_seqlens_k"]
            )

            # Stage1 optimization metadata update
            metadata.cache_seqlens_int32_stage1[:real_bs].copy_(
                metadata.cache_seqlens_int32[:real_bs] - 1
            )

            # Update flashinfer metadata for CUDA graph replay
            # For sparse mode, use the wrapper-based pattern that preserves sparse_page_table
            if self.attention_kernel_type == "flashinfer":
                sparse_bs = bs * 2
                sparse_real_bs = real_bs * 2

                # Get views of pre-allocated buffers
                # kv_indptr is precomputed and static: [0, 1*K, 2*K, ..., sparse_bs*K]
                kv_indptr_view = self.decode_cuda_graph_metadata[
                    "flashinfer_kv_indptr"
                ][: sparse_bs + 1]
                kv_indptr_view[0] = 0
                if sparse_real_bs > 0:
                    actual_seqlens = metadata.sparse_cache_seqlens_int32[
                        :sparse_real_bs
                    ].clone()
                    actual_seqlens = torch.clamp(
                        actual_seqlens, max=self.num_sparse_topk_tokens
                    )
                    kv_indptr_view[1 : sparse_real_bs + 1] = torch.cumsum(
                        actual_seqlens, dim=0
                    )
                kv_indptr_view[sparse_real_bs:].fill_(kv_indptr_view[sparse_real_bs])

                # kv_indices only needs num_sparse_topk_tokens per batch
                kv_indices_view = self.decode_cuda_graph_metadata[
                    "flashinfer_kv_indices"
                ][: sparse_bs * self.num_sparse_topk_tokens]
                kv_last_page_len_view = self.decode_cuda_graph_metadata[
                    "flashinfer_kv_last_page_len"
                ][:sparse_bs]
                kv_last_page_len_view[sparse_real_bs:].fill_(0)

                # Retrieve the wrapper stored during capture
                wrapper = metadata.decode_wrapper

                # Update wrapper's cached metadata using begin_forward
                # This updates the wrapper internal state stored during capture
                wrapper.begin_forward(
                    kv_indptr_view,
                    kv_indices_view,
                    kv_last_page_len_view,
                    self.attention_kernel.num_qo_heads
                    // 2,  # Query heads (for each head group)
                    self.attention_kernel.num_kv_heads
                    // 2,  # KV heads (for each head group)
                    self.head_dim,
                    self.page_size,
                    q_data_type=self.attention_kernel.q_data_type,
                    kv_data_type=self.attention_kernel.data_type,
                    non_blocking=True,
                )

                # Synchronize to ensure GPU operations complete before graph replay
                torch.cuda.synchronize()

                # Store the views for reference (not used in forward, wrapper provides access)
                metadata.flashinfer_kv_indptr = kv_indptr_view
                metadata.flashinfer_kv_indices = kv_indices_view
                metadata.flashinfer_kv_last_page_len = kv_last_page_len_view

            self.decode_cuda_graph_metadata["compress_k1"][
                : real_bs * self.max_context_len // self.k1_kernel_stride,
                :,
                :,
            ].fill_(float("-inf"))
            self.decode_cuda_graph_metadata["compress_k2"][
                : real_bs * self.max_context_len // self.k2_kernel_stride,
                :,
                :,
            ].fill_(float("-inf"))
            metadata.k1.cu_seqlens[: real_bs + 1].copy_(
                compression_metadata["k1"].cu_seqlens
            )
            metadata.k2.cu_seqlens[: real_bs + 1].copy_(
                compression_metadata["k2"].cu_seqlens
            )

            metadata.k1.history_compress_token_nums[:real_bs].copy_(
                compression_metadata["k1"].history_compress_token_nums
            )
            metadata.k2.history_compress_token_nums[:real_bs].copy_(
                compression_metadata["k2"].history_compress_token_nums
            )

            metadata.k1.new_token_nums[:real_bs].copy_(
                compression_metadata["k1"].new_token_nums
            )
            metadata.k2.new_token_nums[:real_bs].copy_(
                compression_metadata["k2"].new_token_nums
            )
            metadata.k1.cu_new_token_nums[: real_bs + 1].copy_(
                compression_metadata["k1"].cu_new_token_nums
            )
            metadata.k2.cu_new_token_nums[: real_bs + 1].copy_(
                compression_metadata["k2"].cu_new_token_nums
            )

            metadata.k1.new_compress_token_nums[:real_bs].copy_(
                compression_metadata["k1"].new_compress_token_nums
            )
            metadata.k2.new_compress_token_nums[:real_bs].copy_(
                compression_metadata["k2"].new_compress_token_nums
            )
            metadata.k1.cu_new_compress_token_nums[: real_bs + 1].copy_(
                compression_metadata["k1"].cu_new_compress_token_nums
            )
            metadata.k2.cu_new_compress_token_nums[: real_bs + 1].copy_(
                compression_metadata["k2"].cu_new_compress_token_nums
            )

            metadata.k1.total_compress_token_nums[:real_bs].copy_(
                compression_metadata["k1"].total_compress_token_nums
            )
            metadata.k2.total_compress_token_nums[:real_bs].copy_(
                compression_metadata["k2"].total_compress_token_nums
            )
            metadata.k1.cu_total_compress_token_nums[: real_bs + 1].copy_(
                compression_metadata["k1"].cu_total_compress_token_nums
            )
            metadata.k2.cu_total_compress_token_nums[: real_bs + 1].copy_(
                compression_metadata["k2"].cu_total_compress_token_nums
            )

            if real_bs < bs:
                metadata.sparse_cache_seqlens_int32[2 * real_bs :].fill_(0)
                metadata.sparse_cu_seqlens_k[2 * real_bs + 1 :].fill_(
                    decode_metadata["sparse_cu_seqlens_k"][-1]
                )
                metadata.cache_seqlens_int32_stage1[real_bs:].fill_(0)
                metadata.k1.cu_seqlens[real_bs + 1 :].fill_(
                    compression_metadata["k1"].cu_seqlens[-1]
                )
                metadata.k2.cu_seqlens[real_bs + 1 :].fill_(
                    compression_metadata["k2"].cu_seqlens[-1]
                )
                metadata.k1.history_compress_token_nums[real_bs:].fill_(0)
                metadata.k2.history_compress_token_nums[real_bs:].fill_(0)
                metadata.k1.new_token_nums[real_bs:].fill_(0)
                metadata.k2.new_token_nums[real_bs:].fill_(0)
                metadata.k1.cu_new_token_nums[real_bs + 1 :].fill_(
                    compression_metadata["k1"].cu_new_token_nums[-1]
                )
                metadata.k2.cu_new_token_nums[real_bs + 1 :].fill_(
                    compression_metadata["k2"].cu_new_token_nums[-1]
                )
                metadata.k1.new_compress_token_nums[real_bs:].fill_(0)
                metadata.k2.new_compress_token_nums[real_bs:].fill_(0)

                metadata.k1.cu_new_compress_token_nums[real_bs + 1 :].fill_(
                    compression_metadata["k1"].cu_new_compress_token_nums[-1]
                )
                metadata.k2.cu_new_compress_token_nums[real_bs + 1 :].fill_(
                    compression_metadata["k2"].cu_new_compress_token_nums[-1]
                )
                metadata.k1.total_compress_token_nums[real_bs:].fill_(0)
                metadata.k2.total_compress_token_nums[real_bs:].fill_(0)

                metadata.k1.cu_total_compress_token_nums[real_bs + 1 :].fill_(
                    compression_metadata["k1"].cu_total_compress_token_nums[-1]
                )

                metadata.k2.cu_total_compress_token_nums[real_bs + 1 :].fill_(
                    compression_metadata["k2"].cu_total_compress_token_nums[-1]
                )

            metadata.k1.table.copy_(self.req_to_sparse_k1_token[req_pool_indices])
            metadata.k2.table.copy_(self.req_to_sparse_k2_token[req_pool_indices])
        else:
            raise NotImplementedError(
                "MiniCPM backend CUDA graph replay only supports decode/idle mode, "
                f"got {forward_mode}"
            )
            return

        if encoder_lens is not None:
            # Only support encoder size 1 for now
            metadata.encoder_max_seq_len_k = encoder_lens[0]
            metadata.encoder_lens_int32.copy_(encoder_lens[:1])
            metadata.encoder_cu_seqlens_k[1:].copy_(
                torch.cumsum(metadata.encoder_lens_int32, dim=0, dtype=torch.int32)
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

    def get_cuda_graph_seq_len_fill_value(self):
        """Get the fill value for sequence length in CUDA graph."""
        return 1
