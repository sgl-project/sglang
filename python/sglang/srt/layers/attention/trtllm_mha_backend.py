from __future__ import annotations

"""
Support attention backend for TRTLLM MHA kernels from flashinfer.
The kernel supports sm100 only, with sliding window and attention sink features.
"""

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

import torch

from sglang.srt.environ import envs
from sglang.srt.layers.attention.flashinfer_backend import (
    FlashInferAttnBackend,
    FlashInferMultiStepDraftBackend,
)
from sglang.srt.layers.attention.triton_ops.trtllm_fp8_kv_kernel import (
    fused_fp8_set_kv_buffer,
)
from sglang.srt.layers.attention.utils import (
    canonicalize_stride,
    create_flashinfer_kv_indices_triton,
)
from sglang.srt.mem_cache.swa_memory_pool import SWAKVPool, SWATokenToKVPoolAllocator
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode
from sglang.srt.utils import is_flashinfer_available
from sglang.srt.utils.common import is_sm90_supported, is_sm120_supported

logger = logging.getLogger(__name__)

if is_flashinfer_available():
    import flashinfer

if TYPE_CHECKING:
    from sglang.srt.layers.radix_attention import RadixAttention
    from sglang.srt.model_executor.model_runner import ModelRunner
    from sglang.srt.speculative.spec_info import SpecInput

# Constants
# Default workspace size in MB for TRTLLM MHA
# Can be configured via SGLANG_FLASHINFER_WORKSPACE_SIZE environment variable
DEFAULT_WORKSPACE_SIZE_MB = 512

# Reuse this workspace buffer across all TRTLLM MHA wrappers
global_zero_init_workspace_buffer = None


@dataclass
class TRTLLMMHAMetadata:
    # Sequence lengths for the forward batch
    cache_seqlens_int32: torch.Tensor = None
    # Maximum sequence length for query
    max_seq_len_q: int = 1
    # Maximum sequence length for key
    max_seq_len_k: int = 0
    # Cumulative sequence lengths for `query
    cu_seqlens_q: torch.Tensor = None
    # Cumulative sequence lengths for key
    cu_seqlens_k: torch.Tensor = None
    # Page table, the index of KV Cache Tables/Blocks
    page_table: torch.Tensor = None
    # Page table for SWA layers (translated from full pool indices to SWA pool indices)
    swa_page_table: torch.Tensor = None

    # The following fields are used for Flashinfer KV update
    # kv_indices: page indices for all requests (full-attention)
    kv_indices: torch.Tensor = None
    # kv_indices for SWA layers
    swa_kv_indices: torch.Tensor = None
    # kv_indptr: cumulative page count per request
    kv_indptr: torch.Tensor = None
    # batch_indices: which request each token belongs to
    batch_indices: torch.Tensor = None
    # positions: position of each token within its sequence
    positions: torch.Tensor = None


class TRTLLMHAAttnBackend(FlashInferAttnBackend):
    """TRTLLM MHA attention kernel from flashinfer."""

    def __init__(
        self,
        model_runner: ModelRunner,
        skip_prefill: bool = False,
        kv_indptr_buf: Optional[torch.Tensor] = None,
        kv_last_page_len_buf: Optional[torch.Tensor] = None,
        speculative_step_id: int = 0,
    ):
        # Capture workspace size before super().__init__() to preserve user's
        # SGLANG_FLASHINFER_WORKSPACE_SIZE setting (may be overridden by parent)
        env_var = envs.SGLANG_FLASHINFER_WORKSPACE_SIZE
        workspace_size_bytes = (
            env_var.get()
            if env_var.is_set()
            else DEFAULT_WORKSPACE_SIZE_MB * 1024 * 1024
        )

        super().__init__(
            model_runner, skip_prefill, kv_indptr_buf, kv_last_page_len_buf
        )

        config = model_runner.model_config

        # MHA-specific dimensions
        self.max_context_len = model_runner.model_config.context_len
        self.hidden_size = config.hidden_size

        # Runtime parameters
        self.data_type = model_runner.kv_cache_dtype
        self.q_data_type = model_runner.dtype
        self.page_size = model_runner.page_size
        self.req_to_token = model_runner.req_to_token_pool.req_to_token
        self.device = model_runner.device

        # Workspace allocation
        self.workspace_size = workspace_size_bytes
        # Allocate buffers
        global global_zero_init_workspace_buffer
        if global_zero_init_workspace_buffer is None:
            global_zero_init_workspace_buffer = torch.zeros(
                self.workspace_size,
                dtype=torch.uint8,
                device=model_runner.device,
            )
        self.workspace_buffer = global_zero_init_workspace_buffer

        # CUDA graph state
        self.decode_cuda_graph_metadata = {}

        # Speculative decoding
        # Only support topk <= 1 for now.
        self.topk = model_runner.server_args.speculative_eagle_topk or 0
        self.speculative_step_id = speculative_step_id
        self.target_verify_metadata = {}

        self.speculative_num_draft_tokens = (
            model_runner.server_args.speculative_num_draft_tokens
        )

        # Sliding Window Attention(SWA) hybrid model support.
        # For hybrid SWA models, the KV cache is split into two pools (full and SWA)
        # with separate index spaces. We maintain a translated page_table for SWA
        # layers so the trtllm kernel reads from the correct pool.
        allocator = model_runner.token_to_kv_pool_allocator
        self.use_sliding_window_kv_pool = isinstance(
            allocator, SWATokenToKVPoolAllocator
        )
        self._swa_kv_pool: Optional[SWAKVPool] = (
            allocator.get_kvcache() if self.use_sliding_window_kv_pool else None
        )

        # Forward metadata
        self.forward_metadata: Optional[TRTLLMMHAMetadata] = None

        # Init backend (XQA or TRTLLM-GEN)
        # We need to specify q_type and out_type for different backend
        # XQA: (q_type must be bf16)
        #   KV bf16: q_type = bf16, out_type=model_runner.dtype
        #   KV fp8: q_type = bf16, out_type=model_runner.dtype
        # TRTLLM-GEN:
        #   KV bf16: q_type = bf16, out_type=model_runner.dtype
        #   KV fp8: q_type = fp8, out_type=model_runner.dtype
        self.is_xqa_impl = is_sm90_supported() or is_sm120_supported()

        self.enable_rope_fusion = envs.SGLANG_ENABLE_FLASHINFER_ROPE_FUSION.get()
        if self.enable_rope_fusion:
            logger.info("Flashinfer RoPE+Quant+cache update fusion is enabled")
            assert (
                self.data_type == torch.float8_e4m3fn
            ), "RoPE+Quant+cache update fusion is only supported for FP8 KV cache dtype"

    def _maybe_translate_swa(
        self, token_indices: torch.Tensor
    ) -> Optional[torch.Tensor]:
        """Translate full-pool token indices to SWA-pool indices, or return None."""
        if not self.use_sliding_window_kv_pool:
            return None
        shape = token_indices.shape
        return self._swa_kv_pool.translate_loc_from_full_to_swa(
            token_indices.reshape(-1)
        ).reshape(shape)

    def _alloc_swa_page_table(
        self, max_bs: int, max_num_pages: int
    ) -> Optional[torch.Tensor]:
        """Allocate a SWA page_table buffer, or return None for non-SWA models."""
        if not self.use_sliding_window_kv_pool:
            return None
        return torch.zeros(max_bs, max_num_pages, dtype=torch.int32, device=self.device)

    def _copy_swa_page_table(
        self,
        metadata: TRTLLMMHAMetadata,
        page_indices: torch.Tensor,
        num_pages: int,
    ):
        """Translate and copy SWA page indices into metadata. No-op for non-SWA."""
        if metadata.swa_page_table is None:
            return
        swa_indices = self._maybe_translate_swa(page_indices)
        metadata.swa_page_table[:, :num_pages].copy_(swa_indices // self.page_size)

    def _get_layer_cache_loc(
        self,
        layer: RadixAttention,
        cache_loc: torch.Tensor,
    ) -> torch.Tensor:
        """Return cache locations in the correct index space for the given layer."""
        if self.use_sliding_window_kv_pool:
            _, is_swa = self._swa_kv_pool.layers_mapping[layer.layer_id]
            if is_swa:
                return self._swa_kv_pool.translate_loc_from_full_to_swa(cache_loc)
        return cache_loc

    def _bind_swa_page_table(
        self, metadata: TRTLLMMHAMetadata, source: dict, key: str, bs: int
    ):
        """Bind a pre-allocated SWA page_table slice to metadata for CUDA graph."""
        buf = source.get(key)
        if buf is not None:
            metadata.swa_page_table = buf[:bs, :]

    def _get_layer_page_table(
        self, layer: RadixAttention, forward_batch: ForwardBatch
    ) -> torch.Tensor:
        """Return the correct page_table for the given layer (SWA or full)."""
        swa_pt = self.forward_metadata.swa_page_table
        if swa_pt is not None:
            _, is_swa = self._swa_kv_pool.layers_mapping[layer.layer_id]
            if is_swa:
                return swa_pt
        return self.forward_metadata.page_table

    def _get_layer_kv_indices(self, layer: RadixAttention) -> torch.Tensor:
        """Return the correct kv_indices for the RoPE fusion kernel (SWA or full)."""
        swa_kv_indices = self.forward_metadata.swa_kv_indices
        if swa_kv_indices is not None:
            _, is_swa = self._swa_kv_pool.layers_mapping[layer.layer_id]
            if is_swa:
                return swa_kv_indices
        return self.forward_metadata.kv_indices

    def init_cuda_graph_state(
        self,
        max_bs: int,
        max_num_tokens: int,
        kv_indices_buf: Optional[torch.Tensor] = None,
    ):
        """Initialize CUDA graph state for TRTLLM MHA."""
        max_num_pages = (self.max_context_len + self.page_size - 1) // self.page_size
        self.decode_cuda_graph_metadata = {
            "cache_seqlens": torch.zeros(max_bs, dtype=torch.int32, device=self.device),
            "page_table": torch.zeros(
                max_bs,
                max_num_pages,
                dtype=torch.int32,
                device=self.device,
            ),
            "swa_page_table": self._alloc_swa_page_table(max_bs, max_num_pages),
            "strided_indices": torch.arange(
                0, self.max_context_len, self.page_size, device=self.device
            ),
        }

        if self.enable_rope_fusion:
            self.decode_cuda_graph_metadata["kv_indices"] = torch.zeros(
                max_bs * max_num_pages, dtype=torch.int32, device=self.device
            )
            if self.use_sliding_window_kv_pool:
                self.decode_cuda_graph_metadata["swa_kv_indices"] = torch.zeros(
                    max_bs * max_num_pages, dtype=torch.int32, device=self.device
                )
        if (
            self.speculative_num_draft_tokens is not None
            and self.speculative_num_draft_tokens > 0
        ):
            self.decode_cuda_graph_metadata["cu_seqlens_q"] = torch.arange(
                0, max_bs + 1, dtype=torch.int32, device=self.device
            )
            self.decode_cuda_graph_metadata["cu_seqlens_k"] = torch.zeros(
                max_bs + 1, dtype=torch.int32, device=self.device
            )
            self.decode_cuda_graph_metadata["page_table_draft_decode"] = torch.zeros(
                max_bs,
                max_num_pages,
                dtype=torch.int32,
                device=self.device,
            )
            self.decode_cuda_graph_metadata["swa_page_table_draft_decode"] = (
                self._alloc_swa_page_table(max_bs, max_num_pages)
            )

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
                    max_num_pages,
                    dtype=torch.int32,
                    device=self.device,
                ),
                "swa_page_table": self._alloc_swa_page_table(max_bs, max_num_pages),
                "strided_indices": torch.arange(
                    0, self.max_context_len, self.page_size, device=self.device
                ),
            }

            self.draft_extend_metadata = {
                "cache_seqlens": torch.zeros(
                    max_bs, dtype=torch.int32, device=self.device
                ),
                "cu_seqlens_q": torch.zeros(
                    max_bs + 1,
                    dtype=torch.int32,
                    device=self.device,
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
                "swa_page_table": self._alloc_swa_page_table(max_bs, max_num_pages),
                "strided_indices": torch.arange(
                    0, self.max_context_len, self.page_size, device=self.device
                ),
            }

            if self.enable_rope_fusion:
                self.target_verify_metadata["kv_indices"] = torch.zeros(
                    max_bs * max_num_pages, dtype=torch.int32, device=self.device
                )
                self.draft_extend_metadata["kv_indices"] = torch.zeros(
                    max_bs * max_num_pages, dtype=torch.int32, device=self.device
                )
                if self.use_sliding_window_kv_pool:
                    self.target_verify_metadata["swa_kv_indices"] = torch.zeros(
                        max_bs * max_num_pages, dtype=torch.int32, device=self.device
                    )
                    self.draft_extend_metadata["swa_kv_indices"] = torch.zeros(
                        max_bs * max_num_pages, dtype=torch.int32, device=self.device
                    )

    def init_forward_metadata_capture_cuda_graph(
        self,
        bs: int,
        num_tokens: int,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        encoder_lens: Optional[torch.Tensor],
        forward_mode: ForwardMode,
        spec_info: Optional[SpecInput],
    ):
        """Initialize metadata for CUDA graph capture."""
        metadata = TRTLLMMHAMetadata()
        device = seq_lens.device

        if forward_mode.is_decode_or_idle():
            if spec_info is not None:
                # Draft Decode
                # Here we only support topk = 1 for now.
                metadata.cache_seqlens_int32 = self.decode_cuda_graph_metadata[
                    "cache_seqlens"
                ][:bs]
                metadata.max_seq_len_k = seq_lens.max().item() + (
                    self.speculative_step_id + 1
                )
                metadata.cu_seqlens_q = self.decode_cuda_graph_metadata["cu_seqlens_q"][
                    : bs + 1
                ]
                metadata.cu_seqlens_k = torch.nn.functional.pad(
                    torch.cumsum(
                        metadata.cache_seqlens_int32, dim=0, dtype=torch.int32
                    ),
                    (1, 0),
                )
                metadata.page_table = self.decode_cuda_graph_metadata[
                    "page_table_draft_decode"
                ][:bs, :]
                self._bind_swa_page_table(
                    metadata,
                    self.decode_cuda_graph_metadata,
                    "swa_page_table_draft_decode",
                    bs,
                )
                self.decode_cuda_graph_metadata[bs] = metadata
            else:
                # Normal Decode
                # Get sequence information
                metadata.cache_seqlens_int32 = seq_lens[:bs].to(torch.int32)
                batch_size = len(seq_lens)
                metadata.cu_seqlens_k = torch.nn.functional.pad(
                    torch.cumsum(seq_lens, dim=0, dtype=torch.int32), (1, 0)
                )

                # Precompute maximum sequence length
                metadata.max_seq_len_k = seq_lens.max().item()
                # Precompute cumulative sequence lengths
                metadata.cu_seqlens_q = torch.arange(
                    0, batch_size + 1, dtype=torch.int32, device=device
                )
                # Precompute page table
                metadata.page_table = self.decode_cuda_graph_metadata["page_table"][
                    :bs, :
                ]
                self._bind_swa_page_table(
                    metadata,
                    self.decode_cuda_graph_metadata,
                    "swa_page_table",
                    bs,
                )
                self.decode_cuda_graph_metadata[bs] = metadata
        elif forward_mode.is_target_verify():
            # Target Verify
            # Here we only support topk = 1 for now.
            metadata.cache_seqlens_int32 = self.target_verify_metadata["cache_seqlens"][
                :bs
            ]
            metadata.cache_seqlens_int32.copy_(
                (seq_lens + self.speculative_num_draft_tokens)
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

            metadata.max_seq_len_q = self.speculative_num_draft_tokens
            metadata.max_seq_len_k = (
                seq_lens.max().item() + self.speculative_num_draft_tokens
            )

            metadata.page_table = self.target_verify_metadata["page_table"][:bs, :]
            self._bind_swa_page_table(
                metadata,
                self.target_verify_metadata,
                "swa_page_table",
                bs,
            )

            self.target_verify_metadata[bs] = metadata
        elif forward_mode.is_draft_extend():
            metadata.cache_seqlens_int32 = self.draft_extend_metadata["cache_seqlens"][
                :bs
            ]
            metadata.cache_seqlens_int32.copy_(seq_lens)
            num_tokens_per_bs = num_tokens // bs
            metadata.cu_seqlens_q = torch.arange(
                0,
                bs * num_tokens_per_bs + 1,
                num_tokens_per_bs,
                dtype=torch.int32,
                device=device,
            )

            metadata.cu_seqlens_k = self.draft_extend_metadata["cu_seqlens_k"][
                : (bs + 1)
            ]
            num_tokens_per_bs = num_tokens // bs
            metadata.max_seq_len_q = num_tokens_per_bs
            metadata.max_seq_len_k = seq_lens.max().item()

            metadata.page_table = self.draft_extend_metadata["page_table"][:bs, :]
            self._bind_swa_page_table(
                metadata,
                self.draft_extend_metadata,
                "swa_page_table",
                bs,
            )

            self.draft_extend_metadata[bs] = metadata

        if self.enable_rope_fusion:
            if forward_mode.is_decode_or_idle():
                metadata.kv_indices = self.decode_cuda_graph_metadata["kv_indices"]
                metadata.swa_kv_indices = self.decode_cuda_graph_metadata.get(
                    "swa_kv_indices"
                )
            elif forward_mode.is_target_verify():
                metadata.kv_indices = self.target_verify_metadata["kv_indices"]
                metadata.swa_kv_indices = self.target_verify_metadata.get(
                    "swa_kv_indices"
                )
            else:
                metadata.kv_indices = self.draft_extend_metadata["kv_indices"]
                metadata.swa_kv_indices = self.draft_extend_metadata.get(
                    "swa_kv_indices"
                )
            self._init_forward_metadata_for_rope_fusion(metadata, bs, num_tokens)

        self.forward_metadata = metadata

    def init_forward_metadata_replay_cuda_graph(
        self,
        bs: int,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        seq_lens_sum: int,
        encoder_lens: Optional[torch.Tensor],
        forward_mode: ForwardMode,
        spec_info: Optional[SpecInput],
        seq_lens_cpu: Optional[torch.Tensor],
    ):
        """Replay CUDA graph with new inputs."""
        seq_lens = seq_lens[:bs]
        seq_lens_cpu = seq_lens_cpu[:bs]
        req_pool_indices = req_pool_indices[:bs]
        metadata = None
        if forward_mode.is_decode_or_idle():
            if spec_info is not None:
                # Draft Decode
                # Here we only support topk = 1 for now.
                metadata = self.decode_cuda_graph_metadata[bs]
                max_len = seq_lens_cpu.max().item()
                metadata.max_seq_len_k = max_len + self.speculative_step_id + 1

                max_seq_pages = (
                    metadata.max_seq_len_k + self.page_size - 1
                ) // self.page_size

                metadata.cache_seqlens_int32.copy_(
                    seq_lens + self.speculative_step_id + 1
                )
            else:
                # Normal Decode
                metadata = self.decode_cuda_graph_metadata[bs]
                max_len = seq_lens_cpu.max().item()
                max_seq_pages = (max_len + self.page_size - 1) // self.page_size
                metadata.max_seq_len_k = max_len

                metadata.cache_seqlens_int32.copy_(seq_lens)

            metadata.cu_seqlens_k[1:].copy_(
                torch.cumsum(metadata.cache_seqlens_int32, dim=0, dtype=torch.int32)
            )
            page_indices = self.req_to_token[
                req_pool_indices[:, None],
                self.decode_cuda_graph_metadata["strided_indices"][:max_seq_pages][
                    None, :
                ],
            ]
            metadata.page_table[:, :max_seq_pages].copy_(page_indices // self.page_size)
            self._copy_swa_page_table(metadata, page_indices, max_seq_pages)
        elif forward_mode.is_target_verify():
            # Here we only support topk = 1 for now.
            metadata = self.target_verify_metadata[bs]
            metadata.cache_seqlens_int32.copy_(
                (seq_lens + self.speculative_num_draft_tokens)
            )

            metadata.max_seq_len_k = (
                seq_lens_cpu.max().item() + self.speculative_num_draft_tokens
            )
            max_len = seq_lens_cpu.max().item()
            metadata.cu_seqlens_k[1:].copy_(
                torch.cumsum(metadata.cache_seqlens_int32, dim=0, dtype=torch.int32)
            )
            max_seq_pages = (
                metadata.max_seq_len_k + self.page_size - 1
            ) // self.page_size
            page_indices = self.req_to_token[
                req_pool_indices[:, None],
                self.decode_cuda_graph_metadata["strided_indices"][:max_seq_pages],
            ]
            metadata.page_table[:, :max_seq_pages].copy_(page_indices // self.page_size)
            self._copy_swa_page_table(metadata, page_indices, max_seq_pages)
            metadata.max_seq_len_q = self.speculative_num_draft_tokens
        elif forward_mode.is_draft_extend():
            metadata = self.draft_extend_metadata[bs]
            metadata.cache_seqlens_int32.copy_(seq_lens)

            metadata.max_seq_len_k = seq_lens_cpu.max().item()
            max_len = seq_lens_cpu.max().item()
            metadata.cu_seqlens_k[1:].copy_(
                torch.cumsum(metadata.cache_seqlens_int32, dim=0, dtype=torch.int32)
            )
            accept_length = spec_info.accept_length[:bs]
            if spec_info.accept_length_cpu:
                metadata.max_seq_len_q = max(spec_info.accept_length_cpu) + 1
            else:
                metadata.max_seq_len_q = 1

            metadata.cu_seqlens_q[1:].copy_(
                torch.cumsum(accept_length, dim=0, dtype=torch.int32)
            )

            max_seq_pages = (
                metadata.max_seq_len_k + self.page_size - 1
            ) // self.page_size
            page_indices = self.req_to_token[
                req_pool_indices[:, None],
                self.draft_extend_metadata["strided_indices"][:max_seq_pages],
            ]
            metadata.page_table[:, :max_seq_pages].copy_(page_indices // self.page_size)
            self._copy_swa_page_table(metadata, page_indices, max_seq_pages)

        if self.enable_rope_fusion:
            # Compute total number of tokens
            if forward_mode.is_decode_or_idle():
                if spec_info is not None:
                    total_num_tokens = bs * spec_info.num_draft_tokens
                else:
                    total_num_tokens = bs
                metadata.kv_indices = self.decode_cuda_graph_metadata["kv_indices"]
                metadata.swa_kv_indices = self.decode_cuda_graph_metadata.get(
                    "swa_kv_indices"
                )
            elif forward_mode.is_target_verify():
                total_num_tokens = bs * self.speculative_num_draft_tokens
                metadata.kv_indices = self.target_verify_metadata["kv_indices"]
                metadata.swa_kv_indices = self.target_verify_metadata.get(
                    "swa_kv_indices"
                )
            else:  # draft_extend
                total_num_tokens = metadata.cu_seqlens_q[-1].item()
                metadata.kv_indices = self.draft_extend_metadata["kv_indices"]
                metadata.swa_kv_indices = self.draft_extend_metadata.get(
                    "swa_kv_indices"
                )

            self._init_forward_metadata_for_rope_fusion(metadata, bs, total_num_tokens)

        self.forward_metadata = metadata

    def get_cuda_graph_seq_len_fill_value(self) -> int:
        """Get the fill value for sequence lengths in CUDA graph."""
        return 1

    def init_forward_metadata(self, forward_batch: ForwardBatch):
        """Initialize the metadata for a forward pass."""

        metadata = TRTLLMMHAMetadata()
        seqlens_in_batch = forward_batch.seq_lens
        batch_size = forward_batch.batch_size
        device = seqlens_in_batch.device

        if forward_batch.forward_mode.is_decode_or_idle():
            if forward_batch.spec_info is not None:
                # Draft Decode
                # Here we only support topk = 1 for now.
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
        elif forward_batch.forward_mode.is_target_verify():
            # Only support topk = 1 for now.
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
                torch.cumsum(metadata.cache_seqlens_int32, dim=0, dtype=torch.int32),
                (1, 0),
            )
            metadata.page_table = forward_batch.req_to_token_pool.req_to_token[
                forward_batch.req_pool_indices, : metadata.max_seq_len_k
            ]
        else:
            metadata.cache_seqlens_int32 = seqlens_in_batch.to(torch.int32)
            metadata.max_seq_len_k = forward_batch.seq_lens_cpu.max().item()
            metadata.cu_seqlens_k = torch.nn.functional.pad(
                torch.cumsum(seqlens_in_batch, dim=0, dtype=torch.int32), (1, 0)
            )
            metadata.page_table = forward_batch.req_to_token_pool.req_to_token[
                forward_batch.req_pool_indices, : metadata.max_seq_len_k
            ]

            if any(
                forward_batch.extend_prefix_lens_cpu
            ) or forward_batch.forward_mode.is_draft_extend(include_v2=True):
                extend_seq_lens = forward_batch.extend_seq_lens
                # NOTE: in piecewise CUDA graph warmup, extend_seq_lens_cpu is a torch.Tensor;
                # Python's max() returns a 0-d tensor, but flashinfer expects an int.
                max_q = max(forward_batch.extend_seq_lens_cpu)
                metadata.max_seq_len_q = (
                    int(max_q.item()) if isinstance(max_q, torch.Tensor) else int(max_q)
                )
                metadata.cu_seqlens_q = torch.nn.functional.pad(
                    torch.cumsum(extend_seq_lens, dim=0, dtype=torch.int32), (1, 0)
                )
            else:
                metadata.max_seq_len_q = metadata.max_seq_len_k
                metadata.cu_seqlens_q = metadata.cu_seqlens_k

        # Compute SWA page table (None for non-SWA models)
        metadata.swa_page_table = self._maybe_translate_swa(metadata.page_table)

        # Convert the page tables to a strided format
        if self.page_size > 1:
            self.strided_indices = torch.arange(
                0, metadata.page_table.shape[1], self.page_size, device=self.device
            )
            metadata.page_table = (
                metadata.page_table[:, self.strided_indices] // self.page_size
            )
            if metadata.swa_page_table is not None:
                metadata.swa_page_table = (
                    metadata.swa_page_table[:, self.strided_indices] // self.page_size
                )

        if self.enable_rope_fusion:
            # Compute total number of tokens
            if forward_batch.forward_mode.is_decode_or_idle():
                if forward_batch.spec_info is not None:
                    total_num_tokens = (
                        batch_size * forward_batch.spec_info.num_draft_tokens
                    )
                else:
                    total_num_tokens = batch_size
            elif forward_batch.forward_mode.is_target_verify():
                total_num_tokens = batch_size * self.speculative_num_draft_tokens
            else:
                total_num_tokens = forward_batch.extend_num_tokens

            self._init_forward_metadata_for_rope_fusion(
                metadata, batch_size, total_num_tokens
            )

        self.forward_metadata = metadata

    def _init_forward_metadata_for_rope_fusion(
        self,
        metadata: TRTLLMMHAMetadata,
        batch_size: int,
        total_num_tokens: int,
    ):
        """
        Initialize the following metadata for RoPE + FP8 quantization + KV cache update kernel.
        - kv_indptr: cumulative page count per request
        - batch_indices: which request each token belongs to
        - positions: position of each token within its sequence
        - kv_indices: page indices for all requests (full pool)
        - swa_kv_indices: page indices for SWA layers
        """

        # Compute number of pages per request
        num_pages_per_req = (
            metadata.cache_seqlens_int32 + self.page_size - 1
        ) // self.page_size

        # kv_indptr
        if metadata.kv_indptr is None:
            metadata.kv_indptr = torch.nn.functional.pad(
                torch.cumsum(num_pages_per_req, dim=0, dtype=torch.int32), (1, 0)
            )
        else:
            metadata.kv_indptr[1:].copy_(
                torch.cumsum(num_pages_per_req, dim=0, dtype=torch.int32)
            )

        # batch_indices and positions
        metadata.batch_indices, metadata.positions = (
            flashinfer.page.get_batch_indices_positions(
                metadata.cu_seqlens_q,
                metadata.cache_seqlens_int32,
                total_num_tokens,
                metadata.batch_indices,
                metadata.positions,
            )
        )

        # kv_indices (full-attention)
        device = metadata.kv_indptr.device
        batch_idx = torch.arange(batch_size, dtype=torch.int32, device=device)
        if metadata.kv_indices is None:
            # The max number of pages is kv_indptr[-1]
            # Use upper bound to avoid D2H transfer from accessing kv_indptr[-1]
            max_pages_per_req = (
                metadata.max_seq_len_k + self.page_size - 1
            ) // self.page_size
            metadata.kv_indices = torch.zeros(
                batch_size * max_pages_per_req, dtype=torch.int32, device=device
            )
        create_flashinfer_kv_indices_triton[(batch_size,)](
            metadata.page_table,
            batch_idx,
            num_pages_per_req,
            metadata.kv_indptr,
            None,
            metadata.kv_indices,
            metadata.page_table.stride(0),
        )

        # swa_kv_indices (SWA)
        if metadata.swa_page_table is not None:
            if metadata.swa_kv_indices is None:
                max_pages_per_req = (
                    metadata.max_seq_len_k + self.page_size - 1
                ) // self.page_size
                metadata.swa_kv_indices = torch.zeros(
                    batch_size * max_pages_per_req, dtype=torch.int32, device=device
                )
            create_flashinfer_kv_indices_triton[(batch_size,)](
                metadata.swa_page_table,
                batch_idx,
                num_pages_per_req,
                metadata.kv_indptr,
                None,
                metadata.swa_kv_indices,
                metadata.swa_page_table.stride(0),
            )

    def forward_decode(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        save_kv_cache: bool = True,
        **kwargs,
    ) -> torch.Tensor:
        """Run forward for decode using TRTLLM MHA kernel."""
        k_cache, v_cache = forward_batch.token_to_kv_pool.get_kv_buffer(layer.layer_id)

        q_scale_float = 1.0
        k_scale_float = (
            layer.k_scale_float
            if getattr(layer, "k_scale_float", None) is not None
            else 1.0
        )
        v_scale_float = (
            layer.v_scale_float
            if getattr(layer, "v_scale_float", None) is not None
            else 1.0
        )

        if save_kv_cache and k is not None:
            if (
                self.enable_rope_fusion
                and kwargs.get("cos_sin_cache") is not None
                and kwargs.get("is_neox_style") is not None
                and (not self.is_xqa_impl)
            ):
                # Use fused RoPE + FP8 quantization + KV cache update path
                cos_sin_cache = kwargs.get("cos_sin_cache")
                is_neox_style = kwargs.get("is_neox_style")

                q = q.view(-1, layer.tp_q_head_num, layer.head_dim)
                k_cache = k_cache.view(
                    -1, self.page_size, layer.tp_k_head_num, layer.head_dim
                )
                v_cache = v_cache.view(
                    -1, self.page_size, layer.tp_v_head_num, layer.head_dim
                )

                q = flashinfer.rope.rope_quantize_fp8_append_paged_kv_cache(
                    q_rope=q,
                    k_rope=k,
                    q_nope=None,
                    k_nope=None,
                    v=v,
                    cos_sin_cache=cos_sin_cache,
                    pos_ids=forward_batch.positions,
                    paged_kv_cache=(k_cache, v_cache),
                    kv_indices=self._get_layer_kv_indices(layer),
                    kv_indptr=self.forward_metadata.kv_indptr,
                    batch_indices=self.forward_metadata.batch_indices,
                    positions=self.forward_metadata.positions,
                    is_neox=is_neox_style,
                    quantize_dtype=torch.float8_e4m3fn,
                    quant_scale_q=q_scale_float,
                    quant_scale_kv=k_scale_float,
                    page_size=self.page_size,
                    kv_layout="NHD",
                )[0]
            elif self.data_type == torch.float8_e4m3fn:
                # Use fused KV FP8 quantization + KV cache update path
                cache_loc = self._get_layer_cache_loc(
                    layer, forward_batch.out_cache_loc
                )
                fused_fp8_set_kv_buffer(
                    k=k,
                    v=v,
                    k_cache=k_cache,
                    v_cache=v_cache,
                    cache_loc=cache_loc,
                    k_scale=layer.k_scale,
                    v_scale=layer.v_scale,
                    page_size=self.page_size,
                )
            else:
                # Use original set_kv_buffer path
                forward_batch.token_to_kv_pool.set_kv_buffer(
                    layer,
                    forward_batch.out_cache_loc,
                    k,
                    v,
                    layer.k_scale,
                    layer.v_scale,
                )

        # For XQA, q_dtype should be bf16
        if self.data_type == torch.float8_e4m3fn and (not self.is_xqa_impl):
            q = q.to(torch.float8_e4m3fn)
        q = q.contiguous().view(-1, layer.tp_q_head_num, layer.head_dim)

        k_cache = k_cache.view(-1, self.page_size, layer.tp_k_head_num, layer.head_dim)
        v_cache = v_cache.view(-1, self.page_size, layer.tp_v_head_num, layer.head_dim)
        if layer.tp_k_head_num == 1:
            k_cache = canonicalize_stride(k_cache)
        if layer.tp_v_head_num == 1:
            v_cache = canonicalize_stride(v_cache)
        kv_cache = (k_cache, v_cache)

        bmm1_scale = q_scale_float * k_scale_float * layer.scaling
        bmm2_scale = v_scale_float
        # sink: additional value per head in the denominator of the softmax.
        attention_sink = kwargs.get("sinks", None)

        page_table = self._get_layer_page_table(layer, forward_batch)

        # Call TRT-LLM kernel
        # raw_out: like q, [bs, acc_q_len, num_q_heads, head_dim] but with output dtype
        o = flashinfer.decode.trtllm_batch_decode_with_kv_cache(
            query=q,
            kv_cache=kv_cache,
            workspace_buffer=self.workspace_buffer,
            block_tables=page_table,
            seq_lens=self.forward_metadata.cache_seqlens_int32,
            max_seq_len=self.max_context_len,
            bmm1_scale=bmm1_scale,
            bmm2_scale=bmm2_scale,
            window_left=layer.sliding_window_size,
            sinks=attention_sink,
            skip_softmax_threshold_scale_factor=envs.SGLANG_SKIP_SOFTMAX_DECODE_THRESHOLD_SCALE_FACTOR.get(),
            out_dtype=self.q_data_type,  # model_runner.dtype
            kv_layout="NHD",
        )

        return o.view(-1, layer.tp_q_head_num * layer.head_dim)

    def forward_extend(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        save_kv_cache: bool = True,
        **kwargs,
    ) -> torch.Tensor:
        """Run forward for extend using TRTLLM MHA kernel."""
        k_cache, v_cache = forward_batch.token_to_kv_pool.get_kv_buffer(layer.layer_id)

        q_scale_float = 1.0
        k_scale_float = (
            layer.k_scale_float
            if getattr(layer, "k_scale_float", None) is not None
            else 1.0
        )
        v_scale_float = (
            layer.v_scale_float
            if getattr(layer, "v_scale_float", None) is not None
            else 1.0
        )

        if save_kv_cache and k is not None:
            if (
                self.enable_rope_fusion
                and kwargs.get("cos_sin_cache") is not None
                and kwargs.get("is_neox_style") is not None
            ):
                # Use fused RoPE + FP8 quantization + KV cache update path
                cos_sin_cache = kwargs.get("cos_sin_cache")
                is_neox_style = kwargs.get("is_neox_style")

                q = q.view(-1, layer.tp_q_head_num, layer.head_dim)
                k_cache = k_cache.view(
                    -1, self.page_size, layer.tp_k_head_num, layer.head_dim
                )
                v_cache = v_cache.view(
                    -1, self.page_size, layer.tp_v_head_num, layer.head_dim
                )

                q = flashinfer.rope.rope_quantize_fp8_append_paged_kv_cache(
                    q_rope=q,
                    k_rope=k,
                    q_nope=None,
                    k_nope=None,
                    v=v,
                    cos_sin_cache=cos_sin_cache,
                    pos_ids=forward_batch.positions,
                    paged_kv_cache=(k_cache, v_cache),
                    kv_indices=self._get_layer_kv_indices(layer),
                    kv_indptr=self.forward_metadata.kv_indptr,
                    batch_indices=self.forward_metadata.batch_indices,
                    positions=self.forward_metadata.positions,
                    is_neox=is_neox_style,
                    quantize_dtype=torch.float8_e4m3fn,
                    quant_scale_q=q_scale_float,
                    quant_scale_kv=k_scale_float,
                    page_size=self.page_size,
                    kv_layout="NHD",
                )[0]
            elif self.data_type == torch.float8_e4m3fn:
                # Use fused KV FP8 quantization + KV cache update path
                cache_loc = self._get_layer_cache_loc(
                    layer, forward_batch.out_cache_loc
                )
                fused_fp8_set_kv_buffer(
                    k=k,
                    v=v,
                    k_cache=k_cache,
                    v_cache=v_cache,
                    cache_loc=cache_loc,
                    k_scale=layer.k_scale,
                    v_scale=layer.v_scale,
                    page_size=self.page_size,
                )
            else:
                # Use original set_kv_buffer path
                forward_batch.token_to_kv_pool.set_kv_buffer(
                    layer,
                    forward_batch.out_cache_loc,
                    k,
                    v,
                    layer.k_scale,
                    layer.v_scale,
                )

        if self.data_type == torch.float8_e4m3fn:
            q = q.to(torch.float8_e4m3fn)
        q = q.contiguous().view(-1, layer.tp_q_head_num, layer.head_dim)

        k_cache = k_cache.view(-1, self.page_size, layer.tp_k_head_num, layer.head_dim)
        v_cache = v_cache.view(-1, self.page_size, layer.tp_v_head_num, layer.head_dim)
        if layer.tp_k_head_num == 1:
            k_cache = canonicalize_stride(k_cache)
        if layer.tp_v_head_num == 1:
            v_cache = canonicalize_stride(v_cache)
        kv_cache = (k_cache, v_cache)

        bmm1_scale = q_scale_float * k_scale_float * layer.scaling
        bmm2_scale = v_scale_float
        # sink: additional value per head in the denominator of the softmax.
        attention_sink = kwargs.get("sinks", None)

        page_table = self._get_layer_page_table(layer, forward_batch)

        if forward_batch.forward_mode.is_target_verify():
            o = flashinfer.decode.trtllm_batch_decode_with_kv_cache(
                query=q,
                kv_cache=kv_cache,
                workspace_buffer=self.workspace_buffer,
                block_tables=page_table,
                seq_lens=self.forward_metadata.cache_seqlens_int32,
                max_seq_len=self.max_context_len,
                bmm1_scale=bmm1_scale,
                bmm2_scale=bmm2_scale,
                window_left=layer.sliding_window_size,
                sinks=attention_sink,
                skip_softmax_threshold_scale_factor=envs.SGLANG_SKIP_SOFTMAX_DECODE_THRESHOLD_SCALE_FACTOR.get(),
                out_dtype=self.q_data_type,  # model_runner.dtype
                q_len_per_req=self.forward_metadata.max_seq_len_q,
                kv_layout="NHD",
            )
        else:
            o = flashinfer.prefill.trtllm_batch_context_with_kv_cache(
                query=q,
                kv_cache=kv_cache,
                workspace_buffer=self.workspace_buffer,
                block_tables=page_table,
                seq_lens=self.forward_metadata.cache_seqlens_int32,
                max_q_len=self.forward_metadata.max_seq_len_q,
                max_kv_len=self.max_context_len,
                bmm1_scale=bmm1_scale,
                bmm2_scale=bmm2_scale,
                batch_size=self.forward_metadata.cu_seqlens_q.shape[0] - 1,
                cum_seq_lens_q=self.forward_metadata.cu_seqlens_q,
                cum_seq_lens_kv=self.forward_metadata.cu_seqlens_k,
                window_left=layer.sliding_window_size,
                sinks=attention_sink,
                skip_softmax_threshold_scale_factor=envs.SGLANG_SKIP_SOFTMAX_PREFILL_THRESHOLD_SCALE_FACTOR.get(),
                out_dtype=self.q_data_type,  # model_runner.dtype
                kv_layout="NHD",
            )

        return o.view(-1, layer.tp_q_head_num * layer.head_dim)


class TRTLLMHAAttnMultiStepDraftBackend(FlashInferMultiStepDraftBackend):
    """Multi-step TRTLLM MHA attention kernel used by EAGLE."""

    def __init__(
        self, model_runner: ModelRunner, topk: int, speculative_num_steps: int
    ):
        super().__init__(model_runner, topk, speculative_num_steps)
        for i in range(self.speculative_num_steps - 1):
            self.attn_backends[i] = TRTLLMHAAttnBackend(
                model_runner,
                skip_prefill=True,
                kv_indptr_buf=self.kv_indptr[i],
                kv_last_page_len_buf=self.kv_last_page_len,
                speculative_step_id=i,
            )

    def init_forward_metadata(self, forward_batch: ForwardBatch):
        for i in range(self.speculative_num_steps - 1):
            self.attn_backends[i].init_forward_metadata(forward_batch)

    def init_cuda_graph_state(self, max_bs: int, max_num_tokens: int):
        for i in range(self.speculative_num_steps - 1):
            self.attn_backends[i].init_cuda_graph_state(max_bs, max_num_tokens)

    def init_forward_metadata_capture_cuda_graph(
        self,
        forward_batch: ForwardBatch,
    ):
        assert forward_batch.spec_info is not None
        assert forward_batch.spec_info.is_draft_input()

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
        assert forward_batch.spec_info.is_draft_input()

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
            )
