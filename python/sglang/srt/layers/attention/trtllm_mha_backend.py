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
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode
from sglang.srt.utils import is_flashinfer_available

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

        # Forward metadata
        self.forward_metadata: Optional[TRTLLMMHAMetadata] = None

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
            "strided_indices": torch.arange(
                0, self.max_context_len, self.page_size, device=self.device
            ),
        }

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
                "strided_indices": torch.arange(
                    0, self.max_context_len, self.page_size, device=self.device
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

            self.draft_extend_metadata[bs] = metadata
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
            page_indices //= self.page_size
            metadata.page_table[:, :max_seq_pages].copy_(page_indices)
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
        self.forward_metadata = metadata

    def get_cuda_graph_seq_len_fill_value(self) -> int:
        """Get the fill value for sequence lengths in CUDA graph."""
        return 1

    def _should_use_fused_fp8_path(self, save_kv_cache: bool, k: torch.Tensor) -> bool:
        """Check if we should use the fused FP8 KV cache write path."""
        return save_kv_cache and k is not None and self.data_type == torch.float8_e4m3fn

    def _fused_fp8_set_kv_buffer(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        **kwargs,
    ):
        """Fused FP8 quantization and KV cache write."""
        cache_loc = forward_batch.out_cache_loc

        # Get K/V cache buffers from token_to_kv_pool
        k_cache, v_cache = forward_batch.token_to_kv_pool.get_kv_buffer(layer.layer_id)

        fused_fp8_set_kv_buffer(
            k=k,
            v=v,
            k_cache=k_cache,
            v_cache=v_cache,
            cache_loc=cache_loc,
            k_scale=layer.k_scale,  # May be None
            v_scale=layer.v_scale,  # May be None
            page_size=self.page_size,
        )

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

        # Convert the page table to a strided format
        if self.page_size > 1:
            self.strided_indices = torch.arange(
                0, metadata.page_table.shape[1], self.page_size, device=self.device
            )
            metadata.page_table = (
                metadata.page_table[:, self.strided_indices] // self.page_size
            )

        self.forward_metadata = metadata

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
        cache_loc = forward_batch.out_cache_loc

        use_fused_fp8_path = self._should_use_fused_fp8_path(save_kv_cache, k)

        # nvfp4 kv cache path using origin fake nvfp4 path here
        nvfp4_kvcache_path = self.data_type == torch.float4_e2m1fn_x2
        # nvfp4_kvcache_path = False
        logger.debug(
            f"[NVFP4 Decode Path] {nvfp4_kvcache_path=}, {use_fused_fp8_path=}"
        )
        logger.debug(
            f"[NVFP4 Decode Path] q.dtype={q.dtype}, k.dtype={k.dtype}, v.dtype={v.dtype}"
        )

        if not hasattr(layer, "k_scale") or layer.k_scale is None:
            k_scale = 1.0
        else:
            k_scale = layer.k_scale
        if not hasattr(layer, "v_scale") or layer.v_scale is None:
            v_scale = 1.0
        else:
            v_scale = layer.v_scale
        logger.debug(f"[NVFP4 Decode Path] {k_scale=}, {v_scale=}")

        if use_fused_fp8_path:
            # Use fused FP8 quantization + KV cache write path
            self._fused_fp8_set_kv_buffer(
                q=q,
                k=k,
                v=v,
                layer=layer,
                forward_batch=forward_batch,
            )
            k = None
            v = None
        else:
            logger.debug(f"[NVFP4 Decode Path] saving kv cache")
            # Use original set_kv_buffer path
            if save_kv_cache and k is not None:
                # logger.debug(f"[NVFP4 Decode Path] Saving kv: k max/min {k.max()},{k.min()}, v max/min {v.max()},{v.min()}")
                forward_batch.token_to_kv_pool.set_kv_buffer(
                    layer, cache_loc, k, v, k_scale, v_scale
                )

        # logger.debug(f"[NVFP4 Decode Path] Stage2: q max/min {q.float().max()},{q.float().min()}")

        if (
            self.data_type == torch.float8_e4m3fn
            or self.data_type == torch.float4_e2m1fn_x2
        ):
            q = q.to(torch.float8_e4m3fn)

        logger.debug(f"{q.dtype=}")
        q = q.contiguous().view(-1, layer.tp_q_head_num, layer.head_dim)
        if nvfp4_kvcache_path:
            logger.debug(
                f"[NVFP4 Decode Path] Reshaping kv cache and kv cache block scales"
            )
            k_cache, k_cache_scales = forward_batch.token_to_kv_pool.get_fp4_key_buffer(
                layer.layer_id
            )
            v_cache, v_cache_scales = (
                forward_batch.token_to_kv_pool.get_fp4_value_buffer(layer.layer_id)
            )
            k_cache = k_cache.view(
                -1, self.page_size, layer.tp_k_head_num, layer.head_dim // 2
            ).permute(0, 2, 1, 3)
            v_cache = v_cache.view(
                -1, self.page_size, layer.tp_v_head_num, layer.head_dim // 2
            ).permute(0, 2, 1, 3)

            k_cache_scales = k_cache_scales.view(
                -1, self.page_size, layer.tp_k_head_num, layer.head_dim // 16
            ).permute(0, 2, 1, 3)
            v_cache_scales = v_cache_scales.view(
                -1, self.page_size, layer.tp_v_head_num, layer.head_dim // 16
            ).permute(0, 2, 1, 3)
            kv_cache = (k_cache, v_cache)
            kv_cache_block_scales = (k_cache_scales, v_cache_scales)

        else:
            k_cache, v_cache = forward_batch.token_to_kv_pool.get_kv_buffer(
                layer.layer_id
            )
            # shape conversion:
            # [num_pages, page_size, num_kv_heads, head_dim] -> [num_pages, num_kv_heads, page_size, head_dim]
            k_cache = k_cache.view(
                -1, self.page_size, layer.tp_k_head_num, layer.head_dim
            ).permute(0, 2, 1, 3)
            v_cache = v_cache.view(
                -1, self.page_size, layer.tp_v_head_num, layer.head_dim
            ).permute(0, 2, 1, 3)
            kv_cache = (k_cache, v_cache)
            kv_cache_block_scales = None

        # logger.debug(f"[NVFP4 Decode Path] Stage3: q max/min {q.float().max()},{q.float().min()}")

        # TODO: add support for quantization
        q_scale = 1.0
        # k_scale = (
        #     layer.k_scale_float
        #     if getattr(layer, "k_scale_float", None) is not None
        #     else 1.0
        # )

        if nvfp4_kvcache_path:
            k_scale *= 6
            k_cache_scales = (k_cache_scales.float() / 6).to(torch.float8_e4m3fn)
            v_scale *= 6
            v_cache_scales = (v_cache_scales.float() / 6).to(torch.float8_e4m3fn)
            # kv_cache = (k_cache, v_cache)
            kv_cache_block_scales = (k_cache_scales, v_cache_scales)
            logger.debug(
                f"[NVFP4 Decode Path] Adjust kv scales, {k_scale=}, {v_scale=}"
            )

        bmm1_scale = q_scale * k_scale * layer.scaling
        bmm2_scale = 1.0 * v_scale
        # sink: additional value per head in the denominator of the softmax.
        attention_sink = kwargs.get("sinks", None)

        # logger.debug(f"[NVFP4 Decode Path] Calling trtllm kernel, {layer.scaling=}, {bmm1_scale=}, {bmm2_scale=}, {q.shape=}, {kv_cache[0].shape=}, {kv_cache_block_scales[0].shape=}")
        # logger.debug(f"[NVFP4 Decode Path] TRTLLM Kernel params: {self.forward_metadata.cache_seqlens_int32=}, q max/min {q.float().max()},{q.float().min()}")
        # Call TRT-LLM kernel
        # raw_out: like q, [bs, acc_q_len, num_q_heads, head_dim] but with output dtype

        # torch.cuda.synchronize()
        # prefix = "./unit_test/"
        # torch.save(q, f"{prefix}/query.pt")
        # torch.save(kv_cache, f"{prefix}/kv_cache.pt")
        # torch.save(self.workspace_buffer, f"{prefix}/workspace_buffer.pt")
        # torch.save(self.forward_metadata.page_table, f"{prefix}/block_tables.pt")
        # torch.save(self.forward_metadata.cache_seqlens_int32, f"{prefix}/seq_lens.pt")

        # torch.save(self.max_context_len, f"{prefix}/max_seq_len.pt")
        # torch.save(bmm1_scale, f"{prefix}/bmm1_scale.pt")
        # torch.save(bmm2_scale, f"{prefix}/bmm2_scale.pt")
        # torch.save(layer.sliding_window_size, f"{prefix}/window_left.pt")
        # torch.save(attention_sink, f"{prefix}/sinks.pt")
        # torch.save(kv_cache_block_scales, f"{prefix}/kv_block_scales.pt")
        # exit(0)

        o = flashinfer.decode.trtllm_batch_decode_with_kv_cache(
            query=q,
            kv_cache=kv_cache,
            workspace_buffer=self.workspace_buffer,
            block_tables=self.forward_metadata.page_table,
            seq_lens=self.forward_metadata.cache_seqlens_int32,
            max_seq_len=self.max_context_len,
            bmm1_scale=bmm1_scale,
            bmm2_scale=bmm2_scale,
            window_left=layer.sliding_window_size,
            # TODO: add attention_sink operation or nvfp4 scale factor if needed
            sinks=attention_sink,
            out_dtype=q.dtype,  # model_runner.dtype
            kv_block_scales=kv_cache_block_scales,
        )
        o = o.to(self.q_data_type)

        # print(o)
        # exit(0)
        # torch.cuda.synchronize()

        # logger.debug(f"{o.sum()=}")
        # logger.debug(f"o min/max {o.float().min()}, {o.float().max()}")
        # logger.debug(f"[NVFP4 Decode Path] ===========Finish trtllm kernel=============")

        return o.view(-1, layer.tp_q_head_num * layer.head_dim)

    def forward_extend(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        save_kv_cache=True,
        **kwargs,
    ):
        cache_loc = forward_batch.out_cache_loc

        use_fused_fp8_path = self._should_use_fused_fp8_path(save_kv_cache, k)

        nvfp4_kvcache_path = self.data_type == torch.float4_e2m1fn_x2
        # nvfp4_kvcache_path = False
        logger.debug(f"q.dtype={q.dtype}, k.dtype={k.dtype}, v.dtype={v.dtype}")

        if not hasattr(layer, "k_scale") or layer.k_scale is None:
            k_scale = 1.0
        else:
            k_scale = layer.k_scale
        if not hasattr(layer, "v_scale") or layer.v_scale is None:
            v_scale = 1.0
        else:
            v_scale = layer.v_scale

        if nvfp4_kvcache_path:
            assert (
                not forward_batch.forward_mode.is_target_verify()
            ), "only prefill for now"
            # assert self.data_type == torch.float8_e4m3fn, "only fp8 kv cache for now"

            logger.debug(
                f"[NVFP4 Path] batch_size={forward_batch.batch_size}, "
                f"extend_seq_lens={forward_batch.extend_seq_lens.tolist()}, "
                f"seq_lens={forward_batch.seq_lens.tolist()}"
            )

            prefix_lens = (
                forward_batch.seq_lens - forward_batch.extend_seq_lens
            )  # previous length
            batch_size = forward_batch.batch_size

            k_prev_fp8_list = []
            v_prev_fp8_list = []
            k_buffer_nvfp4, k_scales_buffer = (
                forward_batch.token_to_kv_pool.get_fp4_key_buffer(layer.layer_id)
            )
            v_buffer_nvfp4, v_scales_buffer = (
                forward_batch.token_to_kv_pool.get_fp4_value_buffer(layer.layer_id)
            )

            logger.debug(
                f"[NVFP4 Path] k_buffer_nvfp4.shape={k_buffer_nvfp4.shape}, "
                f"k_scales_buffer.shape={k_scales_buffer.shape}"
            )

            for batch_idx in range(batch_size):
                req_pool_idx = forward_batch.req_pool_indices[batch_idx].item()
                prev_len = prefix_lens[batch_idx].item()

                logger.debug(
                    f"[NVFP4 Path] batch_idx={batch_idx}, req_pool_idx={req_pool_idx}, prev_len={prev_len}"
                )

                if prev_len > 0:
                    prev_token_indices = forward_batch.req_to_token_pool.req_to_token[
                        req_pool_idx, :prev_len
                    ]

                    # Gather nvfp4 KV from paged buffer
                    k_prev_nvfp4 = k_buffer_nvfp4[
                        prev_token_indices
                    ]  # [prev_len, num_heads, head_dim/2]
                    k_prev_scales = k_scales_buffer[
                        prev_token_indices
                    ]  # [prev_len, ...]
                    v_prev_nvfp4 = v_buffer_nvfp4[prev_token_indices]
                    v_prev_scales = v_scales_buffer[prev_token_indices]

                    logger.debug(
                        f"[NVFP4 Path] batch_idx={batch_idx}, k_prev_nvfp4.shape={k_prev_nvfp4.shape}, "
                        f"k_prev_scales.shape={k_prev_scales.shape}"
                    )

                    from sglang.srt.layers.quantization.fp4_utils import (
                        NVFP4QuantizeUtil,
                    )

                    k_prev_bf16 = NVFP4QuantizeUtil.batched_dequantize(
                        k_prev_nvfp4.view(torch.uint8), k_prev_scales, k_scale
                    )
                    v_prev_bf16 = NVFP4QuantizeUtil.batched_dequantize(
                        v_prev_nvfp4.view(torch.uint8), v_prev_scales, v_scale
                    )
                    logger.debug(
                        f"[NVFP4 Path] batch_idx={batch_idx}, dequantized k_prev_bf16.shape={k_prev_bf16.shape}"
                    )

                    # bf16 -> fp8
                    k_prev_fp8_list.append(k_prev_bf16.to(torch.float8_e4m3fn))
                    v_prev_fp8_list.append(v_prev_bf16.to(torch.float8_e4m3fn))
                else:
                    # add empty tensor
                    logger.debug(
                        f"[NVFP4 Path] batch_idx={batch_idx}, no previous tokens, creating empty tensor"
                    )
                    k_prev_fp8_list.append(
                        torch.empty(
                            0,
                            k.shape[1],
                            k.shape[2],
                            dtype=torch.float8_e4m3fn,
                            device=k.device,
                        )
                    )
                    v_prev_fp8_list.append(
                        torch.empty(
                            0,
                            v.shape[1],
                            v.shape[2],
                            dtype=torch.float8_e4m3fn,
                            device=v.device,
                        )
                    )

            # convert to fp8
            k_cur_fp8 = k.to(torch.float8_e4m3fn)
            v_cur_fp8 = v.to(torch.float8_e4m3fn)

            logger.debug(
                f"[NVFP4 Path] k_cur_fp8.shape={k_cur_fp8.shape}, v_cur_fp8.shape={v_cur_fp8.shape}"
            )

            # Write current bf16 chunk to nvfp4 buffer
            logger.debug(
                f"[NVFP4 Path] Writing current chunk to nvfp4 buffer, k.shape={k.shape}, v.shape={v.shape}"
            )
            forward_batch.token_to_kv_pool.set_kv_buffer(
                layer=layer,
                loc=cache_loc,
                cache_k=k,
                cache_v=v,
                k_scale=k_scale,
                v_scale=v_scale,
            )
            logger.debug(f"[NVFP4 Path] Current chunk written to nvfp4 buffer")

            # Split current chunk by batch
            extend_lens_list = forward_batch.extend_seq_lens.tolist()
            k_cur_split = torch.split(k_cur_fp8, extend_lens_list, dim=0)
            v_cur_split = torch.split(v_cur_fp8, extend_lens_list, dim=0)

            # Concatenate per request: previous + current
            k_full_list = [
                torch.cat([k_prev, k_cur], dim=0)
                for k_prev, k_cur in zip(k_prev_fp8_list, k_cur_split)
            ]
            v_full_list = [
                torch.cat([v_prev, v_cur], dim=0)
                for v_prev, v_cur in zip(v_prev_fp8_list, v_cur_split)
            ]

            # Calculate total pages needed for temporary paged buffer
            full_lens = [kv.shape[0] for kv in k_full_list]
            pages_per_req = [
                (seq_len + self.page_size - 1) // self.page_size
                for seq_len in full_lens
            ]
            total_pages = sum(pages_per_req)

            logger.debug(
                f"[NVFP4 Path] full_lens={full_lens}, pages_per_req={pages_per_req}, total_pages={total_pages}"
            )

            # Allocate temporary paged buffer
            # [num_pages, num_kv_heads, page_size, head_dim]
            k_temp_paged = torch.zeros(
                total_pages,
                layer.tp_k_head_num,
                self.page_size,
                layer.head_dim,
                dtype=torch.float8_e4m3fn,
                device=k.device,
            )
            v_temp_paged = torch.zeros(
                total_pages,
                layer.tp_v_head_num,
                self.page_size,
                layer.head_dim,
                dtype=torch.float8_e4m3fn,
                device=v.device,
            )

            logger.debug(
                f"[NVFP4 Path] k_temp_paged.shape={k_temp_paged.shape}, v_temp_paged.shape={v_temp_paged.shape}"
            )

            # Build temporary page table
            temp_page_table = torch.zeros(
                batch_size, max(pages_per_req), dtype=torch.int32, device=k.device
            )

            # Fill temporary paged buffer and page table
            page_offset = 0
            for req_idx, (k_full, v_full, num_pages, seq_len) in enumerate(
                zip(k_full_list, v_full_list, pages_per_req, full_lens)
            ):
                logger.debug(
                    f"[NVFP4 Path] Filling req_idx={req_idx}, seq_len={seq_len}, num_pages={num_pages}, "
                    f"page_offset={page_offset}"
                )

                # Fill page table for this request
                temp_page_table[req_idx, :num_pages] = torch.arange(
                    page_offset,
                    page_offset + num_pages,
                    dtype=torch.int32,
                    device=k.device,
                )

                # Write KV data to paged buffer (with page_size padding)
                for page_idx in range(num_pages):
                    start_token = page_idx * self.page_size
                    end_token = min(start_token + self.page_size, seq_len)
                    token_count = end_token - start_token

                    # k_full: [seq_len, num_heads, head_dim] -> [num_heads, seq_len, head_dim]
                    k_page_data = k_full[start_token:end_token].permute(
                        1, 0, 2
                    )  # [num_heads, token_count, head_dim]
                    v_page_data = v_full[start_token:end_token].permute(1, 0, 2)

                    # Write to paged buffer (last page may have padding)
                    k_temp_paged[page_offset + page_idx, :, :token_count, :] = (
                        k_page_data
                    )
                    v_temp_paged[page_offset + page_idx, :, :token_count, :] = (
                        v_page_data
                    )

                page_offset += num_pages

            # Use temporary paged KV cache for attention
            kv_cache = (k_temp_paged, v_temp_paged)

            # Override metadata to use temporary page table
            temp_metadata = TRTLLMMHAMetadata()
            temp_metadata.cache_seqlens_int32 = forward_batch.seq_lens.to(torch.int32)
            temp_metadata.max_seq_len_q = self.forward_metadata.max_seq_len_q
            temp_metadata.max_seq_len_k = max(full_lens)
            temp_metadata.cu_seqlens_q = self.forward_metadata.cu_seqlens_q
            temp_metadata.cu_seqlens_k = torch.nn.functional.pad(
                torch.cumsum(
                    torch.tensor(full_lens, dtype=torch.int32, device=k.device), dim=0
                ),
                (1, 0),
            )
            temp_metadata.page_table = temp_page_table

            logger.debug(
                f"[NVFP4 Path] temp_metadata: max_seq_len_q={temp_metadata.max_seq_len_q}, "
                f"max_seq_len_k={temp_metadata.max_seq_len_k}, "
                f"page_table.shape={temp_metadata.page_table.shape}"
            )

            # Temporarily replace forward_metadata
            original_metadata = self.forward_metadata
            self.forward_metadata = temp_metadata

            # else:
            # target verify
            # k_cache_nvfp4, k_scales = forward_batch.token_to_kv_pool._get_key_nvfp4_from_nvfp4_buffer()
            # v_cache_nvfp4, v_scales = forward_batch.token_to_kv_pool._get_value_nvfp4_from_nvfp4_buffer()
        elif use_fused_fp8_path:
            # Use fused FP8 quantization + KV cache write path
            self._fused_fp8_set_kv_buffer(
                q=q,
                k=k,
                v=v,
                layer=layer,
                forward_batch=forward_batch,
            )
            k = None
            v = None
        else:
            # Use original set_kv_buffer path
            if save_kv_cache and k is not None:
                forward_batch.token_to_kv_pool.set_kv_buffer(
                    layer, cache_loc, k, v, k_scale, v_scale
                )

        if not nvfp4_kvcache_path:
            # [num_pages, page_size, num_kv_heads, head_dim] -> [num_pages, num_kv_heads, page_size, head_dim]
            k_cache, v_cache = forward_batch.token_to_kv_pool.get_kv_buffer(
                layer.layer_id,
                k_scale,
                v_scale,
            )
            k_cache = k_cache.view(
                -1, self.page_size, layer.tp_k_head_num, layer.head_dim
            ).permute(0, 2, 1, 3)
            v_cache = v_cache.view(
                -1, self.page_size, layer.tp_v_head_num, layer.head_dim
            ).permute(0, 2, 1, 3)
            kv_cache = (k_cache, v_cache)

        q = q.contiguous().view(-1, layer.tp_q_head_num, layer.head_dim)
        if self.data_type == torch.float8_e4m3fn or nvfp4_kvcache_path:
            q = q.to(torch.float8_e4m3fn)

        logger.debug(f"[forward_extend] q.shape={q.shape}, q.dtype={q.dtype}")

        # sink: additional value per head in the denominator of the softmax.
        attention_sink = kwargs.get("sinks", None)
        # TODO: add support for quantization
        q_scale = 1.0
        # k_scale = (
        #     # where to load k global and v global scales?
        #     layer.k_scale_float
        #     if getattr(layer, "k_scale_float", None) is not None
        #     else 1.0
        # )
        bmm1_scale = q_scale * k_scale * layer.scaling
        bmm2_scale = 1.0

        if forward_batch.forward_mode.is_target_verify():
            o = flashinfer.decode.trtllm_batch_decode_with_kv_cache(
                query=q,
                kv_cache=kv_cache,
                workspace_buffer=self.workspace_buffer,
                block_tables=self.forward_metadata.page_table,
                seq_lens=self.forward_metadata.cache_seqlens_int32,
                max_seq_len=self.max_context_len,
                bmm1_scale=bmm1_scale,
                bmm2_scale=bmm2_scale,
                window_left=layer.sliding_window_size,
                # TODO: add attention_sink operation or nvfp4 scale factor if needed
                sinks=attention_sink,
                out_dtype=self.q_data_type,  # model_runner.dtype
                q_len_per_req=self.forward_metadata.max_seq_len_q,
            )
        else:
            # TODO: pass catted FP8 cache to trtllm mha kernel
            logger.debug(
                f"[forward_extend] prefill/extend mode, calling trtllm_batch_context_with_kv_cache"
            )
            logger.debug(
                f"[forward_extend] metadata: max_seq_len_q={self.forward_metadata.max_seq_len_q}, "
                f"max_seq_len_k={self.forward_metadata.max_seq_len_k}, "
                f"batch_size={forward_batch.batch_size}"
            )

            # print(f"{q.shape=}, {q.dtype}")

            # ============ Debug Info Before Kernel Call ============
            logger.debug(
                f"[KERNEL DEBUG] ====== trtllm_batch_context_with_kv_cache Parameters ======"
            )
            logger.debug(f"[KERNEL DEBUG] nvfp4_kvcache_path={nvfp4_kvcache_path}")

            # Query info
            logger.debug(f"[KERNEL DEBUG] query.shape={q.shape}, query.dtype={q.dtype}")
            logger.debug(
                f"[KERNEL DEBUG] query.device={q.device}, query.is_contiguous={q.is_contiguous()}"
            )

            # KV cache info
            k_cache, v_cache = kv_cache
            logger.debug(
                f"[KERNEL DEBUG] k_cache.shape={k_cache.shape}, k_cache.dtype={k_cache.dtype}"
            )
            logger.debug(
                f"[KERNEL DEBUG] v_cache.shape={v_cache.shape}, v_cache.dtype={v_cache.dtype}"
            )
            logger.debug(
                f"[KERNEL DEBUG] k_cache.is_contiguous={k_cache.is_contiguous()}"
            )
            logger.debug(
                f"[KERNEL DEBUG] v_cache.is_contiguous={v_cache.is_contiguous()}"
            )
            logger.debug(f"[KERNEL DEBUG] k_cache layout: strides={k_cache.stride()}")
            logger.debug(f"[KERNEL DEBUG] v_cache layout: strides={v_cache.stride()}")

            # Block tables and sequence lengths
            logger.debug(
                f"[KERNEL DEBUG] block_tables.shape={self.forward_metadata.page_table.shape}, "
                f"dtype={self.forward_metadata.page_table.dtype}"
            )
            logger.debug(
                f"[KERNEL DEBUG] block_tables sample (first 2 rows):\n{self.forward_metadata.page_table[:2]}"
            )
            logger.debug(
                f"[KERNEL DEBUG] block_tables min={self.forward_metadata.page_table.min().item()}, "
                f"max={self.forward_metadata.page_table.max().item()}"
            )

            logger.debug(
                f"[KERNEL DEBUG] seq_lens={self.forward_metadata.cache_seqlens_int32.tolist()}"
            )
            logger.debug(
                f"[KERNEL DEBUG] cum_seq_lens_q={self.forward_metadata.cu_seqlens_q.tolist()}"
            )
            logger.debug(
                f"[KERNEL DEBUG] cum_seq_lens_kv={self.forward_metadata.cu_seqlens_k.tolist()}"
            )

            # Max lengths
            logger.debug(
                f"[KERNEL DEBUG] max_q_len={self.forward_metadata.max_seq_len_q}"
            )
            logger.debug(f"[KERNEL DEBUG] max_kv_len={self.max_context_len}")
            logger.debug(f"[KERNEL DEBUG] batch_size={forward_batch.batch_size}")

            # Scales
            logger.debug(
                f"[KERNEL DEBUG] bmm1_scale={bmm1_scale}, bmm2_scale={bmm2_scale}"
            )
            logger.debug(
                f"[KERNEL DEBUG] q_scale={q_scale}, k_scale={k_scale}, layer.scaling={layer.scaling}"
            )

            # Window and other params
            logger.debug(f"[KERNEL DEBUG] window_left={layer.sliding_window_size}")
            logger.debug(f"[KERNEL DEBUG] attention_sink={attention_sink}")
            logger.debug(f"[KERNEL DEBUG] out_dtype={self.q_data_type}")

            # Workspace buffer
            logger.debug(
                f"[KERNEL DEBUG] workspace_buffer.shape={self.workspace_buffer.shape}, "
                f"dtype={self.workspace_buffer.dtype}"
            )
            logger.debug(
                f"[KERNEL DEBUG] workspace_buffer size={self.workspace_buffer.numel() * self.workspace_buffer.element_size() / (1024**2):.2f} MB"
            )

            # Page size info
            logger.debug(f"[KERNEL DEBUG] page_size={self.page_size}")

            # Sanity checks
            expected_total_q_tokens = self.forward_metadata.cu_seqlens_q[-1].item()
            expected_total_kv_tokens = self.forward_metadata.cu_seqlens_k[-1].item()
            logger.debug(
                f"[KERNEL DEBUG] Sanity check: q.shape[0]={q.shape[0]} vs expected_total_q_tokens={expected_total_q_tokens}"
            )
            logger.debug(
                f"[KERNEL DEBUG] Sanity check: expected_total_kv_tokens={expected_total_kv_tokens}"
            )

            # Check if dimensions match expectations
            if nvfp4_kvcache_path:
                logger.debug(f"[KERNEL DEBUG] NVFP4 Path: using temporary paged buffer")
                logger.debug(
                    f"[KERNEL DEBUG] NVFP4 Path: num_pages={k_cache.shape[0]}, "
                    f"page_size={k_cache.shape[2]}, num_heads={k_cache.shape[1]}, head_dim={k_cache.shape[3]}"
                )
            else:
                logger.debug(f"[KERNEL DEBUG] Standard Path: using original KV buffer")

            logger.debug(f"[KERNEL DEBUG] ====== End of Parameter Dump ======")

            o = flashinfer.prefill.trtllm_batch_context_with_kv_cache(
                query=q,
                kv_cache=kv_cache,
                workspace_buffer=self.workspace_buffer,
                block_tables=self.forward_metadata.page_table,
                seq_lens=self.forward_metadata.cache_seqlens_int32,
                max_q_len=self.forward_metadata.max_seq_len_q,
                max_kv_len=self.max_context_len,
                bmm1_scale=bmm1_scale,
                bmm2_scale=bmm2_scale,
                batch_size=forward_batch.batch_size,
                cum_seq_lens_q=self.forward_metadata.cu_seqlens_q,
                cum_seq_lens_kv=self.forward_metadata.cu_seqlens_k,
                window_left=layer.sliding_window_size,
                # TODO: add attention_sink operation or nvfp4 scale factor if needed
                sinks=attention_sink,
                out_dtype=self.q_data_type,  # model_runner.dtype
            )

        if nvfp4_kvcache_path:
            self.forward_metadata = original_metadata
            logger.debug(f"[NVFP4 Path] Restored original metadata")

        logger.debug(f"[forward_extend] o.shape={o.shape}, returning reshaped output")

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
