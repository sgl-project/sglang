"""Attention kernel abstraction for MiniCPM backend.

This module provides a unified interface for different attention kernels,
allowing MiniCPM to use either flash attention or flashinfer as the backend.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional, Tuple

import torch

from sglang.srt.environ import envs
from sglang.srt.runtime_context import get_parallel

if TYPE_CHECKING:
    from flashinfer import (
        BatchDecodeWithPagedKVCacheWrapper,
        BatchPrefillWithPagedKVCacheWrapper,
    )

from sglang.srt.layers.attention.minicpm.sparse_kernels import (
    convert_sparse_page_table_to_flashinfer,
)


@dataclass
class AttentionParams:
    """Parameters for attention computation.

    This dataclass contains all parameters needed for both flash attention
    and flashinfer backends.
    """

    # Query, key, value tensors
    q: torch.Tensor
    k_cache: torch.Tensor
    v_cache: torch.Tensor

    # Page table for paged KV cache
    page_table: torch.Tensor

    # Sequence lengths
    cache_seqlens: torch.Tensor
    cu_seqlens_q: torch.Tensor
    cu_seqlens_k_new: torch.Tensor
    max_seqlen_q: int

    # Attention parameters
    softmax_scale: float
    causal: bool = True
    window_size: Tuple[int, int] = (-1, -1)
    softcap: float = 0.0

    # Quantization
    k_descale: Optional[torch.Tensor] = None
    v_descale: Optional[torch.Tensor] = None

    # Other parameters
    num_splits: int = 0
    fa_impl_ver: int = 3

    # Flashinfer-specific pre-converted metadata
    flashinfer_kv_indptr: Optional[torch.Tensor] = None
    flashinfer_kv_indices: Optional[torch.Tensor] = None
    flashinfer_kv_last_page_len: Optional[torch.Tensor] = None

    # Flashinfer decode wrapper (for CUDA graph mode)
    decode_wrapper: Optional[BatchDecodeWithPagedKVCacheWrapper] = None


class AttentionKernel(ABC):
    """Abstract base class for attention kernels.

    This class defines the interface that all attention kernels must implement.
    """

    @abstractmethod
    def forward(
        self,
        params: AttentionParams,
    ) -> torch.Tensor:
        """Perform attention computation.

        Args:
            params: Attention parameters

        Returns:
            Attention output tensor
        """
        pass


class FlashAttentionKernel(AttentionKernel):
    """Flash Attention kernel implementation.

    This class wraps the flash_attn_with_kvcache function from sgl_kernel.
    """

    def __init__(self):
        from sgl_kernel.flash_attn import flash_attn_with_kvcache

        self.flash_attn_func = flash_attn_with_kvcache

    def forward(
        self,
        params: AttentionParams,
    ) -> torch.Tensor:
        """Perform attention computation using flash attention."""
        # Prepare kwargs based on fa_impl_ver
        kwargs = {}
        if params.fa_impl_ver != 3:
            kwargs["ver"] = params.fa_impl_ver

        return self.flash_attn_func(
            q=params.q,
            k_cache=params.k_cache,
            v_cache=params.v_cache,
            page_table=params.page_table,
            cache_seqlens=params.cache_seqlens,
            cu_seqlens_q=params.cu_seqlens_q,
            cu_seqlens_k_new=params.cu_seqlens_k_new,
            max_seqlen_q=params.max_seqlen_q,
            softmax_scale=params.softmax_scale,
            causal=params.causal,
            window_size=params.window_size,
            softcap=params.softcap,
            k_descale=params.k_descale,
            v_descale=params.v_descale,
            return_softmax_lse=False,
            num_splits=params.num_splits,
            **kwargs,
        )


class FlashInferKernel(AttentionKernel):
    """FlashInfer kernel implementation.

    This class wraps the flashinfer attention wrappers.
    """

    def __init__(self, model_runner):
        self.device = model_runner.device
        self.page_size = model_runner.page_size

        # KV cache attributes
        self.kv_cache_dtype = model_runner.kv_cache_dtype
        self.data_type = self.kv_cache_dtype

        # Model config attributes
        self.num_qo_heads = model_runner.model_config.num_attention_heads
        self.num_kv_heads = model_runner.model_config.get_num_kv_heads(
            get_parallel().attn_tp_size
        )
        self.head_dim = model_runner.model_config.head_dim
        # Query data type (same as KV cache dtype, but flashinfer uses separate parameters)
        self.q_data_type = self.kv_cache_dtype

        # Create workspace buffers for flashinfer
        workspace_size = envs.SGLANG_FLASHINFER_WORKSPACE_SIZE.get()
        self.decode_workspace = torch.empty(
            workspace_size,
            dtype=torch.uint8,
            device=self.device,
        )
        self.prefill_workspace = torch.empty(
            workspace_size,
            dtype=torch.uint8,
            device=self.device,
        )

        # Data format for flashinfer (Num heads, Head dim, Seq length)
        self.kv_layout = "NHD"

        # Wrappers will be created lazily
        self.decode_wrapper: Optional[BatchDecodeWithPagedKVCacheWrapper] = None
        self.prefill_wrapper: Optional[BatchPrefillWithPagedKVCacheWrapper] = None

    def _get_or_create_decode_wrapper(
        self,
    ) -> BatchDecodeWithPagedKVCacheWrapper:
        """Get or create the decode wrapper."""
        if self.decode_wrapper is None:
            from flashinfer import BatchDecodeWithPagedKVCacheWrapper

            # NOTE: use_tensor_cores=True is required for num_kv_heads=1 to avoid plan_info=None bug
            self.decode_wrapper = BatchDecodeWithPagedKVCacheWrapper(
                self.decode_workspace,
                self.kv_layout,
                use_tensor_cores=True,
            )
        return self.decode_wrapper

    def _get_or_create_prefill_wrapper(
        self,
    ) -> BatchPrefillWithPagedKVCacheWrapper:
        """Get or create the prefill wrapper."""
        if self.prefill_wrapper is None:
            from flashinfer import BatchPrefillWithPagedKVCacheWrapper

            self.prefill_wrapper = BatchPrefillWithPagedKVCacheWrapper(
                self.prefill_workspace,
                self.kv_layout,
                backend="fa2",
            )
        return self.prefill_wrapper

    def forward(
        self,
        params: AttentionParams,
    ) -> torch.Tensor:
        """Perform attention computation using flashinfer."""
        # Determine if this is prefill or decode based on max_seqlen_q
        is_prefill = params.max_seqlen_q > 1

        # CUDA graph mode: use the pre-configured wrapper from params
        if params.decode_wrapper is not None and not is_prefill:
            wrapper = params.decode_wrapper
            # For CUDA graph mode, update wrapper's internal buffers
            # Convert sparse_page_table to flashinfer format
            bs = params.cache_seqlens.shape[0]
            max_sparse_tokens = params.page_table.shape[1]

            # Get wrapper's internal buffer shapes to avoid reallocation
            kv_indptr_shape = wrapper._paged_kv_indptr_buf.shape
            kv_indices_shape = wrapper._paged_kv_indices_buf.shape
            kv_last_page_len_shape = wrapper._paged_kv_last_page_len_buf.shape

            # Create temporary buffers for conversion (reuse if possible)
            if (
                kv_indptr_shape[0] >= bs + 1
                and kv_indices_shape[0] >= bs * max_sparse_tokens
                and kv_last_page_len_shape[0] >= bs
            ):
                # Use wrapper's internal buffers directly (no allocation)
                kv_indptr = wrapper._paged_kv_indptr_buf
                kv_indices = wrapper._paged_kv_indices_buf
                kv_last_page_len = wrapper._paged_kv_last_page_len_buf
            else:
                # Allocate new buffers if wrapper's buffers are too small
                kv_indptr = torch.zeros(
                    bs + 1, dtype=torch.int32, device=params.page_table.device
                )
                kv_indices = torch.zeros(
                    bs * max_sparse_tokens,
                    dtype=torch.int32,
                    device=params.page_table.device,
                )
                kv_last_page_len = torch.zeros(
                    bs, dtype=torch.int32, device=params.page_table.device
                )

            # Convert sparse_page_table to flashinfer format
            convert_sparse_page_table_to_flashinfer(
                params.page_table,
                params.cache_seqlens,
                kv_indptr,
                kv_indices,
                kv_last_page_len,
            )
        else:
            # Non-CUDA graph mode: create wrapper and convert on-the-fly
            if is_prefill:
                wrapper = self._get_or_create_prefill_wrapper()
            else:
                wrapper = self._get_or_create_decode_wrapper()

            # Convert page table format for flashinfer
            # Use pre-converted tensors if available (for CUDA graph compatibility),
            # otherwise convert on-the-fly (for non-CUDA graph mode)
            if (
                params.flashinfer_kv_indptr is not None
                and params.flashinfer_kv_indices is not None
                and params.flashinfer_kv_last_page_len is not None
            ):
                # Use pre-converted tensors (CUDA graph safe)
                kv_indptr = params.flashinfer_kv_indptr
                kv_indices = params.flashinfer_kv_indices
                kv_last_page_len = params.flashinfer_kv_last_page_len
            else:
                bs = params.cache_seqlens.shape[0]
                max_sparse_tokens = params.page_table.shape[1]

                kv_indptr = torch.zeros(
                    bs + 1, dtype=torch.int32, device=params.page_table.device
                )
                kv_indices = torch.zeros(
                    bs * max_sparse_tokens,
                    dtype=torch.int32,
                    device=params.page_table.device,
                )
                kv_last_page_len = torch.zeros(
                    bs, dtype=torch.int32, device=params.page_table.device
                )

                kv_indptr, kv_indices, kv_last_page_len = (
                    convert_sparse_page_table_to_flashinfer(
                        params.page_table,
                        params.cache_seqlens,
                        kv_indptr,
                        kv_indices,
                        kv_last_page_len,
                    )
                )

            # Call begin_forward to set up the attention plan
            # This is required by flashinfer to cache data types and metadata
            # Skip only if we're using pre-converted tensors that match the plan
            using_preconverted = params.flashinfer_kv_indptr is not None
            if is_prefill:
                if not using_preconverted:
                    # Prefill wrapper requires qo_indptr (query indptr)
                    qo_indptr = params.cu_seqlens_q
                    wrapper.begin_forward(
                        qo_indptr,
                        kv_indptr,
                        kv_indices,
                        kv_last_page_len,
                        self.num_qo_heads,
                        self.num_kv_heads,
                        self.head_dim,
                        self.page_size,
                        q_data_type=self.q_data_type,
                        kv_data_type=self.data_type,
                        non_blocking=True,
                        causal=params.causal,
                    )
            else:
                if not using_preconverted:
                    # Decode wrapper uses indptr, indices
                    wrapper.begin_forward(
                        kv_indptr,
                        kv_indices,
                        kv_last_page_len,
                        self.num_qo_heads,
                        self.num_kv_heads,
                        self.head_dim,
                        self.page_size,
                        q_data_type=self.q_data_type,
                        kv_data_type=self.data_type,
                        non_blocking=True,
                    )

        # Perform attention
        q_data = params.q
        k_data = (params.k_cache, params.v_cache)

        if is_prefill:
            # Prefill mode: use prefill wrapper
            # flashinfer's forward doesn't need cu_seqlens, they are set in begin_forward
            o = wrapper.forward(
                q_data,
                k_data,
                causal=params.causal,
                sm_scale=params.softmax_scale,
                window_left=(
                    params.window_size[0] if params.window_size[0] != -1 else -1
                ),
                logits_soft_cap=params.softcap if params.softcap > 0 else None,
            )
        else:
            # Decode mode: use decode wrapper
            o = wrapper.forward(
                q_data,
                k_data,
                sm_scale=params.softmax_scale,
                logits_soft_cap=params.softcap if params.softcap > 0 else None,
            )

        return o


def create_attention_kernel(
    kernel_type: str,
    model_runner,
) -> AttentionKernel:
    """Factory function to create the appropriate attention kernel.

    Args:
        kernel_type: Type of kernel to create ('flash_attn' or 'flashinfer')
        model_runner: The model runner instance

    Returns:
        An instance of the requested attention kernel

    Raises:
        ValueError: If kernel_type is not recognized
    """
    if kernel_type == "flash_attn":
        return FlashAttentionKernel()
    elif kernel_type == "flashinfer":
        return FlashInferKernel(model_runner)
    else:
        raise ValueError(f"Unknown attention kernel type: {kernel_type}")
