# SPDX-License-Identifier: Apache-2.0
"""Dense variable-length TileLang FlashAttention for Volta (SM70).

The kernel is adapted from TileLang's variable-length MHA forward example and
uses the SM70 lowering safeguards already exercised by SGLang's vendored
TileLang paged-attention kernels. MiniMax H3 consumes the exact packed contract
implemented here: FP16 ``[tokens, heads, 128]`` Q/K/V, int32 cumulative
sequence lengths, non-causal self-attention, and FP32 softmax accumulation.
"""

import tilelang
import tilelang.language as T
import torch
import torch.nn.functional as F
from sglang.multimodal_gen.runtime.layers.attention.backends.attention_backend import (
    AttentionBackend,
    AttentionImpl,
    AttentionMetadata,
    AttentionMetadataBuilder,
)
from sglang.multimodal_gen.runtime.platforms import AttentionBackendEnum
from tilelang.jit.adapter.base import BaseKernelAdapter as _BaseKernelAdapter

tilelang.set_log_level("WARNING")

_PASS_CONFIGS = {
    tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
    tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
    tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
}

# TileLang 0.1.8 mutates decorator result-index lists in place. Apply the same
# idempotent compatibility patch as the existing V100 paged-attention kernel.
if not getattr(_BaseKernelAdapter, "_legalize_result_idx_patched", False):
    _original_legalize_result_idx = _BaseKernelAdapter._legalize_result_idx

    def _legalize_result_idx_safe(self, result_idx):
        if isinstance(result_idx, list):
            result_idx = list(result_idx)
        return _original_legalize_result_idx(self, result_idx)

    _BaseKernelAdapter._legalize_result_idx = _legalize_result_idx_safe
    _BaseKernelAdapter._legalize_result_idx_patched = True


@tilelang.jit(out_idx=[6], pass_configs=_PASS_CONFIGS)
def _dense_varlen_kernel_func(
    batch: int,
    heads: int,
    dim: int,
    *,
    block_m: int = 64,
    block_n: int = 32,
    num_stages: int = 0,
    threads: int = 256,
):
    total_tokens = T.dynamic("total_tokens")

    @T.prim_func
    def main(
        query: T.Tensor([total_tokens, heads, dim], T.float16),
        key: T.Tensor([total_tokens, heads, dim], T.float16),
        value: T.Tensor([total_tokens, heads, dim], T.float16),
        cu_seqlens: T.Tensor([batch + 1], T.int32),
        max_seqlen: T.int32,
        softmax_scale: T.float32,
        output: T.Tensor([total_tokens, heads, dim], T.float16),
    ):
        with T.Kernel(
            T.ceildiv(max_seqlen, block_m), heads, batch, threads=threads
        ) as (query_block, head, sequence):
            query_shared = T.alloc_shared([block_m, dim], T.float16)
            key_shared = T.alloc_shared([block_n, dim], T.float16)
            value_shared = T.alloc_shared([block_n, dim], T.float16)
            probability_shared = T.alloc_shared([block_m, block_n], T.float16)
            scores = T.alloc_fragment([block_m, block_n], T.float32)
            scores_fp16 = T.alloc_fragment([block_m, block_n], T.float16)
            output_acc = T.alloc_fragment([block_m, dim], T.float32)
            row_max = T.alloc_fragment([block_m], T.float32)
            previous_row_max = T.alloc_fragment([block_m], T.float32)
            row_sum = T.alloc_fragment([block_m], T.float32)
            score_sum = T.alloc_fragment([block_m], T.float32)
            rescale = T.alloc_fragment([block_m], T.float32)

            sequence_start = cu_seqlens[sequence]
            sequence_stop = cu_seqlens[sequence + 1]
            sequence_length = sequence_stop - sequence_start
            query_start = sequence_start + query_block * block_m
            softmax_scale_log2 = softmax_scale * 1.44269504

            T.clear(query_shared)
            for row, column in T.Parallel(block_m, dim):
                if query_start + row < sequence_stop:
                    query_shared[row, column] = query[query_start + row, head, column]

            T.fill(output_acc, 0)
            T.fill(row_max, -T.infinity(T.float32))
            T.fill(row_sum, 0)

            for key_block in T.Pipelined(
                T.ceildiv(sequence_length, block_n), num_stages=num_stages
            ):
                T.clear(key_shared)
                for row, column in T.Parallel(block_n, dim):
                    key_offset = key_block * block_n + row
                    if key_offset < sequence_length:
                        key_shared[row, column] = key[
                            sequence_start + key_offset, head, column
                        ]

                for row, column in T.Parallel(block_m, block_n):
                    scores[row, column] = T.if_then_else(
                        (query_block * block_m + row < sequence_length)
                        & (key_block * block_n + column < sequence_length),
                        0,
                        -T.infinity(T.float32),
                    )
                T.gemm(
                    query_shared,
                    key_shared,
                    scores,
                    transpose_B=True,
                    policy=T.GemmWarpPolicy.FullRow,
                )

                T.copy(row_max, previous_row_max)
                T.reduce_max(scores, row_max, dim=1, clear=False)
                for row in T.Parallel(block_m):
                    row_max[row] = T.if_then_else(
                        row_max[row] == -T.infinity(T.float32),
                        0,
                        T.max(row_max[row], previous_row_max[row]),
                    )
                    rescale[row] = T.exp2(
                        previous_row_max[row] * softmax_scale_log2
                        - row_max[row] * softmax_scale_log2
                    )
                    row_sum[row] *= rescale[row]

                for row, column in T.Parallel(block_m, dim):
                    output_acc[row, column] *= rescale[row]
                for row, column in T.Parallel(block_m, block_n):
                    scores[row, column] = T.exp2(
                        scores[row, column] * softmax_scale_log2
                        - row_max[row] * softmax_scale_log2
                    )
                T.reduce_sum(scores, score_sum, dim=1)
                for row in T.Parallel(block_m):
                    row_sum[row] += score_sum[row]

                T.clear(value_shared)
                for row, column in T.Parallel(block_n, dim):
                    value_offset = key_block * block_n + row
                    if value_offset < sequence_length:
                        value_shared[row, column] = value[
                            sequence_start + value_offset, head, column
                        ]
                for row, column in T.Parallel(block_m, block_n):
                    probability_shared[row, column] = T.cast(
                        scores[row, column], T.float16
                    )
                T.copy(probability_shared, scores_fp16)
                T.gemm(
                    scores_fp16,
                    value_shared,
                    output_acc,
                    policy=T.GemmWarpPolicy.Square,
                )

            for row, column in T.Parallel(block_m, dim):
                if query_start + row < sequence_stop:
                    output[query_start + row, head, column] = T.cast(
                        output_acc[row, column] / row_sum[row], T.float16
                    )

    return main


_KERNEL_CACHE = {}


def _get_dense_varlen_kernel(*, batch: int, heads: int, dim: int):
    key = (batch, heads, dim)
    kernel = _KERNEL_CACHE.get(key)
    if kernel is None:
        kernel = _dense_varlen_kernel_func(batch=batch, heads=heads, dim=dim)
        _KERNEL_CACHE[key] = kernel
    return kernel


def tilelang_flash_attn_varlen(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    cu_seqlens: torch.Tensor,
    max_seqlen: int,
    softmax_scale: float,
) -> torch.Tensor:
    """Run dense packed non-causal attention through TileLang on SM70."""

    if query.ndim != 3 or key.shape != query.shape or value.shape != query.shape:
        raise ValueError(
            "TileLang V100 attention requires matching [tokens, heads, dim] Q/K/V"
        )
    if query.dtype != torch.float16:
        raise ValueError(
            f"TileLang V100 attention requires FP16 Q/K/V, got {query.dtype}"
        )
    if not query.is_cuda:
        raise ValueError("TileLang V100 attention requires CUDA tensors")
    if torch.cuda.get_device_capability(query.device) != (7, 0):
        raise ValueError("TileLang V100 attention is restricted to SM70")
    if query.shape[-1] != 128:
        raise ValueError(
            f"TileLang V100 attention requires head dimension 128, got {query.shape[-1]}"
        )
    if cu_seqlens.ndim != 1 or cu_seqlens.numel() < 2:
        raise ValueError("cu_seqlens must contain at least one packed sequence")

    cu_seqlens = cu_seqlens.to(device=query.device, dtype=torch.int32).contiguous()
    kernel = _get_dense_varlen_kernel(
        batch=cu_seqlens.numel() - 1,
        heads=query.shape[1],
        dim=query.shape[2],
    )
    return kernel(
        query.contiguous(),
        key.contiguous(),
        value.contiguous(),
        cu_seqlens,
        int(max_seqlen),
        float(softmax_scale),
    )


class TileLangFlashAttentionV100Backend(AttentionBackend):
    accept_output_buffer = True

    @staticmethod
    def get_supported_head_sizes() -> list[int]:
        return [128]

    @staticmethod
    def get_enum() -> AttentionBackendEnum:
        return AttentionBackendEnum.TILELANG_FA_V100

    @staticmethod
    def get_impl_cls() -> type["TileLangFlashAttentionV100Impl"]:
        return TileLangFlashAttentionV100Impl

    @staticmethod
    def get_metadata_cls() -> type[AttentionMetadata]:
        raise NotImplementedError

    @staticmethod
    def get_builder_cls() -> type[AttentionMetadataBuilder]:
        raise NotImplementedError


class TileLangFlashAttentionV100Impl(AttentionImpl):
    def __init__(
        self,
        num_heads: int,
        head_size: int,
        causal: bool,
        softmax_scale: float,
        num_kv_heads: int | None = None,
        prefix: str = "",
        **extra_impl_args,
    ) -> None:
        del num_kv_heads, prefix, extra_impl_args
        if causal:
            raise ValueError("MiniMax H3 TileLang attention must be non-causal")
        if head_size != 128:
            raise ValueError(
                f"TileLang V100 attention requires head dimension 128, got {head_size}"
            )
        self.num_heads = num_heads
        self.softmax_scale = softmax_scale

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_metadata: AttentionMetadata,
    ) -> torch.Tensor:
        del attn_metadata
        # This backend is selected only for H3's packed path. Keep a correct
        # dense implementation for diagnostics and direct module calls.
        return F.scaled_dot_product_attention(
            query.transpose(1, 2),
            key.transpose(1, 2),
            value.transpose(1, 2),
            scale=self.softmax_scale,
        ).transpose(1, 2)

    def forward_varlen(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        *,
        cu_seqlens: torch.Tensor,
        max_seqlen: int,
        cu_seqlens_host: tuple[int, ...] | None = None,
    ) -> torch.Tensor:
        del cu_seqlens_host
        return tilelang_flash_attn_varlen(
            query,
            key,
            value,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
            softmax_scale=self.softmax_scale,
        )


__all__ = [
    "TileLangFlashAttentionV100Backend",
    "TileLangFlashAttentionV100Impl",
    "tilelang_flash_attn_varlen",
]
