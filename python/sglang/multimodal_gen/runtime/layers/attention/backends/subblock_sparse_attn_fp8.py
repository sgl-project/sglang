# SPDX-License-Identifier: Apache-2.0
"""SM100 SubBlock Sage FP8 attention for MiniMax-H3.

Uses BF16 Q/K routing, uniform Q64 x K64 top-k, TRT-LLM E4M3 quantization
(Q per token/head, K per 16 tokens/head, V per head/channel), and no K
smoothing. The true sequence length is passed through without padding.
Select with transformer=subblock_sparse_attn_fp8; this is independent of
architecture-specific compute_mode choices on the general SubBlock backend.
"""

from __future__ import annotations

import functools

import flashinfer
import msgspec
import torch

from sglang.multimodal_gen.runtime.layers.attention.backends.attention_backend import (
    AttentionBackend,
    AttentionImpl,
)
from sglang.multimodal_gen.runtime.layers.attention.backends.subblock_sparse import (
    SubBlockRouter,
)
from sglang.multimodal_gen.runtime.layers.attention.backends.subblock_sparse_attn import (
    SubBlockSparseAttentionMetadata,
    SubBlockSparseAttentionMetadataBuilder,
    SubBlockSparseSchedule,
    _dit_layer_index,
)
from sglang.multimodal_gen.runtime.managers.forward_context import get_forward_context
from sglang.multimodal_gen.runtime.platforms import AttentionBackendEnum
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

logger = init_logger(__name__)


@functools.lru_cache(maxsize=1)
def load_sage_attention():
    from flashinfer.cute_dsl.sparse.bsa_attn_sm100_blk64 import (
        bsa_attn_sm100_blk64_fwd,
    )

    return bsa_attn_sm100_blk64_fwd


SUBBLOCK_SPARSE_HEAD_DIM = 128


@functools.lru_cache(maxsize=32)
def _quant_cu_seqlens(seq_len: int, device: torch.device) -> torch.Tensor:
    # Reuse immutable metadata instead of uploading two tiny tensors per layer.
    return torch.tensor([0, seq_len], dtype=torch.int32, device=device)


def _quantize_sage_fp8(
    q: torch.Tensor,  # [1, S, H, D] bf16 — S need NOT be a multiple of 64
    k: torch.Tensor,
    v: torch.Tensor,
) -> tuple[
    torch.Tensor,  # q_fp8   [1, S, H, D]  float8_e4m3fn
    torch.Tensor,  # k_fp8   [1, S, H, D]  float8_e4m3fn
    torch.Tensor,  # v_fp8   [1, S, H, D]  float8_e4m3fn
    torch.Tensor,  # q_scale [1, H, S]     float32
    torch.Tensor,  # k_scale [1, H, ceil(S/16)]  float32
    torch.Tensor,  # v_scale [H, D]        float32
]:
    """Sage-FP8 quantisation for the blk64 Sage-FP8 kernel, via flashinfer's
    fused ``trtllm_sage_attention_quantize`` kernel.

    Granularity follows the upstream kernel contract (see
    ``validate_sm100_blk64_fp8_sage`` in flashinfer):
      Q: one scale per (token, head).
      K: one scale per (16-token bucket, head).
      V: one scale per (head, dim) — global across the sequence.

    Quantizes and returns at the true (possibly non-64-aligned) sequence
    length -- the attention kernel handles the tail natively, so there is
    nothing to pad here.
    """
    # Plane-major Ulysses views have packed heads/channels and a strided
    # token dimension. Preserve these views to avoid three input copies.
    q_fp8, k_fp8, v_fp8, q_scale, k_scale, v_scale = (
        flashinfer.trtllm_sage_attention_quantize(
            q.squeeze(0),
            k.squeeze(0),
            v.squeeze(0),
            q_block_size=1,
            k_block_size=16,
            qk_quant_dtype=torch.float8_e4m3fn,
            cum_seq_lens_q=_quant_cu_seqlens(q.shape[1], q.device),
            cum_seq_lens_kv=_quant_cu_seqlens(k.shape[1], k.device),
            smooth_k=False,
            allow_strided_input=True,
        )
    )
    # q_fp8/k_fp8/v_fp8: [S, H, D]; q_scale: [H, S]; k_scale: [H, ceil(S/16)]; v_scale: [H, D]

    q_fp8 = q_fp8.unsqueeze(0)  # [1, S, H, D]
    k_fp8 = k_fp8.unsqueeze(0)
    v_fp8 = v_fp8.unsqueeze(0)
    q_scale = q_scale.unsqueeze(0)  # [1, H, S]
    k_scale = k_scale.unsqueeze(0)  # [1, H, ceil(S/16)]

    return q_fp8, k_fp8, v_fp8, q_scale, k_scale, v_scale


# ---------------------------------------------------------------------------
# Backend / metadata classes (thin wrappers reusing the bf16 counterparts)
# ---------------------------------------------------------------------------


class SubBlockSparseAttentionFP8Backend(AttentionBackend):
    @staticmethod
    def get_supported_head_sizes() -> list[int]:
        return [SUBBLOCK_SPARSE_HEAD_DIM]

    @staticmethod
    def get_enum() -> AttentionBackendEnum:
        return AttentionBackendEnum.SUBBLOCK_SPARSE_ATTN_FP8

    @staticmethod
    def get_impl_cls() -> type[SubBlockSparseAttentionFP8Impl]:
        return SubBlockSparseAttentionFP8Impl

    @staticmethod
    def get_metadata_cls() -> type[SubBlockSparseAttentionMetadata]:
        return SubBlockSparseAttentionMetadata

    @staticmethod
    def get_builder_cls() -> type[SubBlockSparseAttentionMetadataBuilder]:
        return SubBlockSparseAttentionMetadataBuilder


# ---------------------------------------------------------------------------
# Implementation
# ---------------------------------------------------------------------------


class SubBlockSparseAttentionFP8Impl(AttentionImpl):
    """VSA blk64 + Sage-FP8: router in bf16, attention kernel in FP8.

    Shares the ``SubBlockSparseSchedule`` config (sparsity, skip_first_steps,
    etc.) with the bf16 backend.  Dense fallback behaviour is identical.
    """

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        causal: bool = False,
        softmax_scale: float | None = None,
        num_kv_heads: int | None = None,
        prefix: str = "",
        **extra_impl_args,
    ) -> None:
        self.prefix = prefix
        self.num_heads = num_heads
        self.head_size = head_size
        self.causal = causal
        self.softmax_scale = (
            softmax_scale if softmax_scale is not None else head_size**-0.5
        )
        self.num_kv_heads = num_kv_heads if num_kv_heads is not None else num_heads

        self.schedule = SubBlockSparseSchedule.from_server_args()
        from sglang.multimodal_gen.runtime.server_args import get_global_server_args

        config = get_global_server_args().attention_backend_config or {}
        self.schedule = msgspec.structs.replace(
            self.schedule, n_k=int(config.get("n_k", 4)), compute_mode="sage_fp8"
        )
        self.layer_idx = _dit_layer_index(prefix)

        self.layer_enabled = (
            self.layer_idx is not None
            and self.layer_idx >= self.schedule.skip_first_layers
            and head_size == SUBBLOCK_SPARSE_HEAD_DIM
            and self.schedule.sparsity > 0.0
        )

        self.router = (
            SubBlockRouter(n_k=self.schedule.n_k, n_q=self.schedule.n_q)
            if self.layer_enabled
            else None
        )
        self.dense_impl = self._build_dense_impl(causal=causal)
        if self.layer_enabled:
            logger.info_once(
                f"SubBlock FP8 sparse attention [SM100 blk64 Sage-FP8]: "
                f"sparsity={self.schedule.sparsity:.3f} "
                f"n_k={self.schedule.n_k} n_q={self.schedule.n_q} "
                f"num_heads={num_heads}, dense for the first "
                f"{self.schedule.skip_first_steps} denoise steps and the first "
                f"{self.schedule.skip_first_layers} DiT layers"
            )

    def _build_dense_impl(self, *, causal: bool) -> AttentionImpl:
        from sglang.multimodal_gen.runtime.layers.attention.selector import (
            get_attn_backend,
        )

        backend = get_attn_backend(
            self.head_size,
            torch.bfloat16,
            supported_attention_backends={
                AttentionBackendEnum.FA,
                AttentionBackendEnum.TORCH_SDPA,
            },
            selected_attention_backend=AttentionBackendEnum.FA,
        )
        return backend.get_impl_cls()(
            num_heads=self.num_heads,
            head_size=self.head_size,
            causal=causal,
            softmax_scale=self.softmax_scale,
            num_kv_heads=self.num_kv_heads,
            prefix=f"{self.prefix}.dense",
        )

    def _step_enabled(self) -> bool:
        return get_forward_context().current_timestep >= self.schedule.skip_first_steps

    def _sparse_ready(self, q: torch.Tensor, k: torch.Tensor) -> bool:
        return (
            self.layer_enabled
            and self._step_enabled()
            and q.dtype == torch.bfloat16
            and (q.ndim == 3 or q.shape[0] == 1)
            and q.shape[-2] == k.shape[-2]
            and k.shape[-3] >= self.schedule.min_seq_len
            and not self.causal
        )

    def _sparse_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        *,
        out: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Route BF16 inputs and run Sage FP8 at the true sequence length."""
        plan = self.router.route(
            q, k, sparsity=self.schedule.sparsity, softmax_scale=self.softmax_scale
        )
        logger.info_once(
            f"SubBlock FP8 sparse attention active: S={k.shape[1]} heads={q.shape[2]} "
            f"keeping {plan.topk}/{plan.num_blocks} key blocks per query block "
            f"(sparsity {1 - plan.density:.4f})"
        )

        # Step 2: Sage-FP8 quantisation (at true length, no padding).
        q_fp8, k_fp8, v_fp8, q_scale, k_scale, v_scale = _quantize_sage_fp8(q, k, v)

        # Step 3: FP8 sparse attention.
        #   - block_sizes=None: the kernel masks a non-64-aligned tail natively.
        #   - q2k_block_nums=None: uniform topk (guaranteed by SubBlockRouter).
        #   - kv_splits="auto": uses sm100_blk64_auto_fp8_kv_splits heuristic.
        kernel = load_sage_attention()
        if out is None:
            out = torch.empty(q.shape, dtype=torch.bfloat16, device=q.device)
        out = kernel(
            q_fp8,
            k_fp8,
            v_fp8,
            plan.index,
            plan.topk,
            block_sizes=None,
            q2k_block_nums=None,
            softmax_scale=self.softmax_scale,
            kv_splits="auto",
            out=out,
            q_scale=q_scale,
            k_scale=k_scale,
            v_scale=v_scale,
        )
        return out[0] if isinstance(out, tuple) else out

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_metadata: SubBlockSparseAttentionMetadata | None = None,
    ) -> torch.Tensor:
        """query/key/value: ``[B, S, H, D]``."""
        if not self._sparse_ready(query, key):
            return self.dense_impl.forward(query, key, value, attn_metadata)
        return self._sparse_attention(query, key, value)

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
        """Packed ``[T, H, D]`` rows split into documents by ``cu_seqlens``."""

        def all_dense() -> torch.Tensor:
            return self.dense_impl.forward_varlen(
                query,
                key,
                value,
                cu_seqlens=cu_seqlens,
                max_seqlen=max_seqlen,
                cu_seqlens_host=cu_seqlens_host,
            )

        if cu_seqlens_host is None or not self._sparse_ready(query, key):
            return all_dense()

        segments = [
            (start, stop)
            for start, stop in zip(cu_seqlens_host[:-1], cu_seqlens_host[1:])
            if stop > start
        ]
        sparse_segments = {
            (start, stop)
            for start, stop in segments
            if stop - start >= self.schedule.min_seq_len
        }
        if not sparse_segments:
            return all_dense()

        if len(segments) == 1 and segments[0] == (0, query.shape[0]):
            # Return the single document directly, avoiding output assembly.
            start, stop = segments[0]
            q_seg = query.unsqueeze(0)
            k_seg = key.unsqueeze(0)
            v_seg = value.unsqueeze(0)
            seg_out = (
                self._sparse_attention(q_seg, k_seg, v_seg)
                if (start, stop) in sparse_segments
                else self._dense_segment(q_seg, k_seg, v_seg)
            )
            return seg_out[0]

        out = torch.empty(query.shape, dtype=query.dtype, device=query.device)
        if segments[-1][1] < query.shape[0]:
            out[segments[-1][1] :].zero_()
        for start, stop in segments:
            q_seg = query[start:stop].unsqueeze(0)
            k_seg = key[start:stop].unsqueeze(0)
            v_seg = value[start:stop].unsqueeze(0)
            if (start, stop) in sparse_segments:
                self._sparse_attention(
                    q_seg, k_seg, v_seg, out=out[start:stop].unsqueeze(0)
                )
            else:
                seg_out = self._dense_segment(q_seg, k_seg, v_seg)
                out[start:stop] = seg_out[0]
        return out

    def _dense_segment(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor
    ) -> torch.Tensor:
        """Dense attention over one packed document, ``[1, S, H, D]``."""
        return torch.nn.functional.scaled_dot_product_attention(
            q.transpose(1, 2),
            k.transpose(1, 2),
            v.transpose(1, 2),
            is_causal=self.causal,
            scale=self.softmax_scale,
        ).transpose(1, 2)


__all__ = [
    "SubBlockSparseAttentionFP8Backend",
    "SubBlockSparseAttentionFP8Impl",
]
