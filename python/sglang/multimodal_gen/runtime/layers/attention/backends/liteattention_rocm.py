# Copied and adapted from: https://github.com/hao-ai-lab/FastVideo

# SPDX-License-Identifier: Apache-2.0

import logging

import torch

from sglang.multimodal_gen.runtime.layers.attention.backends.attention_backend import (
    AttentionBackend,
    AttentionImpl,
    AttentionMetadata,
    AttentionMetadataBuilder,
)
from sglang.multimodal_gen.runtime.managers.forward_context import get_forward_context
from sglang.multimodal_gen.runtime.platforms import AttentionBackendEnum
from sglang.srt.utils import is_gfx942_supported

logger = logging.getLogger(__name__)


class LiteAttentionROCMBackend(AttentionBackend):
    """
    Backend for LiteAttention (moonmath) implementation on AMD MI300X (gfx942).

    This backend wraps the hand-tuned moonmath_attention kernel which has strict
    constraints:
    - bf16 inputs/outputs
    - head_dim == 128
    - No causal masking
    - No GQA (query/key/value must have same shape)
    - gfx942 / MI300X only (CDNA3)
    """

    accept_output_buffer: bool = False

    @staticmethod
    def get_supported_head_sizes() -> list[int]:
        return [128]

    @staticmethod
    def get_supported_dtypes() -> list[torch.dtype]:
        return [torch.bfloat16]

    @staticmethod
    def get_supported_round_modes() -> list[str]:
        return ["rtna", "rtne", "rtz"]

    @staticmethod
    def get_enum() -> AttentionBackendEnum:
        return AttentionBackendEnum.LITEATTENTION_ROCM

    @staticmethod
    def get_impl_cls() -> type["LiteAttentionROCMImpl"]:
        return LiteAttentionROCMImpl

    @staticmethod
    def get_metadata_cls() -> type["AttentionMetadata"]:
        return AttentionMetadata

    @staticmethod
    def get_builder_cls() -> type["AttentionMetadataBuilder"]:
        raise NotImplementedError(
            "LiteAttentionROCM backend does not have a metadata builder."
        )


class LiteAttentionROCMImpl(AttentionImpl):
    """
    Implementation of attention using moonmath_attention kernel.

    The moonmath kernel is a hand-tuned bf16 forward attention kernel for AMD
    MI300X with specific constraints. This implementation validates all inputs
    against these constraints and raises NotImplementedError for unsupported cases.
    """

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

        try:
            from moonmath_attention import LiteAttention
            from moonmath_attention import forward as moonmath_forward
        except ImportError as e:
            raise ImportError(
                "LiteAttentionROCM backend requires moonmath_attention package, which is "
                "only available on AMD MI300X (gfx942). Install it from the "
                "moonmath_attention repository or use a different attention backend."
            ) from e

        self.num_heads = num_heads
        self.head_size = head_size
        self.causal = causal
        self.softmax_scale = softmax_scale
        self.num_kv_heads = num_kv_heads if num_kv_heads is not None else num_heads
        self.prefix = prefix

        # Validate head_size at construction time
        if self.head_size not in LiteAttentionROCMBackend.get_supported_head_sizes():
            raise NotImplementedError(
                f"LiteAttentionROCM backend only supports head_dim in "
                f"{LiteAttentionROCMBackend.get_supported_head_sizes()}, got {self.head_size}. "
                f"Use a different attention backend for this model."
            )

        # Validate hardware: moonmath kernels target AMD CDNA3 (gfx942 / MI300X) only.
        if not is_gfx942_supported():
            raise NotImplementedError(
                "LiteAttentionROCM backend requires AMD gfx942 (MI300X / MI325X)."
            )

        # Validate causal at construction time
        if self.causal:
            raise NotImplementedError(
                "LiteAttentionROCM backend does not support causal attention. "
                "Use a different attention backend for causal models."
            )

        # Validate GQA at construction time
        if self.num_kv_heads != self.num_heads:
            raise NotImplementedError(
                f"LiteAttentionROCM backend does not support GQA (num_heads={self.num_heads}, "
                f"num_kv_heads={self.num_kv_heads}). Use a different attention backend."
            )

        self._round_mode = extra_impl_args.get("lite_round_mode", "rtz")
        if self._round_mode not in LiteAttentionROCMBackend.get_supported_round_modes():
            raise NotImplementedError(
                f"LiteAttentionROCM backend only supports round_mode in "
                f"{LiteAttentionROCMBackend.get_supported_round_modes()}, got {self._round_mode!r}."
            )
        self._threshold = float(extra_impl_args.get("lite_threshold", -6.0))
        if self._threshold >= 0:
            raise NotImplementedError(
                f"LiteAttentionROCM backend requires threshold < 0 (log2 units), "
                f"got {self._threshold}."
            )

        self._lite = LiteAttention(
            threshold=self._threshold, round_mode=self._round_mode, layout="bshd"
        )
        self._moonmath_forward = moonmath_forward
        self._last_timestep: int = 0
        logger.info(
            f"LiteAttentionROCMImpl initialized with round_mode: {self._round_mode}, threshold: {self._threshold}"
        )

        # Validate softmax scale (moonmath bakes in 1/sqrt(D))
        expected_scale = self.head_size**-0.5
        if abs(self.softmax_scale - expected_scale) > 1e-6:
            raise NotImplementedError(
                f"LiteAttentionROCM backend requires softmax_scale == 1/sqrt(head_dim) = {expected_scale}, "
                f"got {self.softmax_scale}. Use a different attention backend."
            )

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_metadata: AttentionMetadata,
    ) -> torch.Tensor:
        """
        Forward pass using liteattention ROCM kernel.

        Args:
            query: Query tensor of shape [batch_size, seq_len, num_heads, head_dim]
            key: Key tensor of shape [batch_size, seq_len, num_heads, head_dim]
            value: Value tensor of shape [batch_size, seq_len, num_heads, head_dim]
            attn_metadata: Metadata for the attention operation (unused by moonmath).

        Returns:
            Output tensor of shape [batch_size, seq_len, num_heads, head_dim]
        """
        # Validate dtype
        if query.dtype not in LiteAttentionROCMBackend.get_supported_dtypes():
            raise NotImplementedError(
                f"LiteAttentionROCM backend only supports dtypes "
                f"{LiteAttentionROCMBackend.get_supported_dtypes()}, got {query.dtype}. "
                f"Use a different attention backend or cast inputs to a supported dtype."
            )

        # Validate 4D tensors
        if query.dim() != 4:
            raise ValueError(
                f"LiteAttentionROCM backend expects 4D tensors [B, S, H, D], got {query.dim()}D"
            )

        # moonmath kernels require contiguous inputs; enforce it at the boundary.
        query = query.contiguous()
        key = key.contiguous()
        value = value.contiguous()

        # Cross-attention uses the exact forward; self/joint-attention uses the
        # skip-optimized kernel.
        if self._is_cross_attention(query, key):
            output = self._moonmath_forward(
                query, key, value, round_mode=self._round_mode, layout="bshd"
            )
        else:
            # Reset cross-request skip-state at the start of each generation.
            self._maybe_reset_skip_state()
            output = self._lite(query, key, value)

        return output

    @staticmethod
    def _is_cross_attention(query: torch.Tensor, key: torch.Tensor) -> bool:
        # k from a different sequence than q (Sq != Skv) => cross-attention.
        return query.shape[1] != key.shape[1]

    def _maybe_reset_skip_state(self) -> None:
        try:
            ctx = get_forward_context()
        except AssertionError:
            return
        ts = ctx.current_timestep
        if ts == 0 and self._last_timestep != 0:
            self._lite.reset_skip_state()
        self._last_timestep = ts
