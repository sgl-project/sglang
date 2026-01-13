# SPDX-License-Identifier: Apache-2.0
# Adapted from turbo_layer.py for Attention Backend integration

from dataclasses import dataclass
from typing import Any, Dict

import torch
import torch.nn.functional as F

from sglang.multimodal_gen.runtime.layers.attention.backends.attention_backend import (
    AttentionBackend,
    AttentionImpl,
    AttentionMetadata,
    AttentionMetadataBuilder,
)
from sglang.multimodal_gen.runtime.platforms import AttentionBackendEnum

# Import from turbo_layer
from ..turbo_layer import _attention, get_block_map


class SparseLinearAttentionBackend(AttentionBackend):
    """Sparse Linear Attention Backend for efficient attention computation."""

    accept_output_buffer: bool = True

    @staticmethod
    def get_supported_head_sizes() -> list[int]:
        return [64, 128]

    @staticmethod
    def get_enum() -> AttentionBackendEnum:
        return AttentionBackendEnum.SPARSE_LINEAR_ATTN

    @staticmethod
    def get_impl_cls() -> type["SparseLinearAttentionImpl"]:
        return SparseLinearAttentionImpl

    @staticmethod
    def get_metadata_cls() -> type["SparseLinearAttentionMetadata"]:
        return SparseLinearAttentionMetadata

    @staticmethod
    def get_builder_cls() -> type["SparseLinearAttentionMetadataBuilder"]:
        return SparseLinearAttentionMetadataBuilder


@dataclass
class SparseLinearAttentionMetadata(AttentionMetadata):
    """Metadata for Sparse Linear Attention computation."""

    # Basic attention parameters
    current_timestep: int

    # Sparse attention configuration
    topk: float
    feature_map: str = "softmax"
    BLKQ: int = 64
    BLKK: int = 64
    use_bf16: bool = True
    tie_feature_map_qk: bool = True

    # Runtime computed values
    head_dim: int = 64
    real_topk_ratio: float = 0.1

    def asdict_zerocopy(self, skip_fields: set[str] | None = None) -> Dict[str, Any]:
        """Convert metadata to dict for zero-copy operations."""
        if skip_fields is None:
            skip_fields = set()
        return {
            field.name: getattr(self, field.name)
            for field in self.__dataclass_fields__.values()
            if field.name not in skip_fields
        }


class SparseLinearAttentionMetadataBuilder(AttentionMetadataBuilder):
    """Builder for SparseLinearAttentionMetadata."""

    def __init__(
        self,
        topk: float = 0.1,
        feature_map: str = "softmax",
        BLKQ: int = 64,
        BLKK: int = 64,
        use_bf16: bool = True,
        tie_feature_map_qk: bool = True,
    ) -> None:
        """Initialize the builder with configuration parameters.

        Args:
            topk: ratio of keys selected for sparse attention
            feature_map: feature map type ['hedgehog', 'elu', 'relu', 'softmax']
            BLKQ: block size for query
            BLKK: block size for key
            use_bf16: whether to use bfloat16
            tie_feature_map_qk: whether to use same feature map for q and k
        """
        self.topk = topk
        self.feature_map = feature_map
        self.BLKQ = BLKQ
        self.BLKK = BLKK
        self.use_bf16 = use_bf16
        self.tie_feature_map_qk = tie_feature_map_qk

    def prepare(self) -> None:
        """Prepare for one batch - no special preparation needed."""
        pass

    def build(
        self,
        current_timestep: int,
        head_dim: int,
        **kwargs: Dict[str, Any],
    ) -> SparseLinearAttentionMetadata:
        """Build sparse linear attention metadata.

        Args:
            current_timestep: current diffusion timestep
            head_dim: dimension of each attention head
            **kwargs: additional parameters

        Returns:
            SparseLinearAttentionMetadata instance
        """
        return SparseLinearAttentionMetadata(
            current_timestep=current_timestep,
            topk=self.topk,
            feature_map=self.feature_map,
            BLKQ=self.BLKQ,
            BLKK=self.BLKK,
            use_bf16=self.use_bf16,
            tie_feature_map_qk=self.tie_feature_map_qk,
            head_dim=head_dim,
            real_topk_ratio=self.topk,
        )


class SparseLinearAttentionImpl(AttentionImpl):
    """Implementation of sparse linear attention for the backend."""

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        softmax_scale: float,
        causal: bool = False,
        num_kv_heads: int | None = None,
        prefix: str = "",
        **extra_impl_args,
    ) -> None:
        """Initialize sparse linear attention implementation.

        Args:
            num_heads: number of attention heads
            head_size: dimension of each attention head
            softmax_scale: scaling factor for attention scores
            causal: whether to use causal attention
            num_kv_heads: number of key/value heads (for GQA)
            prefix: prefix for parameter names
            **extra_impl_args: additional implementation arguments
        """
        super().__init__(
            num_heads=num_heads,
            head_size=head_size,
            softmax_scale=softmax_scale,
            causal=causal,
            num_kv_heads=num_kv_heads,
            prefix=prefix,
            **extra_impl_args,
        )

        self.num_heads = num_heads
        self.head_size = head_size
        self.prefix = prefix

        # Initialize projection layer for linear attention
        self.proj_l = torch.nn.Linear(head_size, head_size, dtype=torch.float32)
        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize weights for the projection layer."""
        with torch.no_grad():
            torch.nn.init.zeros_(self.proj_l.weight)
            torch.nn.init.zeros_(self.proj_l.bias)

    def _get_feature_map(self, feature_map_type: str):
        """Get feature map function based on type."""
        if feature_map_type == "elu":

            def elu_feature_map(x):
                return F.elu(x) + 1

            return elu_feature_map
        elif feature_map_type == "relu":
            return torch.nn.ReLU()
        elif feature_map_type == "softmax":

            def softmax_feature_map(x):
                return F.softmax(x, dim=-1)

            return softmax_feature_map
        else:
            raise NotImplementedError(f"Not supported feature map {feature_map_type}.")

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_metadata: SparseLinearAttentionMetadata,
    ) -> torch.Tensor:
        """Forward pass for sparse linear attention.

        Args:
            query: query tensor of shape (B, H, L, D)
            key: key tensor of shape (B, H, L, D)
            value: value tensor of shape (B, H, L, D)
            attn_metadata: attention metadata containing configuration

        Returns:
            output tensor of shape (B, H, L, D)
        """
        dtype = query.dtype

        # Get configuration from metadata
        topk_ratio = attn_metadata.topk
        BLKQ = attn_metadata.BLKQ
        BLKK = attn_metadata.BLKK
        feature_map_type = attn_metadata.feature_map
        use_bf16 = attn_metadata.use_bf16
        tie_feature_map_qk = attn_metadata.tie_feature_map_qk

        # Determine computation dtype
        compute_dtype = torch.bfloat16 if use_bf16 else torch.float16

        # Transpose for computation
        q = query.transpose(1, 2).contiguous()
        k = key.transpose(1, 2).contiguous()
        v = value.transpose(1, 2).contiguous()

        # Get sparse attention map
        sparse_map, lut, real_topk = get_block_map(
            q, k, topk_ratio=topk_ratio, BLKQ=BLKQ, BLKK=BLKK
        )

        # Convert to computation dtype
        q = q.to(compute_dtype)
        k = k.to(compute_dtype)
        v = v.to(compute_dtype)

        # Sparse attention computation
        o_s = _attention.apply(q, k, v, sparse_map, lut, real_topk, BLKQ, BLKK)

        # Get feature maps
        feature_map_q = self._get_feature_map(feature_map_type)
        feature_map_k = (
            feature_map_q
            if tie_feature_map_qk
            else self._get_feature_map(feature_map_type)
        )

        # Apply feature maps
        q_feat = feature_map_q(q).contiguous().to(compute_dtype)
        k_feat = feature_map_k(k).contiguous().to(compute_dtype)

        # Linear attention computation
        def calc_linear(q_linear, k_linear, v_linear):
            kvsum = k_linear.transpose(-1, -2) @ v_linear
            ksum = torch.sum(k_linear, dim=-2, keepdim=True)
            return (q_linear @ kvsum) / (
                1e-5 + (q_linear * ksum).sum(dim=-1, keepdim=True)
            )

        o_l = calc_linear(q_feat, k_feat, v)

        # Apply projection and combine results
        with torch.amp.autocast("cuda", dtype=compute_dtype):
            o_l = self.proj_l(o_l)

        # Combine sparse and linear attention
        output = (o_s + o_l).to(dtype).transpose(1, 2)

        return output
