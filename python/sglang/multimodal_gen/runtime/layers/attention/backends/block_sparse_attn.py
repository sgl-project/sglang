from dataclasses import dataclass
from typing import Any

import attentions  # noqa: F401
import torch

from sglang.multimodal_gen.runtime.layers.attention.backends.attention_backend import (
    AttentionBackend,
    AttentionImpl,
    AttentionMetadata,
    AttentionMetadataBuilder,
)
from sglang.multimodal_gen.runtime.layers.attention.backends.laser_attn import (
    LaserAttentionBackend,
)
from sglang.multimodal_gen.runtime.platforms import AttentionBackendEnum
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

logger = init_logger(__name__)
BSA_BLOCK_SIZE = 128


class BlockSparseAttentionBackend(AttentionBackend):

    accept_output_buffer: bool = True

    @staticmethod
    def get_supported_head_sizes() -> list[int]:
        return [32, 64, 96, 128]

    @staticmethod
    def get_enum() -> AttentionBackendEnum:
        return AttentionBackendEnum.BLOCK_SPARSE_ATTN

    @staticmethod
    def get_impl_cls() -> type["BlockSparseAttentionImpl"]:
        return BlockSparseAttentionImpl

    @staticmethod
    def get_metadata_cls() -> type["BlockSparseAttentionMetadata"]:
        return BlockSparseAttentionMetadata

    @staticmethod
    def get_builder_cls() -> type["BlockSparseAttentionMetadataBuilder"]:
        return BlockSparseAttentionMetadataBuilder


@dataclass
class BlockSparseAttentionMetadata(AttentionMetadata):
    current_timestep: int
    skip_first_steps: int
    sparsity: float
    block_frame_stride: int


class BlockSparseAttentionMetadataBuilder(AttentionMetadataBuilder):
    def __init__(self) -> None:
        pass

    def prepare(self) -> None:
        pass

    def build(
        self,
        current_timestep: int,
        skip_first_steps: int,
        sparsity: float,
        raw_latent_shape: list[int],
        patch_size: tuple[int, int, int],
        **kwargs: dict[str, Any],
    ) -> BlockSparseAttentionMetadata:
        """
        Builds BlockSparseAttention metadata.

        Args:
            current_timestep: The current diffusion timestep.
            skip_first_steps: Number of initial timesteps to skip before applying
                sparsity. Must be non‑negative.
            sparsity: Fraction of tokens to drop (block‑wise) in the block sparse
                attention mechanism. Must be in the range [0.0, 1.0).
            raw_latent_shape: Shape of the latent tensor before patching.
            patch_size: Patch size as (T, height, width). Only the height
                and width components are used to divide the latent dimensions.
            **kwargs: Additional keyword arguments (ignored, but accepted for
                compatibility with base class or calling conventions).

        Returns:
            BlockSparseAttentionMetadata
        Note:
            The `block_frame_stride` is needed to set the first blocks to be non‑sparse.
        """
        if not (skip_first_steps >= 0 and 0.0 <= sparsity < 1.0):
            raise ValueError(
                (
                    "Invalid attention metadata values."
                    f"Sparsity should be in [0, 1), skip_first_steps should be non-negative."
                    f"Got sparsity={sparsity}, skip_first_steps={skip_first_steps}"
                )
            )

        if sparsity == 0.0:
            logger.warning(
                (
                    "Sparsity is set to 0.0, which means no tokens will be dropped."
                    "For better performance use Laser Attention or increase sparsity."
                )
            )

        if len(raw_latent_shape) >= 5:
            latent_height, latent_width = raw_latent_shape[3:5]
        else:
            latent_height, latent_width = raw_latent_shape[-2:]

        latent_height //= patch_size[1]
        latent_width //= patch_size[2]

        frame_stride = latent_height * latent_width
        block_frame_stride = (frame_stride + BSA_BLOCK_SIZE - 1) // BSA_BLOCK_SIZE

        return BlockSparseAttentionMetadata(
            current_timestep=current_timestep,
            skip_first_steps=skip_first_steps,
            sparsity=sparsity,
            block_frame_stride=block_frame_stride,
        )


class BlockSparseAttentionImpl(AttentionImpl):

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
        self.causal = causal
        self.softmax_scale = softmax_scale
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads or num_heads
        self.block_size = BSA_BLOCK_SIZE
        self.stride = 8
        self.default_tokens = 214748647

        self.laser_attn_impl = LaserAttentionBackend.get_impl_cls()(
            num_heads,
            head_size,
            causal,
            softmax_scale,
            num_kv_heads,
            prefix,
            **extra_impl_args,
        )

    def _get_estimate_mask(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        sparsity: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return torch.ops.attentions.sparse_block_estimate(
            query=query,
            key=key,
            actual_seq_lengths=None,
            actual_seq_lengths_kv=None,
            input_layout="BNSD",
            stride=self.stride,
            sparse_size=self.block_size,
            num_heads=query.shape[1],
            num_key_value_heads=key.shape[1],
            scale_value=self.softmax_scale / self.stride,
            threshold=1.0,
            causal=self.causal,
            keep_sink=True,
            keep_recent=True,
            row_sparse=1.0 - sparsity,
        )

    def _block_sparse_attention(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        smask: torch.Tensor,
        sct: torch.Tensor,
    ) -> torch.Tensor:
        return torch.ops.attentions.ada_block_sparse_attention(
            query=query,
            key=key,
            value=value,
            sparse_mask=smask,
            sparse_count_table=sct,
            input_layout="BNSD",
            sparse_size=self.block_size,
            num_heads=query.shape[1],
            num_key_value_heads=key.shape[1],
            scale_value=self.softmax_scale,
            causal=self.causal,
            inner_precise=1,
            pre_tokens=self.default_tokens,
            next_tokens=self.default_tokens,
            actual_seq_lengths=None,
            actual_seq_lengths_kv=None,
        )

    def _get_smask(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        block_frame_stride: int,
        sparsity: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        smask, sct = self._get_estimate_mask(
            query,
            key,
            sparsity,
        )

        seq_len = smask.shape[2]

        # Set the first blocks to be non-sparse to ensure the quality of the first few steps
        smask[:, :, :block_frame_stride, :seq_len] = 1
        smask[:, :, :seq_len, :block_frame_stride] = 1
        smask = smask.to(torch.int8)
        sct = smask.sum(dim=-1, dtype=torch.int32)
        return smask, sct

    def _adaptive_block_sparse_attention(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        block_frame_stride: int,
        sparsity: float,
    ) -> torch.Tensor:
        # TODO Currently implementation for BSND input layout has quality issues
        # When the implementation is improved, transposes can be removed
        q = query.permute(0, 2, 1, 3).contiguous()
        k = key.permute(0, 2, 1, 3).contiguous()
        v = value.permute(0, 2, 1, 3).contiguous()

        smask, sct = self._get_smask(
            q,
            k,
            block_frame_stride,
            sparsity,
        )
        output = self._block_sparse_attention(q, k, v, smask, sct)
        output = output.permute(0, 2, 1, 3).contiguous()

        return output

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_metadata: AttentionMetadata,
    ) -> torch.Tensor:
        if attn_metadata.current_timestep < attn_metadata.skip_first_steps:
            output = self.laser_attn_impl.forward(
                query,
                key,
                value,
                attn_metadata,
            )
        else:
            output = self._adaptive_block_sparse_attention(
                query,
                key,
                value,
                attn_metadata.block_frame_stride,
                attn_metadata.sparsity,
            )

        return output
