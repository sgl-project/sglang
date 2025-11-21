# Copied and adapted from: https://github.com/hao-ai-lab/FastVideo

# SPDX-License-Identifier: Apache-2.0

import re
from dataclasses import dataclass

import torch
from einops import rearrange
from kernel.attn.vmoba_attn.vmoba import (
    moba_attn_varlen,
    process_moba_input,
    process_moba_output,
)

from sglang.multimodal_gen.runtime.layers.attention.backends.attention_backend import (
    AttentionBackend,
    AttentionImpl,
    AttentionMetadata,
    AttentionMetadataBuilder,
)
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

logger = init_logger(__name__)


class VMOBAAttentionBackend(AttentionBackend):

    accept_output_buffer: bool = True

    @staticmethod
    def get_name() -> str:
        return "VMOBA_ATTN"

    @staticmethod
    def get_impl_cls() -> type["VMOBAAttentionImpl"]:
        return VMOBAAttentionImpl

    @staticmethod
    def get_metadata_cls() -> type["VideoMobaAttentionMetadata"]:
        return VideoMobaAttentionMetadata

    @staticmethod
    def get_builder_cls() -> type["VideoMobaAttentionMetadataBuilder"]:
        return VideoMobaAttentionMetadataBuilder


@dataclass
class VideoMobaAttentionMetadata(AttentionMetadata):
    current_timestep: int

    temporal_chunk_size: int
    temporal_topk: int
    spatial_chunk_size: tuple[int, int]
    spatial_topk: int
    st_chunk_size: tuple[int, int, int]
    st_topk: int

    moba_select_mode: str
    moba_threshold: float
    moba_threshold_type: str
    patch_resolution: list[int]

    first_full_step: int = 12
    first_full_layer: int = 0
    # temporal_layer -> spatial_layer -> st_layer
    temporal_layer: int = 1
    spatial_layer: int = 1
    st_layer: int = 1


def pad_input(hidden_states, indices, batch, seqlen):
    """
    Arguments:
        hidden_states: (total_nnz, ...), where total_nnz = number of tokens in selected in attention_mask.
        indices: (total_nnz), the indices that represent the non-masked tokens of the original padded input sequence.
        batch: int, batch size for the padded sequence.
        seqlen: int, maximum sequence length for the padded sequence.
    Return:
        hidden_states: (batch, seqlen, ...)
    """
    dim = hidden_states.shape[1:]
    output = torch.zeros(
        (batch * seqlen), *dim, device=hidden_states.device, dtype=hidden_states.dtype
    )
    output[indices] = hidden_states
    return rearrange(output, "(b s) ... -> b s ...", b=batch)


class VideoMobaAttentionMetadataBuilder(AttentionMetadataBuilder):

    def __init__(self):
        pass

    def prepare(self):
        pass

    def build(  # type: ignore
        self,
        current_timestep: int,
        raw_latent_shape: tuple[int, int, int],
        patch_size: tuple[int, int, int],
        temporal_chunk_size: int,
        temporal_topk: int,
        spatial_chunk_size: tuple[int, int],
        spatial_topk: int,
        st_chunk_size: tuple[int, int, int],
        st_topk: int,
        moba_select_mode: str = "threshold",
        moba_threshold: float = 0.25,
        moba_threshold_type: str = "query_head",
        device: torch.device = None,
        first_full_layer: int = 0,
        first_full_step: int = 12,
        temporal_layer: int = 1,
        spatial_layer: int = 1,
        st_layer: int = 1,
        **kwargs,
    ) -> VideoMobaAttentionMetadata:
        if device is None:
            device = torch.device("cpu")
        assert (
            raw_latent_shape[0] % patch_size[0] == 0
            and raw_latent_shape[1] % patch_size[1] == 0
            and raw_latent_shape[2] % patch_size[2] == 0
        ), f"spatial patch_resolution {raw_latent_shape} should be divisible by patch_size {patch_size}"
        patch_resolution = [
            t // pt for t, pt in zip(raw_latent_shape, patch_size, strict=False)
        ]

        return VideoMobaAttentionMetadata(
            current_timestep=current_timestep,
            temporal_chunk_size=temporal_chunk_size,
            temporal_topk=temporal_topk,
            spatial_chunk_size=spatial_chunk_size,
            spatial_topk=spatial_topk,
            st_chunk_size=st_chunk_size,
            st_topk=st_topk,
            moba_select_mode=moba_select_mode,
            moba_threshold=moba_threshold,
            moba_threshold_type=moba_threshold_type,
            patch_resolution=patch_resolution,
            first_full_layer=first_full_layer,
            first_full_step=first_full_step,
            temporal_layer=temporal_layer,
            spatial_layer=spatial_layer,
            st_layer=st_layer,
        )


class VMOBAAttentionImpl(AttentionImpl):

    def __init__(
        self,
        num_heads,
        head_size,
        softmax_scale,
        causal=False,
        num_kv_heads=None,
        prefix="",
        **extra_impl_args,
    ) -> None:
        self.prefix = prefix
        self.layer_idx = self._get_layer_idx(prefix)

        self.pad_input = pad_input

    def _get_layer_idx(self, prefix: str) -> int | None:
        match = re.search(r"blocks\.(\d+)", prefix)
        if not match:
            raise ValueError(f"Invalid prefix: {prefix}")
        return int(match.group(1))

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_metadata: AttentionMetadata,
    ) -> torch.Tensor:
        """
        query: [B, L, H, D]
        key:   [B, L, H, D]
        value: [B, L, H, D]
        attn_metadata: AttentionMetadata
        """
        batch_size, sequence_length, num_heads, head_dim = query.shape

        # select chunk type according to layer idx:
        loop_layer_num = (
            attn_metadata.temporal_layer
            + attn_metadata.spatial_layer
            + attn_metadata.st_layer
        )
        moba_layer = self.layer_idx - attn_metadata.first_full_layer
        if moba_layer % loop_layer_num < attn_metadata.temporal_layer:
            moba_chunk_size = attn_metadata.temporal_chunk_size
            moba_topk = attn_metadata.temporal_topk
        elif (
            moba_layer % loop_layer_num
            < attn_metadata.temporal_layer + attn_metadata.spatial_layer
        ):
            moba_chunk_size = attn_metadata.spatial_chunk_size
            moba_topk = attn_metadata.spatial_topk
        elif (
            moba_layer % loop_layer_num
            < attn_metadata.temporal_layer
            + attn_metadata.spatial_layer
            + attn_metadata.st_layer
        ):
            moba_chunk_size = attn_metadata.st_chunk_size
            moba_topk = attn_metadata.st_topk

        query, chunk_size = process_moba_input(
            query, attn_metadata.patch_resolution, moba_chunk_size
        )
        key, chunk_size = process_moba_input(
            key, attn_metadata.patch_resolution, moba_chunk_size
        )
        value, chunk_size = process_moba_input(
            value, attn_metadata.patch_resolution, moba_chunk_size
        )
        max_seqlen = query.shape[1]
        indices_q = torch.arange(
            0, query.shape[0] * query.shape[1], device=query.device
        )
        cu_seqlens = torch.arange(
            0,
            query.shape[0] * query.shape[1] + 1,
            query.shape[1],
            dtype=torch.int32,
            device=query.device,
        )
        query = rearrange(query, "b s ... -> (b s) ...")
        key = rearrange(key, "b s ... -> (b s) ...")
        value = rearrange(value, "b s ... -> (b s) ...")

        # current_timestep=attn_metadata.current_timestep
        hidden_states = moba_attn_varlen(
            query,
            key,
            value,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
            moba_chunk_size=chunk_size,
            moba_topk=moba_topk,
            select_mode=attn_metadata.moba_select_mode,
            simsum_threshold=attn_metadata.moba_threshold,
            threshold_type=attn_metadata.moba_threshold_type,
        )
        hidden_states = self.pad_input(
            hidden_states, indices_q, batch_size, sequence_length
        )
        hidden_states = process_moba_output(
            hidden_states, attn_metadata.patch_resolution, moba_chunk_size
        )

        return hidden_states
