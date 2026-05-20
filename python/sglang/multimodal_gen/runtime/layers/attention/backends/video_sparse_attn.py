# Copied and adapted from: https://github.com/hao-ai-lab/FastVideo

# SPDX-License-Identifier: Apache-2.0
import functools
import math
from dataclasses import dataclass

import torch

try:
    from vsa import torch_attention, video_sparse_attn

    from vsa.block_sparse_attn_triton import triton_block_sparse_attn_forward
except ImportError:
    torch_attention = None
    triton_block_sparse_attn_forward = None
    video_sparse_attn = None

from collections.abc import Callable

from typing import Any

from sglang.multimodal_gen.runtime.distributed import get_sp_group
from sglang.multimodal_gen.runtime.layers.attention.backends.attention_backend import (
    AttentionBackend,
    AttentionImpl,
    AttentionMetadata,
    AttentionMetadataBuilder,
)
from sglang.multimodal_gen.runtime.platforms import AttentionBackendEnum
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

logger = init_logger(__name__)
VSA_TILE_SIZE = (4, 4, 4)


@functools.lru_cache(maxsize=10)
def get_tile_partition_indices(
    dit_seq_shape: tuple[int, int, int],
    tile_size: tuple[int, int, int],
    device: torch.device,
) -> torch.LongTensor:
    T, H, W = dit_seq_shape
    ts, hs, ws = tile_size
    indices = torch.arange(T * H * W, device=device, dtype=torch.long).reshape(T, H, W)
    ls = []
    for t in range(math.ceil(T / ts)):
        for h in range(math.ceil(H / hs)):
            for w in range(math.ceil(W / ws)):
                ls.append(
                    indices[
                        t * ts : min(t * ts + ts, T),
                        h * hs : min(h * hs + hs, H),
                        w * ws : min(w * ws + ws, W),
                    ].flatten()
                )
    index = torch.cat(ls, dim=0)
    return index


@functools.lru_cache(maxsize=10)
def get_reverse_tile_partition_indices(
    dit_seq_shape: tuple[int, int, int],
    tile_size: tuple[int, int, int],
    device: torch.device,
) -> torch.LongTensor:
    return torch.argsort(get_tile_partition_indices(dit_seq_shape, tile_size, device))


@functools.lru_cache(maxsize=10)
def construct_variable_block_sizes(
    dit_seq_shape: tuple[int, int, int],
    num_tiles: tuple[int, int, int],
    device: torch.device,
) -> torch.LongTensor:
    """
    Compute the number of valid (non‑padded) tokens inside every
    (ts_t × ts_h × ts_w) tile after padding ‑‑ flattened in the order
    (t‑tile, h‑tile, w‑tile) that `rearrange` uses.

    Returns
    -------
    torch.LongTensor  # shape: [∏ full_window_size]
    """
    # unpack
    t, h, w = dit_seq_shape
    ts_t, ts_h, ts_w = VSA_TILE_SIZE
    n_t, n_h, n_w = num_tiles

    def _sizes(dim_len: int, tile: int, n_tiles: int) -> torch.LongTensor:
        """Vector with the size of each tile along one dimension."""
        sizes = torch.full((n_tiles,), tile, dtype=torch.int, device=device)
        # size of last (possibly partial) tile
        remainder = dim_len - (n_tiles - 1) * tile
        sizes[-1] = remainder if remainder > 0 else tile
        return sizes

    t_sizes = _sizes(t, ts_t, n_t)  # [n_t]
    h_sizes = _sizes(h, ts_h, n_h)  # [n_h]
    w_sizes = _sizes(w, ts_w, n_w)  # [n_w]

    # broadcast‑multiply to get voxels per tile, then flatten
    block_sizes = (
        t_sizes[:, None, None]  # [n_t, 1,   1]
        * h_sizes[None, :, None]  # [1,   n_h, 1]
        * w_sizes[None, None, :]  # [1,   1,   n_w]
    ).reshape(
        -1
    )  # [n_t * n_h * n_w]

    return block_sizes


@functools.lru_cache(maxsize=10)
def get_non_pad_index(
    variable_block_sizes: torch.LongTensor,
    max_block_size: int,
):
    n_win = variable_block_sizes.shape[0]
    device = variable_block_sizes.device
    starts_pad = torch.arange(n_win, device=device) * max_block_size
    index_pad = (
        starts_pad[:, None] + torch.arange(max_block_size, device=device)[None, :]
    )
    index_mask = (
        torch.arange(max_block_size, device=device)[None, :]
        < variable_block_sizes[:, None]
    )
    return index_pad[index_mask]


def _use_index_native_vsa(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    gate_compress: torch.Tensor | None,
) -> bool:
    if torch_attention is None or triton_block_sparse_attn_forward is None:
        return False
    if not torch.is_grad_enabled():
        return True
    return not any(
        tensor is not None and tensor.requires_grad
        for tensor in (query, key, value, gate_compress)
    )


def _video_sparse_attn_index_native(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    variable_block_sizes: torch.Tensor,
    topk: int,
    block_size: int | tuple[int, int, int],
    compress_attn_weight: torch.Tensor | None = None,
) -> torch.Tensor:
    if isinstance(block_size, int):
        block_size = (block_size, block_size, block_size)

    block_elements = math.prod(block_size)
    assert block_elements == math.prod(VSA_TILE_SIZE)
    assert q.shape[2] % block_elements == 0

    batch_size, num_heads, seq_len, head_dim = q.shape
    num_blocks = seq_len // block_elements
    block_sizes = variable_block_sizes.view(1, 1, -1, 1)
    q_compress = (
        q.view(batch_size, num_heads, num_blocks, block_elements, head_dim)
        .float()
        .sum(dim=3)
        / block_sizes
    ).to(q.dtype)
    k_compress = (
        k.view(batch_size, num_heads, num_blocks, block_elements, head_dim)
        .float()
        .sum(dim=3)
        / block_sizes
    ).to(k.dtype)
    v_compress = (
        v.view(batch_size, num_heads, num_blocks, block_elements, head_dim)
        .float()
        .sum(dim=3)
        / block_sizes
    ).to(v.dtype)

    output_compress, block_attn_score = torch_attention(q_compress, k_compress, v_compress)
    output_compress = (
        output_compress.view(batch_size, num_heads, num_blocks, 1, head_dim)
        .repeat(1, 1, 1, block_elements, 1)
        .view(batch_size, num_heads, seq_len, head_dim)
    )

    q2k_idx = torch.topk(block_attn_score, topk, dim=-1).indices.to(torch.int32)
    q2k_idx = q2k_idx.contiguous()
    q2k_num = torch.full(
        q2k_idx.shape[:-1], topk, dtype=torch.int32, device=q2k_idx.device
    )
    output_select, _ = triton_block_sparse_attn_forward(
        q.contiguous(),
        k.contiguous(),
        v.contiguous(),
        q2k_idx,
        q2k_num,
        variable_block_sizes.to(torch.int32).contiguous(),
    )

    if compress_attn_weight is not None:
        return output_compress * compress_attn_weight + output_select
    return output_compress + output_select


class VideoSparseAttentionBackend(AttentionBackend):

    accept_output_buffer: bool = True

    @staticmethod
    def get_supported_head_sizes() -> list[int]:
        return [64, 128]

    @staticmethod
    def get_enum() -> AttentionBackendEnum:
        return AttentionBackendEnum.VIDEO_SPARSE_ATTN

    @staticmethod
    def get_impl_cls() -> type["VideoSparseAttentionImpl"]:
        return VideoSparseAttentionImpl

    @staticmethod
    def get_metadata_cls() -> type["VideoSparseAttentionMetadata"]:
        return VideoSparseAttentionMetadata

    @staticmethod
    def get_builder_cls() -> type["VideoSparseAttentionMetadataBuilder"]:
        return VideoSparseAttentionMetadataBuilder


@dataclass
class VideoSparseAttentionMetadata(AttentionMetadata):
    current_timestep: int
    dit_seq_shape: list[int]
    VSA_sparsity: float
    num_tiles: list[int]
    total_seq_length: int
    tile_partition_indices: torch.LongTensor
    reverse_tile_partition_indices: torch.LongTensor
    variable_block_sizes: torch.LongTensor
    non_pad_index: torch.LongTensor

    # adaption for FastWan2.1-T2V-1.3B-Diffusers
    # Sequence lengths for the forward batch
    # Maximum sequence length for query
    max_seqlen_q: int = 1
    # Maximum sequence length for key
    max_seqlen_k: int = 0


class VideoSparseAttentionMetadataBuilder(AttentionMetadataBuilder):

    def __init__(self):
        pass

    def prepare(self):
        pass

    def build(  # type: ignore
        self,
        current_timestep: int,
        raw_latent_shape: tuple[int, int, int],
        patch_size: tuple[int, int, int],
        VSA_sparsity: float,
        device: torch.device,
        **kwargs: dict[str, Any],
    ) -> VideoSparseAttentionMetadata:
        patch_size = patch_size
        dit_seq_shape = (
            raw_latent_shape[0] // patch_size[0],
            raw_latent_shape[1] // patch_size[1],
            raw_latent_shape[2] // patch_size[2],
        )

        num_tiles = (
            math.ceil(dit_seq_shape[0] / VSA_TILE_SIZE[0]),
            math.ceil(dit_seq_shape[1] / VSA_TILE_SIZE[1]),
            math.ceil(dit_seq_shape[2] / VSA_TILE_SIZE[2]),
        )
        total_seq_length = math.prod(dit_seq_shape)

        tile_partition_indices = get_tile_partition_indices(
            dit_seq_shape, VSA_TILE_SIZE, device
        )
        reverse_tile_partition_indices = get_reverse_tile_partition_indices(
            dit_seq_shape, VSA_TILE_SIZE, device
        )
        variable_block_sizes = construct_variable_block_sizes(
            dit_seq_shape, num_tiles, device
        )
        non_pad_index = get_non_pad_index(
            variable_block_sizes, math.prod(VSA_TILE_SIZE)
        )

        return VideoSparseAttentionMetadata(
            current_timestep=current_timestep,
            dit_seq_shape=dit_seq_shape,  # type: ignore
            VSA_sparsity=VSA_sparsity,  # type: ignore
            num_tiles=num_tiles,  # type: ignore
            total_seq_length=total_seq_length,  # type: ignore
            tile_partition_indices=tile_partition_indices,  # type: ignore
            reverse_tile_partition_indices=reverse_tile_partition_indices,
            variable_block_sizes=variable_block_sizes,
            non_pad_index=non_pad_index,
        )


class VideoSparseAttentionImpl(AttentionImpl):

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
        self.prefix = prefix
        sp_group = get_sp_group()
        self.sp_size = sp_group.world_size

    def tile(
        self,
        x: torch.Tensor,
        num_tiles: list[int],
        tile_partition_indices: torch.LongTensor,
        non_pad_index: torch.LongTensor,
    ) -> torch.Tensor:
        t_padded_size = num_tiles[0] * VSA_TILE_SIZE[0]
        h_padded_size = num_tiles[1] * VSA_TILE_SIZE[1]
        w_padded_size = num_tiles[2] * VSA_TILE_SIZE[2]

        x_padded = torch.zeros(
            (
                x.shape[0],
                t_padded_size * h_padded_size * w_padded_size,
                x.shape[-2],
                x.shape[-1],
            ),
            device=x.device,
            dtype=x.dtype,
        )
        x_padded[:, non_pad_index] = x[:, tile_partition_indices]
        return x_padded

    def untile(
        self,
        x: torch.Tensor,
        reverse_tile_partition_indices: torch.LongTensor,
        non_pad_index: torch.LongTensor,
    ) -> torch.Tensor:
        x = x[:, non_pad_index][:, reverse_tile_partition_indices]
        return x

    def preprocess_qkv(
        self,
        qkv: torch.Tensor,
        attn_metadata: VideoSparseAttentionMetadata,
    ) -> torch.Tensor:
        return self.tile(
            qkv,
            attn_metadata.num_tiles,
            attn_metadata.tile_partition_indices,
            attn_metadata.non_pad_index,
        )

    def postprocess_output(
        self,
        output: torch.Tensor,
        attn_metadata: VideoSparseAttentionMetadata,
    ) -> torch.Tensor:
        return self.untile(
            output,
            attn_metadata.reverse_tile_partition_indices,
            attn_metadata.non_pad_index,
        )

    def forward(  # type: ignore[override]
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        gate_compress: torch.Tensor,
        attn_metadata: VideoSparseAttentionMetadata,
    ) -> torch.Tensor:
        query = query.transpose(1, 2).contiguous()
        key = key.transpose(1, 2).contiguous()
        value = value.transpose(1, 2).contiguous()
        gate_compress = gate_compress.transpose(1, 2).contiguous()

        VSA_sparsity = attn_metadata.VSA_sparsity

        cur_topk = math.ceil(
            (1 - VSA_sparsity)
            * (attn_metadata.total_seq_length / math.prod(VSA_TILE_SIZE))
        )

        if video_sparse_attn is None:
            raise NotImplementedError("video_sparse_attn is not installed")
        attn_fn: Callable[..., torch.Tensor]
        if _use_index_native_vsa(query, key, value, gate_compress):
            attn_fn = _video_sparse_attn_index_native
        else:
            attn_fn = video_sparse_attn
        hidden_states = attn_fn(
            query,
            key,
            value,
            variable_block_sizes=attn_metadata.variable_block_sizes,
            topk=cur_topk,
            block_size=VSA_TILE_SIZE,
            compress_attn_weight=gate_compress,
        ).transpose(1, 2)

        return hidden_states
