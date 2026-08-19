# Copied and adapted from: https://github.com/hao-ai-lab/FastVideo

# SPDX-License-Identifier: Apache-2.0
import functools
import math
from dataclasses import dataclass

import torch

try:
    from vsa import video_sparse_attn
except ImportError:
    video_sparse_attn = None

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
    ).reshape(-1)  # [n_t * n_h * n_w]

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
    untile_combined_index: torch.LongTensor
    tile_buf: torch.Tensor | None = None

    # adaption for FastWan2.1-T2V-1.3B-Diffusers
    # Sequence lengths for the forward batch
    # Maximum sequence length for query
    max_seqlen_q: int = 1
    # Maximum sequence length for key
    max_seqlen_k: int = 0


def _compute_cur_topk(attn_metadata: VideoSparseAttentionMetadata) -> int:
    num_kv_blocks = attn_metadata.variable_block_sizes.numel()
    cur_topk = math.ceil((1 - attn_metadata.VSA_sparsity) * num_kv_blocks)
    return max(1, min(cur_topk, num_kv_blocks))


def _compressed_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    variable_block_sizes: torch.Tensor,
    topk: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run the VSA compressed branch and return its direct top-k metadata."""
    batch_size, num_heads, seq_len, head_dim = query.shape
    block_elements = math.prod(VSA_TILE_SIZE)
    denominator = variable_block_sizes.view(1, 1, -1, 1)
    compressed = [
        (
            tensor.view(
                batch_size,
                num_heads,
                seq_len // block_elements,
                block_elements,
                head_dim,
            )
            .float()
            .sum(dim=3)
            / denominator
        ).to(tensor.dtype)
        for tensor in (query, key, value)
    ]
    query_compress, key_compress, value_compress = compressed
    scores = torch.matmul(query_compress, key_compress.transpose(-2, -1))
    scores /= math.sqrt(head_dim)
    probabilities = torch.softmax(scores, dim=-1)
    output = torch.matmul(probabilities, value_compress)
    output = (
        output.view(batch_size, num_heads, -1, 1, head_dim)
        .repeat(1, 1, 1, block_elements, 1)
        .view(batch_size, num_heads, seq_len, head_dim)
    )
    topk_indices = torch.topk(probabilities, topk, dim=-1).indices
    return output, topk_indices


def _create_cake_wrapper(device: torch.device):
    from flashinfer.sparse import BlockSparseAttentionWrapper

    # Cake consumes direct q2k metadata and does not use FlashInfer's generic
    # sparse-planner workspace. One wrapper is created per transformer layer.
    workspace = torch.empty((0,), dtype=torch.uint8, device=device)
    return BlockSparseAttentionWrapper(workspace, backend="cake")


def _validate_cake_inputs(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    gate_compress: torch.Tensor,
) -> None:
    if query.device.type != "cuda":
        raise ValueError("Cake VSA requires CUDA inputs")
    if torch.cuda.get_device_capability(query.device) not in ((10, 0), (10, 3)):
        raise ValueError("Cake VSA requires SM100 or SM103")
    if not (query.shape == key.shape == value.shape == gate_compress.shape):
        raise ValueError("Cake VSA requires matching Q/K/V/gate shapes")
    if query.shape[0] != 1:
        raise ValueError("Cake VSA currently requires batch size 1")
    if query.shape[-1] != 128:
        raise ValueError("Cake VSA currently requires head size 128")
    if not (query.dtype == key.dtype == value.dtype == torch.bfloat16):
        raise ValueError("Cake VSA currently requires BF16 Q/K/V")


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
        untile_combined_index = non_pad_index[reverse_tile_partition_indices]

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
            untile_combined_index=untile_combined_index,
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
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads if num_kv_heads is not None else num_heads
        self.head_size = head_size
        self.softmax_scale = softmax_scale
        sp_group = get_sp_group()
        self.sp_size = sp_group.world_size
        from sglang.multimodal_gen.runtime.server_args import get_global_server_args

        config = get_global_server_args().attention_backend_config or {}
        self.stage2_backend = str(config.get("stage2_backend", "vsa")).lower()
        if self.stage2_backend not in ("vsa", "cake"):
            raise ValueError(
                "video sparse attention stage2_backend must be 'vsa' or 'cake', "
                f"got {self.stage2_backend!r}"
            )
        self._cake_wrapper = None
        self._cake_q2k_num: dict[tuple[int, ...], torch.Tensor] = {}

    def _forward_cake(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        gate_compress: torch.Tensor,
        attn_metadata: VideoSparseAttentionMetadata,
        cur_topk: int,
    ) -> torch.Tensor:
        _validate_cake_inputs(query, key, value, gate_compress)
        query_hsd, key_hsd, value_hsd, gate_hsd = [
            tensor.transpose(1, 2).contiguous()
            for tensor in (query, key, value, gate_compress)
        ]
        output_compress, topk_indices = _compressed_attention(
            query_hsd,
            key_hsd,
            value_hsd,
            attn_metadata.variable_block_sizes,
            cur_topk,
        )

        q2k_indices = topk_indices[0].to(torch.int32).contiguous()
        q2k_shape = tuple(q2k_indices.shape)
        q2k_num = self._cake_q2k_num.get(q2k_shape)
        if q2k_num is None or q2k_num.device != query.device:
            q2k_num = torch.full(
                q2k_shape[:2],
                q2k_shape[2],
                dtype=torch.int32,
                device=query.device,
            )
            self._cake_q2k_num[q2k_shape] = q2k_num

        if self._cake_wrapper is None:
            self._cake_wrapper = _create_cake_wrapper(query.device)
        sequence = query.shape[1]
        num_heads = query.shape[2]
        self._cake_wrapper.plan(
            None,
            None,
            sequence,
            sequence,
            math.prod(VSA_TILE_SIZE),
            math.prod(VSA_TILE_SIZE),
            num_heads,
            num_heads,
            self.head_size,
            q_data_type=query.dtype,
            kv_data_type=key.dtype,
            o_data_type=query.dtype,
            sm_scale=self.softmax_scale,
            kv_block_lens=attn_metadata.variable_block_sizes,
            q2k_indices=q2k_indices,
            q2k_num=q2k_num,
        )
        output_select = self._cake_wrapper.run(query[0], key[0], value[0])
        output_select_hsd = output_select.transpose(0, 1).unsqueeze(0)
        return (output_compress * gate_hsd + output_select_hsd).transpose(1, 2)

    def tile(
        self,
        x: torch.Tensor,
        attn_metadata: VideoSparseAttentionMetadata,
    ) -> torch.Tensor:
        num_tiles = attn_metadata.num_tiles
        t_padded_size = num_tiles[0] * VSA_TILE_SIZE[0]
        h_padded_size = num_tiles[1] * VSA_TILE_SIZE[1]
        w_padded_size = num_tiles[2] * VSA_TILE_SIZE[2]
        target_shape = (
            x.shape[0],
            t_padded_size * h_padded_size * w_padded_size,
            x.shape[-2],
            x.shape[-1],
        )

        buf = attn_metadata.tile_buf
        if (
            buf is None
            or buf.shape != target_shape
            or buf.dtype != x.dtype
            or buf.device != x.device
        ):
            buf = torch.zeros(target_shape, device=x.device, dtype=x.dtype)
            attn_metadata.tile_buf = buf

        buf[:, attn_metadata.non_pad_index] = x[:, attn_metadata.tile_partition_indices]
        return buf

    def untile(
        self,
        x: torch.Tensor,
        untile_combined_index: torch.LongTensor,
    ) -> torch.Tensor:
        return x[:, untile_combined_index]

    def preprocess_qkv(
        self,
        qkv: torch.Tensor,
        attn_metadata: VideoSparseAttentionMetadata,
    ) -> torch.Tensor:
        return self.tile(qkv, attn_metadata)

    def postprocess_output(
        self,
        output: torch.Tensor,
        attn_metadata: VideoSparseAttentionMetadata,
    ) -> torch.Tensor:
        return self.untile(output, attn_metadata.untile_combined_index)

    def forward(  # type: ignore[override]
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        gate_compress: torch.Tensor,
        attn_metadata: VideoSparseAttentionMetadata,
    ) -> torch.Tensor:
        cur_topk = _compute_cur_topk(attn_metadata)
        if self.stage2_backend == "cake":
            return self._forward_cake(
                query,
                key,
                value,
                gate_compress,
                attn_metadata,
                cur_topk,
            )

        query = query.transpose(1, 2).contiguous()
        key = key.transpose(1, 2).contiguous()
        value = value.transpose(1, 2).contiguous()
        gate_compress = gate_compress.transpose(1, 2).contiguous()

        if video_sparse_attn is None:
            raise NotImplementedError("video_sparse_attn is not installed")
        hidden_states = video_sparse_attn(
            query,
            key,
            value,
            variable_block_sizes=attn_metadata.variable_block_sizes,
            topk=cur_topk,
            block_size=VSA_TILE_SIZE,
            compress_attn_weight=gate_compress,
        ).transpose(1, 2)

        return hidden_states
