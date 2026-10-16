"""NaViT vision encoder for Muse 1.2 Omni."""

from __future__ import annotations

import functools
import math
from copy import deepcopy
from typing import Sequence

import torch
import torch.nn.functional as F
from torch import nn

from sglang.srt.configs.muse_spark import MuseSparkVisionConfig
from sglang.srt.layers.attention.vision import (
    VisionAttention,
    VisionAttentionMetadata,
    prepare_vision_attention_metadata,
)
from sglang.srt.layers.linear import (
    ColumnParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
)
from sglang.srt.runtime_context import get_parallel
from sglang.srt.utils import add_prefix, is_cpu


def pad_navit_for_unaligned_cpu_tp(
    config: MuseSparkVisionConfig, tp_size: int
) -> MuseSparkVisionConfig:
    """Pad NaViT dimensions for unaligned CPU tensor parallelism.

    VisionAttention sizes its projections for ``num_dummy_heads + num_heads`` and
    shards that, so head_dim stays derived from the real head count.
    """
    from sglang.srt.configs.update_config import DEFAULT_MOE_PADDING_SIZE
    from sglang.srt.layers.vocab_parallel_embedding import pad_vocab_size

    heads = config.num_attention_heads
    config.num_dummy_heads = (
        0 if heads % tp_size == 0 else (heads // tp_size + 1) * tp_size - heads
    )
    # The MLP is column/row parallel, so widen it too. CPU linear loaders narrow
    # padded parameters against the unpadded checkpoint tensors and zero-fill
    # the padded tails.
    block = tp_size * DEFAULT_MOE_PADDING_SIZE
    if config.intermediate_size % tp_size:
        config.intermediate_size = pad_vocab_size(config.intermediate_size, block)
    return config


def _rope_frequencies(
    head_dim: int,
    grid_h: int,
    grid_w: int,
    theta: float,
    device: torch.device,
) -> torch.Tensor:
    if head_dim % 4:
        raise ValueError("NaViT attention head size must be divisible by four")
    half_dim = head_dim // 2
    exponents = torch.arange(0, half_dim, 2, dtype=torch.float32, device=device)
    inverse = theta ** (-exponents / half_dim)
    pos_h = torch.arange(1, grid_h + 1, dtype=torch.float32, device=device)
    pos_w = torch.arange(1, grid_w + 1, dtype=torch.float32, device=device)
    mesh_h, mesh_w = torch.meshgrid(pos_h, pos_w, indexing="ij")
    freq_h = torch.einsum("hw,d->hwd", mesh_h, inverse)
    freq_w = torch.einsum("hw,d->hwd", mesh_w, inverse)
    angles = torch.cat((freq_w, freq_h), dim=-1)
    return torch.stack((angles.cos(), angles.sin()), dim=-1)


def _patch_grid_positions(
    grid_h: int,
    grid_w: int,
    device: torch.device,
) -> torch.Tensor:
    """Return fp32 grid-sample coordinates for the checkpoint's xy (non-ij) layout."""
    pos_h = torch.linspace(
        -1 + 1 / grid_h,
        1 - 1 / grid_h,
        grid_h,
        device=device,
        dtype=torch.float32,
    )
    pos_w = torch.linspace(
        -1 + 1 / grid_w,
        1 - 1 / grid_w,
        grid_w,
        device=device,
        dtype=torch.float32,
    )
    return torch.stack(torch.meshgrid(pos_h, pos_w, indexing="xy"), dim=-1).reshape(
        -1, 2
    )


def _apply_rope(
    query: torch.Tensor,
    key: torch.Tensor,
    frequencies: torch.Tensor,
    x_shape=None,
) -> tuple[torch.Tensor, torch.Tensor]:
    cosine = frequencies[..., 0].unsqueeze(1)
    sine = frequencies[..., 1].unsqueeze(1)

    def rotate(value: torch.Tensor) -> torch.Tensor:
        pairs = value.float().reshape(*value.shape[:-1], -1, 2)
        first, second = pairs.unbind(dim=-1)
        rotated = torch.stack(
            (first * cosine - second * sine, first * sine + second * cosine),
            dim=-1,
        ).flatten(-2)
        return rotated.to(value.dtype)

    return rotate(query), rotate(key)


@functools.lru_cache(maxsize=64)
def _cached_rope_frequencies(
    head_dim: int,
    grid_h: int,
    grid_w: int,
    theta: float,
    device: torch.device,
) -> torch.Tensor:
    return _rope_frequencies(head_dim, grid_h, grid_w, theta, device).reshape(
        -1, head_dim // 2, 2
    )


@functools.lru_cache(maxsize=64)
def _cached_grid_positions(
    grid_h: int, grid_w: int, device: torch.device
) -> torch.Tensor:
    return _patch_grid_positions(grid_h, grid_w, device)


def _pixel_shuffle_permutation(
    offset: int, grid_h: int, grid_w: int, factor: int, device: torch.device
) -> tuple[torch.Tensor, int]:
    if grid_h % factor or grid_w % factor:
        raise ValueError(
            "preprocessed image grid is not divisible by downsample factor"
        )
    order = torch.arange(offset, offset + grid_h * grid_w, device=device)
    order = (
        order.reshape(grid_h // factor, factor, grid_w // factor, factor)
        .permute(0, 2, 1, 3)
        .reshape(-1)
    )
    return order, offset + grid_h * grid_w


def _sparse_permutation(
    offset: int,
    grid_h: int,
    grid_w: int,
    block_h: int,
    block_w: int,
    device: torch.device,
) -> tuple[torch.Tensor, list[int], int]:
    padded_h = math.ceil(grid_h / block_h) * block_h
    padded_w = math.ceil(grid_w / block_w) * block_w
    order = torch.arange(offset, offset + grid_h * grid_w, device=device).reshape(
        grid_h, grid_w
    )
    order = F.pad(order, (0, padded_w - grid_w, 0, padded_h - grid_h), value=-1)
    rows, cols = padded_h // block_h, padded_w // block_w
    order = (
        order.reshape(rows, block_h, cols, block_w)
        .permute(0, 2, 1, 3)
        .reshape(rows * cols, block_h * block_w)
    )
    lengths = (order != -1).sum(dim=1).tolist()
    order = order[order != -1]
    return order, lengths, offset + grid_h * grid_w


@functools.lru_cache(maxsize=64)
def _cached_pixel_shuffle_permutation(
    grid_h: int, grid_w: int, factor: int, device: torch.device
) -> torch.Tensor:
    order, _ = _pixel_shuffle_permutation(0, grid_h, grid_w, factor, device)
    return order


@functools.lru_cache(maxsize=64)
def _cached_sparse_permutation(
    grid_h: int,
    grid_w: int,
    block_h: int,
    block_w: int,
    device: torch.device,
) -> tuple[torch.Tensor, tuple[int, ...]]:
    order, lengths, _ = _sparse_permutation(0, grid_h, grid_w, block_h, block_w, device)
    return order, tuple(lengths)


def _build_vision_attention_metadata(
    sequence_lengths: Sequence[int], device: torch.device
) -> VisionAttentionMetadata:
    cu_seqlens = torch.zeros(len(sequence_lengths) + 1, dtype=torch.int32)
    if sequence_lengths:
        cu_seqlens[1:] = torch.tensor(sequence_lengths, dtype=torch.int32).cumsum(0)
    return prepare_vision_attention_metadata(
        cu_seqlens,
        device,
        max_seqlen=max(sequence_lengths) if sequence_lengths else 0,
    )


class NaViTMLP(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        prefix: str = "",
        use_data_parallel: bool = False,
    ) -> None:
        super().__init__()
        if use_data_parallel:
            self.fc1 = ReplicatedLinear(
                hidden_size,
                intermediate_size,
                bias=True,
                prefix=add_prefix("fc1", prefix),
            )
            self.fc2 = ReplicatedLinear(
                intermediate_size,
                hidden_size,
                bias=True,
                prefix=add_prefix("fc2", prefix),
            )
        else:
            self.fc1 = ColumnParallelLinear(
                input_size=hidden_size,
                output_size=intermediate_size,
                bias=True,
                prefix=add_prefix("fc1", prefix),
            )
            self.fc2 = RowParallelLinear(
                input_size=intermediate_size,
                output_size=hidden_size,
                bias=True,
                prefix=add_prefix("fc2", prefix),
            )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        value, _ = self.fc1(value)
        value, _ = self.fc2(F.gelu(value))
        return value


class NaViTBlock(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        num_heads: int,
        use_rope: bool,
        prefix: str = "",
        use_data_parallel: bool = False,
        num_dummy_heads: int = 0,
        norm_eps: float = 1e-5,
    ) -> None:
        super().__init__()
        if hidden_size % num_heads:
            raise ValueError("vision hidden size must be divisible by attention heads")
        self.norm1 = nn.LayerNorm(hidden_size, eps=norm_eps)
        self.attn = VisionAttention(
            embed_dim=hidden_size,
            num_heads=num_heads,
            num_dummy_heads=num_dummy_heads,
            head_dim=hidden_size // num_heads,
            projection_size=hidden_size,
            use_qkv_parallel=True,
            qkv_bias=True,
            proj_bias=True,
            softmax_scale=(hidden_size // num_heads) ** -0.5,
            flatten_batch=True,
            customized_position_embedding_applier=(_apply_rope if use_rope else None),
            use_data_parallel=use_data_parallel,
            prefix=add_prefix("attn", prefix),
        )
        self.use_rope = use_rope
        self.norm2 = nn.LayerNorm(hidden_size, eps=norm_eps)
        self.mlp = NaViTMLP(
            hidden_size,
            intermediate_size,
            add_prefix("mlp", prefix),
            use_data_parallel=use_data_parallel,
        )

    def forward(
        self,
        value: torch.Tensor,
        frequencies: torch.Tensor,
        metadata: VisionAttentionMetadata,
    ) -> torch.Tensor:
        attn_output = self.attn(
            self.norm1(value),
            cu_seqlens=metadata.cu_seqlens,
            position_embeddings=frequencies if self.use_rope else None,
            forward_metadata=metadata,
            max_seqlen=metadata.max_seqlen,
        )
        value = value + attn_output.squeeze(0)
        return value + self.mlp(self.norm2(value))


class NaViTAdapter(nn.Module):
    def __init__(
        self,
        input_size: int,
        output_size: int,
        prefix: str = "",
        use_data_parallel: bool = False,
    ) -> None:
        super().__init__()
        if use_data_parallel:
            self.fc1 = ReplicatedLinear(
                input_size,
                output_size,
                bias=False,
                prefix=add_prefix("fc1", prefix),
            )
            self.fc2 = ReplicatedLinear(
                output_size,
                output_size,
                bias=False,
                prefix=add_prefix("fc2", prefix),
            )
        else:
            self.fc1 = ColumnParallelLinear(
                input_size=input_size,
                output_size=output_size,
                bias=False,
                prefix=add_prefix("fc1", prefix),
            )
            self.fc2 = RowParallelLinear(
                input_size=output_size,
                output_size=output_size,
                bias=False,
                prefix=add_prefix("fc2", prefix),
            )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        value, _ = self.fc1(value)
        value, _ = self.fc2(F.gelu(value))
        return F.gelu(value)


class NaViT(nn.Module):
    """Variable-resolution image/video encoder using only public PyTorch ops."""

    def __init__(
        self,
        config: MuseSparkVisionConfig,
        prefix: str = "",
        use_data_parallel: bool = False,
    ) -> None:
        super().__init__()
        config = deepcopy(config)
        if is_cpu() and not use_data_parallel:
            config = pad_navit_for_unaligned_cpu_tp(config, get_parallel().tp_size)
        self.config = config
        self.patch_height = config.patch_size
        self.patch_width = config.patch_size
        self.patch_temporal = config.patch_temporal
        self.downsample_factor = config.merge_size
        self.downsample_strategy = "ps"
        self.has_window_attention = "window_attention" in config.layer_types
        self.grid_sample_mode = "bilinear"
        self.use_cls_token = False

        if config.hidden_act != "gelu" or config.rope_parameters["rope_type"] not in (
            "default",
            "axial",
        ):
            raise ValueError("Muse vision supports gelu and axial RoPE only")
        if len(config.layer_types) != config.num_hidden_layers or set(
            config.layer_types
        ) - {"window_attention", "full_attention"}:
            raise ValueError("Invalid Muse vision layer_types")
        latent_size = config.hidden_size
        self.hidden_size = latent_size
        self.output_size = latent_size * config.merge_size**2
        self.head_dim = latent_size // config.num_attention_heads
        self.pos_grid_h = config.pos_emb_height
        self.pos_grid_w = config.pos_emb_width

        patch_input_size = (
            3 * config.patch_size * config.patch_size * config.patch_temporal
        )
        self.patch_embedder = nn.Module()
        if use_data_parallel:
            self.patch_embedder.patch_embedding = ReplicatedLinear(
                patch_input_size,
                latent_size,
                bias=False,
                prefix=add_prefix("patch_embedder.patch_embedding", prefix),
            )
        else:
            self.patch_embedder.patch_embedding = ColumnParallelLinear(
                input_size=patch_input_size,
                output_size=latent_size,
                bias=False,
                gather_output=True,
                prefix=add_prefix("patch_embedder.patch_embedding", prefix),
            )
        position_count = self.pos_grid_h * self.pos_grid_w + int(self.use_cls_token)
        self.patch_embedder.position_embedding_table = nn.Embedding(
            position_count, latent_size
        )
        nn.init.normal_(
            self.patch_embedder.position_embedding_table.weight, std=latent_size**-0.5
        )
        if self.use_cls_token:
            raise ValueError("Muse 1.2 checkpoints do not use a NaViT CLS token")
        self.register_parameter("class_embedding", None)
        self.ln_pre = nn.LayerNorm(latent_size, eps=config.layer_norm_eps)
        self.ln_post = nn.LayerNorm(latent_size, eps=config.layer_norm_eps)
        self.layers = nn.ModuleList(
            NaViTBlock(
                config.hidden_size,
                config.intermediate_size,
                config.num_attention_heads,
                True,
                add_prefix(f"layers.{i}", prefix),
                use_data_parallel=use_data_parallel,
                num_dummy_heads=getattr(config, "num_dummy_heads", 0),
                norm_eps=config.layer_norm_eps,
            )
            for i in range(config.num_hidden_layers)
        )

    def _split_visions(self, visions: Sequence[torch.Tensor]) -> list[torch.Tensor]:
        result = []
        for vision in visions:
            if vision.ndim == 3:
                result.append(vision)
            elif vision.ndim == 4:
                if vision.shape[0] % self.patch_temporal:
                    raise ValueError("video frame count must align to patch_temporal")
                result.extend(
                    vision[index : index + self.patch_temporal]
                    for index in range(0, vision.shape[0], self.patch_temporal)
                )
            else:
                raise ValueError(
                    f"unexpected vision tensor shape {tuple(vision.shape)}"
                )
        return result

    def forward(self, visions: Sequence[torch.Tensor]) -> torch.Tensor:
        clips = self._split_visions(visions)
        if not clips:
            return self.patch_embedder.position_embedding_table.weight.new_empty(
                (0, self.output_size)
            )

        device = clips[0].device
        patches_list: list[torch.Tensor] = []
        positions_list: list[torch.Tensor] = []
        frequencies_list: list[torch.Tensor] = []
        full_sequence_lengths: list[int] = []
        sparse_sequence_lengths: list[int] = []
        downsample_orders: list[torch.Tensor] = []
        sparse_orders: list[torch.Tensor] = []
        class_indices: list[int] = []
        downsample_offset = 0
        sparse_offset = 0

        for clip in clips:
            if clip.ndim == 3:
                clip = clip.unsqueeze(0)
            if clip.shape[0] == 1:
                clip = clip.expand(self.patch_temporal, -1, -1, -1)
            if clip.shape[0] != self.patch_temporal:
                raise ValueError("each NaViT clip must contain patch_temporal frames")
            _, channels, height, width = clip.shape
            if height % self.patch_height or width % self.patch_width:
                raise ValueError("vision inputs must be aligned to the patch size")
            grid_h = height // self.patch_height
            grid_w = width // self.patch_width
            patches = (
                clip.reshape(
                    self.patch_temporal,
                    channels,
                    grid_h,
                    self.patch_height,
                    grid_w,
                    self.patch_width,
                )
                .permute(2, 4, 0, 1, 3, 5)
                .reshape(grid_h * grid_w, -1)
            )
            positions = _cached_grid_positions(grid_h, grid_w, device)
            frequencies = _cached_rope_frequencies(
                self.head_dim,
                grid_h,
                grid_w,
                self.config.rope_parameters["rope_theta"],
                device,
            )

            downsample_order = (
                _cached_pixel_shuffle_permutation(
                    grid_h, grid_w, self.downsample_factor, device
                )
                + downsample_offset
            )
            downsample_offset += grid_h * grid_w
            downsample_orders.append(downsample_order)

            if self.has_window_attention:
                sparse_order, sparse_lengths = _cached_sparse_permutation(
                    grid_h,
                    grid_w,
                    self.pos_grid_h,
                    self.pos_grid_w,
                    device,
                )
                sparse_order = sparse_order + sparse_offset
                sparse_offset += grid_h * grid_w
                sparse_orders.append(sparse_order)
                sparse_sequence_lengths.extend(sparse_lengths)
            else:
                sparse_sequence_lengths.append(grid_h * grid_w)

            sequence_length = grid_h * grid_w
            patches_list.append(patches)
            positions_list.append(positions)
            frequencies_list.append(frequencies)
            if self.use_cls_token:
                class_indices.append(sum(full_sequence_lengths) + sequence_length)
                patches_list.append(patches.new_zeros(1, patches.shape[1]))
                positions_list.append(positions.new_zeros(1, 2))
                frequencies_list.append(
                    torch.stack(
                        (
                            torch.ones(self.head_dim // 2, device=device),
                            torch.zeros(self.head_dim // 2, device=device),
                        ),
                        dim=-1,
                    ).unsqueeze(0)
                )
                sequence_length += 1
                sparse_sequence_lengths.append(1)
            full_sequence_lengths.append(sequence_length)

        patches = torch.cat(patches_list)
        positions = torch.cat(positions_list)
        frequencies = torch.cat(frequencies_list)
        downsample_order = torch.cat(downsample_orders)
        value, _ = self.patch_embedder.patch_embedding(patches)

        position_weights = self.patch_embedder.position_embedding_table.weight
        class_position = None
        if self.use_cls_token:
            class_position = position_weights[-1]
            position_weights = position_weights[:-1]
        position_grid = (
            position_weights.float()
            .reshape(self.pos_grid_h, self.pos_grid_w, -1)
            .permute(2, 0, 1)
            .unsqueeze(0)
        )
        interpolated = F.grid_sample(
            position_grid,
            positions[None, None],
            mode=self.grid_sample_mode,
            align_corners=False,
        )[0, :, 0].transpose(0, 1)
        value = value + interpolated.to(value.dtype)
        if self.use_cls_token:
            indices = torch.tensor(class_indices, device=device)
            value[indices] = self.class_embedding + class_position
        value = self.ln_pre(value)

        full_metadata = _build_vision_attention_metadata(full_sequence_lengths, device)
        if self.has_window_attention:
            sparse_order = torch.cat(sparse_orders)
            value = value[sparse_order]
            frequencies = frequencies[sparse_order]
            sparse_metadata = _build_vision_attention_metadata(
                sparse_sequence_lengths, device
            )
        else:
            sparse_order = None
            sparse_metadata = full_metadata
        for index, layer in enumerate(self.layers):
            use_full = self.config.layer_types[index] == "full_attention"
            value = layer(
                value, frequencies, full_metadata if use_full else sparse_metadata
            )
        if sparse_order is not None:
            restored = torch.empty_like(value)
            restored.index_copy_(0, sparse_order, value)
            value = restored
        value = self.ln_post(value)
        if self.use_cls_token:
            keep = torch.ones(value.shape[0], dtype=torch.bool, device=device)
            keep[torch.tensor(class_indices, device=device)] = False
            value = value[keep]

        value = value[downsample_order]
        factor_squared = self.downsample_factor**2
        value = value.reshape(-1, factor_squared, value.shape[-1])
        value = value.transpose(1, 2).reshape(value.shape[0], -1)
        return value


def get_navit_patch_count(
    img_h: int,
    img_w: int,
    patch_h: int,
    patch_w: int,
    downsample_factor: int,
) -> int:
    return (img_h // patch_h) * (img_w // patch_w) // (downsample_factor**2)


def get_navit_video_patch_count(
    video_t: int,
    video_h: int,
    video_w: int,
    patch_t: int,
    patch_h: int,
    patch_w: int,
    downsample_factor: int,
) -> int:
    return (video_t // patch_t) * get_navit_patch_count(
        video_h, video_w, patch_h, patch_w, downsample_factor
    )


__all__ = [
    "NaViT",
    "get_navit_patch_count",
    "get_navit_video_patch_count",
]
