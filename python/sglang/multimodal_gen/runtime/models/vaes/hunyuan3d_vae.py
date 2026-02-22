# Copied and adapted from: https://github.com/Tencent-Hunyuan/Hunyuan3D-2


from __future__ import annotations

from typing import Callable, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from tqdm import tqdm

from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

logger = init_logger(__name__)

# Attention backend selection
scaled_dot_product_attention = F.scaled_dot_product_attention


class CrossAttentionProcessor:
    def __call__(self, attn, q, k, v):
        out = scaled_dot_product_attention(q, k, v)
        return out


class FlashVDMCrossAttentionProcessor:
    def __init__(self, topk=None):
        self.topk = topk

    def __call__(self, attn, q, k, v):
        if k.shape[-2] == 3072:
            topk = 1024
        elif k.shape[-2] == 512:
            topk = 256
        else:
            topk = k.shape[-2] // 3

        if self.topk is True:
            q1 = q[:, :, ::100, :]
            sim = q1 @ k.transpose(-1, -2)
            sim = torch.mean(sim, -2)
            topk_ind = torch.topk(sim, dim=-1, k=topk).indices.squeeze(-2).unsqueeze(-1)
            topk_ind = topk_ind.expand(-1, -1, -1, v.shape[-1])
            v0 = torch.gather(v, dim=-2, index=topk_ind)
            k0 = torch.gather(k, dim=-2, index=topk_ind)
            out = scaled_dot_product_attention(q, k0, v0)
        elif self.topk is False:
            out = scaled_dot_product_attention(q, k, v)
        else:
            idx, counts = self.topk
            start = 0
            outs = []
            for grid_coord, count in zip(idx, counts):
                end = start + count
                q_chunk = q[:, :, start:end, :]
                k0, v0 = self.select_topkv(q_chunk, k, v, topk)
                out = scaled_dot_product_attention(q_chunk, k0, v0)
                outs.append(out)
                start += count
            out = torch.cat(outs, dim=-2)
        self.topk = False
        return out

    def select_topkv(self, q_chunk, k, v, topk):
        q1 = q_chunk[:, :, ::50, :]
        sim = q1 @ k.transpose(-1, -2)
        sim = torch.mean(sim, -2)
        topk_ind = torch.topk(sim, dim=-1, k=topk).indices.squeeze(-2).unsqueeze(-1)
        topk_ind = topk_ind.expand(-1, -1, -1, v.shape[-1])
        v0 = torch.gather(v, dim=-2, index=topk_ind)
        k0 = torch.gather(k, dim=-2, index=topk_ind)
        return k0, v0


class FlashVDMTopMCrossAttentionProcessor(FlashVDMCrossAttentionProcessor):
    def select_topkv(self, q_chunk, k, v, topk):
        q1 = q_chunk[:, :, ::30, :]
        sim = q1 @ k.transpose(-1, -2)
        # sim = sim.to(torch.float32)
        sim = sim.softmax(-1)
        sim = torch.mean(sim, 1)
        activated_token = torch.where(sim > 1e-6)[2]
        index = (
            torch.unique(activated_token, return_counts=True)[0]
            .unsqueeze(0)
            .unsqueeze(0)
            .unsqueeze(-1)
        )
        index = index.expand(-1, v.shape[1], -1, v.shape[-1])
        v0 = torch.gather(v, dim=-2, index=index)
        k0 = torch.gather(k, dim=-2, index=index)
        return k0, v0


class FourierEmbedder(nn.Module):
    def __init__(
        self,
        num_freqs: int = 6,
        logspace: bool = True,
        input_dim: int = 3,
        include_input: bool = True,
        include_pi: bool = True,
    ) -> None:
        """The initialization"""

        super().__init__()

        if logspace:
            frequencies = 2.0 ** torch.arange(num_freqs, dtype=torch.float32)
        else:
            frequencies = torch.linspace(
                1.0, 2.0 ** (num_freqs - 1), num_freqs, dtype=torch.float32
            )

        if include_pi:
            frequencies *= torch.pi

        self.register_buffer("frequencies", frequencies, persistent=False)
        self.include_input = include_input
        self.num_freqs = num_freqs

        self.out_dim = self.get_dims(input_dim)

    def get_dims(self, input_dim):
        temp = 1 if self.include_input or self.num_freqs == 0 else 0
        out_dim = input_dim * (self.num_freqs * 2 + temp)

        return out_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward process."""

        if self.num_freqs > 0:
            embed = (x[..., None].contiguous() * self.frequencies).view(
                *x.shape[:-1], -1
            )
            if self.include_input:
                return torch.cat((x, embed.sin(), embed.cos()), dim=-1)
            else:
                return torch.cat((embed.sin(), embed.cos()), dim=-1)
        else:
            return x


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks)."""

    def __init__(self, drop_prob: float = 0.0, scale_by_keep: bool = True):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def forward(self, x):
        """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks)."""
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (
            x.ndim - 1
        )  # work with diff dim tensors, not just 2D ConvNets
        random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
        if keep_prob > 0.0 and self.scale_by_keep:
            random_tensor.div_(keep_prob)
        return x * random_tensor

    def extra_repr(self):
        return f"drop_prob={round(self.drop_prob, 3):0.3f}"


class MLP(nn.Module):
    def __init__(
        self,
        *,
        width: int,
        expand_ratio: int = 4,
        output_width: int = None,
        drop_path_rate: float = 0.0,
    ):
        super().__init__()
        self.width = width
        self.c_fc = nn.Linear(width, width * expand_ratio)
        self.c_proj = nn.Linear(
            width * expand_ratio, output_width if output_width is not None else width
        )
        self.gelu = nn.GELU()
        self.drop_path = (
            DropPath(drop_path_rate) if drop_path_rate > 0.0 else nn.Identity()
        )

    def forward(self, x):
        return self.drop_path(self.c_proj(self.gelu(self.c_fc(x))))


class QKVMultiheadCrossAttention(nn.Module):
    def __init__(
        self,
        *,
        heads: int,
        n_data: Optional[int] = None,
        width=None,
        qk_norm=False,
        norm_layer=nn.LayerNorm,
    ):
        super().__init__()
        self.heads = heads
        self.n_data = n_data
        self.q_norm = (
            norm_layer(width // heads, elementwise_affine=True, eps=1e-6)
            if qk_norm
            else nn.Identity()
        )
        self.k_norm = (
            norm_layer(width // heads, elementwise_affine=True, eps=1e-6)
            if qk_norm
            else nn.Identity()
        )

        self.attn_processor = CrossAttentionProcessor()

    def forward(self, q, kv):
        _, n_ctx, _ = q.shape
        bs, n_data, width = kv.shape
        attn_ch = width // self.heads // 2
        q = q.view(bs, n_ctx, self.heads, -1)
        kv = kv.view(bs, n_data, self.heads, -1)
        k, v = torch.split(kv, attn_ch, dim=-1)

        q = self.q_norm(q)
        k = self.k_norm(k)
        q, k, v = map(
            lambda t: rearrange(t, "b n h d -> b h n d", h=self.heads), (q, k, v)
        )
        out = self.attn_processor(self, q, k, v)
        out = out.transpose(1, 2).reshape(bs, n_ctx, -1)
        return out


class MultiheadCrossAttention(nn.Module):
    def __init__(
        self,
        *,
        width: int,
        heads: int,
        qkv_bias: bool = True,
        n_data: Optional[int] = None,
        data_width: Optional[int] = None,
        norm_layer=nn.LayerNorm,
        qk_norm: bool = False,
        kv_cache: bool = False,
    ):
        super().__init__()
        self.n_data = n_data
        self.width = width
        self.heads = heads
        self.data_width = width if data_width is None else data_width
        self.c_q = nn.Linear(width, width, bias=qkv_bias)
        self.c_kv = nn.Linear(self.data_width, width * 2, bias=qkv_bias)
        self.c_proj = nn.Linear(width, width)
        self.attention = QKVMultiheadCrossAttention(
            heads=heads,
            n_data=n_data,
            width=width,
            norm_layer=norm_layer,
            qk_norm=qk_norm,
        )
        self.kv_cache = kv_cache
        self.data = None

    def forward(self, x, data):
        x = self.c_q(x)
        if self.kv_cache:
            if self.data is None:
                self.data = self.c_kv(data)
                logger.info(
                    "Save kv cache,this should be called only once for one mesh"
                )
            data = self.data
        else:
            data = self.c_kv(data)
        x = self.attention(x, data)
        x = self.c_proj(x)
        return x


class ResidualCrossAttentionBlock(nn.Module):
    def __init__(
        self,
        *,
        n_data: Optional[int] = None,
        width: int,
        heads: int,
        mlp_expand_ratio: int = 4,
        data_width: Optional[int] = None,
        qkv_bias: bool = True,
        norm_layer=nn.LayerNorm,
        qk_norm: bool = False,
    ):
        super().__init__()

        if data_width is None:
            data_width = width

        self.attn = MultiheadCrossAttention(
            n_data=n_data,
            width=width,
            heads=heads,
            data_width=data_width,
            qkv_bias=qkv_bias,
            norm_layer=norm_layer,
            qk_norm=qk_norm,
        )
        self.ln_1 = norm_layer(width, elementwise_affine=True, eps=1e-6)
        self.ln_2 = norm_layer(data_width, elementwise_affine=True, eps=1e-6)
        self.ln_3 = norm_layer(width, elementwise_affine=True, eps=1e-6)
        self.mlp = MLP(width=width, expand_ratio=mlp_expand_ratio)

    def forward(self, x: torch.Tensor, data: torch.Tensor):
        x = x + self.attn(self.ln_1(x), self.ln_2(data))
        x = x + self.mlp(self.ln_3(x))
        return x


class QKVMultiheadAttention(nn.Module):
    def __init__(
        self,
        *,
        heads: int,
        n_ctx: int,
        width=None,
        qk_norm=False,
        norm_layer=nn.LayerNorm,
    ):
        super().__init__()
        self.heads = heads
        self.n_ctx = n_ctx
        self.q_norm = (
            norm_layer(width // heads, elementwise_affine=True, eps=1e-6)
            if qk_norm
            else nn.Identity()
        )
        self.k_norm = (
            norm_layer(width // heads, elementwise_affine=True, eps=1e-6)
            if qk_norm
            else nn.Identity()
        )

    def forward(self, qkv):
        bs, n_ctx, width = qkv.shape
        attn_ch = width // self.heads // 3
        qkv = qkv.view(bs, n_ctx, self.heads, -1)
        q, k, v = torch.split(qkv, attn_ch, dim=-1)

        q = self.q_norm(q)
        k = self.k_norm(k)

        q, k, v = map(
            lambda t: rearrange(t, "b n h d -> b h n d", h=self.heads), (q, k, v)
        )
        out = (
            scaled_dot_product_attention(q, k, v).transpose(1, 2).reshape(bs, n_ctx, -1)
        )
        return out


class MultiheadAttention(nn.Module):
    def __init__(
        self,
        *,
        n_ctx: int,
        width: int,
        heads: int,
        qkv_bias: bool,
        norm_layer=nn.LayerNorm,
        qk_norm: bool = False,
        drop_path_rate: float = 0.0,
    ):
        super().__init__()
        self.n_ctx = n_ctx
        self.width = width
        self.heads = heads
        self.c_qkv = nn.Linear(width, width * 3, bias=qkv_bias)
        self.c_proj = nn.Linear(width, width)
        self.attention = QKVMultiheadAttention(
            heads=heads,
            n_ctx=n_ctx,
            width=width,
            norm_layer=norm_layer,
            qk_norm=qk_norm,
        )
        self.drop_path = (
            DropPath(drop_path_rate) if drop_path_rate > 0.0 else nn.Identity()
        )

    def forward(self, x):
        x = self.c_qkv(x)
        x = self.attention(x)
        x = self.drop_path(self.c_proj(x))
        return x


class ResidualAttentionBlock(nn.Module):
    def __init__(
        self,
        *,
        n_ctx: int,
        width: int,
        heads: int,
        qkv_bias: bool = True,
        norm_layer=nn.LayerNorm,
        qk_norm: bool = False,
        drop_path_rate: float = 0.0,
    ):
        super().__init__()
        self.attn = MultiheadAttention(
            n_ctx=n_ctx,
            width=width,
            heads=heads,
            qkv_bias=qkv_bias,
            norm_layer=norm_layer,
            qk_norm=qk_norm,
            drop_path_rate=drop_path_rate,
        )
        self.ln_1 = norm_layer(width, elementwise_affine=True, eps=1e-6)
        self.mlp = MLP(width=width, drop_path_rate=drop_path_rate)
        self.ln_2 = norm_layer(width, elementwise_affine=True, eps=1e-6)

    def forward(self, x: torch.Tensor):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class Transformer(nn.Module):
    def __init__(
        self,
        *,
        n_ctx: int,
        width: int,
        layers: int,
        heads: int,
        qkv_bias: bool = True,
        norm_layer=nn.LayerNorm,
        qk_norm: bool = False,
        drop_path_rate: float = 0.0,
    ):
        super().__init__()
        self.n_ctx = n_ctx
        self.width = width
        self.layers = layers
        self.resblocks = nn.ModuleList(
            [
                ResidualAttentionBlock(
                    n_ctx=n_ctx,
                    width=width,
                    heads=heads,
                    qkv_bias=qkv_bias,
                    norm_layer=norm_layer,
                    qk_norm=qk_norm,
                    drop_path_rate=drop_path_rate,
                )
                for _ in range(layers)
            ]
        )

    def forward(self, x: torch.Tensor):
        for block in self.resblocks:
            x = block(x)
        return x


class CrossAttentionDecoder(nn.Module):

    def __init__(
        self,
        *,
        num_latents: int,
        out_channels: int,
        fourier_embedder: FourierEmbedder,
        width: int,
        heads: int,
        mlp_expand_ratio: int = 4,
        downsample_ratio: int = 1,
        enable_ln_post: bool = True,
        qkv_bias: bool = True,
        qk_norm: bool = False,
        label_type: str = "binary",
    ):
        super().__init__()

        self.enable_ln_post = enable_ln_post
        self.fourier_embedder = fourier_embedder
        self.downsample_ratio = downsample_ratio
        self.query_proj = nn.Linear(self.fourier_embedder.out_dim, width)
        if self.downsample_ratio != 1:
            self.latents_proj = nn.Linear(width * downsample_ratio, width)
        if self.enable_ln_post == False:
            qk_norm = False
        self.cross_attn_decoder = ResidualCrossAttentionBlock(
            n_data=num_latents,
            width=width,
            mlp_expand_ratio=mlp_expand_ratio,
            heads=heads,
            qkv_bias=qkv_bias,
            qk_norm=qk_norm,
        )

        if self.enable_ln_post:
            self.ln_post = nn.LayerNorm(width)
        self.output_proj = nn.Linear(width, out_channels)
        self.label_type = label_type

    def set_cross_attention_processor(self, processor):
        self.cross_attn_decoder.attn.attention.attn_processor = processor

    def forward(self, queries=None, query_embeddings=None, latents=None):
        if query_embeddings is None:
            fourier_out = self.fourier_embedder(queries)
            query_embeddings = self.query_proj(fourier_out.to(latents.dtype))

        if self.downsample_ratio != 1:
            latents = self.latents_proj(latents)

        x = self.cross_attn_decoder(query_embeddings, latents)

        if self.enable_ln_post:
            x = self.ln_post(x)

        occ = self.output_proj(x)
        return occ


def generate_dense_grid_points(
    bbox_min: np.ndarray,
    bbox_max: np.ndarray,
    octree_resolution: int,
    indexing: str = "ij",
):
    length = bbox_max - bbox_min
    num_cells = octree_resolution

    x = np.linspace(bbox_min[0], bbox_max[0], int(num_cells) + 1, dtype=np.float32)
    y = np.linspace(bbox_min[1], bbox_max[1], int(num_cells) + 1, dtype=np.float32)
    z = np.linspace(bbox_min[2], bbox_max[2], int(num_cells) + 1, dtype=np.float32)
    [xs, ys, zs] = np.meshgrid(x, y, z, indexing=indexing)
    xyz = np.stack((xs, ys, zs), axis=-1)
    grid_size = [int(num_cells) + 1, int(num_cells) + 1, int(num_cells) + 1]

    return xyz, grid_size, length


def extract_near_surface_volume_fn(input_tensor: torch.Tensor, alpha: float):
    """Extract near-surface voxels for hierarchical decoding."""
    device = input_tensor.device

    val = input_tensor + alpha
    valid_mask = val > -9000

    def get_neighbor(t, shift, axis):
        if shift == 0:
            return t.clone()
        pad_dims = [0, 0, 0, 0, 0, 0]
        if axis == 0:
            pad_idx = 0 if shift > 0 else 1
            pad_dims[pad_idx] = abs(shift)
        elif axis == 1:
            pad_idx = 2 if shift > 0 else 3
            pad_dims[pad_idx] = abs(shift)
        elif axis == 2:
            pad_idx = 4 if shift > 0 else 5
            pad_dims[pad_idx] = abs(shift)

        padded = F.pad(t.unsqueeze(0).unsqueeze(0), pad_dims[::-1], mode="replicate")

        slice_dims = [slice(None)] * 3
        if axis == 0:
            slice_dims[0] = slice(shift, None) if shift > 0 else slice(None, shift)
        elif axis == 1:
            slice_dims[1] = slice(shift, None) if shift > 0 else slice(None, shift)
        elif axis == 2:
            slice_dims[2] = slice(shift, None) if shift > 0 else slice(None, shift)

        padded = padded.squeeze(0).squeeze(0)
        return padded[slice_dims]

    left = get_neighbor(val, 1, axis=0)
    right = get_neighbor(val, -1, axis=0)
    back = get_neighbor(val, 1, axis=1)
    front = get_neighbor(val, -1, axis=1)
    down = get_neighbor(val, 1, axis=2)
    up = get_neighbor(val, -1, axis=2)

    def safe_where(neighbor):
        return torch.where(neighbor > -9000, neighbor, val)

    left, right = safe_where(left), safe_where(right)
    back, front = safe_where(back), safe_where(front)
    down, up = safe_where(down), safe_where(up)

    sign = torch.sign(val.to(torch.float32))
    neighbors_sign = torch.stack(
        [
            torch.sign(left.to(torch.float32)),
            torch.sign(right.to(torch.float32)),
            torch.sign(back.to(torch.float32)),
            torch.sign(front.to(torch.float32)),
            torch.sign(down.to(torch.float32)),
            torch.sign(up.to(torch.float32)),
        ],
        dim=0,
    )

    same_sign = torch.all(neighbors_sign == sign, dim=0)
    mask = (~same_sign).to(torch.int32)
    return mask * valid_mask.to(torch.int32)


class VanillaVolumeDecoder:
    """Standard volume decoder using dense grid evaluation."""

    @torch.no_grad()
    def __call__(
        self,
        latents: torch.FloatTensor,
        geo_decoder: Callable,
        bounds: Union[Tuple[float], List[float], float] = 1.01,
        num_chunks: int = 10000,
        octree_resolution: int = None,
        enable_pbar: bool = True,
        **kwargs,
    ):
        device = latents.device
        dtype = latents.dtype
        batch_size = latents.shape[0]

        if isinstance(bounds, float):
            bounds = [-bounds, -bounds, -bounds, bounds, bounds, bounds]

        bbox_min, bbox_max = np.array(bounds[0:3]), np.array(bounds[3:6])
        xyz_samples, grid_size, length = generate_dense_grid_points(
            bbox_min=bbox_min,
            bbox_max=bbox_max,
            octree_resolution=octree_resolution,
            indexing="ij",
        )
        xyz_samples = (
            torch.from_numpy(xyz_samples)
            .to(device, dtype=dtype)
            .contiguous()
            .reshape(-1, 3)
        )

        batch_logits = []
        for start in tqdm(
            range(0, xyz_samples.shape[0], num_chunks),
            desc="Volume Decoding",
            disable=not enable_pbar,
        ):
            chunk_queries = xyz_samples[start : start + num_chunks, :]
            chunk_queries = repeat(chunk_queries, "p c -> b p c", b=batch_size)
            logits = geo_decoder(queries=chunk_queries, latents=latents)
            batch_logits.append(logits)

        grid_logits = torch.cat(batch_logits, dim=1)
        grid_logits = grid_logits.view((batch_size, *grid_size)).float()

        return grid_logits


class HierarchicalVolumeDecoding:
    """Hierarchical volume decoder with multi-resolution refinement."""

    @torch.no_grad()
    def __call__(
        self,
        latents: torch.FloatTensor,
        geo_decoder: Callable,
        bounds: Union[Tuple[float], List[float], float] = 1.01,
        num_chunks: int = 10000,
        mc_level: float = 0.0,
        octree_resolution: int = None,
        min_resolution: int = 63,
        enable_pbar: bool = True,
        **kwargs,
    ):
        device = latents.device
        dtype = latents.dtype

        resolutions = []
        if octree_resolution < min_resolution:
            resolutions.append(octree_resolution)
        while octree_resolution >= min_resolution:
            resolutions.append(octree_resolution)
            octree_resolution = octree_resolution // 2
        resolutions.reverse()

        # 1. generate query points
        if isinstance(bounds, float):
            bounds = [-bounds, -bounds, -bounds, bounds, bounds, bounds]
        bbox_min = np.array(bounds[0:3])
        bbox_max = np.array(bounds[3:6])
        bbox_size = bbox_max - bbox_min

        xyz_samples, grid_size, length = generate_dense_grid_points(
            bbox_min=bbox_min,
            bbox_max=bbox_max,
            octree_resolution=resolutions[0],
            indexing="ij",
        )

        dilate = nn.Conv3d(1, 1, 3, padding=1, bias=False, device=device, dtype=dtype)
        dilate.weight = torch.nn.Parameter(
            torch.ones(dilate.weight.shape, dtype=dtype, device=device)
        )

        grid_size = np.array(grid_size)
        xyz_samples = (
            torch.from_numpy(xyz_samples)
            .to(device, dtype=dtype)
            .contiguous()
            .reshape(-1, 3)
        )

        # 2. latents to 3d volume
        batch_logits = []
        batch_size = latents.shape[0]
        for start in tqdm(
            range(0, xyz_samples.shape[0], num_chunks),
            desc=f"Hierarchical Volume Decoding [r{resolutions[0] + 1}]",
            disable=not enable_pbar,
        ):
            queries = xyz_samples[start : start + num_chunks, :]
            batch_queries = repeat(queries, "p c -> b p c", b=batch_size)
            logits = geo_decoder(queries=batch_queries, latents=latents)
            batch_logits.append(logits)

        grid_logits = torch.cat(batch_logits, dim=1).view(
            (batch_size, grid_size[0], grid_size[1], grid_size[2])
        )

        for octree_depth_now in resolutions[1:]:
            grid_size = np.array([octree_depth_now + 1] * 3)
            resolution = bbox_size / octree_depth_now
            next_index = torch.zeros(tuple(grid_size), dtype=dtype, device=device)
            next_logits = torch.full(
                next_index.shape, -10000.0, dtype=dtype, device=device
            )
            curr_points = extract_near_surface_volume_fn(
                grid_logits.squeeze(0), mc_level
            )
            curr_points += grid_logits.squeeze(0).abs() < 0.95

            if octree_depth_now == resolutions[-1]:
                expand_num = 0
            else:
                expand_num = 1
            for i in range(expand_num):
                curr_points = dilate(curr_points.unsqueeze(0).to(dtype)).squeeze(0)
            cidx_x, cidx_y, cidx_z = torch.where(curr_points > 0)
            next_index[cidx_x * 2, cidx_y * 2, cidx_z * 2] = 1
            for i in range(2 - expand_num):
                next_index = dilate(next_index.unsqueeze(0)).squeeze(0)
            nidx = torch.where(next_index > 0)

            next_points = torch.stack(nidx, dim=1)
            next_points = next_points * torch.tensor(
                resolution, dtype=next_points.dtype, device=device
            ) + torch.tensor(bbox_min, dtype=next_points.dtype, device=device)

            # Check if next_points is empty
            if next_points.shape[0] == 0:
                logger.warning(
                    f"No valid surface points found at resolution {octree_depth_now}, "
                    f"skipping this level"
                )
                continue

            batch_logits = []
            for start in tqdm(
                range(0, next_points.shape[0], num_chunks),
                desc=f"Hierarchical Volume Decoding [r{octree_depth_now + 1}]",
                disable=not enable_pbar,
            ):
                queries = next_points[start : start + num_chunks, :]
                batch_queries = repeat(queries, "p c -> b p c", b=batch_size)
                logits = geo_decoder(
                    queries=batch_queries.to(latents.dtype), latents=latents
                )
                batch_logits.append(logits)
            grid_logits = torch.cat(batch_logits, dim=1)
            next_logits[nidx] = grid_logits[0, ..., 0]
            grid_logits = next_logits.unsqueeze(0)
        grid_logits[grid_logits == -10000.0] = float("nan")

        return grid_logits


class FlashVDMVolumeDecoding:
    """Flash VDM volume decoder with adaptive KV selection."""

    def __init__(self, topk_mode="mean"):
        if topk_mode not in ["mean", "merge"]:
            raise ValueError(f"Unsupported topk_mode {topk_mode}")

        if topk_mode == "mean":
            self.processor = FlashVDMCrossAttentionProcessor()
        else:
            self.processor = FlashVDMTopMCrossAttentionProcessor()

    @torch.no_grad()
    def __call__(
        self,
        latents: torch.FloatTensor,
        geo_decoder: CrossAttentionDecoder,
        bounds: Union[Tuple[float], List[float], float] = 1.01,
        num_chunks: int = 10000,
        mc_level: float = 0.0,
        octree_resolution: int = None,
        min_resolution: int = 63,
        mini_grid_num: int = 4,
        enable_pbar: bool = True,
        **kwargs,
    ):
        processor = self.processor
        geo_decoder.set_cross_attention_processor(processor)

        device = latents.device
        dtype = latents.dtype

        resolutions = []
        orig_resolution = octree_resolution
        if octree_resolution < min_resolution:
            resolutions.append(octree_resolution)
        while octree_resolution >= min_resolution:
            resolutions.append(octree_resolution)
            octree_resolution = octree_resolution // 2
        resolutions.reverse()
        resolutions[0] = round(resolutions[0] / mini_grid_num) * mini_grid_num - 1
        for i, resolution in enumerate(resolutions[1:]):
            resolutions[i + 1] = resolutions[0] * 2 ** (i + 1)

        if isinstance(bounds, float):
            bounds = [-bounds, -bounds, -bounds, bounds, bounds, bounds]
        bbox_min = np.array(bounds[0:3])
        bbox_max = np.array(bounds[3:6])
        bbox_size = bbox_max - bbox_min

        xyz_samples, grid_size, length = generate_dense_grid_points(
            bbox_min=bbox_min,
            bbox_max=bbox_max,
            octree_resolution=resolutions[0],
            indexing="ij",
        )

        logger.info(f"FlashVDMVolumeDecoding Resolution: {resolutions}")

        dilate = nn.Conv3d(1, 1, 3, padding=1, bias=False, device=device, dtype=dtype)
        dilate.weight = torch.nn.Parameter(
            torch.ones(dilate.weight.shape, dtype=dtype, device=device)
        )

        grid_size = np.array(grid_size)

        xyz_samples = torch.from_numpy(xyz_samples).to(device, dtype=dtype)
        batch_size = latents.shape[0]
        mini_grid_size = xyz_samples.shape[0] // mini_grid_num
        xyz_samples = (
            xyz_samples.view(
                mini_grid_num,
                mini_grid_size,
                mini_grid_num,
                mini_grid_size,
                mini_grid_num,
                mini_grid_size,
                3,
            )
            .permute(0, 2, 4, 1, 3, 5, 6)
            .reshape(-1, mini_grid_size * mini_grid_size * mini_grid_size, 3)
        )

        batch_logits = []
        num_batchs = max(num_chunks // xyz_samples.shape[1], 1)
        for start in tqdm(
            range(0, xyz_samples.shape[0], num_batchs),
            desc="FlashVDM Volume Decoding",
            disable=not enable_pbar,
        ):
            queries = xyz_samples[start : start + num_batchs, :]
            batch = queries.shape[0]
            batch_latents = repeat(latents.squeeze(0), "p c -> b p c", b=batch)
            processor.topk = True
            logits = geo_decoder(queries=queries, latents=batch_latents)
            batch_logits.append(logits)

        grid_logits = (
            torch.cat(batch_logits, dim=0)
            .reshape(
                mini_grid_num,
                mini_grid_num,
                mini_grid_num,
                mini_grid_size,
                mini_grid_size,
                mini_grid_size,
            )
            .permute(0, 3, 1, 4, 2, 5)
            .contiguous()
            .view((batch_size, grid_size[0], grid_size[1], grid_size[2]))
        )

        for octree_depth_now in resolutions[1:]:
            grid_size = np.array([octree_depth_now + 1] * 3)
            resolution = bbox_size / octree_depth_now
            next_index = torch.zeros(tuple(grid_size), dtype=dtype, device=device)
            next_logits = torch.full(
                next_index.shape, -10000.0, dtype=dtype, device=device
            )
            curr_points = extract_near_surface_volume_fn(
                grid_logits.squeeze(0), mc_level
            )
            curr_points += grid_logits.squeeze(0).abs() < 0.95

            expand_num = 0 if octree_depth_now == resolutions[-1] else 1
            for _ in range(expand_num):
                curr_points = dilate(curr_points.unsqueeze(0).to(dtype)).squeeze(0)

            cidx_x, cidx_y, cidx_z = torch.where(curr_points > 0)
            next_index[cidx_x * 2, cidx_y * 2, cidx_z * 2] = 1
            for _ in range(2 - expand_num):
                next_index = dilate(next_index.unsqueeze(0)).squeeze(0)
            nidx = torch.where(next_index > 0)

            next_points = torch.stack(nidx, dim=1)
            next_points = next_points * torch.tensor(
                resolution, dtype=torch.float32, device=device
            ) + torch.tensor(bbox_min, dtype=torch.float32, device=device)

            # Check if next_points is empty (no valid surface points found)
            if next_points.shape[0] == 0:
                # Skip this resolution level if no points found
                # Use the previous grid_logits as fallback
                logger.warning(
                    f"No valid surface points found at resolution {octree_depth_now}, "
                    f"skipping this level and using previous resolution grid_logits"
                )
                continue

            query_grid_num = 6
            min_val = next_points.min(axis=0).values
            max_val = next_points.max(axis=0).values
            vol_queries_index = (
                (next_points - min_val) / (max_val - min_val) * (query_grid_num - 0.001)
            )
            index = torch.floor(vol_queries_index).long()
            index = (
                index[..., 0] * (query_grid_num**2)
                + index[..., 1] * query_grid_num
                + index[..., 2]
            )
            index = index.sort()
            next_points = next_points[index.indices].unsqueeze(0).contiguous()
            unique_values = torch.unique(index.values, return_counts=True)
            grid_logits_flat = torch.zeros(
                (next_points.shape[1]), dtype=latents.dtype, device=latents.device
            )
            input_grid = [[], []]
            logits_grid_list = []
            start_num = 0
            sum_num = 0
            for grid_index, count in zip(
                unique_values[0].cpu().tolist(), unique_values[1].cpu().tolist()
            ):
                if sum_num + count < num_chunks or sum_num == 0:
                    sum_num += count
                    input_grid[0].append(grid_index)
                    input_grid[1].append(count)
                else:
                    processor.topk = input_grid
                    logits_grid = geo_decoder(
                        queries=next_points[:, start_num : start_num + sum_num],
                        latents=latents,
                    )
                    start_num = start_num + sum_num
                    logits_grid_list.append(logits_grid)
                    input_grid = [[grid_index], [count]]
                    sum_num = count
            if sum_num > 0:
                processor.topk = input_grid
                logits_grid = geo_decoder(
                    queries=next_points[:, start_num : start_num + sum_num],
                    latents=latents,
                )
                logits_grid_list.append(logits_grid)
            logits_grid = torch.cat(logits_grid_list, dim=1)
            grid_logits_flat[index.indices] = logits_grid.squeeze(0).squeeze(-1)
            next_logits[nidx] = grid_logits_flat
            grid_logits = next_logits.unsqueeze(0)

        grid_logits[grid_logits == -10000.0] = float("nan")
        return grid_logits


class Latent2MeshOutput:
    """Container for mesh output from VAE decoder."""

    def __init__(self, mesh_v=None, mesh_f=None):
        self.mesh_v = mesh_v
        self.mesh_f = mesh_f


def center_vertices(vertices):
    """Translate vertices so bounding box is centered at zero."""
    vert_min = vertices.min(dim=0)[0]
    vert_max = vertices.max(dim=0)[0]
    vert_center = 0.5 * (vert_min + vert_max)
    return vertices - vert_center


class SurfaceExtractor:
    """Base class for surface extraction algorithms."""

    def _compute_box_stat(
        self, bounds: Union[Tuple[float], List[float], float], octree_resolution: int
    ):
        if isinstance(bounds, float):
            bounds = [-bounds, -bounds, -bounds, bounds, bounds, bounds]

        bbox_min, bbox_max = np.array(bounds[0:3]), np.array(bounds[3:6])
        bbox_size = bbox_max - bbox_min
        grid_size = [
            int(octree_resolution) + 1,
            int(octree_resolution) + 1,
            int(octree_resolution) + 1,
        ]
        return grid_size, bbox_min, bbox_size

    def run(self, *args, **kwargs):
        raise NotImplementedError

    def __call__(self, grid_logits, **kwargs):
        outputs = []
        for i in range(grid_logits.shape[0]):
            try:
                vertices, faces = self.run(grid_logits[i], **kwargs)
                vertices = vertices.astype(np.float32)
                faces = np.ascontiguousarray(faces)
                outputs.append(Latent2MeshOutput(mesh_v=vertices, mesh_f=faces))
            except Exception:
                import traceback

                traceback.print_exc()
                outputs.append(None)
        return outputs


class MCSurfaceExtractor(SurfaceExtractor):
    """Marching Cubes surface extractor."""

    def run(self, grid_logit, *, mc_level, bounds, octree_resolution, **kwargs):
        from skimage import measure

        vertices, faces, normals, _ = measure.marching_cubes(
            grid_logit.cpu().numpy(), mc_level, method="lewiner"
        )
        grid_size, bbox_min, bbox_size = self._compute_box_stat(
            bounds, octree_resolution
        )
        vertices = vertices / grid_size * bbox_size + bbox_min
        return vertices, faces


class DMCSurfaceExtractor(SurfaceExtractor):
    """Differentiable Marching Cubes surface extractor."""

    def run(self, grid_logit, *, octree_resolution, **kwargs):
        device = grid_logit.device
        if not hasattr(self, "dmc"):
            try:
                from diso import DiffDMC

                self.dmc = DiffDMC(dtype=torch.float32).to(device)
            except ImportError:
                raise ImportError(
                    "Please install diso via `pip install diso`, or set mc_algo to 'mc'"
                )
        sdf = -grid_logit / octree_resolution
        sdf = sdf.to(torch.float32).contiguous()
        verts, faces = self.dmc(sdf, deform=None, return_quads=False, normalize=True)
        verts = center_vertices(verts)
        vertices = verts.detach().cpu().numpy()
        faces = faces.detach().cpu().numpy()[:, ::-1]
        return vertices, faces


SurfaceExtractors = {
    "mc": MCSurfaceExtractor,
    "dmc": DMCSurfaceExtractor,
}


class VectsetVAE(nn.Module):
    """Base VAE class for vector set encoding."""

    def __init__(self, volume_decoder=None, surface_extractor=None):
        super().__init__()
        if volume_decoder is None:
            volume_decoder = VanillaVolumeDecoder()
        if surface_extractor is None:
            surface_extractor = MCSurfaceExtractor()
        self.volume_decoder = volume_decoder
        self.surface_extractor = surface_extractor

    def latents2mesh(self, latents: torch.FloatTensor, **kwargs):
        """Convert latents to mesh."""
        grid_logits = self.volume_decoder(latents, self.geo_decoder, **kwargs)
        outputs = self.surface_extractor(grid_logits, **kwargs)
        return outputs

    def enable_flashvdm_decoder(
        self,
        enabled: bool = True,
        adaptive_kv_selection=True,
        topk_mode="mean",
        mc_algo="dmc",
    ):
        """Enable or disable FlashVDM decoder for faster inference."""
        if enabled:
            if adaptive_kv_selection:
                self.volume_decoder = FlashVDMVolumeDecoding(topk_mode)
            else:
                self.volume_decoder = HierarchicalVolumeDecoding()
            if mc_algo not in SurfaceExtractors:
                raise ValueError(
                    f"Unsupported mc_algo {mc_algo}, available: {list(SurfaceExtractors.keys())}"
                )
            self.surface_extractor = SurfaceExtractors[mc_algo]()
        else:
            self.volume_decoder = VanillaVolumeDecoder()
            self.surface_extractor = MCSurfaceExtractor()


class ShapeVAE(VectsetVAE):
    """Shape VAE for 3D mesh generation from latent codes."""

    _aliases = ["hy3dgen.shapegen.models.ShapeVAE"]

    def __init__(
        self,
        *,
        num_latents: int,
        embed_dim: int,
        width: int,
        heads: int,
        num_decoder_layers: int,
        num_encoder_layers: int = 8,
        pc_size: int = 5120,
        pc_sharpedge_size: int = 5120,
        point_feats: int = 3,
        downsample_ratio: int = 20,
        geo_decoder_downsample_ratio: int = 1,
        geo_decoder_mlp_expand_ratio: int = 4,
        geo_decoder_ln_post: bool = True,
        num_freqs: int = 8,
        include_pi: bool = True,
        qkv_bias: bool = True,
        qk_norm: bool = False,
        label_type: str = "binary",
        drop_path_rate: float = 0.0,
        scale_factor: float = 1.0,
        use_ln_post: bool = True,
        ckpt_path=None,
    ):
        super().__init__()
        self.geo_decoder_ln_post = geo_decoder_ln_post
        self.downsample_ratio = downsample_ratio

        self.fourier_embedder = FourierEmbedder(
            num_freqs=num_freqs, include_pi=include_pi
        )

        self.post_kl = nn.Linear(embed_dim, width)

        self.transformer = Transformer(
            n_ctx=num_latents,
            width=width,
            layers=num_decoder_layers,
            heads=heads,
            qkv_bias=qkv_bias,
            qk_norm=qk_norm,
            drop_path_rate=drop_path_rate,
        )

        self.geo_decoder = CrossAttentionDecoder(
            fourier_embedder=self.fourier_embedder,
            out_channels=1,
            num_latents=num_latents,
            mlp_expand_ratio=geo_decoder_mlp_expand_ratio,
            downsample_ratio=geo_decoder_downsample_ratio,
            enable_ln_post=self.geo_decoder_ln_post,
            width=width // geo_decoder_downsample_ratio,
            heads=heads // geo_decoder_downsample_ratio,
            qkv_bias=qkv_bias,
            qk_norm=qk_norm,
            label_type=label_type,
        )

        self.scale_factor = scale_factor
        self.latent_shape = (num_latents, embed_dim)

    def forward(self, latents):
        latents = self.post_kl(latents)
        latents = self.transformer(latents)
        return latents

    def decode(self, latents):
        """Decode latents to features."""
        latents = self.post_kl(latents)
        latents = self.transformer(latents)
        return latents


# Entry class for model registry
EntryClass = ShapeVAE
