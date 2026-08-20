# SPDX-License-Identifier: Apache-2.0
"""LTX-2.5 diffusion video decoder.

An alternative to the convolutional VAE decoder: stages 1-4 deterministically
upsample the latent into a context volume, and stage 5 denoises patchified
pixels conditioned on it. As shipped (`model_output_type="x0"`, one step) that
single prediction *is* the output.

Attention is 3D *neighborhood* attention -- each query attends to a fixed window
that shifts inward at the grid borders. Prefers NATTEN's fused `na3d` like
upstream, falling back to a FlexAttention block mask when NATTEN is missing.
"""

import math

import torch
import torch.nn.functional as F
from torch import nn

from sglang.multimodal_gen.configs.models.decoders.ltx_2_5_diffusion_decoder import (
    LTX25DiffusionDecoderConfig,
)
from sglang.multimodal_gen.runtime.layers.visual_embedding import (
    timestep_embedding,
)
from sglang.multimodal_gen.runtime.managers.memory_managers.layerwise_offload import (
    LayerwiseOffloadableModuleMixin,
)
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

logger = init_logger(__name__)

# Bounds the three hidden-width temporaries the SwiGLU holds live. Exact: the
# MLP is pointwise across tokens.
_SWIGLU_TILE_SIZE = 16384

_na3d_fn: object = None
_NA3D_UNAVAILABLE = object()


def _na3d():
    """NATTEN's fused 3D neighborhood attention, or `None` if unavailable.

    ~4.8x the compiled `flex_attention` fallback, and needs no block mask.
    """
    global _na3d_fn
    if _na3d_fn is None:
        try:
            from natten.functional import na3d

            _na3d_fn = na3d
        except ImportError:
            _na3d_fn = _NA3D_UNAVAILABLE
    return None if _na3d_fn is _NA3D_UNAVAILABLE else _na3d_fn


_compiled_flex_attention = None


class LTX2VideoDecoderTimestepEmbedder(nn.Module):
    """Replicated native timestep MLP used by the standalone decoder.

    The decoder is replicated across TP ranks, so these projections must remain
    ordinary linear layers. Reusing the DiT's tensor-parallel embedder shards
    their parameters and makes the unsharded decoder checkpoint unloadable.
    """

    def __init__(self, embedding_dim: int, in_channels: int = 256) -> None:
        super().__init__()
        self.linear_1 = nn.Linear(in_channels, embedding_dim, bias=True)
        self.linear_2 = nn.Linear(embedding_dim, embedding_dim, bias=True)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.linear_1(hidden_states)
        hidden_states = F.silu(hidden_states)
        return self.linear_2(hidden_states)


class LTX2VideoDecoderCombinedTimestepEmbeddings(nn.Module):
    def __init__(self, embedding_dim: int) -> None:
        super().__init__()
        self.timestep_embedder = LTX2VideoDecoderTimestepEmbedder(embedding_dim)

    def forward(
        self, timestep: torch.Tensor, hidden_dtype: torch.dtype | None = None
    ) -> torch.Tensor:
        timestep = timestep.reshape(-1).to(dtype=torch.float32)
        hidden_states = timestep_embedding(
            timestep, dim=256, max_period=10000, dtype=torch.float32
        )
        if hidden_dtype is not None:
            hidden_states = hidden_states.to(dtype=hidden_dtype)
        return self.timestep_embedder(hidden_states)


def _flex_attention_fn():
    """`flex_attention`, compiled.

    Uncompiled it falls back to materializing the full `S x S` score matrix,
    which is tens of GiB at these grids -- compiling is what makes the
    neighborhood window actually sparse. Compiled once and cached; the handful
    of distinct decoder-stage shapes each trigger one recompile.
    """
    global _compiled_flex_attention
    if _compiled_flex_attention is None:
        from torch.nn.attention.flex_attention import flex_attention

        _compiled_flex_attention = torch.compile(flex_attention, dynamic=False)
    return _compiled_flex_attention


def _patchify(x: torch.Tensor, patch_size: int) -> torch.Tensor:
    """Space-to-depth on H/W only: `(B,C,F,H,W)` -> `(B, C*p**2, F, H//p, W//p)`.

    Channel packing order is `(channel, width_offset, height_offset)`.
    """
    batch_size, num_channels, num_frames, height, width = x.shape
    x = x.reshape(
        batch_size,
        num_channels,
        num_frames,
        height // patch_size,
        patch_size,
        width // patch_size,
        patch_size,
    )
    x = x.permute(0, 1, 6, 4, 2, 3, 5)
    return x.reshape(
        batch_size,
        num_channels * patch_size * patch_size,
        num_frames,
        height // patch_size,
        width // patch_size,
    )


def _unpatchify(x: torch.Tensor, patch_size: int) -> torch.Tensor:
    """Depth-to-space on H/W only; the exact inverse of `_patchify`."""
    batch_size, num_channels, num_frames, height, width = x.shape
    num_channels = num_channels // (patch_size * patch_size)
    x = x.reshape(
        batch_size, num_channels, patch_size, patch_size, num_frames, height, width
    )
    x = x.permute(0, 1, 4, 5, 3, 6, 2)
    return x.reshape(
        batch_size, num_channels, num_frames, height * patch_size, width * patch_size
    )


# O(S^2) to build (17.5 s at 1.08M tokens) and a pure function of grid and
# kernel, so a server at one resolution pays it once.
_BLOCK_MASK_CACHE: dict = {}
_BLOCK_MASK_CACHE_MAX = 16


def _neighborhood_block_mask(
    num_frames: int,
    height: int,
    width: int,
    kernel_size: tuple[int, int, int],
    device: torch.device,
):
    """FlexAttention `BlockMask` for a 3D neighborhood window.

    The window is centered where possible and shifted inward at the borders so
    it always holds exactly `kernel_size` positions -- that inward shift, rather
    than truncation, is what NATTEN's `na3d` does.
    """
    from torch.nn.attention.flex_attention import create_block_mask

    cache_key = (num_frames, height, width, tuple(kernel_size), str(device))
    cached = _BLOCK_MASK_CACHE.get(cache_key)
    if cached is not None:
        return cached

    kernel_t, kernel_h, kernel_w = kernel_size
    kernel_t = min(kernel_t, num_frames)
    kernel_h = min(kernel_h, height)
    kernel_w = min(kernel_w, width)
    hw = height * width

    def mask_mod(batch_idx, head_idx, q_idx, kv_idx):
        q_t, q_rem = q_idx // hw, q_idx % hw
        q_h, q_w = q_rem // width, q_rem % width
        k_t, k_rem = kv_idx // hw, kv_idx % hw
        k_h, k_w = k_rem // width, k_rem % width

        start_t = torch.clamp(q_t - kernel_t // 2, 0, num_frames - kernel_t)
        start_h = torch.clamp(q_h - kernel_h // 2, 0, height - kernel_h)
        start_w = torch.clamp(q_w - kernel_w // 2, 0, width - kernel_w)
        window_t = (k_t >= start_t) & (k_t < start_t + kernel_t)
        window_h = (k_h >= start_h) & (k_h < start_h + kernel_h)
        window_w = (k_w >= start_w) & (k_w < start_w + kernel_w)
        return window_t & window_h & window_w

    seq_len = num_frames * hw
    # `_compile=True` is required, not an optimisation: the eager path
    # materialises O(S^2) booleans, tens of GiB at production grids.
    block_mask = create_block_mask(
        mask_mod,
        B=None,
        H=None,
        Q_LEN=seq_len,
        KV_LEN=seq_len,
        device=device,
        _compile=True,
    )
    if len(_BLOCK_MASK_CACHE) >= _BLOCK_MASK_CACHE_MAX:
        _BLOCK_MASK_CACHE.pop(next(iter(_BLOCK_MASK_CACHE)))
    _BLOCK_MASK_CACHE[cache_key] = block_mask
    return block_mask


class LTX2VideoVaeRotaryPosEmbed3D(nn.Module):
    """Absolute 3D rotary embedding over the (T, H, W) grid.

    `head_dim` splits into (T, H, W) chunks, each rotated by its own axis
    position.
    """

    def __init__(self, head_dim: int, base: float = 10000.0) -> None:
        super().__init__()
        if head_dim % 8 != 0:
            raise ValueError(f"head_dim must be a multiple of 8, got {head_dim}.")
        # A quarter to T, the rest split H/W, both kept even for whole
        # rotation pairs.
        dim_t = (head_dim // 4) // 2 * 2
        dim_hw = (head_dim - dim_t) // 2
        if dim_hw % 2 != 0:
            dim_t -= 2
            dim_hw = (head_dim - dim_t) // 2
        self.rope_dim_split = (dim_t, dim_hw, dim_hw)
        self.base = base

    def _inv_freqs(self, dim: int, device: torch.device) -> torch.Tensor:
        exponents = torch.arange(0, dim, 2, dtype=torch.float64, device=device) / dim
        return (1.0 / self.base**exponents).to(torch.float32)

    def _rotate_axis(
        self,
        x: torch.Tensor,
        positions: torch.Tensor,
        inv_freqs: torch.Tensor,
        axis: int,
    ) -> torch.Tensor:
        out_dtype = x.dtype
        pairs = x.reshape(*x.shape[:-1], x.shape[-1] // 2, 2)
        even = pairs[..., 0].float()
        odd = pairs[..., 1].float()
        # Broadcast over (B, T, H, W, heads, dim // 2), varying only along `axis`.
        shape = [1, 1, 1, 1, 1, inv_freqs.shape[0]]
        shape[axis] = positions.shape[0]
        angles = (positions[:, None] * inv_freqs[None, :]).reshape(shape)
        cos, sin = angles.cos(), angles.sin()
        rotated = torch.stack([even * cos - odd * sin, even * sin + odd * cos], dim=-1)
        return rotated.reshape(x.shape).to(out_dtype)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """`hidden_states`: `(B, T, H, W, heads, head_dim)`."""
        dim_t, dim_h, _ = self.rope_dim_split
        num_frames, height, width = hidden_states.shape[1:4]
        device = hidden_states.device
        inv_t, inv_h, inv_w = (
            self._inv_freqs(dim, device) for dim in self.rope_dim_split
        )

        positions_t = torch.arange(num_frames, dtype=torch.float32, device=device)
        positions_h = torch.arange(height, dtype=torch.float32, device=device)
        positions_w = torch.arange(width, dtype=torch.float32, device=device)
        rotated_t = self._rotate_axis(
            hidden_states[..., :dim_t], positions_t, inv_t, axis=1
        )
        rotated_h = self._rotate_axis(
            hidden_states[..., dim_t : dim_t + dim_h], positions_h, inv_h, axis=2
        )
        rotated_w = self._rotate_axis(
            hidden_states[..., dim_t + dim_h :], positions_w, inv_w, axis=3
        )
        return torch.cat([rotated_t, rotated_h, rotated_w], dim=-1)


class LTX2VideoVaeNeighborhoodAttention(nn.Module):
    """3D neighborhood attention over a channels-last `(B, T, H, W, C)` volume."""

    def __init__(
        self,
        dim: int,
        kernel_size: tuple[int, int, int],
        head_dim: int = 64,
        rope_base: float = 10000.0,
    ) -> None:
        super().__init__()
        if dim % head_dim != 0:
            raise ValueError(f"dim {dim} must be divisible by head_dim {head_dim}.")
        self.heads = dim // head_dim
        self.head_dim = head_dim
        self.kernel_size = tuple(kernel_size)
        self.scale = head_dim**-0.5

        self.to_q = nn.Linear(dim, dim, bias=True)
        self.to_k = nn.Linear(dim, dim, bias=True)
        self.to_v = nn.Linear(dim, dim, bias=True)
        self.to_out = nn.ModuleList([nn.Linear(dim, dim, bias=True), nn.Dropout(0.0)])
        self.norm_q = nn.RMSNorm(head_dim, eps=1e-6)
        self.norm_k = nn.RMSNorm(head_dim, eps=1e-6)
        self.rope = LTX2VideoVaeRotaryPosEmbed3D(head_dim, base=rope_base)

    def project_qkv(
        self, hidden_states: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Q/K/V as `(B, T, H, W, heads, head_dim)`: normed, query pre-scaled, rotated.

        The query carries the `1/sqrt(head_dim)` factor so attention runs with
        `scale=1.0` -- upstream's order is norm, scale, then rotate.
        """
        batch_size, num_frames, height, width, _ = hidden_states.shape
        shape = (batch_size, num_frames, height, width, self.heads, self.head_dim)
        query = self.to_q(hidden_states).view(shape)
        key = self.to_k(hidden_states).view(shape)
        value = self.to_v(hidden_states).view(shape)

        query = self.norm_q(query)
        key = self.norm_k(key)
        query = query * self.scale
        return self.rope(query), self.rope(key), value

    def build_block_mask(self, hidden_states: torch.Tensor):
        """The window mask for this grid, or `None` when NATTEN handles it.

        Fixed within a stage, so built once.
        """
        if _na3d() is not None:
            return None
        num_frames, height, width = hidden_states.shape[1:4]
        return _neighborhood_block_mask(
            num_frames, height, width, self.kernel_size, hidden_states.device
        )

    def forward(self, hidden_states: torch.Tensor, block_mask=None) -> torch.Tensor:
        batch_size, num_frames, height, width, _ = hidden_states.shape
        kernel_t, kernel_h, kernel_w = self.kernel_size
        if num_frames < kernel_t or height < kernel_h or width < kernel_w:
            raise ValueError(
                "Neighborhood attention requires each dim to be at least its "
                f"kernel size; got (T, H, W) = ({num_frames}, {height}, {width}) "
                f"with kernel_size {self.kernel_size}."
            )

        query, key, value = self.project_qkv(hidden_states)

        na3d = _na3d()
        if na3d is not None:
            # `project_qkv` already yields NATTEN's layout. scale=1.0: the
            # query is pre-scaled there.
            hidden_states = na3d(
                query, key, value, kernel_size=self.kernel_size, scale=1.0
            )
            hidden_states = hidden_states.reshape(
                batch_size, num_frames, height, width, self.heads * self.head_dim
            )
            return self.to_out[0](hidden_states)

        seq_len = num_frames * height * width
        # flex_attention wants (B, heads, S, head_dim).
        query = query.reshape(batch_size, seq_len, self.heads, self.head_dim).transpose(
            1, 2
        )
        key = key.reshape(batch_size, seq_len, self.heads, self.head_dim).transpose(
            1, 2
        )
        value = value.reshape(batch_size, seq_len, self.heads, self.head_dim).transpose(
            1, 2
        )

        if block_mask is None:
            block_mask = self.build_block_mask(hidden_states)

        # scale=1.0: the query is already scaled in project_qkv.
        hidden_states = _flex_attention_fn()(
            query, key, value, block_mask=block_mask, scale=1.0
        )
        hidden_states = hidden_states.transpose(1, 2).reshape(
            batch_size, num_frames, height, width, self.heads * self.head_dim
        )
        return self.to_out[0](hidden_states)


class LTX2VideoVaeSwiGLU(nn.Module):
    """`w_down(silu(w_gate(x)) * w_up(x))`, evaluated in token tiles."""

    def __init__(self, dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.w_up = nn.Linear(dim, hidden_dim, bias=False)
        self.w_gate = nn.Linear(dim, hidden_dim, bias=False)
        self.w_down = nn.Linear(hidden_dim, dim, bias=False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size, *token_dims, channels = hidden_states.shape
        num_tokens = math.prod(token_dims)
        if num_tokens <= _SWIGLU_TILE_SIZE:
            return self.w_down(
                F.silu(self.w_gate(hidden_states)) * self.w_up(hidden_states)
            )

        flat = hidden_states.reshape(batch_size, num_tokens, channels)
        out = torch.empty_like(flat)
        for start in range(0, num_tokens, _SWIGLU_TILE_SIZE):
            tile = flat[:, start : start + _SWIGLU_TILE_SIZE]
            out[:, start : start + _SWIGLU_TILE_SIZE] = self.w_down(
                F.silu(self.w_gate(tile)) * self.w_up(tile)
            )
        return out.reshape(hidden_states.shape)


def _swiglu_hidden_dim(dim: int, mlp_ratio: float) -> int:
    return (int(dim * mlp_ratio) + 15) // 16 * 16


class LTX2VideoVaeNABlock(nn.Module):
    """Pre-norm neighborhood-attention block used by the deterministic stages."""

    def __init__(
        self,
        dim: int,
        kernel_size: tuple[int, int, int],
        head_dim: int = 64,
        mlp_ratio: float = 4.0,
    ) -> None:
        super().__init__()
        self.norm1 = nn.RMSNorm(dim, eps=1e-6)
        self.attn = LTX2VideoVaeNeighborhoodAttention(
            dim, kernel_size, head_dim=head_dim
        )
        self.norm2 = nn.RMSNorm(dim, eps=1e-6)
        self.mlp = LTX2VideoVaeSwiGLU(dim, _swiglu_hidden_dim(dim, mlp_ratio))

    def forward(self, hidden_states: torch.Tensor, block_mask=None) -> torch.Tensor:
        hidden_states = hidden_states + self.attn(self.norm1(hidden_states), block_mask)
        hidden_states = hidden_states + self.mlp(self.norm2(hidden_states))
        return hidden_states


class LTX2VideoVaeAdaLNZero(nn.Module):
    """Timestep embedding to seven `(B, 1, 1, 1, C)` modulation chunks.

    Seven is upstream's shape (scale/shift/gate for attention and MLP, plus a
    context gate); only the four scale/shift chunks are consumed, since the
    residuals here are ungated.
    """

    def __init__(self, dim: int, t_emb_dim: int, num_chunks: int = 7) -> None:
        super().__init__()
        self.num_chunks = num_chunks
        self.proj = nn.Linear(t_emb_dim, num_chunks * dim, bias=True)

    def forward(self, t_emb: torch.Tensor) -> tuple[torch.Tensor, ...]:
        chunks = self.proj(F.silu(t_emb)).chunk(self.num_chunks, dim=-1)
        return tuple(chunk[:, None, None, None, :] for chunk in chunks)


class LTX2VideoVaeDiffusionNABlock(nn.Module):
    """Stage-5 block: neighborhood attention + SwiGLU under AdaLN-Zero modulation."""

    def __init__(
        self,
        dim: int,
        kernel_size: tuple[int, int, int],
        context_channels: int,
        head_dim: int = 64,
        mlp_ratio: float = 4.0,
        num_mod_params: int = 7,
    ) -> None:
        super().__init__()
        self.context_channels = context_channels
        self.num_mod_params = num_mod_params
        self.context_proj = nn.Linear(context_channels, dim, bias=True)
        self.scale_shift_table = nn.Parameter(torch.zeros(num_mod_params, dim))

        self.norm1 = nn.RMSNorm(dim, eps=1e-6)
        self.attn = LTX2VideoVaeNeighborhoodAttention(
            dim, kernel_size, head_dim=head_dim
        )
        self.norm2 = nn.RMSNorm(dim, eps=1e-6)
        self.mlp = LTX2VideoVaeSwiGLU(dim, _swiglu_hidden_dim(dim, mlp_ratio))

    def forward(
        self,
        hidden_states: torch.Tensor,
        latent_context: torch.Tensor,
        modulation: tuple[torch.Tensor, ...],
        block_mask=None,
    ) -> torch.Tensor:
        scale_msa, shift_msa, _, scale_mlp, shift_mlp, _, _ = [
            modulation[i] + self.scale_shift_table[i].view(1, 1, 1, 1, -1)
            for i in range(self.num_mod_params)
        ]

        hidden_states = hidden_states + self.context_proj(latent_context)
        hidden_states = hidden_states + self.attn(
            self.norm1(hidden_states) * (1 + scale_msa) + shift_msa, block_mask
        )
        hidden_states = hidden_states + self.mlp(
            self.norm2(hidden_states) * (1 + scale_mlp) + shift_mlp
        )
        return hidden_states


class LTX2VideoVaePixelShuffleUpsampler(nn.Module):
    """Linear channel expansion then a channels-last pixel shuffle.

    A temporal stride of 2 produces a duplicate leading frame, dropped to keep
    the causal 1:2 (composed 1:8) frame mapping.
    """

    def __init__(
        self,
        in_channels: int,
        stride: tuple[int, int, int],
        out_channels_reduction_factor: int = 1,
    ) -> None:
        super().__init__()
        self.stride = tuple(stride)
        proj_out_channels = (
            math.prod(self.stride) * in_channels // out_channels_reduction_factor
        )
        self.out_channels = proj_out_channels // math.prod(self.stride)
        self.proj = nn.Linear(in_channels, proj_out_channels, bias=True)

    def forward(
        self, hidden_states: torch.Tensor, drop_leading_frame: bool = True
    ) -> torch.Tensor:
        batch_size, num_frames, height, width, _ = hidden_states.shape
        stride_t, stride_h, stride_w = self.stride
        hidden_states = self.proj(hidden_states)
        hidden_states = hidden_states.reshape(
            batch_size,
            num_frames,
            height,
            width,
            self.out_channels,
            stride_t,
            stride_h,
            stride_w,
        )
        hidden_states = hidden_states.permute(0, 1, 5, 2, 6, 3, 7, 4)
        hidden_states = hidden_states.reshape(
            batch_size,
            num_frames * stride_t,
            height * stride_h,
            width * stride_w,
            self.out_channels,
        )
        if stride_t == 2 and drop_leading_frame:
            hidden_states = hidden_states[:, 1:]
        return hidden_states


class LTX2VideoDiffusionDecoder3d(nn.Module):
    """Stages 1-4 upsample the latent into a context volume; stage 5 denoises
    patchified pixels conditioned on it."""

    def __init__(self, config: LTX25DiffusionDecoderConfig) -> None:
        super().__init__()
        arch = config.arch_config
        stage_channels = tuple(arch.decoder_stage_channels)
        stage_depths = tuple(arch.decoder_stage_depths)
        stage_kernels = tuple(tuple(k) for k in arch.decoder_stage_kernels)
        upsample_strides = tuple(tuple(s) for s in arch.decoder_upsample_strides)
        reductions = tuple(arch.decoder_upsample_channel_reductions)

        if arch.decoder_model_output_type not in ("x0", "v"):
            raise ValueError(
                "decoder_model_output_type must be 'x0' or 'v', got "
                f"{arch.decoder_model_output_type!r}."
            )
        # An inconsistent pair would only fail deep inside the first block.
        for stage_idx, reduction in enumerate(reductions):
            expected = stage_channels[stage_idx] // reduction
            if stage_channels[stage_idx + 1] != expected:
                raise ValueError(
                    f"decoder_stage_channels[{stage_idx + 1}] must be "
                    f"{expected}, got {stage_channels[stage_idx + 1]}."
                )

        self.patch_size = arch.patch_size
        self.out_channels = arch.out_channels
        self.timestep_scale_multiplier = arch.decoder_timestep_scale_multiplier
        self.model_output_type = arch.decoder_model_output_type
        self.default_num_inference_steps = arch.decoder_num_inference_steps
        self.temporal_compression_ratio = arch.temporal_compression_ratio
        self.context_channels = stage_channels[-1]
        # Replicated through stages 1-4 and cropped before stage 5, moving the
        # border effect past the frames that are kept.
        self.trailing_pad_latent_frames = (stage_kernels[0][0] // 2) * 2

        self.conv_in = nn.Linear(arch.latent_channels, stage_channels[0], bias=True)

        self.det_stages = nn.ModuleList()
        self.upsamples = nn.ModuleList()
        for stage_idx, stride in enumerate(upsample_strides):
            channels = stage_channels[stage_idx]
            self.det_stages.append(
                nn.ModuleList(
                    [
                        LTX2VideoVaeNABlock(
                            dim=channels,
                            kernel_size=stage_kernels[stage_idx],
                            head_dim=arch.decoder_head_dim,
                        )
                        for _ in range(stage_depths[stage_idx])
                    ]
                )
            )
            self.upsamples.append(
                LTX2VideoVaePixelShuffleUpsampler(
                    in_channels=channels,
                    stride=stride,
                    out_channels_reduction_factor=reductions[stage_idx],
                )
            )

        self.t_embedder = LTX2VideoDecoderCombinedTimestepEmbeddings(
            embedding_dim=arch.decoder_t_emb_dim
        )

        stage5_channels = stage_channels[-1]
        noised_pixel_channels = arch.out_channels * arch.patch_size**2
        self.conv_in_x_t = nn.Linear(noised_pixel_channels, stage5_channels, bias=True)
        self.shared_adaln = LTX2VideoVaeAdaLNZero(
            dim=stage5_channels, t_emb_dim=arch.decoder_t_emb_dim
        )
        self.diff_blocks = nn.ModuleList(
            [
                LTX2VideoVaeDiffusionNABlock(
                    dim=stage5_channels,
                    kernel_size=tuple(arch.decoder_stage5_kernel),
                    context_channels=self.context_channels,
                    head_dim=arch.decoder_head_dim,
                    num_mod_params=self.shared_adaln.num_chunks,
                )
                for _ in range(stage_depths[-1])
            ]
        )
        self.norm_out = nn.RMSNorm(stage5_channels, eps=1e-6)
        self.conv_out = nn.Linear(stage5_channels, noised_pixel_channels, bias=True)

    def forward_stages_1_to_3(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Latent `(B, C, T, H, W)` to a channels-last feature volume."""
        num_pad = self.trailing_pad_latent_frames
        if num_pad > 0:
            trailing = hidden_states[:, :, -1:].expand(-1, -1, num_pad, -1, -1)
            hidden_states = torch.cat([hidden_states, trailing], dim=2)

        hidden_states = hidden_states.permute(0, 2, 3, 4, 1)
        hidden_states = self.conv_in(hidden_states)
        for blocks, upsample in zip(self.det_stages[:-1], self.upsamples[:-1]):
            # Fixed within a stage, so one mask serves all of its blocks.
            block_mask = blocks[0].attn.build_block_mask(hidden_states)
            for block in blocks:
                hidden_states = block(hidden_states, block_mask)
            hidden_states = upsample(hidden_states)
        return hidden_states

    def forward_stage_4(
        self,
        hidden_states: torch.Tensor,
        drop_leading_frame: bool = True,
        crop_trailing_ghost: bool = True,
    ) -> torch.Tensor:
        """Last deterministic stage -> context `(B, T5, H5, W5, C5)`."""
        blocks = self.det_stages[-1]
        block_mask = blocks[0].attn.build_block_mask(hidden_states)
        for block in blocks:
            hidden_states = block(hidden_states, block_mask)
        hidden_states = self.upsamples[-1](
            hidden_states, drop_leading_frame=drop_leading_frame
        )

        num_pad = self.trailing_pad_latent_frames
        if crop_trailing_ghost and num_pad > 0:
            hidden_states = hidden_states[
                :, : -num_pad * self.temporal_compression_ratio
            ]
        return hidden_states

    def forward_diffusion_step(
        self, latent_context: torch.Tensor, x_t: torch.Tensor, timestep: torch.Tensor
    ) -> torch.Tensor:
        """One stage-5 step; returns a pixel-space prediction `(B, C, F, H, W)`."""
        t_emb = self.t_embedder(
            self.timestep_scale_multiplier * timestep,
            hidden_dtype=latent_context.dtype,
        )
        modulation = self.shared_adaln(t_emb)

        hidden_states = _patchify(x_t, self.patch_size).permute(0, 2, 3, 4, 1)
        hidden_states = self.conv_in_x_t(hidden_states)
        block_mask = self.diff_blocks[0].attn.build_block_mask(hidden_states)
        for block in self.diff_blocks:
            hidden_states = block(hidden_states, latent_context, modulation, block_mask)

        hidden_states = self.norm_out(hidden_states)
        hidden_states = self.conv_out(hidden_states)
        hidden_states = hidden_states.permute(0, 4, 1, 2, 3).contiguous()
        return _unpatchify(hidden_states, self.patch_size)

    def denoise(
        self,
        latent_context: torch.Tensor,
        x_t: torch.Tensor,
        num_inference_steps: int,
    ) -> torch.Tensor:
        batch_size = latent_context.shape[0]
        timesteps = torch.linspace(
            1.0,
            1.0 / num_inference_steps,
            num_inference_steps,
            device=latent_context.device,
            dtype=torch.float32,
        )

        # How LTX-2.5 ships: one step whose x0 prediction is the output.
        if num_inference_steps == 1 and self.model_output_type == "x0":
            return self.forward_diffusion_step(
                latent_context, x_t, timesteps[:1].expand(batch_size)
            )

        for step_idx in range(num_inference_steps):
            t_now = timesteps[step_idx].expand(batch_size)
            t_next = (
                timesteps[step_idx + 1]
                if step_idx + 1 < num_inference_steps
                else torch.zeros_like(t_now)
            )
            model_out = self.forward_diffusion_step(latent_context, x_t, t_now).float()
            x_t_fp32 = x_t.float()
            if self.model_output_type == "x0":
                sigma = t_now.view(-1, *([1] * (x_t.ndim - 1)))
                model_out = (x_t_fp32 - model_out) / sigma
            dt = (t_now - t_next).view(-1, *([1] * (x_t.ndim - 1)))
            x_t = (x_t_fp32 - dt * model_out).to(x_t.dtype)
        return x_t

    def forward(
        self,
        hidden_states: torch.Tensor,
        generator: torch.Generator | None = None,
        num_inference_steps: int | None = None,
    ) -> torch.Tensor:
        num_inference_steps = num_inference_steps or self.default_num_inference_steps
        latent_context = self.forward_stage_4(self.forward_stages_1_to_3(hidden_states))
        # Pixel canvas = stage-5 token grid times the patch size.
        pixel_shape = (
            hidden_states.shape[0],
            self.out_channels,
            latent_context.shape[1],
            latent_context.shape[2] * self.patch_size,
            latent_context.shape[3] * self.patch_size,
        )
        x_t = torch.randn(
            pixel_shape,
            generator=generator,
            device=hidden_states.device,
            dtype=hidden_states.dtype,
        )
        return self.denoise(latent_context, x_t, num_inference_steps)


def _tile_intervals(
    length: int, tile_size: int, stride: int, min_size: int
) -> list[tuple[int, int]]:
    """Overlapping `[start, end)` tiles covering `[0, length)`.

    A trailing remnant shorter than `min_size` is merged into the previous tile
    rather than decoded alone: neighborhood attention rejects any grid smaller
    than its kernel, so a short remnant cannot always stand on its own.
    """
    if length <= tile_size:
        return [(0, length)]
    starts = list(range(0, length, stride))
    while len(starts) > 1 and length - starts[-1] < min_size:
        starts.pop()
    return [(start, min(start + tile_size, length)) for start in starts[:-1]] + [
        (starts[-1], length)
    ]


class LTX2VideoDiffusionDecoderModel(nn.Module, LayerwiseOffloadableModuleMixin):
    """Checkpoint-level wrapper: the decoder plus the latent statistics.

    `diffusion_decoder/` stores `latents_mean` / `latents_std` alongside a
    `decoder.` submodule, so this mirrors that layout rather than flattening it.
    """

    layerwise_offload_dit_group_enabled = False

    def __init__(self, config: LTX25DiffusionDecoderConfig) -> None:
        super().__init__()
        self.config = config
        latent_channels = config.arch_config.latent_channels
        self.decoder = LTX2VideoDiffusionDecoder3d(config)
        self.layer_names = [
            *(
                f"decoder.det_stages.{index}"
                for index in range(len(self.decoder.det_stages))
            ),
            "decoder.diff_blocks",
        ]
        self.register_buffer(
            "latents_mean", torch.zeros(latent_channels), persistent=True
        )
        self.register_buffer(
            "latents_std", torch.ones(latent_channels), persistent=True
        )

        # Tiles the last deterministic stage and the diffusion blocks, so the
        # output only moves near tile borders. Set by the decoding stage from
        # `--diffusion-decoder-tiling`; tile sizes match upstream.
        self.use_tiling = False
        self.tile_sample_min_height = 768
        self.tile_sample_min_width = 768
        self.tile_sample_min_num_frames = 32
        self.tile_sample_stride_height = 512
        self.tile_sample_stride_width = 512
        self.tile_sample_stride_num_frames = 16

    @staticmethod
    def _blend(a: torch.Tensor, b: torch.Tensor, extent: int, dim: int) -> torch.Tensor:
        """Linear cross-fade of `a`'s tail into `b`'s head along `dim`."""
        extent = min(a.shape[dim], b.shape[dim], extent)
        if extent <= 0:
            return b
        ramp = torch.arange(extent, device=b.device, dtype=torch.float32) / extent
        shape = [1] * b.ndim
        shape[dim] = extent
        ramp = ramp.reshape(shape).to(b.dtype)
        a_tail = a.narrow(dim, a.shape[dim] - extent, extent)
        b_head = b.narrow(dim, 0, extent)
        b_head.copy_(a_tail * (1 - ramp) + b_head * ramp)
        return b

    def _should_tile(self, hidden_states: torch.Tensor) -> bool:
        if not self.use_tiling:
            return False
        arch = self.config.arch_config
        return (
            hidden_states.shape[2]
            > self.tile_sample_min_num_frames // arch.temporal_compression_ratio
            or hidden_states.shape[3]
            > self.tile_sample_min_height // arch.spatial_compression_ratio
            or hidden_states.shape[4]
            > self.tile_sample_min_width // arch.spatial_compression_ratio
        )

    def tiled_decode(
        self,
        hidden_states: torch.Tensor,
        generator: torch.Generator | None = None,
        num_inference_steps: int | None = None,
    ) -> torch.Tensor:
        """Decode with stage 4 and the diffusion stage running per tile.

        Tiles live on the grid entering the last deterministic stage, where one
        cell maps to a fixed block of output pixels. Temporal tiles follow the
        causal frame mapping: only the tile holding t=0 drops the temporal
        upsample's duplicate leading frame, and only the tile holding the end of
        the video carries the border padding to crop.
        """
        decoder = self.decoder
        arch = self.config.arch_config
        num_inference_steps = num_inference_steps or decoder.default_num_inference_steps
        batch_size = hidden_states.shape[0]
        patch_size = decoder.patch_size

        # Pixels per tiling-grid cell: the last upsample's stride times the patch.
        stride_up = decoder.upsamples[-1].stride
        scale_t, scale_h, scale_w = (
            stride_up[0],
            stride_up[1] * patch_size,
            stride_up[2] * patch_size,
        )
        tile_t = self.tile_sample_min_num_frames // scale_t
        step_t = self.tile_sample_stride_num_frames // scale_t
        tile_h = self.tile_sample_min_height // scale_h
        step_h = self.tile_sample_stride_height // scale_h
        tile_w = self.tile_sample_min_width // scale_w
        step_w = self.tile_sample_stride_width // scale_w
        # Stage 4 sees the tile as-is, stage 5 sees it scaled by the stride.
        min_sizes = [
            max(k4, -(-k5 // stride))
            for k4, k5, stride in zip(
                arch.decoder_stage_kernels[-1], arch.decoder_stage5_kernel, stride_up
            )
        ]

        features = decoder.forward_stages_1_to_3(hidden_states)
        # Trailing ghost frames replicate through the earlier temporal
        # upsamples; the composed mapping is affine in their stride product.
        ghost = decoder.trailing_pad_latent_frames * math.prod(
            up.stride[0] for up in decoder.upsamples[:-1]
        )
        num_frames = features.shape[1] - ghost
        height, width = features.shape[2], features.shape[3]

        temporal_tiles = _tile_intervals(num_frames, tile_t, step_t, min_sizes[0])
        height_tiles = _tile_intervals(height, tile_h, step_h, min_sizes[1])
        width_tiles = _tile_intervals(width, tile_w, step_w, min_sizes[2])
        blend_frames = (tile_t - step_t) * scale_t
        blend_height = (tile_h - step_h) * scale_h
        blend_width = (tile_w - step_w) * scale_w

        # Single-step x0 predicts from pure noise, so tiles may draw their own.
        # Multi-step integrates across steps and needs one shared canvas.
        single_step_x0 = num_inference_steps == 1 and decoder.model_output_type == "x0"
        x_t_full = None
        if not single_step_x0:
            pixel_frames = num_frames * scale_t - (1 if scale_t == 2 else 0)
            x_t_full = torch.randn(
                (
                    batch_size,
                    decoder.out_channels,
                    pixel_frames,
                    height * scale_h,
                    width * scale_w,
                ),
                generator=generator,
                device=hidden_states.device,
                dtype=hidden_states.dtype,
            )

        frame_groups = []
        for t0, t1 in temporal_tiles:
            is_origin = t0 == 0
            is_trailing = t1 == num_frames
            feature_t1 = features.shape[1] if is_trailing else t1
            rows = []
            for h0, h1 in height_tiles:
                row = []
                for w0, w1 in width_tiles:
                    context = decoder.forward_stage_4(
                        features[:, t0:feature_t1, h0:h1, w0:w1],
                        drop_leading_frame=is_origin,
                        crop_trailing_ghost=is_trailing,
                    )
                    tile_shape = (
                        batch_size,
                        decoder.out_channels,
                        context.shape[1],
                        context.shape[2] * patch_size,
                        context.shape[3] * patch_size,
                    )
                    if single_step_x0:
                        x_t = torch.randn(
                            tile_shape,
                            generator=generator,
                            device=hidden_states.device,
                            dtype=hidden_states.dtype,
                        )
                    else:
                        # A non-origin tile keeps its duplicate leading frame, so
                        # it starts one pixel frame earlier than t0 * scale_t.
                        pixel_t0 = t0 * scale_t - (
                            1 if not is_origin and scale_t == 2 else 0
                        )
                        x_t = x_t_full[
                            :,
                            :,
                            pixel_t0 : pixel_t0 + tile_shape[2],
                            h0 * scale_h : h0 * scale_h + tile_shape[3],
                            w0 * scale_w : w0 * scale_w + tile_shape[4],
                        ]
                    row.append(decoder.denoise(context, x_t, num_inference_steps))
                rows.append(row)

            result_rows = []
            for i, row in enumerate(rows):
                result_row = []
                for j, tile in enumerate(row):
                    if i > 0:
                        tile = self._blend(rows[i - 1][j], tile, blend_height, dim=3)
                    if j > 0:
                        tile = self._blend(row[j - 1], tile, blend_width, dim=4)
                    # The last tile can run past the stride grid, since a short
                    # remnant is merged into it, so it keeps its full extent.
                    keep_h = step_h * scale_h if i < len(rows) - 1 else tile.shape[3]
                    keep_w = step_w * scale_w if j < len(row) - 1 else tile.shape[4]
                    result_row.append(tile[:, :, :, :keep_h, :keep_w])
                result_rows.append(torch.cat(result_row, dim=4))
            frame_groups.append(torch.cat(result_rows, dim=3))

        result = []
        for k, group in enumerate(frame_groups):
            if k > 0:
                group = self._blend(frame_groups[k - 1], group, blend_frames, dim=2)
            if k < len(frame_groups) - 1:
                # The origin group is one frame short of stride * scale: its
                # first cell decodes to a single pixel frame under the causal
                # mapping.
                keep_frames = step_t * scale_t - (1 if k == 0 and scale_t == 2 else 0)
                group = group[:, :, :keep_frames]
            result.append(group)
        return torch.cat(result, dim=2)

    def forward(
        self,
        hidden_states: torch.Tensor,
        generator: torch.Generator | None = None,
        num_inference_steps: int | None = None,
    ) -> torch.Tensor:
        if self._should_tile(hidden_states):
            return self.tiled_decode(hidden_states, generator, num_inference_steps)
        return self.decoder(hidden_states, generator, num_inference_steps)

    def decode(
        self,
        hidden_states: torch.Tensor,
        generator: torch.Generator | None = None,
        num_inference_steps: int | None = None,
    ) -> torch.Tensor:
        return self.forward(hidden_states, generator, num_inference_steps)


EntryClass = LTX2VideoDiffusionDecoderModel
