# SPDX-License-Identifier: Apache-2.0
# Adapted from: https://github.com/Tencent-Hunyuan/Hy-Nomas
"""Native PixelUMM decoder for Nomas: the ported network and the SGLang decode contract.

The first part is the GenExpert transformer adapted from HY-Nomas; the second
part adds upstream temporal windowing, seeded noise, the stable-prefix
streaming timeline and the ``.decode(latent)`` interface the common
DecodingStage consumes. No encoder, spatial tiling or cross-request state.
"""

from __future__ import annotations

import math
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass, fields
from functools import lru_cache

import torch
from msgspec import Struct
from torch import nn
from torch.nn import functional as F

from sglang.multimodal_gen.configs.models.vaes.nomas import NomasVAEConfig
from sglang.multimodal_gen.runtime.distributed.parallel_state import (
    get_decode_parallel_group_coordinator,
    get_decode_parallel_world_size,
    model_parallel_is_initialized,
)

TUBELET = (4, 16, 16)
TUBE_DIM = TUBELET[0] * TUBELET[1] * TUBELET[2] * 3
LATENT_CHANNELS = 32
# A DCAE latent cell is (4, 32, 32) pixels: 1 tubelet in time, 2 x 2 in space.
LATENT_TOKEN_GRID = 2
# Target FP32 bytes per fused Q/K chunk; only the squares are materialised.
_QK_FUSED_CHUNK_BYTES = 256 * 2**20


@dataclass(frozen=True)
class GenExpertArch:
    """Qwen3-8B (hidden 4096) with PixelUMM's SenseNova T/H/W MRoPE."""

    hidden_size: int = 4096
    intermediate_size: int = 12288
    num_layers: int = 36
    num_heads: int = 32
    num_kv_heads: int = 8
    head_dim: int = 128
    rms_norm_eps: float = 1e-6
    rope_theta_t: float = 1_000_000.0
    rope_theta_hw: float = 10_000.0

    def __post_init__(self):
        if self.head_dim % 4 or self.num_heads % self.num_kv_heads:
            raise ValueError("head_dim must be divisible by 4 and heads by kv heads")

    @property
    def rope_sections(self) -> tuple[int, int, int]:
        """Head-dim slices rotated by the T, H and W position respectively."""
        return self.head_dim // 2, self.head_dim // 4, self.head_dim // 4


QWEN3_8B_GEN = GenExpertArch()


def patchify(video: torch.Tensor) -> torch.Tensor:
    """[B,3,T,H,W] -> [B, (T/4)(H/16)(W/16), 3072]; token order T,H,W, values (t,p,q,c)."""
    if video.ndim != 5 or video.shape[1] != 3:
        raise ValueError(f"expected RGB video [B,3,T,H,W], got {tuple(video.shape)}")
    pt, ph, pw = TUBELET
    batch, channels, frames, height, width = video.shape
    if frames % pt or height % ph or width % pw:
        raise ValueError(
            f"video {tuple(video.shape[2:])} must be a multiple of {TUBELET}"
        )
    x = video.reshape(
        batch, channels, frames // pt, pt, height // ph, ph, width // pw, pw
    )
    return x.permute(0, 2, 4, 6, 3, 5, 7, 1).reshape(batch, -1, TUBE_DIM)


def unpatchify(tokens: torch.Tensor, grid: tuple[int, int, int]) -> torch.Tensor:
    batch, count, dim = tokens.shape
    nt, nh, nw = grid
    pt, ph, pw = TUBELET
    if count != nt * nh * nw or dim != TUBE_DIM:
        raise ValueError(f"tokens {tuple(tokens.shape)} do not match grid {grid}")
    x = tokens.reshape(batch, nt, nh, nw, pt, ph, pw, 3)
    return x.permute(0, 7, 1, 4, 2, 5, 3, 6).reshape(
        batch, 3, nt * pt, nh * ph, nw * pw
    )


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)


def mrope_tables(
    arch: GenExpertArch, grid: tuple[int, int, int], device
) -> tuple[torch.Tensor, torch.Tensor]:
    """cos/sin [1, T*H*W, 1, head_dim] for a THW-ordered grid, laid out [T | H | W]."""
    positions = torch.meshgrid(
        *(torch.arange(n, device=device) for n in grid), indexing="ij"
    )
    thetas = (arch.rope_theta_t, arch.rope_theta_hw, arch.rope_theta_hw)
    cos_parts, sin_parts = [], []
    for position, dim, theta in zip(positions, arch.rope_sections, thetas):
        inv_freq = theta ** (
            -torch.arange(0, dim, 2, device=device, dtype=torch.float32) / dim
        )
        angles = position.flatten().float()[:, None] * inv_freq[None]
        angles = torch.cat((angles, angles), dim=-1)
        cos_parts.append(angles.cos())
        sin_parts.append(angles.sin())
    return (
        torch.cat(cos_parts, dim=-1)[None, :, None, :],
        torch.cat(sin_parts, dim=-1)[None, :, None, :],
    )


def apply_mrope(
    x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, sections
) -> torch.Tensor:
    """x [B,N,H,D]; each section is rotated within itself (rotate_half per axis)."""
    rotated = torch.cat(
        [_rotate_half(part) for part in x.split(tuple(sections), dim=-1)], dim=-1
    )
    return x * cos + rotated * sin


def _fused_ok(x: torch.Tensor) -> bool:
    return (
        x.is_cuda and not torch.is_grad_enabled() and not torch.compiler.is_compiling()
    )


def _compile_exact(fn):
    # emulate_precision_casts keeps every intermediate rounding, so the fused
    # elementwise kernels are bit-identical to the eager sequences they replace.
    # Mean-square reductions stay in ATen to keep their summation order.
    return torch.compile(
        fn, dynamic=True, fullgraph=True, options={"emulate_precision_casts": True}
    )


def _square(x):
    return x.float().pow(2)


def _rms_scale(x, mean_square, eps, weight, dtype):
    return weight * (x.float() * torch.rsqrt(mean_square + eps)).to(dtype)


def _qk_norm_mrope(x, ms_t, ms_hw, eps, w_t, w_hw, cos, sin, sections, dtype):
    t_part, hw_part = x.float().chunk(2, dim=-1)
    normalized = torch.cat(
        (
            w_t * (t_part * torch.rsqrt(ms_t + eps)),
            w_hw * (hw_part * torch.rsqrt(ms_hw + eps)),
        ),
        dim=-1,
    )
    return apply_mrope(normalized, cos, sin, sections).to(dtype)


@lru_cache(maxsize=1)
def _compiled_square():
    return _compile_exact(_square)


@lru_cache(maxsize=1)
def _compiled_rms_scale():
    return _compile_exact(_rms_scale)


@lru_cache(maxsize=1)
def _compiled_qk_norm_mrope():
    return _compile_exact(_qk_norm_mrope)


def _silu_mul(gate, up):
    return F.silu(gate) * up


@lru_cache(maxsize=1)
def _compiled_silu_mul():
    return _compile_exact(_silu_mul)


def _mean_square(x):
    return _compiled_square()(x).mean(-1, keepdim=True)


@lru_cache(maxsize=1)
def _block_mask_builder():
    from torch.nn.attention.flex_attention import create_block_mask

    return torch.compile(create_block_mask, dynamic=True, fullgraph=True)


def local_attention_layout(grid: tuple[int, int, int], tile: int, device):
    """``(perm, inverse, block_mask)`` of the sliding-tile attention, or None when the plane has
    at most 3 x 3 tiles (full attention). ``tokens[:, perm]`` is the tile-major order the mask is
    built in; cached per shape."""
    _, height, width = grid
    if -(-height // tile) <= 3 and -(-width // tile) <= 3:
        return None
    return _cached_local_layout(tuple(grid), tile, str(device))


@lru_cache(maxsize=8)
def _cached_local_layout(grid, tile, device):
    from torch.nn.attention.flex_attention import create_block_mask

    frames, height, width = grid
    tiles_h, tiles_w = -(-height // tile), -(-width // tile)
    t, h, w = torch.meshgrid(
        *(torch.arange(n, device=device) for n in grid), indexing="ij"
    )
    tile_h, tile_w = (h // tile).flatten(), (w // tile).flatten()
    order = (
        ((tile_h * tiles_w + tile_w) * frames + t.flatten()) * tile
        + (h % tile).flatten()
    ) * tile
    perm = torch.argsort(order + (w % tile).flatten())
    tile_h, tile_w = tile_h[perm], tile_w[perm]
    center_h = tile_h.clamp(1, max(tiles_h - 2, 1))
    center_w = tile_w.clamp(1, max(tiles_w - 2, 1))

    def mask_mod(batch, head, q_idx, kv_idx):
        return ((center_h[q_idx] - tile_h[kv_idx]).abs() <= 1) & (
            (center_w[q_idx] - tile_w[kv_idx]).abs() <= 1
        )

    count = frames * height * width
    on_cuda = torch.device(device).type == "cuda"
    if not on_cuda and count * count > 16_777_216:
        raise ValueError("PixelUMM CPU reference mask is too large; use CUDA")
    builder = _block_mask_builder() if on_cuda else create_block_mask
    block_mask = builder(mask_mod, None, None, count, count, device=device)
    return perm, torch.argsort(perm), block_mask


def _flex_attention(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, block_mask
) -> torch.Tensor:
    """[B,heads,N,D] x GQA [B,kv,N,D]. Compiled (fused Triton kernel) on CUDA; eager elsewhere.

    The compile is required, not an optimization: eager flex materializes the full score matrix.
    """
    from torch.nn.attention.flex_attention import flex_attention

    if q.device.type != "cuda" or torch.compiler.is_compiling():
        return flex_attention(q, k, v, block_mask=block_mask, enable_gqa=True)
    return _compiled_flex()(q, k, v, block_mask=block_mask, enable_gqa=True)


@lru_cache(maxsize=1)
def _compiled_flex():
    from torch.nn.attention.flex_attention import flex_attention

    # Never allow an eager quadratic-memory fallback after graph specialization.
    return torch.compile(flex_attention, dynamic=True, fullgraph=True)


class RMSNorm(nn.Module):
    """Qwen3RMSNorm: normalize in fp32, cast back to the input dtype, then scale."""

    def __init__(self, dim: int, eps: float):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        if _fused_ok(x):
            return _compiled_rms_scale()(
                x, _mean_square(x), self.eps, self.weight, dtype
            )
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return self.weight * x.to(dtype)


class GenAttention(nn.Module):
    """GQA, split T / HW QK-norm, T/H/W MRoPE; full or sliding-tile attention."""

    def __init__(self, arch: GenExpertArch):
        super().__init__()
        self.arch = arch
        hidden, heads, kv, dim = (
            arch.hidden_size,
            arch.num_heads,
            arch.num_kv_heads,
            arch.head_dim,
        )
        self.q_proj = nn.Linear(hidden, heads * dim, bias=False)
        self.k_proj = nn.Linear(hidden, kv * dim, bias=False)
        self.v_proj = nn.Linear(hidden, kv * dim, bias=False)
        self.o_proj = nn.Linear(heads * dim, hidden, bias=False)
        self.q_norm_t = RMSNorm(dim // 2, arch.rms_norm_eps)
        self.q_norm_hw = RMSNorm(dim // 2, arch.rms_norm_eps)
        self.k_norm_t = RMSNorm(dim // 2, arch.rms_norm_eps)
        self.k_norm_hw = RMSNorm(dim // 2, arch.rms_norm_eps)

    @staticmethod
    def _split_norm(x: torch.Tensor, norm_t: RMSNorm, norm_hw: RMSNorm) -> torch.Tensor:
        t_part, hw_part = x.chunk(2, dim=-1)
        return torch.cat((norm_t(t_part), norm_hw(hw_part)), dim=-1)

    def _prepare_qk(self, x, cos, sin, norm_t, norm_hw, dtype):
        if _fused_ok(x):
            return self._prepare_qk_fused(x, cos, sin, norm_t, norm_hw, dtype)
        # CPU / autograd reference: FP32 QK norm, MRoPE, then the cast.
        normalized = self._split_norm(x.float(), norm_t, norm_hw)
        return apply_mrope(normalized, cos, sin, self.arch.rope_sections).to(dtype)

    def _prepare_qk_fused(self, x, cos, sin, norm_t, norm_hw, dtype):
        """Same values as the eager path: ATen mean squares, one fused kernel for the rest."""
        half = x.shape[-1] // 2
        result = torch.empty_like(x, dtype=dtype)
        # Only the FP32 squares are materialised now, so far larger chunks fit.
        chunk = max(
            1, _QK_FUSED_CHUNK_BYTES // (x.shape[0] * x.shape[2] * x.shape[3] * 4)
        )
        for first in range(0, x.shape[1], chunk):
            part = x[:, first : first + chunk]
            result[:, first : first + chunk] = _compiled_qk_norm_mrope()(
                part,
                _mean_square(part[..., :half]),
                _mean_square(part[..., half:]),
                norm_t.eps,
                norm_t.weight,
                norm_hw.weight,
                cos[:, first : first + chunk],
                sin[:, first : first + chunk],
                self.arch.rope_sections,
                dtype,
            )
        return result

    def forward(
        self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, block_mask=None
    ) -> torch.Tensor:
        arch = self.arch
        batch, count, _ = x.shape
        q = self.q_proj(x).view(batch, count, arch.num_heads, arch.head_dim)
        k = self.k_proj(x).view(batch, count, arch.num_kv_heads, arch.head_dim)
        v = self.v_proj(x).view(batch, count, arch.num_kv_heads, arch.head_dim)
        # PixelUMM's generation path keeps q/k in fp32 through QK-norm and RoPE.
        q = self._prepare_qk(q, cos, sin, self.q_norm_t, self.q_norm_hw, v.dtype)
        k = self._prepare_qk(k, cos, sin, self.k_norm_t, self.k_norm_hw, v.dtype)
        if block_mask is not None:
            out = _flex_attention(
                q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), block_mask
            )
        else:
            repeat = arch.num_heads // arch.num_kv_heads
            k, v = (
                k.repeat_interleave(repeat, dim=2),
                v.repeat_interleave(repeat, dim=2),
            )
            out = F.scaled_dot_product_attention(
                q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
            )
        return self.o_proj(
            out.transpose(1, 2).reshape(batch, count, arch.num_heads * arch.head_dim)
        )


class GenMLP(nn.Module):
    def __init__(self, arch: GenExpertArch):
        super().__init__()
        self.gate_proj = nn.Linear(arch.hidden_size, arch.intermediate_size, bias=False)
        self.up_proj = nn.Linear(arch.hidden_size, arch.intermediate_size, bias=False)
        self.down_proj = nn.Linear(arch.intermediate_size, arch.hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate, up = self.gate_proj(x), self.up_proj(x)
        fused = _compiled_silu_mul() if _fused_ok(x) else _silu_mul
        return self.down_proj(fused(gate, up))


class GenLayer(nn.Module):
    def __init__(self, arch: GenExpertArch):
        super().__init__()
        self.input_norm = RMSNorm(arch.hidden_size, arch.rms_norm_eps)
        self.attn = GenAttention(arch)
        self.post_attn_norm = RMSNorm(arch.hidden_size, arch.rms_norm_eps)
        self.mlp = GenMLP(arch)

    def forward(
        self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, block_mask=None
    ) -> torch.Tensor:
        x = x + self.attn(self.input_norm(x), cos, sin, block_mask)
        return x + self.mlp(self.post_attn_norm(x))


class PixelHead(nn.Module):
    """PixelUMM ``VideoUnpatchifyHead``: RMSNorm(eps=1e-6) -> Linear(hidden -> 3072)."""

    def __init__(self, hidden: int):
        super().__init__()
        self.norm = nn.RMSNorm(hidden, eps=1e-6)
        self.linear = nn.Linear(hidden, TUBE_DIM, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(self.norm(x))


class DCAEVPixelUMMDecoder(nn.Module):
    """``forward(latent [B,32,L,h,w], xt [B,3,4L,32h,32w]) -> x0 prediction``."""

    def __init__(
        self, arch: GenExpertArch = QWEN3_8B_GEN, *, local_attention_tile: int = 0
    ):
        super().__init__()
        self.arch = arch
        self.local_attention_tile = int(local_attention_tile)
        hidden = arch.hidden_size
        self.embed = nn.Linear(TUBE_DIM, hidden)
        self.cond_proj = nn.Linear(LATENT_CHANNELS, LATENT_TOKEN_GRID**2 * hidden)
        self.layers = nn.ModuleList(GenLayer(arch) for _ in range(arch.num_layers))
        self.final_norm = RMSNorm(hidden, arch.rms_norm_eps)
        self.head = PixelHead(hidden)

    @staticmethod
    def pixel_shape(latent: torch.Tensor) -> tuple[int, int, int, int, int]:
        batch, _, frames, height, width = latent.shape
        return batch, 3, frames * TUBELET[0], height * 32, width * 32

    def _condition_tokens(self, latent: torch.Tensor) -> torch.Tensor:
        batch, _, frames, height, width = latent.shape
        g = LATENT_TOKEN_GRID
        x = self.cond_proj(latent.permute(0, 2, 3, 4, 1))  # [B,L,h,w,4*hidden]
        x = x.view(batch, frames, height, width, g, g, self.arch.hidden_size)
        return x.permute(0, 1, 2, 4, 3, 5, 6).reshape(batch, -1, self.arch.hidden_size)

    def forward(self, latent: torch.Tensor, xt: torch.Tensor) -> torch.Tensor:
        if latent.ndim != 5 or latent.shape[1] != LATENT_CHANNELS:
            raise ValueError(
                f"expected DCAE latent [B,{LATENT_CHANNELS},L,h,w], got {tuple(latent.shape)}"
            )
        if tuple(xt.shape) != self.pixel_shape(latent):
            raise ValueError(
                f"xt {tuple(xt.shape)} != expected {self.pixel_shape(latent)}"
            )
        dtype = self.embed.weight.dtype
        context = (
            torch.autocast(device_type=latent.device.type, dtype=dtype)
            if dtype in (torch.bfloat16, torch.float16)
            else nullcontext()
        )
        _, _, frames, height, width = latent.shape
        grid = (frames, height * LATENT_TOKEN_GRID, width * LATENT_TOKEN_GRID)
        with context:
            tokens = self.embed(patchify(xt.to(dtype))) + self._condition_tokens(
                latent.to(dtype)
            )
            cos, sin = mrope_tables(self.arch, grid, latent.device)
            layout = (
                local_attention_layout(grid, self.local_attention_tile, latent.device)
                if self.local_attention_tile
                else None
            )
            block_mask = None
            if layout is not None:
                perm, inverse, block_mask = layout
                tokens, cos, sin = tokens[:, perm], cos[:, perm], sin[:, perm]
            for layer in self.layers:
                tokens = layer(tokens, cos, sin, block_mask)
            tokens = self.final_norm(tokens)
            if layout is not None:
                tokens = tokens[:, inverse]
        # The source runs the head after leaving autocast (forward_features returns first).
        return unpatchify(self.head(tokens), grid)


class NomasDecodeOutput(Struct):
    sample: torch.Tensor


def window_starts(length, window, stride):
    if length < 1 or not 1 <= stride <= window:
        raise ValueError("Invalid PixelUMM temporal window")
    if length <= window:
        return [0]
    starts = list(range(0, length - window + 1, stride))
    if starts[-1] != length - window:
        starts.append(length - window)
    return starts


def merge_windows(parts, starts):
    result = parts[0]
    for part, start in zip(parts[1:], starts[1:]):
        overlap = result.shape[2] - start
        if overlap == 0:
            result = torch.cat((result, part), dim=2)
            continue
        if not 0 < overlap < part.shape[2]:
            raise ValueError("PixelUMM windows must extend the decoded timeline")
        position = torch.arange(1, overlap + 1, device=part.device, dtype=torch.float32)
        weight = 0.5 - 0.5 * torch.cos(math.pi * position / (overlap + 1))
        weight = weight.view(1, 1, -1, 1, 1).to(part.dtype)
        result = torch.cat(
            (
                result[:, :, :-overlap],
                result[:, :, -overlap:] * (1 - weight) + part[:, :, :overlap] * weight,
                part[:, :, overlap:],
            ),
            dim=2,
        )
    return result


class NomasStreamingDecoder:
    """Append-only PixelUMM timeline that emits only future-stable frames.

    Mirrors upstream ``WindowStreamingDecodeSession.push``/``finish`` (the
    stable-prefix protocol of the 1-step decoder). Windows sit on one global
    stride grid and draw noise by absolute latent index, so the frames emitted
    over a session, concatenated, equal one full-clip decode of every pushed
    latent with each push's padded tail frames dropped. A frame is emitted once
    no later window can overlap it; ``finish`` emits the rest.

    ``decode_fn(latents, frame_offset)`` returns the raw (merged, not yet
    normalized) decoder pixels of ``latents``, whose first frame is absolute
    latent ``frame_offset``.
    """

    def __init__(self, window: int, overlap: int, temporal_ratio: int = 4):
        self.window, self.stride = window, window - overlap
        self.ratio = temporal_ratio
        # Latents still needed by a future window; the first is absolute latent_start.
        self.latents: torch.Tensor | None = None
        self.latent_start = 0
        self.available = 0
        self.next_window = 0
        # Merged raw pixels of the decoded regular windows from raw frame pending_start.
        self.pending: torch.Tensor | None = None
        self.pending_start = 0
        # Raw frames already emitted (or dropped as push padding).
        self.committed = 0
        # (raw_start, raw_visible_end) of every push not yet fully committed.
        self.spans: list[tuple[int, int]] = []
        self.finished = False

    def _decode(self, decode_fn, start: int, end: int) -> torch.Tensor:
        offset = self.latent_start
        return decode_fn(self.latents[:, :, start - offset : end - offset], start)

    def _commit(self, raw: torch.Tensor, end: int) -> torch.Tensor:
        """Emit raw frames [committed, end), dropping each push's padded tail."""
        start = self.committed
        parts = [
            raw[:, :, max(a, start) - start : min(b, end) - start]
            for a, b in self.spans
            if min(b, end) > max(a, start)
        ]
        self.committed = end
        self.spans = [span for span in self.spans if span[1] > end]
        return torch.cat(parts, dim=2) if parts else raw[:, :, :0]

    def push(
        self, decode_fn, latent: torch.Tensor, pixel_frames: int
    ) -> torch.Tensor | None:
        """Append one segment showing ``pixel_frames`` frames; return newly stable frames."""
        if self.finished:
            raise RuntimeError("cannot push PixelUMM latents after finish")
        length = latent.shape[2]
        if latent.ndim != 5 or length < 1:
            raise ValueError("PixelUMM streaming expects nonempty [B,C,T,H,W] latents")
        if not (length - 1) * self.ratio < pixel_frames <= length * self.ratio:
            raise ValueError(
                f"{pixel_frames} pixel frames do not match {length} latent frames"
            )
        raw_start = self.available * self.ratio
        self.spans.append((raw_start, raw_start + pixel_frames))
        self.latents = (
            latent if self.latents is None else torch.cat((self.latents, latent), 2)
        )
        self.available += length
        window, stride, ratio = self.window, self.stride, self.ratio
        starts = []
        while self.next_window + window <= self.available:
            starts.append(self.next_window)
            self.next_window += stride
        # Like upstream, the stable boundary only moves when a regular window
        # is decoded.
        if not starts:
            return None
        video = self._decode(decode_fn, starts[0], starts[-1] + window)
        if self.pending is None:
            self.pending, self.pending_start = video, starts[0] * ratio
        else:
            self.pending = merge_windows(
                [self.pending, video], [0, starts[0] * ratio - self.pending_start]
            )
        # A later push can add a window anchored at the new tail, at least one
        # latent right of the current tail window, so only frames before that
        # are stable. A tail window on the grid is already decoded.
        tail = self.available - window
        stable = tail + 1 if tail >= 0 and tail % stride == 0 else max(0, tail)
        output = None
        if stable * ratio > self.committed:
            local = stable * ratio - self.pending_start
            output = self._commit(
                self.pending[:, :, self.committed - self.pending_start : local],
                stable * ratio,
            )
            self.pending, self.pending_start = (
                self.pending[:, :, local:],
                stable * ratio,
            )
        # finish() re-decodes from the first window that reaches past `stable`.
        keep = max(0, min(self.next_window, stable - window + 1))
        if keep > self.latent_start:
            self.latents = self.latents[:, :, keep - self.latent_start :]
            self.latent_start = keep
        return output

    def finish(self, decode_fn) -> torch.Tensor | None:
        """Emit every frame not yet stable; None when nothing is withheld."""
        if self.finished:
            raise RuntimeError("PixelUMM streaming decode is already finished")
        self.finished = True
        end = self.available * self.ratio
        if self.latents is None or self.committed == end:
            return None
        if (
            self.pending is not None
            and self.pending_start + self.pending.shape[2] == end
        ):
            # The tail window sits on the grid, so it was decoded as a regular window.
            return self._commit(
                self.pending[:, :, self.committed - self.pending_start :], end
            )
        starts = window_starts(self.available, self.window, self.stride)
        committed_latent = self.committed // self.ratio
        # Earlier windows end inside the committed frames, so decoding from
        # this window reproduces the full-clip blend of every later frame.
        first = next(
            (s for s in starts if s + self.window > committed_latent), starts[-1]
        )
        video = self._decode(decode_fn, first, self.available)
        return self._commit(video[:, :, self.committed - first * self.ratio :], end)


class NomasPixelUMMDecoder(nn.Module):
    def __init__(self, config: NomasVAEConfig):
        super().__init__()
        self.config = config
        arch = config.arch_config
        self.decoder = DCAEVPixelUMMDecoder(
            GenExpertArch(
                **{f.name: getattr(arch, f.name) for f in fields(GenExpertArch)}
            ),
            local_attention_tile=arch.local_attention_tile,
        )
        # Absolute latent index of the first decoded frame. Streaming decodes
        # slices of a session timeline whose noise follows that index.
        self.noise_frame_offset = 0

    @contextmanager
    def timeline_offset(self, frame_offset: int):
        previous, self.noise_frame_offset = self.noise_frame_offset, frame_offset
        try:
            yield
        finally:
            self.noise_frame_offset = previous

    def _noise(self, latent, start, end):
        batch, _, _, height, width = latent.shape
        return torch.cat(
            [
                torch.randn(
                    (batch, 3, 4, height * 32, width * 32),
                    device=latent.device,
                    dtype=torch.float32,
                    generator=torch.Generator(device=latent.device).manual_seed(
                        self.config.arch_config.seed * 1_000_003 + index
                    ),
                )
                for index in range(start, end)
            ],
            dim=2,
        )

    @staticmethod
    def _decode_windows(decode_window, starts):
        """Decode independent windows, spread over the decode-parallel group.

        Every window depends only on its own latents and frame-indexed noise, so
        rank ``r`` decodes windows ``r, r + W, ...`` and an all-gather hands every
        rank all windows; values and merge order are those of a serial decode.
        """
        group = None
        if model_parallel_is_initialized() and get_decode_parallel_world_size() > 1:
            group = get_decode_parallel_group_coordinator()
        if group is None or len(starts) < group.world_size:
            return [decode_window(start) for start in starts]
        world, rank = group.world_size, group.rank_in_group
        mine = [decode_window(start) for start in starts[rank::world]]
        per_rank = -(-len(starts) // world)
        mine += [torch.zeros_like(mine[0])] * (per_rank - len(mine))
        gathered = group.all_gather(torch.stack(mine), dim=0)
        return [
            gathered[(i % world) * per_rank + i // world] for i in range(len(starts))
        ]

    @torch.inference_mode()
    def decode(self, latent, return_dict=True, generator=None):
        if latent.ndim != 5 or latent.shape[1] != 32:
            raise ValueError("PixelUMM expects [B,32,T,H,W] latents")
        arch = self.config.arch_config
        length = latent.shape[2]
        starts = window_starts(
            length,
            arch.temporal_window_latents,
            arch.temporal_window_latents - arch.temporal_overlap_latents,
        )
        steps = arch.denoising_step_list[: arch.num_steps]
        offset = self.noise_frame_offset
        # A single step generates only one window of noise at a time.
        state = self._noise(latent, offset, offset + length) if len(steps) > 1 else None
        for t_cur, t_next in zip(steps, (*steps[1:], 0.0)):

            def decode_window(start):
                end = min(start + arch.temporal_window_latents, length)
                noise = (
                    self._noise(latent, offset + start, offset + end)
                    if state is None
                    else state[:, :, start * 4 : end * 4]
                )
                return self.decoder(latent[:, :, start:end], noise)

            parts = self._decode_windows(decode_window, starts)
            video = merge_windows(parts, [s * 4 for s in starts])
            if t_next:
                state = video.float() + (t_next / t_cur) * (state - video.float())
        return NomasDecodeOutput(video) if return_dict else (video,)


EntryClass = NomasPixelUMMDecoder
