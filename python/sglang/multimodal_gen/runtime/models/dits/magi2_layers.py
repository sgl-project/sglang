# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import math

import msgspec
import torch
from torch import nn
from torch.utils.weak import WeakTensorKeyDictionary

from sglang.multimodal_gen.runtime.distributed.sp_shard_utils import (
    SpShard,
    build_shard_plan,
)
from sglang.multimodal_gen.runtime.models.dits.magi2_moe import (
    SWIGLU7_ALPHA,
    SWIGLU7_LIMIT,
)

# Coordinate columns are (t, h, w, T, H, W, ref_T, ref_H, ref_W).
MAGI2_COORD_COLUMNS = 9
MAGI2_COORD_AXES = 3

MAGI2_MODALITY_VIDEO = 0
MAGI2_MODALITY_AUDIO = 1
MAGI2_MODALITY_TEXT = 2


class Magi2SegmentLayout(msgspec.Struct, frozen=True):
    total_tokens: int
    video_index: torch.Tensor
    audio_index: torch.Tensor
    text_index: torch.Tensor
    modality_ids: torch.Tensor
    cu_seqlens: torch.Tensor
    max_seqlen: int
    video_latent_thw: tuple[int, int, int]

    ref_special_index: torch.Tensor = msgspec.field(
        default_factory=lambda: torch.empty(0, dtype=torch.long)
    )
    ref_patch_index: torch.Tensor = msgspec.field(
        default_factory=lambda: torch.empty(0, dtype=torch.long)
    )


def freq_bands(num_bands: int, temperature: float = 10000.0) -> torch.Tensor:
    exp = torch.arange(num_bands, dtype=torch.float32) / num_bands
    return 1.0 / (temperature**exp)


class Magi2FourierRope(nn.Module):
    """16 bands give a 96-channel rotary span; the trailing 32 of a 128-wide head stay unrotated, intentionally."""

    def __init__(self, num_bands: int) -> None:
        super().__init__()
        self.num_bands = num_bands
        self.bands = nn.Parameter(freq_bands(num_bands))

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        positions = coords[:, 0:3]
        sizes = coords[:, 3:6]
        refs = coords[:, 6:9]

        scales = (refs - 1) / (sizes - 1)
        scales = torch.where(
            (refs == 1) & (sizes == 1), torch.ones_like(scales), scales
        )

        centers = (sizes - 1) / 2
        centers = torch.cat([torch.zeros_like(centers[:, :1]), centers[:, 1:]], dim=1)

        proj = (positions - centers).unsqueeze(-1) * scales.unsqueeze(-1) * self.bands
        return torch.cat((proj.sin(), proj.cos()), dim=1).flatten(1)


def apply_partial_rope(
    x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> torch.Tensor:
    """fp32 cos/sin in, ``x`` dtype out. Channels past the rotary span pass through."""
    rotary_dim = cos.shape[-1] * 2
    if rotary_dim > x.shape[-1]:
        raise ValueError(f"rotary_dim {rotary_dim} exceeds head_dim {x.shape[-1]}")

    rotated, passthrough = x[..., :rotary_dim], x[..., rotary_dim:]
    # Tile, not interleave: the rotation pairs channel i with i + rotary_dim // 2.
    cos = cos.repeat(1, 2).unsqueeze(1)
    sin = sin.repeat(1, 2).unsqueeze(1)
    first, second = rotated.chunk(2, dim=-1)
    half_rotated = torch.cat((-second, first), dim=-1)
    out = (rotated * cos + half_rotated * sin).to(x.dtype)
    return torch.cat((out, passthrough), dim=-1) if passthrough.numel() else out


def swiglu7_interleaved(x: torch.Tensor) -> torch.Tensor:
    """Interleaved spelling; the MoE kernel uses the chunked one. Not interchangeable."""
    # fp32 in, input dtype out: bf16 spacing is coarse at the +/-7.0 clamp.
    dtype = x.dtype
    x = x.float()
    gate, up = x[..., ::2], x[..., 1::2]
    gate = gate.clamp(max=SWIGLU7_LIMIT)
    up = up.clamp(min=-SWIGLU7_LIMIT, max=SWIGLU7_LIMIT)
    return (gate * torch.sigmoid(SWIGLU7_ALPHA * gate) * (up + 1)).to(dtype)


def shard_packed_rows(
    *tensors: torch.Tensor,
) -> tuple[list[torch.Tensor], object]:
    from sglang.multimodal_gen.runtime.distributed.sp_shard_utils import (
        build_shard_plan,
        shard_like,
    )

    plan = build_shard_plan(tensors[0].shape[0])
    sharded = [shard_like(t, plan, dim=0, pad_mode="repeat_last") for t in tensors]
    return sharded, plan


def local_row_range(total_tokens: int) -> tuple[SpShard, tuple[int, int] | None]:
    """This rank's packed rows ``[lo, hi)``, so the adapters touch only those; None
    without sequence parallelism."""
    plan = build_shard_plan(total_tokens)
    if plan.sp_size <= 1:
        return plan, None
    lo = plan.sp_rank * plan.local_len
    return plan, (lo, lo + plan.local_real_len)


def pad_local_rows(plan: SpShard, *tensors: torch.Tensor) -> list[torch.Tensor]:
    """The last rank's tail pad repeats its last real row, as ``shard_like`` does."""
    if not plan.local_pad:
        return list(tensors)
    return [
        torch.cat([t, t[-1:].expand(plan.local_pad, *t.shape[1:])]) for t in tensors
    ]


_LOCAL_ROWS = WeakTensorKeyDictionary()


def local_rows(
    index: torch.Tensor, lo: int, hi: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """The inputs whose packed position falls in ``[lo, hi)``, and that position minus
    ``lo``. Memoized per index tensor, since the selection syncs with the host and every
    step of a request reuses the layout."""
    cached = _LOCAL_ROWS.get(index)
    if cached is not None and cached[0] == (lo, hi):
        return cached[1]
    keep = ((index >= lo) & (index < hi)).nonzero().flatten()
    rows = keep, index.index_select(0, keep) - lo
    _LOCAL_ROWS[index] = ((lo, hi), rows)
    return rows


def pad_rows_to_multiple(
    *tensors: torch.Tensor, multiple: int
) -> tuple[list[torch.Tensor], int]:
    """Repeat-last rather than zero-fill, because a zeroed coordinate row is a valid grid position."""
    num_pad = -tensors[0].shape[0] % multiple
    if num_pad == 0:
        return list(tensors), 0
    padded = [
        torch.cat(
            [t, t.narrow(0, t.shape[0] - 1, 1).expand(num_pad, *t.shape[1:])], dim=0
        )
        for t in tensors
    ]
    return padded, num_pad


def sharded_cu_seqlens(*, plan, device: torch.device) -> tuple[torch.Tensor, int]:
    """Pad rows become their own varlen segment so real tokens cannot attend to them."""
    total = plan.sp_size * plan.local_len
    if plan.num_pad == 0:
        return (
            torch.tensor([0, total], dtype=torch.int32, device=device),
            total,
        )
    valid = plan.orig_len
    return (
        torch.tensor([0, valid, total], dtype=torch.int32, device=device),
        max(valid, plan.num_pad),
    )


def gather_packed_rows(local: torch.Tensor, *, plan) -> torch.Tensor:
    from sglang.multimodal_gen.runtime.distributed.sp_shard_utils import gather_seq

    if plan.sp_size <= 1:
        return local
    return gather_seq(local.unsqueeze(0), plan.orig_len, dim=1).squeeze(0)


_MODALITY_RUNS = WeakTensorKeyDictionary()


def modality_runs(modality_ids: torch.Tensor) -> list[tuple[int, int, int]]:
    """Memoized per tensor, since the scan syncs with the host and every block of a
    forward reads the same layout. Weak keys, so a finished request is not retained."""
    runs = _MODALITY_RUNS.get(modality_ids)
    if runs is not None:
        return runs
    # Only the boundary positions cross to the host, not the tags themselves.
    changes = (modality_ids[1:] != modality_ids[:-1]).nonzero().flatten() + 1
    edges = [0, *changes.tolist(), int(modality_ids.numel())]
    runs = [
        (start, end - start, int(modality_ids[start]))
        for start, end in zip(edges, edges[1:])
        if end > start
    ]
    _MODALITY_RUNS[modality_ids] = runs
    return runs


class Magi2ModalityRMSNorm(nn.Module):
    """Modality selects which weight applies; it is not a batch axis."""

    def __init__(self, width: int, *, num_modality: int = 1, eps: float = 1e-6) -> None:
        super().__init__()
        self.width = width
        self.num_modality = num_modality
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(num_modality * width))

    def forward(
        self,
        x: torch.Tensor,
        modality_ids: torch.Tensor | None = None,
        *,
        out_dtype: torch.dtype | None = None,
    ) -> torch.Tensor:
        dtype = out_dtype or x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        if self.num_modality == 1:
            return (x * self.weight.float()).to(dtype)

        weight = self.weight.float().view(self.num_modality, self.width)

        if modality_ids.numel() == 1:
            return (x * weight[int(modality_ids[0])]).to(dtype)

        # Gathered, not a run loop: the loop's boundary scan crosses to the host,
        # which both syncs and breaks the compiled graph in every MoE layer.
        gathered = weight.index_select(0, modality_ids)
        # q_norm/k_norm pass [T, heads, dim], so the gather needs the head axis.
        return (x * gathered.reshape(-1, *(1,) * (x.dim() - 2), self.width)).to(dtype)


class Magi2ModalityLinear(nn.Module):
    """Modality selects which block of weight rows applies; it is not a batch axis."""

    def __init__(
        self, in_features: int, out_features: int, *, num_modality: int = 1
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_modality = num_modality
        self.weight = nn.Parameter(
            torch.empty(num_modality * out_features, in_features)
        )

    def forward(
        self, x: torch.Tensor, modality_ids: torch.Tensor | None = None
    ) -> torch.Tensor:
        # Precision boundary: callers pass fp32, cross-dtype matmul raises.
        x = x.to(self.weight.dtype)

        if self.num_modality == 1:
            return torch.nn.functional.linear(x, self.weight)

        weight = self.weight.view(
            self.num_modality, self.out_features, self.in_features
        )
        if modality_ids.numel() == 1:
            return torch.nn.functional.linear(x, weight[int(modality_ids[0])])

        out = x.new_empty(x.shape[:-1] + (self.out_features,))
        for start, count, modality in modality_runs(modality_ids):
            # Slice assignment, not linear(out=...), which rejects grad inputs.
            out[start : start + count] = torch.nn.functional.linear(
                x.narrow(0, start, count), weight[modality]
            )
        return out


def sinusoidal_embedding_1d(dim: int, position: torch.Tensor) -> torch.Tensor:
    half = dim // 2
    position = position.float()
    div = torch.exp(
        -math.log(10000.0)
        * torch.arange(half, dtype=torch.float32, device=position.device)
        / half
    )
    angles = position.unsqueeze(-1) * div.unsqueeze(0)
    return torch.cat((angles.cos(), angles.sin()), dim=-1)


class Magi2PreAdapter(nn.Module):
    """Kept fp32: this is where the raw latent enters the network."""

    def __init__(self, config) -> None:
        super().__init__()
        self.stream_dim = config.residual_stream_dim
        self.time_channel_dim = config.time_channel_dim

        self.video_embedder = nn.Linear(
            config.video_in_channels, self.stream_dim, dtype=torch.float32
        )
        self.audio_embedder = nn.Linear(
            config.audio_in_channels, self.stream_dim, dtype=torch.float32
        )
        self.text_embedder = nn.Linear(
            config.text_in_channels, self.stream_dim, dtype=torch.float32
        )
        self.rope = Magi2FourierRope(config.rope_bands)

    def forward(
        self,
        *,
        video: torch.Tensor,
        audio: torch.Tensor | None,
        text: torch.Tensor,
        layout: Magi2SegmentLayout,
        coords: torch.Tensor,
        timestep: torch.Tensor,
        ref_patches: torch.Tensor | None = None,
        ref_special: torch.Tensor | None = None,
        row_range: tuple[int, int] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Embeds the packed rows ``row_range`` (all of them by default)."""
        lo, hi = row_range or (0, layout.total_tokens)
        rows = torch.zeros(
            hi - lo, self.stream_dim, device=video.device, dtype=torch.float32
        )

        def embed(index, inputs, embedder):
            if row_range is not None:
                keep, index = local_rows(index, lo, hi)
                inputs = inputs.index_select(0, keep)
            rows.index_copy_(0, index, embedder(inputs.float()))

        embed(layout.video_index, video, self.video_embedder)
        embed(layout.text_index, text, self.text_embedder)
        if audio is not None:
            embed(layout.audio_index, audio, self.audio_embedder)
        if ref_special is not None and layout.ref_special_index.numel():
            embed(layout.ref_special_index, ref_special, self.text_embedder)
        if ref_patches is not None and layout.ref_patch_index.numel():
            embed(layout.ref_patch_index, ref_patches, self.video_embedder)

        # The timestep embedding is not a token: it overwrites the leading channels of
        # every row, which is what allows a per-token schedule. time_channel_dim is 0
        # on the refiner, where it would discard 64 real embedded channels.
        if self.time_channel_dim:
            time_embed = sinusoidal_embedding_1d(self.time_channel_dim, timestep[lo:hi])
            rows[:, : self.time_channel_dim] = time_embed.to(rows.dtype)

        return rows, self.rope(coords[lo:hi])


class Magi2PostAdapter(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        stream_dim = config.residual_stream_dim
        self.final_norm_video = Magi2ModalityRMSNorm(stream_dim)
        self.final_norm_audio = Magi2ModalityRMSNorm(stream_dim)
        self.final_linear_video = nn.Linear(
            stream_dim, config.video_in_channels, bias=False, dtype=torch.float32
        )
        self.final_linear_audio = nn.Linear(
            stream_dim, config.audio_in_channels, bias=False, dtype=torch.float32
        )

    def project(self, rows: torch.Tensor) -> torch.Tensor:
        """Both output heads on every row, ``[rows, video + audio channels]``: a rank
        projects its own rows, so only these narrow outputs cross ranks."""
        rows = rows.float()
        video = self.final_linear_video(self.final_norm_video(rows))
        audio = self.final_linear_audio(self.final_norm_audio(rows))
        return torch.cat((video, audio), dim=1)

    def forward(
        self, projected: torch.Tensor, *, layout: Magi2SegmentLayout
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Picks each modality's rows out of ``project``'s output."""
        num_video = self.final_linear_video.out_features
        video = projected[:, :num_video].index_select(0, layout.video_index)
        audio = None
        if layout.audio_index.numel():
            audio = projected[:, num_video:].index_select(0, layout.audio_index)
        return video, audio
