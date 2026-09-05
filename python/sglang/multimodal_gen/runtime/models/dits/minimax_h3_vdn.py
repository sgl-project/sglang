# SPDX-License-Identifier: Apache-2.0
"""VDN-H3 (Video DeltaNet MiniMax-H3) linear attention branch.

Port of OpenVDN's ``BidirectionalLinearBranch`` (github.com/OpenVDN/
vdn-minimax-h3, ``src/models/linear_attention``) to SGLang's packed MiniMax-H3
attention. The branch summarises everything the chunked window softmax cannot
see, for every video token, in five steps:

    0. text state      the prompt rows written once into a zero state; both
                       directional scans start from half of it
    1. features        SiLU (+ separable 5x5 spatial / 5-tap temporal depthwise
                       conv on k, v), L2-normalised q/k, NoPE
    2. frame stats     A = K^T diag(beta) K (fp32), B = V^T diag(beta) K per frame
    3. two scans       Video Delta rule S_t = (S_{t-1} diag(alpha_t) + B_t)(I + A_t)^-1
                       forward and reverse over frames
    4. boundary gather prefix[lo-1] + suffix[hi+1] decayed to frame t by
                       prod alpha over the window (the exact complement of the
                       softmax window; ends read the text state)
    5. readout         q . S -> RMSNorm(head_dim) -> low-rank sigmoid gate

Inference-only and eager. Parameter names follow VDN one level below
``blocks.N.attn.linear_attention``; heads shard under TP, the per-token
``alpha.down`` and ``output_gate.down`` are replicated.
"""

from __future__ import annotations

import functools
import math

import msgspec
import torch
import torch.nn.functional as F
from torch import nn

from sglang.kernels.ops.diffusion import (
    can_use_vdn_delta_factors,
    can_use_vdn_frame_stats_prep,
    can_use_vdn_gather_linear_state,
    can_use_vdn_linear_epilogue,
    can_use_vdn_silu_l2norm,
    can_use_vdn_temporal_conv_act,
    vdn_delta_factors,
    vdn_frame_stats_prep,
    vdn_gather_linear_state,
    vdn_linear_epilogue,
    vdn_silu_l2norm,
    vdn_temporal_conv_act,
)
from sglang.multimodal_gen.configs.models.dits.minimax_h3 import (
    MiniMaxH3DiTArchConfig,
    VDNHybridAttentionArchConfig,
)
from sglang.multimodal_gen.runtime.distributed import (
    get_tp_rank,
    get_tp_world_size,
)
from sglang.multimodal_gen.runtime.layers.linear import (
    ColumnParallelLinear,
    ReplicatedLinear,
)

_BF16 = torch.bfloat16
_FP32 = torch.float32

# Each directional scan starts from TEXT_STATE_SCALE * S_text. Baked into the
# trained checkpoints, not a knob (see VDN BidirectionalLinearBranch).
TEXT_STATE_SCALE = 0.5
SHORT_CONV_KERNEL = 5


# --------------------------------------------------------------------------
# Packed-sequence geometry
# --------------------------------------------------------------------------


class VDNH3Layout(msgspec.Struct, frozen=True):
    """Where the modalities sit in SGLang's packed H3 sequence
    ``[text L | cond C | audio A | video V | pad P]``.

    ``used`` is ``cu_seqlens[1]``: rows at and past it are padding and sit
    outside every attention mask. Text and audio rows are "global" for the
    softmax branch (dense both ways); only the text rows seed the linear
    branch's state.
    """

    seq_len: int
    used: int
    text_len: int
    video_start: int
    num_frames: int
    tokens_per_frame: int
    frame_height: int
    frame_width: int

    def __post_init__(self) -> None:
        if self.frame_height * self.frame_width != self.tokens_per_frame:
            raise ValueError(
                f"frame grid {self.frame_height}x{self.frame_width} != "
                f"{self.tokens_per_frame} tokens per frame"
            )
        if self.video_end > self.used or self.used > self.seq_len:
            raise ValueError(
                f"video rows [{self.video_start}, {self.video_end}) exceed used "
                f"rows {self.used} (seq_len {self.seq_len})"
            )
        if self.text_len > self.video_start:
            raise ValueError("text rows must precede the video rows")

    @property
    def video_end(self) -> int:
        return self.video_start + self.num_frames * self.tokens_per_frame

    @property
    def text_range(self) -> tuple[int, int]:
        return 0, self.text_len

    @property
    def frame_size(self) -> tuple[int, int]:
        return self.frame_height, self.frame_width

    @property
    def global_ranges(self) -> list[tuple[int, int]]:
        """Non-video, non-padding row ranges (text, condition, audio)."""
        return [
            (start, stop)
            for start, stop in ((0, self.video_start), (self.video_end, self.used))
            if start < stop
        ]

    def frame_rows(self, frame: int) -> tuple[int, int]:
        start = self.video_start + frame * self.tokens_per_frame
        return start, start + self.tokens_per_frame


def vdn_h3_layout_from_packed(
    packed: dict, *, latent_t: int, latent_h: int, latent_w: int
) -> VDNH3Layout:
    """Layout from a ``minimax_h3_packed_sequence`` result (t2va / fl2va)."""
    img_pos = packed["img_pos"].view(-1)
    update_mask = packed["update_mask"].view(-1).to(torch.bool)
    video_pos = img_pos[update_mask]
    frame_h, frame_w = latent_h // 2, latent_w // 2
    tokens_per_frame = frame_h * frame_w
    video_start = int(video_pos[0])
    if int(video_pos[-1]) - video_start + 1 != int(video_pos.numel()):
        raise ValueError("video rows are not contiguous in the packed sequence")
    if int(video_pos.numel()) != latent_t * tokens_per_frame:
        raise ValueError(
            f"{int(video_pos.numel())} video rows != {latent_t} frames x "
            f"{tokens_per_frame} tokens per frame"
        )
    cu = packed["cu_seqlens"].view(-1).tolist()
    return VDNH3Layout(
        seq_len=int(packed["seq_len"]),
        used=int(cu[1]),
        text_len=int(packed["text_pos"].numel()),
        video_start=video_start,
        num_frames=latent_t,
        tokens_per_frame=tokens_per_frame,
        frame_height=frame_h,
        frame_width=frame_w,
    )


# --------------------------------------------------------------------------
# Head-sharded plain parameters
# --------------------------------------------------------------------------


def _head_sharded_loader(shard_dim: int):
    # head-major output rows: TP shards by head, the checkpoint stores all heads

    def _loader(param: torch.Tensor, loaded_weight: torch.Tensor) -> None:
        tp_size = get_tp_world_size()
        if tp_size > 1:
            shard = param.shape[shard_dim]
            loaded_weight = loaded_weight.narrow(
                shard_dim, get_tp_rank() * shard, shard
            )
        assert param.shape == loaded_weight.shape, (
            f"VDN branch parameter shape {tuple(param.shape)} != checkpoint "
            f"{tuple(loaded_weight.shape)}"
        )
        param.data.copy_(loaded_weight)

    return _loader


def _make_param(
    shape: tuple[int, ...], *, dtype: torch.dtype, shard_dim: int | None
) -> nn.Parameter:
    param = nn.Parameter(torch.empty(shape, dtype=dtype), requires_grad=False)
    if shard_dim is not None:
        param.weight_loader = _head_sharded_loader(shard_dim)
    return param


def _head_sharded_linear(
    in_features: int, out_features: int, *, bias: bool, prefix: str
) -> ColumnParallelLinear:
    return ColumnParallelLinear(
        in_features,
        out_features,
        bias=bias,
        gather_output=False,
        params_dtype=_BF16,
        quant_config=None,
        prefix=prefix,
    )


def _replicated_linear(
    in_features: int, out_features: int, *, prefix: str
) -> ReplicatedLinear:
    return ReplicatedLinear(
        in_features,
        out_features,
        bias=False,
        params_dtype=_BF16,
        quant_config=None,
        prefix=prefix,
    )


class VDNFrameAlpha(nn.Module):
    """alpha_t = exp(-exp(A_log) * softplus(up(down(frame_mean)) + dt_bias)),
    per frame / head / key channel, in fp32 (KDA's double-exponential gate)."""

    def __init__(
        self,
        hidden_size: int,
        heads: int,
        local_heads: int,
        head_dim: int,
        *,
        prefix: str,
    ) -> None:
        super().__init__()
        self.local_heads, self.head_dim = local_heads, head_dim
        self.down = _replicated_linear(hidden_size, head_dim, prefix=f"{prefix}.down")
        self.up = _head_sharded_linear(
            head_dim, heads * head_dim, bias=False, prefix=f"{prefix}.up"
        )
        # fp32: the scan multiplies alpha over ~100 frames, so bf16 error compounds
        self.A_log = _make_param((local_heads,), dtype=_FP32, shard_dim=0)
        self.dt_bias = _make_param((local_heads * head_dim,), dtype=_FP32, shard_dim=0)

    def forward(
        self, frame_mean: torch.Tensor, heads: slice | None = None
    ) -> torch.Tensor:
        """frame_mean [F, hidden] fp32 -> alpha [F, H, d] fp32 for the head
        range ``heads`` (Ulysses: this rank's shard of the TP-local heads)."""
        if heads is None:
            up_w, dt_bias, a_log, n_heads = (
                self.up.weight,
                self.dt_bias,
                self.A_log,
                self.local_heads,
            )
        else:
            rows = slice(heads.start * self.head_dim, heads.stop * self.head_dim)
            up_w, dt_bias, a_log = (
                self.up.weight[rows],
                self.dt_bias[rows],
                self.A_log[heads],
            )
            n_heads = heads.stop - heads.start
        delta = F.linear(frame_mean.float(), self.down.weight.float())
        delta = F.linear(delta, up_w.float()) + dt_bias
        scale = torch.exp(a_log)[:, None]
        delta = delta.view(-1, n_heads, self.head_dim)
        return torch.exp(-scale * F.softplus(delta))


class VDNOutputGate(nn.Module):
    """Low-rank sigmoid gate: sigmoid(up(down(x))) -> [T, H_local, d]."""

    def __init__(
        self,
        hidden_size: int,
        heads: int,
        local_heads: int,
        head_dim: int,
        *,
        prefix: str,
    ) -> None:
        super().__init__()
        self.local_heads, self.head_dim = local_heads, head_dim
        self.down = _replicated_linear(hidden_size, head_dim, prefix=f"{prefix}.down")
        self.up = _head_sharded_linear(
            head_dim, heads * head_dim, bias=True, prefix=f"{prefix}.up"
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden, _ = self.down(x)
        return self.up_gate(hidden)

    def up_gate(self, hidden: torch.Tensor, heads: slice | None = None) -> torch.Tensor:
        """sigmoid(up(hidden)) -> [T, h, d]; ``heads`` selects a head range of
        the up projection (Ulysses computes the gate on its head shard from the
        all-gathered ``down`` hidden)."""
        if heads is None:
            gate, _ = self.up(hidden)
            return torch.sigmoid(gate).view(-1, self.local_heads, self.head_dim)
        rows = slice(heads.start * self.head_dim, heads.stop * self.head_dim)
        bias = None if self.up.bias is None else self.up.bias[rows]
        gate = F.linear(hidden, self.up.weight[rows], bias)
        return torch.sigmoid(gate).view(-1, heads.stop - heads.start, self.head_dim)


class VDNSoftmaxGate(nn.Module):
    """Per-(token, head) sigmoid gate on the softmax branch output."""

    def __init__(self, hidden_size: int, heads: int, *, prefix: str) -> None:
        super().__init__()
        self.up = _head_sharded_linear(
            hidden_size, heads, bias=True, prefix=f"{prefix}.up"
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate, _ = self.up(x)
        return torch.sigmoid(gate)


class VDNShortConv(nn.Module):
    """Separable depthwise short conv (5x5 spatial per frame, then 5 taps
    across frames) on the projections named in ``targets``; channels are
    head-major, so TP takes a channel slice."""

    def __init__(self, channels: int, targets: tuple[str, ...]) -> None:
        super().__init__()
        self.targets = tuple(targets)
        k = SHORT_CONV_KERNEL
        for name in self.targets:
            setattr(
                self,
                f"{name}_sp",
                nn.ParameterDict(
                    {
                        "weight": _make_param(
                            (channels, 1, k, k), dtype=_BF16, shard_dim=0
                        )
                    }
                ),
            )
            setattr(
                self,
                f"{name}_tm",
                nn.ParameterDict(
                    {"weight": _make_param((channels, 1, k), dtype=_BF16, shard_dim=0)}
                ),
            )

    def spatial(
        self,
        proj: str,
        tokens: torch.Tensor,
        num_frames: int,
        frame_size: tuple[int, int],
        heads: slice | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """The 5x5 depthwise half on [F*S, H, d] tokens -> ([F, S, C], w_tm [C, 5]);
        ``heads`` selects the weight channels of a head range."""
        n_heads, head_dim = tokens.shape[-2], tokens.shape[-1]
        grid_h, grid_w = frame_size
        channels = n_heads * head_dim
        w_sp = getattr(self, f"{proj}_sp")["weight"]
        w_tm = getattr(self, f"{proj}_tm")["weight"]
        if heads is not None:
            rows = slice(heads.start * head_dim, heads.stop * head_dim)
            w_sp, w_tm = w_sp[rows], w_tm[rows]
        # [F*S, H, d] read as channels_last [F, C, gh, gw]: cuDNN NHWC depthwise
        volume = tokens.reshape(num_frames, grid_h, grid_w, channels).permute(
            0, 3, 1, 2
        )
        volume = F.conv2d(volume, w_sp, padding=SHORT_CONV_KERNEL // 2, groups=channels)
        x = volume.permute(0, 2, 3, 1).reshape(num_frames, grid_h * grid_w, channels)
        return x, w_tm.squeeze(1).to(x.dtype)


def _branch_norm(dim: int, eps: float = 1e-6) -> nn.RMSNorm:
    # weight holder only; the arithmetic runs in the epilogue (fp32 second moment)
    return nn.RMSNorm(dim, eps=eps, dtype=_BF16)


# --------------------------------------------------------------------------
# The algorithm (eager, inference-only)
# --------------------------------------------------------------------------


def _temporal_shift(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    # depthwise 5-tap conv over frames, zero padded, symmetric; x [F, S, C], w [C, 5]
    k = SHORT_CONV_KERNEL
    pad = k // 2
    xp = F.pad(x, (0, 0, 0, 0, pad, pad))
    out = None
    for dt in range(k):
        part = xp[dt : dt + x.shape[0]] * w[:, dt].view(1, 1, -1)
        out = part if out is None else out + part
    return out


def _activate(tokens: torch.Tensor, l2norm: bool) -> torch.Tensor:
    x = F.silu(tokens)
    return F.normalize(x, dim=-1, eps=1e-6).to(x.dtype) if l2norm else x


def linear_features(
    tokens: torch.Tensor,
    *,
    proj: str,
    conv: VDNShortConv | None,
    num_frames: int | None,
    frame_size: tuple[int, int] | None,
    heads: slice | None = None,
    frame_major: bool = False,
    fused: bool = True,
) -> torch.Tensor:
    """[N, H, d] raw projection -> [N, H, d] branch features:
    [short conv ->] SiLU [-> L2 norm for q, k]. ``frame_major`` returns
    [F, H, S, d] instead (the readout's bmm layout), written by the fused
    kernels directly; the eager path permutes."""
    l2norm = proj != "v"
    n_heads, head_dim = tokens.shape[-2], tokens.shape[-1]
    if frame_major and (num_frames is None or frame_size is None):
        raise ValueError("frame_major needs the (frames, height, width) grid")
    if conv is not None and proj in conv.targets:
        if frame_size is None or num_frames is None:
            raise ValueError("the short conv needs the (frames, height, width) grid")
        x, w_tm = conv.spatial(proj, tokens, num_frames, frame_size, heads=heads)
        if fused and can_use_vdn_temporal_conv_act(x, n_heads, head_dim):
            # one kernel: 5 taps + SiLU + L2 norm, the conv output never hits HBM
            return vdn_temporal_conv_act(
                x, w_tm, n_heads, head_dim, l2norm, frame_major=frame_major
            )
        out = _activate(_temporal_shift(x, w_tm).reshape(-1, n_heads, head_dim), l2norm)
    elif fused and can_use_vdn_silu_l2norm(tokens):
        per_frame = frame_size[0] * frame_size[1] if frame_major else None
        return vdn_silu_l2norm(tokens, l2norm, per_frame=per_frame)
    else:
        out = _activate(tokens, l2norm)
    if frame_major:
        per_frame = frame_size[0] * frame_size[1]
        return out.view(num_frames, per_frame, n_heads, head_dim).permute(0, 2, 1, 3)
    return out


def frame_statistics(
    kf: torch.Tensor,
    vf: torch.Tensor,
    beta: torch.Tensor,
    *,
    a_fp32: bool,
    prepared: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
    | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """kf, vf [F, H, S, d], beta [F, H, S] -> A [F, H, dk, dk] fp32 symmetric,
    B [F, H, dv, dk] fp32; ``prepared`` carries the operands from
    ``vdn_frame_stats_prep``. A is inverted downstream, so it needs fp32;
    B enters the state linearly and takes bf16 tensor cores."""
    if prepared is not None:
        kf, kf32, scaled32, vf_b = prepared
    else:
        kf = kf.contiguous()
        vf_b = (vf * beta.unsqueeze(-1).to(vf.dtype)).contiguous()
        kf32 = scaled32 = None
    if a_fp32:
        if kf32 is None:
            kf32 = kf.float()
            scaled32 = (kf32 * beta.unsqueeze(-1).float()).contiguous()
        # TF32 keeps I + A well conditioned where bf16 does not; scoped to this matmul
        prev = torch.backends.cuda.matmul.allow_tf32
        torch.backends.cuda.matmul.allow_tf32 = True
        try:
            A = torch.matmul(scaled32.transpose(-1, -2), kf32)
        finally:
            torch.backends.cuda.matmul.allow_tf32 = prev
    else:
        A = torch.matmul(
            (kf * beta.unsqueeze(-1).to(kf.dtype)).contiguous().transpose(-1, -2), kf
        ).float()
    A = 0.5 * (A + A.transpose(-1, -2))
    B = torch.matmul(vf_b.transpose(-1, -2), kf).float()
    return A, B


def delta_factor_apply(
    rule: str,
    alpha: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    *,
    tokens_per_frame: int,
    fused: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """One frame's statistics -> (transition [F,H,dk,dk], injection [F,H,dv,dk]) fp32.

    vdn_solve  (the released checkpoints): S' = (S diag(alpha) + B)(I + A)^-1,
               exact Cholesky inverse.
    sana_scaled: S' = (S diag(alpha))(I - c^2 A) + c B, c = 1/sqrt(S).
    vdn_scaled: S' = (S diag(alpha) + c B)(I + c^2 A)^-1.

    ``fused`` takes the inverse and both products through one CUDA kernel
    (``vdn_delta_factors``, fp32 / head_dim 128) with the same accuracy as the
    Cholesky chain; anything the kernel does not cover falls back to eager.
    """
    A32, B32 = A.float(), B.float()
    eye = torch.eye(A32.shape[-1], device=A32.device, dtype=_FP32).expand_as(A32)
    if rule == "sana_scaled":
        inv_tokens = 1.0 / tokens_per_frame
        transition = alpha.unsqueeze(-1) * (eye - inv_tokens * A32)
        injection = math.sqrt(inv_tokens) * B32
        return transition, injection
    if rule == "vdn_scaled":
        inv_tokens = 1.0 / tokens_per_frame
        A32 = A32 * inv_tokens
        B32 = B32 * math.sqrt(inv_tokens)
    elif rule != "vdn_solve":
        raise ValueError(f"unknown delta rule {rule!r}")
    if fused:
        A32, B32, alpha32 = (
            A32.contiguous(),
            B32.contiguous(),
            alpha.float().contiguous(),
        )
        if can_use_vdn_delta_factors(A32, B32, alpha32):
            return vdn_delta_factors(A32, B32, alpha32)
    chol = torch.linalg.cholesky(A32 + eye)
    # (I+A)^-1 = L^-T L^-1: a batched trsm at 128x128 is far slower than the GEMM
    linv = torch.linalg.solve_triangular(chol, eye, upper=False, left=True)
    inv = linv.transpose(-1, -2) @ linv
    transition = alpha.unsqueeze(-1) * inv
    injection = B32 @ inv
    return transition, injection


def run_scans(
    transitions: torch.Tensor,
    injections: torch.Tensor,
    text_state: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """prefix[t] = frames 0..t, suffix[t] = frames t..F-1 (both fp32
    [F, H, dv, dk]); both start from ``text_state`` (or zero)."""
    num_frames = transitions.shape[0]
    start = (
        torch.zeros_like(injections[0])
        if text_state is None
        else text_state.to(injections.dtype)
    )
    prefix = torch.empty_like(injections)
    suffix = torch.empty_like(injections)
    state = start
    for frame in range(num_frames):
        torch.baddbmm(injections[frame], state, transitions[frame], out=prefix[frame])
        state = prefix[frame]
    state = start
    for frame in range(num_frames - 1, -1, -1):
        torch.baddbmm(injections[frame], state, transitions[frame], out=suffix[frame])
        state = suffix[frame]
    return prefix, suffix


def _compose_chunk(
    transitions: torch.Tensor, injections: torch.Tensor, reverse: bool
) -> tuple[torch.Tensor, torch.Tensor]:
    # fold each chunk's frames into one affine map S -> S @ M + C, batched over chunks
    order = list(range(transitions.shape[0]))
    if reverse:
        order.reverse()
    chunks, heads = transitions.shape[1], transitions.shape[2]
    dk, dv = transitions.shape[-1], injections.shape[-2]
    folded_t = transitions[order[0]]
    folded_b = injections[order[0]]
    for j in order[1:]:
        step_t = transitions[j].view(chunks * heads, dk, dk)
        folded_b = torch.baddbmm(
            injections[j].view(chunks * heads, dv, dk),
            folded_b.view(chunks * heads, dv, dk),
            step_t,
        ).view(chunks, heads, dv, dk)
        folded_t = torch.bmm(folded_t.view(chunks * heads, dk, dk), step_t).view(
            chunks, heads, dk, dk
        )
    return folded_t, folded_b


@functools.lru_cache(maxsize=64)
def _boundary_frames(
    num_frames: int, chunk: int, frame_offset: int, device: str
) -> tuple[int, torch.Tensor, torch.Tensor, torch.Tensor]:
    # the gather reads prefix at chunk ends and suffix at chunk starts, on the offset grid
    padded = frame_offset + num_frames
    num_chunks = -(-padded // chunk)
    ends = [
        min((c + 1) * chunk - 1, padded - 1) - frame_offset for c in range(num_chunks)
    ]
    starts = [c * chunk - frame_offset for c in range(num_chunks)]
    dev = torch.device(device)
    ends = [(f, c) for c, f in enumerate(ends) if f >= 0]
    starts = [(f, c) for c, f in enumerate(starts) if f >= 0]
    return (
        num_chunks,
        torch.tensor([f for f, _ in ends], device=dev),
        torch.tensor([c for _, c in ends], device=dev),
        torch.tensor([f for f, _ in starts], device=dev),
        torch.tensor([c for _, c in starts], device=dev),
    )


def run_boundary_scans(
    transitions: torch.Tensor,
    injections: torch.Tensor,
    text_state: torch.Tensor | None,
    *,
    chunk: int,
    frame_offset: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """``run_scans`` restricted to what the chunked gather reads: prefix at each
    chunk's last frame, suffix at each chunk's first frame, zero elsewhere.
    ``frame_offset`` is frame 0's position on the chunk grid (1 when the anchor
    frames were dropped). Same fp32 math, the products re-associated."""
    if chunk <= 1:
        return run_scans(transitions, injections, text_state)
    num_frames, heads, dv, dk = injections.shape
    num_chunks, ends, end_chunks, starts, start_chunks = _boundary_frames(
        num_frames, chunk, frame_offset, str(injections.device)
    )
    # identity / zero padding fills the leading offset and the partial last chunk
    lead, tail = frame_offset, num_chunks * chunk - frame_offset - num_frames
    eye = torch.eye(dk, device=transitions.device, dtype=transitions.dtype)
    transitions = torch.cat(
        [eye.expand(lead, heads, dk, dk), transitions, eye.expand(tail, heads, dk, dk)]
    )
    injections = torch.cat(
        [
            injections.new_zeros(lead, heads, dv, dk),
            injections,
            injections.new_zeros(tail, heads, dv, dk),
        ]
    )
    # frame-major so each composition step reads contiguous operands
    by_frame_t = (
        transitions.view(num_chunks, chunk, heads, dk, dk).transpose(0, 1).contiguous()
    )
    by_frame_b = (
        injections.view(num_chunks, chunk, heads, dv, dk).transpose(0, 1).contiguous()
    )
    start = (
        torch.zeros(heads, dv, dk, dtype=injections.dtype, device=injections.device)
        if text_state is None
        else text_state.to(injections.dtype)
    )
    # step c: the forward chain on chunk c and the reverse chain on chunk C-1-c
    fwd_t, fwd_b = _compose_chunk(by_frame_t, by_frame_b, reverse=False)
    rev_t, rev_b = _compose_chunk(by_frame_t, by_frame_b, reverse=True)
    chunk_t = torch.stack([fwd_t, rev_t.flip(0)], dim=1)  # [C, 2, H, dk, dk]
    boundary = torch.stack([fwd_b, rev_b.flip(0)], dim=1)  # [C, 2, H, dv, dk]
    flat = boundary.view(num_chunks, 2 * heads, dv, dk)
    state = torch.stack([start, start], dim=0).view(2 * heads, dv, dk)
    for c in range(num_chunks):
        flat[c].baddbmm_(state, chunk_t[c].view(2 * heads, dk, dk))
        state = flat[c]
    prefix = torch.zeros(
        num_frames, heads, dv, dk, dtype=injections.dtype, device=injections.device
    )
    suffix = torch.zeros_like(prefix)
    # step c holds chunk c's forward state and chunk C-1-c's reverse state
    prefix.index_copy_(0, ends, boundary[end_chunks, 0])
    suffix.index_copy_(0, starts, boundary[num_chunks - 1 - start_chunks, 1])
    return prefix, suffix


@functools.lru_cache(maxsize=64)
def _gather_indices(
    bounds: tuple[tuple[int, int], ...], num_frames: int, device: str
) -> tuple[torch.Tensor, ...]:
    # cached: rebuilding from Python lists per block costs two synchronizing H2D copies
    dev = torch.device(device)
    last_before = torch.tensor([lo for lo, _ in bounds], device=dev) - 1
    first_after = torch.tensor([hi for _, hi in bounds], device=dev) + 1
    return (
        last_before,
        first_after,
        last_before.clamp(min=0),
        first_after.clamp(max=num_frames - 1),
        last_before >= 0,
        first_after < num_frames,
        torch.arange(num_frames, device=dev),
    )


def gather_linear_state(
    prefix: torch.Tensor,
    suffix: torch.Tensor,
    alpha: torch.Tensor,
    bounds: list[tuple[int, int]],
    *,
    bridge: str,
    text_state: torch.Tensor | None,
    out_dtype: torch.dtype,
    fused: bool = True,
) -> torch.Tensor:
    """Everything OUTSIDE the softmax window of frame t, decayed to t:
    prefix[lo-1] * prod_{u=lo..t} alpha_u + suffix[hi+1] * prod_{u=t..hi} alpha_u.
    Out-of-range sides read the text state (the scans' virtual start) when one
    was given, else contribute nothing. -> [F, H, dv, dk] in ``out_dtype``."""
    num_frames = prefix.shape[0]
    (
        last_before,
        first_after,
        before_idx,
        after_idx,
        has_before,
        has_after,
        frames,
    ) = _gather_indices(tuple(bounds), num_frames, str(prefix.device))
    if fused and can_use_vdn_gather_linear_state(prefix):
        return vdn_gather_linear_state(
            prefix,
            suffix,
            alpha,
            text_state,
            before_idx=before_idx,
            after_idx=after_idx,
            has_before=has_before,
            has_after=has_after,
            bridge_before=(last_before + 1).clamp(min=0),
            bridge_after=first_after.clamp(max=num_frames),
            bridge=bridge == "alpha",
            out_dtype=out_dtype,
        )

    state_before = prefix[before_idx]
    state_after = suffix[after_idx]
    if text_state is not None:
        ts = text_state.to(state_before.dtype)
        state_before = torch.where(has_before.view(-1, 1, 1, 1), state_before, ts)
        state_after = torch.where(has_after.view(-1, 1, 1, 1), state_after, ts)
    if bridge == "alpha":
        log_alpha = torch.log(alpha.clamp_min(1e-12))
        log_prefix = torch.cat([torch.zeros_like(log_alpha[:1]), log_alpha.cumsum(0)])
        # an out-of-range side decays the text state over [0..t] or [t..F-1]
        bridge_before = (last_before + 1).clamp(min=0)
        bridge_after = first_after.clamp(max=num_frames)
        alpha_from_before = torch.exp(
            log_prefix[frames + 1] - log_prefix[bridge_before]
        )
        alpha_from_after = torch.exp(log_prefix[bridge_after] - log_prefix[frames])
        # alpha is per KEY channel: broadcast over dv, not dk
        state_before = state_before * alpha_from_before.unsqueeze(2)
        state_after = state_after * alpha_from_after.unsqueeze(2)
    elif bridge != "none":
        raise ValueError(f"unknown bridge {bridge!r}")
    if text_state is not None:
        out = state_before + state_after
    else:
        out = state_before * has_before.view(
            -1, 1, 1, 1
        ) + state_after * has_after.view(-1, 1, 1, 1)
    return out.to(out_dtype)


def linear_epilogue(
    readout: torch.Tensor, norm_weight: torch.Tensor, gate: torch.Tensor, eps: float
) -> torch.Tensor:
    """readout [F, H, S, dv] -> RMSNorm over dv -> * gate [F*S, H, dv] -> [F*S, H*dv]."""
    ms = (
        torch.linalg.vector_norm(readout, dim=-1, keepdim=True, dtype=_FP32).pow(2)
        / (readout.shape[-1])
    )
    normed = (
        readout
        * torch.rsqrt(ms + eps).to(readout.dtype)
        * norm_weight.to(readout.dtype)
    )
    frames, heads, per_frame, dim = normed.shape
    rows = frames * per_frame
    return normed.permute(0, 2, 1, 3).reshape(rows, heads * dim) * gate.reshape(
        rows, heads * dim
    )


# --------------------------------------------------------------------------
# The module
# --------------------------------------------------------------------------


class MiniMaxH3VDNLinearBranch(nn.Module):
    """VDN's BidirectionalLinearBranch on SGLang's TP-local head shard."""

    def __init__(
        self,
        arch: MiniMaxH3DiTArchConfig,
        hybrid: VDNHybridAttentionArchConfig,
        *,
        local_heads: int,
        prefix: str = "linear_attention",
    ) -> None:
        super().__init__()
        if hybrid.linear_head_dim != arch.attention_head_dim:
            # the branch reads the attention projections, so the head dims must agree
            raise ValueError(
                f"hybrid_attention.linear_head_dim={hybrid.linear_head_dim} != "
                f"attention_head_dim={arch.attention_head_dim}"
            )
        self.hybrid = hybrid
        self.local_heads = local_heads
        self.head_dim = arch.attention_head_dim
        # tests flip this to compare the fused Triton stages with the eager chain
        self.fused_kernels = True
        hidden = arch.hidden_size
        channels = local_heads * self.head_dim
        self.short_conv = (
            VDNShortConv(channels, hybrid.short_conv) if hybrid.short_conv else None
        )
        heads = arch.num_attention_heads
        self.alpha = VDNFrameAlpha(
            hidden, heads, local_heads, self.head_dim, prefix=f"{prefix}.alpha"
        )
        self.beta_proj = _head_sharded_linear(
            hidden, heads, bias=False, prefix=f"{prefix}.beta_proj"
        )
        self.output_gate = VDNOutputGate(
            hidden, heads, local_heads, self.head_dim, prefix=f"{prefix}.output_gate"
        )
        self.norm = _branch_norm(self.head_dim)

    # ---- pieces the attention module computes on the row shard (Ulysses) ----

    def beta(self, x: torch.Tensor) -> torch.Tensor:
        """x [T, hidden] -> beta [T, H_local] (sigmoid)."""
        beta, _ = self.beta_proj(x)
        return torch.sigmoid(beta)

    def gate(self, x: torch.Tensor) -> torch.Tensor:
        """x [T, hidden] -> output gate [T, H_local, d]."""
        return self.output_gate(x)

    # ---- the text state -----------------------------------------------------

    def text_statistics(
        self,
        text_k_raw: torch.Tensor,
        text_v_raw: torch.Tensor,
        text_beta: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, int]:
        """(A [1, H, dk, dk], B [1, H, dv, dk]) fp32 of the prompt rows, no conv."""
        length = text_k_raw.shape[0]
        heads, head_dim = text_k_raw.shape[1], self.head_dim
        key = linear_features(
            text_k_raw,
            proj="k",
            conv=None,
            num_frames=None,
            frame_size=None,
            fused=self.fused_kernels,
        )
        value = linear_features(
            text_v_raw,
            proj="v",
            conv=None,
            num_frames=None,
            frame_size=None,
            fused=self.fused_kernels,
        )
        key = key.view(1, length, heads, head_dim).permute(0, 2, 1, 3)
        value = value.view(1, length, heads, head_dim).permute(0, 2, 1, 3)
        beta = text_beta.view(1, length, heads).permute(0, 2, 1)
        A, B = frame_statistics(key, value, beta, a_fp32=self.hybrid.a_fp32)
        return A, B, length

    def text_state(
        self,
        text_k_raw: torch.Tensor,
        text_v_raw: torch.Tensor,
        text_beta: torch.Tensor,
    ) -> torch.Tensor:
        """S_text [H, dv, dk] fp32: the prompt written into a zero state as one
        delta-rule chunk, scaled by TEXT_STATE_SCALE."""
        A, B, length = self.text_statistics(text_k_raw, text_v_raw, text_beta)
        heads, head_dim = A.shape[1], self.head_dim
        ones = torch.ones(1, heads, head_dim, device=A.device, dtype=_FP32)
        _, injection = delta_factor_apply(
            self.hybrid.delta_rule,
            ones,
            A,
            B,
            tokens_per_frame=length,
            fused=self.fused_kernels,
        )
        return TEXT_STATE_SCALE * injection[0]

    # ---- the branch --------------------------------------------------------

    def forward(
        self,
        *,
        q_raw: torch.Tensor,
        k_raw: torch.Tensor,
        v_raw: torch.Tensor,
        beta: torch.Tensor,
        gate: torch.Tensor,
        frame_mean: torch.Tensor,
        layout: VDNH3Layout,
        text_k_raw: torch.Tensor | None = None,
        text_v_raw: torch.Tensor | None = None,
        text_beta: torch.Tensor | None = None,
        heads: slice | None = None,
    ) -> torch.Tensor:
        """Linear readout of the video rows, [V, H * d] in q's dtype.

        q/k/v/beta/gate are the video rows' raw (pre-norm, pre-RoPE) values on
        this rank's heads; ``heads`` is the head range of the full sequence a
        Ulysses rank processes, applied to the per-head parameters. Under
        anchor_frames == "both" frames 0 and F-1 read zero.
        """
        hybrid = self.hybrid
        num_frames, per_frame = layout.num_frames, layout.tokens_per_frame
        bounds = hybrid.window_bounds(num_frames)
        text_state = None
        text_stats = None
        if hybrid.enable_text_state:
            if text_k_raw is None or text_v_raw is None or text_beta is None:
                raise ValueError("enable_text_state needs the prompt rows' k/v/beta")
            if text_k_raw.shape[0] > 0:
                if hybrid.delta_rule == "vdn_solve":
                    # vdn_solve ignores tokens_per_frame, so the prompt joins the batch
                    A_text, B_text, _ = self.text_statistics(
                        text_k_raw, text_v_raw, text_beta
                    )
                    text_stats = (A_text, B_text)
                else:
                    text_state = self.text_state(text_k_raw, text_v_raw, text_beta)

        skip_ends = hybrid.anchor_frames == "both"
        n_heads = q_raw.shape[1]
        if not skip_ends:
            return self._readout(
                q_raw,
                k_raw,
                v_raw,
                beta,
                gate,
                frame_mean,
                num_frames,
                per_frame,
                bounds,
                layout.frame_size,
                text_state,
                heads,
                text_stats=text_stats,
            )
        out = q_raw.new_empty(num_frames * per_frame, n_heads * self.head_dim)
        if num_frames <= 2:
            return out.zero_()
        inner = slice(per_frame, (num_frames - 1) * per_frame)
        readout = self._readout(
            q_raw[inner],
            k_raw[inner],
            v_raw[inner],
            beta[inner],
            gate[inner],
            frame_mean[1:-1],
            num_frames - 2,
            per_frame,
            [(lo - 1, hi - 1) for lo, hi in bounds[1 : num_frames - 1]],
            layout.frame_size,
            text_state,
            heads,
            frame_offset=1,
            text_stats=text_stats,
        )
        out[:per_frame].zero_()
        out[(num_frames - 1) * per_frame :].zero_()
        out[inner] = readout
        return out

    def _readout(
        self,
        q_raw: torch.Tensor,
        k_raw: torch.Tensor,
        v_raw: torch.Tensor,
        beta: torch.Tensor,
        gate: torch.Tensor,
        frame_mean: torch.Tensor,
        num_frames: int,
        per_frame: int,
        bounds: list[tuple[int, int]],
        frame_size: tuple[int, int],
        text_state: torch.Tensor | None,
        heads: slice | None,
        frame_offset: int = 0,
        text_stats: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        n_heads, head_dim = q_raw.shape[1], self.head_dim
        shape = (num_frames, per_frame, n_heads, head_dim)
        fused = self.fused_kernels
        features = functools.partial(
            linear_features,
            conv=self.short_conv,
            num_frames=num_frames,
            frame_size=frame_size,
            heads=heads,
            fused=fused,
        )
        query_by_frame = features(q_raw, proj="q", frame_major=True)
        key = features(k_raw, proj="k")
        value = features(v_raw, proj="v")
        key_by_frame = key.view(shape).permute(0, 2, 1, 3)
        value_by_frame = value.view(shape).permute(0, 2, 1, 3)
        beta_by_frame = beta.view(num_frames, per_frame, n_heads).permute(0, 2, 1)
        prepared = (
            vdn_frame_stats_prep(key, value, beta, num_frames, per_frame)
            if fused and self.hybrid.a_fp32 and can_use_vdn_frame_stats_prep(key, value)
            else None
        )
        A, B = frame_statistics(
            key_by_frame,
            value_by_frame,
            beta_by_frame,
            a_fp32=self.hybrid.a_fp32,
            prepared=prepared,
        )
        del prepared
        alpha = self.alpha(frame_mean, heads=heads)
        if text_stats is not None:
            # the prompt leads as a virtual frame; alpha 1 since its old state is zero
            A = torch.cat([text_stats[0], A])
            B = torch.cat([text_stats[1], B])
            alpha_all = torch.cat([alpha.new_ones((1,) + alpha.shape[1:]), alpha])
        else:
            alpha_all = alpha
        transitions, injections = delta_factor_apply(
            self.hybrid.delta_rule,
            alpha_all,
            A,
            B,
            tokens_per_frame=per_frame,
            fused=self.fused_kernels,
        )
        if text_stats is not None:
            text_state = TEXT_STATE_SCALE * injections[0]
            transitions, injections = transitions[1:], injections[1:]
        prefix, suffix = run_boundary_scans(
            transitions,
            injections,
            text_state,
            chunk=self.hybrid.chunk,
            frame_offset=frame_offset,
        )
        del transitions, injections
        linear_state = gather_linear_state(
            prefix,
            suffix,
            alpha,
            bounds,
            bridge=self.hybrid.bridge,
            text_state=text_state,
            out_dtype=q_raw.dtype,
            fused=fused,
        )
        del prefix, suffix
        readout = torch.matmul(query_by_frame, linear_state.transpose(-1, -2))
        if fused and can_use_vdn_linear_epilogue(readout):
            return vdn_linear_epilogue(readout, self.norm.weight, gate, self.norm.eps)
        return linear_epilogue(readout, self.norm.weight, gate, self.norm.eps)


__all__ = [
    "MiniMaxH3VDNLinearBranch",
    "TEXT_STATE_SCALE",
    "VDNFrameAlpha",
    "VDNH3Layout",
    "VDNOutputGate",
    "VDNShortConv",
    "VDNSoftmaxGate",
    "delta_factor_apply",
    "frame_statistics",
    "gather_linear_state",
    "linear_epilogue",
    "linear_features",
    "run_boundary_scans",
    "run_scans",
    "vdn_h3_layout_from_packed",
]
