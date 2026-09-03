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

Everything here is inference-only and eager (no autograd). The module keeps
VDN's parameter names one level below ``blocks.N.attn.linear_attention`` so
the linear-branch safetensors load through ``param_names_mapping`` untouched.
Heads are sharded under tensor parallelism; ``alpha.down`` and
``output_gate.down`` are replicated (they are per-token bottlenecks shared by
every head).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from sglang.multimodal_gen.configs.models.dits.minimax_h3 import (
    MiniMaxH3DiTArchConfig,
    VDNHybridAttentionArchConfig,
)
from sglang.kernels.ops.diffusion import (
    vdn_frame_stats_prep,
    vdn_linear_epilogue,
    vdn_silu_l2norm,
    vdn_temporal_conv_act,
)
from sglang.multimodal_gen.runtime.distributed import (
    get_tp_rank,
    get_tp_world_size,
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


@dataclass(frozen=True)
class VDNH3Layout:
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
    """Weight loader that takes this TP rank's contiguous slice along
    ``shard_dim`` (rows laid out head-major, so a head slice is contiguous)."""

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


class _Linear(nn.Module):
    """``nn.Linear``-shaped parameters with optional head-sharded output rows."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        bias: bool,
        dtype: torch.dtype,
        shard_output: bool,
    ) -> None:
        super().__init__()
        self.weight = _make_param(
            (out_features, in_features),
            dtype=dtype,
            shard_dim=0 if shard_output else None,
        )
        if bias:
            self.bias = _make_param(
                (out_features,), dtype=dtype, shard_dim=0 if shard_output else None
            )
        else:
            self.register_parameter("bias", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.weight, self.bias)


class VDNFrameAlpha(nn.Module):
    """alpha_t = exp(-exp(A_log) * softplus(up(down(frame_mean)) + dt_bias)),
    per frame / head / key channel, in fp32 (KDA's double-exponential gate)."""

    def __init__(self, hidden_size: int, local_heads: int, head_dim: int) -> None:
        super().__init__()
        self.local_heads, self.head_dim = local_heads, head_dim
        self.down = _Linear(hidden_size, head_dim, bias=False, dtype=_BF16, shard_output=False)
        self.up = _Linear(head_dim, local_heads * head_dim, bias=False, dtype=_BF16, shard_output=True)
        # fp32 islands: the scan multiplies alpha across ~100 frames, so a bf16
        # gate compounds into tens of percent of retention error.
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

    def __init__(self, hidden_size: int, local_heads: int, head_dim: int) -> None:
        super().__init__()
        self.local_heads, self.head_dim = local_heads, head_dim
        self.down = _Linear(hidden_size, head_dim, bias=False, dtype=_BF16, shard_output=False)
        self.up = _Linear(head_dim, local_heads * head_dim, bias=True, dtype=_BF16, shard_output=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.up(self.down(x))).view(
            -1, self.local_heads, self.head_dim
        )


class VDNSoftmaxGate(nn.Module):
    """Per-(token, head) sigmoid gate on the softmax branch output."""

    def __init__(self, hidden_size: int, local_heads: int) -> None:
        super().__init__()
        self.up = _Linear(hidden_size, local_heads, bias=True, dtype=_BF16, shard_output=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.up(x))


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
                    {"weight": _make_param((channels, 1, k, k), dtype=_BF16, shard_dim=0)}
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
        volume = tokens.reshape(num_frames, grid_h, grid_w, channels).permute(0, 3, 1, 2)
        volume = F.conv2d(volume, w_sp, padding=SHORT_CONV_KERNEL // 2, groups=channels)
        x = volume.permute(0, 2, 3, 1).reshape(num_frames, grid_h * grid_w, channels)
        return x, w_tm.squeeze(1).to(x.dtype)


class VDNRMSNorm(nn.Module):
    """RMSNorm(head_dim) with fp32 second-moment accumulation (VDN ops.rms_norm)."""

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = _make_param((dim,), dtype=_BF16, shard_dim=None)


# --------------------------------------------------------------------------
# The algorithm (eager, inference-only)
# --------------------------------------------------------------------------


def _temporal_shift(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """Depthwise 5-tap conv over frames as shift-multiply-add. x [F, S, C];
    w [C, 5] in x's dtype; zero padded, symmetric."""
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
) -> torch.Tensor:
    """[N, H, d] raw projection -> [N, H, d] branch features:
    [short conv ->] SiLU [-> L2 norm for q, k]."""
    l2norm = proj != "v"
    fused = tokens.is_cuda and _use_fused_kernels()
    if conv is not None and proj in conv.targets:
        if frame_size is None or num_frames is None:
            raise ValueError("the short conv needs the (frames, height, width) grid")
        heads_n, head_dim = tokens.shape[-2], tokens.shape[-1]
        x, w_tm = conv.spatial(proj, tokens, num_frames, frame_size, heads=heads)
        if fused:
            # one kernel: 5 taps + SiLU + L2 norm, the conv output never hits HBM
            return vdn_temporal_conv_act(x, w_tm, heads_n, head_dim, l2norm)
        out = _temporal_shift(x, w_tm).reshape(-1, heads_n, head_dim)
        return _activate(out, l2norm)
    if fused:
        return vdn_silu_l2norm(tokens, l2norm)
    return _activate(tokens, l2norm)


_FUSED_KERNELS_ENABLED = True


def _use_fused_kernels() -> bool:
    return _FUSED_KERNELS_ENABLED


def set_fused_kernels_enabled(enabled: bool) -> None:
    """Test/debug switch between the fused Triton stages and the eager chain."""
    global _FUSED_KERNELS_ENABLED
    _FUSED_KERNELS_ENABLED = bool(enabled)


def frame_statistics(
    kf: torch.Tensor,
    vf: torch.Tensor,
    beta: torch.Tensor,
    *,
    a_fp32: bool,
    prepared: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor] | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """kf, vf [F, H, S, d]; beta [F, H, S] -> A [F, H, dk, dk] fp32 (symmetric),
    B [F, H, dv, dk] fp32.

    A is the matrix the scan inverts; computed in fp32 (bf16 breaks the
    conditioning of I + A on real, strongly correlated frame tokens). B enters
    the state linearly, so bf16 tensor cores are fine. ``prepared`` carries
    the four GEMM operands from ``vdn_frame_stats_prep`` (one fused pass).
    """
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
        # TF32 (10 mantissa bits) keeps I + A well conditioned where bf16
        # does not, and runs the GEMM on tensor cores; scoped to this matmul.
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
) -> tuple[torch.Tensor, torch.Tensor]:
    """One frame's statistics -> (transition [F,H,dk,dk], injection [F,H,dv,dk]) fp32.

    vdn_solve  (the released checkpoints): S' = (S diag(alpha) + B)(I + A)^-1,
               exact Cholesky inverse.
    sana_scaled: S' = (S diag(alpha))(I - c^2 A) + c B, c = 1/sqrt(S).
    vdn_scaled: S' = (S diag(alpha) + c B)(I + c^2 A)^-1.
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
    chol = torch.linalg.cholesky(A32 + eye)
    # (I+A)^-1 = L^-T L^-1 as one triangular solve and a matmul (VDN's choice:
    # a batched trsm at 128x128 is far slower than a batched GEMM).
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


def gather_linear_state(
    prefix: torch.Tensor,
    suffix: torch.Tensor,
    alpha: torch.Tensor,
    bounds: list[tuple[int, int]],
    *,
    bridge: str,
    text_state: torch.Tensor | None,
    out_dtype: torch.dtype,
) -> torch.Tensor:
    """Everything OUTSIDE the softmax window of frame t, decayed to t:
    prefix[lo-1] * prod_{u=lo..t} alpha_u + suffix[hi+1] * prod_{u=t..hi} alpha_u.
    Out-of-range sides read the text state (the scans' virtual start) when one
    was given, else contribute nothing. -> [F, H, dv, dk] in ``out_dtype``."""
    num_frames = prefix.shape[0]
    device = prefix.device
    last_before = torch.tensor([lo for lo, _ in bounds], device=device) - 1
    first_after = torch.tensor([hi for _, hi in bounds], device=device) + 1
    before_idx = last_before.clamp(min=0)
    after_idx = first_after.clamp(max=num_frames - 1)
    has_before = last_before >= 0
    has_after = first_after < num_frames
    frames = torch.arange(num_frames, device=device)

    state_before = prefix[before_idx]
    state_after = suffix[after_idx]
    if text_state is not None:
        ts = text_state.to(state_before.dtype)
        state_before = torch.where(has_before.view(-1, 1, 1, 1), state_before, ts)
        state_after = torch.where(has_after.view(-1, 1, 1, 1), state_after, ts)
    if bridge == "alpha":
        log_alpha = torch.log(alpha.clamp_min(1e-12))
        log_prefix = torch.cat([torch.zeros_like(log_alpha[:1]), log_alpha.cumsum(0)])
        # boundary rows decay the text state over the frames they really
        # skipped: from virtual -1 that is [0..t], from virtual F it is [t..F-1]
        bridge_before = (last_before + 1).clamp(min=0)
        bridge_after = first_after.clamp(max=num_frames)
        alpha_from_before = torch.exp(log_prefix[frames + 1] - log_prefix[bridge_before])
        alpha_from_after = torch.exp(log_prefix[bridge_after] - log_prefix[frames])
        # alpha is per KEY channel: broadcast over dv, not dk
        state_before = state_before * alpha_from_before.unsqueeze(2)
        state_after = state_after * alpha_from_after.unsqueeze(2)
    elif bridge != "none":
        raise ValueError(f"unknown bridge {bridge!r}")
    if text_state is not None:
        out = state_before + state_after
    else:
        out = state_before * has_before.view(-1, 1, 1, 1) + state_after * has_after.view(
            -1, 1, 1, 1
        )
    return out.to(out_dtype)


def linear_epilogue(
    readout: torch.Tensor, norm_weight: torch.Tensor, gate: torch.Tensor, eps: float
) -> torch.Tensor:
    """readout [F, H, S, dv] -> RMSNorm over dv -> * gate [F*S, H, dv] -> [F*S, H*dv]."""
    ms = torch.linalg.vector_norm(readout, dim=-1, keepdim=True, dtype=_FP32).pow(2) / (
        readout.shape[-1]
    )
    normed = readout * torch.rsqrt(ms + eps).to(readout.dtype) * norm_weight.to(readout.dtype)
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
    ) -> None:
        super().__init__()
        if hybrid.linear_head_dim != arch.attention_head_dim:
            # The branch shares the softmax branch's raw q/k/v, so the linear
            # head dim is the attention head dim by construction.
            raise ValueError(
                f"hybrid_attention.linear_head_dim={hybrid.linear_head_dim} != "
                f"attention_head_dim={arch.attention_head_dim}"
            )
        self.hybrid = hybrid
        self.local_heads = local_heads
        self.head_dim = arch.attention_head_dim
        hidden = arch.hidden_size
        channels = local_heads * self.head_dim
        self.short_conv = (
            VDNShortConv(channels, hybrid.short_conv) if hybrid.short_conv else None
        )
        self.alpha = VDNFrameAlpha(hidden, local_heads, self.head_dim)
        self.beta_proj = _Linear(hidden, local_heads, bias=False, dtype=_BF16, shard_output=True)
        self.output_gate = VDNOutputGate(hidden, local_heads, self.head_dim)
        self.norm = VDNRMSNorm(self.head_dim)

    # ---- pieces the attention module computes on the row shard (Ulysses) ----

    def beta(self, x: torch.Tensor) -> torch.Tensor:
        """x [T, hidden] -> beta [T, H_local] (sigmoid)."""
        return torch.sigmoid(self.beta_proj(x))

    def gate(self, x: torch.Tensor) -> torch.Tensor:
        """x [T, hidden] -> output gate [T, H_local, d]."""
        return self.output_gate(x)

    # ---- the text state -----------------------------------------------------

    def text_state(
        self,
        text_k_raw: torch.Tensor,
        text_v_raw: torch.Tensor,
        text_beta: torch.Tensor,
    ) -> torch.Tensor:
        """S_text [H, dv, dk] fp32: the whole prompt written into a zero state as
        one delta-rule chunk (no conv, no causal scan; alpha plays no part
        because the old state is zero), scaled by TEXT_STATE_SCALE."""
        length = text_k_raw.shape[0]
        heads, head_dim = text_k_raw.shape[1], self.head_dim
        key = linear_features(text_k_raw, proj="k", conv=None, num_frames=None, frame_size=None)
        value = linear_features(text_v_raw, proj="v", conv=None, num_frames=None, frame_size=None)
        key = key.view(1, length, heads, head_dim).permute(0, 2, 1, 3)
        value = value.view(1, length, heads, head_dim).permute(0, 2, 1, 3)
        beta = text_beta.view(1, length, heads).permute(0, 2, 1)
        A, B = frame_statistics(key, value, beta, a_fp32=self.hybrid.a_fp32)
        ones = torch.ones(1, heads, head_dim, device=A.device, dtype=_FP32)
        _, injection = delta_factor_apply(
            self.hybrid.delta_rule, ones, A, B, tokens_per_frame=length
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
        """Linear readout for every video row, [V, H * d] in q's dtype.

        ``heads``: under Ulysses each rank owns every head's weights but
        processes one head range of the full sequence after the all-to-all;
        the per-head parameters (alpha's up/dt_bias/A_log, the conv channels)
        take that slice. beta / gate arrive already head-sharded.

        q_raw/k_raw/v_raw: the VIDEO rows' raw (pre-QK-norm, pre-RoPE)
        projections [V, H_local, d]; beta [V, H_local]; gate [V, H_local, d];
        frame_mean [F, hidden] fp32 over the video rows of each frame; the
        text_* arguments are the prompt rows (enable_text_state).
        Under anchor_frames == "both" frames 0 and F-1 are exact in the softmax
        branch, so they are dropped from the input and their rows read zero.
        """
        hybrid = self.hybrid
        num_frames, per_frame = layout.num_frames, layout.tokens_per_frame
        bounds = hybrid.window_bounds(num_frames)
        text_state = None
        if hybrid.enable_text_state:
            if text_k_raw is None or text_v_raw is None or text_beta is None:
                raise ValueError("enable_text_state needs the prompt rows' k/v/beta")
            if text_k_raw.shape[0] > 0:
                text_state = self.text_state(text_k_raw, text_v_raw, text_beta)

        skip_ends = hybrid.anchor_frames == "both"
        n_heads = q_raw.shape[1]
        if not skip_ends:
            return self._readout(
                q_raw, k_raw, v_raw, beta, gate, frame_mean, num_frames, per_frame,
                bounds, layout.frame_size, text_state, heads,
            )
        out = q_raw.new_empty(num_frames * per_frame, n_heads * self.head_dim)
        if num_frames <= 2:
            return out.zero_()
        inner = slice(per_frame, (num_frames - 1) * per_frame)
        readout = self._readout(
            q_raw[inner], k_raw[inner], v_raw[inner], beta[inner], gate[inner],
            frame_mean[1:-1], num_frames - 2, per_frame,
            [(lo - 1, hi - 1) for lo, hi in bounds[1 : num_frames - 1]],
            layout.frame_size, text_state, heads,
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
    ) -> torch.Tensor:
        n_heads, head_dim = q_raw.shape[1], self.head_dim
        shape = (num_frames, per_frame, n_heads, head_dim)
        conv = self.short_conv
        # 1. features (q frame-major for the bmm readout)
        query = linear_features(q_raw, proj="q", conv=conv, num_frames=num_frames, frame_size=frame_size, heads=heads)
        key = linear_features(k_raw, proj="k", conv=conv, num_frames=num_frames, frame_size=frame_size, heads=heads)
        value = linear_features(v_raw, proj="v", conv=conv, num_frames=num_frames, frame_size=frame_size, heads=heads)
        query_by_frame = query.view(shape).permute(0, 2, 1, 3)
        key_by_frame = key.view(shape).permute(0, 2, 1, 3)
        value_by_frame = value.view(shape).permute(0, 2, 1, 3)
        beta_by_frame = beta.view(num_frames, per_frame, n_heads).permute(0, 2, 1)
        # 2. per-frame statistics
        fused = q_raw.is_cuda and _use_fused_kernels()
        prepared = (
            vdn_frame_stats_prep(key, value, beta, num_frames, per_frame)
            if fused and self.hybrid.a_fp32
            else None
        )
        A, B = frame_statistics(
            key_by_frame, value_by_frame, beta_by_frame,
            a_fp32=self.hybrid.a_fp32, prepared=prepared,
        )
        del prepared
        alpha = self.alpha(frame_mean, heads=heads)
        # 3. scans
        transitions, injections = delta_factor_apply(
            self.hybrid.delta_rule, alpha, A, B, tokens_per_frame=per_frame
        )
        prefix, suffix = run_scans(transitions, injections, text_state)
        del transitions, injections
        # 4. boundary gather
        linear_state = gather_linear_state(
            prefix, suffix, alpha, bounds,
            bridge=self.hybrid.bridge, text_state=text_state, out_dtype=q_raw.dtype,
        )
        del prefix, suffix
        # 5. readout
        readout = torch.matmul(query_by_frame, linear_state.transpose(-1, -2))
        if fused:
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
    "run_scans",
    "vdn_h3_layout_from_packed",
]
