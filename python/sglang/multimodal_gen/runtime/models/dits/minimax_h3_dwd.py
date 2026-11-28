# SPDX-License-Identifier: Apache-2.0
# Adapted from LakonLab (Apache-2.0, https://github.com/Lakonik/LakonLab,
# lakonlab/models/architectures/sol_h3/), itself adapted from the Apache-2.0
# Sol-H3-VAE reference decoder.
"""Decoder-free (DWD) video output head of the pi-H3 MiniMax-H3 adapters.

At the last denoising step the DiT's final hidden states pass through their
own AdaLN output norm (``dwd_norm_out``) and a pixel refiner: the last eight
layers of the Sol-H3-VAE decoder, fed by a linear expansion of every DiT token
onto 16-pixel Sol tokens. The refiner predicts ImageNet-normalized RGB frames
directly, so the video skips the VAE decoder.

Sol layers attend within shifted windows of 5 latent frames x 16 x 16 tokens,
plus a few global tokens per native 5-latent block that see the whole block.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

# Sol-H3-VAE decoder constants
NUM_HEADS = 32
HEAD_DIM = 64
DIM = NUM_HEADS * HEAD_DIM
FFN_MULT = 4
NUM_LAYERS = 24
NORM_EPS = 1e-5
NUM_REGISTER_TOKENS = 4
ROPE_THETA = 100.0
ROPE_DIM = int(HEAD_DIM * 0.75)
WINDOW = 16  # spatial window in tokens; odd layers shift it by half
BLOCK = 5  # native latent frames per block (17 video frames)
T_SHIFT = 2  # temporal window shift of odd layers
TIME_NORM = 5
PATCH, PATCH_T, OUT_CH = 16, 4, 3
FRAMES_TO_TRIM = 3  # frames dropped from the first latent of every block


def _varlen_attention(q, k, v, cu_seqlens_q, cu_seqlens_k, max_q, max_k):
    from torch.nn.attention.varlen import varlen_attn

    return varlen_attn(q, k, v, cu_seqlens_q, cu_seqlens_k, max_q, max_k)


def _rope(T: int, H: int, W: int, s: int, st: int, device) -> tuple:
    """cos/sin (1, T*H*W, 1, ROPE_DIM // 2) with window-local time and space ids."""
    lt = ((torch.arange(T, device=device) + st) % BLOCK).float()
    ly = ((torch.arange(H, device=device) + s) % WINDOW).float()
    lx = ((torch.arange(W, device=device) + s) % WINDOW).float()
    t = (lt + 0.5) / TIME_NORM * 2 - 1
    hy = (ly + 0.5) / WINDOW * 2 - 1
    wx = (lx + 0.5) / WINDOW * 2 - 1
    grid = torch.stack(torch.meshgrid(t, hy, wx, indexing="ij"), -1).reshape(1, -1, 3)
    inv_freq = 1.0 / ROPE_THETA ** torch.arange(
        0, 1, 6 / ROPE_DIM, dtype=torch.float32, device=device
    )
    angles = 2.0 * math.pi * grid[:, :, :, None] * inv_freq[None, None, None, :]
    angles = angles.flatten(2, 3).unsqueeze(2)
    return angles.cos(), angles.sin()


def _rotate_(x: torch.Tensor, emb: tuple) -> None:
    """Rotate the leading ROPE_DIM dims of (B, L, heads, head_dim) x in place."""
    cos, sin = (t.to(x.dtype) for t in emb)
    half = cos.size(-1)
    first, second = x[..., :half], x[..., half : 2 * half]
    first_sin, second_sin = first * sin, second * sin
    first.mul_(cos).sub_(second_sin)
    second.mul_(cos).add_(first_sin)


def _partition(x: torch.Tensor, s: int, st: int):
    """(B, T, H, W, ...) -> windows (B*nW, BLOCK*WINDOW*WINDOW, ...), validity, layout."""
    B, T, H, W = x.shape[:4]
    rest = x.shape[4:]
    Tp = ((T + st + BLOCK - 1) // BLOCK) * BLOCK
    Hp = ((H + s + WINDOW - 1) // WINDOW) * WINDOW
    Wp = ((W + s + WINDOW - 1) // WINDOW) * WINDOW
    xp = x.new_zeros(B, Tp, Hp, Wp, *rest)
    xp[:, st : st + T, s : s + H, s : s + W] = x
    valid = torch.zeros(Tp, Hp, Wp, dtype=torch.bool, device=x.device)
    valid[st : st + T, s : s + H, s : s + W] = True
    nt, nh, nw = Tp // BLOCK, Hp // WINDOW, Wp // WINDOW
    r = len(rest)
    xw = xp.view(B, nt, BLOCK, nh, WINDOW, nw, WINDOW, *rest).permute(
        0, 1, 3, 5, 2, 4, 6, *range(7, 7 + r)
    )
    xw = xw.reshape(B * nt * nh * nw, BLOCK * WINDOW * WINDOW, *rest)
    vw = (
        valid.view(nt, BLOCK, nh, WINDOW, nw, WINDOW)
        .permute(0, 2, 4, 1, 3, 5)
        .reshape(nt * nh * nw, BLOCK * WINDOW * WINDOW)
    )
    return xw, vw, (nt, nh, nw)


def _pack(rows: torch.Tensor, valid: torch.Tensor, total: int):
    """The ``total`` valid entries of (B, n, L) rows in order, with cu_seqlens."""
    valid = valid.expand(rows.shape).flatten(0, 1)
    order = valid.logical_not().flatten().to(torch.int8).argsort(stable=True)[:total]
    cu_seqlens = F.pad(
        valid.sum(1, dtype=torch.int32).cumsum(0, dtype=torch.int32), (1, 0)
    )
    return rows.flatten()[order], cu_seqlens


def _window_rows(B, T, H, W, nb, S, s, st, device):
    """Packed query/key rows and cu_seqlens of the shifted-window attention.

    Rows index the B*T*H*W tokens followed by the (B, nb, S) global tokens. Each
    window's keys append the global tokens of the native blocks it overlaps.
    """
    N = T * H * W
    tokens, valid, (nt, nh, nw) = _partition(
        torch.arange(1, N + 1, device=device).view(1, T, H, W), s, st
    )
    # native blocks of the first and last latent of temporal window j
    blocks = [
        (
            max(j * BLOCK - st, 0) // BLOCK,
            (min((j + 1) * BLOCK - st, T) - 1) // BLOCK,
        )
        for j in range(nt)
    ]
    global_rows = torch.tensor(
        [[b * S + i for b in pair for i in range(S)] for pair in blocks], device=device
    )
    global_valid = torch.tensor(
        [[True] * S + [b1 != b0] * S for b0, b1 in blocks], device=device
    )
    global_rows, global_valid = (
        t[:, None, None].expand(nt, nh, nw, 2 * S).reshape(-1, 2 * S)
        for t in (global_rows, global_valid)
    )
    batch = torch.arange(B, device=device)[:, None, None]
    token_rows = batch * N + tokens - 1
    key_rows = torch.cat([token_rows, B * N + batch * nb * S + global_rows], -1)
    key_valid = torch.cat([valid, global_valid], -1)
    q_rows, cu_seqlens_q = _pack(token_rows, valid, B * N)
    k_rows, cu_seqlens_k = _pack(
        key_rows,
        key_valid,
        B * (N + nh * nw * S * sum(1 + (b1 != b0) for b0, b1 in blocks)),
    )
    return q_rows, k_rows, cu_seqlens_q, cu_seqlens_k


def _global_rows(B, T, H, W, nb, S, device):
    """Packed key rows of the global tokens: their native block's tokens, then
    the block's global tokens; with the query and key cu_seqlens."""
    N = T * H * W
    tokens = torch.arange(nb * BLOCK * H * W, device=device).view(nb, -1)
    batch = torch.arange(B, device=device)[:, None, None]
    globals_ = torch.arange(nb * S, device=device).view(nb, S)
    rows = torch.cat([batch * N + tokens, B * N + batch * nb * S + globals_], -1)
    valid = torch.cat(
        [tokens < N, torch.ones(nb, S, dtype=torch.bool, device=device)], -1
    )
    rows, cu_seqlens_k = _pack(rows, valid, B * (N + nb * S))
    cu_seqlens_q = torch.arange(B * nb + 1, dtype=torch.int32, device=device) * S
    return rows, cu_seqlens_q, cu_seqlens_k


class _SolAttention(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.to_qkv = nn.Linear(DIM, 3 * DIM)
        self.to_out = nn.Linear(DIM, DIM)


class _SolSwiGLU(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.w1 = nn.Linear(DIM, 2 * DIM * FFN_MULT)
        self.w2 = nn.Linear(DIM * FFN_MULT, DIM)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate, up = self.w1(x).chunk(2, dim=-1)
        return self.w2(F.silu(gate) * up)


class _SolBlock(nn.Module):
    """One Sol decoder layer on tokens x (B, T, H, W, D) and global tokens g (B, nb, S, D)."""

    def __init__(self) -> None:
        super().__init__()
        self.norm1 = nn.RMSNorm(DIM, eps=NORM_EPS)
        self.norm2 = nn.RMSNorm(DIM, eps=NORM_EPS)
        self.scale1 = nn.Parameter(torch.empty(DIM))
        self.scale2 = nn.Parameter(torch.empty(DIM))
        self.attn = _SolAttention()
        self.ff = _SolSwiGLU()

    def forward(self, x, g, shifted, rope, global_indices, window_indices):
        x, g = self._attention(x, g, shifted, rope, global_indices, window_indices)
        x = x + self.ff(self.norm2(x.float()).to(x.dtype)) * self.scale2
        g = g + self.ff(self.norm2(g.float()).to(g.dtype)) * self.scale2
        return x, g

    def _attention(self, x, g, shifted, rope, global_indices, window_indices):
        B, T, H, W, _ = x.shape
        nb, S = g.shape[1:3]
        N = B * T * H * W
        # token rows, then global-token rows
        qkv = self.attn.to_qkv(
            self.norm1(torch.cat([x.flatten(0, 3), g.flatten(0, 2)]).float()).to(
                x.dtype
            )
        )
        qkv = qkv.view(-1, NUM_HEADS, 3, HEAD_DIM)
        q = F.rms_norm(qkv[:, :, 0].float(), (HEAD_DIM,), eps=NORM_EPS).to(x.dtype)
        k = F.rms_norm(qkv[:, :, 1].float(), (HEAD_DIM,), eps=NORM_EPS).to(x.dtype)
        v = qkv[:, :, 2]
        _rotate_(q[:N].view(B, -1, NUM_HEADS, HEAD_DIM), rope)
        _rotate_(k[:N].view(B, -1, NUM_HEADS, HEAD_DIM), rope)
        rows, cu_q, cu_k = global_indices
        og = _varlen_attention(
            q[N:], k[rows], v[rows], cu_q, cu_k, S, BLOCK * H * W + S
        )
        q_rows, k_rows, cu_q, cu_k = window_indices
        o = _varlen_attention(
            q[q_rows],
            k[k_rows],
            v[k_rows],
            cu_q,
            cu_k,
            BLOCK * WINDOW**2,
            BLOCK * WINDOW**2 + 2 * S,
        )
        del qkv, q, k, v
        o = torch.empty_like(o).index_copy_(0, q_rows, o)  # back to token order
        x = x + self.attn.to_out(o.flatten(1)).view_as(x) * self.scale1
        g = g + self.attn.to_out(og.flatten(1)).view_as(g) * self.scale1
        return x, g


class MiniMaxH3DWDNormOut(nn.Module):
    """DWD output norm: affine-free LayerNorm, AdaLN from SiLU of the pruned curve."""

    def __init__(self, hidden_size: int, condition_dim: int, eps: float) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.eps = eps
        self.linear = nn.Linear(condition_dim, 2 * hidden_size, dtype=torch.float32)

    def forward(
        self, x: torch.Tensor, adaln_input: torch.Tensor, indices: torch.Tensor
    ) -> torch.Tensor:
        scale, shift = self.linear(F.silu(adaln_input.float())).chunk(2, dim=-1)
        h = F.layer_norm(x, (self.hidden_size,), eps=self.eps)
        return (h * (1 + scale[indices]) + shift[indices]).to(torch.bfloat16)


class MiniMaxH3DWDPixelRefiner(nn.Module):
    """The last ``num_layers`` Sol-H3-VAE decoder layers over expanded DiT tokens.

    Returns ImageNet-normalized RGB (B, 3, F, 16H, 16W) for a (T, H, W) grid of
    DiT tokens that each cover one latent frame and ``src_patch`` pixels; the
    first latent of every 5-latent block decodes one frame, the others four.
    """

    def __init__(
        self,
        in_channels: int,
        src_patch_size: tuple[int, int, int],
        first_frame_period: int,
        num_layers: int = 8,
    ) -> None:
        super().__init__()
        _, src_h, src_w = src_patch_size
        if (
            src_patch_size[0] != PATCH_T
            or src_h % PATCH
            or src_w % PATCH
            or first_frame_period != BLOCK
        ):
            raise ValueError(
                "the DWD pixel geometry must follow Sol's frame trimming at every "
                f"5-latent block, got src_patch_size={src_patch_size}, "
                f"first_frame_period={first_frame_period}"
            )
        self.expansion = (src_h // PATCH, src_w // PATCH)
        self.num_layers = num_layers
        self.feature_expand = nn.Linear(
            in_channels, DIM * self.expansion[0] * self.expansion[1]
        )
        self.register_tokens = nn.Parameter(torch.empty(1, NUM_REGISTER_TOKENS, DIM))
        self.blocks = nn.ModuleList([_SolBlock() for _ in range(num_layers)])
        self.norm_out = nn.LayerNorm(DIM, eps=NORM_EPS)
        self.proj_out = nn.Linear(DIM, OUT_CH * PATCH_T * PATCH * PATCH)
        self.first_frame_proj_out = nn.Linear(
            DIM, OUT_CH * (PATCH_T - FRAMES_TO_TRIM) * PATCH * PATCH
        )

    @torch.no_grad()
    def forward(
        self, features: torch.Tensor, token_grid: tuple[int, int, int]
    ) -> torch.Tensor:
        """Decode (B, t*h*w, in_channels) features on a (t, h, w) token grid."""
        B = features.size(0)
        t, h, w = token_grid
        e_h, e_w = self.expansion
        x = self.feature_expand(features.to(self.register_tokens.dtype))
        x = (
            x.view(B, t, h, w, e_h, e_w, DIM)
            .permute(0, 1, 2, 4, 3, 5, 6)
            .reshape(B, t, h * e_h, w * e_w, DIM)
        )
        T, H, W = t, h * e_h, w * e_w
        nb = (T + BLOCK - 1) // BLOCK
        g = torch.cat(
            [self.register_tokens.expand(B, -1, -1), x.new_zeros(B, 1, DIM)], 1
        )
        g = g.unsqueeze(1).expand(B, nb, -1, -1).contiguous().to(x.dtype)
        S = g.shape[2]
        device = x.device
        global_indices = _global_rows(B, T, H, W, nb, S, device)
        window_indices = {}
        ropes = {}
        for index, block in enumerate(self.blocks, NUM_LAYERS - self.num_layers):
            shifted = index % 2 == 1
            if shifted not in window_indices:
                s, st = (WINDOW // 2, T_SHIFT) if shifted else (0, 0)
                window_indices[shifted] = _window_rows(B, T, H, W, nb, S, s, st, device)
                ropes[shifted] = _rope(T, H, W, s, st, device)
            x, g = block(
                x, g, shifted, ropes[shifted], global_indices, window_indices[shifted]
            )
        x = self.norm_out(x).to(self.proj_out.weight.dtype)
        num_chunks, tail = divmod(T, BLOCK)
        outs = []
        if num_chunks:
            outs.append(
                self._project_pixels(
                    x[:, : num_chunks * BLOCK].unflatten(1, (num_chunks, BLOCK))
                ).flatten(1, 2)
            )
        if tail:
            outs.append(
                self._project_pixels(x[:, num_chunks * BLOCK :].unsqueeze(1)).flatten(
                    1, 2
                )
            )
        # (B, F, 3, H, W) -> (B, 3, F, H, W)
        return torch.cat(outs, dim=1).transpose(1, 2)

    def _project_pixels(self, x: torch.Tensor) -> torch.Tensor:
        """Project chunks (B, n, t, H, W, D) into (B, n, F, 3, 16H, 16W)."""
        B, n, t, H, W, _ = x.shape
        first = self.first_frame_proj_out(x[:, :, :1])
        first = (
            first.view(B, n, H, W, OUT_CH, PATCH_T - FRAMES_TO_TRIM, PATCH, PATCH)
            .permute(0, 1, 5, 4, 2, 6, 3, 7)
            .reshape(B, n, PATCH_T - FRAMES_TO_TRIM, OUT_CH, H * PATCH, W * PATCH)
        )
        if t > 1:
            rest = self.proj_out(x[:, :, 1:])
            rest = (
                rest.view(B, n, t - 1, H, W, OUT_CH, PATCH_T, PATCH, PATCH)
                .permute(0, 1, 2, 6, 5, 3, 7, 4, 8)
                .reshape(B, n, (t - 1) * PATCH_T, OUT_CH, H * PATCH, W * PATCH)
            )
            first = torch.cat((first, rest), dim=2)
        return first


__all__ = ["MiniMaxH3DWDNormOut", "MiniMaxH3DWDPixelRefiner"]
