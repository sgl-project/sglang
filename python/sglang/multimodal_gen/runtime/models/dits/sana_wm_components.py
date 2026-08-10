# SPDX-License-Identifier: Apache-2.0
"""SANA-WM DiT building blocks (components).

Reusable primitives, RoPE / UCPE camera geometry, GDN scan kernels, embedders,
FFN, and the SANA-WM attention modules.
"""

import math
from typing import Callable, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers.models.embeddings import get_1d_rotary_pos_embed

from sglang.multimodal_gen.runtime.layers.attention import LocalAttention
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

logger = init_logger(__name__)

_SANA_WM_TRITON_GDN_DISABLED_REASON: Optional[str] = None
_SANA_WM_TRITON_GDN_FALLBACK_LOGGED = False
_SANA_WM_TRITON_CAM_GDN_DISABLED_REASON: Optional[str] = None
_SANA_WM_TRITON_CAM_GDN_FALLBACK_LOGGED = False

# Streaming per-block cache: a 10-slot list per block (mirrors the reference
# SANA-WM forward_long). GDN blocks store recurrent state (0/1 main, 2 cam),
# the K short-conv prefix (4), and a STATE type flag (6); softmax blocks store
# a K/V concat-window (0/1 main, 2/3 cam) and a CONCAT type flag (6); all blocks
# store the FFN temporal-conv tail (9). Slots 5/7/8 are unused.
_NUM_STREAM_CACHE_SLOTS = 10
_SLOT_K = 0
_SLOT_V = 1
_SLOT_CAM_K = 2
_SLOT_CAM_V = 3
_SLOT_SHORTCONV = 4
_SLOT_TYPE_FLAG = 6
_SLOT_FFN_TCONV = 9
_CACHE_TYPE_CONCAT = 0.0
_CACHE_TYPE_STATE = 1.0


def _tensor_cache_key(tensor: torch.Tensor) -> Tuple:
    # Inference-mode tensors (e.g. an un-laundered camera tensor on the cfg=1.0
    # streaming path) raise on ._version access; they are never mutated, so 0 is safe.
    try:
        version = int(tensor._version)
    except (RuntimeError, AttributeError):
        version = 0
    return (
        tuple(tensor.shape),
        tuple(tensor.stride()),
        str(tensor.device),
        tensor.dtype,
        tensor.data_ptr(),
        version,
    )


def _log_sana_wm_triton_gdn_fallback(reason: str) -> None:
    global _SANA_WM_TRITON_GDN_FALLBACK_LOGGED
    if not _SANA_WM_TRITON_GDN_FALLBACK_LOGGED:
        logger.warning(
            "SANA-WM Triton GDN fast path is unavailable; falling back to torch "
            "GDN scan. reason=%s",
            reason,
        )
        _SANA_WM_TRITON_GDN_FALLBACK_LOGGED = True


def _log_sana_wm_triton_cam_gdn_fallback(reason: str) -> None:
    global _SANA_WM_TRITON_CAM_GDN_FALLBACK_LOGGED
    if not _SANA_WM_TRITON_CAM_GDN_FALLBACK_LOGGED:
        logger.warning(
            "SANA-WM Triton camera GDN fast path is unavailable; falling back "
            "to torch camera scan. reason=%s",
            reason,
        )
        _SANA_WM_TRITON_CAM_GDN_FALLBACK_LOGGED = True


# ---------------------------------------------------------------------------
# Small primitives (RMSNorm, ShortConvolution). Parameter names/shapes match
# the released checkpoint so it loads cleanly.
# ---------------------------------------------------------------------------


class _RMSNorm(nn.Module):
    """RMSNorm with signature ``RMSNorm(dim, scale_factor=1.0, eps=1e-6)``.

    Parameter name: ``weight`` (shape ``(dim,)``).
    """

    def __init__(
        self,
        dim: int,
        scale_factor: float = 1.0,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim) * scale_factor)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Upstream Sana RMSNorm does both normalization and weight multiply in
        # fp32 before casting back to the input dtype.
        x_in = x
        x32 = x.float()
        rms = x32.pow(2).mean(dim=-1, keepdim=True).add(self.eps).rsqrt()
        return (x32 * rms * self.weight.to(dtype=x32.dtype)).type_as(x_in)


class _ShortConvolution(nn.Module):
    """Depth-wise causal Conv1d along the temporal axis.

    Weight shape ``(hidden_size, 1, K)`` (groups=hidden_size). Input is
    ``(B, T, C)`` with causal padding of ``K-1`` on the left.
    """

    def __init__(self, hidden_size: int, kernel_size: int) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.kernel_size = kernel_size
        self.weight = nn.Parameter(torch.zeros(hidden_size, 1, kernel_size))
        # identity init: last tap = 1. The released SANA-WM checkpoint has no
        # ShortConvolution bias, so keep this module bias-free.
        with torch.no_grad():
            self.weight[:, 0, -1] = 1.0

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, None]:
        # x: (B, T, C) -> (B, C, T) for conv1d
        x_bct = x.transpose(1, 2)
        x_pad = F.pad(x_bct, (self.kernel_size - 1, 0))
        y = F.conv1d(x_pad, self.weight, bias=None, groups=self.hidden_size)
        return y.transpose(1, 2), None


def _bidirectional_short_conv(
    x: torch.Tensor,  # (B*S, T, C)
    conv: _ShortConvolution,
) -> torch.Tensor:
    """Forward + backward causal pass minus the shared center tap (symmetric/non-causal filter)."""
    y_fwd, _ = conv(x)
    y_bwd, _ = conv(x.flip(1))
    y_bwd = y_bwd.flip(1)
    w_center = conv.weight[:, 0, -1]  # (C,)
    center = x * w_center.view(1, 1, -1)
    return (y_fwd + y_bwd - center).to(x.dtype)


def _temporal_short_conv_cached(
    x: torch.Tensor,  # (B, N=T*S, C)
    conv: "_ShortConvolution",
    HW: Tuple[int, int, int],
    *,
    prefix: Optional[torch.Tensor] = None,
    save_prefix: bool = True,
    bidirectional: bool = True,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Chunk-causal short conv on K for streaming `forward_long` (cache slot 4).

    The FORWARD (causal) pass prepends the previous chunk's last ``kernel-1``
    frames so it stays continuous across chunks; the BACKWARD pass (when
    ``bidirectional``) is recomputed intra-chunk. The last ``kernel-1`` input
    frames are returned as the next chunk's ``prefix``. A single chunk with no
    prefix reduces to ``_bidirectional_short_conv``.
    """
    B, N, C = x.shape
    T, H, W = HW
    S = H * W
    pad = conv.kernel_size - 1
    xt = x.view(B, T, S, C).permute(0, 2, 1, 3).contiguous().reshape(B * S, T, C)

    if prefix is not None:
        prefix = prefix.to(device=xt.device, dtype=xt.dtype)
        y_fwd, _ = conv(torch.cat([prefix, xt], dim=1))
        y_fwd = y_fwd[:, -T:]
    else:
        y_fwd, _ = conv(xt)

    if bidirectional:
        y_bwd, _ = conv(xt.flip(1))
        y_bwd = y_bwd.flip(1)
        center = xt * conv.weight[:, 0, -1].view(1, 1, -1)
        y = (y_fwd + y_bwd - center).to(xt.dtype)
    else:
        y = y_fwd.to(xt.dtype)

    new_prefix = xt[:, -pad:].detach().clone() if (save_prefix and pad > 0) else prefix
    y = y.reshape(B, S, T, C).permute(0, 2, 1, 3).reshape(B, N, C)
    return y, new_prefix


# ---------------------------------------------------------------------------
# 3D RoPE
# ---------------------------------------------------------------------------


class WanRotaryPosEmbed(nn.Module):
    """3D rotary position embeddings split across (t, h, w) head dims.

    Returns complex ``freqs`` of shape ``(1, 1, T*H*W, D/2)``.
    """

    def __init__(
        self,
        attention_head_dim: int,
        patch_size: Tuple[int, int, int],
        max_seq_len: int = 1024,
        theta: float = 10000.0,
    ) -> None:
        super().__init__()
        self.attention_head_dim = attention_head_dim
        self.patch_size = patch_size
        self.max_seq_len = max_seq_len
        self.theta = theta
        self._init_freqs_buffer()

    def _init_freqs_buffer(self) -> None:
        # This is a persistent=False buffer, so it is not in the checkpoint and
        # stays on meta after FSDP weight load; post_load_weights re-runs this
        # to rematerialize it.
        h_dim = w_dim = 2 * (self.attention_head_dim // 6)
        t_dim = self.attention_head_dim - h_dim - w_dim

        freqs = []
        for dim in [t_dim, h_dim, w_dim]:
            freq = get_1d_rotary_pos_embed(
                dim,
                self.max_seq_len,
                self.theta,
                use_real=False,
                repeat_interleave_real=False,
                freqs_dtype=torch.float64,
            )
            freqs.append(freq)
        self.register_buffer("_freqs", torch.cat(freqs, dim=1), persistent=False)

    def forward(
        self,
        fhw,
        device: torch.device,
        frame_index: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # `fhw[0]` is either an int frame count (dense; positions 0..T-1) or a
        # `(start, end)` tuple selecting GLOBAL frame positions for a streaming
        # chunk. `frame_index`, when given, supplies per-token global frame
        # positions directly and overrides `fhw[0]`, so chunk-by-chunk RoPE
        # stays at absolute positions.
        fspec, pph, ppw = fhw
        freqs = self._freqs.to(device)
        d = self.attention_head_dim
        t_size = d // 2 - 2 * (d // 6)
        h_size = d // 6
        w_size = d // 6
        ft, fh, fw = freqs.split_with_sizes([t_size, h_size, w_size], dim=1)

        if frame_index is not None:
            f_pos = frame_index.to(device=device, dtype=torch.long)
        elif isinstance(fspec, (tuple, list)):
            f_pos = torch.arange(
                int(fspec[0]), int(fspec[1]), device=device, dtype=torch.long
            )
        else:
            f_pos = torch.arange(int(fspec), device=device, dtype=torch.long)
        ppf = int(f_pos.shape[0])

        freqs_t = ft[f_pos].view(ppf, 1, 1, -1).expand(ppf, pph, ppw, -1)
        freqs_h = fh[:pph].view(1, pph, 1, -1).expand(ppf, pph, ppw, -1)
        freqs_w = fw[:ppw].view(1, 1, ppw, -1).expand(ppf, pph, ppw, -1)
        out = torch.cat([freqs_t, freqs_h, freqs_w], dim=-1)
        return out.reshape(1, 1, ppf * pph * ppw, -1)


def _slice_rope_to_current_chunk(
    rotary_emb: Optional[torch.Tensor], current_tokens: int
) -> Optional[torch.Tensor]:
    """Defensive no-op: keep the trailing ``current_tokens`` of a RoPE table.

    ``forward_long`` builds ``freqs`` windowed to exactly the chunk, so this is
    a no-op there; it only trims if a caller hands in a wider table."""
    if rotary_emb is None or rotary_emb.shape[-2] == current_tokens:
        return rotary_emb
    return rotary_emb[..., -current_tokens:, :]


def _apply_rotary_emb_dn(
    hidden_states: torch.Tensor, freqs: torch.Tensor
) -> torch.Tensor:
    """Apply complex RoPE to a tensor of shape ``(B, H, D, N)`` (GDN layout).

    ``freqs`` is complex with shape ``(1, 1, N, D/2)``.
    """
    # (B, H, D, N) -> (B, H, N, D)
    x = hidden_states.permute(0, 1, 3, 2).to(torch.float64).contiguous()
    x_c = torch.view_as_complex(x.unflatten(-1, (-1, 2)))
    y = torch.view_as_real(x_c * freqs).flatten(-2, -1)
    return y.permute(0, 1, 3, 2).type_as(hidden_states)


def _apply_rotary_emb_bhnd(
    hidden_states: torch.Tensor, freqs: torch.Tensor
) -> torch.Tensor:
    """Apply complex RoPE to ``(B, H, N, D)`` (softmax attention layout)."""
    x = hidden_states.to(torch.float64).contiguous()
    x_c = torch.view_as_complex(x.unflatten(-1, (-1, 2)))
    y = torch.view_as_real(x_c * freqs).flatten(-2, -1)
    return y.type_as(hidden_states)


# ---------------------------------------------------------------------------
# UCPE block-diagonal apply primitives
# ---------------------------------------------------------------------------


def _apply_ray_projmat(feats: torch.Tensor, matrix: torch.Tensor) -> torch.Tensor:
    """Per-token 4x4 projmat applied to channels grouped by 4.

    feats: (B, H, N, D), matrix: (B, N, 4, 4).
    """
    B, Hh, N, D = feats.shape
    return torch.einsum(
        "bnij,bhnkj->bhnki",
        matrix,
        feats.reshape(B, Hh, N, -1, 4),
    ).reshape(feats.shape)


def _apply_complex_rope(
    hidden_states: torch.Tensor, freqs: torch.Tensor, inverse: bool = False
) -> torch.Tensor:
    if inverse:
        freqs = freqs.conj()
    x_real = hidden_states.to(torch.float64)
    if x_real.stride(-1) != 1:
        x_real = x_real.contiguous()
    x_c = torch.view_as_complex(x_real.unflatten(-1, (-1, 2)))
    return torch.view_as_real(x_c * freqs).flatten(-2, -1).type_as(hidden_states)


def _apply_block_diagonal(
    feats: torch.Tensor,
    func_size_pairs: List[Tuple[Callable[[torch.Tensor], torch.Tensor], int]],
) -> torch.Tensor:
    funcs, block_sizes = zip(*func_size_pairs)
    assert feats.shape[-1] == sum(block_sizes), (feats.shape, block_sizes)
    x_blocks = torch.split(feats, list(block_sizes), dim=-1)
    return torch.cat([f(b) for f, b in zip(funcs, x_blocks)], dim=-1)


def _sana_wm_chunk_index_from_chunk_size(
    T: int,
    chunk_size: int,
    strategy: str = "uniform",
) -> list[int]:
    """Return temporal chunk start indices."""
    if chunk_size <= 0:
        raise ValueError(f"chunk_size must be > 0, got {chunk_size}.")
    if T <= 0:
        raise ValueError(f"T must be > 0, got {T}.")

    strategy = "uniform" if strategy is None else str(strategy).lower()

    if strategy in ("uniform", "default"):
        indices = list(range(0, T, chunk_size))
        if len(indices) > 1 and (T - indices[-1]) < chunk_size:
            indices.pop()
        return indices

    if strategy in ("first_frame", "first_frame_alone", "first_frame_only"):
        if T <= 1:
            return [0]
        indices = [0] + list(range(1, T, chunk_size))
        if len(indices) > 2 and (T - indices[-1]) < chunk_size:
            indices.pop()
        return indices

    if strategy in ("first_plus_one", "first_chunk_plus_one"):
        if T <= chunk_size + 1:
            return [0]
        indices = [0] + list(range(chunk_size + 1, T, chunk_size))
        if len(indices) > 1 and (T - indices[-1]) < chunk_size:
            indices.pop()
        return indices

    raise ValueError(
        f"Unknown chunk_split_strategy '{strategy}'. Supported: "
        "uniform, first_frame, first_plus_one."
    )


def _sana_wm_normalize_chunk_index(
    chunk_index: Optional[List[int]],
    T: int,
    chunk_size: Optional[int] = None,
    chunk_split_strategy: str = "uniform",
) -> list[int]:
    if chunk_index is not None:
        normalized = [int(idx) for idx in chunk_index]
        if not normalized or normalized[0] != 0:
            normalized = [0] + [idx for idx in normalized if idx > 0]
        normalized = [idx for idx in normalized if idx < T]
        if not normalized:
            normalized = [0]
    else:
        if chunk_size is None:
            raise ValueError("Either chunk_index or chunk_size must be provided.")
        normalized = _sana_wm_chunk_index_from_chunk_size(
            T,
            int(chunk_size),
            strategy=chunk_split_strategy,
        )

    if normalized[-1] != T:
        normalized.append(T)
    if any(end <= start for start, end in zip(normalized[:-1], normalized[1:])):
        raise ValueError(f"chunk_index must be strictly increasing, got {normalized}.")
    return normalized


def _sana_wm_chunk_boundaries_for_attention(
    HW: Tuple[int, int, int],
    chunk_size: Optional[int],
    chunk_split_strategy: str,
    chunk_index: Optional[List[int]],
) -> Optional[list[int]]:
    T, _, _ = HW
    if chunk_index is None and (chunk_size is None or int(chunk_size) >= T):
        return None

    boundaries = _sana_wm_normalize_chunk_index(
        chunk_index,
        T,
        chunk_size=chunk_size,
        chunk_split_strategy=chunk_split_strategy,
    )
    if boundaries == [0, T]:
        return None
    return boundaries


def _sana_wm_sdpa(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    softmax_scale: float,
) -> torch.Tensor:
    q_sdpa = q.transpose(1, 2)
    k_sdpa = k.transpose(1, 2)
    v_sdpa = v.transpose(1, 2)

    dtype_orig = q_sdpa.dtype
    head_dim = q_sdpa.shape[-1]
    need_pad = head_dim not in (32, 64, 128, 256) and head_dim < 256
    if need_pad:
        pad_to = 128 if head_dim <= 128 else 256
        pad_size = pad_to - head_dim
        q_sdpa = F.pad(q_sdpa, (0, pad_size))
        k_sdpa = F.pad(k_sdpa, (0, pad_size))
        v_sdpa = F.pad(v_sdpa, (0, pad_size))

    # CUDA SDPA cannot use flash kernels for fp32; cast to bf16 on that path
    # and cast the output back to the caller dtype.
    if q_sdpa.device.type == "cuda" and q_sdpa.dtype == torch.float32:
        q_sdpa = q_sdpa.bfloat16()
        k_sdpa = k_sdpa.bfloat16()
        v_sdpa = v_sdpa.bfloat16()

    out = F.scaled_dot_product_attention(
        q_sdpa,
        k_sdpa,
        v_sdpa,
        dropout_p=0.0,
        is_causal=False,
        scale=softmax_scale,
    )
    if need_pad:
        out = out[..., :head_dim]
    return out.transpose(1, 2).to(dtype_orig)


def _sana_wm_padded_scale(head_dim: int) -> float:
    """Softmax scale for the head-padding SDPA path: pad the head dim to 128
    (or 256) and let SDPA use the *padded*-dim default scale. For SANA-WM's
    head_dim=112 this is 1/sqrt(128), NOT 1/sqrt(112)."""
    if head_dim in (32, 64, 128, 256) or head_dim >= 256:
        return head_dim**-0.5
    pad_to = 128 if head_dim <= 128 else 256
    return pad_to**-0.5


def _sana_wm_chunked_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    HW: Tuple[int, int, int],
    chunk_size: Optional[int],
    chunk_split_strategy: str,
    chunk_index: Optional[List[int]],
    softmax_scale: float,
) -> Optional[torch.Tensor]:
    """Exact chunk-causal softmax attention without materializing an NxN mask."""
    boundaries = _sana_wm_chunk_boundaries_for_attention(
        HW,
        chunk_size,
        chunk_split_strategy,
        chunk_index,
    )
    if boundaries is None:
        return None

    _, H_sp, W_sp = HW
    tokens_per_frame = H_sp * W_sp
    out_chunks = []
    for start_frame, end_frame in zip(boundaries[:-1], boundaries[1:]):
        query_start = start_frame * tokens_per_frame
        query_end = end_frame * tokens_per_frame
        kv_end = end_frame * tokens_per_frame
        out_chunks.append(
            _sana_wm_sdpa(
                q[:, query_start:query_end],
                k[:, :kv_end],
                v[:, :kv_end],
                softmax_scale=softmax_scale,
            )
        )
    return torch.cat(out_chunks, dim=1)


def _invert_SE3(transforms: torch.Tensor) -> torch.Tensor:
    Rinv = transforms[..., :3, :3].transpose(-1, -2)
    out = torch.zeros_like(transforms)
    out[..., :3, :3] = Rinv
    out[..., :3, 3] = -torch.einsum("...ij,...j->...i", Rinv, transforms[..., :3, 3])
    out[..., 3, 3] = 1.0
    return out


def _slice_rope_for_cam(
    rotary_emb: Optional[torch.Tensor],
    head_dim: int,
    rope_dim: int,
) -> Optional[torch.Tensor]:
    """Re-slice WanRotaryPosEmbed output to a smaller rope_dim."""
    if rotary_emb is None:
        return None
    orig_t_size = head_dim // 2 - 2 * (head_dim // 6)
    orig_h_size = head_dim // 6
    new_t_size = rope_dim // 2 - 2 * (rope_dim // 6)
    new_h_size = rope_dim // 6
    new_w_size = rope_dim // 6
    t_part = rotary_emb[..., :new_t_size]
    h_part = rotary_emb[..., orig_t_size : orig_t_size + new_h_size]
    w_part = rotary_emb[
        ..., orig_t_size + orig_h_size : orig_t_size + orig_h_size + new_w_size
    ]
    return torch.cat([t_part, h_part, w_part], dim=-1)


def _build_ucpe_apply_fns(
    head_dim: int,
    raymats: torch.Tensor,  # (B, N, 4, 4) -- ray<-world
    rotary_emb: Optional[torch.Tensor],
) -> Tuple[Callable, Callable, Callable]:
    """Build the (apply_q, apply_kv, apply_o) callables used in the camera
    branch.  Splits the head_dim in two: half for 4x4 projmat tiling, half
    for complex-RoPE rotation.
    """
    P = raymats
    P_T = P.transpose(-1, -2)
    P_inv = _invert_SE3(P)

    rotary_emb_cam = _slice_rope_for_cam(rotary_emb, head_dim, head_dim // 2)
    if rotary_emb_cam is not None:

        def rope_fn(x: torch.Tensor) -> torch.Tensor:
            return _apply_complex_rope(x, rotary_emb_cam, inverse=False)

        def rope_fn_inv(x: torch.Tensor) -> torch.Tensor:
            return _apply_complex_rope(x, rotary_emb_cam, inverse=True)

    else:

        def rope_fn(x: torch.Tensor) -> torch.Tensor:
            return x

        def rope_fn_inv(x: torch.Tensor) -> torch.Tensor:
            return x

    half = head_dim // 2

    def ray_proj_t(y: torch.Tensor) -> torch.Tensor:
        return _apply_ray_projmat(y, P_T)

    def ray_proj_inv(y: torch.Tensor) -> torch.Tensor:
        return _apply_ray_projmat(y, P_inv)

    def ray_proj(y: torch.Tensor) -> torch.Tensor:
        return _apply_ray_projmat(y, P)

    def apply_q(x: torch.Tensor) -> torch.Tensor:
        return _apply_block_diagonal(x, [(ray_proj_t, half), (rope_fn, half)])

    def apply_kv(x: torch.Tensor) -> torch.Tensor:
        return _apply_block_diagonal(x, [(ray_proj_inv, half), (rope_fn, half)])

    def apply_o(x: torch.Tensor) -> torch.Tensor:
        return _apply_block_diagonal(x, [(ray_proj, half), (rope_fn_inv, half)])

    return apply_q, apply_kv, apply_o


def _compute_fov_from_focal(focal: torch.Tensor, image_size: int) -> torch.Tensor:
    """fov = 2 * atan(image_size / (2 * focal))"""
    return 2.0 * torch.atan(image_size / (2.0 * focal.clamp(min=1e-6)))


def _unproject_grid(
    x_fov: torch.Tensor,
    y_fov: torch.Tensor,
    H: int,
    W: int,
    cx: torch.Tensor,
    cy: torch.Tensor,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Compute camera-space unit ray directions for each latent token.

    Returns (B, F, H, W, 3).
    """
    B, F_dim = x_fov.shape
    # Upstream `create_grid` uses integer pixel coordinates [0, W-1] /
    # [0, H-1] rather than half-pixel centers.
    u = torch.arange(W, device=device, dtype=dtype)
    v = torch.arange(H, device=device, dtype=dtype)
    u = u.view(1, 1, 1, W).expand(B, F_dim, H, W)
    v = v.view(1, 1, H, 1).expand(B, F_dim, H, W)
    cx_e = cx.view(B, F_dim, 1, 1)
    cy_e = cy.view(B, F_dim, 1, 1)
    tan_x = torch.tan(x_fov / 2.0).view(B, F_dim, 1, 1)
    tan_y = torch.tan(y_fov / 2.0).view(B, F_dim, 1, 1)
    # Map pixel (u, v) -> (x_dir, y_dir) on the z=1 plane.
    dx = (u - cx_e) / max(W, 1) * 2.0 * tan_x
    dy = (v - cy_e) / max(H, 1) * 2.0 * tan_y
    dz = torch.ones_like(dx)
    d = torch.stack([dx, dy, dz], dim=-1)  # (B, F, H, W, 3)
    return F.normalize(d, dim=-1)


def process_camera_conditions_ucpe(
    camera_conditions: torch.Tensor,  # (B, F, 20)
    HW: Tuple[int, int, int],
    patch_size: Tuple[int, int, int] = (1, 1, 1),
) -> torch.Tensor:
    """Convert ``(B, F, 20)`` flat camera conditions into ``raymats``.

    Layout: first 16 = c2w 4x4 flatten, last 4 = ``(fx, fy, cx, cy)``.
    Returns ``raymats`` of shape ``(B, F, H, W, 4, 4)`` (ray<-world).
    """
    B, F_dim, _ = camera_conditions.shape
    _, H, W = HW
    device = camera_conditions.device
    dtype = camera_conditions.dtype

    c2w_flat = camera_conditions[..., :16]
    C_to_W = c2w_flat.view(B, F_dim, 4, 4)

    fx = camera_conditions[..., 16]
    fy = camera_conditions[..., 17]
    cx = camera_conditions[..., 18]
    cy = camera_conditions[..., 19]

    image_h = H * patch_size[1]
    image_w = W * patch_size[2]
    x_fov = _compute_fov_from_focal(fx, image_w)
    y_fov = _compute_fov_from_focal(fy, image_h)

    # cx/cy are in pixel units -- scale to latent grid like upstream
    # (``cx / patch_size[2]``).
    cx_lat = cx / float(patch_size[2])
    cy_lat = cy / float(patch_size[1])

    d_cam = _unproject_grid(
        x_fov, y_fov, H, W, cx_lat, cy_lat, device, dtype
    )  # (B,F,H,W,3)

    # Build per-token "ray<-world" 4x4 following upstream `world_to_ray_mats`.
    R_c2w = C_to_W[..., :3, :3]  # (B, F, 3, 3)
    t_c2w = C_to_W[..., :3, 3]  # (B, F, 3)
    d_world = torch.einsum("bfij,bfhwj->bfhwi", R_c2w, d_cam)
    z_ray = F.normalize(d_world, dim=-1, eps=1e-6)
    cam_y = R_c2w[..., :, 1].view(B, F_dim, 1, 1, 3).expand(B, F_dim, H, W, 3)
    x_ray = F.normalize(torch.cross(cam_y, z_ray, dim=-1), dim=-1, eps=1e-6)
    y_ray = F.normalize(torch.cross(z_ray, x_ray, dim=-1), dim=-1, eps=1e-6)
    R_ray_to_world = torch.stack([x_ray, y_ray, z_ray], dim=-1)

    # P = ray<-world = inverse of [R_ray_to_world | t_c2w]
    R_w_to_ray = R_ray_to_world.transpose(-1, -2)
    t_w_to_ray = -torch.einsum(
        "bfhwij,bfj->bfhwi",
        R_w_to_ray,
        t_c2w,
    )
    raymats = torch.zeros(B, F_dim, H, W, 4, 4, device=device, dtype=dtype)
    raymats[..., :3, :3] = R_w_to_ray
    raymats[..., :3, 3] = t_w_to_ray
    raymats[..., 3, 3] = 1.0
    invalid = torch.isnan(d_world).any(dim=-1)
    if bool(invalid.any()):
        eye = torch.eye(4, device=device, dtype=dtype)
        raymats[invalid] = eye
    return raymats


def compute_chunk_plucker(
    camera_conditions: torch.Tensor,  # (B, F_orig, 20)
    HW: Tuple[int, int, int],  # latent (T, H, W)
    vae_temporal_stride: int = 8,
    patch_size: Tuple[int, int, int] = (1, 1, 1),
) -> torch.Tensor:
    """Compute the 48-channel packed Plücker raymap consumed by
    ``plucker_embedder``.

    Official SANA-WM centers each chunk on the latent-frame timestamp:
    ``0, stride, 2*stride, ...``. For timestamp 0 the chunk is clamped to the
    first ``stride`` frames; for later timestamps it uses the preceding
    ``stride - 1`` frames plus the current frame. Short tail chunks are padded
    by repeating the final available frame.

    Each latent frame packs ``vae_temporal_stride`` original-frame Plücker
    coords ``[d, o x d]`` (6D each) into 48 channels. Output shape is
    ``(B, 48, T, H, W)`` for direct consumption by Conv3d.
    """
    B, F_orig, _ = camera_conditions.shape
    T, H, W = HW
    device = camera_conditions.device
    dtype = camera_conditions.dtype

    c2w = camera_conditions[..., :16].view(B, F_orig, 4, 4)
    fx = camera_conditions[..., 16]
    fy = camera_conditions[..., 17]
    cx = camera_conditions[..., 18]
    cy = camera_conditions[..., 19]

    image_h = H * patch_size[1]
    image_w = W * patch_size[2]
    x_fov = _compute_fov_from_focal(fx, image_w)
    y_fov = _compute_fov_from_focal(fy, image_h)

    d_cam = _unproject_grid(
        x_fov,
        y_fov,
        H,
        W,
        cx / float(patch_size[2]),
        cy / float(patch_size[1]),
        device,
        dtype,
    )  # (B, F_orig, H, W, 3)

    R = c2w[..., :3, :3]
    o = c2w[..., :3, 3]  # (B, F_orig, 3)
    d_world = F.normalize(torch.einsum("bfij,bfhwj->bfhwi", R, d_cam), dim=-1)
    o_exp = o.view(B, F_orig, 1, 1, 3).expand_as(d_world)
    moment = torch.cross(o_exp, d_world, dim=-1)
    plucker = torch.cat([d_world, moment], dim=-1)  # (B, F_orig, H, W, 6)

    time_indices = torch.arange(
        0, F_orig, vae_temporal_stride, device=device, dtype=torch.long
    )
    if time_indices.numel() < T:
        pad = T - int(time_indices.numel())
        last = (
            time_indices[-1:]
            if time_indices.numel()
            else torch.zeros(1, device=device, dtype=torch.long)
        )
        time_indices = torch.cat([time_indices, last.repeat(pad)], dim=0)
    time_indices = time_indices[:T]

    chunks = []
    for time_index in time_indices.tolist():
        start = max(0, int(time_index) - vae_temporal_stride + 1)
        end = min(start + vae_temporal_stride, F_orig)
        chunk = plucker[:, start:end]
        if chunk.shape[1] < vae_temporal_stride:
            pad = vae_temporal_stride - chunk.shape[1]
            pad_chunk = chunk[:, -1:].repeat(1, pad, 1, 1, 1)
            chunk = torch.cat([chunk, pad_chunk], dim=1)
        chunks.append(chunk)

    plucker = torch.stack(chunks, dim=1)  # (B, T, stride, H, W, 6)
    plucker = plucker.permute(0, 1, 3, 4, 2, 5).reshape(
        B, T, H, W, vae_temporal_stride * 6
    )
    # (B, 48, T, H, W) for Conv3d
    plucker = plucker.permute(0, 4, 1, 2, 3).contiguous()
    return plucker


# ---------------------------------------------------------------------------
# Patch embedding / caption embedder / final layer -- upstream key names
# ---------------------------------------------------------------------------


class PatchEmbedMS3D(nn.Module):
    """3D patch embedder used by SANA-WM for ``x_embedder``,
    ``raymap_embedder``, and ``plucker_embedder``.

    Parameter names: ``proj.weight`` / ``proj.bias``.
    """

    def __init__(
        self,
        patch_size: Tuple[int, int, int],
        in_chans: int,
        embed_dim: int,
        kernel_size: Optional[Tuple[int, int, int]] = None,
        bias: bool = True,
    ) -> None:
        super().__init__()
        kernel_size = kernel_size or patch_size
        assert patch_size[0] == 1, "Temporal patch must be 1 for SANA-WM."
        self.patch_size = patch_size
        self.proj = nn.Conv3d(
            in_chans, embed_dim, kernel_size=kernel_size, stride=patch_size, bias=bias
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)  # (B, D, T, H, W)
        return x.flatten(2).transpose(1, 2)  # (B, T*H*W, D)


class _UpstreamMlp(nn.Module):
    """timm-style Mlp used by ``y_embedder.y_proj`` (fc1/fc2 with bias=True)."""

    def __init__(
        self, in_features: int, hidden_features: int, out_features: int
    ) -> None:
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features, bias=True)
        self.act = nn.GELU(approximate="tanh")
        self.fc2 = nn.Linear(hidden_features, out_features, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act(self.fc1(x)))


class CaptionEmbedder(nn.Module):
    """Upstream ``CaptionEmbedder``: projects text embeddings to hidden size, plus a learned null-caption table for CFG.

    Inference-only forward: ``y`` may be ``(B, 1, L, in_channels)`` or
    ``(B, L, in_channels)``; the null table is not applied (no inference-time
    dropout). Returns ``(B, 1, L, hidden_size)``.
    """

    def __init__(
        self, in_channels: int, hidden_size: int, token_num: int = 300
    ) -> None:
        super().__init__()
        self.y_proj = _UpstreamMlp(in_channels, hidden_size, hidden_size)
        # buffer in upstream -- registered with nn.Parameter wrapper but as buffer.
        self.register_buffer(
            "y_embedding",
            torch.randn(token_num, in_channels) / in_channels**0.5,
            persistent=True,
        )

    def forward(self, y: torch.Tensor) -> torch.Tensor:
        if y.dim() == 3:
            y = y.unsqueeze(1)
        return self.y_proj(y)


class T2IFinalLayer(nn.Module):
    """Output AdaLN + Linear, with the scale_shift_table living *inside* the
    module (upstream stores it as ``final_layer.scale_shift_table``).
    """

    def __init__(
        self, hidden_size: int, patch_size: Tuple[int, int, int], out_channels: int
    ) -> None:
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(
            hidden_size, math.prod(patch_size) * out_channels, bias=True
        )
        self.scale_shift_table = nn.Parameter(
            torch.randn(2, hidden_size) / hidden_size**0.5
        )
        self.out_channels = out_channels

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        # t can be either (B, D) for scalar timesteps or (B, 1, T, D) for
        # SANA-WM's first-frame-conditioned flow_euler_ltx sampler.
        if t.dim() == 2:
            shift, scale = (self.scale_shift_table[None] + t[:, None]).chunk(2, dim=1)
            x = self.norm_final(x) * (1 + scale) + shift
        else:
            B, N, D = x.shape
            t = t.reshape(B, -1, D)
            num_frames = t.shape[1]
            tokens_per_frame = N // num_frames
            shift, scale = (
                self.scale_shift_table[None, None, :, :] + t[:, :, None, :]
            ).chunk(2, dim=2)
            x = self.norm_final(x).reshape(B, num_frames, tokens_per_frame, D)
            x = x * (1 + scale) + shift
            x = x.reshape(B, N, D)
        return self.linear(x)


# ---------------------------------------------------------------------------
# Timestep embedder (upstream sana_blocks.TimestepEmbedder, key names
# ``t_embedder.mlp.0/2.{weight,bias}``)
# ---------------------------------------------------------------------------


def _sinusoidal_timestep_embedding(
    t: torch.Tensor, dim: int, max_period: float = 10000.0
) -> torch.Tensor:
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period)
        * torch.arange(half, dtype=torch.float32, device=t.device)
        / half
    )
    args = t.float()[:, None] * freqs[None]
    emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
    return emb


class TimestepEmbedder(nn.Module):
    """Upstream ``TimestepEmbedder``.  Stored as ``mlp.0/2`` (Linear, SiLU,
    Linear) -- we mirror that exact layout."""

    def __init__(self, hidden_size: int, frequency_embedding_size: int = 256) -> None:
        super().__init__()
        self.frequency_embedding_size = frequency_embedding_size
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        t_freq = _sinusoidal_timestep_embedding(t, self.frequency_embedding_size)
        # first linear's weight dtype matches upstream
        return self.mlp(t_freq.to(self.mlp[0].weight.dtype))


# ---------------------------------------------------------------------------
# GLUMBConvTemp -- upstream basic_modules.GLUMBConvTemp.
#
# Stored sub-modules (all referenced by their checkpoint key prefix):
#   * inverted_conv.conv      Conv2d(in, hidden*2, 1)
#   * depth_conv.conv         Conv2d(hidden*2, hidden*2, 3, groups=hidden*2)
#   * point_conv.conv         Conv2d(hidden, out, 1, bias=False)
#   * t_conv                  Conv2d(out, out, kernel=(t_k, 1), padding=(t_pad, 0), bias=False)
#
# Upstream wraps each Conv2d in a ``ConvLayer`` that owns the conv as
# ``.conv``; that's why the key has the extra ``.conv`` segment.
# ---------------------------------------------------------------------------

_INT32_SAFE_CONV_ELEMENTS = 1 << 30


class _ConvLayer(nn.Module):
    """Thin wrapper -- has a single ``conv`` member to match upstream
    ``inverted_conv.conv`` / ``depth_conv.conv`` / ``point_conv.conv`` keys.
    """

    def __init__(self, conv: nn.Module, act: Optional[nn.Module] = None) -> None:
        super().__init__()
        self.conv = conv
        self.act = act

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        if self.act is not None:
            x = self.act(x)
        return x


class GLUMBConvTemp(nn.Module):
    """Spatial GLU MBConv + additive temporal Conv2d (zero-init).

    Operates on ``(B, T*H*W, C)``; reshape to ``(B*T, C, H, W)`` for spatial
    convs, then ``(B, C, T, H*W)`` for ``t_conv`` (a 2D depth-wise temporal
    conv), residually added.
    """

    def __init__(
        self, in_features: int, hidden_features: int, t_kernel_size: int = 3
    ) -> None:
        super().__init__()
        self.inverted_conv = _ConvLayer(
            nn.Conv2d(in_features, hidden_features * 2, 1, 1, 0, bias=True),
            act=nn.SiLU(inplace=False),
        )
        self.depth_conv = _ConvLayer(
            nn.Conv2d(
                hidden_features * 2,
                hidden_features * 2,
                3,
                1,
                1,
                groups=hidden_features * 2,
                bias=True,
            ),
            act=None,
        )
        self.point_conv = _ConvLayer(
            nn.Conv2d(hidden_features, in_features, 1, 1, 0, bias=False),
            act=None,
        )
        self.glu_act = nn.SiLU(inplace=False)

        t_padding = t_kernel_size // 2
        self.t_conv = nn.Conv2d(
            in_features,
            in_features,
            kernel_size=(t_kernel_size, 1),
            stride=1,
            padding=(t_padding, 0),
            bias=False,
        )
        nn.init.zeros_(self.t_conv.weight)

    def _apply_spatial(self, x: torch.Tensor) -> torch.Tensor:
        x = self.inverted_conv(x)
        x = self.depth_conv(x)
        a, g = x.chunk(2, dim=1)
        return self.point_conv(a * self.glu_act(g))

    def _apply_spatial_autochunked(self, x: torch.Tensor) -> torch.Tensor:
        """Avoid oversized Conv2d calls on long videos while keeping short path fused."""
        BT, _, H, W = x.shape
        elements_per_bt = self.inverted_conv.conv.out_channels * H * W
        max_bt = max(1, _INT32_SAFE_CONV_ELEMENTS // elements_per_bt)
        if BT <= max_bt:
            return self._apply_spatial(x)
        return torch.cat(
            [
                self._apply_spatial(x[start : start + max_bt])
                for start in range(0, BT, max_bt)
            ],
            dim=0,
        )

    def forward(
        self,
        x: torch.Tensor,
        HW: Tuple[int, int, int],
        *,
        ffn_tail: Optional[torch.Tensor] = None,
        save_ffn_tail: bool = False,
    ):
        B, N, C = x.shape
        T, H, W = HW
        assert N == T * H * W, f"GLUMBConvTemp: N={N} != T*H*W={T * H * W}"

        # Spatial path is frame-local, so identical chunked or whole.
        x_sp = x.reshape(B * T, H, W, C).permute(0, 3, 1, 2).contiguous()
        x_sp = self._apply_spatial_autochunked(x_sp)  # (B*T, C, H, W)

        x_t = (
            x_sp.view(B, T, C, H * W).permute(0, 2, 1, 3).contiguous()
        )  # (B, C, T, S=H*W)

        if ffn_tail is None and not save_ffn_tail:
            # Dense (bidirectional / symmetric-padding) path.
            x_out = x_t + self.t_conv(x_t)
            return x_out.permute(0, 2, 3, 1).reshape(B, N, C)

        # Streaming causal path (cache slot 9): prepend the previous chunk's last
        # `pad` frames as real left context, then drop them from the output. The
        # right edge still sees the conv's own zero-pad, so a chunk matches the
        # whole-sequence pass for every frame whose right context lies in-chunk
        # (always true for the final chunk). Mirrors the reference GLUMBConvTemp.
        pad = self.t_conv.kernel_size[0] // 2
        conv_in = x_t
        padded_size = 0
        if ffn_tail is not None:
            prefix = ffn_tail.to(device=x_t.device, dtype=x_t.dtype)
            conv_in = torch.cat([prefix[:, :, -pad:], x_t], dim=2)
            padded_size = conv_in.shape[2] - x_t.shape[2]
        new_tail = (
            x_t[:, :, -pad:].detach().clone()
            if (save_ffn_tail and pad > 0)
            else ffn_tail
        )
        tconv_out = self.t_conv(conv_in)[:, :, padded_size:]
        x_out = x_t + tconv_out
        return x_out.permute(0, 2, 3, 1).reshape(B, N, C), new_tail


# ---------------------------------------------------------------------------
# Frame-gate and DeltaNet update rule. The recurrent and chunk-parallel forms
# are numerically equivalent.
# ---------------------------------------------------------------------------


def _compute_frame_gates(
    x: torch.Tensor,  # (B, N, C)
    HW: Tuple[int, int, int],
    heads: int,
    beta_proj: nn.Linear,
    gate_proj: nn.Linear,
    dt_bias: torch.Tensor,
    A_log: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Frame-level beta/decay gates.

    Returns:
        beta:  (B, H, T, S)   in (0, 1) via sigmoid
        decay: (B, H, T)      in (0, 1) via exp(-A * softplus(.))
    """
    B, N, C = x.shape
    T, H, W = HW
    S = H * W
    beta = beta_proj(x).sigmoid().reshape(B, T, S, heads).permute(0, 3, 1, 2)
    x_frame = x.reshape(B, T, S, C).mean(dim=2)
    a_out = gate_proj(x_frame).float()
    dt = dt_bias.float().view(1, 1, -1)
    A_val = A_log.float().exp().view(1, 1, -1)
    decay = (-A_val * F.softplus(a_out + dt)).exp().transpose(1, 2)  # (B, H, T)
    return beta, decay


def _gdn_scan_forward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    q_rot: torch.Tensor,
    k_rot: torch.Tensor,
    beta: torch.Tensor,
    decay: torch.Tensor,
    eps: float = 1e-6,
    return_components: bool = False,
) -> torch.Tensor:
    """Causal recurrent GDN scan over T. Tensors are in (B, H, D, N=T*S) layout."""
    B, H, D, N = q.shape
    T = beta.shape[2]
    S = N // T

    def fold(x):
        return x.view(B, H, D, T, S).permute(0, 1, 3, 2, 4)  # (B, H, T, D, S)

    q, k, v = fold(q), fold(k), fold(v)
    q_rot, k_rot = fold(q_rot), fold(k_rot)
    if beta.ndim == 4:
        beta_e = beta.unsqueeze(3)  # (B, H, T, 1, S)
    else:
        beta_e = beta.view(B, H, T, 1, 1)
    decay_e = decay.view(B, H, T, 1, 1)

    state_kv = torch.zeros(B, H, D, D, device=q.device, dtype=q.dtype)
    state_z = torch.zeros(B, H, D, 1, device=q.device, dtype=q.dtype)
    num_list, den_list = [], []
    target_z = 1.0
    for t in range(T):
        qt, kt, vt = q[:, :, t], k[:, :, t], v[:, :, t]
        qrt, krt = q_rot[:, :, t], k_rot[:, :, t]
        bt, gt = beta_e[:, :, t], decay_e[:, :, t]
        state_kv = state_kv * gt
        state_z = state_z * gt
        v_pred = torch.matmul(state_kv, krt)
        delta_v = (vt - v_pred) * bt
        state_kv = state_kv + torch.matmul(delta_v, krt.transpose(-1, -2))
        z_pred = torch.matmul(state_z.transpose(-1, -2), kt)
        delta_z = (target_z - z_pred) * bt
        state_z = state_z + torch.matmul(kt, delta_z.transpose(-1, -2))
        num_list.append(torch.matmul(state_kv, qrt))  # (B, H, D, S)
        den_list.append(torch.matmul(state_z.transpose(-1, -2), qt))  # (B, H, 1, S)

    num_stacked = torch.stack(num_list, dim=2)  # (B, H, T, D, S)
    den_stacked = torch.stack(den_list, dim=2)  # (B, H, T, 1, S)

    def restore(tensor, d_out):
        return tensor.permute(0, 1, 3, 2, 4).reshape(B, H, d_out, N)

    num = restore(num_stacked, D)
    den = restore(den_stacked, 1)
    if return_components:
        return num, den
    return num / (den + eps)


# ---------------------------------------------------------------------------
# Streaming (autoregressive) state-carrying scans for the `forward_long` path:
# a forward-only recurrence that SEEDS its state from a prior chunk and RETURNS
# the final state, so a long video can be generated chunk-by-chunk while
# remaining numerically identical to the monolithic forward scan.
# ---------------------------------------------------------------------------


def _gdn_scan_forward_stateful(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    q_rot: torch.Tensor,
    k_rot: torch.Tensor,
    beta: torch.Tensor,
    decay: torch.Tensor,
    *,
    init_state_kv: Optional[torch.Tensor] = None,
    init_state_z: Optional[torch.Tensor] = None,
    eps: float = 1e-6,
    return_components: bool = False,
    return_state: bool = False,
):
    """Main-branch GDN causal scan that carries KV/Z state across chunks.

    The state is seeded from
    ``init_state_kv``/``init_state_z`` (None → zeros, i.e. the first chunk) and,
    when ``return_state`` is set, the final ``(state_kv, state_z)`` is returned so
    the next chunk continues the recurrence. Tensors are ``(B, H, D, N=T*S)``.
    """
    B, H, D, N = q.shape
    T = beta.shape[2]
    S = N // T

    def fold(x):
        return x.view(B, H, D, T, S).permute(0, 1, 3, 2, 4)

    q, k, v = fold(q), fold(k), fold(v)
    q_rot, k_rot = fold(q_rot), fold(k_rot)
    beta_e = beta.unsqueeze(3) if beta.ndim == 4 else beta.view(B, H, T, 1, 1)
    decay_e = decay.view(B, H, T, 1, 1)

    state_kv = (
        torch.zeros(B, H, D, D, device=q.device, dtype=q.dtype)
        if init_state_kv is None
        else init_state_kv.to(device=q.device, dtype=q.dtype).clone()
    )
    state_z = (
        torch.zeros(B, H, D, 1, device=q.device, dtype=q.dtype)
        if init_state_z is None
        else init_state_z.to(device=q.device, dtype=q.dtype).clone()
    )
    num_list, den_list = [], []
    target_z = 1.0
    for t in range(T):
        qt, kt, vt = q[:, :, t], k[:, :, t], v[:, :, t]
        qrt, krt = q_rot[:, :, t], k_rot[:, :, t]
        bt, gt = beta_e[:, :, t], decay_e[:, :, t]
        state_kv = state_kv * gt
        state_z = state_z * gt
        delta_v = (vt - torch.matmul(state_kv, krt)) * bt
        state_kv = state_kv + torch.matmul(delta_v, krt.transpose(-1, -2))
        delta_z = (target_z - torch.matmul(state_z.transpose(-1, -2), kt)) * bt
        state_z = state_z + torch.matmul(kt, delta_z.transpose(-1, -2))
        num_list.append(torch.matmul(state_kv, qrt))
        den_list.append(torch.matmul(state_z.transpose(-1, -2), qt))

    num = torch.stack(num_list, dim=2).permute(0, 1, 3, 2, 4).reshape(B, H, D, N)
    den = torch.stack(den_list, dim=2).permute(0, 1, 3, 2, 4).reshape(B, H, 1, N)
    out = (num, den) if return_components else num / (den + eps)
    if return_state:
        return out, (state_kv, state_z)
    return out


def _single_path_delta_scan_forward_stateful(
    q_rot: torch.Tensor,
    k_rot: torch.Tensor,
    v: torch.Tensor,
    beta: torch.Tensor,
    decay: torch.Tensor,
    *,
    init_state_kv: Optional[torch.Tensor] = None,
    return_state: bool = False,
):
    """Camera-branch (numerator-only) delta-rule scan that carries state across
    chunks via a seedable / returnable ``state_kv`` for chunked autoregressive
    generation.
    """
    B, H, D, N = q_rot.shape
    T = beta.shape[2]
    S = N // T

    def fold(x):
        return x.view(B, H, D, T, S).permute(0, 1, 3, 2, 4)

    q_rot, k_rot, v = fold(q_rot), fold(k_rot), fold(v)
    beta_e = beta.unsqueeze(3) if beta.ndim == 4 else beta.view(B, H, T, 1, 1)
    decay_e = decay.view(B, H, T, 1, 1)

    state_kv = (
        torch.zeros(B, H, D, D, device=q_rot.device, dtype=q_rot.dtype)
        if init_state_kv is None
        else init_state_kv.to(device=q_rot.device, dtype=q_rot.dtype).clone()
    )
    out_list = []
    for t in range(T):
        qrt, krt, vt = q_rot[:, :, t], k_rot[:, :, t], v[:, :, t]
        bt, gt = beta_e[:, :, t], decay_e[:, :, t]
        state_kv = state_kv * gt
        delta_v = (vt - torch.matmul(state_kv, krt)) * bt
        state_kv = state_kv + torch.matmul(delta_v, krt.transpose(-1, -2))
        out_list.append(torch.matmul(state_kv, qrt))

    out = torch.stack(out_list, dim=2).permute(0, 1, 3, 2, 4).reshape(B, H, D, N)
    if return_state:
        return out, state_kv
    return out


def _gdn_chunk_scan_forward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    q_rot: torch.Tensor,
    k_rot: torch.Tensor,
    beta: torch.Tensor,
    decay: torch.Tensor,
    chunk_size: Optional[int] = 21,
    eps: float = 1e-6,
    return_components: bool = False,
) -> torch.Tensor:
    """Chunk-scan form of SANA GDN.

    Computes W/U per chunk instead of materializing all temporal transitions at
    once, keeping peak memory closer to the recurrent path on long videos.
    """
    B, H, D, N = q.shape
    T = beta.shape[2]
    S = N // T

    def fold(x: torch.Tensor) -> torch.Tensor:
        return x.view(B, H, D, T, S).permute(0, 1, 3, 2, 4)

    q, k, v = fold(q), fold(k), fold(v)
    q_rot, k_rot = fold(q_rot), fold(k_rot)
    if beta.ndim == 4:
        beta_e = beta.unsqueeze(3)
    else:
        beta_e = beta.view(B, H, T, 1, 1)
    decay_e = decay.view(B, H, T, 1, 1)

    if chunk_size is None or int(chunk_size) <= 0:
        boundaries = [0, T]
    else:
        boundaries = _sana_wm_normalize_chunk_index(
            None,
            T,
            chunk_size=int(chunk_size),
            chunk_split_strategy="uniform",
        )

    eye = torch.eye(D, device=q.device, dtype=q.dtype).view(1, 1, 1, D, D)
    state_kv = torch.zeros(B, H, D, D, device=q.device, dtype=q.dtype)
    state_z = torch.zeros(B, H, D, 1, device=q.device, dtype=q.dtype)
    num_chunks, den_chunks = [], []

    for start, end in zip(boundaries[:-1], boundaries[1:]):
        q_c = q[:, :, start:end]
        k_c = k[:, :, start:end]
        v_c = v[:, :, start:end]
        q_rot_c = q_rot[:, :, start:end]
        k_rot_c = k_rot[:, :, start:end]
        beta_c = beta_e[:, :, start:end]
        decay_c = decay_e[:, :, start:end]

        k_rot_beta = k_rot_c * beta_c
        w_kv = decay_c * (eye - torch.matmul(k_rot_beta, k_rot_c.transpose(-1, -2)))
        u_kv = torch.matmul(v_c * beta_c, k_rot_c.transpose(-1, -2))

        k_beta = k_c * beta_c
        w_z = decay_c * (eye - torch.matmul(k_beta, k_c.transpose(-1, -2)))
        u_z = k_beta.sum(dim=-1, keepdim=True)

        state_kv_frames, state_z_frames = [], []
        for offset in range(end - start):
            state_kv = torch.matmul(state_kv, w_kv[:, :, offset]) + u_kv[:, :, offset]
            state_z = torch.matmul(w_z[:, :, offset], state_z) + u_z[:, :, offset]
            state_kv_frames.append(state_kv)
            state_z_frames.append(state_z)

        state_kv_all = torch.stack(state_kv_frames, dim=2)
        state_z_all = torch.stack(state_z_frames, dim=2)
        num_chunks.append(torch.matmul(state_kv_all, q_rot_c))
        den_chunks.append(torch.matmul(state_z_all.transpose(-1, -2), q_c))

    num = torch.cat(num_chunks, dim=2)
    den = torch.cat(den_chunks, dim=2)

    def restore(tensor: torch.Tensor, d_out: int) -> torch.Tensor:
        return tensor.permute(0, 1, 3, 2, 4).reshape(B, H, d_out, N)

    num = restore(num, D)
    den = restore(den, 1)
    if return_components:
        return num, den
    return num / (den + eps)


def _flip_and_shift(x: torch.Tensor, dim: int, shift_val: float = 0.0) -> torch.Tensor:
    """Flip along ``dim`` then shift by one with ``shift_val`` filling the head.
    Used for the backward pass of bidirectional GDN."""
    x_flipped = x.flip(dim)
    idx = [slice(None)] * x.ndim
    idx[dim] = slice(0, 1)
    head = torch.full_like(x_flipped[tuple(idx)], shift_val)
    idx[dim] = slice(0, -1)
    return torch.cat([head, x_flipped[tuple(idx)]], dim=dim)


def _gdn_scan_bidirectional(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    q_rot: torch.Tensor,
    k_rot: torch.Tensor,
    beta: torch.Tensor,
    decay: torch.Tensor,
    HW: Tuple[int, int, int],
    chunk_size: Optional[int] = None,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Bidirectional GDN: forward (inclusive) + backward (exclusive) scan, summed in numerator/denominator space."""

    def run_scan(
        q_in: torch.Tensor,
        k_in: torch.Tensor,
        v_in: torch.Tensor,
        q_rot_in: torch.Tensor,
        k_rot_in: torch.Tensor,
        beta_in: torch.Tensor,
        decay_in: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if chunk_size is None:
            return _gdn_scan_forward(
                q_in,
                k_in,
                v_in,
                q_rot_in,
                k_rot_in,
                beta_in,
                decay_in,
                eps=eps,
                return_components=True,
            )
        return _gdn_chunk_scan_forward(
            q_in,
            k_in,
            v_in,
            q_rot_in,
            k_rot_in,
            beta_in,
            decay_in,
            chunk_size=chunk_size,
            eps=eps,
            return_components=True,
        )

    num_fwd, den_fwd = run_scan(q, k, v, q_rot, k_rot, beta, decay)

    # Backward pass: flip Q, flip+shift K/V/k_rot/beta and shift decay-by-1.
    B, H, D, N = q.shape
    T, H_sp, W_sp = HW
    S = H_sp * W_sp

    def to_time(x, d):
        return x.view(B, H, d, T, S).permute(0, 1, 3, 2, 4)

    def from_time(x, d):
        return x.permute(0, 1, 3, 2, 4).reshape(B, H, d, N)

    q_t = to_time(q, D)
    k_t = to_time(k, D)
    v_t = to_time(v, D)
    q_rot_t = to_time(q_rot, D)
    k_rot_t = to_time(k_rot, D)

    q_bwd = torch.flip(q_t, dims=[2])
    q_rot_bwd = torch.flip(q_rot_t, dims=[2])
    k_bwd = _flip_and_shift(k_t, dim=2, shift_val=0.0)
    v_bwd = _flip_and_shift(v_t, dim=2, shift_val=0.0)
    k_rot_bwd = _flip_and_shift(k_rot_t, dim=2, shift_val=0.0)
    beta_bwd = _flip_and_shift(beta, dim=2, shift_val=0.0)
    decay_bwd = _flip_and_shift(decay, dim=2, shift_val=1.0)

    q_bwd_f = from_time(q_bwd, D)
    k_bwd_f = from_time(k_bwd, D)
    v_bwd_f = from_time(v_bwd, D)
    q_rot_bwd_f = from_time(q_rot_bwd, D)
    k_rot_bwd_f = from_time(k_rot_bwd, D)

    num_bwd_flipped, den_bwd_flipped = run_scan(
        q_bwd_f,
        k_bwd_f,
        v_bwd_f,
        q_rot_bwd_f,
        k_rot_bwd_f,
        beta_bwd,
        decay_bwd,
    )

    def flip_back(tensor):
        d_actual = tensor.shape[2]
        t_struct = tensor.view(B, H, d_actual, T, S)
        return torch.flip(t_struct, dims=[3]).reshape(B, H, d_actual, N)

    num_bwd = flip_back(num_bwd_flipped)
    den_bwd = flip_back(den_bwd_flipped)
    return (num_fwd + num_bwd) / (den_fwd + den_bwd + eps)


def _single_path_delta_scan_forward(
    q_rot: torch.Tensor,
    k_rot: torch.Tensor,
    v: torch.Tensor,
    beta: torch.Tensor,
    decay: torch.Tensor,
) -> torch.Tensor:
    """Numerator-only camera delta-rule recurrence.

    This is intentionally separate from ``_gdn_scan_forward``: the SANA-WM
    camera branch does not use the GDN denominator path; reusing the
    main-branch GDN recurrence here changes the latent distribution
    substantially once camera conditioning is enabled.
    """
    B, H, D, N = q_rot.shape
    T = beta.shape[2]
    S = N // T

    def fold(x: torch.Tensor) -> torch.Tensor:
        return x.view(B, H, D, T, S).permute(0, 1, 3, 2, 4)

    q_rot = fold(q_rot)
    k_rot = fold(k_rot)
    v = fold(v)

    if beta.ndim == 4:
        beta_e = beta.unsqueeze(3)
    else:
        beta_e = beta.view(B, H, T, 1, 1)
    decay_e = decay.view(B, H, T, 1, 1)

    state_kv = torch.zeros(B, H, D, D, device=q_rot.device, dtype=q_rot.dtype)
    out_list = []
    for t in range(T):
        qrt = q_rot[:, :, t]
        krt = k_rot[:, :, t]
        vt = v[:, :, t]
        bt = beta_e[:, :, t]
        gt = decay_e[:, :, t]

        state_kv = state_kv * gt
        v_pred = torch.matmul(state_kv, krt)
        delta_v = (vt - v_pred) * bt
        state_kv = state_kv + torch.matmul(delta_v, krt.transpose(-1, -2))
        out_list.append(torch.matmul(state_kv, qrt))

    out = torch.stack(out_list, dim=2)
    return out.permute(0, 1, 3, 2, 4).reshape(B, H, D, N)


def _single_path_delta_chunk_scan_forward(
    q_rot: torch.Tensor,
    k_rot: torch.Tensor,
    v: torch.Tensor,
    beta: torch.Tensor,
    decay: torch.Tensor,
    chunk_size: Optional[int] = 21,
) -> torch.Tensor:
    """Chunk-scan form of the camera single-path delta rule."""
    B, H, D, N = q_rot.shape
    T = beta.shape[2]
    S = N // T

    def fold(x: torch.Tensor) -> torch.Tensor:
        return x.view(B, H, D, T, S).permute(0, 1, 3, 2, 4)

    q_rot = fold(q_rot)
    k_rot = fold(k_rot)
    v = fold(v)
    if beta.ndim == 4:
        beta_e = beta.unsqueeze(3)
    else:
        beta_e = beta.view(B, H, T, 1, 1)
    decay_e = decay.view(B, H, T, 1, 1)

    if chunk_size is None or int(chunk_size) <= 0:
        boundaries = [0, T]
    else:
        boundaries = _sana_wm_normalize_chunk_index(
            None,
            T,
            chunk_size=int(chunk_size),
            chunk_split_strategy="uniform",
        )

    eye = torch.eye(D, device=q_rot.device, dtype=q_rot.dtype).view(1, 1, 1, D, D)
    state_kv = torch.zeros(B, H, D, D, device=q_rot.device, dtype=q_rot.dtype)
    out_chunks = []

    for start, end in zip(boundaries[:-1], boundaries[1:]):
        q_rot_c = q_rot[:, :, start:end]
        k_rot_c = k_rot[:, :, start:end]
        v_c = v[:, :, start:end]
        beta_c = beta_e[:, :, start:end]
        decay_c = decay_e[:, :, start:end]

        k_rot_beta = k_rot_c * beta_c
        w_kv = decay_c * (eye - torch.matmul(k_rot_beta, k_rot_c.transpose(-1, -2)))
        u_kv = torch.matmul(v_c * beta_c, k_rot_c.transpose(-1, -2))

        state_frames = []
        for offset in range(end - start):
            state_kv = torch.matmul(state_kv, w_kv[:, :, offset]) + u_kv[:, :, offset]
            state_frames.append(state_kv)

        state_all = torch.stack(state_frames, dim=2)
        out_chunks.append(torch.matmul(state_all, q_rot_c))

    out = torch.cat(out_chunks, dim=2)
    return out.permute(0, 1, 3, 2, 4).reshape(B, H, D, N)


def _single_path_delta_scan_bidirectional(
    q_rot: torch.Tensor,
    k_rot: torch.Tensor,
    v: torch.Tensor,
    beta: torch.Tensor,
    decay: torch.Tensor,
    HW: Tuple[int, int, int],
    chunk_size: Optional[int] = None,
) -> torch.Tensor:
    """Bidirectional single-path camera scan: inclusive forward + exclusive backward (``flip_and_shift`` on K/V, beta, decay)."""
    scan_forward = (
        _single_path_delta_chunk_scan_forward
        if chunk_size is not None
        else _single_path_delta_scan_forward
    )
    if chunk_size is None:
        out_fwd = scan_forward(q_rot, k_rot, v, beta, decay)
    else:
        out_fwd = scan_forward(q_rot, k_rot, v, beta, decay, chunk_size=chunk_size)

    B, H, D, N = q_rot.shape
    T, H_sp, W_sp = HW
    S = H_sp * W_sp

    def to_time(x: torch.Tensor) -> torch.Tensor:
        return x.view(B, H, D, T, S).permute(0, 1, 3, 2, 4)

    def from_time(x: torch.Tensor) -> torch.Tensor:
        return x.permute(0, 1, 3, 2, 4).reshape(B, H, D, N)

    q_rot_t = to_time(q_rot)
    k_rot_t = to_time(k_rot)
    v_t = to_time(v)

    q_rot_bwd = torch.flip(q_rot_t, dims=[2])
    k_rot_bwd = _flip_and_shift(k_rot_t, dim=2, shift_val=0.0)
    v_bwd = _flip_and_shift(v_t, dim=2, shift_val=0.0)
    beta_bwd = _flip_and_shift(beta, dim=2, shift_val=0.0)
    decay_bwd = _flip_and_shift(decay, dim=2, shift_val=1.0)

    if chunk_size is None:
        out_bwd_flipped = scan_forward(
            from_time(q_rot_bwd),
            from_time(k_rot_bwd),
            from_time(v_bwd),
            beta_bwd,
            decay_bwd,
        )
    else:
        out_bwd_flipped = scan_forward(
            from_time(q_rot_bwd),
            from_time(k_rot_bwd),
            from_time(v_bwd),
            beta_bwd,
            decay_bwd,
            chunk_size=chunk_size,
        )
    out_bwd = torch.flip(
        out_bwd_flipped.view(B, H, D, T, S),
        dims=[3],
    ).reshape(B, H, D, N)
    return out_fwd + out_bwd


# ---------------------------------------------------------------------------
# Chunk-causal cached scans for the streaming `forward_long` path: the FORWARD
# (inclusive) pass carries recurrent state across chunks; the BACKWARD
# (exclusive) pass is recomputed intra-chunk and stateless. A single chunk with
# no carried state reduces exactly to the bidirectional scans, while a sequence
# processed chunk-by-chunk stays continuous in the forward direction.
# ---------------------------------------------------------------------------


def _gdn_scan_cached(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    q_rot: torch.Tensor,
    k_rot: torch.Tensor,
    beta: torch.Tensor,
    decay: torch.Tensor,
    *,
    init_state_kv: Optional[torch.Tensor] = None,
    init_state_z: Optional[torch.Tensor] = None,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
    """Chunk-causal main-branch GDN scan for streaming `forward_long`.

    The forward pass seeds/returns ``(state_kv, state_z)`` so chunks stay
    continuous; the backward pass is intra-chunk and stateless. Returns
    ``(out, (state_kv, state_z))``.
    """
    (num_fwd, den_fwd), (state_kv, state_z) = _gdn_scan_forward_stateful(
        q,
        k,
        v,
        q_rot,
        k_rot,
        beta,
        decay,
        init_state_kv=init_state_kv,
        init_state_z=init_state_z,
        eps=eps,
        return_components=True,
        return_state=True,
    )

    B, H, D, N = q.shape
    T = beta.shape[2]
    S = N // T

    def to_time(x, d):
        return x.view(B, H, d, T, S).permute(0, 1, 3, 2, 4)

    def from_time(x, d):
        return x.permute(0, 1, 3, 2, 4).reshape(B, H, d, N)

    q_bwd = from_time(torch.flip(to_time(q, D), dims=[2]), D)
    q_rot_bwd = from_time(torch.flip(to_time(q_rot, D), dims=[2]), D)
    k_bwd = from_time(_flip_and_shift(to_time(k, D), dim=2, shift_val=0.0), D)
    v_bwd = from_time(_flip_and_shift(to_time(v, D), dim=2, shift_val=0.0), D)
    k_rot_bwd = from_time(_flip_and_shift(to_time(k_rot, D), dim=2, shift_val=0.0), D)
    beta_bwd = _flip_and_shift(beta, dim=2, shift_val=0.0)
    decay_bwd = _flip_and_shift(decay, dim=2, shift_val=1.0)

    num_bwd_flipped, den_bwd_flipped = _gdn_scan_forward(
        q_bwd,
        k_bwd,
        v_bwd,
        q_rot_bwd,
        k_rot_bwd,
        beta_bwd,
        decay_bwd,
        eps=eps,
        return_components=True,
    )

    def flip_back(tensor, d):
        return torch.flip(tensor.view(B, H, d, T, S), dims=[3]).reshape(B, H, d, N)

    num_bwd = flip_back(num_bwd_flipped, D)
    den_bwd = flip_back(den_bwd_flipped, 1)
    out = (num_fwd + num_bwd) / (den_fwd + den_bwd + eps)
    return out, (state_kv, state_z)


def _single_path_delta_scan_cached(
    q_rot: torch.Tensor,
    k_rot: torch.Tensor,
    v: torch.Tensor,
    beta: torch.Tensor,
    decay: torch.Tensor,
    *,
    init_state_kv: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Chunk-causal camera single-path delta scan for streaming `forward_long`.

    The forward pass carries ``state_kv`` across chunks; the backward pass is
    intra-chunk/stateless. Returns ``(out, state_kv)``.
    """
    out_fwd, state_kv = _single_path_delta_scan_forward_stateful(
        q_rot,
        k_rot,
        v,
        beta,
        decay,
        init_state_kv=init_state_kv,
        return_state=True,
    )

    B, H, D, N = q_rot.shape
    T = beta.shape[2]
    S = N // T

    def to_time(x):
        return x.view(B, H, D, T, S).permute(0, 1, 3, 2, 4)

    def from_time(x):
        return x.permute(0, 1, 3, 2, 4).reshape(B, H, D, N)

    q_rot_bwd = from_time(torch.flip(to_time(q_rot), dims=[2]))
    k_rot_bwd = from_time(_flip_and_shift(to_time(k_rot), dim=2, shift_val=0.0))
    v_bwd = from_time(_flip_and_shift(to_time(v), dim=2, shift_val=0.0))
    beta_bwd = _flip_and_shift(beta, dim=2, shift_val=0.0)
    decay_bwd = _flip_and_shift(decay, dim=2, shift_val=1.0)

    out_bwd_flipped = _single_path_delta_scan_forward(
        q_rot_bwd,
        k_rot_bwd,
        v_bwd,
        beta_bwd,
        decay_bwd,
    )
    out_bwd = torch.flip(out_bwd_flipped.view(B, H, D, T, S), dims=[3]).reshape(
        B, H, D, N
    )
    return out_fwd + out_bwd, state_kv


def _downscale_to_reference_rms(
    ref: torch.Tensor,
    transformed: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Clamp UCPE-transformed channel RMS to the pre-transform envelope.

    The UCPE matrices include translations that can inflate transformed Q/K/V
    magnitudes; downscale per (batch, head, token) before the camera recurrence
    so inference stays on the training distribution.
    """
    ref_rms = ref.square().mean(dim=2, keepdim=True).add(eps).sqrt()
    transformed_rms = transformed.square().mean(dim=2, keepdim=True).add(eps).sqrt()
    scale = (ref_rms / transformed_rms.clamp_min(eps)).clamp(max=1.0)
    return transformed * scale


# ---------------------------------------------------------------------------
# SANA-WM attention block: main GDN (or softmax) + UCPE camera branch. The cam
# branch shares ``proj`` / ``output_gate`` with the main branch.
# ---------------------------------------------------------------------------


class BidirectionalGDNUCPESinglePathLiteLA(nn.Module):
    """Bidirectional GDN main branch + UCPE camera branch (single-path
    output: ``main + out_proj_cam(cam_raw)`` then shared output gate +
    shared output projection).
    """

    def __init__(
        self,
        in_dim: int,
        heads: int,
        head_dim: int,
        qk_norm: bool = True,
        conv_kernel_size: int = 4,
        k_conv_only: bool = True,
        eps: float = 1e-8,
        softmax_main: bool = False,
        update_rule: str = "torch_chunk",
        cam_update_rule: str = "torch_chunk",
        chunk_gdn_chunk_size: int = 21,
        use_chunked_softmax_attention: bool = False,
        gdn_backend: str = "auto",
    ) -> None:
        super().__init__()
        out_dim = heads * head_dim
        assert (
            out_dim == in_dim
        ), f"in_dim ({in_dim}) must equal heads*head_dim ({out_dim})"
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.heads = heads
        self.dim = head_dim
        self.eps = eps
        self.softmax_main = softmax_main
        self.update_rule = update_rule
        self.cam_update_rule = cam_update_rule
        self.chunk_gdn_chunk_size = chunk_gdn_chunk_size
        self.use_chunked_softmax_attention = use_chunked_softmax_attention
        self.gdn_backend = gdn_backend
        if self.update_rule not in ("torch_chunk", "torch_recurrent"):
            raise ValueError(f"Unsupported SANA-WM update_rule: {self.update_rule}")
        if self.cam_update_rule not in ("torch_chunk", "torch_recurrent"):
            raise ValueError(
                f"Unsupported SANA-WM cam_update_rule: {self.cam_update_rule}"
            )
        if self.gdn_backend not in ("auto", "torch", "triton"):
            raise ValueError(
                "Unsupported SANA-WM gdn_backend: "
                f"{self.gdn_backend}. Expected one of auto, torch, triton."
            )

        # Fused QKV + output proj (proj shared with cam branch).
        self.qkv = nn.Linear(in_dim, 3 * out_dim, bias=False)
        self.proj = nn.Linear(out_dim, out_dim, bias=True)

        if qk_norm:
            self.q_norm = _RMSNorm(in_dim, eps=1e-5)
            self.k_norm = _RMSNorm(in_dim, eps=1e-5)
            self.q_norm_cam = _RMSNorm(in_dim, eps=1e-5)
            self.k_norm_cam = _RMSNorm(in_dim, eps=1e-5)
        else:
            self.q_norm = nn.Identity()
            self.k_norm = nn.Identity()
            self.q_norm_cam = nn.Identity()
            self.k_norm_cam = nn.Identity()

        # Also held by the softmax variant for state_dict compat.
        self.beta_proj = nn.Linear(in_dim, heads, bias=True)
        self.gate_proj = nn.Linear(in_dim, heads, bias=True)
        self.A_log = nn.Parameter(torch.log(torch.empty(heads).uniform_(0, 16)))
        self.dt_bias = nn.Parameter(torch.full((heads,), -5.0))
        self.register_buffer("recall_gate", torch.zeros(1))
        self.output_gate = nn.Linear(in_dim, out_dim, bias=True)

        # Short convs on K (k_conv_only=True). The upstream softmax variant
        # ``BidirectionalSoftmaxUCPESinglePathLiteLA`` does NOT carry these
        # short convs in either the main or the cam branch, so the released
        # checkpoint has no conv_k/conv_k_cam tensors for softmax blocks
        # (indices {3, 7, 11, 15, 19} when softmax_every_n=4). Creating them
        # here would leave the loader staring at missing checkpoint keys it
        # cannot synthesize. Skip creation when softmax_main=True.
        if conv_kernel_size > 0 and not softmax_main:
            self.conv_k = _ShortConvolution(out_dim, conv_kernel_size)
            if not k_conv_only:
                self.conv_q = _ShortConvolution(out_dim, conv_kernel_size)
                self.conv_v = _ShortConvolution(out_dim, conv_kernel_size)
            else:
                self.conv_q = None
                self.conv_v = None
        else:
            self.conv_k = self.conv_q = self.conv_v = None
        self.conv_kernel_size = conv_kernel_size
        self.k_conv_only = k_conv_only

        # Camera branch: separate QKV + zero-init output proj + separate K
        # ShortConvolution. Both branches share ``proj``.
        self.q_proj_cam = nn.Linear(in_dim, out_dim, bias=True)
        self.k_proj_cam = nn.Linear(in_dim, out_dim, bias=True)
        self.v_proj_cam = nn.Linear(in_dim, out_dim, bias=True)
        self.out_proj_cam = nn.Linear(out_dim, out_dim, bias=True)
        nn.init.zeros_(self.out_proj_cam.weight)
        nn.init.zeros_(self.out_proj_cam.bias)
        if conv_kernel_size > 0 and not softmax_main:
            self.conv_k_cam = _ShortConvolution(out_dim, conv_kernel_size)
            if not k_conv_only:
                self.conv_q_cam = _ShortConvolution(out_dim, conv_kernel_size)
                self.conv_v_cam = _ShortConvolution(out_dim, conv_kernel_size)
            else:
                self.conv_q_cam = None
                self.conv_v_cam = None
        else:
            self.conv_k_cam = self.conv_q_cam = self.conv_v_cam = None

        # Softmax-variant blocks route attention through SGLang's pluggable
        # backend (FA3 / FlashInfer / Triton / SDPA). GDN blocks compute
        # attention via the scan path and don't go through this.
        if softmax_main:
            self.softmax_attn = LocalAttention(
                num_heads=heads,
                head_size=head_dim,
            )
        self._triton_rope_tables_cache: Optional[
            Tuple[Tuple, Tuple[torch.Tensor, torch.Tensor]]
        ] = None
        self._triton_norm_weights_cache: Optional[
            Tuple[Tuple, Tuple[torch.Tensor, torch.Tensor]]
        ] = None
        self._cam_qkv_params_cache: Optional[
            Tuple[Tuple, Tuple[torch.Tensor, torch.Tensor]]
        ] = None

    # ------------------------------------------------------------------ #

    @staticmethod
    def _temporal_short_conv(
        x: torch.Tensor,  # (B, N, C)
        conv: _ShortConvolution,
        HW: Tuple[int, int, int],
        bidirectional: bool = True,
    ) -> torch.Tensor:
        B, N, C = x.shape
        T, H, W = HW
        S = H * W
        # (B, T, S, C) -> (B*S, T, C): T onto the time axis for the conv.
        y = x.view(B, T, S, C).permute(0, 2, 1, 3).contiguous().reshape(B * S, T, C)
        if bidirectional:
            y = _bidirectional_short_conv(y, conv)
        else:
            y, _ = conv(y)
        # back to (B, T*S, C)
        return y.reshape(B, S, T, C).permute(0, 2, 1, 3).reshape(B, N, C)

    # ------------------------------------------------------------------ #

    def _get_cam_qkv_params(self) -> Tuple[torch.Tensor, torch.Tensor]:
        weights = (
            self.q_proj_cam.weight,
            self.k_proj_cam.weight,
            self.v_proj_cam.weight,
        )
        biases = (
            self.q_proj_cam.bias,
            self.k_proj_cam.bias,
            self.v_proj_cam.bias,
        )
        if torch.is_grad_enabled():
            return torch.cat(weights, dim=0), torch.cat(biases, dim=0)

        key = (
            "cam_qkv_params",
            tuple(_tensor_cache_key(weight) for weight in weights),
            tuple(_tensor_cache_key(bias) for bias in biases),
        )
        cached = self._cam_qkv_params_cache
        if cached is not None and cached[0] == key:
            return cached[1]

        params = (
            torch.cat(weights, dim=0).contiguous(),
            torch.cat(biases, dim=0).contiguous(),
        )
        self._cam_qkv_params_cache = (key, params)
        return params

    def _cam_qkv(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        qkv_weight, qkv_bias = self._get_cam_qkv_params()
        return F.linear(x, qkv_weight, qkv_bias).chunk(3, dim=-1)

    # ------------------------------------------------------------------ #

    def _triton_gdn_unavailable_reason(
        self,
        qkv: torch.Tensor,
        beta: torch.Tensor,
        decay: torch.Tensor,
        HW: Tuple[int, int, int],
    ) -> Optional[str]:
        if self.gdn_backend == "torch":
            return "gdn_backend=torch"
        if _SANA_WM_TRITON_GDN_DISABLED_REASON is not None:
            return _SANA_WM_TRITON_GDN_DISABLED_REASON
        if self.training or torch.is_grad_enabled():
            return "requires eval/inference mode"
        if not qkv.is_cuda:
            return "requires CUDA tensor"
        if qkv.dtype not in (torch.float16, torch.bfloat16):
            return f"requires fp16/bf16 qkv, got {qkv.dtype}"
        if not qkv.is_contiguous():
            return "qkv must be contiguous"
        if beta.ndim != 4:
            return f"requires beta shape (B, H, T, S), got {tuple(beta.shape)}"

        B, N, three, heads, head_dim = qkv.shape
        T, H_sp, W_sp = HW
        S = H_sp * W_sp
        if three != 3:
            return f"requires qkv third dim=3, got {three}"
        if N != T * S:
            return f"requires N=T*S, got N={N}, T*S={T * S}"
        if head_dim > 128:
            return f"requires head_dim <= 128, got {head_dim}"
        if beta.shape != (B, heads, T, S):
            return f"requires beta shape {(B, heads, T, S)}, got {tuple(beta.shape)}"
        if decay.shape != (B, heads, T):
            return f"requires decay shape {(B, heads, T)}, got {tuple(decay.shape)}"
        if not hasattr(self.q_norm, "weight") or not hasattr(self.k_norm, "weight"):
            return "requires learned q/k RMSNorm weights"
        return None

    def _get_triton_norm_weights(self) -> Tuple[torch.Tensor, torch.Tensor]:
        q_weight = self.q_norm.weight
        k_weight = self.k_norm.weight
        key = (
            "triton_norm_weights",
            _tensor_cache_key(q_weight),
            _tensor_cache_key(k_weight),
        )
        cached = self._triton_norm_weights_cache
        if cached is not None and cached[0] == key:
            return cached[1]

        weights = (
            q_weight.float().contiguous(),
            k_weight.float().contiguous(),
        )
        self._triton_norm_weights_cache = (key, weights)
        return weights

    def _get_triton_rope_tables(
        self,
        prepare_rope_tables: Callable,
        rotary_emb: Optional[torch.Tensor],
        *,
        N: int,
        head_dim: int,
        device: torch.device,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        key = (
            "triton_rope_tables",
            N,
            head_dim,
            str(device),
            None if rotary_emb is None else _tensor_cache_key(rotary_emb),
        )
        cached = self._triton_rope_tables_cache
        if cached is not None and cached[0] == key:
            return cached[1]

        tables = prepare_rope_tables(rotary_emb, N, head_dim, device)
        self._triton_rope_tables_cache = (key, tables)
        return tables

    def _maybe_main_branch_triton_gdn(
        self,
        qkv: torch.Tensor,
        beta: torch.Tensor,
        decay: torch.Tensor,
        HW: Tuple[int, int, int],
        rotary_emb: Optional[torch.Tensor],
    ) -> Optional[torch.Tensor]:
        global _SANA_WM_TRITON_GDN_DISABLED_REASON

        reason = self._triton_gdn_unavailable_reason(qkv, beta, decay, HW)
        if reason is not None:
            if self.gdn_backend == "triton":
                raise RuntimeError(f"SANA-WM Triton GDN backend unavailable: {reason}")
            return None

        try:
            from sglang.kernels.ops.diffusion.triton.sana_wm_gdn import (
                fused_bigdn_func,
                fused_qk_inv_rms,
                prepare_rope_tables,
            )

            B, N, _, heads, head_dim = qkv.shape
            T, H_sp, W_sp = HW
            S = H_sp * W_sp
            q_norm_weight, k_norm_weight = self._get_triton_norm_weights()
            norm_eps = float(getattr(self.q_norm, "eps", 1e-5))
            q_inv_rms, k_inv_rms = fused_qk_inv_rms(qkv, eps=norm_eps)
            rope_cos, rope_sin = self._get_triton_rope_tables(
                prepare_rope_tables,
                rotary_emb,
                N=N,
                head_dim=head_dim,
                device=qkv.device,
            )
            out = fused_bigdn_func(
                qkv,
                q_inv_rms,
                k_inv_rms,
                q_norm_weight=q_norm_weight,
                k_norm_weight=k_norm_weight,
                rope_cos=rope_cos,
                rope_sin=rope_sin,
                beta=beta.contiguous(),
                decay=decay.contiguous(),
                F=T,
                S=S,
                k_scale=(head_dim**-0.5) * (S**-0.5),
                eps=self.eps,
            )
            return out.reshape(B, N, heads * head_dim)
        except Exception as exc:
            if self.gdn_backend == "triton":
                raise
            _SANA_WM_TRITON_GDN_DISABLED_REASON = str(exc)
            _log_sana_wm_triton_gdn_fallback(str(exc))
            return None

    # ------------------------------------------------------------------ #

    def _triton_cam_gdn_unavailable_reason(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        beta: torch.Tensor,
        decay: torch.Tensor,
        HW: Tuple[int, int, int],
    ) -> Optional[str]:
        if self.gdn_backend == "torch":
            return "gdn_backend=torch"
        if _SANA_WM_TRITON_CAM_GDN_DISABLED_REASON is not None:
            return _SANA_WM_TRITON_CAM_GDN_DISABLED_REASON
        if self.training or torch.is_grad_enabled():
            return "requires eval/inference mode"
        if not q.is_cuda:
            return "requires CUDA tensor"
        if (
            q.dtype != torch.float32
            or k.dtype != torch.float32
            or v.dtype != torch.float32
        ):
            return f"requires fp32 q/k/v, got {q.dtype}/{k.dtype}/{v.dtype}"
        if not q.is_contiguous() or not k.is_contiguous() or not v.is_contiguous():
            return "q/k/v must be contiguous"
        if beta.ndim not in (3, 4):
            return f"requires beta rank 3 or 4, got {tuple(beta.shape)}"

        B, heads, head_dim, N = q.shape
        T, H_sp, W_sp = HW
        S = H_sp * W_sp
        if k.shape != q.shape or v.shape != q.shape:
            return f"q/k/v shape mismatch: {q.shape}/{k.shape}/{v.shape}"
        if N != T * S:
            return f"requires N=T*S, got N={N}, T*S={T * S}"
        if beta.ndim == 3 and beta.shape != (B, heads, T):
            return f"requires beta shape {(B, heads, T)}, got {tuple(beta.shape)}"
        if beta.ndim == 4 and beta.shape != (B, heads, T, S):
            return (
                f"requires beta shape {(B, heads, T, S)}, " f"got {tuple(beta.shape)}"
            )
        if decay.shape != (B, heads, T):
            return f"requires decay shape {(B, heads, T)}, got {tuple(decay.shape)}"
        if head_dim > 128:
            return f"requires head_dim <= 128, got {head_dim}"
        return None

    def _maybe_cam_branch_triton_scan(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        beta: torch.Tensor,
        decay: torch.Tensor,
        HW: Tuple[int, int, int],
    ) -> Optional[torch.Tensor]:
        global _SANA_WM_TRITON_CAM_GDN_DISABLED_REASON

        precheck_reason = None
        if self.gdn_backend == "torch":
            precheck_reason = "gdn_backend=torch"
        elif _SANA_WM_TRITON_CAM_GDN_DISABLED_REASON is not None:
            precheck_reason = _SANA_WM_TRITON_CAM_GDN_DISABLED_REASON
        elif self.training or torch.is_grad_enabled():
            precheck_reason = "requires eval/inference mode"
        elif not q.is_cuda:
            precheck_reason = "requires CUDA tensor"

        if precheck_reason is not None:
            if self.gdn_backend == "triton":
                raise RuntimeError(
                    "SANA-WM Triton camera GDN backend unavailable: "
                    f"{precheck_reason}"
                )
            return None

        q = q.float().contiguous()
        k = k.float().contiguous()
        v = v.float().contiguous()
        reason = self._triton_cam_gdn_unavailable_reason(q, k, v, beta, decay, HW)
        if reason is not None:
            if self.gdn_backend == "triton":
                raise RuntimeError(
                    f"SANA-WM Triton camera GDN backend unavailable: {reason}"
                )
            return None

        try:
            from sglang.kernels.ops.diffusion.triton.sana_wm_gdn_chunkwise import (
                cam_scan_bidi_chunkwise,
            )

            B, heads, _, _ = q.shape
            T, H_sp, W_sp = HW
            S = H_sp * W_sp
            if beta.ndim == 3:
                beta_in = beta.unsqueeze(-1).expand(B, heads, T, S).contiguous()
            else:
                beta_in = beta.contiguous()
            out = cam_scan_bidi_chunkwise(
                q,
                k,
                v,
                beta_in.float(),
                decay.float().contiguous(),
            )
            return out
        except Exception as exc:
            if self.gdn_backend == "triton":
                raise
            _SANA_WM_TRITON_CAM_GDN_DISABLED_REASON = str(exc)
            _log_sana_wm_triton_cam_gdn_fallback(str(exc))
            return None

    # ------------------------------------------------------------------ #

    def _main_branch_gdn(
        self,
        x: torch.Tensor,
        HW: Tuple[int, int, int],
        rotary_emb: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return ``(out_raw, beta, decay)`` where ``out_raw`` is the GDN
        scan result before output gate / proj.
        """
        B, N, C = x.shape
        T, H_sp, W_sp = HW
        S = H_sp * W_sp

        qkv = self.qkv(x).reshape(B, N, 3, self.heads, self.dim).contiguous()
        q, k, v = qkv.unbind(2)

        if self.conv_k is not None:
            k = self._temporal_short_conv(
                k.reshape(B, N, C), self.conv_k, HW, bidirectional=True
            )
            k = k.reshape(B, N, self.heads, self.dim)
            qkv = torch.stack((q, k, v), dim=2).contiguous()

        beta, decay = _compute_frame_gates(
            x,
            HW,
            self.heads,
            self.beta_proj,
            self.gate_proj,
            self.dt_bias,
            self.A_log,
        )

        triton_out = self._maybe_main_branch_triton_gdn(
            qkv,
            beta,
            decay,
            HW,
            rotary_emb,
        )
        if triton_out is not None:
            return triton_out, beta, decay

        q, k, v = qkv.unbind(2)

        q = self.q_norm(q.reshape(B, N, C)).reshape(B, N, self.heads, self.dim)
        k = self.k_norm(k.reshape(B, N, C)).reshape(B, N, self.heads, self.dim)

        # ReLU kernel + key scale
        q = F.relu(q)
        k = F.relu(k)
        k_scale = (self.dim**-0.5) * (S**-0.5)
        k = k * k_scale

        # (B, H, D, N)
        q = q.permute(0, 2, 3, 1)
        k = k.permute(0, 2, 3, 1)
        v = v.permute(0, 2, 3, 1)

        if rotary_emb is not None:
            q_rot = _apply_rotary_emb_dn(q, rotary_emb)
            k_rot = _apply_rotary_emb_dn(k, rotary_emb)
        else:
            q_rot = q
            k_rot = k

        # fp32 scan for stability
        dtype = q.dtype
        scan_chunk_size = (
            self.chunk_gdn_chunk_size if self.update_rule == "torch_chunk" else None
        )
        out = _gdn_scan_bidirectional(
            q.float(),
            k.float(),
            v.float(),
            q_rot.float(),
            k_rot.float(),
            beta.float(),
            decay.float(),
            HW=HW,
            chunk_size=scan_chunk_size,
            eps=self.eps,
        ).to(dtype)

        out = out.permute(0, 3, 1, 2).reshape(B, N, C)
        return out, beta, decay

    def _main_branch_softmax(
        self,
        x: torch.Tensor,
        HW: Tuple[int, int, int],
        rotary_emb: Optional[torch.Tensor],
        chunk_size: Optional[int] = None,
        chunk_split_strategy: str = "uniform",
        chunk_index: Optional[List[int]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Softmax variant of the main branch, via SGLang's pluggable attention backend.

        Returns ``(out_raw, beta, decay)`` so the cam branch can reuse the shared
        gates -- like upstream ``BidirectionalSoftmaxUCPESinglePathLiteLA``.
        """
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.heads, self.dim)
        q, k, v = qkv.unbind(2)
        if self.conv_k is not None:
            k = self._temporal_short_conv(
                k.reshape(B, N, C), self.conv_k, HW, bidirectional=True
            )
            k = k.reshape(B, N, self.heads, self.dim)
        q = self.q_norm(q.reshape(B, N, C)).reshape(B, N, self.heads, self.dim)
        k = self.k_norm(k.reshape(B, N, C)).reshape(B, N, self.heads, self.dim)
        # RoPE primitives use (B, H, N, D); LocalAttention takes (B, N, H, D).
        q = q.permute(0, 2, 1, 3)  # (B, H, N, D)
        k = k.permute(0, 2, 1, 3)
        v = v.permute(0, 2, 1, 3)
        if rotary_emb is not None:
            q = _apply_rotary_emb_bhnd(q, rotary_emb)
            k = _apply_rotary_emb_bhnd(k, rotary_emb)
        q_in = q.transpose(1, 2).contiguous()
        k_in = k.transpose(1, 2).contiguous()
        v_in = v.transpose(1, 2).contiguous()
        out = None
        if self.use_chunked_softmax_attention:
            out = _sana_wm_chunked_attention(
                q_in,
                k_in,
                v_in,
                HW=HW,
                chunk_size=chunk_size,
                chunk_split_strategy=chunk_split_strategy,
                chunk_index=chunk_index,
                softmax_scale=self.softmax_attn.softmax_scale,
            )
        if out is None:
            out = self.softmax_attn(q_in, k_in, v_in)
        out = out.reshape(B, N, C)

        # Gates are needed by the cam branch and also exist in the softmax
        # variant's state dict.
        beta, decay = _compute_frame_gates(
            x,
            HW,
            self.heads,
            self.beta_proj,
            self.gate_proj,
            self.dt_bias,
            self.A_log,
        )
        return out, beta, decay

    # ------------------------------------------------------------------ #

    def _cam_branch(
        self,
        x: torch.Tensor,
        HW: Tuple[int, int, int],
        apply_q: Callable,
        apply_kv: Callable,
        apply_o: Callable,
        beta: torch.Tensor,
        decay: torch.Tensor,
    ) -> torch.Tensor:
        B, N, C = x.shape
        T, H_sp, W_sp = HW
        S = H_sp * W_sp

        q, k, v = self._cam_qkv(x)
        q = q.reshape(B, N, self.heads, self.dim)
        k = k.reshape(B, N, self.heads, self.dim)
        v = v.reshape(B, N, self.heads, self.dim)

        if self.conv_k_cam is not None:
            k = self._temporal_short_conv(
                k.reshape(B, N, C), self.conv_k_cam, HW, bidirectional=True
            )
            k = k.reshape(B, N, self.heads, self.dim)

        q = self.q_norm_cam(q.reshape(B, N, C)).reshape(B, N, self.heads, self.dim)
        k = self.k_norm_cam(k.reshape(B, N, C)).reshape(B, N, self.heads, self.dim)

        q = F.relu(q)
        k = F.relu(k)
        k_scale = (self.dim**-0.5) * (S**-0.5)
        k = k * k_scale

        # (B, H, N, D) for UCPE apply, then back to (B, H, D, N) for the scan.
        q_bhnd = q.permute(0, 2, 1, 3)
        k_bhnd = k.permute(0, 2, 1, 3)
        v_bhnd = v.permute(0, 2, 1, 3)

        q_proj = apply_q(q_bhnd)
        kv_proj = apply_kv(torch.cat([k_bhnd, v_bhnd], dim=1))
        k_proj, v_proj = torch.chunk(kv_proj, chunks=2, dim=1)

        q_pre_dn = q_bhnd.permute(0, 1, 3, 2)
        q_dn = q_proj.permute(0, 1, 3, 2)
        k_pre_dn = k_bhnd.permute(0, 1, 3, 2)
        k_dn = k_proj.permute(0, 1, 3, 2)
        v_pre_dn = v_bhnd.permute(0, 1, 3, 2)
        v_dn = v_proj.permute(0, 1, 3, 2)

        q_dn = _downscale_to_reference_rms(q_pre_dn, q_dn)
        k_dn = _downscale_to_reference_rms(k_pre_dn, k_dn)
        v_dn = _downscale_to_reference_rms(v_pre_dn, v_dn)

        pre_ucpe_k_norm = torch.linalg.vector_norm(
            k_pre_dn.float(), dim=2, keepdim=True
        ).clamp_min(1e-6)
        post_ucpe_k_norm = torch.linalg.vector_norm(
            k_dn.float(), dim=2, keepdim=True
        ).clamp_min(1e-6)
        inflation_sq = (post_ucpe_k_norm / pre_ucpe_k_norm) ** 2
        frame_inflation_sq = inflation_sq.view(B, self.heads, T, S).mean(dim=-1)
        if beta.ndim == 3:
            beta = beta / frame_inflation_sq.clamp_min(1.0)
        elif beta.ndim == 4:
            beta = beta / frame_inflation_sq.unsqueeze(-1).clamp_min(1.0)

        dtype = q_dn.dtype
        out = self._maybe_cam_branch_triton_scan(q_dn, k_dn, v_dn, beta, decay, HW)
        if out is None:
            scan_chunk_size = (
                self.chunk_gdn_chunk_size
                if self.cam_update_rule == "torch_chunk"
                else None
            )
            out = _single_path_delta_scan_bidirectional(
                q_dn.float(),
                k_dn.float(),
                v_dn.float(),
                beta.float(),
                decay.float(),
                HW=HW,
                chunk_size=scan_chunk_size,
            )
        out = out.to(dtype)
        out_bhnd = out.permute(0, 1, 3, 2)
        out_bhnd = apply_o(out_bhnd)
        return out_bhnd.permute(0, 2, 1, 3).reshape(B, N, C)

    def _cam_branch_softmax(
        self,
        x: torch.Tensor,
        HW: Tuple[int, int, int],
        apply_q: Callable,
        apply_kv: Callable,
        apply_o: Callable,
        chunk_size: Optional[int] = None,
        chunk_split_strategy: str = "uniform",
        chunk_index: Optional[List[int]] = None,
    ) -> torch.Tensor:
        B, N, C = x.shape

        q, k, v = self._cam_qkv(x)
        q = q.reshape(B, N, self.heads, self.dim)
        k = k.reshape(B, N, self.heads, self.dim)
        v = v.reshape(B, N, self.heads, self.dim)

        if self.conv_k_cam is not None:
            k = self._temporal_short_conv(
                k.reshape(B, N, C), self.conv_k_cam, HW, bidirectional=True
            )
            k = k.reshape(B, N, self.heads, self.dim)

        q = self.q_norm_cam(q.reshape(B, N, C)).reshape(B, N, self.heads, self.dim)
        k = self.k_norm_cam(k.reshape(B, N, C)).reshape(B, N, self.heads, self.dim)

        q_bhnd = q.permute(0, 2, 1, 3)
        k_bhnd = k.permute(0, 2, 1, 3)
        v_bhnd = v.permute(0, 2, 1, 3)

        q_proj = apply_q(q_bhnd)
        kv_proj = apply_kv(torch.cat([k_bhnd, v_bhnd], dim=1))
        k_proj, v_proj = torch.chunk(kv_proj, chunks=2, dim=1)

        q_pre_dn = q_bhnd.permute(0, 1, 3, 2)
        k_pre_dn = k_bhnd.permute(0, 1, 3, 2)
        v_pre_dn = v_bhnd.permute(0, 1, 3, 2)
        q_dn = _downscale_to_reference_rms(q_pre_dn, q_proj.permute(0, 1, 3, 2))
        k_dn = _downscale_to_reference_rms(k_pre_dn, k_proj.permute(0, 1, 3, 2))
        v_dn = _downscale_to_reference_rms(v_pre_dn, v_proj.permute(0, 1, 3, 2))

        q_in = q_dn.permute(0, 3, 1, 2).contiguous()
        k_in = k_dn.permute(0, 3, 1, 2).contiguous()
        v_in = v_dn.permute(0, 3, 1, 2).contiguous()
        out = None
        if self.use_chunked_softmax_attention:
            out = _sana_wm_chunked_attention(
                q_in,
                k_in,
                v_in,
                HW=HW,
                chunk_size=chunk_size,
                chunk_split_strategy=chunk_split_strategy,
                chunk_index=chunk_index,
                softmax_scale=self.softmax_attn.softmax_scale,
            )
        if out is None:
            out = self.softmax_attn(q_in, k_in, v_in)  # (B, N, H, D)

        out_bhnd = out.transpose(1, 2).contiguous()
        out_bhnd = apply_o(out_bhnd)
        return out_bhnd.transpose(1, 2).reshape(B, N, C)

    # ------------------------------------------------------------------ #

    def forward(
        self,
        x: torch.Tensor,
        HW: Tuple[int, int, int],
        rotary_emb: Optional[torch.Tensor] = None,
        prope_fns: Optional[Tuple[Callable, Callable, Callable]] = None,
        chunk_size: Optional[int] = None,
        chunk_split_strategy: str = "uniform",
        chunk_index: Optional[List[int]] = None,
    ) -> torch.Tensor:
        if self.softmax_main:
            main_raw, beta, decay = self._main_branch_softmax(
                x,
                HW,
                rotary_emb,
                chunk_size=chunk_size,
                chunk_split_strategy=chunk_split_strategy,
                chunk_index=chunk_index,
            )
        else:
            main_raw, beta, decay = self._main_branch_gdn(x, HW, rotary_emb)

        if prope_fns is not None:
            apply_q, apply_kv, apply_o = prope_fns
            if self.softmax_main:
                cam_raw = self._cam_branch_softmax(
                    x,
                    HW,
                    apply_q,
                    apply_kv,
                    apply_o,
                    chunk_size=chunk_size,
                    chunk_split_strategy=chunk_split_strategy,
                    chunk_index=chunk_index,
                )
            else:
                cam_raw = self._cam_branch(
                    x, HW, apply_q, apply_kv, apply_o, beta, decay
                )
            combined = main_raw + self.out_proj_cam(cam_raw)
        else:
            combined = main_raw

        # Shared output gate + shared output projection. The SiLU gate is
        # evaluated in fp32 and multiplied before casting for proj.
        gate = F.silu(self.output_gate(x).to(torch.float32))
        combined = combined * gate
        return self.proj(combined.to(self.proj.weight.dtype))

    # ------------------------------------------------------------------ #
    # Streaming `forward_long`: chunk-causal cached branch methods + dispatcher.
    # Each takes a per-block 10-slot `kv_cache` list (mutated in place) +
    # `save_kv_cache`. GDN/cam scans carry recurrent state (slots 0/1/2);
    # softmax blocks use a concat-window (slots 0/1 main, 2/3 cam); short-conv
    # prefix (slot 4); type flag (slot 6). A single chunk with an empty cache
    # reduces to the dense bidirectional path.
    # ------------------------------------------------------------------ #

    def _main_branch_gdn_cached(
        self,
        x: torch.Tensor,
        HW: Tuple[int, int, int],
        rotary_emb: Optional[torch.Tensor],
        kv_cache: list,
        save_kv_cache: bool,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        B, N, C = x.shape
        T, H_sp, W_sp = HW
        S = H_sp * W_sp

        qkv = self.qkv(x).reshape(B, N, 3, self.heads, self.dim).contiguous()
        q, k, v = qkv.unbind(2)

        # Cached forward prefix + intra-chunk backward short conv on K.
        if self.conv_k is not None:
            k, new_prefix = _temporal_short_conv_cached(
                k.reshape(B, N, C),
                self.conv_k,
                HW,
                prefix=kv_cache[_SLOT_SHORTCONV],
                save_prefix=save_kv_cache,
                bidirectional=True,
            )
            k = k.reshape(B, N, self.heads, self.dim)
            if save_kv_cache:
                kv_cache[_SLOT_SHORTCONV] = new_prefix

        beta, decay = _compute_frame_gates(
            x, HW, self.heads, self.beta_proj, self.gate_proj, self.dt_bias, self.A_log
        )

        # Bypass the Triton fast path: it carries no state.
        q = self.q_norm(q.reshape(B, N, C)).reshape(B, N, self.heads, self.dim)
        k = self.k_norm(k.reshape(B, N, C)).reshape(B, N, self.heads, self.dim)
        q = F.relu(q)
        k = F.relu(k)
        k = k * ((self.dim**-0.5) * (S**-0.5))

        q = q.permute(0, 2, 3, 1)
        k = k.permute(0, 2, 3, 1)
        v = v.permute(0, 2, 3, 1)

        rotary_emb = _slice_rope_to_current_chunk(rotary_emb, N)
        if rotary_emb is not None:
            q_rot = _apply_rotary_emb_dn(q, rotary_emb)
            k_rot = _apply_rotary_emb_dn(k, rotary_emb)
        else:
            q_rot = q
            k_rot = k

        dtype = q.dtype
        out, (state_kv, state_z) = _gdn_scan_cached(
            q.float(),
            k.float(),
            v.float(),
            q_rot.float(),
            k_rot.float(),
            beta.float(),
            decay.float(),
            init_state_kv=kv_cache[_SLOT_K],
            init_state_z=kv_cache[_SLOT_V],
            eps=self.eps,
        )
        out = out.to(dtype)
        if save_kv_cache:
            kv_cache[_SLOT_K] = state_kv.detach().clone()
            kv_cache[_SLOT_V] = state_z.detach().clone()
            kv_cache[_SLOT_TYPE_FLAG] = torch.tensor(
                [_CACHE_TYPE_STATE], device=x.device
            )

        out = out.permute(0, 3, 1, 2).reshape(B, N, C)
        return out, beta, decay

    def _main_branch_softmax_cached(
        self,
        x: torch.Tensor,
        HW: Tuple[int, int, int],
        rotary_emb: Optional[torch.Tensor],
        kv_cache: list,
        save_kv_cache: bool,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.heads, self.dim)
        q, k, v = qkv.unbind(2)
        if self.conv_k is not None:  # softmax blocks have conv_k=None
            k, new_prefix = _temporal_short_conv_cached(
                k.reshape(B, N, C),
                self.conv_k,
                HW,
                prefix=kv_cache[_SLOT_SHORTCONV],
                save_prefix=save_kv_cache,
                bidirectional=True,
            )
            k = k.reshape(B, N, self.heads, self.dim)
            if save_kv_cache:
                kv_cache[_SLOT_SHORTCONV] = new_prefix
        q = self.q_norm(q.reshape(B, N, C)).reshape(B, N, self.heads, self.dim)
        k = self.k_norm(k.reshape(B, N, C)).reshape(B, N, self.heads, self.dim)
        q = q.permute(0, 2, 1, 3)  # (B, H, N, D)
        k = k.permute(0, 2, 1, 3)
        v = v.permute(0, 2, 1, 3)
        rotary_emb = _slice_rope_to_current_chunk(rotary_emb, N)
        if rotary_emb is not None:
            q = _apply_rotary_emb_bhnd(q, rotary_emb)
            k = _apply_rotary_emb_bhnd(k, rotary_emb)
        q_in = q.transpose(1, 2).contiguous()  # (B, N_cur, H, D)
        k_in = k.transpose(1, 2).contiguous()
        v_in = v.transpose(1, 2).contiguous()

        # Concat-window: store the CURRENT chunk K/V (RoPE'd at absolute
        # positions), then prepend the cached prefix. Q stays current-chunk.
        cached_k = kv_cache[_SLOT_K]
        cached_v = kv_cache[_SLOT_V]
        if save_kv_cache:
            kv_cache[_SLOT_K] = k_in.detach().clone()
            kv_cache[_SLOT_V] = v_in.detach().clone()
            kv_cache[_SLOT_TYPE_FLAG] = torch.tensor(
                [_CACHE_TYPE_CONCAT], device=x.device
            )
        if cached_k is not None:
            k_in = torch.cat([cached_k.to(k_in.dtype), k_in], dim=1)
            v_in = torch.cat([cached_v.to(v_in.dtype), v_in], dim=1)

        out = _sana_wm_sdpa(
            q_in, k_in, v_in, softmax_scale=_sana_wm_padded_scale(self.dim)
        )
        out = out.reshape(B, N, C)

        beta, decay = _compute_frame_gates(
            x, HW, self.heads, self.beta_proj, self.gate_proj, self.dt_bias, self.A_log
        )
        return out, beta, decay

    def _cam_branch_cached(
        self,
        x: torch.Tensor,
        HW: Tuple[int, int, int],
        apply_q: Callable,
        apply_kv: Callable,
        apply_o: Callable,
        beta: torch.Tensor,
        decay: torch.Tensor,
        kv_cache: list,
        save_kv_cache: bool,
    ) -> torch.Tensor:
        B, N, C = x.shape
        T, H_sp, W_sp = HW
        S = H_sp * W_sp

        q, k, v = self._cam_qkv(x)
        q = q.reshape(B, N, self.heads, self.dim)
        k = k.reshape(B, N, self.heads, self.dim)
        v = v.reshape(B, N, self.heads, self.dim)

        # Cam K short conv stays bidirectional/uncached; slot 4 is main-K only.
        if self.conv_k_cam is not None:
            k = self._temporal_short_conv(
                k.reshape(B, N, C), self.conv_k_cam, HW, bidirectional=True
            )
            k = k.reshape(B, N, self.heads, self.dim)

        q = self.q_norm_cam(q.reshape(B, N, C)).reshape(B, N, self.heads, self.dim)
        k = self.k_norm_cam(k.reshape(B, N, C)).reshape(B, N, self.heads, self.dim)
        q = F.relu(q)
        k = F.relu(k)
        k = k * ((self.dim**-0.5) * (S**-0.5))

        q_bhnd = q.permute(0, 2, 1, 3)
        k_bhnd = k.permute(0, 2, 1, 3)
        v_bhnd = v.permute(0, 2, 1, 3)

        q_proj = apply_q(q_bhnd)
        kv_proj = apply_kv(torch.cat([k_bhnd, v_bhnd], dim=1))
        k_proj, v_proj = torch.chunk(kv_proj, chunks=2, dim=1)

        q_dn = q_proj.permute(0, 1, 3, 2)
        k_pre_dn = k_bhnd.permute(0, 1, 3, 2)
        k_dn = k_proj.permute(0, 1, 3, 2)
        v_dn = v_proj.permute(0, 1, 3, 2)

        # No RMS downscale here: full post-UCPE q/k/v feed the scan; inflation
        # is computed from full post-UCPE K vs pre-UCPE K and absorbed only into
        # beta.
        pre_ucpe_k_norm = torch.linalg.vector_norm(
            k_pre_dn.float(), dim=2, keepdim=True
        ).clamp_min(1e-6)
        post_ucpe_k_norm = torch.linalg.vector_norm(
            k_dn.float(), dim=2, keepdim=True
        ).clamp_min(1e-6)
        inflation_sq = (post_ucpe_k_norm / pre_ucpe_k_norm) ** 2
        frame_inflation_sq = inflation_sq.view(B, self.heads, T, S).mean(dim=-1)
        if beta.ndim == 3:
            beta = beta / frame_inflation_sq.clamp_min(1.0)
        elif beta.ndim == 4:
            beta = beta / frame_inflation_sq.unsqueeze(-1).clamp_min(1.0)

        dtype = q_dn.dtype
        # Bypass the Triton cam scan: it carries no state.
        out, cam_state = _single_path_delta_scan_cached(
            q_dn.float(),
            k_dn.float(),
            v_dn.float(),
            beta.float(),
            decay.float(),
            init_state_kv=kv_cache[_SLOT_CAM_K],
        )
        out = out.to(dtype)
        if save_kv_cache:
            kv_cache[_SLOT_CAM_K] = cam_state.detach().clone()

        out_bhnd = out.permute(0, 1, 3, 2)
        out_bhnd = apply_o(out_bhnd)
        return out_bhnd.permute(0, 2, 1, 3).reshape(B, N, C)

    def _cam_branch_softmax_cached(
        self,
        x: torch.Tensor,
        HW: Tuple[int, int, int],
        apply_q: Callable,
        apply_kv: Callable,
        apply_o: Callable,
        kv_cache: list,
        save_kv_cache: bool,
    ) -> torch.Tensor:
        B, N, C = x.shape

        q, k, v = self._cam_qkv(x)
        q = q.reshape(B, N, self.heads, self.dim)
        k = k.reshape(B, N, self.heads, self.dim)
        v = v.reshape(B, N, self.heads, self.dim)

        if self.conv_k_cam is not None:
            k = self._temporal_short_conv(
                k.reshape(B, N, C), self.conv_k_cam, HW, bidirectional=True
            )
            k = k.reshape(B, N, self.heads, self.dim)

        q = self.q_norm_cam(q.reshape(B, N, C)).reshape(B, N, self.heads, self.dim)
        k = self.k_norm_cam(k.reshape(B, N, C)).reshape(B, N, self.heads, self.dim)

        q_bhnd = q.permute(0, 2, 1, 3)
        k_bhnd = k.permute(0, 2, 1, 3)
        v_bhnd = v.permute(0, 2, 1, 3)

        q_proj = apply_q(q_bhnd)
        kv_proj = apply_kv(torch.cat([k_bhnd, v_bhnd], dim=1))
        k_proj, v_proj = torch.chunk(kv_proj, chunks=2, dim=1)

        # No RMS downscale here: full post-UCPE q/k/v feed SDPA directly.
        q_dn = q_proj.permute(0, 1, 3, 2)
        k_dn = k_proj.permute(0, 1, 3, 2)
        v_dn = v_proj.permute(0, 1, 3, 2)

        q_in = q_dn.permute(0, 3, 1, 2).contiguous()  # (B, N_cur, H, D)
        k_in = k_dn.permute(0, 3, 1, 2).contiguous()
        v_in = v_dn.permute(0, 3, 1, 2).contiguous()

        cached_cam_k = kv_cache[_SLOT_CAM_K]
        cached_cam_v = kv_cache[_SLOT_CAM_V]
        if save_kv_cache:
            kv_cache[_SLOT_CAM_K] = k_in.detach().clone()
            kv_cache[_SLOT_CAM_V] = v_in.detach().clone()
        if cached_cam_k is not None:
            k_in = torch.cat([cached_cam_k.to(k_in.dtype), k_in], dim=1)
            v_in = torch.cat([cached_cam_v.to(v_in.dtype), v_in], dim=1)

        out = _sana_wm_sdpa(
            q_in, k_in, v_in, softmax_scale=_sana_wm_padded_scale(self.dim)
        )  # (B, N_cur, H, D)
        out_bhnd = out.transpose(1, 2).contiguous()
        out_bhnd = apply_o(out_bhnd)
        return out_bhnd.transpose(1, 2).reshape(B, N, C)

    def forward_long(
        self,
        x: torch.Tensor,
        HW: Tuple[int, int, int],
        rotary_emb: Optional[torch.Tensor] = None,
        prope_fns: Optional[Tuple[Callable, Callable, Callable]] = None,
        *,
        kv_cache: list,
        save_kv_cache: bool,
    ) -> Tuple[torch.Tensor, list]:
        if self.softmax_main:
            main_raw, beta, decay = self._main_branch_softmax_cached(
                x, HW, rotary_emb, kv_cache, save_kv_cache
            )
        else:
            main_raw, beta, decay = self._main_branch_gdn_cached(
                x, HW, rotary_emb, kv_cache, save_kv_cache
            )

        if prope_fns is not None:
            apply_q, apply_kv, apply_o = prope_fns
            if self.softmax_main:
                cam_raw = self._cam_branch_softmax_cached(
                    x, HW, apply_q, apply_kv, apply_o, kv_cache, save_kv_cache
                )
            else:
                cam_raw = self._cam_branch_cached(
                    x,
                    HW,
                    apply_q,
                    apply_kv,
                    apply_o,
                    beta,
                    decay,
                    kv_cache,
                    save_kv_cache,
                )
            combined = main_raw + self.out_proj_cam(cam_raw)
        else:
            combined = main_raw

        gate = F.silu(self.output_gate(x).to(torch.float32))
        combined = combined * gate
        return self.proj(combined.to(self.proj.weight.dtype)), kv_cache


# ---------------------------------------------------------------------------
# Cross-attention (text conditioning). Stored as
# ``cross_attn.{q_linear, kv_linear, proj, q_norm, k_norm}``.
# ---------------------------------------------------------------------------


class MultiHeadCrossAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, qk_norm: bool = True) -> None:
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        self.q_linear = nn.Linear(d_model, d_model, bias=True)
        self.kv_linear = nn.Linear(d_model, d_model * 2, bias=True)
        self.proj = nn.Linear(d_model, d_model, bias=True)
        if qk_norm:
            self.q_norm = _RMSNorm(d_model, eps=1e-6)
            self.k_norm = _RMSNorm(d_model, eps=1e-6)
        else:
            self.q_norm = nn.Identity()
            self.k_norm = nn.Identity()
        # Padding-mask path falls back to SDPA internally; the unmasked path
        # can pick FA3 / FlashInfer / etc.
        self.attn = LocalAttention(
            num_heads=num_heads,
            head_size=self.head_dim,
        )

    def forward(
        self,
        x: torch.Tensor,  # (B, N, D)
        cond: torch.Tensor,  # (B, L, D)
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, N, D = x.shape

        q = self.q_linear(x)
        kv = self.kv_linear(cond).view(B, -1, 2, D)
        k, v = kv.unbind(2)
        # LocalAttention takes (B, N, H, D).
        q = self.q_norm(q).view(B, N, self.num_heads, self.head_dim)
        k = self.k_norm(k).view(B, -1, self.num_heads, self.head_dim)
        v = v.view(B, -1, self.num_heads, self.head_dim)

        attn_mask = mask.bool() if mask is not None else None
        out = self.attn(q, k, v, attn_mask=attn_mask)  # (B, N, H, D)
        out = out.reshape(B, N, D)
        return self.proj(out)
