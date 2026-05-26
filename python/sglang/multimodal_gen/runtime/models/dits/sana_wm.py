# SPDX-License-Identifier: Apache-2.0
#
# SANA-WM (Sana World Model) DiT for SGLang Diffusion.
#
# This is an inference-only torch port of NVlabs/Sana's
# ``SanaMSVideoCamCtrl_1600M_P1_D20`` (depth=20, hidden=2240, heads=20,
# linear_head_dim=112), the 2.6B TI2V world model behind the released
# ``Efficient-Large-Model/SANA-WM_bidirectional`` checkpoint.
#
# Parameter names and module layout track the upstream training code so the
# raw safetensors at ``dit/sana_wm_1600m_720p.safetensors`` load cleanly
# without key surgery.  Upstream sources mirrored here:
#   * ``diffusion/model/nets/sana_multi_scale_video_camctrl.py``
#   * ``diffusion/model/nets/sana_gdn_blocks.py``
#   * ``diffusion/model/nets/sana_gdn_camctrl_blocks.py``
#   * ``diffusion/model/nets/sana_camctrl_blocks.py``
#   * ``diffusion/model/nets/sana_blocks.py``
#   * ``diffusion/model/nets/basic_modules.py``
#
# Key architectural features (matching upstream):
#   1. 5D video latent (B, C, T, H, W) — patch_embed is Conv3d with kernel
#      (1, P, P); each latent voxel becomes one token.
#   2. Hybrid main-branch attention: 15 GDN blocks + 5 softmax blocks at
#      indices {3, 7, 11, 15, 19} (``softmax_every_n=4``). All blocks share
#      the same parameter names; the softmax variant just swaps the main
#      scan for SDPA.
#   3. Dual-branch attention: every block has an extra GDN-style camera
#      branch (``q_proj_cam`` / ``k_proj_cam`` / ``v_proj_cam`` /
#      ``out_proj_cam`` + ``conv_k_cam``) whose K is rotated by per-ray
#      UCPE matrices.  ``out_proj_cam`` is zero-init so the camera branch
#      contributes zero at the start of training.
#   4. Camera conditioning input is ``(B, F, 20)``: 16 c2w flat + 4
#      intrinsics ``(fx, fy, cx, cy)``.  Upstream's
#      ``_process_camera_conditions_ucpe`` converts this to ``raymats``
#      ``(B, F, H, W, 4, 4)`` and ``absmap`` ``(B, F, H, W, 3)``.
#   5. Plücker post-attn mixing: a single shared
#      ``plucker_embedder = PatchEmbedMS3D(48, hidden)`` (zero-init) plus
#      a per-block ``plucker_proj = Linear(hidden, hidden)`` (zero-init).
#   6. ``GLUMBConvTemp`` FFN: 2D spatial ``inverted_conv / depth_conv /
#      point_conv`` followed by an additive temporal Conv2d ``t_conv``
#      (zero-init).
#   7. 3D RoPE via ``WanRotaryPosEmbed`` (split across t / h / w head dims).
#   8. AdaLN-single: ``t_embedder`` produces a timestep embedding, then
#      ``t_block = SiLU + Linear(D, 6D)`` produces per-block
#      ``(shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp)``
#      modulation factors via ``scale_shift_table``.
#   9. Output: ``final_layer`` = LayerNorm + scale_shift_table + Linear.
#
# Inference simplifications vs upstream:
#   * Single GDN update rule (torch chunk-parallel) instead of Triton kernels.
#   * No ``frame_valid_mask``, ``chunk_index``, ``block_mask``, or
#     CP/SP-aware halo exchange (single-GPU inference only).
#   * No xformers code path; cross-attention uses SDPA.
#
# Reference checkpoint: ``Efficient-Large-Model/SANA-WM_bidirectional``.

import math
from typing import Callable, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers.models.embeddings import get_1d_rotary_pos_embed

from sglang.multimodal_gen.configs.models.dits.sana_wm import SanaWMConfig
from sglang.multimodal_gen.runtime.layers.attention import LocalAttention
from sglang.multimodal_gen.runtime.managers.memory_managers.layerwise_offload import (
    LayerwiseOffloadableModuleMixin,
)
from sglang.multimodal_gen.runtime.models.dits.base import CachableDiT
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

logger = init_logger(__name__)


# ---------------------------------------------------------------------------
# Small primitives (RMSNorm, ShortConvolution) -- shapes match upstream
# parameter names exactly so the released checkpoint loads cleanly.
# ---------------------------------------------------------------------------


class _RMSNorm(nn.Module):
    """RMSNorm matching upstream signature ``RMSNorm(dim, scale_factor=1.0, eps=1e-6)``.

    Parameter name: ``weight`` (shape ``(dim,)``).
    """

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # fp32 reduction for stability
        x_in = x
        x32 = x.float()
        rms = x32.pow(2).mean(dim=-1, keepdim=True).add(self.eps).rsqrt()
        return (x32 * rms).to(x_in.dtype) * self.weight


class _ShortConvolution(nn.Module):
    """Depth-wise causal Conv1d along the temporal axis.

    Mirrors FLA's ``ShortConvolution(hidden_size, kernel_size=K)``:
      * weight shape: ``(hidden_size, 1, K)`` (groups=hidden_size)

    Input is ``(B, T, C)``.  Causal padding of ``K-1`` on the left.
    """

    def __init__(self, hidden_size: int, kernel_size: int) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.kernel_size = kernel_size
        self.weight = nn.Parameter(torch.zeros(hidden_size, 1, kernel_size))
        # identity init: last tap = 1. The released SANA-WM checkpoint has no
        # ShortConvolution bias tensors, so keep this module bias-free.
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
    """Forward + backward causal pass with shared kernel, minus the shared
    center tap.  Equivalent to a symmetric (non-causal) filter, matching
    upstream's ``BidirectionalGDN._bidirectional_causal_conv_1d``.
    """
    y_fwd, _ = conv(x)
    y_bwd, _ = conv(x.flip(1))
    y_bwd = y_bwd.flip(1)
    w_center = conv.weight[:, 0, -1]  # (C,)
    center = x * w_center.view(1, 1, -1)
    return (y_fwd + y_bwd - center).to(x.dtype)


# ---------------------------------------------------------------------------
# 3D RoPE -- ``WanRotaryPosEmbed`` from upstream sana_blocks.py
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
        # Extracted so SanaWMTransformer3DModel.post_load_weights can re-run
        # it: this is a persistent=False buffer, so it is not in the upstream
        # checkpoint and stays on meta after FSDP weight load.
        h_dim = w_dim = 2 * (self.attention_head_dim // 6)
        t_dim = self.attention_head_dim - h_dim - w_dim

        freqs = []
        for dim in [t_dim, h_dim, w_dim]:
            freq = get_1d_rotary_pos_embed(
                dim, self.max_seq_len, self.theta,
                use_real=False, repeat_interleave_real=False,
                freqs_dtype=torch.float64,
            )
            freqs.append(freq)
        self.register_buffer("_freqs", torch.cat(freqs, dim=1), persistent=False)

    def forward(self, fhw: Tuple[int, int, int], device: torch.device) -> torch.Tensor:
        ppf, pph, ppw = fhw
        freqs = self._freqs.to(device)
        d = self.attention_head_dim
        t_size = d // 2 - 2 * (d // 6)
        h_size = d // 6
        w_size = d // 6
        ft, fh, fw = freqs.split_with_sizes([t_size, h_size, w_size], dim=1)

        freqs_t = ft[:ppf].view(ppf, 1, 1, -1).expand(ppf, pph, ppw, -1)
        freqs_h = fh[:pph].view(1, pph, 1, -1).expand(ppf, pph, ppw, -1)
        freqs_w = fw[:ppw].view(1, 1, ppw, -1).expand(ppf, pph, ppw, -1)
        out = torch.cat([freqs_t, freqs_h, freqs_w], dim=-1)
        return out.reshape(1, 1, ppf * pph * ppw, -1)


def _apply_rotary_emb_dn(hidden_states: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
    """Apply complex RoPE to a tensor of shape ``(B, H, D, N)`` (GDN layout).

    ``freqs`` is complex with shape ``(1, 1, N, D/2)``.
    """
    # (B, H, D, N) -> (B, H, N, D)
    x = hidden_states.permute(0, 1, 3, 2).to(torch.float64).contiguous()
    x_c = torch.view_as_complex(x.unflatten(-1, (-1, 2)))
    y = torch.view_as_real(x_c * freqs).flatten(-2, -1)
    return y.permute(0, 1, 3, 2).type_as(hidden_states)


def _apply_rotary_emb_bhnd(hidden_states: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
    """Apply complex RoPE to ``(B, H, N, D)`` (softmax attention layout)."""
    x = hidden_states.to(torch.float64).contiguous()
    x_c = torch.view_as_complex(x.unflatten(-1, (-1, 2)))
    y = torch.view_as_real(x_c * freqs).flatten(-2, -1)
    return y.type_as(hidden_states)


# ---------------------------------------------------------------------------
# UCPE block-diagonal apply primitives (mirror upstream
# diffusion/model/nets/sana_camctrl_blocks.py)
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


def _apply_complex_rope(hidden_states: torch.Tensor, freqs: torch.Tensor, inverse: bool = False) -> torch.Tensor:
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
    h_part = rotary_emb[..., orig_t_size:orig_t_size + new_h_size]
    w_part = rotary_emb[..., orig_t_size + orig_h_size:orig_t_size + orig_h_size + new_w_size]
    return torch.cat([t_part, h_part, w_part], dim=-1)


def _build_ucpe_apply_fns(
    head_dim: int,
    raymats: torch.Tensor,           # (B, N, 4, 4) -- ray<-world
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
        rope_fn = lambda x: _apply_complex_rope(x, rotary_emb_cam, inverse=False)
        rope_fn_inv = lambda x: _apply_complex_rope(x, rotary_emb_cam, inverse=True)
    else:
        rope_fn = lambda x: x
        rope_fn_inv = lambda x: x

    half = head_dim // 2
    apply_q = lambda x: _apply_block_diagonal(
        x, [(lambda y: _apply_ray_projmat(y, P_T), half), (rope_fn, half)]
    )
    apply_kv = lambda x: _apply_block_diagonal(
        x, [(lambda y: _apply_ray_projmat(y, P_inv), half), (rope_fn, half)]
    )
    apply_o = lambda x: _apply_block_diagonal(
        x, [(lambda y: _apply_ray_projmat(y, P), half), (rope_fn_inv, half)]
    )
    return apply_q, apply_kv, apply_o


def _compute_fov_from_focal(focal: torch.Tensor, image_size: int) -> torch.Tensor:
    """fov = 2 * atan(image_size / (2 * focal))"""
    return 2.0 * torch.atan(image_size / (2.0 * focal.clamp(min=1e-6)))


def _unproject_grid(
    x_fov: torch.Tensor, y_fov: torch.Tensor,
    H: int, W: int, cx: torch.Tensor, cy: torch.Tensor,
    device: torch.device, dtype: torch.dtype,
) -> torch.Tensor:
    """Compute camera-space unit ray directions for each latent token.

    Returns (B, F, H, W, 3).
    """
    B, F_dim = x_fov.shape
    # Pixel centers in latent grid coords (matches upstream's
    # ``ucm_unproject_grid_fov`` with xi=0 pinhole assumption).
    u = (torch.arange(W, device=device, dtype=dtype) + 0.5)
    v = (torch.arange(H, device=device, dtype=dtype) + 0.5)
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
    camera_conditions: torch.Tensor,   # (B, F, 20)
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

    d_cam = _unproject_grid(x_fov, y_fov, H, W, cx_lat, cy_lat, device, dtype)  # (B,F,H,W,3)

    # Build per-token "ray<-world" 4x4. Rotation = orthonormal frame with
    # third column = world-space ray direction. Translation = camera origin.
    R_c2w = C_to_W[..., :3, :3]                 # (B, F, 3, 3)
    t_c2w = C_to_W[..., :3, 3]                  # (B, F, 3)
    # d_world: (B, F, H, W, 3)
    d_world = torch.einsum("bfij,bfhwj->bfhwi", R_c2w, d_cam)
    d_world = F.normalize(d_world, dim=-1)

    # Build orthonormal local frame [e1 | e2 | d_world]
    up = torch.zeros_like(d_world)
    up[..., 1] = 1.0
    is_par = (d_world * up).sum(-1).abs() > 0.99
    up_alt = torch.zeros_like(d_world)
    up_alt[..., 0] = 1.0
    up = torch.where(is_par.unsqueeze(-1), up_alt, up)
    e1 = F.normalize(torch.cross(up, d_world, dim=-1), dim=-1)
    e2 = F.normalize(torch.cross(d_world, e1, dim=-1), dim=-1)
    R_ray_to_world = torch.stack([e1, e2, d_world], dim=-1)  # (B,F,H,W,3,3)

    # P = ray<-world = inverse of [R_ray_to_world | t_c2w]
    R_w_to_ray = R_ray_to_world.transpose(-1, -2)
    t_w_to_ray = -torch.einsum(
        "bfhwij,bfj->bfhwi", R_w_to_ray,
        t_c2w,
    )
    raymats = torch.zeros(B, F_dim, H, W, 4, 4, device=device, dtype=dtype)
    raymats[..., :3, :3] = R_w_to_ray
    raymats[..., :3, 3] = t_w_to_ray
    raymats[..., 3, 3] = 1.0
    return raymats


def compute_chunk_plucker(
    camera_conditions: torch.Tensor,   # (B, F_orig, 20)
    HW: Tuple[int, int, int],          # latent (T, H, W) where T = F_orig // vae_temporal_stride
    vae_temporal_stride: int = 8,
    patch_size: Tuple[int, int, int] = (1, 1, 1),
) -> torch.Tensor:
    """Compute the 48-channel packed Plücker raymap consumed by
    ``plucker_embedder``.

    Each latent frame packs ``vae_temporal_stride`` original frames'
    Plücker coords ``[d, o×d]`` (6D each) into 48 channels.  Output shape
    ``(B, 48, T, H, W)`` for direct consumption by Conv3d.
    """
    B, F_orig, _ = camera_conditions.shape
    T, H, W = HW
    assert F_orig == T * vae_temporal_stride, (
        f"camera_conditions has {F_orig} frames but expected T*stride = "
        f"{T}*{vae_temporal_stride} = {T * vae_temporal_stride}"
    )
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
        x_fov, y_fov, H, W,
        cx / float(patch_size[2]), cy / float(patch_size[1]),
        device, dtype,
    )  # (B, F_orig, H, W, 3)

    R = c2w[..., :3, :3]
    o = c2w[..., :3, 3]                          # (B, F_orig, 3)
    d_world = F.normalize(torch.einsum("bfij,bfhwj->bfhwi", R, d_cam), dim=-1)
    o_exp = o.view(B, F_orig, 1, 1, 3).expand_as(d_world)
    moment = torch.cross(o_exp, d_world, dim=-1)
    plucker = torch.cat([d_world, moment], dim=-1)  # (B, F_orig, H, W, 6)

    # Pack `vae_temporal_stride` orig frames into one latent frame ->
    # 48 channels.
    plucker = plucker.view(B, T, vae_temporal_stride, H, W, 6)
    plucker = plucker.permute(0, 1, 3, 4, 2, 5).reshape(B, T, H, W, vae_temporal_stride * 6)
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
        self.proj = nn.Conv3d(in_chans, embed_dim, kernel_size=kernel_size, stride=patch_size, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)              # (B, D, T, H, W)
        return x.flatten(2).transpose(1, 2)  # (B, T*H*W, D)


class _UpstreamMlp(nn.Module):
    """timm-style Mlp used by ``y_embedder.y_proj`` (fc1/fc2 with bias=True)."""

    def __init__(self, in_features: int, hidden_features: int, out_features: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features, bias=True)
        self.act = nn.GELU(approximate="tanh")
        self.fc2 = nn.Linear(hidden_features, out_features, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act(self.fc1(x)))


class CaptionEmbedder(nn.Module):
    """Upstream ``CaptionEmbedder``: projects text encoder embeddings to
    model hidden size, plus a learned null-caption table used by CFG.

    Inference-only forward: ``y`` may be ``(B, 1, L, in_channels)`` or
    ``(B, L, in_channels)``; the null table is not applied (no dropout at
    inference time).  Returns ``(B, 1, L, hidden_size)``.
    """

    def __init__(self, in_channels: int, hidden_size: int, token_num: int = 300) -> None:
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

    def __init__(self, hidden_size: int, patch_size: Tuple[int, int, int], out_channels: int) -> None:
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, math.prod(patch_size) * out_channels, bias=True)
        self.scale_shift_table = nn.Parameter(torch.randn(2, hidden_size) / hidden_size**0.5)
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


def _sinusoidal_timestep_embedding(t: torch.Tensor, dim: int, max_period: float = 10000.0) -> torch.Tensor:
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(half, dtype=torch.float32, device=t.device) / half
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
        # use the dtype of the first linear's weight to match upstream
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

    def __init__(self, in_features: int, hidden_features: int, t_kernel_size: int = 3) -> None:
        super().__init__()
        self.inverted_conv = _ConvLayer(
            nn.Conv2d(in_features, hidden_features * 2, 1, 1, 0, bias=True),
            act=nn.SiLU(inplace=False),
        )
        self.depth_conv = _ConvLayer(
            nn.Conv2d(
                hidden_features * 2, hidden_features * 2,
                3, 1, 1,
                groups=hidden_features * 2, bias=True,
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
            in_features, in_features,
            kernel_size=(t_kernel_size, 1),
            stride=1,
            padding=(t_padding, 0),
            bias=False,
        )
        nn.init.zeros_(self.t_conv.weight)

    def forward(self, x: torch.Tensor, HW: Tuple[int, int, int]) -> torch.Tensor:
        B, N, C = x.shape
        T, H, W = HW
        assert N == T * H * W, f"GLUMBConvTemp: N={N} != T*H*W={T * H * W}"

        # Spatial path -- (B*T, C, H, W)
        x_sp = x.reshape(B * T, H, W, C).permute(0, 3, 1, 2).contiguous()
        x_sp = self.inverted_conv(x_sp)
        x_sp = self.depth_conv(x_sp)
        a, g = x_sp.chunk(2, dim=1)
        x_sp = a * self.glu_act(g)
        x_sp = self.point_conv(x_sp)   # (B*T, C, H, W)

        # Temporal additive path -- (B, C, T, S=H*W)
        x_t = x_sp.view(B, T, C, H * W).permute(0, 2, 1, 3).contiguous()
        x_out = x_t + self.t_conv(x_t)

        # back to (B, N, C)
        return x_out.permute(0, 2, 3, 1).reshape(B, N, C)


# ---------------------------------------------------------------------------
# Frame-gate and DeltaNet update rule (torch port of upstream
# torch_chunk_sana_gdn / torch_recurrent_sana_gdn).  Inference path uses
# the recurrent form for simplicity and clarity; chunk-parallel form is
# numerically equivalent.
# ---------------------------------------------------------------------------


def _compute_frame_gates(
    x: torch.Tensor,                 # (B, N, C)
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
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
    q_rot: torch.Tensor, k_rot: torch.Tensor,
    beta: torch.Tensor, decay: torch.Tensor,
    eps: float = 1e-6,
    return_components: bool = False,
) -> torch.Tensor:
    """Causal recurrent GDN scan over T (mirrors
    ``torch_recurrent_sana_gdn``).  Tensors are in (B, H, D, N=T*S) layout.
    """
    B, H, D, N = q.shape
    T = beta.shape[2]
    S = N // T

    def fold(x):
        return x.view(B, H, D, T, S).permute(0, 1, 3, 2, 4)  # (B, H, T, D, S)

    q, k, v = fold(q), fold(k), fold(v)
    q_rot, k_rot = fold(q_rot), fold(k_rot)
    if beta.ndim == 4:
        beta_e = beta.unsqueeze(3)        # (B, H, T, 1, S)
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
        # decay
        state_kv = state_kv * gt
        state_z = state_z * gt
        # KV delta update
        v_pred = torch.matmul(state_kv, krt)
        delta_v = (vt - v_pred) * bt
        state_kv = state_kv + torch.matmul(delta_v, krt.transpose(-1, -2))
        # Z delta update
        z_pred = torch.matmul(state_z.transpose(-1, -2), kt)
        delta_z = (target_z - z_pred) * bt
        state_z = state_z + torch.matmul(kt, delta_z.transpose(-1, -2))
        # output components
        num_list.append(torch.matmul(state_kv, qrt))               # (B, H, D, S)
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


def _flip_and_shift(x: torch.Tensor, dim: int, shift_val: float = 0.0) -> torch.Tensor:
    """``flip_and_shift`` helper from upstream: flip along ``dim`` then shift
    by one with ``shift_val`` filling the head.  Used for the backward pass
    of bidirectional GDN."""
    x_flipped = x.flip(dim)
    idx = [slice(None)] * x.ndim
    idx[dim] = slice(0, 1)
    head = torch.full_like(x_flipped[tuple(idx)], shift_val)
    idx[dim] = slice(0, -1)
    return torch.cat([head, x_flipped[tuple(idx)]], dim=dim)


def _gdn_scan_bidirectional(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
    q_rot: torch.Tensor, k_rot: torch.Tensor,
    beta: torch.Tensor, decay: torch.Tensor,
    HW: Tuple[int, int, int],
    eps: float = 1e-6,
) -> torch.Tensor:
    """Bidirectional GDN: forward (inclusive) + backward (exclusive) scan,
    summed in numerator/denominator space (mirrors upstream
    ``BidirectionalGDN.forward``).
    """
    num_fwd, den_fwd = _gdn_scan_forward(q, k, v, q_rot, k_rot, beta, decay, eps=eps, return_components=True)

    # Backward pass: flip Q, flip+shift K/V/k_rot/beta and shift decay-by-1.
    B, H, D, N = q.shape
    T, H_sp, W_sp = HW
    S = H_sp * W_sp

    def to_time(x, d):
        return x.view(B, H, d, T, S).permute(0, 1, 3, 2, 4)

    def from_time(x, d):
        return x.permute(0, 1, 3, 2, 4).reshape(B, H, d, N)

    q_t = to_time(q, D); k_t = to_time(k, D); v_t = to_time(v, D)
    q_rot_t = to_time(q_rot, D); k_rot_t = to_time(k_rot, D)

    q_bwd = torch.flip(q_t, dims=[2])
    q_rot_bwd = torch.flip(q_rot_t, dims=[2])
    k_bwd = _flip_and_shift(k_t, dim=2, shift_val=0.0)
    v_bwd = _flip_and_shift(v_t, dim=2, shift_val=0.0)
    k_rot_bwd = _flip_and_shift(k_rot_t, dim=2, shift_val=0.0)
    beta_bwd = _flip_and_shift(beta, dim=2, shift_val=0.0)
    decay_bwd = _flip_and_shift(decay, dim=2, shift_val=1.0)

    q_bwd_f = from_time(q_bwd, D); k_bwd_f = from_time(k_bwd, D); v_bwd_f = from_time(v_bwd, D)
    q_rot_bwd_f = from_time(q_rot_bwd, D); k_rot_bwd_f = from_time(k_rot_bwd, D)

    num_bwd_flipped, den_bwd_flipped = _gdn_scan_forward(
        q_bwd_f, k_bwd_f, v_bwd_f, q_rot_bwd_f, k_rot_bwd_f,
        beta_bwd, decay_bwd, eps=eps, return_components=True,
    )

    def flip_back(tensor):
        d_actual = tensor.shape[2]
        t_struct = tensor.view(B, H, d_actual, T, S)
        return torch.flip(t_struct, dims=[3]).reshape(B, H, d_actual, N)

    num_bwd = flip_back(num_bwd_flipped)
    den_bwd = flip_back(den_bwd_flipped)
    return (num_fwd + num_bwd) / (den_fwd + den_bwd + eps)


# ---------------------------------------------------------------------------
# SANA-WM attention block: main GDN (or softmax) + UCPE camera branch
# (single SinglePath module; cam branch shares ``proj`` / ``output_gate``
# with the main branch).
# ---------------------------------------------------------------------------


class BidirectionalGDNUCPESinglePathLiteLA(nn.Module):
    """Bidirectional GDN main branch + UCPE camera branch (single-path
    output: ``main + out_proj_cam(cam_raw)`` then shared output gate +
    shared output projection).

    Parameter naming follows upstream exactly so the checkpoint loads.
    """

    def __init__(
        self,
        in_dim: int,
        heads: int,
        head_dim: int,
        qk_norm: bool = True,
        conv_kernel_size: int = 4,
        k_conv_only: bool = True,
        eps: float = 1e-6,
        softmax_main: bool = False,
    ) -> None:
        super().__init__()
        out_dim = heads * head_dim
        assert out_dim == in_dim, f"in_dim ({in_dim}) must equal heads*head_dim ({out_dim})"
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.heads = heads
        self.dim = head_dim
        self.eps = eps
        self.softmax_main = softmax_main

        # Main branch: fused QKV + output proj (shared with cam branch)
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

        # GDN-specific (also held by softmax variant for state_dict compat)
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

        # Camera branch -- separate QKV + zero-init output proj, plus a
        # separate ShortConvolution for K.  Both branches share ``proj``.
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

        # Softmax-variant blocks (every Nth block, controlled by softmax_main)
        # route attention through SGLang's pluggable backend (FA3 / FlashInfer /
        # Triton / SDPA). GDN blocks compute attention via _gdn_scan_bidirectional
        # and don't go through this path.
        if softmax_main:
            self.softmax_attn = LocalAttention(
                num_heads=heads,
                head_size=head_dim,
            )

    # ------------------------------------------------------------------ #

    @staticmethod
    def _temporal_short_conv(
        x: torch.Tensor,                 # (B, N, C)
        conv: _ShortConvolution,
        HW: Tuple[int, int, int],
        bidirectional: bool = True,
    ) -> torch.Tensor:
        B, N, C = x.shape
        T, H, W = HW
        S = H * W
        # Move T into the time axis: (B, T, S, C) -> (B*S, T, C)
        y = x.view(B, T, S, C).permute(0, 2, 1, 3).contiguous().reshape(B * S, T, C)
        if bidirectional:
            y = _bidirectional_short_conv(y, conv)
        else:
            y, _ = conv(y)
        # back to (B, T*S, C)
        return y.reshape(B, S, T, C).permute(0, 2, 1, 3).reshape(B, N, C)

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

        qkv = self.qkv(x).reshape(B, N, 3, self.heads, self.dim)
        q, k, v = qkv.unbind(2)

        # Short conv on K only.
        if self.conv_k is not None:
            k = self._temporal_short_conv(k.reshape(B, N, C), self.conv_k, HW, bidirectional=True)
            k = k.reshape(B, N, self.heads, self.dim)

        # Q/K norm on the flattened channel dim (B, N, C)
        q = self.q_norm(q.reshape(B, N, C)).reshape(B, N, self.heads, self.dim)
        k = self.k_norm(k.reshape(B, N, C)).reshape(B, N, self.heads, self.dim)

        # ReLU kernel + key scale
        q = F.relu(q)
        k = F.relu(k)
        k_scale = (self.dim ** -0.5) * (S ** -0.5)
        k = k * k_scale

        # Move to (B, H, D, N)
        q = q.permute(0, 2, 3, 1)
        k = k.permute(0, 2, 3, 1)
        v = v.permute(0, 2, 3, 1)

        if rotary_emb is not None:
            q_rot = _apply_rotary_emb_dn(q, rotary_emb)
            k_rot = _apply_rotary_emb_dn(k, rotary_emb)
        else:
            q_rot = q
            k_rot = k

        beta, decay = _compute_frame_gates(
            x, HW, self.heads, self.beta_proj, self.gate_proj, self.dt_bias, self.A_log,
        )

        # fp32 scan for stability
        dtype = q.dtype
        out = _gdn_scan_bidirectional(
            q.float(), k.float(), v.float(),
            q_rot.float(), k_rot.float(),
            beta.float(), decay.float(),
            HW=HW, eps=self.eps,
        ).to(dtype)

        out = out.permute(0, 3, 1, 2).reshape(B, N, C)
        return out, beta, decay

    def _main_branch_softmax(
        self,
        x: torch.Tensor,
        HW: Tuple[int, int, int],
        rotary_emb: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Softmax variant of the main branch, dispatched through SGLang's
        pluggable attention backend.

        Returns ``(out_raw, beta, decay)`` so the cam branch can reuse the
        shared gates -- exactly like upstream
        ``BidirectionalSoftmaxUCPESinglePathLiteLA``.
        """
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.heads, self.dim)
        q, k, v = qkv.unbind(2)
        if self.conv_k is not None:
            k = self._temporal_short_conv(k.reshape(B, N, C), self.conv_k, HW, bidirectional=True)
            k = k.reshape(B, N, self.heads, self.dim)
        q = self.q_norm(q.reshape(B, N, C)).reshape(B, N, self.heads, self.dim)
        k = self.k_norm(k.reshape(B, N, C)).reshape(B, N, self.heads, self.dim)
        # RoPE primitives are written for (B, H, N, D); LocalAttention takes
        # (B, N, H, D). Permute for RoPE, then transpose back at the call site.
        q = q.permute(0, 2, 1, 3)  # (B, H, N, D)
        k = k.permute(0, 2, 1, 3)
        v = v.permute(0, 2, 1, 3)
        if rotary_emb is not None:
            q = _apply_rotary_emb_bhnd(q, rotary_emb)
            k = _apply_rotary_emb_bhnd(k, rotary_emb)
        out = self.softmax_attn(
            q.transpose(1, 2).contiguous(),
            k.transpose(1, 2).contiguous(),
            v.transpose(1, 2).contiguous(),
        )  # (B, N, H, D)
        out = out.reshape(B, N, C)

        # Compute the gates anyway -- they are needed by the cam branch and
        # also exist in the softmax variant's state dict.
        beta, decay = _compute_frame_gates(
            x, HW, self.heads, self.beta_proj, self.gate_proj, self.dt_bias, self.A_log,
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

        q = self.q_proj_cam(x).reshape(B, N, self.heads, self.dim)
        k = self.k_proj_cam(x).reshape(B, N, self.heads, self.dim)
        v = self.v_proj_cam(x).reshape(B, N, self.heads, self.dim)

        # Short conv on K only (k_conv_only)
        if self.conv_k_cam is not None:
            k = self._temporal_short_conv(k.reshape(B, N, C), self.conv_k_cam, HW, bidirectional=True)
            k = k.reshape(B, N, self.heads, self.dim)

        q = self.q_norm_cam(q.reshape(B, N, C)).reshape(B, N, self.heads, self.dim)
        k = self.k_norm_cam(k.reshape(B, N, C)).reshape(B, N, self.heads, self.dim)

        q = F.relu(q)
        k = F.relu(k)
        k_scale = (self.dim ** -0.5) * (S ** -0.5)
        k = k * k_scale

        # Move to (B, H, N, D) for UCPE application then back to (B, H, D, N)
        q_bhnd = q.permute(0, 2, 1, 3)
        k_bhnd = k.permute(0, 2, 1, 3)
        v_bhnd = v.permute(0, 2, 1, 3)

        q_proj = apply_q(q_bhnd)
        k_proj = apply_kv(k_bhnd)
        v_proj = v_bhnd  # value is untransformed

        q_dn = q_proj.permute(0, 1, 3, 2)
        k_dn = k_proj.permute(0, 1, 3, 2)
        v_dn = v_proj.permute(0, 1, 3, 2)

        dtype = q_dn.dtype
        out = _gdn_scan_bidirectional(
            q_dn.float(), k_dn.float(), v_dn.float(),
            q_dn.float(), k_dn.float(),  # UCPE already applied (no extra RoPE on numerator)
            beta.float(), decay.float(),
            HW=HW, eps=self.eps,
        ).to(dtype)
        # apply inverse UCPE projection on output
        out_bhnd = out.permute(0, 1, 3, 2)
        out_bhnd = apply_o(out_bhnd)
        return out_bhnd.permute(0, 2, 1, 3).reshape(B, N, C)

    # ------------------------------------------------------------------ #

    def forward(
        self,
        x: torch.Tensor,
        HW: Tuple[int, int, int],
        rotary_emb: Optional[torch.Tensor] = None,
        prope_fns: Optional[Tuple[Callable, Callable, Callable]] = None,
    ) -> torch.Tensor:
        if self.softmax_main:
            main_raw, beta, decay = self._main_branch_softmax(x, HW, rotary_emb)
        else:
            main_raw, beta, decay = self._main_branch_gdn(x, HW, rotary_emb)

        if prope_fns is not None:
            apply_q, apply_kv, apply_o = prope_fns
            cam_raw = self._cam_branch(x, HW, apply_q, apply_kv, apply_o, beta, decay)
            combined = main_raw + self.out_proj_cam(cam_raw)
        else:
            combined = main_raw

        # Shared output gate + shared output projection
        gate = F.silu(self.output_gate(x).float())
        combined = combined * gate.to(combined.dtype)
        return self.proj(combined.to(self.proj.weight.dtype))


# ---------------------------------------------------------------------------
# Cross-attention (text conditioning) -- upstream MultiHeadCrossAttention,
# stored as ``cross_attn.{q_linear, kv_linear, proj, q_norm, k_norm}``.
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
        # Cross-attention dispatched through SGLang's pluggable backend.
        # The padding-mask path falls back to SDPA internally; the unmasked
        # path can pick FA3 / FlashInfer / etc.
        self.attn = LocalAttention(
            num_heads=num_heads,
            head_size=self.head_dim,
        )

    def forward(
        self,
        x: torch.Tensor,                       # (B, N, D)
        cond: torch.Tensor,                    # (B, L, D)
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, N, D = x.shape

        q = self.q_linear(x)
        kv = self.kv_linear(cond).view(B, -1, 2, D)
        k, v = kv.unbind(2)
        # LocalAttention takes (B, N, H, D); skip the legacy BHND transpose.
        q = self.q_norm(q).view(B, N, self.num_heads, self.head_dim)
        k = self.k_norm(k).view(B, -1, self.num_heads, self.head_dim)
        v = v.view(B, -1, self.num_heads, self.head_dim)

        attn_mask = mask.bool() if mask is not None else None
        out = self.attn(q, k, v, attn_mask=attn_mask)  # (B, N, H, D)
        out = out.reshape(B, N, D)
        return self.proj(out)


# ---------------------------------------------------------------------------
# Transformer block
# ---------------------------------------------------------------------------


class SanaWMBlock(nn.Module):
    """One transformer block of SANA-WM."""

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        head_dim: int,
        mlp_ratio: float,
        t_kernel_size: int,
        qk_norm: bool,
        cross_norm: bool,
        conv_kernel_size: int,
        k_conv_only: bool,
        softmax_main: bool,
        use_chunk_plucker_post_attn: bool,
    ) -> None:
        super().__init__()
        self.softmax_main = softmax_main

        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)

        self.attn = BidirectionalGDNUCPESinglePathLiteLA(
            in_dim=hidden_size,
            heads=num_heads,
            head_dim=head_dim,
            qk_norm=qk_norm,
            conv_kernel_size=conv_kernel_size,
            k_conv_only=k_conv_only,
            softmax_main=softmax_main,
        )

        self.cross_attn = MultiHeadCrossAttention(
            d_model=hidden_size,
            num_heads=num_heads,
            qk_norm=cross_norm,
        )

        self.mlp = GLUMBConvTemp(
            in_features=hidden_size,
            hidden_features=int(hidden_size * mlp_ratio),
            t_kernel_size=t_kernel_size,
        )

        self.scale_shift_table = nn.Parameter(torch.randn(6, hidden_size) / hidden_size**0.5)

        if use_chunk_plucker_post_attn:
            self.plucker_proj = nn.Linear(hidden_size, hidden_size, bias=True)
            nn.init.zeros_(self.plucker_proj.weight)
            nn.init.zeros_(self.plucker_proj.bias)
        else:
            self.plucker_proj = None

    @staticmethod
    def _modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
        return x * (1 + scale) + shift

    @staticmethod
    def _reshape_framewise_modulation(
        x: torch.Tensor,
        num_frames: int,
    ) -> tuple[torch.Tensor, int]:
        B, N, C = x.shape
        tokens_per_frame = N // num_frames
        return x.reshape(B, num_frames, tokens_per_frame, C), tokens_per_frame

    def forward(
        self,
        x: torch.Tensor,                          # (B, N, D)
        y: torch.Tensor,                          # (B, L, D) text embeds
        t: torch.Tensor,                          # (B, 6*D) AdaLN-single
        HW: Tuple[int, int, int],
        rotary_emb: Optional[torch.Tensor],
        prope_fns: Optional[Tuple[Callable, Callable, Callable]],
        plucker_emb: Optional[torch.Tensor],
        mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        B = x.shape[0]
        if t.dim() == 2:
            num_frames = None
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
                self.scale_shift_table[None] + t.reshape(B, 6, -1)
            ).chunk(6, dim=1)
        else:
            num_frames = t.reshape(B, -1, 6, t.shape[-1] // 6).shape[1]
            t = t.reshape(B, num_frames, 6, -1)
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
                self.scale_shift_table[None, None, :, :] + t
            ).chunk(6, dim=2)

        # Self-attention with UCPE camera branch
        if num_frames is None:
            x_in = self._modulate(self.norm1(x), shift_msa, scale_msa)
        else:
            x_norm, tokens_per_frame = self._reshape_framewise_modulation(
                self.norm1(x), num_frames
            )
            x_in = self._modulate(x_norm, shift_msa, scale_msa).reshape_as(x)
        attn_out = self.attn(x_in, HW=HW, rotary_emb=rotary_emb, prope_fns=prope_fns)
        if num_frames is None:
            x = x + gate_msa * attn_out
        else:
            attn_out = attn_out.reshape(B, num_frames, tokens_per_frame, -1)
            x = x + (gate_msa * attn_out).reshape_as(x)

        # Plücker post-attn injection (zero-init linear)
        if self.plucker_proj is not None and plucker_emb is not None:
            x = x + self.plucker_proj(plucker_emb)

        # Cross-attention
        x = x + self.cross_attn(x, y, mask=mask)

        # FFN
        if num_frames is None:
            x_in = self._modulate(self.norm2(x), shift_mlp, scale_mlp)
            x = x + gate_mlp * self.mlp(x_in, HW=HW)
        else:
            x_norm, tokens_per_frame = self._reshape_framewise_modulation(
                self.norm2(x), num_frames
            )
            x_in = self._modulate(x_norm, shift_mlp, scale_mlp).reshape_as(x)
            mlp_out = self.mlp(x_in, HW=HW).reshape(
                B, num_frames, tokens_per_frame, -1
            )
            x = x + (gate_mlp * mlp_out).reshape_as(x)
        return x


# ---------------------------------------------------------------------------
# Top-level model
# ---------------------------------------------------------------------------


class SanaWMTransformer3DModel(CachableDiT, LayerwiseOffloadableModuleMixin):
    """SANA-WM 2.6B TI2V world model.

    Forward inputs:
        hidden_states:           (B, C, T, H, W)        128-ch LTX-2 latent
        encoder_hidden_states:   (B, L, 2304)           Gemma-2 embeddings
        timestep:                (B,)
        encoder_attention_mask:  (B, L) optional bool
        camera_conditions:       (B, F_orig, 20)        16 c2w + (fx,fy,cx,cy)
        chunk_plucker:           (B, 48, T, H, W)       optional, computed
                                                        from camera_conditions
                                                        if absent.

    Returns: ``(B, C, T, H, W)`` predicted velocity / noise.
    """

    _fsdp_shard_conditions = SanaWMConfig()._fsdp_shard_conditions
    _compile_conditions = SanaWMConfig()._compile_conditions
    _supported_attention_backends = SanaWMConfig()._supported_attention_backends
    param_names_mapping = SanaWMConfig().param_names_mapping
    reverse_param_names_mapping = SanaWMConfig().reverse_param_names_mapping
    lora_param_names_mapping: dict = {}

    def __init__(self, config: SanaWMConfig, hf_config=None, **kwargs) -> None:
        super().__init__(config, hf_config=hf_config or {}, **kwargs)
        arch = config.arch_config

        self.patch_size = (arch.patch_size_t, arch.patch_size, arch.patch_size)
        self.inner_dim = arch.num_attention_heads * arch.attention_head_dim
        self.hidden_size = self.inner_dim
        self.num_attention_heads = arch.num_attention_heads
        self.attention_head_dim = arch.attention_head_dim
        self.out_channels = arch.out_channels
        self.num_channels_latents = arch.num_channels_latents
        self.vae_temporal_stride = arch.vae_temporal_stride

        # --- Embedders (upstream names: x_embedder, t_embedder, t_block,
        # y_embedder, attention_y_norm, raymap_embedder, plucker_embedder) ---
        self.x_embedder = PatchEmbedMS3D(
            self.patch_size, arch.in_channels, self.inner_dim, bias=True,
        )

        self.t_embedder = TimestepEmbedder(self.inner_dim, frequency_embedding_size=256)
        self.t_block = nn.Sequential(
            nn.SiLU(),
            nn.Linear(self.inner_dim, 6 * self.inner_dim, bias=True),
        )

        self.y_embedder = CaptionEmbedder(
            in_channels=arch.caption_channels,
            hidden_size=self.inner_dim,
            token_num=arch.model_max_length,
        )
        self.attention_y_norm = _RMSNorm(self.inner_dim, eps=1e-6)

        # 3-channel raymap embedder -- kept for state_dict compatibility but
        # only invoked when ``use_chunk_plucker_post_attn`` is False.
        # When ``True`` (the case for the released checkpoint) the absmap
        # path is skipped entirely.
        self.raymap_embedder = PatchEmbedMS3D(
            self.patch_size, 3, self.inner_dim, bias=True,
        )
        # 48-channel plucker embedder (chunk-packed)
        if arch.use_chunk_plucker_post_attn or arch.use_chunk_plucker_input:
            self.plucker_embedder = PatchEmbedMS3D(
                self.patch_size, arch.chunk_plucker_channels, self.inner_dim, bias=True,
            )
            nn.init.zeros_(self.plucker_embedder.proj.weight)
            nn.init.zeros_(self.plucker_embedder.proj.bias)
        else:
            self.plucker_embedder = None
        self.use_chunk_plucker_post_attn = arch.use_chunk_plucker_post_attn
        self.use_chunk_plucker_input = arch.use_chunk_plucker_input

        # --- RoPE ---
        self.rope = WanRotaryPosEmbed(
            attention_head_dim=arch.linear_head_dim,
            patch_size=self.patch_size,
            max_seq_len=1024,
        )

        # --- Transformer blocks ---
        depth = arch.num_layers
        self.softmax_every_n = arch.softmax_every_n
        softmax_idx = set(
            i for i in range(depth) if arch.softmax_every_n > 0 and (i + 1) % arch.softmax_every_n == 0
        )
        self.softmax_block_indices = tuple(sorted(softmax_idx))

        self.blocks = nn.ModuleList([
            SanaWMBlock(
                hidden_size=self.inner_dim,
                num_heads=arch.num_attention_heads,
                head_dim=arch.linear_head_dim,
                mlp_ratio=arch.mlp_ratio,
                t_kernel_size=arch.t_kernel_size,
                qk_norm=arch.qk_norm,
                cross_norm=arch.cross_norm,
                conv_kernel_size=arch.conv_kernel_size,
                k_conv_only=arch.k_conv_only,
                softmax_main=(i in softmax_idx),
                use_chunk_plucker_post_attn=(
                    arch.use_chunk_plucker_post_attn
                    and (arch.chunk_plucker_post_attn_blocks < 0
                         or i < arch.chunk_plucker_post_attn_blocks)
                ),
            )
            for i in range(depth)
        ])

        self.final_layer = T2IFinalLayer(self.inner_dim, self.patch_size, self.out_channels)

        # Cache RoPE freqs per shape -- avoids recomputation across denoising
        # steps with constant latent shapes.
        self._freqs_cache: dict = {}

        # FSDP shard targets
        self.layer_names = ["blocks"]

    def post_load_weights(self) -> None:
        # FSDP loader initializes the model on meta and only materializes
        # tensors that appear in the checkpoint. WanRotaryPosEmbed._freqs is a
        # derived, non-persistent constant, so recompute it deterministically.
        for module in self.modules():
            if isinstance(module, WanRotaryPosEmbed):
                if module._freqs.is_meta:
                    module._init_freqs_buffer()

    # ------------------------------------------------------------------ #
    # Forward
    # ------------------------------------------------------------------ #

    def _get_freqs(self, T: int, H: int, W: int, device: torch.device) -> torch.Tensor:
        key = (T, H, W, str(device))
        if key not in self._freqs_cache:
            self._freqs_cache[key] = self.rope((T, H, W), device)
        return self._freqs_cache[key]

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        timestep: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        camera_conditions: Optional[torch.Tensor] = None,
        chunk_plucker: Optional[torch.Tensor] = None,
        guidance: Optional[torch.Tensor] = None,  # kept for compat
        **kwargs,
    ) -> torch.Tensor:
        if encoder_hidden_states is None:
            raise ValueError("SANA-WM forward requires encoder_hidden_states.")
        if timestep is None:
            raise ValueError("SANA-WM forward requires timestep.")

        B, C, T_raw, H_raw, W_raw = hidden_states.shape
        p_t, p_h, p_w = self.patch_size
        T = T_raw // p_t
        H = H_raw // p_h
        W = W_raw // p_w

        # --- 1. Patch embed: (B, C, T, H, W) -> (B, T*H*W, D) ---
        x = self.x_embedder(hidden_states.to(dtype=self.x_embedder.proj.weight.dtype))

        # --- 2. Timestep AdaLN-single ---
        # SANA-WM's LTX sampler passes per-frame timesteps shaped (B, 1, T)
        # so the clean first-frame condition can stay at timestep 0 while the
        # remaining latent frames denoise. Keep the scalar path for generic
        # scheduler compatibility.
        if timestep.dim() == 1:
            t_emb = self.t_embedder(timestep)          # (B, D)
            t6 = self.t_block(t_emb)                   # (B, 6D)
        else:
            timestep_shape = tuple(timestep.shape)
            t_flat = self.t_embedder(timestep.flatten())
            t6_flat = self.t_block(t_flat)
            t_emb = t_flat.unflatten(0, timestep_shape)
            t6 = t6_flat.unflatten(0, timestep_shape)

        # --- 3. Caption projection + y_norm ---
        if isinstance(encoder_attention_mask, (list, tuple)):
            encoder_attention_mask = encoder_attention_mask[0]
        y = encoder_hidden_states
        if y.dim() == 3:
            y = y.unsqueeze(1)
        y = self.y_embedder(y).squeeze(1)              # (B, L, D)
        if y.shape[0] != B:
            y = y.expand(B, -1, -1).contiguous()
        y = self.attention_y_norm(y)
        if encoder_attention_mask is not None and encoder_attention_mask.shape[0] != B:
            encoder_attention_mask = encoder_attention_mask.expand(B, -1).contiguous()

        # --- 4. RoPE ---
        freqs = self._get_freqs(T, H, W, x.device)

        # --- 5. Camera conditioning: compute UCPE prope_fns + Plücker ---
        prope_fns = None
        if camera_conditions is not None:
            head_dim = self.attention_head_dim
            raymats = process_camera_conditions_ucpe(
                camera_conditions, HW=(T, H, W), patch_size=self.patch_size,
            )
            raymats_flat = raymats.reshape(B, -1, 4, 4)
            prope_fns = _build_ucpe_apply_fns(head_dim, raymats_flat, freqs)

        # Plücker post-attn embedding (shared across all blocks)
        plucker_emb = None
        if (
            self.use_chunk_plucker_post_attn
            and chunk_plucker is not None
            and self.plucker_embedder is not None
        ):
            plucker_emb = self.plucker_embedder(
                chunk_plucker.to(self.plucker_embedder.proj.weight.dtype),
            )  # (B, T*H*W, D)
            if plucker_emb.shape[1] != x.shape[1]:
                raise ValueError(
                    f"plucker_emb token count {plucker_emb.shape[1]} != latent token count {x.shape[1]}; "
                    f"expected chunk_plucker shape (B, 48, T={T}, H={H}, W={W})."
                )

        if (
            self.use_chunk_plucker_input
            and chunk_plucker is not None
            and self.plucker_embedder is not None
        ):
            x = x + self.plucker_embedder(
                chunk_plucker.to(self.plucker_embedder.proj.weight.dtype)
            )

        # --- 6. Transformer blocks ---
        HW = (T, H, W)
        for block in self.blocks:
            x = block(
                x,
                y=y,
                t=t6,
                HW=HW,
                rotary_emb=freqs,
                prope_fns=prope_fns,
                plucker_emb=plucker_emb,
                mask=encoder_attention_mask,
            )

        # --- 7. Final layer ---
        x = self.final_layer(x, t_emb)                 # (B, N, p_t*p_h*p_w*C_out)

        # --- 8. Un-patch ---
        x = x.reshape(B, T, H, W, p_t, p_h, p_w, self.out_channels)
        x = x.permute(0, 7, 1, 4, 2, 5, 3, 6).contiguous()
        x = x.reshape(B, self.out_channels, T * p_t, H * p_h, W * p_w)
        return x


EntryClass = SanaWMTransformer3DModel
