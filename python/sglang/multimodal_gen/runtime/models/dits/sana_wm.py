# SPDX-License-Identifier: Apache-2.0
#
# SANA-WM (World Model) DiT implementation.
#
# Key architectural differences vs SANA T2I (sana.py):
#   1. 5D video latent (B, C, T, H, W) instead of 4D image (B, C, H, W)
#   2. Hybrid GDN/Softmax self-attention:
#      - 15 GDN blocks: Frame-wise linear recurrent (Gated Delta Network),
#        state matrix dim = (B, H, D, D), memory-constant in num_frames
#      - 5 Softmax blocks at indices {3,7,11,15,19}: full spatial-temporal attn
#   3. Dual-branch camera conditioning:
#      - Coarse: Camera branch with UCPE (Ray-Local Unified Camera Positional Encoding)
#        Geometric head channels rotated by ray-local D_i transform; zero-init output proj
#      - Fine: Post-attention Plücker Raymap Mixing
#        48-ch packed Plücker (8 orig frames × 6D) embedded via zero-init 3D conv
#   4. GLUMBConvTemp: temporal extension of GLUMBConv, depth-wise conv is 3D (t_kernel=3)
#   5. wan_rope 3D rotary position embeddings (spatial h, w + temporal t)
#
# Reference: https://arxiv.org/abs/2605.15178

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers.models.embeddings import PixArtAlphaTextProjection, TimestepEmbedding

from sglang.multimodal_gen.configs.models.dits.sana_wm import SanaWMConfig
from sglang.multimodal_gen.runtime.layers.attention.layer import LocalAttention
from sglang.multimodal_gen.runtime.layers.layernorm import LayerNormScaleShift, RMSNorm
from sglang.multimodal_gen.runtime.layers.visual_embedding import Timesteps
from sglang.multimodal_gen.runtime.managers.memory_managers.layerwise_offload import (
    LayerwiseOffloadableModuleMixin,
)
from sglang.multimodal_gen.runtime.models.dits.base import CachableDiT
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

logger = init_logger(__name__)


# ---------------------------------------------------------------------------
# Time embedding (reused from SANA T2I)
# ---------------------------------------------------------------------------


class SanaWMCombinedTimestepEmbeddings(nn.Module):
    def __init__(self, embedding_dim: int):
        super().__init__()
        self.time_proj = Timesteps(
            num_channels=256, flip_sin_to_cos=True, downscale_freq_shift=0
        )
        self.timestep_embedder = TimestepEmbedding(
            in_channels=256, time_embed_dim=embedding_dim
        )

    def forward(self, timestep: torch.Tensor, hidden_dtype=None):
        ts_proj = self.time_proj(timestep)
        if hidden_dtype is not None:
            ts_proj = ts_proj.to(dtype=hidden_dtype)
        return self.timestep_embedder(ts_proj)


class SanaWMAdaLayerNormSingle(nn.Module):
    def __init__(self, embedding_dim: int):
        super().__init__()
        self.emb = SanaWMCombinedTimestepEmbeddings(embedding_dim)
        self.silu = nn.SiLU()
        self.linear = nn.Linear(embedding_dim, 6 * embedding_dim, bias=True)

    def forward(self, timestep: torch.Tensor, hidden_dtype=None):
        embedded = self.emb(timestep, hidden_dtype=hidden_dtype)
        out = self.linear(self.silu(embedded))
        return out, embedded


# ---------------------------------------------------------------------------
# 3D Rotary Positional Embeddings (wan_rope style)
# ---------------------------------------------------------------------------


def precompute_freqs_cis_3d(
    head_dim: int,
    max_t: int,
    max_h: int,
    max_w: int,
    theta: float = 10000.0,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """
    Pre-compute 3D (temporal, height, width) RoPE frequencies.
    head_dim is split into 3 parts: dim_t, dim_h, dim_w (each head_dim // 3, padded if needed).
    Returns complex tensor of shape (max_t * max_h * max_w, head_dim // 2).
    """
    dim_t = head_dim // 3
    dim_h = head_dim // 3
    dim_w = head_dim - dim_t - dim_h  # absorb remainder

    def _1d_freqs(dim, max_pos, base=theta):
        freqs = 1.0 / (base ** (torch.arange(0, dim, 2, device=device).float() / dim))
        t = torch.arange(max_pos, device=device).float()
        freqs = torch.outer(t, freqs)  # (max_pos, dim/2)
        freqs = torch.polar(torch.ones_like(freqs), freqs)  # complex
        return freqs  # (max_pos, dim/2)

    ft = _1d_freqs(dim_t, max_t)  # (T, dim_t/2)
    fh = _1d_freqs(dim_h, max_h)  # (H, dim_h/2)
    fw = _1d_freqs(dim_w, max_w)  # (W, dim_w/2)

    # Broadcast to (T, H, W, total_complex_dim) then flatten (T*H*W, total)
    T, H, W = max_t, max_h, max_w
    ft_ = ft.unsqueeze(1).unsqueeze(1).expand(T, H, W, -1)  # (T,H,W,dim_t/2)
    fh_ = fh.unsqueeze(0).unsqueeze(2).expand(T, H, W, -1)  # (T,H,W,dim_h/2)
    fw_ = fw.unsqueeze(0).unsqueeze(0).expand(T, H, W, -1)  # (T,H,W,dim_w/2)

    freqs = torch.cat([ft_, fh_, fw_], dim=-1)  # (T,H,W, head_dim/2)
    return freqs.reshape(T * H * W, -1)  # (T*H*W, head_dim/2)


def apply_rope(x: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
    """
    Apply rotary embeddings to query or key tensor.
    x: (B, H, S, D) where S = T*H_sp*W_sp, D = head_dim
    freqs: (S, D//2) complex
    Returns same shape as x.
    """
    B, num_heads, S, D = x.shape
    x_c = torch.view_as_complex(x.float().reshape(B, num_heads, S, D // 2, 2))
    x_rot = x_c * freqs[None, None, :, :]  # broadcast over B and heads
    return torch.view_as_real(x_rot).reshape(B, num_heads, S, D)


# ---------------------------------------------------------------------------
# Frame-wise GDN Attention (Gated Delta Network, frame-level recurrent scan)
# ---------------------------------------------------------------------------


class FrameWiseGDNAttention(nn.Module):
    """
    Frame-wise linear recurrent attention (GDN) for video tokens.

    Each video latent has shape (B, T, H_sp, W_sp, D) in this module.
    The GDN scan is performed *per-frame* in temporal order (or both directions
    for bidirectional). The key innovation: the accumulated state matrix is
    (B, H, D, D) — constant in T — enabling O(D^2) per-step memory.

    Key scaling: 1/sqrt(D * S_frame) applied to Keys to prevent spatial token
    accumulation blowup (S_frame = H_sp * W_sp).

    Ref: https://arxiv.org/abs/2605.15178, Section 3.1
    """

    def __init__(
        self,
        query_dim: int,
        num_heads: int,
        head_dim: int,
        bidirectional: bool = True,
        bias: bool = False,
    ):
        super().__init__()
        inner_dim = num_heads * head_dim
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.bidirectional = bidirectional

        self.to_q = nn.Linear(query_dim, inner_dim, bias=bias)
        self.to_k = nn.Linear(query_dim, inner_dim, bias=bias)
        self.to_v = nn.Linear(query_dim, inner_dim, bias=bias)
        self.to_out = nn.ModuleList(
            [nn.Linear(inner_dim, query_dim, bias=True), nn.Identity()]
        )

        # Gate parameter (learned per-head beta that controls delta update strength)
        # Shape: (num_heads, head_dim) — shared across spatial tokens, per-head
        self.gate = nn.Parameter(torch.ones(num_heads, head_dim))

    def _gdn_forward_scan(
        self,
        q: torch.Tensor,   # (B, H, T, S, D)
        k: torch.Tensor,   # (B, H, T, S, D)
        v: torch.Tensor,   # (B, H, T, S, D)
        key_scale: float,
    ) -> torch.Tensor:
        """
        Causal (forward) GDN scan over T frames.
        State: S_t = S_{t-1} + gate * (k_t^T @ v_t) where k_t is (B,H,S,D), v_t is (B,H,S,D)
        Output per frame: out_t = q_t @ S_t / (q_t @ k_sum_t)
        """
        B, H, T, S, D = q.shape
        device = q.device
        dtype = q.dtype

        # ReLU activation (as in original SANA linear attention)
        q = F.relu(q)
        k = F.relu(k) * key_scale

        # Gate broadcast: (1, H, 1, 1, D) → (B, H, T, S, D)
        gate = self.gate.unsqueeze(0).unsqueeze(2).unsqueeze(2)  # (1,H,1,1,D)

        # Accumulate kv state (B, H, D, D) and key sum (B, H, 1, D)
        state = torch.zeros(B, H, D, D, device=device, dtype=dtype)
        key_sum = torch.zeros(B, H, 1, D, device=device, dtype=dtype)
        outputs = []

        for t in range(T):
            kt = k[:, :, t]   # (B, H, S, D)
            vt = v[:, :, t]   # (B, H, S, D)
            qt = q[:, :, t]   # (B, H, S, D)

            # Delta update: kv_t = k_t^T @ v_t = (B, H, D, D)
            kv_t = torch.einsum("bhsd,bhse->bhde", kt, vt)
            state = state + gate * kv_t   # element-wise gate on (B,H,D,D)

            # Key sum accumulation for normalization
            key_sum = key_sum + kt.mean(dim=2, keepdim=True)  # (B, H, 1, D)

            # Output: q_t @ state / (q_t @ key_sum + eps)
            out_t = torch.einsum("bhsd,bhde->bhse", qt, state)  # (B, H, S, D)
            norm = (qt * key_sum).sum(dim=-1, keepdim=True).clamp(min=1e-6)  # (B,H,S,1)
            out_t = out_t / norm
            outputs.append(out_t)

        return torch.stack(outputs, dim=2)  # (B, H, T, S, D)

    def _gdn_backward_scan(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        key_scale: float,
    ) -> torch.Tensor:
        """Backward (reverse temporal) GDN scan."""
        B, H, T, S, D = q.shape
        # Flip along T, scan forward, then flip output back
        q_r = q.flip(2)
        k_r = k.flip(2)
        v_r = v.flip(2)
        out_r = self._gdn_forward_scan(q_r, k_r, v_r, key_scale)
        return out_r.flip(2)

    def forward(
        self,
        hidden_states: torch.Tensor,  # (B, T*H_sp*W_sp, D)
        num_frames: int,
        sp_h: int,
        sp_w: int,
        freqs: Optional[torch.Tensor] = None,  # (T*H*W, D//2) complex, for RoPE
    ) -> torch.Tensor:
        B, S_total, D = hidden_states.shape
        S_frame = sp_h * sp_w
        T = num_frames

        q = self.to_q(hidden_states)
        k = self.to_k(hidden_states)
        v = self.to_v(hidden_states)

        # Reshape: (B, T*S, H*D) → (B, H, T, S, D)
        def reshape_qkv(x):
            x = x.view(B, T, S_frame, self.num_heads, self.head_dim)
            return x.permute(0, 3, 1, 2, 4)  # (B, H, T, S, D)

        q, k, v = reshape_qkv(q), reshape_qkv(k), reshape_qkv(v)

        # Apply wan_rope if provided (reshape to (B, H, T*S, D) for rope, then back)
        if freqs is not None:
            q_flat = q.reshape(B, self.num_heads, T * S_frame, self.head_dim)
            k_flat = k.reshape(B, self.num_heads, T * S_frame, self.head_dim)
            q_flat = apply_rope(q_flat, freqs)
            k_flat = apply_rope(k_flat, freqs)
            q = q_flat.reshape(B, self.num_heads, T, S_frame, self.head_dim)
            k = k_flat.reshape(B, self.num_heads, T, S_frame, self.head_dim)

        # Key scaling: 1/sqrt(D * S_frame) to prevent spatial accumulation blowup
        key_scale = 1.0 / math.sqrt(self.head_dim * S_frame)

        out_fwd = self._gdn_forward_scan(q, k, v, key_scale)  # (B, H, T, S, D)

        if self.bidirectional:
            out_bwd = self._gdn_backward_scan(q, k, v, key_scale)  # (B, H, T, S, D)
            hidden_states_out = (out_fwd + out_bwd) * 0.5
        else:
            hidden_states_out = out_fwd

        # Reshape back: (B, H, T, S, D) → (B, T*S, H*D)
        hidden_states_out = hidden_states_out.permute(0, 2, 3, 1, 4)  # (B, T, S, H, D)
        hidden_states_out = hidden_states_out.reshape(B, T * S_frame, self.num_heads * self.head_dim)

        return self.to_out[0](hidden_states_out)


# ---------------------------------------------------------------------------
# Full Softmax Attention (used at blocks {3, 7, 11, 15, 19})
# ---------------------------------------------------------------------------


class SoftmaxTemporalAttention(nn.Module):
    """Full spatial-temporal softmax attention for the designated blocks (e.g. {3,7,11,15,19}).
    Uses LocalAttention, which picks the best available backend (FA > SageAttn > SDPA) at init.
    """

    def __init__(self, query_dim: int, num_heads: int, head_dim: int, bias: bool = False):
        super().__init__()
        inner_dim = num_heads * head_dim
        self.num_heads = num_heads
        self.head_dim = head_dim

        self.to_q = nn.Linear(query_dim, inner_dim, bias=bias)
        self.to_k = nn.Linear(query_dim, inner_dim, bias=bias)
        self.to_v = nn.Linear(query_dim, inner_dim, bias=bias)
        self.to_out = nn.ModuleList(
            [nn.Linear(inner_dim, query_dim, bias=True), nn.Identity()]
        )
        self.attn = LocalAttention(num_heads=num_heads, head_size=head_dim, causal=False)

    def forward(
        self,
        hidden_states: torch.Tensor,   # (B, T*H*W, D)
        freqs: Optional[torch.Tensor] = None,   # (T*H*W, D//2) complex
    ) -> torch.Tensor:
        B, S, _ = hidden_states.shape

        q = self.to_q(hidden_states).reshape(B, S, self.num_heads, self.head_dim)
        k = self.to_k(hidden_states).reshape(B, S, self.num_heads, self.head_dim)
        v = self.to_v(hidden_states).reshape(B, S, self.num_heads, self.head_dim)

        if freqs is not None:
            q = apply_rope(q.transpose(1, 2), freqs).transpose(1, 2)
            k = apply_rope(k.transpose(1, 2), freqs).transpose(1, 2)

        # LocalAttention expects [B, S, H, D] and returns [B, S, H, D]
        hidden_states = self.attn(q, k, v).reshape(B, S, -1)
        return self.to_out[0](hidden_states)


# ---------------------------------------------------------------------------
# Cross-attention for text conditioning (unchanged from SANA T2I)
# ---------------------------------------------------------------------------


class SanaWMCrossAttention(nn.Module):
    def __init__(
        self,
        query_dim: int,
        cross_attention_dim: int,
        num_heads: int,
        head_dim: int,
        bias: bool = False,
    ):
        super().__init__()
        inner_dim = num_heads * head_dim
        self.num_heads = num_heads
        self.head_dim = head_dim

        self.to_q = nn.Linear(query_dim, inner_dim, bias=bias)
        self.to_k = nn.Linear(cross_attention_dim, inner_dim, bias=bias)
        self.to_v = nn.Linear(cross_attention_dim, inner_dim, bias=bias)
        self.to_out = nn.ModuleList(
            [nn.Linear(inner_dim, query_dim, bias=True), nn.Identity()]
        )
        # skip_sequence_parallel because KV (text tokens) is replicated across SP ranks
        self.attn = LocalAttention(num_heads=num_heads, head_size=head_dim, causal=False)

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        encoder_attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, S, _ = hidden_states.shape
        T_enc = encoder_hidden_states.shape[1]

        q = self.to_q(hidden_states).reshape(B, S, self.num_heads, self.head_dim)
        k = self.to_k(encoder_hidden_states).reshape(B, T_enc, self.num_heads, self.head_dim)
        v = self.to_v(encoder_hidden_states).reshape(B, T_enc, self.num_heads, self.head_dim)

        # LocalAttention handles the attn_mask path via SDPA when mask is non-None
        hidden_states = self.attn(q, k, v, attn_mask=encoder_attention_mask).reshape(B, S, -1)
        return self.to_out[0](hidden_states)


# ---------------------------------------------------------------------------
# GLUMBConvTemp (temporal extension of SANA's GLUMBConv)
# ---------------------------------------------------------------------------


class GLUMBConvTemp(nn.Module):
    """
    GLUMBConv extended to video: the depth-wise convolution is 3D with a
    temporal kernel (t_kernel_size, 3, 3) instead of purely spatial (3, 3).

    Input: (B, T*H*W, C) tokens — internally reshaped to (B, C, T, H, W).
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        expand_ratio: float = 3.0,
        t_kernel_size: int = 3,
    ):
        super().__init__()
        hidden = int(expand_ratio * in_channels)
        self.nonlinearity = nn.SiLU()

        self.conv_inverted = nn.Conv3d(in_channels, hidden * 2, kernel_size=1)
        t_pad = t_kernel_size // 2
        self.conv_depth = nn.Conv3d(
            hidden * 2,
            hidden * 2,
            kernel_size=(t_kernel_size, 3, 3),
            padding=(t_pad, 1, 1),
            groups=hidden * 2,  # depth-wise
        )
        self.conv_point = nn.Conv3d(hidden, out_channels, kernel_size=1, bias=False)

    def forward(
        self,
        hidden_states: torch.Tensor,  # (B, T*H*W, C)
        num_frames: int,
        sp_h: int,
        sp_w: int,
    ) -> torch.Tensor:
        B, S, C = hidden_states.shape

        # Reshape to (B, C, T, H, W)
        x = hidden_states.view(B, num_frames, sp_h, sp_w, C).permute(0, 4, 1, 2, 3)

        x = self.conv_inverted(x)
        x = self.nonlinearity(x)
        x = self.conv_depth(x)
        x, gate = x.chunk(2, dim=1)  # split hidden*2 → two halves
        x = x * self.nonlinearity(gate)
        x = self.conv_point(x)

        # Reshape back to (B, T*H*W, C)
        x = x.permute(0, 2, 3, 4, 1).reshape(B, S, -1)
        return x


# ---------------------------------------------------------------------------
# Camera Branch — UCPE (Coarse camera conditioning)
# ---------------------------------------------------------------------------


class UCPECameraAttention(nn.Module):
    """
    Camera branch GDN attention with UCPE (Ray-Local Unified Camera Positional Encoding).

    Each token in a frame has a ray-local rotation matrix D_i derived from its
    camera-to-world pose. Geometric head channels are rotated by D_i (Q × D_i^T,
    K × D_i^{-1}); remaining channels use standard spatiotemporal RoPE.

    The output is zero-initialized projected and added to the main branch output.
    """

    def __init__(
        self,
        query_dim: int,
        num_heads: int,
        head_dim: int,
        num_geometric_dims: int = 32,   # head channels dedicated to geometry rotation
        cam_attn_compress: int = 1,
        bidirectional: bool = True,
        bias: bool = False,
    ):
        super().__init__()
        inner_dim = num_heads * head_dim
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.num_geometric_dims = num_geometric_dims
        self.bidirectional = bidirectional
        self.cam_attn_compress = cam_attn_compress

        # Independent QKV for camera branch
        self.to_q = nn.Linear(query_dim, inner_dim, bias=bias)
        self.to_k = nn.Linear(query_dim, inner_dim, bias=bias)
        self.to_v = nn.Linear(query_dim, inner_dim, bias=bias)

        # Zero-initialized output projection (ensures cam branch = 0 at init)
        out_proj = nn.Linear(inner_dim, query_dim, bias=True)
        nn.init.zeros_(out_proj.weight)
        nn.init.zeros_(out_proj.bias)
        self.to_out = nn.ModuleList([out_proj, nn.Identity()])

        # GDN gate (shared with main branch via param tie — handled in SanaWMBlock)
        self.gate = nn.Parameter(torch.ones(num_heads, head_dim))

    @staticmethod
    def compute_ucpe_transforms(
        camera_to_world: torch.Tensor,  # (B, T, 4, 4)
        intrinsics: torch.Tensor,       # (B, 3, 3) or (B, T, 3, 3)
        sp_h: int,
        sp_w: int,
        latent_scale: int = 32,         # spatial VAE stride (32x)
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute per-token ray-local rotation matrices D_i for UCPE.

        For each latent token (t, h, w), we:
        1. Map (h, w) to pixel coords via latent_scale and intrinsics
        2. Unproject to unit ray direction d in world coords using R = R^c2w[:3,:3]
        3. Construct a local frame {e1, e2, e3} with e3 = d
        4. D_i is the 3x3 rotation matrix [e1|e2|e3]

        Returns:
            D_q: (B, T, H*W, 3, 3) — for Q rotation: Q' = Q * D_i
            D_k_inv: (B, T, H*W, 3, 3) — for K rotation: K' = K * D_i^{-1} = K * D_i^T
        """
        B, T = camera_to_world.shape[:2]
        device = camera_to_world.device
        dtype = camera_to_world.dtype

        # Handle per-frame or single intrinsics
        if intrinsics.dim() == 3:
            intrinsics = intrinsics.unsqueeze(1).expand(B, T, 3, 3)  # (B, T, 3, 3)

        # Build pixel grid (h_idx, w_idx) for each latent token
        hs = torch.arange(sp_h, device=device, dtype=dtype)
        ws = torch.arange(sp_w, device=device, dtype=dtype)
        grid_h, grid_w = torch.meshgrid(hs, ws, indexing="ij")
        # Map latent index → pixel center: pixel = (idx + 0.5) * latent_scale
        px = (grid_w.flatten() + 0.5) * latent_scale  # (H*W,)
        py = (grid_h.flatten() + 0.5) * latent_scale  # (H*W,)

        # Unproject to camera space ray (homogeneous): (H*W, 3)
        # d_cam = K^{-1} [px, py, 1]^T
        # We compute per-batch per-frame since intrinsics may vary
        S_frame = sp_h * sp_w
        ones = torch.ones(S_frame, device=device, dtype=dtype)
        pts_cam_h = torch.stack([px, py, ones], dim=-1)  # (S, 3)
        pts_cam_h = pts_cam_h.unsqueeze(0).unsqueeze(0).expand(B, T, S_frame, 3)  # (B,T,S,3)

        K = intrinsics  # (B, T, 3, 3)
        K_inv = torch.inverse(K.reshape(B * T, 3, 3)).reshape(B, T, 3, 3)
        # (B, T, S, 3) @ (B, T, 3, 3)^T → (B, T, S, 3)
        d_cam = torch.einsum("btij,btsj->btsi", K_inv, pts_cam_h)  # (B, T, S, 3)

        # Rotate to world space: d_world = R^{c2w} @ d_cam
        R = camera_to_world[:, :, :3, :3]  # (B, T, 3, 3)
        d_world = torch.einsum("btij,btsj->btsi", R, d_cam)  # (B, T, S, 3)
        d_world = F.normalize(d_world, dim=-1)  # unit rays

        # Build orthonormal local frame with e3 = d_world
        # Choose arbitrary vector not parallel to d_world for cross product
        up = torch.zeros_like(d_world)
        up[..., 1] = 1.0  # world-up (0,1,0)
        # Handle degenerate case: if d ≈ up, use (1,0,0)
        is_parallel = (d_world * up).sum(-1).abs() > 0.99
        up_alt = torch.zeros_like(d_world)
        up_alt[..., 0] = 1.0
        up = torch.where(is_parallel.unsqueeze(-1), up_alt, up)

        e1 = F.normalize(torch.cross(up, d_world, dim=-1), dim=-1)   # (B, T, S, 3)
        e2 = F.normalize(torch.cross(d_world, e1, dim=-1), dim=-1)   # (B, T, S, 3)
        e3 = d_world                                                   # (B, T, S, 3)

        # D_i = [e1 | e2 | e3] column matrix: (B, T, S, 3, 3)
        D = torch.stack([e1, e2, e3], dim=-1)  # (B, T, S, 3, 3)

        # For Q: multiply by D_i^T (same as D since columns are orthonormal → D^T = D^{-1})
        # For K: multiply by D_i^{-1} = D_i^T
        D_q = D.transpose(-1, -2)   # D^T, for Q: (B, T, S, 3, 3)
        D_k_inv = D.transpose(-1, -2)  # D^{-1} = D^T, for K: (B, T, S, 3, 3)

        return D_q, D_k_inv

    def _apply_ucpe(
        self,
        qk: torch.Tensor,      # (B, H, T*S, D)
        D: torch.Tensor,        # (B, T, S, 3, 3)
        is_query: bool,
        num_frames: int,
        sp_h: int,
        sp_w: int,
    ) -> torch.Tensor:
        """Rotate geometric head channels by ray-local D matrix."""
        B, H, TS, D_hd = qk.shape
        S_frame = sp_h * sp_w
        g = min(self.num_geometric_dims, D_hd)  # geometric channels

        # Split: geometric vs RoPE channels
        qk_geo = qk[..., :g]   # (B, H, T*S, g)
        qk_std = qk[..., g:]   # (B, H, T*S, D-g)

        # Reshape geometric part: (B, H, T, S, g) → apply 3x3 rotation on first 3 dims
        qk_geo = qk_geo.reshape(B, H, num_frames, S_frame, g)

        if g >= 3:
            # Rotate first 3 geometric dims by D (B, T, S, 3, 3)
            geo3 = qk_geo[..., :3]  # (B, H, T, S, 3)
            D_mat = D.unsqueeze(1).expand(B, H, num_frames, S_frame, 3, 3)
            geo3_rotated = torch.einsum("bhtsi,bhtsij->bhtsi", geo3, D_mat)
            qk_geo = torch.cat([geo3_rotated, qk_geo[..., 3:]], dim=-1)

        qk_geo = qk_geo.reshape(B, H, num_frames * S_frame, g)
        return torch.cat([qk_geo, qk_std], dim=-1)

    def forward(
        self,
        hidden_states: torch.Tensor,  # (B, T*S, D)
        D_q: Optional[torch.Tensor],  # (B, T, S, 3, 3) or None
        D_k_inv: Optional[torch.Tensor],
        num_frames: int,
        sp_h: int,
        sp_w: int,
        freqs: Optional[torch.Tensor] = None,  # RoPE
    ) -> torch.Tensor:
        """Camera-branch attention with UCPE conditioning."""
        B, S_total, D = hidden_states.shape
        S_frame = sp_h * sp_w

        q = self.to_q(hidden_states)
        k = self.to_k(hidden_states)
        v = self.to_v(hidden_states)

        def reshape(x):
            return x.view(B, S_total, self.num_heads, self.head_dim).transpose(1, 2)

        q, k, v = reshape(q), reshape(k), reshape(v)  # (B, H, S_total, D)

        # Apply UCPE to geometric head channels
        if D_q is not None and D_k_inv is not None:
            q = self._apply_ucpe(q, D_q, is_query=True, num_frames=num_frames, sp_h=sp_h, sp_w=sp_w)
            k = self._apply_ucpe(k, D_k_inv, is_query=False, num_frames=num_frames, sp_h=sp_h, sp_w=sp_w)

        # Apply RoPE to remaining channels
        if freqs is not None:
            q = apply_rope(q, freqs)
            k = apply_rope(k, freqs)

        # GDN scan on camera-branch QKV (using same gate logic)
        q = q.view(B, self.num_heads, num_frames, S_frame, self.head_dim)
        k = k.view(B, self.num_heads, num_frames, S_frame, self.head_dim)
        v = v.view(B, self.num_heads, num_frames, S_frame, self.head_dim)

        q = F.relu(q)
        k = F.relu(k)
        key_scale = 1.0 / math.sqrt(self.head_dim * S_frame)
        k = k * key_scale

        gate = self.gate.unsqueeze(0).unsqueeze(2).unsqueeze(2)

        state = torch.zeros(B, self.num_heads, self.head_dim, self.head_dim,
                            device=hidden_states.device, dtype=hidden_states.dtype)
        key_sum = torch.zeros(B, self.num_heads, 1, self.head_dim,
                              device=hidden_states.device, dtype=hidden_states.dtype)
        outputs_fwd = []

        for t in range(num_frames):
            kt, vt, qt = k[:, :, t], v[:, :, t], q[:, :, t]
            kv_t = torch.einsum("bhsd,bhse->bhde", kt, vt)
            state = state + gate * kv_t
            key_sum = key_sum + kt.mean(dim=2, keepdim=True)
            out_t = torch.einsum("bhsd,bhde->bhse", qt, state)
            norm = (qt * key_sum).sum(dim=-1, keepdim=True).clamp(min=1e-6)
            outputs_fwd.append(out_t / norm)

        out_fwd = torch.stack(outputs_fwd, dim=2)  # (B, H, T, S, D)

        if self.bidirectional:
            # Backward scan
            state = torch.zeros_like(state)
            key_sum = torch.zeros_like(key_sum)
            outputs_bwd = []
            for t in range(num_frames - 1, -1, -1):
                kt, vt, qt = k[:, :, t], v[:, :, t], q[:, :, t]
                kv_t = torch.einsum("bhsd,bhse->bhde", kt, vt)
                state = state + gate * kv_t
                key_sum = key_sum + kt.mean(dim=2, keepdim=True)
                out_t = torch.einsum("bhsd,bhde->bhse", qt, state)
                norm = (qt * key_sum).sum(dim=-1, keepdim=True).clamp(min=1e-6)
                outputs_bwd.append(out_t / norm)
            outputs_bwd.reverse()
            out_bwd = torch.stack(outputs_bwd, dim=2)
            out = (out_fwd + out_bwd) * 0.5
        else:
            out = out_fwd

        out = out.permute(0, 2, 3, 1, 4).reshape(B, S_total, -1)
        return self.to_out[0](out)


# ---------------------------------------------------------------------------
# Plücker Raymap Mixing (Fine camera conditioning, post-attention)
# ---------------------------------------------------------------------------


class PluckerRaymapMixer(nn.Module):
    """
    Post-attention Plücker Raymap Mixing.

    For each latent frame, we pack 8 original (pre-VAE) frames' Plücker coordinates
    (d, o×d) — 6D per frame — into a 48-channel tensor. This is embedded via a
    zero-initialized 3D patch embedder and added after each self-attention.

    Input:
        plucker: (B, T_latent, C_plucker, H_sp, W_sp) where C_plucker = 48
    Output:
        hidden_states: (B, T*H*W, D) addition delta
    """

    def __init__(
        self,
        plucker_channels: int,   # 48 = 8 orig_frames × 6D
        hidden_dim: int,
        t_kernel_size: int = 3,
    ):
        super().__init__()
        t_pad = t_kernel_size // 2
        self.embed = nn.Conv3d(
            plucker_channels,
            hidden_dim,
            kernel_size=(t_kernel_size, 1, 1),
            padding=(t_pad, 0, 0),
        )
        # Zero-initialize: camera conditioning is identity at init
        nn.init.zeros_(self.embed.weight)
        nn.init.zeros_(self.embed.bias)

    def forward(
        self,
        plucker: Optional[torch.Tensor],  # (B, T, 48, H_sp, W_sp)
    ) -> Optional[torch.Tensor]:
        """Returns (B, T*H*W, D) or None if plucker is None."""
        if plucker is None:
            return None
        B, T, C, H_sp, W_sp = plucker.shape
        # Conv3D expects (B, C, T, H, W)
        x = plucker.permute(0, 2, 1, 3, 4)  # (B, C, T, H, W)
        x = self.embed(x)                   # (B, D, T, H, W)
        x = x.permute(0, 2, 3, 4, 1).reshape(B, T * H_sp * W_sp, -1)  # (B, T*H*W, D)
        return x


# ---------------------------------------------------------------------------
# SANA-WM Transformer Block (Hybrid GDN + Camera branch + Plücker mixing)
# ---------------------------------------------------------------------------


class SanaWMTransformerBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_attention_heads: int,
        attention_head_dim: int,
        num_cross_attention_heads: int,
        cross_attention_head_dim: int,
        cross_attention_dim: int,
        mlp_ratio: float,
        norm_eps: float,
        softmax_attention: bool,       # True → use SoftmaxTemporalAttention, False → GDN
        plucker_channels: int,
        t_kernel_size: int,
        gdn_bidirectional: bool,
        cam_attn_compress: int,
        num_geometric_dims: int = 32,
    ):
        super().__init__()

        self.softmax_attention = softmax_attention

        # AdaLN-scale-shift parameters (6 scalars per token)
        self.scale_shift_table = nn.Parameter(torch.randn(6, dim) / dim**0.5)

        # Pre-attention and pre-FFN norms — use SGLang's fused LayerNormScaleShift
        # (D=2240 triggers the native fuse_scale_shift_kernel fallback, faster than raw PyTorch)
        self.norm1 = LayerNormScaleShift(
            dim, elementwise_affine=False, eps=norm_eps, dtype=torch.float32
        )
        self.norm2 = LayerNormScaleShift(
            dim, elementwise_affine=False, eps=norm_eps, dtype=torch.float32
        )

        # Self-attention: either GDN or Softmax depending on block index
        if softmax_attention:
            self.attn1 = SoftmaxTemporalAttention(
                query_dim=dim,
                num_heads=num_attention_heads,
                head_dim=attention_head_dim,
            )
        else:
            self.attn1 = FrameWiseGDNAttention(
                query_dim=dim,
                num_heads=num_attention_heads,
                head_dim=attention_head_dim,
                bidirectional=gdn_bidirectional,
            )

        # Camera branch (always GDN-based, with UCPE)
        self.cam_attn = UCPECameraAttention(
            query_dim=dim,
            num_heads=num_attention_heads,
            head_dim=attention_head_dim,
            num_geometric_dims=num_geometric_dims,
            cam_attn_compress=cam_attn_compress,
            bidirectional=gdn_bidirectional,
        )

        # Plücker Raymap Mixer (post self-attn)
        self.plucker_mixer = PluckerRaymapMixer(
            plucker_channels=plucker_channels,
            hidden_dim=dim,
            t_kernel_size=t_kernel_size,
        )

        # Cross-attention for text
        self.attn2 = SanaWMCrossAttention(
            query_dim=dim,
            cross_attention_dim=cross_attention_dim,
            num_heads=num_cross_attention_heads,
            head_dim=cross_attention_head_dim,
        )

        # Temporal FFN
        self.ff = GLUMBConvTemp(
            in_channels=dim,
            out_channels=dim,
            expand_ratio=mlp_ratio,
            t_kernel_size=t_kernel_size,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,           # (B, T*H*W, D)
        encoder_hidden_states: torch.Tensor,   # (B, L, D)
        timestep_emb: torch.Tensor,            # (B, 6*D) from AdaLN-single
        num_frames: int,
        sp_h: int,
        sp_w: int,
        freqs: Optional[torch.Tensor] = None,        # RoPE freqs (T*H*W, D//2) complex
        plucker: Optional[torch.Tensor] = None,      # (B, T, 48, H_sp, W_sp)
        D_q: Optional[torch.Tensor] = None,          # UCPE (B, T, S, 3, 3)
        D_k_inv: Optional[torch.Tensor] = None,      # UCPE (B, T, S, 3, 3)
        encoder_attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        batch_size = hidden_states.shape[0]

        # AdaLN modulation from timestep embedding
        # scale_shift_table: (6, D), timestep_emb: (B, 6*D)
        e = self.scale_shift_table[None].float() + timestep_emb.float().reshape(
            batch_size, 6, -1
        )
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = e.chunk(6, dim=1)
        # Each: (B, 1, D)

        # --- Self-attention with fused LayerNormScaleShift ---
        norm_h = self.norm1(hidden_states, shift_msa, scale_msa)

        if self.softmax_attention:
            attn_out = self.attn1(norm_h, freqs=freqs)
        else:
            attn_out = self.attn1(
                norm_h,
                num_frames=num_frames,
                sp_h=sp_h,
                sp_w=sp_w,
                freqs=freqs,
            )

        # Camera branch output (zero-init → no effect until trained)
        cam_out = self.cam_attn(
            norm_h,
            D_q=D_q,
            D_k_inv=D_k_inv,
            num_frames=num_frames,
            sp_h=sp_h,
            sp_w=sp_w,
            freqs=freqs,
        )
        attn_out = attn_out + cam_out  # zero-init projection ensures cam_out=0 at init

        hidden_states = hidden_states + gate_msa * attn_out

        # Plücker raymap post-attn injection (also zero-init)
        plucker_delta = self.plucker_mixer(plucker)
        if plucker_delta is not None:
            hidden_states = hidden_states + plucker_delta

        # --- Cross-attention (text conditioning) ---
        hidden_states = hidden_states + self.attn2(
            hidden_states, encoder_hidden_states, encoder_attention_mask
        )

        # --- Temporal FFN (GLUMBConvTemp) with fused LayerNormScaleShift ---
        norm_h = self.norm2(hidden_states, shift_mlp, scale_mlp)
        hidden_states = hidden_states + gate_mlp * self.ff(
            norm_h, num_frames=num_frames, sp_h=sp_h, sp_w=sp_w
        )

        return hidden_states


# ---------------------------------------------------------------------------
# SANA-WM Transformer 3D Model (top-level DiT)
# ---------------------------------------------------------------------------


class SanaWMTransformer3DModel(CachableDiT, LayerwiseOffloadableModuleMixin):
    """
    SANA-WM: 2.6B dual-branch DiT for 6-DoF camera-controlled TI2V generation.

    Forward inputs:
        hidden_states: (B, C, T, H, W) — video latent from LTX-2 VAE (128 ch)
        encoder_hidden_states: (B, L, 2304) — Gemma-2 text embeddings
        timestep: (B,) long — diffusion timestep
        camera_to_world: (B, T_orig, 4, 4) — per-original-frame c2w matrices (optional)
        intrinsics: (B, 3, 3) or (B, T_orig, 3, 3) — camera intrinsics (optional)
        plucker: (B, T_latent, 48, H_sp, W_sp) — pre-computed Plücker raymaps (optional)
            If None and camera_to_world provided, computed internally.
        encoder_attention_mask: (B, L) bool mask for text tokens

    Returns:
        (B, C, T, H, W) — predicted noise / velocity field
    """

    # Class-level attributes required by BaseDiT / CachableDiT
    _fsdp_shard_conditions = SanaWMConfig()._fsdp_shard_conditions
    _compile_conditions = SanaWMConfig()._compile_conditions
    _supported_attention_backends = SanaWMConfig()._supported_attention_backends
    param_names_mapping = SanaWMConfig().param_names_mapping
    reverse_param_names_mapping = SanaWMConfig().reverse_param_names_mapping
    lora_param_names_mapping: dict = {}

    def __init__(self, config: SanaWMConfig, hf_config=None, **kwargs):
        super().__init__(config, hf_config=hf_config or {}, **kwargs)
        arch = config.arch_config
        self.patch_size = arch.patch_size
        self.inner_dim = arch.num_attention_heads * arch.attention_head_dim
        self.hidden_size = self.inner_dim
        self.num_attention_heads = arch.num_attention_heads
        self.num_channels_latents = arch.num_channels_latents
        self.out_channels = arch.out_channels
        self.attention_head_dim = arch.attention_head_dim
        self.softmax_block_indices = set(arch.softmax_block_indices)

        # Video patch embedding: project (B, C, T, H, W) spatial tokens
        # patch_size=1 means each latent voxel is one token
        self.patch_embed = nn.ModuleDict({
            "proj": nn.Conv3d(
                arch.in_channels,
                self.inner_dim,
                kernel_size=(1, arch.patch_size, arch.patch_size),
                stride=(1, arch.patch_size, arch.patch_size),
                bias=True,
            )
        })

        # Time embedding (AdaLN-single shared across all layers)
        self.time_embed = SanaWMAdaLayerNormSingle(self.inner_dim)

        # Caption projection (Gemma-2 → inner_dim)
        self.caption_projection = PixArtAlphaTextProjection(
            in_features=arch.caption_channels,
            hidden_size=self.inner_dim,
        )
        self.caption_norm = RMSNorm(self.inner_dim)

        # Transformer blocks
        self.transformer_blocks = nn.ModuleList([
            SanaWMTransformerBlock(
                dim=self.inner_dim,
                num_attention_heads=arch.num_attention_heads,
                attention_head_dim=arch.attention_head_dim,
                num_cross_attention_heads=arch.num_cross_attention_heads,
                cross_attention_head_dim=arch.cross_attention_head_dim,
                cross_attention_dim=arch.cross_attention_dim,
                mlp_ratio=arch.mlp_ratio,
                norm_eps=arch.norm_eps,
                softmax_attention=(i in self.softmax_block_indices),
                plucker_channels=arch.chunk_plucker_channels,
                t_kernel_size=arch.t_kernel_size,
                gdn_bidirectional=arch.gdn_bidirectional,
                cam_attn_compress=arch.cam_attn_compress,
            )
            for i in range(arch.num_layers)
        ])

        # Output normalization and projection — fused LayerNormScaleShift
        self.scale_shift_table = nn.Parameter(
            torch.randn(2, self.inner_dim) / self.inner_dim**0.5
        )
        self.norm_out = LayerNormScaleShift(
            self.inner_dim,
            elementwise_affine=False,
            eps=arch.norm_eps,
            dtype=torch.float32,
        )
        self.proj_out = nn.Linear(
            self.inner_dim,
            arch.patch_size * arch.patch_size * self.out_channels,
            bias=True,
        )

        # RoPE: cache computed lazily for efficiency
        self._freqs_cache: dict = {}

        self.layer_names = ["transformer_blocks"]

    def _get_freqs(
        self, num_frames: int, sp_h: int, sp_w: int, device: torch.device
    ) -> torch.Tensor:
        """Retrieve or compute 3D RoPE frequencies, caching by shape."""
        key = (num_frames, sp_h, sp_w, str(device))
        if key not in self._freqs_cache:
            freqs = precompute_freqs_cis_3d(
                head_dim=self.attention_head_dim,  # = 112 = D per head
                max_t=num_frames,
                max_h=sp_h,
                max_w=sp_w,
                device=device,
            )
            self._freqs_cache[key] = freqs
        return self._freqs_cache[key]

    @staticmethod
    def compute_plucker(
        camera_to_world: torch.Tensor,  # (B, T_orig, 4, 4)
        intrinsics: torch.Tensor,        # (B, 3, 3) or (B, T_orig, 3, 3)
        sp_h: int,
        sp_w: int,
        latent_scale: int = 32,
        vae_temporal_stride: int = 8,
    ) -> torch.Tensor:
        """
        Compute packed Plücker raymaps for Plücker Mixing.

        Each latent frame corresponds to `vae_temporal_stride` original frames.
        For each original frame, we compute Plücker coordinates:
            d = normalized ray direction (world space)
            m = o × d (moment, where o is camera origin)
            plucker_frame = [d, m] → 6D

        Returns:
            plucker: (B, T_latent, 48, H_sp, W_sp) — packed 48-ch Plücker
                     where T_latent = T_orig // vae_temporal_stride
        """
        B, T_orig, _, _ = camera_to_world.shape
        device = camera_to_world.device
        dtype = camera_to_world.dtype
        T_latent = T_orig // vae_temporal_stride

        # Handle intrinsics shape
        if intrinsics.dim() == 3:
            intrinsics = intrinsics.unsqueeze(1).expand(B, T_orig, 3, 3)

        # Build pixel grid
        hs = torch.arange(sp_h, device=device, dtype=dtype)
        ws = torch.arange(sp_w, device=device, dtype=dtype)
        grid_h, grid_w = torch.meshgrid(hs, ws, indexing="ij")
        px = (grid_w.flatten() + 0.5) * latent_scale  # (S,)
        py = (grid_h.flatten() + 0.5) * latent_scale  # (S,)
        S = sp_h * sp_w
        ones = torch.ones(S, device=device, dtype=dtype)
        pts = torch.stack([px, py, ones], dim=-1)  # (S, 3)
        pts = pts.unsqueeze(0).unsqueeze(0).expand(B, T_orig, S, 3)  # (B, T, S, 3)

        # Camera origins (translation part)
        origin = camera_to_world[:, :, :3, 3]  # (B, T, 3)
        origin = origin.unsqueeze(2).expand(B, T_orig, S, 3)  # (B, T, S, 3)

        # Unproject to world rays
        K_inv = torch.inverse(intrinsics.reshape(B * T_orig, 3, 3)).reshape(B, T_orig, 3, 3)
        d_cam = torch.einsum("btij,btsj->btsi", K_inv, pts)   # (B, T, S, 3)
        R = camera_to_world[:, :, :3, :3]
        d_world = F.normalize(torch.einsum("btij,btsj->btsi", R, d_cam), dim=-1)

        # Plücker: [d, o × d]
        moment = torch.cross(origin, d_world, dim=-1)  # (B, T, S, 3)
        plucker_per_frame = torch.cat([d_world, moment], dim=-1)  # (B, T, S, 6)

        # Pack: group vae_temporal_stride orig frames into each latent frame
        # Shape: (B, T_latent, vae_temporal_stride, S, 6) → (B, T_latent, S, 48)
        plucker_packed = plucker_per_frame.reshape(
            B, T_latent, vae_temporal_stride, S, 6
        ).permute(0, 1, 3, 2, 4).reshape(B, T_latent, S, vae_temporal_stride * 6)
        # Reshape S → (H_sp, W_sp) and move channel dim: (B, T_latent, 48, H_sp, W_sp)
        plucker_packed = plucker_packed.reshape(B, T_latent, sp_h, sp_w, -1)
        plucker_packed = plucker_packed.permute(0, 1, 4, 2, 3)  # (B, T, 48, H, W)

        return plucker_packed

    def forward(
        self,
        hidden_states: torch.Tensor,                         # (B, C, T, H, W)
        encoder_hidden_states: Optional[torch.Tensor] = None,  # (B, L, 2304)
        timestep: Optional[torch.Tensor] = None,             # (B,)
        camera_to_world: Optional[torch.Tensor] = None,      # (B, T_orig, 4, 4)
        intrinsics: Optional[torch.Tensor] = None,           # (B, 3, 3) or (B, T_orig, 3, 3)
        plucker: Optional[torch.Tensor] = None,              # (B, T, 48, H_sp, W_sp) pre-computed
        encoder_attention_mask: Optional[torch.Tensor] = None,
        guidance: Optional[torch.Tensor] = None,             # unused, kept for compat
        **kwargs,
    ) -> torch.Tensor:

        if encoder_hidden_states is None:
            raise ValueError("encoder_hidden_states is required for SANA-WM forward")

        B, C, T, H, W = hidden_states.shape
        p = self.patch_size
        sp_h = H // p
        sp_w = W // p

        # --- 1. Patch embedding: (B, C, T, H, W) → (B, T*H_sp*W_sp, D) ---
        x = self.patch_embed["proj"](hidden_states)          # (B, D, T, sp_h, sp_w)
        x = x.permute(0, 2, 3, 4, 1).reshape(B, T * sp_h * sp_w, self.inner_dim)

        # --- 2. Timestep embedding ---
        timestep_emb, embedded_timestep = self.time_embed(
            timestep, hidden_dtype=x.dtype
        )

        # --- 3. Text conditioning ---
        if isinstance(encoder_attention_mask, (list, tuple)):
            encoder_attention_mask = encoder_attention_mask[0]

        enc_hs = self.caption_projection(encoder_hidden_states)
        if enc_hs.shape[0] != B:
            enc_hs = enc_hs.expand(B, -1, -1).contiguous()
        enc_hs = enc_hs.view(B, -1, x.shape[-1])
        enc_hs = self.caption_norm(enc_hs)

        if encoder_attention_mask is not None and encoder_attention_mask.shape[0] != B:
            encoder_attention_mask = encoder_attention_mask.expand(B, -1).contiguous()

        # --- 4. 3D RoPE frequencies ---
        freqs = self._get_freqs(T, sp_h, sp_w, device=x.device)

        # --- 5. Compute UCPE transforms (coarse camera) ---
        D_q, D_k_inv = None, None
        if camera_to_world is not None and intrinsics is not None:
            try:
                D_q, D_k_inv = UCPECameraAttention.compute_ucpe_transforms(
                    camera_to_world=camera_to_world,
                    intrinsics=intrinsics,
                    sp_h=sp_h,
                    sp_w=sp_w,
                    latent_scale=p * 32,  # patch_size × VAE spatial stride
                )
            except Exception as e:
                logger.warning(f"UCPE transform computation failed: {e}. Skipping camera branch.")

        # --- 6. Compute Plücker raymaps (fine camera) if not pre-computed ---
        if plucker is None and camera_to_world is not None and intrinsics is not None:
            try:
                vae_temporal_stride = 8  # LTX-2 VAE temporal compression
                plucker = self.compute_plucker(
                    camera_to_world=camera_to_world,
                    intrinsics=intrinsics,
                    sp_h=sp_h,
                    sp_w=sp_w,
                    latent_scale=p * 32,
                    vae_temporal_stride=vae_temporal_stride,
                )
            except Exception as e:
                logger.warning(f"Plücker computation failed: {e}. Disabling Plücker mixing.")
                plucker = None

        # --- 7. Transformer blocks ---
        for block in self.transformer_blocks:
            x = block(
                hidden_states=x,
                encoder_hidden_states=enc_hs,
                timestep_emb=timestep_emb,
                num_frames=T,
                sp_h=sp_h,
                sp_w=sp_w,
                freqs=freqs,
                plucker=plucker,
                D_q=D_q,
                D_k_inv=D_k_inv,
                encoder_attention_mask=encoder_attention_mask,
            )

        # --- 8. Output projection ---
        shift, scale = (
            self.scale_shift_table[None].float() + embedded_timestep.float()[:, None]
        ).chunk(2, dim=1)
        x = self.norm_out(x, shift, scale)
        x = self.proj_out(x)

        # --- 9. Un-patch: (B, T*sp_h*sp_w, p*p*C_out) → (B, C_out, T, H, W) ---
        x = x.reshape(B, T, sp_h, sp_w, p, p, self.out_channels)
        x = x.permute(0, 6, 1, 2, 4, 3, 5)   # (B, C, T, sp_h, p, sp_w, p)
        x = x.reshape(B, self.out_channels, T, H, W)

        return x


EntryClass = SanaWMTransformer3DModel
