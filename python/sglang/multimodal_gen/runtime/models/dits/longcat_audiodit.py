# Copied and adapted from: https://github.com/meituan-longcat/LongCat-AudioDiT
"""PyTorch LongCatAudioDiT model — Conditional Flow Matching TTS with DiT backbone."""

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.utils import weight_norm
from torch.nn.utils.rnn import pad_sequence
from transformers import PreTrainedModel, logging
from transformers.modeling_outputs import ModelOutput

from sglang.multimodal_gen.configs.models.dits.longcat_audiodit import (
    LongCatAudioDiTConfig,
    LongCatAudioDiTVaeConfig,
)

logger = logging.get_logger(__name__)


# ---------------------------------------------------------------------------
# Output dataclass
# ---------------------------------------------------------------------------


@dataclass
class LongCatAudioDiTOutput(ModelOutput):
    """
    Output of [`LongCatAudioDiTModel`].

    Args:
        waveform (`torch.FloatTensor` of shape `(batch_size, num_samples)`):
            Generated audio waveform.
        latent (`torch.FloatTensor` of shape `(batch_size, latent_dim, num_frames)`):
            Predicted latent representation before VAE decoding.
    """

    waveform: torch.FloatTensor | None = None
    latent: torch.FloatTensor | None = None


# ---------------------------------------------------------------------------
# ODE solver (inline Euler — replaces torchdiffeq dependency)
# ---------------------------------------------------------------------------


def odeint_euler(fn, y0, t):
    """Simple Euler ODE integrator (equivalent to `torchdiffeq.odeint` with `method='euler'`).

    Args:
        fn: callable(t, y) → dy/dt
        y0: initial state tensor
        t: 1-D tensor of time steps (must be monotonically increasing)

    Returns:
        Tensor of shape `(len(t), *y0.shape)` containing the trajectory.
    """
    ys = [y0]
    y = y0
    for i in range(len(t) - 1):
        dt = t[i + 1] - t[i]
        y = y + fn(t[i], y) * dt
        ys.append(y)
    return torch.stack(ys)


# ---------------------------------------------------------------------------
# Utility helpers (from model/utils.py)
# ---------------------------------------------------------------------------


def lens_to_mask(lengths: torch.Tensor, length: int | None = None) -> torch.BoolTensor:
    if length is None:
        length = lengths.amax()
    seq = torch.arange(length, device=lengths.device)
    return seq[None, :] < lengths[:, None]


# ---------------------------------------------------------------------------
# Low-level modules (from model/modules.py)
# ---------------------------------------------------------------------------


class LongCatAudioDiTRMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._norm(x.float()).type_as(x) * self.weight

    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)


class LongCatAudioDiTSinusPositionEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor, scale: float = 1000.0) -> torch.Tensor:
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device).float() * -emb)
        emb = scale * x.unsqueeze(1) * emb.unsqueeze(0)
        return torch.cat((emb.sin(), emb.cos()), dim=-1)


class LongCatAudioDiTTimestepEmbedding(nn.Module):
    def __init__(self, dim: int, freq_embed_dim: int = 256):
        super().__init__()
        self.time_embed = LongCatAudioDiTSinusPositionEmbedding(freq_embed_dim)
        self.time_mlp = nn.Sequential(
            nn.Linear(freq_embed_dim, dim), nn.SiLU(), nn.Linear(dim, dim)
        )

    def forward(self, timestep: torch.Tensor) -> torch.Tensor:
        time_hidden = self.time_embed(timestep)
        time_hidden = time_hidden.to(timestep.dtype)
        return self.time_mlp(time_hidden)


class LongCatAudioDiTRotaryEmbedding(nn.Module):
    """Qwen2-style rotary position embedding.

    All state (inv_freq, cos/sin caches) is built lazily on first ``forward``
    call.  This avoids corruption from ``from_pretrained`` meta-device
    construction while producing bit-identical results to the original
    ``Qwen2RotaryEmbedding`` (which creates ``inv_freq`` on CPU then moves
    the whole model to CUDA with ``.to(device)``).
    """

    def __init__(
        self, dim: int, max_position_embeddings: int = 2048, base: float = 100000.0
    ):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        # Do NOT register any buffers here — they get corrupted by meta-device.
        # Everything is built lazily in forward().
        self._cos: torch.Tensor | None = None
        self._sin: torch.Tensor | None = None
        self._cached_len: int = 0
        self._cached_device: torch.device | None = None

    def _build(self, seq_len: int, device: torch.device, dtype: torch.dtype):
        """Build cos/sin tables entirely on CPU (matching original
        Qwen2RotaryEmbedding which builds in __init__ on CPU, then the
        whole model is moved with .to(device)), then move to target."""
        inv_freq = 1.0 / (
            self.base
            ** (torch.arange(0, self.dim, 2, dtype=torch.int64).float() / self.dim)
        )
        t = torch.arange(seq_len, dtype=torch.int64).type_as(inv_freq)
        freqs = torch.outer(t, inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self._cos = emb.cos().to(dtype=dtype, device=device)
        self._sin = emb.sin().to(dtype=dtype, device=device)
        self._cached_len = seq_len
        self._cached_device = device

    def forward(
        self, x: torch.Tensor, seq_len: int | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if seq_len is None:
            seq_len = x.shape[1]
        if (
            self._cos is None
            or seq_len > self._cached_len
            or self._cached_device != x.device
        ):
            self._build(max(seq_len, self.max_position_embeddings), x.device, x.dtype)
        return (
            self._cos[:seq_len].to(dtype=x.dtype),
            self._sin[:seq_len].to(dtype=x.dtype),
        )


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)


def _apply_rotary_emb(
    x: torch.Tensor, freqs_cis: tuple[torch.Tensor, torch.Tensor]
) -> torch.Tensor:
    cos, sin = freqs_cis
    cos = cos[None, None].to(x.device)
    sin = sin[None, None].to(x.device)
    return (x.float() * cos + _rotate_half(x).float() * sin).to(x.dtype)


# ---------------------------------------------------------------------------
# GRN + ConvNeXtV2 (for text conv)
# ---------------------------------------------------------------------------


class LongCatAudioDiTGRN(nn.Module):
    """Global Response Normalization."""

    def __init__(self, dim: int):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1, 1, dim))
        self.beta = nn.Parameter(torch.zeros(1, 1, dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gx = torch.norm(x, p=2, dim=1, keepdim=True)
        nx = gx / (gx.mean(dim=-1, keepdim=True) + 1e-6)
        return self.gamma * (x * nx) + self.beta + x


class LongCatAudioDiTConvNeXtV2Block(nn.Module):
    def __init__(
        self,
        dim: int,
        intermediate_dim: int,
        dilation: int = 1,
        kernel_size: int = 7,
        bias: bool = True,
        eps: float = 1e-6,
    ):
        super().__init__()
        padding = (dilation * (kernel_size - 1)) // 2
        self.dwconv = nn.Conv1d(
            dim,
            dim,
            kernel_size=kernel_size,
            padding=padding,
            groups=dim,
            dilation=dilation,
            bias=bias,
        )
        self.norm = nn.LayerNorm(dim, eps=eps)
        self.pwconv1 = nn.Linear(dim, intermediate_dim, bias=bias)
        self.act = nn.SiLU()
        self.grn = LongCatAudioDiTGRN(intermediate_dim)
        self.pwconv2 = nn.Linear(intermediate_dim, dim, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = x.transpose(1, 2)
        x = self.dwconv(x)
        x = x.transpose(1, 2)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.grn(x)
        x = self.pwconv2(x)
        return residual + x


# ---------------------------------------------------------------------------
# Embedder (shared for input / text / latent)
# ---------------------------------------------------------------------------


class LongCatAudioDiTEmbedder(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(in_dim, out_dim), nn.SiLU(), nn.Linear(out_dim, out_dim)
        )

    def forward(
        self, x: torch.Tensor, mask: torch.BoolTensor | None = None
    ) -> torch.Tensor:
        if mask is not None:
            x = x.masked_fill(mask.logical_not().unsqueeze(-1), 0.0)
        x = self.proj(x)
        if mask is not None:
            x = x.masked_fill(mask.logical_not().unsqueeze(-1), 0.0)
        return x


# ---------------------------------------------------------------------------
# AdaLN modules
# ---------------------------------------------------------------------------


class LongCatAudioDiTAdaLNMLP(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, bias: bool = True):
        super().__init__()
        self.mlp = nn.Sequential(nn.SiLU(), nn.Linear(in_dim, out_dim, bias=bias))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)


class LongCatAudioDiTAdaLayerNormZeroFinal(nn.Module):
    def __init__(self, dim: int, bias: bool = True, eps: float = 1e-6):
        super().__init__()
        self.silu = nn.SiLU()
        self.linear = nn.Linear(dim, dim * 2, bias=bias)
        self.norm = nn.LayerNorm(dim, elementwise_affine=False, eps=eps)

    def forward(self, x: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        emb = self.linear(self.silu(emb))
        scale, shift = torch.chunk(emb, 2, dim=-1)
        x = self.norm(x.float()).type_as(x)
        if scale.ndim == 2:
            x = x * (1 + scale)[:, None, :] + shift[:, None, :]
        else:
            x = x * (1 + scale) + shift
        return x


# ---------------------------------------------------------------------------
# Attention
# ---------------------------------------------------------------------------


def _modulate(
    x: torch.Tensor, scale: torch.Tensor, shift: torch.Tensor, eps: float = 1e-6
) -> torch.Tensor:
    """LayerNorm without affine + modulate."""
    x = F.layer_norm(x.float(), (x.shape[-1],), eps=eps).type_as(x)
    if scale.ndim == 2:
        return x * (1 + scale[:, None]) + shift[:, None]
    return x * (1 + scale) + shift


class LongCatAudioDiTSelfAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        heads: int,
        dim_head: int,
        dropout: float = 0.0,
        bias: bool = True,
        qk_norm: bool = False,
        eps: float = 1e-6,
    ):
        super().__init__()
        self.heads = heads
        self.inner_dim = dim_head * heads
        self.to_q = nn.Linear(dim, self.inner_dim, bias=bias)
        self.to_k = nn.Linear(dim, self.inner_dim, bias=bias)
        self.to_v = nn.Linear(dim, self.inner_dim, bias=bias)
        self.qk_norm = qk_norm
        if qk_norm:
            self.q_norm = LongCatAudioDiTRMSNorm(self.inner_dim, eps=eps)
            self.k_norm = LongCatAudioDiTRMSNorm(self.inner_dim, eps=eps)
        self.to_out = nn.ModuleList(
            [nn.Linear(self.inner_dim, dim, bias=bias), nn.Dropout(dropout)]
        )

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.BoolTensor | None = None,
        rope: tuple | None = None,
    ) -> torch.Tensor:
        batch_size = x.shape[0]
        query = self.to_q(x)
        key = self.to_k(x)
        value = self.to_v(x)
        if self.qk_norm:
            query = self.q_norm(query)
            key = self.k_norm(key)
        head_dim = self.inner_dim // self.heads
        query = query.view(batch_size, -1, self.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, self.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, self.heads, head_dim).transpose(1, 2)
        if rope is not None:
            query = _apply_rotary_emb(query, rope)
            key = _apply_rotary_emb(key, rope)
        attn_mask = None
        if mask is not None:
            attn_mask = (
                mask.unsqueeze(1)
                .unsqueeze(1)
                .expand(batch_size, self.heads, query.shape[-2], key.shape[-2])
            )
        x = F.scaled_dot_product_attention(
            query, key, value, attn_mask=attn_mask, dropout_p=0.0, is_causal=False
        )
        x = x.transpose(1, 2).reshape(batch_size, -1, self.inner_dim).to(query.dtype)
        x = self.to_out[0](x)
        x = self.to_out[1](x)
        return x


class LongCatAudioDiTCrossAttention(nn.Module):
    def __init__(
        self,
        q_dim: int,
        kv_dim: int,
        heads: int,
        dim_head: int,
        dropout: float = 0.0,
        bias: bool = True,
        qk_norm: bool = False,
        eps: float = 1e-6,
    ):
        super().__init__()
        self.heads = heads
        self.inner_dim = dim_head * heads
        self.to_q = nn.Linear(q_dim, self.inner_dim, bias=bias)
        self.to_k = nn.Linear(kv_dim, self.inner_dim, bias=bias)
        self.to_v = nn.Linear(kv_dim, self.inner_dim, bias=bias)
        self.qk_norm = qk_norm
        if qk_norm:
            self.q_norm = LongCatAudioDiTRMSNorm(self.inner_dim, eps=eps)
            self.k_norm = LongCatAudioDiTRMSNorm(self.inner_dim, eps=eps)
        self.to_out = nn.ModuleList(
            [nn.Linear(self.inner_dim, q_dim, bias=bias), nn.Dropout(dropout)]
        )

    def forward(
        self,
        x: torch.Tensor,
        cond: torch.Tensor,
        mask: torch.BoolTensor | None = None,
        cond_mask: torch.BoolTensor | None = None,
        rope: tuple | None = None,
        cond_rope: tuple | None = None,
    ) -> torch.Tensor:
        batch_size = x.shape[0]
        query = self.to_q(x)
        key = self.to_k(cond)
        value = self.to_v(cond)
        if self.qk_norm:
            query = self.q_norm(query)
            key = self.k_norm(key)
        head_dim = self.inner_dim // self.heads
        query = query.view(batch_size, -1, self.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, self.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, self.heads, head_dim).transpose(1, 2)
        if rope is not None:
            query = _apply_rotary_emb(query, rope)
        if cond_rope is not None:
            key = _apply_rotary_emb(key, cond_rope)
        attn_mask = None
        if mask is not None:
            attn_mask = (
                cond_mask.unsqueeze(1).expand(-1, mask.shape[1], -1).unsqueeze(1)
            )
            attn_mask = attn_mask.expand(
                batch_size, self.heads, query.shape[-2], key.shape[-2]
            )
        x = F.scaled_dot_product_attention(
            query, key, value, attn_mask=attn_mask, dropout_p=0.0, is_causal=False
        )
        x = x.transpose(1, 2).reshape(batch_size, -1, self.inner_dim).to(query.dtype)
        x = self.to_out[0](x)
        x = self.to_out[1](x)
        return x


# ---------------------------------------------------------------------------
# FeedForward
# ---------------------------------------------------------------------------


class LongCatAudioDiTFeedForward(nn.Module):
    def __init__(
        self, dim: int, mult: float = 4.0, dropout: float = 0.0, bias: bool = True
    ):
        super().__init__()
        inner_dim = int(dim * mult)
        self.ff = nn.Sequential(
            nn.Linear(dim, inner_dim, bias=bias),
            nn.GELU(approximate="tanh"),
            nn.Dropout(dropout),
            nn.Linear(inner_dim, dim, bias=bias),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.ff(x)


# ---------------------------------------------------------------------------
# Transformer Block (CrossDiTBlock)
# ---------------------------------------------------------------------------


class LongCatAudioDiTBlock(nn.Module):
    """Single DiT block with self-attention, optional cross-attention, FFN, and AdaLN modulation."""

    def __init__(self, config: LongCatAudioDiTConfig):
        super().__init__()
        dim = config.dit_dim
        cond_dim = config.dit_dim  # after text embedding, cond_dim == dim
        heads = config.dit_heads
        dim_head = dim // heads
        bias = config.dit_bias
        eps = config.dit_eps

        self.adaln_type = config.dit_adaln_type
        self.adaln_use_text_cond = config.dit_adaln_use_text_cond
        if config.dit_adaln_type == "local":
            self.adaln_mlp = LongCatAudioDiTAdaLNMLP(dim, dim * 6, bias=True)
        elif config.dit_adaln_type == "global":
            self.adaln_scale_shift = nn.Parameter(torch.randn(dim * 6) / dim**0.5)

        self.self_attn = LongCatAudioDiTSelfAttention(
            dim=dim,
            heads=heads,
            dim_head=dim_head,
            dropout=config.dit_dropout,
            bias=bias,
            qk_norm=config.dit_qk_norm,
            eps=eps,
        )

        self.use_cross_attn = config.dit_cross_attn
        if config.dit_cross_attn:
            self.cross_attn = LongCatAudioDiTCrossAttention(
                q_dim=dim,
                kv_dim=cond_dim,
                heads=heads,
                dim_head=dim_head,
                dropout=config.dit_dropout,
                bias=bias,
                qk_norm=config.dit_qk_norm,
                eps=eps,
            )
            self.cross_attn_norm = (
                nn.LayerNorm(dim, elementwise_affine=True, eps=eps)
                if config.dit_cross_attn_norm
                else nn.Identity()
            )
            self.cross_attn_norm_c = (
                nn.LayerNorm(cond_dim, elementwise_affine=True, eps=eps)
                if config.dit_cross_attn_norm
                else nn.Identity()
            )

        self.ffn = LongCatAudioDiTFeedForward(
            dim=dim, mult=config.dit_ff_mult, dropout=config.dit_dropout, bias=bias
        )

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        cond: torch.Tensor,
        mask: torch.BoolTensor | None = None,
        cond_mask: torch.BoolTensor | None = None,
        rope: tuple | None = None,
        cond_rope: tuple | None = None,
        adaln_global_out: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.adaln_type == "local" and adaln_global_out is None:
            if self.adaln_use_text_cond:
                cond_mean = cond.sum(1) / cond_mask.sum(1, keepdim=True)
                norm_cond = t + cond_mean
            else:
                norm_cond = t
            adaln_out = self.adaln_mlp(norm_cond)
            gate_sa, scale_sa, shift_sa, gate_ffn, scale_ffn, shift_ffn = torch.chunk(
                adaln_out, 6, dim=-1
            )
        else:
            from einops import rearrange

            adaln_out = adaln_global_out + rearrange(self.adaln_scale_shift, "f -> 1 f")
            gate_sa, scale_sa, shift_sa, gate_ffn, scale_ffn, shift_ffn = torch.chunk(
                adaln_out, 6, dim=-1
            )

        # Self-attention
        norm = _modulate(x, scale_sa, shift_sa)
        attn_output = self.self_attn(norm, mask=mask, rope=rope)
        if gate_sa.ndim == 2:
            gate_sa = gate_sa.unsqueeze(1)
        x = x + gate_sa * attn_output

        # Cross-attention
        if self.use_cross_attn:
            cross_out = self.cross_attn(
                x=self.cross_attn_norm(x),
                cond=self.cross_attn_norm_c(cond),
                mask=mask,
                cond_mask=cond_mask,
                rope=rope,
                cond_rope=cond_rope,
            )
            x = x + cross_out

        # FFN
        norm = _modulate(x, scale_ffn, shift_ffn)
        ff_output = self.ffn(norm)
        if gate_ffn.ndim == 2:
            gate_ffn = gate_ffn.unsqueeze(1)
        x = x + gate_ffn * ff_output
        return x


# ---------------------------------------------------------------------------
# LongCatAudioDiTTransformer (CrossDiT backbone)
# ---------------------------------------------------------------------------


class LongCatAudioDiTTransformer(nn.Module):
    """The core DiT transformer backbone for LongCatAudioDiT."""

    def __init__(self, config: LongCatAudioDiTConfig):
        super().__init__()
        dim = config.dit_dim
        latent_dim = config.latent_dim  # 64
        text_dim = config.dit_text_dim
        dim_head = dim // config.dit_heads

        self.config = config
        self.dim = dim
        self.depth = config.dit_depth
        self.long_skip = config.dit_long_skip
        self.adaln_type = config.dit_adaln_type
        self.adaln_use_text_cond = config.dit_adaln_use_text_cond

        self.time_embed = LongCatAudioDiTTimestepEmbedding(dim)
        self.input_embed = LongCatAudioDiTEmbedder(latent_dim, dim)
        self.text_embed = LongCatAudioDiTEmbedder(text_dim, dim)
        self.rotary_embed = LongCatAudioDiTRotaryEmbedding(
            dim_head, 2048, base=100000.0
        )

        self.blocks = nn.ModuleList(
            [LongCatAudioDiTBlock(config) for _ in range(config.dit_depth)]
        )

        self.norm_out = LongCatAudioDiTAdaLayerNormZeroFinal(
            dim, bias=True, eps=config.dit_eps
        )
        self.proj_out = nn.Linear(dim, latent_dim)

        if config.dit_adaln_type == "global":
            self.adaln_global_mlp = LongCatAudioDiTAdaLNMLP(dim, dim * 6, bias=True)

        self.text_conv = config.dit_text_conv
        if config.dit_text_conv:
            self.text_conv_layer = nn.Sequential(
                *[
                    LongCatAudioDiTConvNeXtV2Block(
                        dim, dim * 2, bias=config.dit_bias, eps=config.dit_eps
                    )
                    for _ in range(4)
                ]
            )

        self.use_latent_condition = config.dit_use_latent_condition
        if config.dit_use_latent_condition:
            self.latent_embed = LongCatAudioDiTEmbedder(latent_dim, dim)
            self.latent_cond_embedder = LongCatAudioDiTEmbedder(dim * 2, dim)

        self._initialize_weights()

    def _initialize_weights(self):
        """Zero-out AdaLN and output projection weights for stable training init."""
        bias = self.config.dit_bias
        if self.adaln_type == "local":
            for block in self.blocks:
                nn.init.constant_(block.adaln_mlp.mlp[-1].weight, 0)
                if bias:
                    nn.init.constant_(block.adaln_mlp.mlp[-1].bias, 0)
        elif self.adaln_type == "global":
            nn.init.constant_(self.adaln_global_mlp.mlp[-1].weight, 0)
            if bias:
                nn.init.constant_(self.adaln_global_mlp.mlp[-1].bias, 0)

        nn.init.constant_(self.norm_out.linear.weight, 0)
        nn.init.constant_(self.proj_out.weight, 0)
        if bias:
            nn.init.constant_(self.norm_out.linear.bias, 0)
            nn.init.constant_(self.proj_out.bias, 0)

        for m in self.time_embed.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        for m in self.text_embed.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(
        self,
        x: torch.Tensor,
        text: torch.Tensor,
        text_len: torch.Tensor,
        time: torch.Tensor,
        mask: torch.BoolTensor | None = None,
        cond_mask: torch.BoolTensor | None = None,
        return_ith_layer: int | None = None,
        latent_cond: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor | None]:
        dtype = next(self.parameters()).dtype
        x = x.to(dtype)
        text = text.to(dtype)
        time = time.to(dtype)

        batch = x.shape[0]
        text_seq_len = text.shape[1]
        if time.ndim == 0:
            time = time.repeat(batch)

        t = self.time_embed(time)
        text = self.text_embed(text, cond_mask)
        if self.text_conv:
            text = self.text_conv_layer(text)
            text = text.masked_fill(cond_mask.logical_not().unsqueeze(-1), 0.0)

        x = self.input_embed(x, mask)
        if self.use_latent_condition:
            latent_cond = latent_cond.to(dtype)
            latent_cond = self.latent_embed(latent_cond, mask)
            x = self.latent_cond_embedder(torch.cat([x, latent_cond], dim=-1))

        if self.long_skip:
            x_clone = x.clone()

        seq_len = x.shape[1]
        rope = self.rotary_embed(x, seq_len)
        cond_rope = self.rotary_embed(text, text_seq_len)

        if self.adaln_type == "global":
            if self.adaln_use_text_cond:
                text_mean = text.sum(1) / text_len.unsqueeze(1).to(text.dtype)
                norm_cond = t + text_mean
            else:
                norm_cond = t
            adaln_mlp_out = self.adaln_global_mlp(norm_cond)
        else:
            adaln_mlp_out = None
            norm_cond = None

        hidden_state = None
        for i, block in enumerate(self.blocks):
            x = block(
                x=x,
                t=t,
                cond=text,
                mask=mask,
                cond_mask=cond_mask,
                rope=rope,
                cond_rope=cond_rope,
                adaln_global_out=adaln_mlp_out,
            )
            if return_ith_layer == i + 1:
                hidden_state = x.clone()
                if self.long_skip:
                    x = x + x_clone

        if self.long_skip:
            x = x + x_clone

        x = self.norm_out(x, norm_cond if norm_cond is not None else t)
        output = self.proj_out(x)
        return {"last_hidden_state": output, "hidden_state": hidden_state}


# ---------------------------------------------------------------------------
# WAV-VAE components (from wav_vae.py)
# ---------------------------------------------------------------------------


def _snake_beta(
    x: torch.Tensor, alpha: torch.Tensor, beta: torch.Tensor
) -> torch.Tensor:
    return x + (1.0 / (beta + 1e-9)) * torch.sin(x * alpha).pow(2)


class LongCatAudioDiTSnakeBeta(nn.Module):
    def __init__(self, in_features: int, alpha_logscale: bool = True):
        super().__init__()
        self.alpha_logscale = alpha_logscale
        self.alpha = nn.Parameter(torch.zeros(in_features))
        self.beta = nn.Parameter(torch.zeros(in_features))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        alpha = self.alpha.unsqueeze(0).unsqueeze(-1)
        beta = self.beta.unsqueeze(0).unsqueeze(-1)
        if self.alpha_logscale:
            alpha = torch.exp(alpha)
            beta = torch.exp(beta)
        return _snake_beta(x, alpha, beta)


def _get_vae_activation(activation: str, channels: int | None = None) -> nn.Module:
    if activation == "elu":
        return nn.ELU()
    elif activation == "snake":
        return LongCatAudioDiTSnakeBeta(channels)
    elif activation == "none":
        return nn.Identity()
    raise ValueError(f"Unknown activation {activation}")


def _wn_conv1d(*args, **kwargs):
    return weight_norm(nn.Conv1d(*args, **kwargs))


def _wn_conv_transpose1d(*args, **kwargs):
    return weight_norm(nn.ConvTranspose1d(*args, **kwargs))


def _pixel_unshuffle_1d(x: torch.Tensor, factor: int) -> torch.Tensor:
    b, c, w = x.size()
    return (
        x.view(b, c, w // factor, factor)
        .permute(0, 1, 3, 2)
        .contiguous()
        .view(b, c * factor, w // factor)
    )


def _pixel_shuffle_1d(x: torch.Tensor, factor: int) -> torch.Tensor:
    b, c, w = x.size()
    c = c // factor
    return (
        x.view(b, c, factor, w).permute(0, 1, 3, 2).contiguous().view(b, c, w * factor)
    )


class _DownsampleShortcut(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, factor: int):
        super().__init__()
        self.factor = factor
        self.group_size = in_channels * factor // out_channels
        self.out_channels = out_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = _pixel_unshuffle_1d(x, self.factor)
        b, c, n = x.shape
        return x.view(b, self.out_channels, self.group_size, n).mean(dim=2)


class _UpsampleShortcut(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, factor: int):
        super().__init__()
        self.factor = factor
        self.repeats = out_channels * factor // in_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.repeat_interleave(self.repeats, dim=1)
        return _pixel_shuffle_1d(x, self.factor)


class _VaeResidualUnit(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        dilation: int,
        kernel_size: int = 7,
        use_snake: bool = False,
    ):
        super().__init__()
        padding = (dilation * (kernel_size - 1)) // 2
        act = "snake" if use_snake else "elu"
        self.layers = nn.Sequential(
            _get_vae_activation(act, channels=out_channels),
            _wn_conv1d(
                in_channels,
                out_channels,
                kernel_size,
                dilation=dilation,
                padding=padding,
            ),
            _get_vae_activation(act, channels=out_channels),
            _wn_conv1d(out_channels, out_channels, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.layers(x)


class _VaeEncoderBlock(nn.Module):
    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        stride: int,
        use_snake: bool = False,
        downsample_shortcut: str = "none",
    ):
        super().__init__()
        layers = []
        for d in [1, 3, 9]:
            layers.append(
                _VaeResidualUnit(in_ch, in_ch, dilation=d, use_snake=use_snake)
            )
        act = "snake" if use_snake else "elu"
        layers.append(_get_vae_activation(act, channels=in_ch))
        layers.append(
            _wn_conv1d(
                in_ch,
                out_ch,
                kernel_size=2 * stride,
                stride=stride,
                padding=math.ceil(stride / 2),
            )
        )
        self.layers = nn.Sequential(*layers)
        self.res = (
            _DownsampleShortcut(in_ch, out_ch, stride)
            if downsample_shortcut == "averaging"
            else None
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.res is not None:
            return self.layers(x) + self.res(x)
        return self.layers(x)


class _VaeDecoderBlock(nn.Module):
    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        stride: int,
        use_snake: bool = False,
        upsample_shortcut: str = "none",
    ):
        super().__init__()
        act = "snake" if use_snake else "elu"
        layers = [
            _get_vae_activation(act, channels=in_ch),
            _wn_conv_transpose1d(
                in_ch,
                out_ch,
                kernel_size=2 * stride,
                stride=stride,
                padding=math.ceil(stride / 2),
            ),
        ]
        for d in [1, 3, 9]:
            layers.append(
                _VaeResidualUnit(out_ch, out_ch, dilation=d, use_snake=use_snake)
            )
        self.layers = nn.Sequential(*layers)
        self.res = (
            _UpsampleShortcut(in_ch, out_ch, stride)
            if upsample_shortcut == "duplicating"
            else None
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.res is not None:
            return self.layers(x) + self.res(x)
        return self.layers(x)


class LongCatAudioDiTVaeEncoder(nn.Module):
    def __init__(self, config: LongCatAudioDiTVaeConfig):
        super().__init__()
        c_mults = [1] + config.c_mults
        ch = config.channels
        layers = [
            _wn_conv1d(config.in_channels, c_mults[0] * ch, kernel_size=7, padding=3)
        ]
        for i in range(len(c_mults) - 1):
            layers.append(
                _VaeEncoderBlock(
                    c_mults[i] * ch,
                    c_mults[i + 1] * ch,
                    config.strides[i],
                    use_snake=config.use_snake,
                    downsample_shortcut=config.downsample_shortcut,
                )
            )
        layers.append(
            _wn_conv1d(
                c_mults[-1] * ch, config.encoder_latent_dim, kernel_size=3, padding=1
            )
        )
        self.layers = nn.Sequential(*layers)

        if config.out_shortcut == "averaging":
            self.shortcut = _DownsampleShortcut(
                c_mults[-1] * ch, config.encoder_latent_dim, 1
            )
        else:
            self.shortcut = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.shortcut is None:
            return self.layers(x)
        x = self.layers[:-1](x)
        return self.layers[-1](x) + self.shortcut(x)


class LongCatAudioDiTVaeDecoder(nn.Module):
    def __init__(self, config: LongCatAudioDiTVaeConfig):
        super().__init__()
        c_mults = [1] + config.c_mults
        ch = config.channels

        if config.in_shortcut == "duplicating":
            self.shortcut = _UpsampleShortcut(config.latent_dim, c_mults[-1] * ch, 1)
        else:
            self.shortcut = None

        layers = [
            _wn_conv1d(config.latent_dim, c_mults[-1] * ch, kernel_size=7, padding=3)
        ]
        for i in range(len(c_mults) - 1, 0, -1):
            layers.append(
                _VaeDecoderBlock(
                    c_mults[i] * ch,
                    c_mults[i - 1] * ch,
                    config.strides[i - 1],
                    use_snake=config.use_snake,
                    upsample_shortcut=config.upsample_shortcut,
                )
            )
        act = "snake" if config.use_snake else "elu"
        layers.append(_get_vae_activation(act, channels=c_mults[0] * ch))
        layers.append(
            _wn_conv1d(
                c_mults[0] * ch,
                config.in_channels,
                kernel_size=7,
                padding=3,
                bias=False,
            )
        )
        if config.final_tanh:
            layers.append(nn.Tanh())
        else:
            layers.append(nn.Identity())
        self.layers = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.shortcut is None:
            return self.layers(x)
        x_short = self.shortcut(x) + self.layers[0](x)
        return self.layers[1:](x_short)


class LongCatAudioDiTVae(nn.Module):
    """WAV-VAE audio autoencoder with VAE bottleneck and scale factor.

    The original checkpoint runs encode/decode in **float16** (``model_half=True``
    in ``AutoencoderPretransform``).  We replicate this behaviour so that the
    outputs are numerically identical to the original codebase.
    """

    def __init__(self, config: LongCatAudioDiTVaeConfig):
        super().__init__()
        self.config = config
        self.encoder = LongCatAudioDiTVaeEncoder(config)
        self.decoder = LongCatAudioDiTVaeDecoder(config)
        self.scale = config.scale
        self.downsampling_ratio = config.downsampling_ratio

    def to_half(self):
        """Convert encoder and decoder weights to float16 (matching original behaviour)."""
        self.encoder.half()
        self.decoder.half()
        return self

    def encode(self, audio: torch.Tensor) -> torch.Tensor:
        """Encode audio to latent space.

        Runs encoder **and** VAE bottleneck in float16 when weights are float16,
        matching the original ``AutoencoderPretransform(model_half=True)`` +
        ``AudioAutoencoder.encode`` behaviour where the bottleneck operates on
        the fp16 encoder output before the final ``.float()`` conversion.

        Args:
            audio: ``(batch, 1, num_samples)`` raw waveform.

        Returns:
            Latent tensor ``(batch, latent_dim, num_frames)`` in float32.
        """
        is_half = next(self.encoder.parameters()).dtype == torch.float16
        if is_half:
            audio = audio.half()
        latents = self.encoder(audio)
        # VAE bottleneck runs in the same dtype as encoder output (fp16)
        # to match original: bottleneck.encode(latents) happens before .float()
        mean, scale_param = latents.chunk(2, dim=1)
        stdev = F.softplus(scale_param) + 1e-4
        latents = torch.randn_like(mean) * stdev + mean
        # Convert to fp32 after bottleneck, matching original AutoencoderPretransform
        if is_half:
            latents = latents.float()
        return latents / self.scale

    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        """Decode latents to audio waveform.

        Runs decoder in float16 when weights are float16, matching the original
        ``AutoencoderPretransform(model_half=True)`` behaviour.

        Args:
            latents: ``(batch, latent_dim, num_frames)``.

        Returns:
            Waveform tensor ``(batch, 1, num_samples)`` in float32.
        """
        z = latents * self.scale
        is_half = next(self.decoder.parameters()).dtype == torch.float16
        if is_half:
            z = z.half()
        decoded = self.decoder(z)
        if is_half:
            decoded = decoded.float()
        return decoded


# ---------------------------------------------------------------------------
# Top-level LongCatAudioDiTModel
# ---------------------------------------------------------------------------


class LongCatAudioDiTPreTrainedModel(PreTrainedModel):
    config_class = LongCatAudioDiTConfig
    base_model_prefix = "audiodit"
    supports_gradient_checkpointing = True
    _supports_sdpa = True

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, std=0.02)


class LongCatAudioDiTModel(LongCatAudioDiTPreTrainedModel):
    """LongCatAudioDiT: Conditional Flow Matching TTS model with DiT backbone, UMT5 text encoder, and WAV-VAE.

    All sub-models (text_encoder, transformer, vae) are constructed from config
    and their weights are loaded together via ``from_pretrained``.

    Example::

        model = LongCatAudioDiTModel.from_pretrained("hf_audiodit_1b")
        tokenizer = AutoTokenizer.from_pretrained(model.config.text_encoder_model)
        output = model(text=["Hello world"], tokenizer=tokenizer)
        waveform = output.waveform  # (B, num_samples)
    """

    def __init__(self, config: LongCatAudioDiTConfig):
        super().__init__(config)
        self.config = config

        # Text encoder — constructed from embedded config, weights loaded by from_pretrained
        from transformers import UMT5Config, UMT5EncoderModel

        if config.text_encoder_config is not None:
            self.text_encoder = UMT5EncoderModel(config.text_encoder_config)
        else:
            te_config = UMT5Config.from_pretrained(config.text_encoder_model)
            self.text_encoder = UMT5EncoderModel(te_config)
        self.text_encoder.requires_grad_(False)

        # DiT transformer
        self.transformer = LongCatAudioDiTTransformer(config)

        # WAV-VAE
        self.vae = LongCatAudioDiTVae(config.vae_config)
        self.vae.requires_grad_(False)

        self.post_init()

    def encode_text(
        self,
        input_ids: torch.LongTensor,
        attention_mask: torch.LongTensor,
    ) -> torch.FloatTensor:
        """Encode tokenized text using the UMT5 text encoder.

        Args:
            input_ids: Token ids ``(batch, seq_len)``.
            attention_mask: Attention mask ``(batch, seq_len)``.

        Returns:
            Text embeddings ``(batch, seq_len, text_dim)`` in float32.
        """
        with torch.no_grad():
            output = self.text_encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
            )
        emb = output.last_hidden_state
        d_model = self.text_encoder.config.d_model

        if self.config.text_norm_feat:
            emb = F.layer_norm(emb, (d_model,), eps=1e-6)

        if self.config.text_add_embed:
            first_hidden = output.hidden_states[0]
            if self.config.text_norm_feat:
                first_hidden = F.layer_norm(first_hidden, (d_model,), eps=1e-6)
            emb = emb + first_hidden

        return emb.float()

    def encode_prompt_audio(
        self, prompt_audio: torch.FloatTensor
    ) -> tuple[torch.FloatTensor, int]:
        """Encode prompt audio to latent space.

        Args:
            prompt_audio: Waveform tensor ``(batch, 1, num_samples)`` or ``(batch, num_samples)``.

        Returns:
            Tuple of (prompt_latent ``(batch, num_frames, latent_dim)``, prompt_duration_frames).
        """
        full_hop = self.config.latent_hop
        off = 3
        wav = prompt_audio.to(self.device)
        if wav.ndim == 2:
            wav = wav.unsqueeze(1)
        if wav.shape[-1] % full_hop != 0:
            wav = F.pad(wav, (0, full_hop - wav.shape[-1] % full_hop))
        wav = F.pad(wav, (0, full_hop * off))
        latent = self.vae.encode(wav)
        if off != 0:
            latent = latent[..., :-off]
        prompt_duration_frames = latent.shape[-1]
        return latent.permute(0, 2, 1), prompt_duration_frames

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.LongTensor | None = None,
        text_embedding: torch.FloatTensor | None = None,
        prompt_audio: torch.FloatTensor | None = None,
        duration: int | None = None,
        steps: int = 16,
        cfg_strength: float = 4.0,
        guidance_method: str = "cfg",
        return_dict: bool = True,
    ) -> LongCatAudioDiTOutput | tuple:
        """Generate audio from text (and optional prompt audio).

        Args:
            input_ids: Tokenized text ``(batch, seq_len)``. Use with ``attention_mask``.
            attention_mask: Attention mask ``(batch, seq_len)``.
            text_embedding: Pre-computed text embeddings ``(batch, seq_len, dim)``. Alternative to input_ids.
            prompt_audio: Optional prompt audio ``(batch, 1, num_samples)`` for voice cloning.
            duration: Target duration in latent frames (prompt + gen). If None, uses max_wav_duration.
            steps: Number of ODE Euler steps (default 16).
            cfg_strength: Guidance strength for CFG/APG (default 4.0).
            guidance_method: ``"cfg"`` or ``"apg"`` (default ``"cfg"``).
            return_dict: Whether to return ``LongCatAudioDiTOutput`` or tuple.
        """
        device = self.device
        sr = self.config.sampling_rate
        full_hop = self.config.latent_hop
        max_duration_frames = int(self.config.max_wav_duration * sr // full_hop)
        repa_layer = self.config.repa_dit_layer

        # ── text encoding ─────────────────────────────────────────────
        if text_embedding is not None:
            text_condition = text_embedding.to(device, torch.float32)
            if attention_mask is not None:
                text_condition_len = attention_mask.sum(dim=1).to(device)
            else:
                text_condition_len = torch.full(
                    (text_condition.shape[0],),
                    text_condition.shape[1],
                    device=device,
                )
        else:
            text_condition = self.encode_text(
                input_ids.to(device),
                attention_mask.to(device),
            )
            text_condition_len = attention_mask.sum(dim=1).to(device)

        batch = text_condition.shape[0]

        # ── prompt audio encoding ─────────────────────────────────────
        if prompt_audio is not None:
            prompt_latent, prompt_dur = self.encode_prompt_audio(prompt_audio)
        else:
            prompt_latent = torch.empty(batch, 0, self.config.latent_dim, device=device)
            prompt_dur = 0

        # ── duration ──────────────────────────────────────────────────
        if duration is None:
            duration = max_duration_frames
        total_duration = min(duration, max_duration_frames)

        # ── masks & conditioning ──────────────────────────────────────
        duration_tensor = torch.full(
            (batch,), total_duration, device=device, dtype=torch.long
        )
        max_dur = total_duration
        mask = lens_to_mask(duration_tensor)
        text_mask = lens_to_mask(text_condition_len, length=text_condition.shape[1])

        neg_text = torch.zeros_like(text_condition)
        neg_text_len = text_condition_len

        latent_len = prompt_dur
        if prompt_audio is not None:
            gen_len = max_dur - latent_len
            latent_cond = F.pad(prompt_latent, (0, 0, 0, gen_len))
            empty_latent_cond = torch.zeros_like(latent_cond)
        else:
            latent_cond = torch.zeros(
                batch, max_dur, self.config.latent_dim, device=device
            )
            empty_latent_cond = latent_cond

        # ── APG buffer ────────────────────────────────────────────────
        if guidance_method == "apg":
            apg_buffer = _MomentumBuffer(momentum=-0.3)

        # ── ODE function ──────────────────────────────────────────────
        def fn(t, x):
            x[:, :latent_len] = prompt_noise * (1 - t) + latent_cond[:, :latent_len] * t
            output = self.transformer(
                x=x,
                text=text_condition,
                text_len=text_condition_len,
                time=t,
                mask=mask,
                cond_mask=text_mask,
                return_ith_layer=repa_layer,
                latent_cond=latent_cond,
            )
            pred = output["last_hidden_state"]

            if cfg_strength < 1e-5:
                return pred

            x[:, :latent_len] = 0
            null_output = self.transformer(
                x=x,
                text=neg_text,
                text_len=neg_text_len,
                time=t,
                mask=mask,
                cond_mask=text_mask,
                return_ith_layer=repa_layer,
                latent_cond=empty_latent_cond,
            )
            null_pred = null_output["last_hidden_state"]

            if guidance_method == "cfg":
                return pred + (pred - null_pred) * cfg_strength

            # APG
            x_s = x[:, latent_len:]
            pred_s = pred[:, latent_len:]
            null_s = null_pred[:, latent_len:]
            pred_sample = x_s + (1 - t) * pred_s
            null_sample = x_s + (1 - t) * null_s
            out = _apg_forward(
                pred_sample,
                null_sample,
                cfg_strength,
                apg_buffer,
                eta=0.5,
                norm_threshold=0.0,
                dims=[-1, -2],
            )
            out = (out - x_s) / (1 - t)
            return F.pad(out, (0, 0, latent_len, 0), value=0.0)

        # ── initial noise ─────────────────────────────────────────────
        y0 = []
        for dur in duration_tensor:
            noise = torch.randn(dur.item(), self.config.latent_dim, device=device)
            y0.append(noise)
        y0 = pad_sequence(y0, padding_value=0, batch_first=True)

        # ── ODE solve ─────────────────────────────────────────────────
        t = torch.linspace(0, 1, steps, device=device)
        prompt_noise = y0[:, :latent_len].clone()
        trajectory = odeint_euler(fn, y0, t)
        sampled = trajectory[-1]

        # ── decode ────────────────────────────────────────────────────
        pred_latent = sampled
        if prompt_audio is not None:
            pred_latent = pred_latent[:, prompt_dur:]

        pred_latent = pred_latent.permute(0, 2, 1).float()
        waveform = self.vae.decode(pred_latent).squeeze(1)

        if not return_dict:
            return (waveform, pred_latent)
        return LongCatAudioDiTOutput(waveform=waveform, latent=pred_latent)


# ---------------------------------------------------------------------------
# APG helpers (from model/cfm.py — Adaptive Projected Guidance)
# ---------------------------------------------------------------------------


class _MomentumBuffer:
    def __init__(self, momentum: float = -0.75):
        self.momentum = momentum
        self.running_average = 0

    def update(self, update_value: torch.Tensor):
        new_average = self.momentum * self.running_average
        self.running_average = update_value + new_average


def _project(v0: torch.Tensor, v1: torch.Tensor, dims=(-1, -2)):
    dtype = v0.dtype
    device_type = v0.device.type
    if device_type == "mps":
        v0, v1 = v0.cpu(), v1.cpu()
    v0, v1 = v0.double(), v1.double()
    v1 = F.normalize(v1, dim=dims)
    v0_parallel = (v0 * v1).sum(dim=dims, keepdim=True) * v1
    v0_orthogonal = v0 - v0_parallel
    return v0_parallel.to(dtype).to(device_type), v0_orthogonal.to(dtype).to(
        device_type
    )


def _apg_forward(
    pred_cond,
    pred_uncond,
    guidance_scale,
    momentum_buffer=None,
    eta=0.0,
    norm_threshold=2.5,
    dims=(-1, -2),
):
    diff = pred_cond - pred_uncond
    if momentum_buffer is not None:
        momentum_buffer.update(diff)
        diff = momentum_buffer.running_average
    if norm_threshold > 0:
        ones = torch.ones_like(diff)
        diff_norm = diff.norm(p=2, dim=dims, keepdim=True)
        scale_factor = torch.minimum(ones, norm_threshold / diff_norm)
        diff = diff * scale_factor
    diff_parallel, diff_orthogonal = _project(diff, pred_cond, dims)
    normalized_update = diff_orthogonal + eta * diff_parallel
    return pred_cond + guidance_scale * normalized_update


__all__ = [
    "LongCatAudioDiTConfig",
    "LongCatAudioDiTVaeConfig",
    "LongCatAudioDiTOutput",
    "LongCatAudioDiTPreTrainedModel",
    "LongCatAudioDiTModel",
    "LongCatAudioDiTTransformer",
    "LongCatAudioDiTVae",
]
