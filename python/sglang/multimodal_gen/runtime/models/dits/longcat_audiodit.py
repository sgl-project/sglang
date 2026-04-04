# Copied and adapted from: https://github.com/meituan-longcat/LongCat-AudioDiT
"""PyTorch LongCatAudioDiT model — Conditional Flow Matching TTS with DiT backbone."""

import math

import torch
import torch.nn.functional as F
from torch import nn
from transformers import PreTrainedModel, logging

from sglang.multimodal_gen.configs.models.dits.base import DiTArchConfig, DiTConfig
from sglang.multimodal_gen.configs.models.dits.longcat_audiodit import (
    LongCatAudioDiTConfig,
)
from sglang.multimodal_gen.runtime.layers.attention import USPAttention
from sglang.multimodal_gen.runtime.managers.memory_managers.layerwise_offload import (
    LayerwiseOffloadableModuleMixin,
)
from sglang.multimodal_gen.runtime.models.dits.base import BaseDiT
from sglang.multimodal_gen.runtime.models.vaes.longcat_audiodit_vae import (
    LongCatAudioDiTVae,
)
from sglang.multimodal_gen.runtime.platforms import AttentionBackendEnum

logger = logging.get_logger(__name__)


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


def _run_usp_attention(
    attn: USPAttention,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    heads: int,
    head_dim: int,
    *,
    rope: tuple | None = None,
    key_rope: tuple | None = None,
    attn_mask: torch.BoolTensor | None = None,
) -> torch.Tensor:
    """Project ``[B, S, H*D]`` QKV to USPAttention layout ``[B, S, H, D]``.

    RoPE stays on ``[B, H, S, D]`` (``_apply_rotary_emb`` broadcasts ``[S, D]``
    as ``[1, 1, S, D]``).
    """
    batch = query.shape[0]
    q = query.view(batch, -1, heads, head_dim)
    k = key.view(batch, -1, heads, head_dim)
    v = value.view(batch, -1, heads, head_dim)
    if rope is not None:
        q = _apply_rotary_emb(q.transpose(1, 2), rope).transpose(1, 2)
    if key_rope is not None:
        k = _apply_rotary_emb(k.transpose(1, 2), key_rope).transpose(1, 2)
    out = attn(q, k, v, attn_mask=attn_mask)
    return out.flatten(2, 3).to(q.dtype)


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
        supported_attention_backends: set[AttentionBackendEnum] | None = None,
    ):
        super().__init__()
        self.heads = heads
        self.head_dim = dim_head
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
        self.attn = USPAttention(
            num_heads=heads,
            head_size=dim_head,
            causal=False,
            dropout_rate=0.0,
            supported_attention_backends=supported_attention_backends,
            # 1D audio latents are not sharded for SP; keep attention replicated.
            skip_sequence_parallel=True,
        )

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.BoolTensor | None = None,
        rope: tuple | None = None,
    ) -> torch.Tensor:
        query = self.to_q(x)
        key = self.to_k(x)
        value = self.to_v(x)
        if self.qk_norm:
            query = self.q_norm(query)
            key = self.k_norm(key)
        x = _run_usp_attention(
            self.attn,
            query,
            key,
            value,
            self.heads,
            self.head_dim,
            rope=rope,
            key_rope=rope,
            attn_mask=mask,
        )
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
        supported_attention_backends: set[AttentionBackendEnum] | None = None,
    ):
        super().__init__()
        self.heads = heads
        self.head_dim = dim_head
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
        self.attn = USPAttention(
            num_heads=heads,
            head_size=dim_head,
            causal=False,
            dropout_rate=0.0,
            skip_sequence_parallel=True,
            supported_attention_backends=supported_attention_backends,
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
        query = self.to_q(x)
        key = self.to_k(cond)
        value = self.to_v(cond)
        if self.qk_norm:
            query = self.q_norm(query)
            key = self.k_norm(key)
        x = _run_usp_attention(
            self.attn,
            query,
            key,
            value,
            self.heads,
            self.head_dim,
            rope=rope,
            key_rope=cond_rope,
            attn_mask=cond_mask,
        )
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

    def __init__(
        self,
        config: LongCatAudioDiTConfig,
        supported_attention_backends: set[AttentionBackendEnum] | None = None,
    ):
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
            supported_attention_backends=supported_attention_backends,
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
                supported_attention_backends=supported_attention_backends,
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
                if cond_mask is not None:
                    cond_mean = cond.sum(1) / cond_mask.sum(1, keepdim=True)
                else:
                    cond_mean = cond.mean(1)
                norm_cond = t + cond_mean
            else:
                norm_cond = t
            adaln_out = self.adaln_mlp(norm_cond)
            gate_sa, scale_sa, shift_sa, gate_ffn, scale_ffn, shift_ffn = torch.chunk(
                adaln_out, 6, dim=-1
            )
        else:
            adaln_out = adaln_global_out + self.adaln_scale_shift.unsqueeze(0)
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


class LongCatAudioDiTTransformer(BaseDiT, LayerwiseOffloadableModuleMixin):
    """DiT backbone for LongCatAudioDiT.

    Single-GPU only: 1D latents are not sequence-sharded, and both self- and
    cross-attention skip USP. Supported attention backends are FA and
    TORCH_SDPA.
    """

    _supported_attention_backends = {
        AttentionBackendEnum.FA,
        AttentionBackendEnum.TORCH_SDPA,
    }
    # -- BaseDiT required class attributes -------------------------------------
    _fsdp_shard_conditions: list = []
    _compile_conditions: list = []
    param_names_mapping: dict = {}
    reverse_param_names_mapping: dict = {}

    def __init__(self, config: LongCatAudioDiTConfig, **kwargs):
        # Wrap HF config into DiTConfig for BaseDiT compatibility.
        dit_config = DiTConfig(
            arch_config=DiTArchConfig(
                hidden_size=config.dit_dim,
                num_attention_heads=config.dit_heads,
                num_channels_latents=config.latent_dim,
            )
        )
        super().__init__(config=dit_config, hf_config={})

        dim = config.dit_dim
        latent_dim = config.latent_dim  # 64
        text_dim = config.dit_text_dim
        dim_head = dim // config.dit_heads

        self.audio_config = config
        self.dim = dim
        self.depth = config.dit_depth
        self.long_skip = config.dit_long_skip
        self.adaln_type = config.dit_adaln_type
        self.adaln_use_text_cond = config.dit_adaln_use_text_cond

        # BaseDiT instance attributes
        self.hidden_size = dim
        self.num_attention_heads = config.dit_heads
        self.num_channels_latents = latent_dim
        self.layer_names = ["blocks"]
        self.__post_init__()

        self.time_embed = LongCatAudioDiTTimestepEmbedding(dim)
        self.input_embed = LongCatAudioDiTEmbedder(latent_dim, dim)
        self.text_embed = LongCatAudioDiTEmbedder(text_dim, dim)
        self.rotary_embed = LongCatAudioDiTRotaryEmbedding(
            dim_head, 2048, base=100000.0
        )

        self.blocks = nn.ModuleList(
            [
                LongCatAudioDiTBlock(
                    config,
                    supported_attention_backends=self._supported_attention_backends,
                )
                for _ in range(config.dit_depth)
            ]
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
        bias = self.audio_config.dit_bias
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
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        timestep: torch.LongTensor,
        text_len: torch.Tensor | None = None,
        mask: torch.BoolTensor | None = None,
        cond_mask: torch.BoolTensor | None = None,
        return_ith_layer: int | None = None,
        latent_cond: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        x = hidden_states
        text = encoder_hidden_states
        time = timestep

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
            if cond_mask is not None:
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
        # Store hidden_state for REPA if requested; return the prediction tensor.
        self._last_hidden_state = hidden_state
        return output


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
    and their weights are loaded together via ``from_pretrained``.  Inference
    is driven by ``LongCatAudioDiTPipeline`` (encode helpers + standard
    ``DenoisingStage`` + WAV-VAE decode).
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
        self,
        prompt_audio: torch.FloatTensor,
        generator: torch.Generator | None = None,
    ) -> tuple[torch.FloatTensor, int]:
        """Encode prompt audio to latent space.

        Args:
            prompt_audio: Waveform tensor ``(batch, 1, num_samples)`` or ``(batch, num_samples)``.
            generator: Optional CPU generator for the VAE posterior sample.

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
        latent = self.vae.encode(wav, generator=generator)
        if off != 0:
            latent = latent[..., :-off]
        prompt_duration_frames = latent.shape[-1]
        return latent.permute(0, 2, 1), prompt_duration_frames

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        timestep: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        """One DiT step. Full TTS / cloning is ``LongCatAudioDiTPipeline``."""
        return self.transformer(
            hidden_states=hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            timestep=timestep,
            **kwargs,
        )


EntryClass = LongCatAudioDiTTransformer
