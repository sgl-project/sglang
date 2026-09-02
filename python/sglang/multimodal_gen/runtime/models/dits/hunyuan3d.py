# Copied and adapted from: https://github.com/Tencent-Hunyuan/Hunyuan3D-2
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from sglang.multimodal_gen.configs.models.dits.hunyuan3d import (
    Hunyuan3DDiTArchConfig,
    Hunyuan3DDiTConfig,
)
from sglang.multimodal_gen.runtime.distributed import divide
from sglang.multimodal_gen.runtime.distributed.parallel_state import get_tp_world_size
from sglang.multimodal_gen.runtime.layers.attention import LocalAttention
from sglang.multimodal_gen.runtime.layers.layernorm import (
    LayerNormScaleShift,
    ScaleResidualLayerNormScaleShift,
    apply_qk_norm,
)
from sglang.multimodal_gen.runtime.layers.linear import (
    MergedColumnParallelLinear,
    RowParallelLinear,
)
from sglang.multimodal_gen.runtime.layers.mlp import MLP
from sglang.multimodal_gen.runtime.managers.memory_managers.layerwise_offload import (
    LayerwiseOffloadableModuleMixin,
)
from sglang.multimodal_gen.runtime.models.dits.base import CachableDiT
from sglang.multimodal_gen.runtime.platforms import AttentionBackendEnum
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

logger = init_logger(__name__)


def _fused_add_gate(
    residual: torch.Tensor, x: torch.Tensor, gate: torch.Tensor
) -> torch.Tensor:
    return torch.addcmul(residual, x, gate)


class MixedRowParallelLinear(RowParallelLinear):
    """RowParallel for inputs concatenated from multiple separately-sharded sources."""

    def __init__(self, input_sizes: list[int], output_size: int, **kwargs):
        self.input_sizes = input_sizes
        super().__init__(sum(input_sizes), output_size, **kwargs)

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        input_dim = getattr(param, "input_dim", None)
        if input_dim is not None:
            shards = []
            offset = 0
            for sz in self.input_sizes:
                part = loaded_weight.narrow(input_dim, offset, sz)
                per_rank = sz // self.tp_size
                shard = part.narrow(input_dim, self.tp_rank * per_rank, per_rank)
                shards.append(shard)
                offset += sz
            param.data.copy_(torch.cat(shards, dim=input_dim))
        else:
            param.data.copy_(loaded_weight)


def _flux_timestep_embedding(
    t: torch.Tensor, dim, max_period=10000, time_factor: float = 1000.0
):
    """Create sinusoidal timestep embeddings for Flux-style model."""
    t = time_factor * t
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period)
        * torch.arange(start=0, end=half, dtype=torch.float32)
        / half
    ).to(t.device)

    args = t[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    if torch.is_floating_point(t):
        embedding = embedding.to(t)
    return embedding


class _FluxGELU(nn.Module):
    def __init__(self, approximate="tanh"):
        super().__init__()
        self.approximate = approximate

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.gelu(x, approximate=self.approximate)


class _FluxMLPEmbedder(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int):
        super().__init__()
        self.in_layer = nn.Linear(in_dim, hidden_dim, bias=True)
        self.silu = nn.SiLU()
        self.out_layer = nn.Linear(hidden_dim, hidden_dim, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.out_layer(self.silu(self.in_layer(x)))


class _FluxRMSNorm(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(dim))
        self.variance_epsilon = 1e-6
        self.hidden_size = dim

    @property
    def weight(self) -> nn.Parameter:
        # Keep the original checkpoint key (`scale`) while exposing the
        # interface expected by the fused QK-norm helper.
        return self.scale

    def forward(self, x: torch.Tensor):
        x_dtype = x.dtype
        x = x.float()
        rrms = torch.rsqrt(
            torch.mean(x**2, dim=-1, keepdim=True) + self.variance_epsilon
        )
        return (x * rrms).to(dtype=x_dtype) * self.scale


class _FluxQKNorm(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        self.query_norm = _FluxRMSNorm(dim)
        self.key_norm = _FluxRMSNorm(dim)

    def forward(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        q, k = apply_qk_norm(
            q=q.contiguous(),
            k=k.contiguous(),
            q_norm=self.query_norm,
            k_norm=self.key_norm,
            head_dim=self.dim,
            allow_inplace=True,
        )
        return q.to(v), k.to(v)


class _FluxSelfAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        supported_attention_backends: set[AttentionBackendEnum] | None = None,
    ):
        super().__init__()
        tp_size = get_tp_world_size()
        self.num_heads = num_heads
        self.local_num_heads = divide(num_heads, tp_size)
        self.head_dim = dim // num_heads

        self.qkv = MergedColumnParallelLinear(
            dim, [dim, dim, dim], bias=qkv_bias, gather_output=False
        )
        self.norm = _FluxQKNorm(self.head_dim)
        self.proj = RowParallelLinear(dim, dim, bias=True, input_is_parallel=True)

        if supported_attention_backends is None:
            supported_attention_backends = {
                AttentionBackendEnum.FA,
                AttentionBackendEnum.TORCH_SDPA,
            }
        self.local_attn = LocalAttention(
            num_heads=self.local_num_heads,
            head_size=self.head_dim,
            causal=False,
            supported_attention_backends=supported_attention_backends,
        )

    def forward(self, x: torch.Tensor, pe: torch.Tensor) -> torch.Tensor:
        qkv, _ = self.qkv(x)
        B, L, _ = qkv.shape
        qkv = qkv.view(B, L, 3, self.local_num_heads, self.head_dim)
        q, k, v = qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v_for_norm = v.transpose(1, 2)
        q, k = self.norm(q, k, v_for_norm)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        x = self.local_attn(q, k, v)
        x = x.flatten(2)
        x, _ = self.proj(x)
        return x


@dataclass
class _FluxModulationOut:
    shift: torch.Tensor
    scale: torch.Tensor
    gate: torch.Tensor


class _FluxModulation(nn.Module):
    def __init__(self, dim: int, double: bool):
        super().__init__()
        self.is_double = double
        self.multiplier = 6 if double else 3
        self.lin = nn.Linear(dim, self.multiplier * dim, bias=True)

    def forward(
        self, vec: torch.Tensor
    ) -> Tuple[_FluxModulationOut, Optional[_FluxModulationOut]]:
        out = self.lin(F.silu(vec))[:, None, :]
        out = out.chunk(self.multiplier, dim=-1)

        return (
            _FluxModulationOut(*out[:3]),
            _FluxModulationOut(*out[3:]) if self.is_double else None,
        )


class _FluxDoubleStreamBlock(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        mlp_ratio: float,
        qkv_bias: bool = False,
        supported_attention_backends: set[AttentionBackendEnum] | None = None,
    ):
        super().__init__()
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        tp_size = get_tp_world_size()
        self.num_heads = num_heads
        self.local_num_heads = divide(num_heads, tp_size)
        self.hidden_size = hidden_size
        self.head_dim = hidden_size // num_heads
        self.img_mod = _FluxModulation(hidden_size, double=True)
        self.img_norm1 = LayerNormScaleShift(
            hidden_size, elementwise_affine=False, eps=1e-6
        )
        self.img_attn = _FluxSelfAttention(
            dim=hidden_size,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            supported_attention_backends=supported_attention_backends,
        )

        self.img_norm2 = ScaleResidualLayerNormScaleShift(
            hidden_size, elementwise_affine=False, eps=1e-6
        )
        self.img_mlp = MLP(hidden_size, mlp_hidden_dim, act_type="gelu_pytorch_tanh")

        self.txt_mod = _FluxModulation(hidden_size, double=True)
        self.txt_norm1 = LayerNormScaleShift(
            hidden_size, elementwise_affine=False, eps=1e-6
        )
        self.txt_attn = _FluxSelfAttention(
            dim=hidden_size,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            supported_attention_backends=supported_attention_backends,
        )

        self.txt_norm2 = ScaleResidualLayerNormScaleShift(
            hidden_size, elementwise_affine=False, eps=1e-6
        )
        self.txt_mlp = MLP(hidden_size, mlp_hidden_dim, act_type="gelu_pytorch_tanh")

        if supported_attention_backends is None:
            supported_attention_backends = {
                AttentionBackendEnum.FA,
                AttentionBackendEnum.TORCH_SDPA,
            }
        self.local_attn_joint = LocalAttention(
            num_heads=self.local_num_heads,
            head_size=self.head_dim,
            causal=False,
            supported_attention_backends=supported_attention_backends,
        )

    def forward(
        self, img: torch.Tensor, txt: torch.Tensor, vec: torch.Tensor, pe: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        img_mod1, img_mod2 = self.img_mod(vec)
        txt_mod1, txt_mod2 = self.txt_mod(vec)

        img_modulated = self.img_norm1(img, shift=img_mod1.shift, scale=img_mod1.scale)

        B, img_L, _ = img_modulated.shape
        img_qkv, _ = self.img_attn.qkv(img_modulated)
        img_qkv = img_qkv.view(B, img_L, 3, self.local_num_heads, self.head_dim)
        img_q, img_k, img_v = img_qkv[:, :, 0], img_qkv[:, :, 1], img_qkv[:, :, 2]
        img_q_t = img_q.transpose(1, 2)
        img_k_t = img_k.transpose(1, 2)
        img_v_t = img_v.transpose(1, 2)
        img_q_t, img_k_t = self.img_attn.norm(img_q_t, img_k_t, img_v_t)
        img_q = img_q_t.transpose(1, 2)
        img_k = img_k_t.transpose(1, 2)

        txt_modulated = self.txt_norm1(txt, shift=txt_mod1.shift, scale=txt_mod1.scale)
        txt_L = txt_modulated.shape[1]
        txt_qkv, _ = self.txt_attn.qkv(txt_modulated)
        txt_qkv = txt_qkv.view(B, txt_L, 3, self.local_num_heads, self.head_dim)
        txt_q, txt_k, txt_v = txt_qkv[:, :, 0], txt_qkv[:, :, 1], txt_qkv[:, :, 2]
        txt_q_t = txt_q.transpose(1, 2)
        txt_k_t = txt_k.transpose(1, 2)
        txt_v_t = txt_v.transpose(1, 2)
        txt_q_t, txt_k_t = self.txt_attn.norm(txt_q_t, txt_k_t, txt_v_t)
        txt_q = txt_q_t.transpose(1, 2)
        txt_k = txt_k_t.transpose(1, 2)

        q = torch.cat((txt_q, img_q), dim=1)
        k = torch.cat((txt_k, img_k), dim=1)
        v = torch.cat((txt_v, img_v), dim=1)

        attn = self.local_attn_joint(q, k, v)
        attn = attn.flatten(2)

        txt_attn, img_attn = attn[:, :txt_L], attn[:, txt_L:]

        img_proj, _ = self.img_attn.proj(img_attn)
        img_modulated, img = self.img_norm2(
            residual=img,
            x=img_proj,
            gate=img_mod1.gate,
            shift=img_mod2.shift,
            scale=img_mod2.scale,
        )
        img = _fused_add_gate(img, self.img_mlp(img_modulated), img_mod2.gate)

        txt_proj, _ = self.txt_attn.proj(txt_attn)
        txt_modulated, txt = self.txt_norm2(
            residual=txt,
            x=txt_proj,
            gate=txt_mod1.gate,
            shift=txt_mod2.shift,
            scale=txt_mod2.scale,
        )
        txt = _fused_add_gate(txt, self.txt_mlp(txt_modulated), txt_mod2.gate)
        return img, txt


class _FluxSingleStreamBlock(nn.Module):
    """
    A DiT block with parallel linear layers as described in
    https://arxiv.org/abs/2302.05442 and adapted modulation interface.
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qk_scale: Optional[float] = None,
        supported_attention_backends: set[AttentionBackendEnum] | None = None,
    ):
        super().__init__()

        tp_size = get_tp_world_size()
        self.hidden_dim = hidden_size
        self.num_heads = num_heads
        self.local_num_heads = divide(num_heads, tp_size)
        self.head_dim = hidden_size // num_heads
        self.tp_size = tp_size

        self.mlp_hidden_dim = int(hidden_size * mlp_ratio)
        self.linear1 = MergedColumnParallelLinear(
            hidden_size,
            [hidden_size, hidden_size, hidden_size, self.mlp_hidden_dim],
            bias=True,
            gather_output=False,
        )
        self.linear2 = MixedRowParallelLinear(
            [hidden_size, self.mlp_hidden_dim],
            hidden_size,
            bias=True,
            input_is_parallel=True,
        )

        self.norm = _FluxQKNorm(self.head_dim)

        self.hidden_size = hidden_size
        self.pre_norm = LayerNormScaleShift(
            hidden_size, elementwise_affine=False, eps=1e-6
        )

        self.mlp_act = _FluxGELU(approximate="tanh")
        self.modulation = _FluxModulation(hidden_size, double=False)

        if supported_attention_backends is None:
            supported_attention_backends = {
                AttentionBackendEnum.FA,
                AttentionBackendEnum.TORCH_SDPA,
            }
        self.local_attn = LocalAttention(
            num_heads=self.local_num_heads,
            head_size=self.head_dim,
            causal=False,
            supported_attention_backends=supported_attention_backends,
        )

    def forward(
        self, x: torch.Tensor, vec: torch.Tensor, pe: torch.Tensor
    ) -> torch.Tensor:
        mod, _ = self.modulation(vec)

        x_mod = self.pre_norm(x, shift=mod.shift, scale=mod.scale)
        linear1_out, _ = self.linear1(x_mod)
        local_qkv_dim = 3 * self.head_dim * self.local_num_heads
        local_mlp_dim = self.mlp_hidden_dim // self.tp_size
        qkv, mlp = torch.split(linear1_out, [local_qkv_dim, local_mlp_dim], dim=-1)

        B, L, _ = qkv.shape
        qkv = qkv.view(B, L, 3, self.local_num_heads, self.head_dim)
        q, k, v = qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]
        q_t = q.transpose(1, 2)
        k_t = k.transpose(1, 2)
        v_t = v.transpose(1, 2)
        q_t, k_t = self.norm(q_t, k_t, v_t)
        q = q_t.transpose(1, 2)
        k = k_t.transpose(1, 2)

        attn = self.local_attn(q, k, v)
        attn = attn.flatten(2)

        output, _ = self.linear2(torch.cat((attn, self.mlp_act(mlp)), 2))
        return _fused_add_gate(x, output, mod.gate)


class _FluxLastLayer(nn.Module):
    def __init__(self, hidden_size: int, patch_size: int, out_channels: int):
        super().__init__()
        self.norm_final = LayerNormScaleShift(
            hidden_size, elementwise_affine=False, eps=1e-6
        )
        self.linear = nn.Linear(
            hidden_size, patch_size * patch_size * out_channels, bias=True
        )
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )

    def forward(self, x: torch.Tensor, vec: torch.Tensor) -> torch.Tensor:
        shift, scale = self.adaLN_modulation(vec).chunk(2, dim=1)
        x = self.norm_final(x, shift=shift[:, None, :], scale=scale[:, None, :])
        x = self.linear(x)
        return x


class Hunyuan3D2DiT(CachableDiT, LayerwiseOffloadableModuleMixin):
    """Hunyuan3D DiT model (Flux-style architecture for Hunyuan3D-2.0)."""

    _aliases = ["hy3dgen.shapegen.models.Hunyuan3DDiT"]

    param_names_mapping = Hunyuan3DDiTConfig().param_names_mapping

    @classmethod
    def build_config_from_params(cls, params: dict) -> Hunyuan3DDiTConfig:
        """Build a DiTConfig from YAML-style parameter dict."""
        field_mapping = {
            "num_heads": "num_attention_heads",
            "depth": "num_layers",
            "depth_single_blocks": "num_single_layers",
        }
        arch_kwargs = {}
        for k, v in params.items():
            if k in ("ckpt_path", "supported_attention_backends"):
                continue
            mapped = field_mapping.get(k, k)
            if k == "axes_dim" and isinstance(v, list):
                v = tuple(v)
            arch_kwargs[mapped] = v
        return Hunyuan3DDiTConfig(arch_config=Hunyuan3DDiTArchConfig(**arch_kwargs))

    def __init__(
        self,
        config: Hunyuan3DDiTConfig,
        hf_config: dict | None = None,
        **kwargs,
    ):
        super().__init__(config=config, hf_config=hf_config or {}, **kwargs)
        arch = self.config

        in_channels = arch.in_channels
        context_in_dim = arch.context_in_dim
        hidden_size = arch.hidden_size
        mlp_ratio = arch.mlp_ratio
        num_heads = arch.num_attention_heads
        depth = arch.num_layers
        depth_single_blocks = arch.num_single_layers
        axes_dim = list(arch.axes_dim)
        theta = arch.theta
        qkv_bias = arch.qkv_bias
        time_factor = arch.time_factor
        guidance_embed = arch.guidance_embed
        supported_attention_backends = self._supported_attention_backends

        self.in_channels = in_channels
        self.context_in_dim = context_in_dim
        self.hidden_size = hidden_size
        self.mlp_ratio = mlp_ratio
        self.num_heads = num_heads
        self.num_attention_heads = num_heads
        self.depth = depth
        self.depth_single_blocks = depth_single_blocks
        self.axes_dim = axes_dim
        self.theta = theta
        self.qkv_bias = qkv_bias
        self.time_factor = time_factor
        self.out_channels = self.in_channels
        self.num_channels_latents = self.in_channels
        self.guidance_embed = guidance_embed

        if hidden_size % num_heads != 0:
            raise ValueError(
                f"Hidden size {hidden_size} must be divisible by num_heads {num_heads}"
            )
        pe_dim = hidden_size // num_heads
        if sum(axes_dim) != pe_dim:
            raise ValueError(f"Got {axes_dim} but expected positional dim {pe_dim}")
        self.latent_in = nn.Linear(self.in_channels, self.hidden_size, bias=True)
        self.time_in = _FluxMLPEmbedder(in_dim=256, hidden_dim=self.hidden_size)
        self.cond_in = nn.Linear(context_in_dim, self.hidden_size)
        self.guidance_in = (
            _FluxMLPEmbedder(in_dim=256, hidden_dim=self.hidden_size)
            if guidance_embed
            else nn.Identity()
        )

        self.double_blocks = nn.ModuleList(
            [
                _FluxDoubleStreamBlock(
                    self.hidden_size,
                    self.num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    supported_attention_backends=supported_attention_backends,
                )
                for _ in range(depth)
            ]
        )

        self.single_blocks = nn.ModuleList(
            [
                _FluxSingleStreamBlock(
                    self.hidden_size,
                    self.num_heads,
                    mlp_ratio=mlp_ratio,
                    supported_attention_backends=supported_attention_backends,
                )
                for _ in range(depth_single_blocks)
            ]
        )

        self.final_layer = _FluxLastLayer(self.hidden_size, 1, self.out_channels)

        # LayerwiseOffloadableModuleMixin
        self.layer_names = ["double_blocks", "single_blocks"]

    def forward(
        self,
        x,
        t,
        contexts,
        **kwargs,
    ) -> torch.Tensor:
        """Forward pass for denoising."""

        cond = contexts["main"]

        latent = self.latent_in(x)

        t_emb = _flux_timestep_embedding(t, 256, self.time_factor).to(
            dtype=latent.dtype
        )

        vec = self.time_in(t_emb)

        if self.guidance_embed:
            guidance = kwargs.get("guidance", None)
            if guidance is None:
                raise ValueError(
                    "Didn't get guidance strength for guidance distilled model."
                )
            vec = vec + self.guidance_in(
                _flux_timestep_embedding(guidance, 256, self.time_factor)
            )

        cond = self.cond_in(cond)

        pe = None

        # Double blocks
        for i, block in enumerate(self.double_blocks):
            latent, cond = block(img=latent, txt=cond, vec=vec, pe=pe)
        latent = torch.cat((cond, latent), 1)

        # Single blocks
        for i, block in enumerate(self.single_blocks):
            latent = block(latent, vec=vec, pe=pe)

        latent = latent[:, cond.shape[1] :, ...]
        latent = self.final_layer(latent, vec)
        return latent


EntryClass = Hunyuan3D2DiT
