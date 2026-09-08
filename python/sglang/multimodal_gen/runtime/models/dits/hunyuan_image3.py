"""HunyuanImage-3 AR backbone + diffusion I/O for multimodal_gen.

Ported from the official HunyuanImage-3 repository
(`modeling_hunyuan_image_3.py`).
"""

import re
import types
from typing import Iterable, Optional, Tuple

import torch
import torch.nn.functional as F
from einops import rearrange
from torch import nn
from transformers import PretrainedConfig

from sglang.srt.layers.linear import (
    ColumnParallelLinear,
    MergedColumnParallelLinear,
    QKVParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
)
from sglang.srt.layers.moe.fused_moe_triton.layer import FusedMoE
from sglang.srt.layers.moe.topk import TopK
from sglang.srt.layers.rotary_embedding import get_rope
from sglang.srt.models.hunyuan import (
    _get_cla_factor,
    _is_moe,
)
from sglang.multimodal_gen.runtime.distributed import get_tp_world_size
from sglang.multimodal_gen.runtime.layers.activation import SiluAndMul
from sglang.multimodal_gen.runtime.layers.attention import LocalAttention
from sglang.multimodal_gen.runtime.layers.layernorm import RMSNorm
from sglang.multimodal_gen.runtime.layers.quantization import QuantizationConfig

from sglang.multimodal_gen.runtime.layers.vocab_parallel_embedding import (
    VocabParallelEmbedding,
)
from sglang.multimodal_gen.runtime.loader.weight_utils import default_weight_loader
from sglang.multimodal_gen.runtime.models.dits.base import CachableDiT
from sglang.multimodal_gen.configs.models.dits.hunyuan_image3 import HunyuanImage3DitConfig

from .hunyuan_image3_utils import (
    CachedRoPE,
    HunYuanRotary2DEmbedder,
    ImageKVCacheManager,
    create_hunyuan_image_attention_meta,
    timestep_embedding,
)

from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

logger = init_logger(__name__)

# Checkpoint weight names of the non-AR parts (VAE, ViT), skipped during
# backbone weight loading.
UNEXPECTED_KEYWORDS = [
    "vae",
    "vision_aligner",
    "vision_model",
]

# =============================================================
# Diffusion I/O helper functions and modules
# (ported from official HunyuanImage-3 model repository)
# =============================================================

def _conv_nd(dims, *args, **kwargs):
    """Create a 1D, 2D, or 3D convolution module."""
    if dims == 1:
        return nn.Conv1d(*args, **kwargs)
    elif dims == 2:
        return nn.Conv2d(*args, **kwargs)
    elif dims == 3:
        return nn.Conv3d(*args, **kwargs)
    raise ValueError(f"unsupported dimensions: {dims}")


def _zero_module(module):
    """Zero out the parameters of a module and return it."""
    for p in module.parameters():
        p.detach().zero_()
    return module


def _normalization(channels, **kwargs):
    """GroupNorm normalization."""
    return nn.GroupNorm(32, channels, **kwargs)


class _Upsample(nn.Module):
    """Upsample layer with optional convolution (dims=3 for spatial 2D)."""

    def __init__(self, channels, use_conv, dims=2, out_channels=None, device=None, dtype=None):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.dims = dims
        if use_conv:
            self.conv = _conv_nd(dims, self.channels, self.out_channels, 3, padding=1, **factory_kwargs)

    def forward(self, x):
        assert x.shape[1] == self.channels
        if self.dims == 3:
            x = F.interpolate(
                x, (x.shape[2], x.shape[3] * 2, x.shape[4] * 2), mode="nearest"
            )
        else:
            x = F.interpolate(x, scale_factor=2, mode="nearest")
        if self.use_conv:
            x = self.conv(x)
        return x


class _Downsample(nn.Module):
    """Downsample layer with optional convolution (dims=3 for spatial 2D)."""

    def __init__(self, channels, use_conv, dims=2, out_channels=None, device=None, dtype=None):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.dims = dims
        stride = 2 if dims != 3 else (1, 2, 2)
        if use_conv:
            self.op = _conv_nd(
                dims, self.channels, self.out_channels, 3, stride=stride, padding=1, **factory_kwargs
            )
        else:
            assert self.channels == self.out_channels
            self.op = nn.AvgPool2d(kernel_size=stride, stride=stride)

    def forward(self, x):
        assert x.shape[1] == self.channels
        return self.op(x)


class _ResBlock(nn.Module):
    """Residual block with timestep embedding conditioning."""

    def __init__(
        self, in_channels, emb_channels, out_channels=None, dropout=0.0,
        use_conv=False, dims=2, up=False, down=False, device=None, dtype=None,
    ):
        factory_kwargs = {"dtype": dtype, "device": device}
        super().__init__()
        self.in_channels = in_channels
        self.dropout = dropout
        self.out_channels = out_channels or self.in_channels

        self.in_layers = nn.Sequential(
            _normalization(self.in_channels, **factory_kwargs),
            nn.SiLU(),
            _conv_nd(dims, self.in_channels, self.out_channels, 3, padding=1, **factory_kwargs),
        )

        self.updown = up or down
        if up:
            self.h_upd = _Upsample(self.in_channels, False, dims, **factory_kwargs)
            self.x_upd = _Upsample(self.in_channels, False, dims, **factory_kwargs)
        elif down:
            self.h_upd = _Downsample(self.in_channels, False, dims, **factory_kwargs)
            self.x_upd = _Downsample(self.in_channels, False, dims, **factory_kwargs)
        else:
            self.h_upd = self.x_upd = nn.Identity()

        self.emb_layers = nn.Sequential(
            nn.SiLU(),
            nn.Linear(emb_channels, 2 * self.out_channels, **factory_kwargs),
        )

        self.out_layers = nn.Sequential(
            _normalization(self.out_channels, **factory_kwargs),
            nn.SiLU(),
            nn.Dropout(p=dropout),
            _zero_module(
                _conv_nd(dims, self.out_channels, self.out_channels, 3, padding=1, **factory_kwargs)
            ),
        )

        if self.out_channels == self.in_channels:
            self.skip_connection = nn.Identity()
        elif use_conv:
            self.skip_connection = _conv_nd(
                dims, self.in_channels, self.out_channels, 3, padding=1, **factory_kwargs
            )
        else:
            self.skip_connection = _conv_nd(
                dims, self.in_channels, self.out_channels, 1, **factory_kwargs
            )

    def forward(self, x, emb):
        if self.updown:
            in_rest, in_conv = self.in_layers[:-1], self.in_layers[-1]
            h = in_rest(x)
            h = self.h_upd(h)
            x = self.x_upd(x)
            h = in_conv(h)
        else:
            h = self.in_layers(x)

        emb_out = self.emb_layers(emb)
        while len(emb_out.shape) < len(h.shape):
            emb_out = emb_out[..., None]

        out_norm, out_rest = self.out_layers[0], self.out_layers[1:]
        scale, shift = torch.chunk(emb_out, 2, dim=1)
        h = out_norm(h) * (1.0 + scale) + shift
        h = out_rest(h)

        return self.skip_connection(x) + h


class TimestepEmbedder(nn.Module):
    """Embeds scalar timesteps into vector representations."""

    def __init__(self, hidden_size, act_layer=nn.GELU, frequency_embedding_size=256,
                 max_period=10000, out_size=None, dtype=None, device=None):
        factory_kwargs = {"dtype": dtype, "device": device}
        super().__init__()
        self.frequency_embedding_size = frequency_embedding_size
        self.max_period = max_period
        if out_size is None:
            out_size = hidden_size
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True, **factory_kwargs),
            act_layer(),
            nn.Linear(hidden_size, out_size, bias=True, **factory_kwargs),
        )

    def forward(self, t):
        t_freq = timestep_embedding(t, self.frequency_embedding_size, self.max_period)
        t_freq = t_freq.type(self.mlp[0].weight.dtype)
        return self.mlp(t_freq)


class UNetDown(nn.Module):
    """Patch embed: converts noise latents (B, C, H, W) into sequence embeddings."""

    def __init__(self, patch_size, in_channels, emb_channels, hidden_channels,
                 out_channels, dropout=0.0, device=None, dtype=None):
        factory_kwargs = {"dtype": dtype, "device": device}
        super().__init__()
        self.patch_size = patch_size

        self.model = nn.ModuleList([
            _conv_nd(2, in_channels=in_channels, out_channels=hidden_channels,
                     kernel_size=3, padding=1, **factory_kwargs)
        ])
        if self.patch_size == 1:
            self.model.append(_ResBlock(
                in_channels=hidden_channels, emb_channels=emb_channels,
                out_channels=out_channels, dropout=dropout, **factory_kwargs,
            ))
        else:
            for i in range(self.patch_size // 2):
                self.model.append(_ResBlock(
                    in_channels=hidden_channels, emb_channels=emb_channels,
                    out_channels=(hidden_channels if (i + 1) * 2 != self.patch_size else out_channels),
                    dropout=dropout, down=True, **factory_kwargs,
                ))

    def forward(self, x, t):
        assert x.shape[2] % self.patch_size == 0 and x.shape[3] % self.patch_size == 0
        for module in self.model:
            if isinstance(module, _ResBlock):
                x = module(x, t)
            else:
                x = module(x)
        _, _, token_h, token_w = x.shape
        x = rearrange(x, "b c h w -> b (h w) c")
        return x, token_h, token_w


class UNetUp(nn.Module):
    """Final layer: converts backbone output sequence into noise predictions."""

    def __init__(self, patch_size, in_channels, emb_channels, hidden_channels,
                 out_channels, dropout=0.0, device=None, dtype=None, out_norm=False):
        factory_kwargs = {"dtype": dtype, "device": device}
        super().__init__()
        self.patch_size = patch_size
        self.model = nn.ModuleList()

        if self.patch_size == 1:
            self.model.append(_ResBlock(
                in_channels=in_channels, emb_channels=emb_channels,
                out_channels=hidden_channels, dropout=dropout, **factory_kwargs,
            ))
        else:
            for i in range(self.patch_size // 2):
                self.model.append(_ResBlock(
                    in_channels=in_channels if i == 0 else hidden_channels,
                    emb_channels=emb_channels, out_channels=hidden_channels,
                    dropout=dropout, up=True, **factory_kwargs,
                ))

        if out_norm:
            self.model.append(nn.Sequential(
                _normalization(hidden_channels, **factory_kwargs),
                nn.SiLU(),
                _conv_nd(2, hidden_channels, out_channels, kernel_size=3,
                         padding=1, **factory_kwargs),
            ))
        else:
            self.model.append(_conv_nd(
                2, hidden_channels, out_channels, kernel_size=3,
                padding=1, **factory_kwargs))

    def forward(self, x, t, token_h, token_w):
        x = rearrange(x, "b (h w) c -> b c h w", h=token_h, w=token_w)
        for module in self.model:
            if isinstance(module, _ResBlock):
                x = module(x, t)
            else:
                x = module(x)
        return x


def _make_rope(config: PretrainedConfig, head_dim: int, rope_theta, rope_scaling, max_position):
    if rope_scaling is not None:
        rope_scaling = dict(rope_scaling)
        rope_scaling["rope_type"] = "default"
    return get_rope(
        head_dim,
        rotary_dim=head_dim,
        max_position=max_position,
        base=rope_theta,
        rope_scaling=rope_scaling,
        is_neox_style=True,
    )


def _get_layer_value(config: PretrainedConfig, field: str, layer_id: int, default=None):
    value = getattr(config, field, default)
    if isinstance(value, list):
        assert layer_id >= 0 and len(value) > layer_id, f"{field}[{layer_id}] missing"
        return value[layer_id]
    return value


class HunYuanMLP(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str,
        quant_config: Optional[QuantizationConfig] = None,
        bias: bool = False,
        prefix: str = "",
        reduce_results: bool = True,
    ) -> None:
        super().__init__()
        self.gate_up_proj = MergedColumnParallelLinear(
            input_size=hidden_size,
            output_sizes=[intermediate_size] * 2,
            bias=bias,
            quant_config=quant_config,
            prefix=f"{prefix}.gate_up_proj",
        )
        self.down_proj = RowParallelLinear(
            input_size=intermediate_size,
            output_size=hidden_size,
            bias=bias,
            quant_config=quant_config,
            reduce_results=reduce_results,
            prefix=f"{prefix}.down_proj",
        )
        if hidden_act != "silu":
            raise ValueError(
                f"Unsupported activation: {hidden_act}. Only silu is supported for now."
            )
        self.act_fn = SiluAndMul()

    def forward(self, x):
        gate_up, _ = self.gate_up_proj(x)
        x = self.act_fn(gate_up)
        x, _ = self.down_proj(x)
        return x


class HunYuanSparseMoeBlock(nn.Module):
    """Sparse MoE block using SRT FusedMoE with separate TopK routing.

    TopK handles softmax + top-k routing, FusedMoE handles expert computation.
    A separate shared MLP (when present) is always applied to all tokens.
    """

    def __init__(
        self, config: PretrainedConfig, layer_id: int,
        quant_config: Optional[QuantizationConfig] = None, prefix: str = "",
    ):
        super().__init__()
        assert layer_id >= 0
        self.tp_size = get_tp_world_size()
        self.n_routed_experts = config.num_experts
        self.layer_id = layer_id

        top_k = _get_layer_value(config, "moe_topk", layer_id)
        intermediate_size = _get_layer_value(config, "intermediate_size", layer_id, 0)
        if getattr(config, "moe_intermediate_size", None) is not None:
            intermediate_size = _get_layer_value(config, "moe_intermediate_size", layer_id)

        self.gate = ReplicatedLinear(
            config.hidden_size,
            config.num_experts,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.gate",
        )

        self.topk = TopK(
            top_k=top_k,
            layer_id=layer_id,
        )

        if getattr(config, "use_mixed_mlp_moe", 0) > 0:
            num_shared_expert = _get_layer_value(config, "num_shared_expert", layer_id)
            self.shared_mlp = HunYuanMLP(
                hidden_size=config.hidden_size,
                intermediate_size=intermediate_size * num_shared_expert,
                hidden_act=config.hidden_act,
                quant_config=quant_config,
                prefix=f"{prefix}.shared_mlp",
                reduce_results=True,
            )
        else:
            self.shared_mlp = None

        self.experts = FusedMoE(
            num_experts=config.num_experts,
            top_k=top_k,
            hidden_size=config.hidden_size,
            intermediate_size=intermediate_size,
            reduce_results=True,
            quant_config=quant_config,
            layer_id=layer_id,
            prefix=f"{prefix}.experts",
        )

    def forward(self, hidden_states):
        orig_shape = hidden_states.shape
        hidden_dim = hidden_states.shape[-1]
        hidden_states = hidden_states.view(-1, hidden_dim)

        # Router logits: [num_tokens, num_experts]
        router_logits, _ = self.gate(hidden_states)

        # TopK routing: softmax + top-k selection
        topk_output = self.topk(hidden_states, router_logits)

        # FusedMoE expert computation
        final_hidden_states = self.experts(hidden_states, topk_output)

        # Shared MLP contribution (always applied to all tokens)
        if self.shared_mlp is not None:
            _shared_out = self.shared_mlp(hidden_states)
            final_hidden_states = final_hidden_states + _shared_out

        # NOTE: The AscendTPDispatcher's finalize routing performs all-gather
        # internally for the FusedMoE output on NPU. The shared MLP's down_proj
        # uses reduce_results=True to all-reduce its output across TP ranks.
        # Both components are now properly TP-synchronized.

        return final_hidden_states.view(orig_shape)


class HunYuanAttention(nn.Module):
    """Self-attention; CLA followers attend to the master's K/V via ``kv_states``."""

    def __init__(
        self,
        config: PretrainedConfig,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        layer_id: int = 0,
        rope_theta: float = 10000,
        rope_scaling: Optional[dict] = None,
        max_position_embeddings: int = 8192,
        quant_config: Optional[QuantizationConfig] = None,
        bias: bool = False,
        prefix: str = "",
        is_cross_attention: bool = False,
    ) -> None:
        super().__init__()
        tp_size = get_tp_world_size()
        self.hidden_size = hidden_size
        self.is_cross_attention = is_cross_attention
        self.total_num_heads = num_heads
        assert self.total_num_heads % tp_size == 0
        self.num_heads = self.total_num_heads // tp_size
        self.total_num_kv_heads = num_kv_heads
        if self.total_num_kv_heads >= tp_size:
            assert self.total_num_kv_heads % tp_size == 0
        else:
            assert tp_size % self.total_num_kv_heads == 0
        self.num_kv_heads = max(1, self.total_num_kv_heads // tp_size)

        self.head_dim = hidden_size // self.total_num_heads
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim**-0.5
        self.use_qk_norm = getattr(config, "use_qk_norm", False)
        self.layer_id = layer_id

        if is_cross_attention:
            # CLA follower: project only Q; K/V are reused from the master layer
            # via ``kv_states``. Matches the reference HunYuanCrossAttention and
            # the weight loader, which skips the fused-qkv mapping for follower
            # layers (layer_id % cla_factor != 0) and routes their ``.q_proj``
            # weights to this standalone q_proj.
            self.q_proj = ColumnParallelLinear(
                hidden_size,
                hidden_size,
                bias=bias,
                quant_config=quant_config,
                prefix=f"{prefix}.q_proj",
            )
        else:
            self.qkv_proj = QKVParallelLinear(
                hidden_size,
                self.head_dim,
                self.total_num_heads,
                self.total_num_kv_heads,
                bias=bias,
                quant_config=quant_config,
                prefix=f"{prefix}.qkv_proj",
            )

        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            hidden_size,
            bias=bias,
            quant_config=quant_config,
            prefix=f"{prefix}.o_proj",
        )

        self.rotary_emb = _make_rope(
            config, self.head_dim, rope_theta, rope_scaling, max_position_embeddings
        )
        self.attn = LocalAttention(
            num_heads=self.num_heads,
            head_size=self.head_dim,
            num_kv_heads=self.num_kv_heads,
            softmax_scale=self.scaling,
            causal=True,
        )

        self.image_attn = ImageKVCacheManager()
        self.image_rope2d_emb = HunYuanRotary2DEmbedder(
            num_heads=self.num_heads,
            num_kv_heads=self.num_kv_heads,
            head_dim=self.head_dim,
        )

        if self.use_qk_norm:
            self.rms_norm_eps = getattr(config, "rms_norm_eps", 1e-5)
            self.query_layernorm = RMSNorm(self.head_dim, eps=self.rms_norm_eps)
            self.key_layernorm = RMSNorm(self.head_dim, eps=self.rms_norm_eps)

    def _apply_rope(
        self,
        positions,
        q: torch.Tensor,
        k: torch.Tensor,
        hidden_states: torch.Tensor,
        custom_pos_emb,
    ):
        """Apply the 2D image RoPE (diffusion path) or the 1D text RoPE."""
        if custom_pos_emb is not None:
            return self.image_rope2d_emb(q, k, hidden_states, custom_pos_emb)
        return self.rotary_emb(positions, q, k)

    def forward(
        self,
        positions,
        hidden_states,
        forward_batch,
        kv_states=None,
        attn_meta=None,
        attention_mask=None,
        custom_pos_emb=None,
    ):
        q_len, hidden_size = hidden_states.size()
        hidden_states = hidden_states.reshape(-1, hidden_size)

        if self.is_cross_attention:
            # CLA follower: attend to the master layer's K/V.
            ori_k, v = kv_states
            k = ori_k
            q, _ = self.q_proj(hidden_states)
            q, _ = self._apply_rope(
                positions, q, torch.empty_like(k), hidden_states, custom_pos_emb,
            )
        else:
            qkv, _ = self.qkv_proj(hidden_states)
            q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
            q, k = self._apply_rope(
                positions, q, k, hidden_states, custom_pos_emb
            )
            ori_k = k

        if self.use_qk_norm:
            # Master layers use NPU fused RMSNorm; followers use generic RMSNorm.
            if self.is_cross_attention:
                q = self.query_layernorm(
                    q.view(-1, self.num_heads, self.head_dim).contiguous()
                )
                k = self.key_layernorm(
                    k.view(-1, self.num_kv_heads, self.head_dim).contiguous()
                )
            else:
                import torch_npu

                q = torch_npu.npu_rms_norm(
                    q.view(-1, self.num_heads, self.head_dim).contiguous(),
                    gamma=self.query_layernorm.weight.float(),
                    epsilon=self.rms_norm_eps,
                )[0]
                k = torch_npu.npu_rms_norm(
                    k.view(-1, self.num_kv_heads, self.head_dim).contiguous(),
                    gamma=self.key_layernorm.weight.float(),
                    epsilon=self.rms_norm_eps,
                )[0]

        if attn_meta is not None:
            attn_output = self.image_attn(q, k, v, attn_meta, attention_mask=attention_mask)
        else:
            q = q.view(-1, self.num_heads, self.head_dim)
            k = k.view(-1, self.num_kv_heads, self.head_dim)
            v = v.view(-1, self.num_kv_heads, self.head_dim)
            attn_output = self.attn(q.unsqueeze(0), k.unsqueeze(0), v.unsqueeze(0))

        attn_output = attn_output.view(q.shape[0], -1)
        output, _ = self.o_proj(attn_output)
        output = output.reshape(q_len, -1)
        return output, (ori_k, v)


class HunyuanImage3DecoderLayer(nn.Module):
    def __init__(
        self, config: PretrainedConfig, layer_id: int,
        quant_config: Optional[QuantizationConfig] = None, prefix: str = "",
    ) -> None:
        super().__init__()
        self.layer_id = layer_id
        self.hidden_size = config.hidden_size
        # intermediate_size may be a scalar or a per-layer list in the config
        intermediate_size = getattr(config, "intermediate_size", 0)
        if isinstance(intermediate_size, list):
            intermediate_size = intermediate_size[layer_id]
        self.intermediate_size = intermediate_size
        rope_theta = getattr(config, "rope_theta", 10000)
        rope_scaling = getattr(config, "rope_scaling", None)
        if rope_scaling is not None and getattr(config, "original_max_position_embeddings", None):
            rope_scaling = dict(rope_scaling)
            rope_scaling["original_max_position_embeddings"] = config.original_max_position_embeddings
        max_position_embeddings = getattr(config, "max_position_embeddings", 8192)
        attention_bias = getattr(config, "attention_bias", False) or getattr(config, "bias", False)

        cla_factor = _get_cla_factor(config)
        attention_type = "cross" if layer_id % cla_factor != 0 else "self"
        attn_kwargs = dict(
            config=config, hidden_size=self.hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=getattr(config, "num_key_value_heads", config.num_attention_heads),
            layer_id=layer_id, rope_theta=rope_theta, rope_scaling=rope_scaling,
            max_position_embeddings=max_position_embeddings,
            quant_config=quant_config, bias=attention_bias,
            prefix=f"{prefix}.self_attn",
        )
        self.self_attn = HunYuanAttention(**attn_kwargs, is_cross_attention=attention_type == "cross")

        if _is_moe(config):
            self.mlp = HunYuanSparseMoeBlock(
                config=config, layer_id=layer_id, quant_config=quant_config,
                prefix=f"{prefix}.mlp",
            )
        else:
            self.mlp = HunYuanMLP(
                hidden_size=self.hidden_size, intermediate_size=self.intermediate_size,
                hidden_act=config.hidden_act, quant_config=quant_config,
                bias=getattr(config, "mlp_bias", False), prefix=f"{prefix}.mlp",
            )

        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self, positions, hidden_states, forward_batch, residual,
        kv_states=None, attn_meta=None, attention_mask=None, custom_pos_emb=None,
    ):
        if attention_mask is not None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
            
            hidden_states, ori_kv_states = self.self_attn(
                positions=positions, hidden_states=hidden_states,
                forward_batch=forward_batch, kv_states=kv_states,
                attn_meta=attn_meta, attention_mask=attention_mask,
                custom_pos_emb=custom_pos_emb,
            )
            hidden_states = residual + hidden_states
            residual = hidden_states
            hidden_states = self.post_attention_layernorm(hidden_states)
            hidden_states = self.mlp(hidden_states)
            hidden_states = residual + hidden_states
        else:
            if residual is None:
                residual = hidden_states
                hidden_states = self.input_layernorm(hidden_states)
            else:
                hidden_states, residual = self.input_layernorm(hidden_states, residual)
            hidden_states, ori_kv_states = self.self_attn(
                positions=positions, hidden_states=hidden_states,
                forward_batch=forward_batch, kv_states=kv_states,
            )
            hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
            hidden_states = self.mlp(hidden_states)
        return hidden_states, residual, ori_kv_states


class HunyuanImage3Model(nn.Module):
    def __init__(
        self, config: PretrainedConfig,
        quant_config: Optional[QuantizationConfig] = None, prefix: str = "",
    ):
        super().__init__()
        self.config = config
        self.quant_config = quant_config
        self.vocab_size = config.vocab_size

        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size, config.hidden_size,
            quant_config=quant_config, prefix=f"{prefix}.embed_tokens",
        )
        self.layers = nn.ModuleList([
            HunyuanImage3DecoderLayer(
                config=config, layer_id=i, quant_config=quant_config,
                prefix=f"{prefix}.layers.{i}",
            )
            for i in range(config.num_hidden_layers)
        ])
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def get_input_embeddings(self, input_ids):
        return self.embed_tokens(input_ids)

    @torch.no_grad()
    def forward(self, input_ids, positions, forward_batch, input_embeds=None):
        if input_embeds is not None:
            hidden_states = input_embeds
        else:
            hidden_states = self.get_input_embeddings(input_ids)
        residual = None

        cla_factor = _get_cla_factor(self.config)
        prev_kv_states = None
        for i, layer in enumerate(self.layers):
            hidden_states, residual, kv_states = layer(
                positions, hidden_states, forward_batch, residual, prev_kv_states,
            )
            if getattr(self.config, "use_cla", False) and i % cla_factor == 0:
                prev_kv_states = kv_states
            else:
                prev_kv_states = None

        hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states

    def forward_block(
        self, hidden_states, attention_mask, custom_pos_emb,
        attn_meta=None, num_image_tokens=None, first_step=False,
    ):
        if attn_meta is None:
            attn_meta = create_hunyuan_image_attention_meta(
                attention_mask, num_image_tokens, first_step
            )

        residual = None
        cla_factor = _get_cla_factor(self.config)
        prev_kv_states = None
        for i, layer in enumerate(self.layers):
            hidden_states, residual, kv_states = layer(
                None, hidden_states, None, residual,
                prev_kv_states, attn_meta, attention_mask, custom_pos_emb,
            )
            if getattr(self.config, "use_cla", False) and i % cla_factor == 0:
                prev_kv_states = kv_states
            else:
                prev_kv_states = None

        return hidden_states.contiguous()

    def _split_qkv_weight(self, qkv):
        num_attention_heads = self.config.num_attention_heads
        num_kv_heads = getattr(self.config, "num_key_value_heads", self.config.num_attention_heads)
        num_key_value_groups = num_attention_heads // num_kv_heads
        hidden_size = self.config.hidden_size
        attention_head_dim = hidden_size // num_attention_heads

        qkv = qkv.reshape(num_kv_heads, num_key_value_groups + 2, attention_head_dim, hidden_size)
        q, k, v = torch.split(qkv, (num_key_value_groups, 1, 1), dim=1)
        q = q.reshape(-1, hidden_size)
        k = k.reshape(-1, hidden_size)
        v = v.reshape(-1, hidden_size)
        return torch.concat((q, k, v))


class HunyuanImage3ForCausalMM(CachableDiT):
    """Top-level HunyuanImage-3 model for diffusion pipeline."""

    def __init__(
        self, config: HunyuanImage3DitConfig, prefix: str = "", **kwargs,
    ):
        super().__init__(config=config, **kwargs)

        arch_config = self.config

        self.model = HunyuanImage3Model(
            arch_config, prefix=f"{prefix}.model",
        )

        self.unpadded_vocab_size = arch_config.vocab_size
        # multimodal_gen has no dedicated LM-head layer; the vocab-parallel
        # embedding shares its layout and only `.weight` is consumed downstream.
        self.lm_head = VocabParallelEmbedding(
            self.unpadded_vocab_size, arch_config.hidden_size,
            org_num_embeddings=self.unpadded_vocab_size,
            prefix=f"{prefix}.lm_head",
        )
        if getattr(arch_config, "tie_word_embeddings", False):
            self.lm_head.weight = self.model.embed_tokens.weight

        # ---- Diffusion I/O modules ----
        patch_size = getattr(arch_config, "patch_size", 1)
        patch_embed_hidden_dim = getattr(arch_config, "patch_embed_hidden_dim", 1024)
        img_proj_type = getattr(arch_config, "img_proj_type", "unet")
        # latent_channels may be top-level or nested under arch_config.vae
        if isinstance(getattr(arch_config, "vae", None), dict):
            latent_channels = arch_config.vae["latent_channels"]
        else:
            latent_channels = arch_config.latent_channels

        if img_proj_type == "unet":
            self.timestep_emb = TimestepEmbedder(hidden_size=arch_config.hidden_size)
            self.patch_embed = UNetDown(
                patch_size=patch_size,
                emb_channels=arch_config.hidden_size,
                in_channels=latent_channels,
                hidden_channels=patch_embed_hidden_dim,
                out_channels=arch_config.hidden_size,
            )
            self.time_embed = TimestepEmbedder(hidden_size=arch_config.hidden_size)
            self.final_layer = UNetUp(
                patch_size=patch_size,
                emb_channels=arch_config.hidden_size,
                in_channels=arch_config.hidden_size,
                hidden_channels=patch_embed_hidden_dim,
                out_channels=latent_channels,
                out_norm=True,
            )
            self.time_embed_2 = TimestepEmbedder(hidden_size=arch_config.hidden_size)
        else:
            raise ValueError(f"Unknown img_proj_type: {img_proj_type}")

        head_dim = arch_config.hidden_size // arch_config.num_attention_heads
        self.cached_rope = CachedRoPE(
            rope_theta=arch_config.rope_theta,
            head_dim=head_dim,
            rope_type=getattr(arch_config, "rope_type", "2d"),
        )

    def forward(self, hidden_states, timestep=None, encoder_hidden_states=None, **kwargs):
        """DiT-style forward for denoising stage."""
        return hidden_states

    def forward_block(
        self, hidden_states, attention_mask, custom_pos_emb,
        num_image_tokens=None, first_step=False, timestep=None,
    ):
        # TeaCache gate: skip layers on similar steps, reuse the cached
        # residual; one decision covers the packed CFG batch.
        if timestep is not None and self.should_skip_forward_for_cached_states(
            timestep=timestep
        ):
            return self.retrieve_cached_states(hidden_states).contiguous()

        output = self.model.forward_block(
            hidden_states, attention_mask, custom_pos_emb,
            num_image_tokens=num_image_tokens, first_step=first_step,
        )

        if timestep is not None:
            self.maybe_cache_states(output, hidden_states)
        return output

    def get_embed_and_head(self):
        return self.model.embed_tokens.weight, self.lm_head.weight

    def set_embed_and_head(self, embed, head):
        del self.model.embed_tokens.weight
        del self.lm_head.weight
        self.model.embed_tokens.weight = embed
        self.lm_head.weight = head
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    # TeaCache — see runtime/cache/teacache.py

    def should_skip_forward_for_cached_states(self, **kwargs) -> bool:
        ctx = self._get_teacache_context()
        # Gate consumed by forward_block / maybe_cache_states.
        self.enable_teacache = ctx is not None
        if ctx is None:
            return False
        # Cond/uncond rows share one packed forward (same timestep), so skip
        # boundaries are step-unit based — no CFG doubling.
        start_skipping, end_skipping = ctx.teacache_params.get_skip_boundaries(
            ctx.num_inference_steps, do_cfg=False
        )
        is_boundary_step = (
            ctx.current_timestep < start_skipping
            or ctx.current_timestep >= end_skipping
        )
        # Timestep-conditioned input for the L1 similarity decision.
        modulated_inp = self.time_embed(kwargs["timestep"])
        should_calc = self._compute_teacache_decision(
            modulated_inp=modulated_inp,
            is_boundary_step=is_boundary_step,
            coefficients=ctx.coefficients,
            teacache_thresh=ctx.teacache_thresh,
        )
        return not should_calc

    def maybe_cache_states(
        self, hidden_states: torch.Tensor, original_hidden_states: torch.Tensor
    ) -> None:
        """Cache the backbone residual for the packed [tokens, hidden] tensor."""
        if not self.enable_teacache:
            return
        self.previous_residual = hidden_states - original_hidden_states

    def retrieve_cached_states(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return hidden_states + self.previous_residual

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        stacked_params_mapping = [
            (".qkv_proj", ".q_proj", "q"),
            (".qkv_proj", ".k_proj", "k"),
            (".qkv_proj", ".v_proj", "v"),
            (".gate_up_proj", ".gate_proj", 0),
            (".gate_up_proj", ".up_proj", 1),
        ]

        num_attention_heads = self.config.num_attention_heads
        num_kv_heads = getattr(
            self.config, "num_key_value_heads", self.config.num_attention_heads
        )
        split_params_mapping = [
            (".gate_up_proj", ".gate_and_up_proj", 2, [(1, 1), (0, 1)], None),
            (
                ".qkv_proj", ".qkv_proj",
                num_attention_heads + num_kv_heads * 2,
                [("q", num_attention_heads), ("k", num_kv_heads), ("v", num_kv_heads)],
                self.model._split_qkv_weight,
            ),
        ]

        cla_factor = _get_cla_factor(self.config)

        # Expert mapping for FusedMoE loading (matching vllm-omni); remaps
        # to fused gate_and_up_proj checkpoint keys.
        expert_weights_remapping = {
            "gate_proj": ("gate_and_up_proj", 1, 2),
            "up_proj": ("gate_and_up_proj", 0, 2),
        }
        expert_params_mapping = []
        if _is_moe(self.config):
            expert_params_mapping = FusedMoE.make_expert_params_mapping(
                ckpt_gate_proj_name="gate_proj",
                ckpt_down_proj_name="down_proj",
                ckpt_up_proj_name="up_proj",
                num_experts=self.config.num_experts,
            )

        params_dict = dict(self.named_parameters())
        loaded_params: set = set()
        _ckpt_dtype_logged = False

        for name, loaded_weight in weights:
            if not _ckpt_dtype_logged:
                logger.info(
                    "  checkpoint weight dtype: %s (param dtype: %s)",
                    loaded_weight.dtype,
                    next(iter(params_dict.values())).dtype if params_dict else "?",
                )
                _ckpt_dtype_logged = True
            if any(keyword in name for keyword in UNEXPECTED_KEYWORDS):
                continue
            if "rotary_emb.inv_freq" in name:
                continue
            if "gate_proj_bias" in name:
                name = name.replace("gate_proj_bias", "gate_proj.bias")
            if "up_proj_bias" in name:
                name = name.replace("up_proj_bias", "up_proj.bias")
            if "rotary_emb.cos_cached" in name or "rotary_emb.sin_cached" in name:
                continue
            if getattr(self.config, "tie_word_embeddings", False) and "lm_head.weight" in name:
                continue

            if name.endswith("wte.weight"):
                name = name.replace("wte.weight", "embed_tokens.weight")
            if name.endswith("ln_f.weight"):
                name = name.replace("ln_f.weight", "norm.weight")
            if "mlp.gate.wg." in name:
                name = name.replace("wg.", "")

            is_found = False
            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in name:
                    continue
                if "mlp.experts" in name:
                    continue
                if weight_name == ".q_proj" and cla_factor > 1:
                    match = re.search(r"layers\.(\d+)", name)
                    if match and int(match.group(1)) % cla_factor != 0:
                        continue
                name = name.replace(weight_name, param_name)
                if name.endswith(".bias") and name not in params_dict:
                    continue
                if name not in params_dict:
                    continue
                param = params_dict[name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                loaded_params.add(name)
                is_found = True
                break
            if is_found:
                continue

            for param_name, weight_name, den, split_param, func in split_params_mapping:
                if weight_name not in name:
                    continue
                if "mlp.experts" in name:
                    continue
                name = name.replace(weight_name, param_name)
                if name.endswith(".bias") and name not in params_dict:
                    continue
                if name not in params_dict:
                    continue
                assert loaded_weight.shape[0] % den == 0
                units = loaded_weight.shape[0] // den
                param = params_dict[name]
                weight_loader = param.weight_loader
                chunk = func(loaded_weight) if func is not None else loaded_weight
                offset = 0
                for shard_id, num in split_param:
                    new_offset = offset + num * units
                    weight_loader(param, chunk[offset:new_offset], shard_id)
                    offset = new_offset
                loaded_params.add(name)
                is_found = True
                break
            if is_found:
                continue

            # Expert weights: FusedMoE.make_expert_params_mapping +
            # expert_weights_remapping handle the fused gate_and_up_proj format.
            is_expert_weight = False
            is_found = False
            found_num = 0
            if _is_moe(self.config) and "mlp.experts" in name:
                if not getattr(self, "_expert_ckpt_logged", False):
                    logger.info("  expert ckpt key sample: %s", name)
                    self._expert_ckpt_logged = True
                for mapping in expert_params_mapping:
                    param_name, weight_name, expert_id, shard_id = mapping
                    offset = 0
                    den = 1
                    for (
                        mapped_weight_substr,
                        origin_weight_info,
                    ) in expert_weights_remapping.items():
                        if mapped_weight_substr in weight_name:
                            origin_weight_name, offset, den = origin_weight_info
                            weight_name = weight_name.replace(
                                mapped_weight_substr, origin_weight_name
                            )
                            break
                    if weight_name not in name:
                        continue
                    is_expert_weight = True
                    name_mapped = name.replace(weight_name, param_name)
                    found_num += 1
                    if name_mapped not in params_dict:
                        continue
                    param = params_dict[name_mapped]
                    weight_loader = param.weight_loader

                    if den > 1:
                        assert loaded_weight.shape[0] % den == 0
                        units = loaded_weight.shape[0] // den
                        loaded_weight_shard = loaded_weight[
                            offset * units : offset * units + units
                        ]
                    else:
                        loaded_weight_shard = loaded_weight

                    weight_loader(
                        param,
                        loaded_weight_shard,
                        name_mapped,
                        shard_id=shard_id,
                        expert_id=expert_id,
                    )
                    loaded_params.add(name_mapped)
                    is_found = True
                    if found_num == den:
                        break
            if is_found:
                continue
            if is_expert_weight:
                # Recognised as expert weight but not mapped locally
                continue

            if name.endswith(".bias") and name not in params_dict:
                continue
            if name not in params_dict:
                continue
            param = params_dict[name]
            weight_loader = getattr(param, "weight_loader", default_weight_loader)
            weight_loader(param, loaded_weight)
            loaded_params.add(name)

        # Log missing weights; filter out expected missing patterns
        all_param_names = set(params_dict.keys())
        missing = all_param_names - loaded_params
        if missing:
            significant_missing = [
                n for n in missing
                if not any(k in n for k in ["rotary_emb", "lm_head"])
            ]
            if significant_missing:
                logger.warning(
                    "Weight loading: %d/%d params loaded, %d MISSING:",
                    len(loaded_params), len(all_param_names), len(significant_missing),
                )
                for n in sorted(significant_missing)[:30]:
                    logger.warning("  MISSING: %s", n)
                if len(significant_missing) > 30:
                    logger.warning("  ... and %d more", len(significant_missing) - 30)
            else:
                logger.info(
                    "Weight loading: %d/%d params loaded (all accounted for)",
                    len(loaded_params), len(all_param_names),
                )
        else:
            logger.info(
                "Weight loading: %d/%d params loaded (complete)",
                len(loaded_params), len(all_param_names),
            )

        return loaded_params


class LightProjector(nn.Module):
    """ViT embedding → transformer dim projection."""

    def __init__(self, config):
        config = types.SimpleNamespace(**config)
        super().__init__()

        if config.projector_type == "linear":
            self.layers = nn.Linear(config.input_dim, config.n_embed)
        elif config.projector_type == "mlp_gelu":
            modules = [nn.Linear(config.input_dim, config.n_embed)]
            for _ in range(1, config.depth):
                modules.append(nn.GELU())
                modules.append(nn.Linear(config.n_embed, config.n_embed))
            self.layers = nn.Sequential(*modules)
        else:
            raise ValueError(f"Unknown projector type: {config.projector_type}")

    def forward(self, x):
        return self.layers(x)


class _Hi3CacheBlock(nn.Module):
    """Pattern_3 view of one decoder layer for cache-dit.

    cache-dit drives blocks as ``(hidden_states, *conds) -> hidden_states``
    (ForwardPattern.Pattern_3); the real layer has the AR-backbone signature
    ``(positions, hidden, forward_batch, residual, kv_states, mask, pos_emb)
    -> (hidden, residual, kv)``. In the masked diffusion path the layer resets
    ``residual = hidden`` internally and CLA is off, so it is a pure
    ``hidden -> hidden`` map with the two condition tensors passed through.

    The real layer is held by plain reference (object.__setattr__) so its
    parameters stay registered only under ``HunyuanImage3Model.layers`` -- no
    double registration in named_parameters/state_dict.
    """

    def __init__(self, layer: nn.Module):
        super().__init__()
        object.__setattr__(self, "_layer", layer)

    def forward(self, hidden_states, attention_mask, custom_pos_emb):
        # The restored pre-compact attention contract dispatches on attn_meta,
        # which Model.forward_block always builds via the factory. This adapter
        # was added after that threading was removed, so it builds the metadata
        # itself (ImageKVCacheManager only reads query_lens from it).
        attn_meta = create_hunyuan_image_attention_meta(attention_mask, None, False)
        hidden_states, _, _ = self._layer(
            None, hidden_states, None, None, None, attn_meta, attention_mask, custom_pos_emb,
        )
        return hidden_states


class Hi3CacheBlockAdapter(nn.Module):
    """cache-dit-wrappable view of the diffusion block loop.

    ``forward`` matches ForwardPattern.Pattern_3 so DBCache caches the
    hidden_states residual across steps and threads ``attention_mask`` /
    ``custom_pos_emb`` through every block unchanged. This is the module handed
    to ``enable_cache_on_transformer``; its spec is registered under this class
    name by the AR stage (``_register_hi3_cache_dit_spec``).

    The name deliberately does NOT start with ``HunyuanImage``: cache-dit's
    ``BlockAdapterRegister.is_supported`` / ``get_adapter`` resolve a module by
    ``cls_name.startswith(prefix)``, so a ``HunyuanImage3...`` name prefix-matches
    cache-dit's built-in HunyuanImage (diffusers) adapter, which reads
    ``transformer_blocks`` and mis-fires on this AR backbone. A non-colliding
    name makes ``is_supported`` return False so the custom Pattern_3 spec
    (``blocks_attr="blocks"``) is used instead.
    """

    def __init__(self, model: "HunyuanImage3Model"):
        super().__init__()
        self.blocks = nn.ModuleList(
            [_Hi3CacheBlock(layer) for layer in model.layers]
        )

    def forward(self, hidden_states, attention_mask, custom_pos_emb):
        for block in self.blocks:
            hidden_states = block(hidden_states, attention_mask, custom_pos_emb)
        return hidden_states.contiguous()


EntryClass = [HunyuanImage3ForCausalMM]
