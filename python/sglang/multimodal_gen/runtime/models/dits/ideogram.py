# SPDX-License-Identifier: Apache-2.0

import math
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from sglang.multimodal_gen.configs.models.dits.ideogram import Ideogram4DiTConfig
from sglang.multimodal_gen.runtime.distributed import (
    divide,
    get_tp_world_size,
    model_parallel_is_initialized,
)
from sglang.multimodal_gen.runtime.layers.attention import (
    USPAttention,
    build_varlen_mask_meta,
)
from sglang.multimodal_gen.runtime.layers.linear import (
    ColumnParallelLinear,
    MergedColumnParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
)
from sglang.multimodal_gen.runtime.layers.quantization.configs.base_config import (
    QuantizationConfig,
)
from sglang.multimodal_gen.runtime.layers.quantization.weight_only_fp8 import (
    WeightOnlyFP8ColumnParallelLinear,
    WeightOnlyFP8Linear,
    WeightOnlyFP8MergedColumnParallelLinear,
    WeightOnlyFP8RowParallelLinear,
)
from sglang.multimodal_gen.runtime.layers.rotary_embedding import (
    Qwen3VLTextRotaryEmbedding,
    qwen3_apply_rotary_pos_emb,
)
from sglang.multimodal_gen.runtime.managers.memory_managers.layerwise_offload import (
    LayerwiseOffloadableModuleMixin,
)
from sglang.multimodal_gen.runtime.models.dits.base import BaseDiT
from sglang.multimodal_gen.runtime.server_args import get_global_server_args

OUTPUT_IMAGE_INDICATOR = 2
LLM_TOKEN_INDICATOR = 3


class Ideogram4RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.rms_norm(x, self.weight.shape, self.weight, self.eps)


class Ideogram4QuantizedLinear(ReplicatedLinear):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return super().forward(x)[0]


class Ideogram4ColumnParallelLinear(ColumnParallelLinear):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return super().forward(x)[0]


class Ideogram4MergedColumnParallelLinear(MergedColumnParallelLinear):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return super().forward(x)[0]


class Ideogram4RowParallelLinear(RowParallelLinear):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return super().forward(x)[0]


def _tp_size() -> int:
    return get_tp_world_size() if model_parallel_is_initialized() else 1


def _linear(
    in_features: int,
    out_features: int,
    bias: bool = True,
    quant_config: QuantizationConfig | None = None,
    prefix: str = "",
    gather_output: bool = True,
    use_weight_only_fp8_linears: bool = True,
):
    tp_size = _tp_size()
    use_column_parallel = tp_size > 1 and out_features % tp_size == 0
    if quant_config is None and use_weight_only_fp8_linears:
        if use_column_parallel:
            return WeightOnlyFP8ColumnParallelLinear(
                in_features,
                out_features,
                bias=bias,
                gather_output=gather_output,
            )
        return WeightOnlyFP8Linear(in_features, out_features, bias=bias)
    if use_column_parallel:
        return Ideogram4ColumnParallelLinear(
            in_features,
            out_features,
            bias=bias,
            gather_output=gather_output,
            quant_config=quant_config,
            prefix=prefix,
        )
    return Ideogram4QuantizedLinear(
        in_features,
        out_features,
        bias=bias,
        quant_config=quant_config,
        prefix=prefix,
    )


def _merged_column_linear(
    in_features: int,
    output_sizes: list[int],
    bias: bool = True,
    quant_config: QuantizationConfig | None = None,
    prefix: str = "",
    use_weight_only_fp8_linears: bool = True,
):
    tp_size = _tp_size()
    use_column_parallel = tp_size > 1 and all(
        output_size % tp_size == 0 for output_size in output_sizes
    )
    out_features = sum(output_sizes)
    if quant_config is None and use_weight_only_fp8_linears:
        if use_column_parallel:
            return WeightOnlyFP8MergedColumnParallelLinear(
                in_features,
                output_sizes,
                bias=bias,
                gather_output=False,
            )
        return WeightOnlyFP8Linear(in_features, out_features, bias=bias)
    if use_column_parallel:
        return Ideogram4MergedColumnParallelLinear(
            in_features,
            output_sizes,
            bias=bias,
            gather_output=False,
            quant_config=quant_config,
            prefix=prefix,
        )
    return Ideogram4QuantizedLinear(
        in_features,
        out_features,
        bias=bias,
        quant_config=quant_config,
        prefix=prefix,
    )


def _row_linear(
    in_features: int,
    out_features: int,
    bias: bool = True,
    quant_config: QuantizationConfig | None = None,
    prefix: str = "",
    use_weight_only_fp8_linears: bool = True,
):
    tp_size = _tp_size()
    use_row_parallel = tp_size > 1 and in_features % tp_size == 0
    if quant_config is None and use_weight_only_fp8_linears:
        if use_row_parallel:
            return WeightOnlyFP8RowParallelLinear(
                in_features,
                out_features,
                bias=bias,
                input_is_parallel=True,
            )
        return WeightOnlyFP8Linear(in_features, out_features, bias=bias)
    if use_row_parallel:
        return Ideogram4RowParallelLinear(
            in_features,
            out_features,
            bias=bias,
            input_is_parallel=True,
            quant_config=quant_config,
            prefix=prefix,
        )
    return Ideogram4QuantizedLinear(
        in_features,
        out_features,
        bias=bias,
        quant_config=quant_config,
        prefix=prefix,
    )


class Ideogram4Attention(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        eps: float,
        supported_attention_backends,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
        use_weight_only_fp8_linears: bool = True,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        tp_size = _tp_size()
        assert num_heads % tp_size == 0
        self.local_num_heads = divide(num_heads, tp_size)
        self.qkv = _merged_column_linear(
            hidden_size,
            [hidden_size, hidden_size, hidden_size],
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.qkv",
            use_weight_only_fp8_linears=use_weight_only_fp8_linears,
        )
        self.norm_q = Ideogram4RMSNorm(self.head_dim, eps=eps)
        self.norm_k = Ideogram4RMSNorm(self.head_dim, eps=eps)
        self.attn = USPAttention(
            num_heads=self.local_num_heads,
            head_size=self.head_dim,
            dropout_rate=0,
            softmax_scale=None,
            causal=False,
            supported_attention_backends=supported_attention_backends,
        )
        self.o = _row_linear(
            hidden_size,
            hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.o",
            use_weight_only_fp8_linears=use_weight_only_fp8_linears,
        )

    def forward(self, x, cos, sin, attn_mask, attn_mask_meta):
        batch_size, seq_len, _ = x.shape
        qkv = self.qkv(x).view(
            batch_size, seq_len, 3, self.local_num_heads, self.head_dim
        )
        q, k, v = qkv.unbind(dim=2)
        q = self.norm_q(q)
        k = self.norm_k(k)
        q, k = qwen3_apply_rotary_pos_emb(q, k, cos, sin)
        out = self.attn(q, k, v, attn_mask=attn_mask, attn_mask_meta=attn_mask_meta)
        out = out.reshape(batch_size, seq_len, self.local_num_heads * self.head_dim)
        return self.o(out)


class Ideogram4MLP(nn.Module):
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
        use_weight_only_fp8_linears: bool = True,
    ) -> None:
        super().__init__()
        self.w1 = _linear(
            dim,
            hidden_dim,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.w1",
            gather_output=False,
            use_weight_only_fp8_linears=use_weight_only_fp8_linears,
        )
        self.w2 = _row_linear(
            hidden_dim,
            dim,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.w2",
            use_weight_only_fp8_linears=use_weight_only_fp8_linears,
        )
        self.w3 = _linear(
            dim,
            hidden_dim,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.w3",
            gather_output=False,
            use_weight_only_fp8_linears=use_weight_only_fp8_linears,
        )

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class Ideogram4TransformerBlock(nn.Module):
    def __init__(
        self,
        hidden_size,
        intermediate_size,
        num_heads,
        norm_eps,
        adaln_dim,
        supported_attention_backends,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
        use_weight_only_fp8_linears: bool = True,
    ):
        super().__init__()
        self.attention = Ideogram4Attention(
            hidden_size,
            num_heads,
            eps=1e-5,
            supported_attention_backends=supported_attention_backends,
            quant_config=quant_config,
            prefix=f"{prefix}.attention",
            use_weight_only_fp8_linears=use_weight_only_fp8_linears,
        )
        self.feed_forward = Ideogram4MLP(
            hidden_size,
            intermediate_size,
            quant_config=quant_config,
            prefix=f"{prefix}.feed_forward",
            use_weight_only_fp8_linears=use_weight_only_fp8_linears,
        )
        self.attention_norm1 = Ideogram4RMSNorm(hidden_size, eps=norm_eps)
        self.ffn_norm1 = Ideogram4RMSNorm(hidden_size, eps=norm_eps)
        self.attention_norm2 = Ideogram4RMSNorm(hidden_size, eps=norm_eps)
        self.ffn_norm2 = Ideogram4RMSNorm(hidden_size, eps=norm_eps)
        self.adaln_modulation = _linear(
            adaln_dim,
            4 * hidden_size,
            bias=True,
            quant_config=quant_config,
            prefix=f"{prefix}.adaln_modulation",
            use_weight_only_fp8_linears=use_weight_only_fp8_linears,
        )

    def forward(self, x, cos, sin, adaln_input, attn_mask, attn_mask_meta):
        scale_msa, gate_msa, scale_mlp, gate_mlp = self.adaln_modulation(
            adaln_input
        ).chunk(4, dim=-1)
        gate_msa = torch.tanh(gate_msa)
        gate_mlp = torch.tanh(gate_mlp)
        attn_out = self.attention(
            self.attention_norm1(x) * (1.0 + scale_msa),
            cos=cos,
            sin=sin,
            attn_mask=attn_mask,
            attn_mask_meta=attn_mask_meta,
        )
        x = x + gate_msa * self.attention_norm2(attn_out)
        x = x + gate_mlp * self.ffn_norm2(
            self.feed_forward(self.ffn_norm1(x) * (1.0 + scale_mlp))
        )
        return x


def _sinusoidal_embedding(t: torch.Tensor, dim: int, scale: float = 1e4):
    t = t.to(torch.float32)
    half = dim // 2
    freq = math.log(scale) / (half - 1)
    freq = torch.exp(torch.arange(half, dtype=torch.float32, device=t.device) * -freq)
    emb = t.unsqueeze(-1) * freq
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
    if dim % 2 == 1:
        emb = F.pad(emb, (0, 1))
    return emb


class Ideogram4EmbedScalar(nn.Module):
    def __init__(
        self,
        dim: int,
        input_range: tuple[float, float],
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
        use_weight_only_fp8_linears: bool = True,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.range_min, self.range_max = input_range
        self.mlp_in = _linear(
            dim,
            dim,
            bias=True,
            quant_config=quant_config,
            prefix=f"{prefix}.mlp_in",
            use_weight_only_fp8_linears=use_weight_only_fp8_linears,
        )
        self.mlp_out = _linear(
            dim,
            dim,
            bias=True,
            quant_config=quant_config,
            prefix=f"{prefix}.mlp_out",
            use_weight_only_fp8_linears=use_weight_only_fp8_linears,
        )

    def forward(self, x):
        compute_dtype = x.dtype
        x = x.to(torch.float32)
        scaled = 1e4 * (x - self.range_min) / (self.range_max - self.range_min)
        emb = _sinusoidal_embedding(scaled, self.dim).to(compute_dtype)
        return self.mlp_out(F.silu(self.mlp_in(emb)))


class Ideogram4FinalLayer(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        out_channels: int,
        adaln_dim: int,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
        use_weight_only_fp8_linears: bool = True,
    ) -> None:
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, eps=1e-6, elementwise_affine=False)
        self.linear = _linear(
            hidden_size,
            out_channels,
            bias=True,
            quant_config=quant_config,
            prefix=f"{prefix}.linear",
            use_weight_only_fp8_linears=use_weight_only_fp8_linears,
        )
        self.adaln_modulation = _linear(
            adaln_dim,
            hidden_size,
            bias=True,
            quant_config=quant_config,
            prefix=f"{prefix}.adaln_modulation",
            use_weight_only_fp8_linears=use_weight_only_fp8_linears,
        )

    def forward(self, x, c):
        scale = 1.0 + self.adaln_modulation(F.silu(c))
        return self.linear(self.norm_final(x) * scale)


class Ideogram4Transformer2DModel(BaseDiT, LayerwiseOffloadableModuleMixin):
    _repeated_blocks = ["Ideogram4TransformerBlock"]
    layer_names = ["layers"]
    _fsdp_shard_conditions = Ideogram4DiTConfig().arch_config._fsdp_shard_conditions
    _compile_conditions = Ideogram4DiTConfig().arch_config._compile_conditions
    _supported_attention_backends = (
        Ideogram4DiTConfig().arch_config._supported_attention_backends
    )
    param_names_mapping = Ideogram4DiTConfig().arch_config.param_names_mapping
    reverse_param_names_mapping = {}

    def __init__(
        self,
        config: Ideogram4DiTConfig,
        hf_config: dict[str, Any],
        quant_config: QuantizationConfig | None = None,
        **kwargs,
    ) -> None:
        super().__init__(config, hf_config, **kwargs)
        cfg = config.arch_config
        use_weight_only_fp8_linears = config.use_weight_only_fp8_linears
        if get_global_server_args().original_dtype == "bfloat16":
            use_weight_only_fp8_linears = False
        self._supported_attention_backends = cfg._supported_attention_backends
        hidden_size = cfg.num_attention_heads * cfg.attention_head_dim
        self.hidden_size = hidden_size
        self.num_attention_heads = cfg.num_attention_heads
        self.num_channels_latents = cfg.in_channels
        self.input_proj = _linear(
            cfg.in_channels,
            hidden_size,
            bias=True,
            quant_config=quant_config,
            prefix="input_proj",
            use_weight_only_fp8_linears=use_weight_only_fp8_linears,
        )
        self.llm_cond_norm = Ideogram4RMSNorm(cfg.llm_features_dim, eps=1e-6)
        self.llm_cond_proj = _linear(
            cfg.llm_features_dim,
            hidden_size,
            bias=True,
            quant_config=quant_config,
            prefix="llm_cond_proj",
            use_weight_only_fp8_linears=use_weight_only_fp8_linears,
        )
        self.t_embedding = Ideogram4EmbedScalar(
            hidden_size,
            input_range=(0.0, 1.0),
            quant_config=quant_config,
            prefix="t_embedding",
            use_weight_only_fp8_linears=use_weight_only_fp8_linears,
        )
        self.adaln_proj = _linear(
            hidden_size,
            cfg.adaln_dim,
            bias=True,
            quant_config=quant_config,
            prefix="adaln_proj",
            use_weight_only_fp8_linears=use_weight_only_fp8_linears,
        )
        self.embed_image_indicator = nn.Embedding(2, hidden_size)
        self.rotary_emb = Qwen3VLTextRotaryEmbedding(
            head_dim=cfg.attention_head_dim,
            rope_theta=cfg.rope_theta,
            mrope_section=cfg.mrope_section,
        )
        self.layers = nn.ModuleList(
            [
                Ideogram4TransformerBlock(
                    hidden_size=hidden_size,
                    intermediate_size=cfg.intermediate_size,
                    num_heads=cfg.num_attention_heads,
                    norm_eps=cfg.norm_eps,
                    adaln_dim=cfg.adaln_dim,
                    supported_attention_backends=self._supported_attention_backends,
                    quant_config=quant_config,
                    prefix=f"layers.{i}",
                    use_weight_only_fp8_linears=use_weight_only_fp8_linears,
                )
                for i in range(cfg.num_layers)
            ]
        )
        self.final_layer = Ideogram4FinalLayer(
            hidden_size=hidden_size,
            out_channels=cfg.in_channels,
            adaln_dim=cfg.adaln_dim,
            quant_config=quant_config,
            prefix="final_layer",
            use_weight_only_fp8_linears=use_weight_only_fp8_linears,
        )

    def post_load_weights(self) -> None:
        if not self.rotary_emb.inv_freq.is_meta:
            return
        cfg = self.config.arch_config
        inv_freq = 1.0 / (
            cfg.rope_theta
            ** (
                torch.arange(
                    0,
                    cfg.attention_head_dim,
                    2,
                    dtype=torch.float32,
                    device=self.input_proj.weight.device,
                )
                / cfg.attention_head_dim
            )
        )
        self.rotary_emb.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(
        self,
        *,
        llm_features: torch.Tensor,
        x: torch.Tensor,
        t: torch.Tensor,
        position_ids: torch.Tensor,
        segment_ids: torch.Tensor,
        indicator: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
        attn_mask_meta: dict | None = None,
        **kwargs,
    ) -> torch.Tensor:
        param_dtype = self.embed_image_indicator.weight.dtype
        x = x.to(param_dtype)
        t = t.to(param_dtype)
        llm_features = llm_features.to(param_dtype)
        indicator = indicator.to(torch.long)
        llm_token_mask = (indicator == LLM_TOKEN_INDICATOR).to(x.dtype).unsqueeze(-1)
        output_image_mask = (
            (indicator == OUTPUT_IMAGE_INDICATOR).to(x.dtype).unsqueeze(-1)
        )
        llm_features = llm_features * llm_token_mask
        x = x * output_image_mask
        x = self.input_proj(x) * output_image_mask
        t_cond = self.t_embedding(t)
        if t.dim() == 1:
            t_cond = t_cond.unsqueeze(1)
        adaln_input = F.silu(self.adaln_proj(t_cond))
        llm_features = self.llm_cond_proj(self.llm_cond_norm(llm_features))
        llm_features = llm_features * llm_token_mask
        h = x + llm_features
        h = h + self.embed_image_indicator(
            (indicator == OUTPUT_IMAGE_INDICATOR).to(torch.long)
        )
        cos, sin = self.rotary_emb(h, position_ids)
        cos = cos.unsqueeze(2)
        sin = sin.unsqueeze(2)
        # ideogram uses -1 padding; varlen meta enables fa packed attention
        if attn_mask is None:
            attn_mask = segment_ids > 0
        if attn_mask_meta is None:
            attn_mask_meta = build_varlen_mask_meta(attn_mask)
        for layer in self.layers:
            h = layer(
                h,
                cos=cos,
                sin=sin,
                adaln_input=adaln_input,
                attn_mask=attn_mask,
                attn_mask_meta=attn_mask_meta,
            )
        return self.final_layer(h, c=adaln_input).to(torch.float32)


EntryClass = Ideogram4Transformer2DModel
