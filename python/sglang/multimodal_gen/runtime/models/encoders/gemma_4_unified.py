# SPDX-License-Identifier: Apache-2.0
# Adapted from transformers 5.19: models/gemma4_unified/modeling_gemma4_unified.py
"""Gemma-4-Unified text encoder (LTX-2.5).

LTX-2.5 reads every hidden state of the text stack (embeddings plus the 48
decoder layers), so only the language model is built; the vision and audio
embedders in the checkpoint are never loaded. Each op follows the transformers
SDPA path in the same order and dtype, so the hidden states match it bit for bit
at TP 1; the fused QKV and gate/up GEMMs compute each output column exactly as
the separate projections do.
"""

from collections.abc import Iterable

import torch
import torch.nn.functional as F
from torch import nn

from sglang.multimodal_gen.configs.models.encoders import BaseEncoderOutput
from sglang.multimodal_gen.configs.models.encoders.gemma_4_unified import (
    Gemma4UnifiedConfig,
)
from sglang.multimodal_gen.runtime.distributed import get_tp_world_size
from sglang.multimodal_gen.runtime.layers.activation import GeluAndMul
from sglang.multimodal_gen.runtime.layers.linear import (
    MergedColumnParallelLinear,
    QKVParallelLinear,
    RowParallelLinear,
)
from sglang.multimodal_gen.runtime.layers.quantization import QuantizationConfig
from sglang.multimodal_gen.runtime.layers.vocab_parallel_embedding import (
    VocabParallelEmbedding,
)
from sglang.multimodal_gen.runtime.loader.weight_utils import (
    default_weight_loader,
    load_stacked_weight,
)
from sglang.multimodal_gen.runtime.managers.memory_managers.layerwise_offload import (
    LayerwiseOffloadableModuleMixin,
)
from sglang.multimodal_gen.runtime.models.encoders.base import (
    TextEncoder,
    get_attention_head_partition,
)
from sglang.multimodal_gen.runtime.utils.weight_attrs import set_weight_attrs


class Gemma4UnifiedRMSNorm(nn.Module):
    """RMSNorm scaled by ``weight`` (not ``1 + weight``); ``torch.pow`` as upstream."""

    def __init__(self, dim: int, eps: float, with_scale: bool = True) -> None:
        super().__init__()
        self.eps = eps
        self.weight = None
        if with_scale:
            self.weight = nn.Parameter(torch.ones(dim), requires_grad=False)
            set_weight_attrs(self.weight, {"weight_loader": default_weight_loader})

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = x.float()
        h = h * torch.pow(h.pow(2).mean(-1, keepdim=True) + self.eps, -0.5)
        if self.weight is not None:
            h = h * self.weight.float()
        return h.type_as(x)


def _rope_inv_freq(rope_parameters: dict, head_dim: int) -> torch.Tensor:
    """fp32 inverse frequencies on the host, as upstream computes them.

    ``proportional`` rotates the first ``partial_rotary_factor`` of the head and
    leaves the rest at frequency 0 (cos 1, sin 0).
    """
    rope_type = rope_parameters["rope_type"]
    if rope_type == "default":
        rotary_dim = head_dim
    elif rope_type == "proportional":
        partial = rope_parameters.get("partial_rotary_factor", 1.0)
        rotary_dim = 2 * int(partial * head_dim // 2)
    else:
        raise ValueError(f"Unsupported Gemma-4 rope_type {rope_type!r}")
    exponents = torch.arange(0, rotary_dim, 2, dtype=torch.float32, device="cpu")
    inv_freq = 1.0 / (rope_parameters["rope_theta"] ** (exponents / head_dim))
    inv_freq = F.pad(inv_freq, (0, head_dim // 2 - rotary_dim // 2))
    return inv_freq / rope_parameters.get("factor", 1.0)


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def _apply_rotary(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
    # x: [B, S, H, D]; cos/sin: [B, S, D]
    cos, sin = cos.unsqueeze(2), sin.unsqueeze(2)
    return x * cos + _rotate_half(x) * sin


def _repeat_kv(x: torch.Tensor, groups: int) -> torch.Tensor:
    # [B, H_kv, S, D] -> [B, H_kv * groups, S, D], materialized like upstream's
    if groups == 1:
        return x
    b, h, s, d = x.shape
    return x[:, :, None].expand(b, h, groups, s, d).reshape(b, h * groups, s, d)


class Gemma4UnifiedAttention(nn.Module):
    def __init__(
        self,
        text_config,
        layer_idx: int,
        quant_config: QuantizationConfig | None,
        prefix: str,
    ) -> None:
        super().__init__()
        layer_config = text_config.per_layer_config[layer_idx]
        self.head_dim = layer_config.head_dim
        total_heads = text_config.num_attention_heads
        total_kv_heads = layer_config.num_key_value_heads
        self.num_heads, self.num_kv_heads = get_attention_head_partition(
            total_heads, total_kv_heads, get_tp_world_size()
        )
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        # full-attention layers with attention_k_eq_v have no v_proj: V is the
        # raw K projection, so the V shard is loaded with K's weights
        self.k_eq_v = (
            text_config.attention_k_eq_v
            and text_config.layer_types[layer_idx] == "full_attention"
        )

        self.qkv_proj = QKVParallelLinear(
            hidden_size=text_config.hidden_size,
            head_size=self.head_dim,
            total_num_heads=total_heads,
            total_num_kv_heads=total_kv_heads,
            bias=text_config.attention_bias,
            quant_config=quant_config,
            prefix=f"{prefix}.qkv_proj",
        )
        self.o_proj = RowParallelLinear(
            input_size=total_heads * self.head_dim,
            output_size=text_config.hidden_size,
            bias=text_config.attention_bias,
            quant_config=quant_config,
            prefix=f"{prefix}.o_proj",
        )
        eps = text_config.rms_norm_eps
        self.q_norm = Gemma4UnifiedRMSNorm(self.head_dim, eps)
        self.k_norm = Gemma4UnifiedRMSNorm(self.head_dim, eps)
        self.v_norm = Gemma4UnifiedRMSNorm(self.head_dim, eps, with_scale=False)

    def forward(
        self,
        hidden_states: torch.Tensor,
        cos_sin: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        b, s, _ = hidden_states.shape
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        cos, sin = cos_sin
        q = _apply_rotary(self.q_norm(q.view(b, s, -1, self.head_dim)), cos, sin)
        k = _apply_rotary(self.k_norm(k.view(b, s, -1, self.head_dim)), cos, sin)
        v = self.v_norm(v.view(b, s, -1, self.head_dim))
        # LTX2 pins this encoder to torch SDPA; QK-norm replaces the 1/sqrt(d) scale
        groups = self.num_heads // self.num_kv_heads
        out = F.scaled_dot_product_attention(
            q.transpose(1, 2),
            _repeat_kv(k.transpose(1, 2), groups),
            _repeat_kv(v.transpose(1, 2), groups),
            attn_mask=attention_mask,
            scale=1.0,
        )
        out, _ = self.o_proj(out.transpose(1, 2).reshape(b, s, -1))
        return out


class Gemma4UnifiedMLP(nn.Module):
    def __init__(
        self, text_config, quant_config: QuantizationConfig | None, prefix: str
    ) -> None:
        super().__init__()
        if text_config.hidden_activation != "gelu_pytorch_tanh":
            raise ValueError(
                f"Gemma-4 MLP expects gelu_pytorch_tanh, got "
                f"{text_config.hidden_activation!r}"
            )
        self.gate_up_proj = MergedColumnParallelLinear(
            input_size=text_config.hidden_size,
            output_sizes=[text_config.intermediate_size] * 2,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.gate_up_proj",
        )
        self.down_proj = RowParallelLinear(
            input_size=text_config.intermediate_size,
            output_size=text_config.hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.down_proj",
        )
        # gelu(gate) * up; on CUDA each op rounds to the activation dtype as upstream
        self.act_fn = GeluAndMul(approximate="tanh")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, _ = self.gate_up_proj(x)
        x, _ = self.down_proj(self.act_fn(x))
        return x


class Gemma4UnifiedDecoderLayer(nn.Module):
    def __init__(
        self,
        text_config,
        layer_idx: int,
        quant_config: QuantizationConfig | None,
        prefix: str,
    ) -> None:
        super().__init__()
        self.self_attn = Gemma4UnifiedAttention(
            text_config, layer_idx, quant_config, f"{prefix}.self_attn"
        )
        self.mlp = Gemma4UnifiedMLP(text_config, quant_config, f"{prefix}.mlp")
        hidden, eps = text_config.hidden_size, text_config.rms_norm_eps
        self.input_layernorm = Gemma4UnifiedRMSNorm(hidden, eps)
        self.post_attention_layernorm = Gemma4UnifiedRMSNorm(hidden, eps)
        self.pre_feedforward_layernorm = Gemma4UnifiedRMSNorm(hidden, eps)
        self.post_feedforward_layernorm = Gemma4UnifiedRMSNorm(hidden, eps)
        # a persistent buffer upstream; a frozen parameter here so the loader
        # and layerwise offload treat it like the other layer weights
        self.layer_scalar = nn.Parameter(torch.ones(1), requires_grad=False)
        set_weight_attrs(self.layer_scalar, {"weight_loader": default_weight_loader})

    def forward(
        self,
        hidden_states: torch.Tensor,
        cos_sin: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(hidden_states, cos_sin, attention_mask)
        hidden_states = residual + self.post_attention_layernorm(hidden_states)

        residual = hidden_states
        hidden_states = self.mlp(self.pre_feedforward_layernorm(hidden_states))
        hidden_states = residual + self.post_feedforward_layernorm(hidden_states)
        return hidden_states * self.layer_scalar


class Gemma4UnifiedTextModel(nn.Module):
    def __init__(self, text_config, quant_config: QuantizationConfig | None) -> None:
        super().__init__()
        if (
            text_config.num_kv_shared_layers
            or text_config.use_bidirectional_attention == "all"
        ):
            raise ValueError(
                "Gemma-4 KV sharing and fully bidirectional text attention are "
                "not implemented"
            )
        self.layer_types = list(text_config.layer_types)
        self.sliding_window = text_config.sliding_window
        self.embed_tokens = VocabParallelEmbedding(
            text_config.vocab_size,
            text_config.hidden_size,
            org_num_embeddings=text_config.vocab_size,
        )
        # upstream casts sqrt(hidden) to the table dtype before scaling
        self.register_buffer(
            "embed_scale",
            torch.tensor(text_config.hidden_size**0.5, dtype=torch.float32),
            persistent=False,
        )
        self.layers = nn.ModuleList(
            [
                Gemma4UnifiedDecoderLayer(
                    text_config, i, quant_config, f"language_model.layers.{i}"
                )
                for i in range(text_config.num_hidden_layers)
            ]
        )
        self.norm = Gemma4UnifiedRMSNorm(
            text_config.hidden_size, text_config.rms_norm_eps
        )
        for layer_type in sorted(set(self.layer_types)):
            self.register_buffer(
                f"{layer_type}_inv_freq",
                _rope_inv_freq(
                    text_config.rope_parameters[layer_type],
                    text_config.per_layer_config[layer_type].head_dim,
                ),
                persistent=False,
            )

    def _attention_masks(
        self, attention_mask: torch.Tensor | None, embeds: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        """[B, 1, S, S] boolean masks per layer type: causal, padded keys
        dropped, sliding layers windowed; the same masks upstream hands SDPA."""
        batch_size, seq_len, _ = embeds.shape
        idx = torch.arange(seq_len, device=embeds.device)
        causal = idx[None, :] <= idx[:, None]
        if attention_mask is None:
            keys = torch.ones(
                batch_size, 1, 1, seq_len, dtype=torch.bool, device=idx.device
            )
        else:
            keys = attention_mask.bool()[:, None, None, :]
        window = idx[None, :] > idx[:, None] - self.sliding_window
        masks = {
            "full_attention": causal & keys,
            "sliding_attention": causal & window & keys,
        }
        return {layer_type: masks[layer_type] for layer_type in set(self.layer_types)}

    def _rotary(
        self, position_ids: torch.Tensor, dtype: torch.dtype
    ) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
        cos_sin = {}
        for layer_type in set(self.layer_types):
            inv_freq = self.get_buffer(f"{layer_type}_inv_freq")
            freqs = position_ids[..., None].float() * inv_freq.to(position_ids.device)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos_sin[layer_type] = (emb.cos().to(dtype), emb.sin().to(dtype))
        return cos_sin

    def forward(
        self,
        input_ids: torch.Tensor | None,
        position_ids: torch.Tensor | None,
        attention_mask: torch.Tensor | None,
        inputs_embeds: torch.Tensor | None,
        output_hidden_states: bool,
    ) -> BaseEncoderOutput:
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
            inputs_embeds = inputs_embeds * self.embed_scale.to(inputs_embeds.dtype)
        if position_ids is None:
            position_ids = torch.arange(
                inputs_embeds.shape[1], device=inputs_embeds.device
            )[None]
        masks = self._attention_masks(attention_mask, inputs_embeds)
        cos_sin = self._rotary(position_ids, inputs_embeds.dtype)

        hidden_states = inputs_embeds
        all_hidden_states = [hidden_states] if output_hidden_states else None
        for layer, layer_type in zip(self.layers, self.layer_types):
            hidden_states = layer(hidden_states, cos_sin[layer_type], masks[layer_type])
            if all_hidden_states is not None:
                all_hidden_states.append(hidden_states)
        hidden_states = self.norm(hidden_states)
        if all_hidden_states is not None:
            # the last entry is the normed output, as in transformers
            all_hidden_states[-1] = hidden_states
            all_hidden_states = tuple(all_hidden_states)
        return BaseEncoderOutput(
            last_hidden_state=hidden_states, hidden_states=all_hidden_states
        )


class Gemma4UnifiedForConditionalGeneration(TextEncoder):
    """Text path of ``Gemma4UnifiedForConditionalGeneration``."""

    layer_names = ["language_model.layers"]
    # the 2.0 GB vocab table is read only through embed_tokens.__call__
    host_resident_table_names = ["language_model.embed_tokens"]

    def __init__(self, config: Gemma4UnifiedConfig) -> None:
        super().__init__(config)
        text_config = config.arch_config.text_config
        self.language_model = Gemma4UnifiedTextModel(text_config, config.quant_config)
        self._k_eq_v_k_projs = {
            f"language_model.layers.{i}.self_attn.k_proj"
            for i, layer in enumerate(self.language_model.layers)
            if layer.self_attn.k_eq_v
        }

    @staticmethod
    def should_materialize_checkpoint_weight(name: str) -> bool:
        return name.startswith("model.language_model.")

    def forward(
        self,
        input_ids: torch.Tensor | None,
        position_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
        output_hidden_states: bool | None = None,
        **kwargs,
    ) -> BaseEncoderOutput:
        return self.language_model(
            input_ids,
            position_ids,
            attention_mask,
            inputs_embeds,
            bool(output_hidden_states),
        )

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        params_dict = dict(self.named_parameters())
        stacked_params_mapping = self.config.arch_config.stacked_params_mapping
        loaded_params: set[str] = set()
        for name, loaded_weight in weights:
            name = name.removeprefix("model.")
            if name.rpartition(".")[0] in self._k_eq_v_k_projs:
                load_stacked_weight(
                    name.replace(".k_proj.", ".v_proj."),
                    loaded_weight,
                    params_dict,
                    stacked_params_mapping,
                )
            loaded_name = load_stacked_weight(
                name, loaded_weight, params_dict, stacked_params_mapping
            )
            if loaded_name is not None:
                loaded_params.add(loaded_name)
        return loaded_params


# -- transformers fallback ---------------------------------------------------
# A checkpoint whose format transformers owns (e.g. bitsandbytes) still loads as
# ``Gemma4UnifiedModel``; its decoder stack is a plain ``ModuleList`` at
# ``language_model.layers``, so the loaded module's class is swapped for a
# subclass that also carries the layerwise mixin. Forward and weights untouched.


class _Gemma4UnifiedLayerwise(LayerwiseOffloadableModuleMixin):
    layerwise_offload_dit_group_enabled = False
    # 48 decoder layers of ~454 MB in bf16: 21.8 of the 23.9 GB.
    layer_names = ["language_model.layers"]
    # The 2.0 GB vocab table is reached only through embed_tokens.__call__;
    # AutoModel builds no tied lm_head.
    host_resident_table_names = ["language_model.embed_tokens"]


class _HostGatherScaledEmbedding:
    """Apply Gemma's 0-dim ``embed_scale`` on whichever device ran the gather.

    Upstream multiplies by the buffer in place; under host residency the gather
    runs on the host while the buffer follows the module to the device.
    """

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        embeds = nn.Embedding.forward(self, input_ids)
        return embeds * self.embed_scale.to(device=embeds.device, dtype=embeds.dtype)


def _swap_class(module: nn.Module, *, extra: type, first: bool) -> None:
    cls = type(module)
    if issubclass(cls, extra):
        return
    bases = (extra, cls) if first else (cls, extra)
    module.__class__ = type(cls.__name__, bases, {"__module__": cls.__module__})


def make_gemma4_unified_layerwise_offloadable(model: nn.Module) -> nn.Module:
    _swap_class(model, extra=_Gemma4UnifiedLayerwise, first=False)
    _swap_class(
        model.language_model.embed_tokens,
        extra=_HostGatherScaledEmbedding,
        first=True,
    )
    return model


# transformers class name -> adapter applied after the native load
NATIVE_LAYERWISE_ADAPTERS = {
    "Gemma4UnifiedModel": make_gemma4_unified_layerwise_offloadable,
}

EntryClass = Gemma4UnifiedForConditionalGeneration
