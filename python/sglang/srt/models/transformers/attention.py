# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Copyright 2025 SGLang Team

from collections.abc import Mapping
from dataclasses import dataclass

from torch import nn
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

from sglang.srt.layers.radix_attention import AttentionType, RadixAttention
from sglang.srt.runtime_context import get_parallel


@dataclass(frozen=True)
class AttentionDescriptor:
    num_heads: int
    num_kv_heads: int
    head_dim: int
    value_head_dim: int
    scaling: float
    sliding_window: int
    causal: bool
    softcap: float

    @classmethod
    def from_module(cls, module, config, tp_size):
        heads = getattr(module, "num_heads", None) or getattr(
            module, "num_attention_heads", config.num_attention_heads
        )
        kv_heads = getattr(module, "num_key_value_heads", None) or getattr(
            config, "num_key_value_heads", heads
        )
        if heads % tp_size or (kv_heads % tp_size and tp_size % kv_heads):
            raise ValueError(
                f"Attention heads ({heads}, {kv_heads}) cannot be partitioned over TP={tp_size}"
            )
        head_dim = (
            getattr(module, "qk_head_dim", None)
            or getattr(module, "head_dim", None)
            or getattr(module, "attention_head_size", None)
            or getattr(config, "head_dim", None)
            or config.hidden_size // heads
        )
        value_dim = getattr(module, "v_head_dim", head_dim)
        causal = getattr(module, "is_causal", True)
        if (
            getattr(module, "is_cross_attention", False)
            or "CrossAttention" in type(module).__name__
        ):
            raise ValueError(
                "Transformers cross-attention requires an encoder-decoder adapter"
            )
        window = getattr(module, "sliding_window", None)
        if not hasattr(module, "sliding_window"):
            layer_types = getattr(config, "layer_types", None)
            if not layer_types or layer_types[module.layer_idx] == "sliding_attention":
                window = getattr(config, "sliding_window", None)
        if window is not None and window <= 0:
            raise ValueError(f"Invalid attention window: {window}")
        return cls(
            heads // tp_size,
            max(1, kv_heads // tp_size),
            head_dim,
            value_dim,
            getattr(module, "scaling", head_dim**-0.5),
            window - 1 if window is not None else -1,
            causal,
            getattr(module, "attn_logit_softcapping", None)
            or getattr(config, "attn_logit_softcapping", None)
            or 0.0,
        )


def sglang_flash_attention_forward(
    module,
    query,
    key,
    value,
    attention_mask=None,
    scaling=None,
    attention_instances: Mapping | None = None,
    forward_batch=None,
    **kwargs,
):
    if attention_instances is None or forward_batch is None:
        raise ValueError("SGLang attention requires engine batch metadata")
    attention_key = getattr(module, "_sglang_attention_key", str(module.layer_idx))
    self_attn = attention_instances[attention_key]
    if attention_mask is not None:
        raise ValueError(
            "Arbitrary Transformers attention masks are not supported by paged attention"
        )
    if kwargs.get("dropout", 0.0):
        raise ValueError("SGLang attention requires inference mode (dropout=0)")
    for name in ("head_mask", "alibi", "position_bias", "score_mod", "block_mask"):
        if kwargs.get(name) is not None:
            raise ValueError(f"Unsupported attention semantics: {name}")
    if scaling is not None:
        self_attn.scaling = scaling
    softcap = kwargs.get("softcap", kwargs.get("soft_cap"))
    if softcap is not None:
        self_attn.logit_cap = softcap
    causal = kwargs.get("is_causal", getattr(module, "is_causal", True))
    if causal != (self_attn.attn_type == AttentionType.DECODER):
        raise ValueError("Attention causality changed after model initialization")
    if query.ndim != 4 or query.shape[0] != 1:
        raise ValueError(
            "Transformers attention expects packed [1, heads, tokens, dim] tensors"
        )
    tokens = query.shape[-2]
    query, key, value = (
        x.transpose(1, 2).reshape(tokens, -1) for x in (query, key, value)
    )
    native_kwargs = {}
    sinks = kwargs.get("sinks", kwargs.get("attention_sinks", kwargs.get("s_aux")))
    if sinks is not None:
        if sinks.ndim != 1:
            raise ValueError("Attention sinks must contain one scalar per head")
        if sinks.shape[0] != self_attn.tp_q_head_num:
            rank = get_parallel().attn_tp_rank
            sinks = sinks.narrow(
                0, rank * self_attn.tp_q_head_num, self_attn.tp_q_head_num
            )
        native_kwargs["sinks"] = sinks
    output = self_attn(query, key, value, forward_batch=forward_batch, **native_kwargs)
    width = getattr(self_attn, "output_head_dim", self_attn.v_head_dim)
    return output.reshape(1, tokens, self_attn.tp_q_head_num, width), None


ALL_ATTENTION_FUNCTIONS["sglang"] = sglang_flash_attention_forward


class AttentionMixin:
    def validate_attention_backend(self, prefill_backend, decode_backend):
        has_bidirectional_window = any(
            layer.attn_type == AttentionType.ENCODER_ONLY
            and layer.sliding_window_size is not None
            and layer.sliding_window_size >= 0
            for layer in self.attention_instances.values()
        )
        if has_bidirectional_window and prefill_backend not in {
            "torch_native",
            "fa3",
            "fa4",
        }:
            raise ValueError(
                "Transformers bidirectional sliding-window attention requires "
                "the torch_native, fa3, or fa4 prefill backend; "
                f"received {prefill_backend!r}"
            )

    def _create_attention_instances(self):
        instances = nn.ModuleDict()
        tp_size = get_parallel().attn_tp_size
        for name, module in self.model.named_modules():
            if (
                not hasattr(module, "is_causal")
                or getattr(module, "layer_idx", None) is None
            ):
                continue
            config = getattr(module, "config", self.text_config)
            if config is not self.text_config and self.config is not self.text_config:
                continue
            idx = module.layer_idx
            if not self.start_layer <= idx < self.end_layer:
                continue
            key = str(idx)
            if key in instances:
                raise ValueError(
                    f"Multiple language attention modules for layer {idx}: {name}"
                )
            desc = AttentionDescriptor.from_module(module, self.text_config, tp_size)
            module._sglang_attention_key = key
            instances[key] = RadixAttention(
                num_heads=desc.num_heads,
                num_kv_heads=desc.num_kv_heads,
                head_dim=desc.head_dim,
                v_head_dim=desc.value_head_dim,
                scaling=desc.scaling,
                layer_id=idx,
                quant_config=self.quant_config,
                sliding_window_size=desc.sliding_window,
                logit_cap=desc.softcap,
                attn_type=AttentionType.DECODER
                if desc.causal
                else AttentionType.ENCODER_ONLY,
                prefix=f"model.{name}.attn",
            )
        if not instances and self.start_layer != self.end_layer:
            raise ValueError("No compatible language attention modules were found")
        return instances
