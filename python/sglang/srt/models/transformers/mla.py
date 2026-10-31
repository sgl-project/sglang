# Copyright 2026 SGLang Team
# SPDX-License-Identifier: Apache-2.0

import re
from types import MethodType
from weakref import ref

import torch
from torch import nn

from sglang.srt.layers.radix_attention import RadixAttention


def validate_mla_backend_options(
    prefill_backend,
    decode_backend,
    *,
    kv_cache_dtype="auto",
    dcp_enabled=False,
    enable_dp_attention=False,
    enable_lora=False,
):
    supported = {"flashinfer"}
    unsupported = {prefill_backend, decode_backend} - supported
    if unsupported:
        raise ValueError(
            f"Transformers native MLA does not support attention backends: {sorted(unsupported, key=str)}"
        )
    if kv_cache_dtype not in ("auto", "float16", "bfloat16"):
        raise ValueError("Transformers native MLA requires an unquantized KV cache")
    if dcp_enabled or enable_dp_attention:
        raise ValueError(
            "Transformers native MLA does not support context or data parallel attention"
        )
    if enable_lora:
        raise ValueError(
            "Transformers native MLA absorption does not support LoRA adapters"
        )


def split_mla_projection(
    weight: torch.Tensor, num_heads: int, nope_dim: int, value_dim: int
) -> tuple[torch.Tensor, torch.Tensor]:
    if weight.ndim != 2 or weight.shape[0] != num_heads * (nope_dim + value_dim):
        raise ValueError("MLA KV projection does not match local head geometry")
    by_head = weight.unflatten(0, (num_heads, nope_dim + value_dim))
    key, value = by_head.split((nope_dim, value_dim), dim=1)
    return key.transpose(1, 2).contiguous(), value.contiguous()


def absorb_mla_query(query: torch.Tensor, key_weight: torch.Tensor) -> torch.Tensor:
    return torch.bmm(query.transpose(0, 1), key_weight.transpose(1, 2)).transpose(0, 1)


def expand_mla_output(latent: torch.Tensor, value_weight: torch.Tensor) -> torch.Tensor:
    return torch.bmm(latent.transpose(0, 1), value_weight.transpose(1, 2)).transpose(
        0, 1
    )


def _latent_kv(self, kv_nope: torch.Tensor, k_rot: torch.Tensor):
    return kv_nope, k_rot


class TransformersMLAAttention(RadixAttention):
    is_mla = True

    def __init__(self, module: nn.Module, attention: RadixAttention):
        self.input_head_dim = module.qk_nope_head_dim + module.qk_rope_head_dim
        self.output_head_dim = module.v_head_dim
        self.nope_dim = module.qk_nope_head_dim
        self.rope_dim = module.qk_rope_head_dim
        self.latent_dim = module.kv_lora_rank
        super().__init__(
            num_heads=attention.tp_q_head_num,
            head_dim=self.latent_dim + self.rope_dim,
            scaling=attention.scaling,
            num_kv_heads=1,
            layer_id=attention.layer_id,
            v_head_dim=self.latent_dim,
            prefix=f"{attention.layer_id}.mla",
        )
        self._source_ref = ref(module)
        self.register_buffer("w_kc", None, persistent=False)
        self.register_buffer("w_vc", None, persistent=False)

    def refresh_weights(self):
        source = self._source_ref()
        if source is None:
            raise RuntimeError("MLA source attention is no longer alive")
        weight = source.kv_b_proj.weight
        if weight.is_meta:
            raise RuntimeError(
                "MLA projection must be loaded before refreshing weights"
            )
        if weight.dtype not in (torch.float16, torch.bfloat16, torch.float32):
            raise ValueError("MLA absorption requires an unquantized KV projection")
        key, value = split_mla_projection(
            weight.detach(), self.tp_q_head_num, self.nope_dim, self.output_head_dim
        )
        for name, tensor in (("w_kc", key), ("w_vc", value)):
            previous = getattr(self, name)
            if previous is not None and previous.shape == tensor.shape:
                previous.copy_(tensor)
            else:
                setattr(self, name, tensor)

    def forward(self, q, k, v, forward_batch, **kwargs):
        if self.w_kc is None or self.w_vc is None:
            raise RuntimeError("MLA weights have not been refreshed after loading")
        query = q.reshape(-1, self.tp_q_head_num, self.input_head_dim)
        q_nope, q_rope = query.split((self.nope_dim, self.rope_dim), dim=-1)
        q_latent = absorb_mla_query(q_nope, self.w_kc)
        latent = k.reshape(-1, 1, self.latent_dim)
        k_rope = v.reshape(-1, 1, self.rope_dim)
        output = super().forward(
            q_latent,
            latent,
            latent,
            forward_batch,
            q_rope=q_rope,
            k_rope=k_rope,
            **kwargs,
        )
        output = output.reshape(-1, self.tp_q_head_num, self.latent_dim)
        return expand_mla_output(output, self.w_vc).reshape(
            -1, self.tp_q_head_num * self.output_head_dim
        )


def supports_mla_module(module: nn.Module) -> bool:
    cls = type(module)
    if (cls.__module__, cls.__name__) not in {
        ("transformers.models.deepseek_v2.modeling_deepseek_v2", "DeepseekV2Attention"),
        ("transformers.models.deepseek_v3.modeling_deepseek_v3", "DeepseekV3Attention"),
    }:
        return False
    if not callable(getattr(module, "expand_kv", None)):
        return False
    projection = getattr(module, "kv_b_proj", None)
    weight = getattr(projection, "weight", None)
    if (
        weight is None
        or weight.ndim != 2
        or getattr(projection, "bias", None) is not None
    ):
        return False
    return all(
        isinstance(getattr(module, name, None), int) and getattr(module, name) > 0
        for name in (
            "qk_nope_head_dim",
            "qk_rope_head_dim",
            "kv_lora_rank",
            "v_head_dim",
        )
    )


def mla_attention_tp_plan(model: nn.Module) -> dict[str, str]:
    plan = {}
    for name, module in model.named_modules():
        if not supports_mla_module(module):
            continue
        for projection, style in (
            ("q_proj", "colwise"),
            ("q_a_proj", "replicate"),
            ("q_b_proj", "colwise"),
            ("kv_a_proj_with_mqa", "replicate"),
            ("kv_b_proj", "colwise"),
            ("o_proj", "rowwise"),
        ):
            if isinstance(getattr(module, projection, None), nn.Linear):
                plan[re.escape(f"{name}.{projection}")] = style
    return plan


def install_mla_adapters(model, attention_instances, quant_config=None) -> int:
    if quant_config is not None:
        raise ValueError(
            "Transformers native MLA currently requires unquantized weights"
        )
    candidates = [
        m
        for m in model.modules()
        if hasattr(m, "kv_lora_rank") and hasattr(m, "layer_idx")
    ]
    if not candidates or not all(supports_mla_module(m) for m in candidates):
        raise ValueError(
            "Transformers MLA requires a supported latent expand_kv contract"
        )
    for module in candidates:
        key = getattr(module, "_sglang_attention_key", str(module.layer_idx))
        attention = attention_instances[key]
        if (
            not getattr(module, "is_causal", False)
            or attention.sliding_window_size != -1
        ):
            raise ValueError("Transformers MLA requires causal full attention")
        projection = module.kv_b_proj
        expected = attention.tp_q_head_num * (
            module.qk_nope_head_dim + module.v_head_dim
        )
        if projection.weight.shape != (expected, module.kv_lora_rank):
            raise ValueError(
                "Transformers MLA KV projection has incompatible TP layout"
            )
    for module in candidates:
        key = getattr(module, "_sglang_attention_key", str(module.layer_idx))
        attention_instances[key] = TransformersMLAAttention(
            module, attention_instances[key]
        )
        module.expand_kv = MethodType(_latent_kv, module)
    return len(candidates)


def refresh_mla_weights(model) -> None:
    for module in model.modules():
        if isinstance(module, TransformersMLAAttention):
            module.refresh_weights()
