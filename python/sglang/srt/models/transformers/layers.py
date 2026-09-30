# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Copyright 2025 SGLang Team

import re
from typing import Literal, Optional, Union

from torch import nn

from sglang.srt.layers.layernorm import GemmaRMSNorm, RMSNorm
from sglang.srt.layers.linear import (
    ColumnParallelLinear,
    MergedColumnParallelLinear,
    QKVParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
)
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.runtime_context import get_parallel

Style = Literal["colwise", "colwise_rep", "rowwise", "rowwise_rep", "replicate"]


class TensorOutputLinear:
    _hf_returns_tensor = True

    def forward_with_bias(self, *args, **kwargs):
        return super().forward(*args, **kwargs)

    def forward(self, *args, **kwargs):
        return self.forward_with_bias(*args, **kwargs)[0]


class HFCompatibleColumnParallelLinear(TensorOutputLinear, ColumnParallelLinear):
    parent_cls = ColumnParallelLinear


class HFCompatibleRowParallelLinear(TensorOutputLinear, RowParallelLinear):
    parent_cls = RowParallelLinear


class HFCompatibleReplicatedLinear(TensorOutputLinear, ReplicatedLinear):
    parent_cls = ReplicatedLinear


class HFCompatibleQKVParallelLinear(TensorOutputLinear, QKVParallelLinear):
    parent_cls = QKVParallelLinear


class HFCompatibleMergedColumnParallelLinear(
    TensorOutputLinear, MergedColumnParallelLinear
):
    parent_cls = MergedColumnParallelLinear


def replace_linear_class(
    linear: nn.Linear,
    style: Style = "replicate",
    quant_config: Optional[QuantizationConfig] = None,
    *,
    prefix: str = "",
    tp_rank: Optional[int] = None,
    tp_size: Optional[int] = None,
) -> Union[ColumnParallelLinear, RowParallelLinear, ReplicatedLinear]:
    if not isinstance(style, str):
        raise ValueError(f"Unsupported parallel style type {type(style)}, expected str")
    style = _normalize_tp_style(style)
    linear_class, linear_kwargs = {
        "colwise": (HFCompatibleColumnParallelLinear, {}),
        "colwise_rep": (HFCompatibleColumnParallelLinear, {"gather_output": True}),
        "rowwise": (HFCompatibleRowParallelLinear, {}),
        "rowwise_rep": (HFCompatibleRowParallelLinear, {"input_is_parallel": False}),
        "replicate": (HFCompatibleReplicatedLinear, {}),
    }[style]
    if tp_rank is not None or tp_size is not None:
        if style != "colwise":
            raise ValueError(
                "Projection shard overrides require an ungathered column parallel layer"
            )
        linear_kwargs.update(tp_rank=tp_rank, tp_size=tp_size)
    return linear_class(
        input_size=linear.in_features,
        output_size=linear.out_features,
        bias=linear.bias is not None,
        params_dtype=linear.weight.dtype,
        quant_config=quant_config,
        prefix=prefix,
        **linear_kwargs,
    )


def _normalize_tp_style(style: str) -> Style:
    style = style.lower().replace("-", "_")
    style = {
        "colwiseparallel": "colwise",
        "colwise_gather_output": "colwise_rep",
        "packed_colwise": "colwise",
        "local_colwise": "colwise",
        "rowwiseparallel": "rowwise",
        "rowwise_split_input": "rowwise_rep",
        "packed_rowwise": "rowwise",
        "local_rowwise": "rowwise",
        "local_packed_rowwise": "rowwise",
        "isolated": "replicate",
        "local": "replicate",
        "replicated_with_grad_allreduce": "replicate",
        "moe_tp_experts": "replicate",
        "mla_kv_a_proj": "replicate",
    }.get(style, style)
    if style not in {"colwise", "colwise_rep", "rowwise", "rowwise_rep", "replicate"}:
        raise ValueError(f"Unsupported TP style '{style}' for Transformers backend.")
    return style


class ShapePreservingNorm:
    def forward_add(self, x, residual):
        from .fusers.residual import residual_norm

        return residual_norm(self, x, residual)

    def forward(self, x, residual=None, **kwargs):
        if residual is None and not kwargs:
            from .fusers.strided_norm import strided_rms_norm

            output = strided_rms_norm(self, x)
            if output is not None:
                return output
        shape = x.shape
        x = x.reshape(-1, shape[-1]).contiguous()
        if residual is not None:
            residual = residual.reshape_as(x).contiguous()
        if kwargs.get("post_residual_addition") is not None:
            kwargs["post_residual_addition"] = (
                kwargs["post_residual_addition"].reshape_as(x).contiguous()
            )
        result = super().forward(x, residual=residual, **kwargs)
        if isinstance(result, tuple):
            return tuple(value.reshape(shape) for value in result)
        return result.reshape(shape)


class HFCompatibleRMSNorm(ShapePreservingNorm, RMSNorm):
    _hf_zero_centered = False


class HFCompatibleGemmaRMSNorm(ShapePreservingNorm, GemmaRMSNorm):
    _hf_zero_centered = True
    cast_x_before_out_mul = False


def replace_rms_norm_class(rms_norm: nn.Module, hidden_size: int) -> nn.Module:
    from .fusers.rms_norm import match_rms_norm

    semantics = match_rms_norm(rms_norm)
    if semantics is None:
        return rms_norm
    weight = getattr(rms_norm, "weight", None)
    if weight is not None:
        if weight.ndim != 1:
            return rms_norm
        width = weight.shape[0]
    else:
        width = next(
            (
                getattr(rms_norm, name)
                for name in ("hidden_size", "dim", "normalized_shape")
                if hasattr(rms_norm, name)
            ),
            None,
        )
        if isinstance(width, (tuple, list)) and len(width) == 1:
            width = width[0]
        if not isinstance(width, int) or width <= 0:
            return rms_norm
    if semantics.zero_centered:
        norm = HFCompatibleGemmaRMSNorm(width, eps=semantics.epsilon).to(
            dtype=weight.dtype
        )
    else:
        norm = HFCompatibleRMSNorm(
            width,
            eps=semantics.epsilon,
            has_weight=weight is not None,
            weight_dtype=weight.dtype if weight is not None else None,
            cast_x_before_out_mul=semantics.cast_before_weight,
        )
        if weight is None:
            tensor = norm.weight
            del norm.weight
            norm.register_buffer("weight", tensor, persistent=False)
    return norm


def get_attention_projection_shards(module, prefix, tp_plan):
    tp_size = get_parallel().attn_tp_size
    head_dim = getattr(module, "head_dim", getattr(module, "attention_head_size", None))
    if tp_size == 1 or not isinstance(head_dim, int) or head_dim <= 0:
        return {}
    for names in (("q_proj", "k_proj", "v_proj"), ("query", "key", "value")):
        if not all(
            isinstance(getattr(module, name, None), nn.Linear) for name in names
        ):
            continue
        q, k, v = (getattr(module, name) for name in names)
        if k.out_features != v.out_features or k.out_features % head_dim:
            continue
        kv_heads = k.out_features // head_dim
        if kv_heads >= tp_size or kv_heads <= 0:
            continue
        if tp_size % kv_heads or q.out_features % (tp_size * head_dim):
            raise ValueError(
                "Attention heads cannot be partitioned over the configured tensor parallel group"
            )
        shards = {}
        for name in names[1:]:
            path = f"{prefix}.{name}"
            style = next(
                (
                    style
                    for pattern, style in tp_plan.items()
                    if re.fullmatch(pattern, path)
                ),
                None,
            )
            if style == "colwise":
                shards[path] = {
                    "tp_rank": get_parallel().attn_tp_rank // (tp_size // kv_heads),
                    "tp_size": kv_heads,
                }
        return shards
    return {}
