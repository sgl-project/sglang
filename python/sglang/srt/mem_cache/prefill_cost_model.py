"""Prefill compute model for hybrid recurrent models.

The ``marconi`` radix eviction policy (Marconi, arXiv:2411.19379) values a
cached recurrent-state checkpoint by the prefill FLOPs it saves per byte it
holds. This module estimates those FLOPs per tensor-parallel rank for every
layer type the MAMBA cache component serves: Mamba2/SSD (Nemotron-H,
Falcon-H1, ...), gated delta rule (GDN: Qwen3-Next, Qwen3.5, ...), Kimi delta
attention (KDA: Kimi Linear, ...) and lightning-style linear attention.

Prefilling ``L`` tokens from scratch costs

    F(L) = linear_flops_per_token * L + attention_flops_per_token_pair * L^2 / 2

where the linear term covers the dense and expert projections (2 FLOPs per
active parameter), the recurrent state update and the short convolution, and
the quadratic term covers the query-key and probability-value products of the
full-attention layers. Extending a cached prefix of ``a`` tokens to ``b``
tokens costs ``F(b) - F(a)``.

Only ratios between checkpoints matter to the eviction policy, so the model
aims at the right ratio between the linear and quadratic terms rather than at
exact FLOP counts.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    import torch

    from sglang.srt.configs.mamba_utils import BaseLinearStateParams
    from sglang.srt.configs.model_config import ModelConfig

logger = logging.getLogger(__name__)

# FLOPs per element of a layer's recurrent state, per token.
RECURRENCE_FLOPS_PER_STATE_ELEMENT = {
    # Mamba2 SSD: h <- exp(dt * A) * h + dt * x B^T (3), y = h C (2).
    "mamba2": 5,
    # Lightning / linear attention: S <- lambda * S + k^T v (3), o = q S (2).
    "lightning": 5,
    # Gated delta rule: S <- g * S (1), u = S k (2),
    # S <- S + beta * (v - u) k^T (2), o = S q (2).
    "gdn": 7,
    # Kimi delta attention: the gated delta rule with a per-channel decay
    # diag(g) S, which still costs one multiply per state element.
    "kda": 7,
}
RECURRENCE_KINDS = tuple(RECURRENCE_FLOPS_PER_STATE_ELEMENT)


@dataclass(frozen=True)
class PrefillCostModel:
    """Per-rank prefill FLOPs, and the bytes of one recurrent-state checkpoint."""

    linear_flops_per_token: float
    attention_flops_per_token_pair: float
    # One checkpoint across all recurrent layers: conv state plus SSM state.
    state_bytes: int

    def __post_init__(self):
        if self.linear_flops_per_token < 0 or self.attention_flops_per_token_pair < 0:
            raise ValueError(f"FLOP coefficients must be non-negative, got {self}")
        if self.linear_flops_per_token == self.attention_flops_per_token_pair == 0:
            raise ValueError("a prefill cost model needs a positive FLOP coefficient")
        if self.state_bytes <= 0:
            raise ValueError(f"state_bytes must be positive, got {self.state_bytes}")

    def prefill_flops(self, num_tokens: int) -> float:
        """FLOPs to prefill ``num_tokens`` tokens from an empty cache."""
        return (
            self.linear_flops_per_token * num_tokens
            + 0.5 * self.attention_flops_per_token_pair * num_tokens * num_tokens
        )

    def extend_flops(self, cached_len: int, prefix_len: int) -> float:
        """FLOPs to extend a cached prefix of ``cached_len`` to ``prefix_len`` tokens."""
        if prefix_len <= cached_len:
            return 0.0
        return self.prefill_flops(prefix_len) - self.prefill_flops(cached_len)

    def flops_per_state_byte(self, cached_len: int, prefix_len: int) -> float:
        """Prefill FLOPs a checkpoint at ``prefix_len`` saves per byte it holds,
        when the nearest shallower checkpoint sits at ``cached_len``."""
        return self.extend_flops(cached_len, prefix_len) / self.state_bytes


# A placeholder for callers that have no model to inspect (unit tests, or a
# model the builder cannot read): savings grow with the tokens recomputed.
TOKEN_COUNT_COST_MODEL = PrefillCostModel(
    linear_flops_per_token=1.0, attention_flops_per_token_pair=0.0, state_bytes=1
)


def recurrent_flops_per_token(kind: str, cache_params: BaseLinearStateParams) -> float:
    """Per-rank FLOPs per token of the state update and short convolution,
    summed over the recurrent layers described by ``cache_params``.

    The projections around the recurrence are model parameters and belong to
    the 2 FLOPs per active parameter term instead.
    """
    try:
        per_element = RECURRENCE_FLOPS_PER_STATE_ELEMENT[kind]
    except KeyError:
        raise ValueError(
            f"unknown recurrence kind {kind!r}; expected one of {RECURRENCE_KINDS}"
        ) from None
    shape = cache_params.shape
    state_elements = math.prod(shape.temporal)
    # Each conv state holds the last (kernel - 1) inputs of every channel, and
    # a token costs one multiply-add per channel and kernel tap.
    conv_numel = sum(math.prod(conv_shape) for conv_shape in shape.conv)
    conv_kernel = getattr(shape, "conv_kernel", None)
    if conv_kernel is not None and conv_kernel > 1:
        conv_flops = 2 * conv_numel * conv_kernel / (conv_kernel - 1)
    else:
        conv_flops = 2 * conv_numel
    return len(cache_params.layers) * (per_element * state_elements + conv_flops)


def recurrence_kind(model_config: ModelConfig) -> str:
    """Name the recurrence of a hybrid model, for RECURRENCE_FLOPS_PER_STATE_ELEMENT."""
    from sglang.srt.configs.hybrid_arch import (
        hybrid_gdn_config,
        hybrid_kda_config,
        hybrid_lightning_config,
        linear_attn_model_spec,
        mamba2_config,
    )

    if hybrid_kda_config(model_config) is not None:
        return "kda"
    if hybrid_gdn_config(model_config) is not None:
        return "gdn"
    if hybrid_lightning_config(model_config) is not None:
        return "lightning"
    if mamba2_config(model_config) is not None:
        return "mamba2"
    spec = linear_attn_model_spec(model_config)
    backend = spec.backend_class_name.lower() if spec is not None else ""
    for kind in ("kda", "gdn", "lightning", "mamba2"):
        if kind in backend:
            return kind
    # A delta-rule update is the most expensive kind, and the term is small
    # next to the projections either way.
    return "gdn"


def _attention_coefficients(model: torch.nn.Module) -> tuple[float, float]:
    """(linear, quadratic) per-rank FLOPs of the model's softmax attention.

    A full-attention layer costs 2 * heads * (qk_dim + v_dim) FLOPs per
    (query, key) pair. A sliding-window layer attends to at most its window,
    so its cost is linear in the prompt and counted per token.
    """
    from sglang.srt.layers.radix_attention import AttentionType, RadixAttention

    linear = quadratic = 0.0
    for module in model.modules():
        if not isinstance(module, RadixAttention) or module.is_cross_attention:
            continue
        if module.attn_type == AttentionType.ENCODER_ONLY:
            continue
        pair_flops = 2 * module.tp_q_head_num * (module.qk_head_dim + module.v_head_dim)
        if module.sliding_window_size is not None and module.sliding_window_size > 0:
            linear += pair_flops * module.sliding_window_size
        else:
            quadratic += pair_flops
    return linear, quadratic


_NON_TEXT_PARAM_PREFIXES = ("visual.", "vision_tower.", "vision_model.", "audio_tower.")


def _active_parameter_count(model: torch.nn.Module) -> float:
    """Per-rank parameters a prompt token multiplies through.

    Embedding and output tables are excluded (a lookup, and logits for only the
    last token of a prefill), as are vision and audio towers. Routed experts
    count with the fraction of experts a token activates.
    """
    from sglang.srt.layers.moe.fused_moe_triton.layer import FusedMoE
    from sglang.srt.layers.vocab_parallel_embedding import VocabParallelEmbedding

    weight: dict[int, float] = {}
    for module in model.modules():
        if isinstance(module, VocabParallelEmbedding):
            fraction = 0.0
        elif isinstance(module, FusedMoE):
            fraction = min(1.0, module.top_k / max(module.num_experts, 1))
        else:
            continue
        for param in module.parameters():
            weight[id(param)] = fraction
    total = 0.0
    for name, param in model.named_parameters():
        if name.startswith(_NON_TEXT_PARAM_PREFIXES):
            continue
        total += param.numel() * weight.get(id(param), 1.0)
    return total


def build_prefill_cost_model(
    model_config: ModelConfig,
    model: torch.nn.Module,
    cache_params: BaseLinearStateParams,
    kind: Optional[str] = None,
) -> PrefillCostModel:
    """Estimate a hybrid model's per-rank prefill FLOPs from its loaded modules."""
    kind = kind or recurrence_kind(model_config)
    window_linear, quadratic = _attention_coefficients(model)
    linear = (
        2 * _active_parameter_count(model)
        + recurrent_flops_per_token(kind, cache_params)
        + window_linear
    )
    cost = PrefillCostModel(
        linear_flops_per_token=linear,
        attention_flops_per_token_pair=quadratic,
        state_bytes=cache_params.mamba_cache_per_req,
    )
    logger.info(
        "Prefill cost model (%s recurrence): %.3g FLOPs/token + %.3g FLOPs/token pair "
        "(attention dominates past %s tokens), %.1f MiB per state checkpoint",
        kind,
        cost.linear_flops_per_token,
        cost.attention_flops_per_token_pair,
        (
            f"{2 * linear / quadratic:.0f}"
            if quadratic > 0
            else "never; no full-attention layers"
        ),
        cost.state_bytes / 2**20,
    )
    return cost


def maybe_build_prefill_cost_model(
    eviction_policy: str, model_config: ModelConfig, model: Any
) -> Optional[PrefillCostModel]:
    """The prefill cost model the ``marconi`` eviction policy reads, else None."""
    if eviction_policy.lower() != "marconi":
        return None
    from sglang.srt.configs.hybrid_arch import mambaish_config

    config = mambaish_config(model_config)
    cache_params = getattr(config, "mamba2_cache_params", None)
    if cache_params is None:
        return None
    try:
        return build_prefill_cost_model(model_config, model, cache_params)
    except (AttributeError, ImportError, TypeError, ValueError) as e:
        logger.warning(
            "Could not build the prefill cost model (%s); the marconi eviction "
            "policy weighs checkpoints by the tokens they save instead.",
            e,
        )
        return None
