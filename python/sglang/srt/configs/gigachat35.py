from __future__ import annotations

from typing import Optional

import torch
from transformers.configuration_utils import PretrainedConfig

from sglang.srt.configs.mamba_utils import (
    Mamba2CacheParams,
    Mamba2StateShape,
    mamba2_state_dtype,
)

FULL_ATTENTION = "attention"
LINEAR_ATTENTION = "linear_attention"
_FULL_ATTENTION_ALIASES = {FULL_ATTENTION, "full_attention"}

_REQUIRED_LINEAR_ATTRS = (
    "linear_conv_kernel_dim",
    "linear_key_head_dim",
    "linear_value_head_dim",
    "linear_num_key_heads",
    "linear_num_value_heads",
)


class GigaChat35Config(PretrainedConfig):
    model_type = "gigachat3_5"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        vocab_size: int = 128256,
        hidden_size: int = 1280,
        intermediate_size: int = 896,
        moe_intermediate_size: int = 896,
        num_hidden_layers: int = 12,
        num_attention_heads: int = 16,
        num_key_value_heads: int = 16,
        hidden_act: str = "silu",
        max_position_embeddings: int = 4096,
        initializer_range: float = 0.006,
        rms_norm_eps: float = 1e-6,
        use_cache: bool = True,
        rope_theta: float = 100000.0,
        rope_scaling: Optional[dict] = None,
        rope_interleave: bool = True,
        attention_bias: bool = False,
        tie_word_embeddings: bool = False,
        q_lora_rank: Optional[int] = 1536,
        kv_lora_rank: int = 512,
        qk_nope_head_dim: int = 128,
        qk_rope_head_dim: int = 64,
        v_head_dim: int = 128,
        head_dim: int = 64,
        n_routed_experts: int = 64,
        n_shared_experts: int = 2,
        num_experts_per_tok: int = 6,
        moe_layer_freq: int = 1,
        first_k_dense_replace: int = 1,
        routed_scaling_factor: float = 2.5,
        n_group: int = 1,
        topk_group: int = 1,
        topk_method: str = "noaux_tc",
        scoring_func: str = "sigmoid",
        norm_topk_prob: bool = True,
        use_shared_expert_sigmoid: bool = False,
        linear_attention_type: str = "Qwen3NextGatedDeltaNet",
        layer_types: Optional[list[str]] = None,
        full_attention_layers: Optional[list[int]] = None,
        linear_conv_kernel_dim: int = 4,
        linear_key_head_dim: int = 128,
        linear_value_head_dim: int = 128,
        linear_num_key_heads: int = 8,
        linear_num_value_heads: int = 16,
        linear_sigmoid_gate_scale: float = 2.0,
        output_gate_type: str = "sigmoid",
        norm_type: str = "ZeroCenteredGatedNorm",
        layernorm_type: str = "pre_post",
        layernorm_gating_weight: float = 2.0,
        gated_attention: bool = True,
        use_mla_scaling_factor: bool = True,
        swiglu_limit: Optional[float] = None,
        num_nextn_predict_layers: int = 0,
        nextn_is_sparse: bool = False,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.moe_intermediate_size = moe_intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.hidden_act = hidden_act
        self.max_position_embeddings = max_position_embeddings
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.use_cache = use_cache
        self.rope_theta = rope_theta
        self.rope_scaling = rope_scaling
        self.rope_interleave = rope_interleave
        self.attention_bias = attention_bias

        self.q_lora_rank = q_lora_rank
        self.kv_lora_rank = kv_lora_rank
        self.qk_nope_head_dim = qk_nope_head_dim
        self.qk_rope_head_dim = qk_rope_head_dim
        self.v_head_dim = v_head_dim
        self.head_dim = head_dim

        self.n_routed_experts = n_routed_experts
        self.n_shared_experts = n_shared_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.moe_layer_freq = moe_layer_freq
        self.first_k_dense_replace = first_k_dense_replace
        self.routed_scaling_factor = routed_scaling_factor
        self.n_group = n_group
        self.topk_group = topk_group
        self.topk_method = topk_method
        self.scoring_func = scoring_func
        self.norm_topk_prob = norm_topk_prob
        self.use_shared_expert_sigmoid = use_shared_expert_sigmoid

        self.linear_attention_type = linear_attention_type
        self.layer_types = layer_types
        self.full_attention_layers = full_attention_layers
        self.linear_conv_kernel_dim = linear_conv_kernel_dim
        self.linear_key_head_dim = linear_key_head_dim
        self.linear_value_head_dim = linear_value_head_dim
        self.linear_num_key_heads = linear_num_key_heads
        self.linear_num_value_heads = linear_num_value_heads
        self.linear_sigmoid_gate_scale = linear_sigmoid_gate_scale
        self.output_gate_type = output_gate_type
        self.linear_num_key_heads_cpu = linear_num_key_heads
        self.linear_num_value_heads_cpu = linear_num_value_heads

        self.norm_type = norm_type
        self.layernorm_type = layernorm_type
        self.layernorm_gating_weight = layernorm_gating_weight
        self.gated_attention = gated_attention
        self.use_mla_scaling_factor = use_mla_scaling_factor
        self.swiglu_limit = (
            swiglu_limit if (swiglu_limit is not None and swiglu_limit > 0) else None
        )

        self.num_nextn_predict_layers = num_nextn_predict_layers
        self.nextn_is_sparse = nextn_is_sparse

        super().__init__(tie_word_embeddings=tie_word_embeddings, **kwargs)

        _dtype = getattr(self, "torch_dtype", None) or getattr(self, "dtype", None)
        if isinstance(_dtype, str):
            _dtype = getattr(torch, _dtype, torch.bfloat16)
        self.torch_dtype = _dtype or torch.bfloat16

    def _resolve_layer_types(self) -> list[str]:
        """Return a normalized per-layer list of FULL_ATTENTION / LINEAR_ATTENTION.

        Resolution order: explicit ``layer_types`` -> ``full_attention_layers``.
        Defaults to all-linear if nothing is specified (degenerate, but
        well-defined).
        """
        n = self.num_hidden_layers

        if self.layer_types is not None:
            if len(self.layer_types) != n:
                raise ValueError(
                    f"layer_types must have length num_hidden_layers ({n}), "
                    f"got {len(self.layer_types)}."
                )
            resolved = []
            for idx, lt in enumerate(self.layer_types):
                if lt == LINEAR_ATTENTION:
                    resolved.append(LINEAR_ATTENTION)
                elif lt in _FULL_ATTENTION_ALIASES:
                    resolved.append(FULL_ATTENTION)
                else:
                    raise ValueError(f"Unsupported layer type {lt!r} at index {idx}.")
            return resolved

        resolved = [LINEAR_ATTENTION] * n
        if self.full_attention_layers is not None:
            for lid in self.full_attention_layers:
                resolved[lid] = FULL_ATTENTION
        return resolved

    @property
    def layers_block_type(self) -> list[str]:
        return self._resolve_layer_types()

    @property
    def linear_layer_ids(self) -> list[int]:
        return [
            i for i, lt in enumerate(self.layers_block_type) if lt == LINEAR_ATTENTION
        ]

    @property
    def full_attention_layer_ids(self) -> list[int]:
        return [
            i for i, lt in enumerate(self.layers_block_type) if lt == FULL_ATTENTION
        ]

    def is_linear_attention_layer(self, layer_id: int) -> bool:
        if layer_id >= self.num_hidden_layers:
            return False
        return self.layers_block_type[layer_id] == LINEAR_ATTENTION

    @property
    def mamba2_cache_params(self) -> Mamba2CacheParams:
        from sglang.srt.runtime_context import get_parallel

        missing = [a for a in _REQUIRED_LINEAR_ATTRS if getattr(self, a, None) is None]
        if missing:
            raise ValueError(
                "GigaChat35 hybrid GDN config is missing required linear-attention "
                "fields: " + ", ".join(missing)
            )

        shape = Mamba2StateShape.create(
            tp_world_size=get_parallel().attn_tp_size,
            intermediate_size=self.linear_value_head_dim * self.linear_num_value_heads,
            n_groups=self.linear_num_key_heads,
            num_heads=self.linear_num_value_heads,
            head_dim=self.linear_value_head_dim,
            state_size=self.linear_key_head_dim,
            conv_kernel=self.linear_conv_kernel_dim,
        )
        return Mamba2CacheParams(
            shape=shape,
            layers=self.linear_layer_ids,
            dtype=mamba2_state_dtype(self),
        )


from sglang.srt.configs.linear_attn_model_registry import (  # noqa: E402
    LinearAttnModelSpec,
    register_linear_attn_model,
)

register_linear_attn_model(
    LinearAttnModelSpec(
        config_class=GigaChat35Config,
        backend_class_name="sglang.srt.layers.attention.linear.gdn_backend.GDNAttnBackend",
        arch_names=[
            "GigaChat35ForCausalLM",
            "GigaChat35ForCausalLMNextN",
        ],
        uses_mamba_radix_cache=True,
        support_mamba_cache=True,
    )
)
