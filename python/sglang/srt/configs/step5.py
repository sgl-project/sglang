# SPDX-License-Identifier: Apache-2.0
"""Step5 text and vision-language checkpoint configurations."""

from typing import Any, Optional, Union

import msgspec
from transformers.configuration_utils import PretrainedConfig

from sglang.srt.configs.step3p7 import Step3p7Config, Step3p7VisionEncoderConfig


class Step5Config(PretrainedConfig):
    model_type = "step5"
    architectures = ["Step5ForCausalLM"]

    def __init__(
        self,
        hidden_size: int = 4096,
        intermediate_size: int = 13824,
        num_attention_heads: int = 64,
        num_attention_groups: int = 4,
        num_hidden_layers: int = 92,
        head_dim: int = 192,
        vocab_size: int = 128896,
        rms_norm_eps: float = 1e-5,
        max_position_embeddings: int = 1048576,
        rope_theta: Optional[list[float]] = None,
        rope_scaling: Optional[dict[str, Any]] = None,
        partial_rotary_factors: Optional[list[float]] = None,
        layer_types: Optional[list[str]] = None,
        sliding_window: int = 512,
        moe_intermediate_size: int = 1536,
        share_expert_dim: int = 1536,
        moe_num_experts: int = 352,
        moe_top_k: int = 8,
        moe_router_scaling_factor: float = 1.0,
        moe_layers_enum: Optional[str] = None,
        swiglu_limits: Optional[list[float]] = None,
        swiglu_limits_shared: Optional[list[float]] = None,
        sparse_config: Optional[dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_attention_heads = num_attention_heads
        self.num_attention_groups = num_attention_groups
        self.num_hidden_layers = num_hidden_layers
        self.head_dim = head_dim
        self.vocab_size = vocab_size
        self.rms_norm_eps = rms_norm_eps
        self.max_position_embeddings = max_position_embeddings
        self.rope_theta = rope_theta
        self.rope_scaling = rope_scaling
        self.partial_rotary_factors = partial_rotary_factors
        self.layer_types = (
            list(layer_types[:num_hidden_layers]) if layer_types else None
        )
        self.sliding_window = sliding_window
        self.is_hybrid_swa = True
        self.moe_intermediate_size = moe_intermediate_size
        self.share_expert_dim = share_expert_dim
        self.moe_num_experts = moe_num_experts
        self.moe_top_k = moe_top_k
        self.moe_router_scaling_factor = moe_router_scaling_factor
        self.moe_layers_enum = moe_layers_enum
        self.moe_layer_list = (
            sorted(int(i) for i in moe_layers_enum.split(","))
            if moe_layers_enum
            else []
        )
        self.swiglu_limits = swiglu_limits
        self.swiglu_limits_shared = swiglu_limits_shared
        self.sparse_config = sparse_config or {}
        super().__init__(**kwargs)

    def wrap_attention_backend(self, runner, full_attn_backend):
        from sglang.srt.layers.attention.step5_backend import (
            wrap_step5_attention_backend,
        )

        return wrap_step5_attention_backend(runner, full_attn_backend)


class Step5VLConfig(Step3p7Config):
    model_type = "step5_vl"
    architectures = ["Step5ForConditionalGeneration"]

    def __init__(
        self,
        vision_config: Optional[Union[dict, Step3p7VisionEncoderConfig]] = None,
        text_config: Optional[Union[dict, PretrainedConfig]] = None,
        **kwargs: Any,
    ) -> None:
        if text_config is None or isinstance(text_config, dict):
            text_config = Step5Config(**(text_config or {}))
        kwargs.setdefault("language_model_only", True)
        super().__init__(vision_config=vision_config, text_config=text_config, **kwargs)


class Step5AttentionProfile(msgspec.Struct, frozen=True, kw_only=True):
    hidden_size: int
    head_dim: int
    num_heads: int
    num_kv_heads: int
    index_head_dim: int
    index_num_heads: int
    index_rope_dim: int
    provider_groups: int


def parse_step5_attention_profile(config: Step5Config) -> Step5AttentionProfile:
    sparse = config.sparse_config
    profile = Step5AttentionProfile(
        hidden_size=config.hidden_size,
        head_dim=config.head_dim,
        num_heads=config.num_attention_heads,
        num_kv_heads=config.num_attention_groups,
        index_head_dim=sparse["proxy_dim"],
        index_num_heads=sparse["sparse_indexer_num_heads"],
        index_rope_dim=sparse["sparse_indexer_rope_dim"],
        provider_groups=sparse["num_provider_groups"],
    )
    if (
        profile.head_dim != 192
        or profile.index_head_dim != 256
        or sparse["region_block_size"] != 8
        or sparse["sparse_indexer_num_k_heads"] != 1
        or config.sliding_window != 512
    ):
        raise ValueError(
            "Step5 kernels support head_dim 192, index_head_dim 256, "
            "8-token regions and a 512-token sliding window"
        )
    return profile
