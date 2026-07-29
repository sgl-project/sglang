# Copyright 2025 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""LFM2-MoE (Liquid Foundation Model 2 - Mixture of Experts) configuration

Note: HF transformers has Lfm2MoeConfig in v5.0.0rc2 (unreleased).
Once released, we could inherit from it like Lfm2Config does with HFLfm2Config.
For now, we define a standalone config to support the model immediately.
"""

from typing import List, Optional

from transformers import CONFIG_MAPPING
from transformers.configuration_utils import PretrainedConfig

from sglang.srt.configs.mamba_utils import Mamba2CacheParams, Mamba2StateShape


class Lfm2MoeConfig(PretrainedConfig):
    """
    Configuration for LFM2-MoE models (e.g., LiquidAI/LFM2-8B-A1B).

    LFM2-MoE is a hybrid architecture with:
    - Attention layers and ShortConv layers (like dense LFM2)
    - MoE (Mixture of Experts) FFN layers with sigmoid routing

    Key MoE specifics:
    - First `num_dense_layers` use dense MLP, rest use MoE
    - Sigmoid routing (not softmax) with expert_bias for load balancing
    - expert_bias is fp32 for numerical stability
    """

    model_type = "lfm2_moe"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        vocab_size: int = 65536,
        hidden_size: int = 2048,
        intermediate_size: int = 7168,
        moe_intermediate_size: int = 1792,
        num_hidden_layers: int = 32,
        num_attention_heads: int = 32,
        num_key_value_heads: int = 8,
        max_position_embeddings: int = 128000,
        initializer_range: float = 0.02,
        norm_eps: float = 1e-5,
        use_cache: bool = True,
        pad_token_id: int = 0,
        bos_token_id: int = 1,
        eos_token_id: int = 2,
        tie_word_embeddings: bool = True,
        rope_parameters: Optional[dict] = None,
        conv_bias: bool = False,
        conv_L_cache: int = 3,
        # MoE-specific parameters
        num_dense_layers: int = 2,
        num_experts: int = 32,
        num_experts_per_tok: int = 4,
        use_expert_bias: bool = True,
        routed_scaling_factor: float = 1.0,
        norm_topk_prob: bool = True,
        # Layer types
        layer_types: Optional[List[str]] = None,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.moe_intermediate_size = moe_intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.max_position_embeddings = max_position_embeddings
        self.initializer_range = initializer_range
        self.norm_eps = norm_eps
        self.use_cache = use_cache

        # Conv parameters
        self.conv_bias = conv_bias
        self.conv_L_cache = conv_L_cache

        # MoE parameters
        self.num_dense_layers = num_dense_layers
        self.num_experts = num_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.use_expert_bias = use_expert_bias
        self.routed_scaling_factor = routed_scaling_factor
        self.norm_topk_prob = norm_topk_prob

        # Layer types (attention vs conv)
        self.layer_types = layer_types

        # RoPE parameters
        self.rope_parameters = rope_parameters

        # Validate layer_types length matches num_hidden_layers
        if layer_types is not None and len(layer_types) != num_hidden_layers:
            raise ValueError(
                f"layer_types length ({len(layer_types)}) must match "
                f"num_hidden_layers ({num_hidden_layers})"
            )

        # Handle tie_embedding alias from original config
        tie_word_embeddings = kwargs.pop("tie_embedding", tie_word_embeddings)

        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )

    @property
    def full_attention_layer_ids(self) -> List[int]:
        """Return indices of attention layers for KV cache."""
        if self.layer_types is None:
            return []
        return [i for i, lt in enumerate(self.layer_types) if lt == "full_attention"]

    @property
    def linear_layer_ids(self) -> List[int]:
        """Return indices of conv layers for conv state cache."""
        if self.layer_types is None:
            return []
        return [
            i for i, lt in enumerate(self.layer_types) if lt in ("conv", "short_conv")
        ]

    @property
    def mamba_chunk_size(self) -> int:
        """Return chunk size for Mamba2 backend. LFM2 doesn't use chunking."""
        return 1

    @property
    def mamba2_cache_params(self) -> Optional[Mamba2CacheParams]:
        """
        Get cache params for HybridReqToTokenPool initialization.

        LFM2-MoE uses ShortConv layers with a small fixed-size cache.
        """
        from sglang.srt.layers.dp_attention import get_attention_tp_size

        conv_layer_ids = self.linear_layer_ids
        if not conv_layer_ids:
            return None

        hidden_size = self.hidden_size
        # conv_L_cache in config is kernel_size (e.g., 3)
        conv_kernel = int(self.conv_L_cache)
        # actual cache size is kernel_size - 1 (e.g., 2 for kernel=3)

        try:
            tp_size = get_attention_tp_size()
        except (AssertionError, RuntimeError):
            tp_size = 1

        shape = Mamba2StateShape.create(
            tp_world_size=tp_size,
            intermediate_size=hidden_size,
            n_groups=1,
            num_heads=tp_size,  # Ensures divide works; temporal state is empty anyway
            head_dim=hidden_size,
            state_size=0,
            conv_kernel=conv_kernel,
        )

        # Uses default mamba2_state_dtype() which reads SGLANG_MAMBA_CONV_DTYPE env var
        # (defaults to bfloat16). Set SGLANG_MAMBA_CONV_DTYPE=float16 for fp16 inference.
        return Mamba2CacheParams(
            shape=shape,
            layers=conv_layer_ids,
        )


# Register with transformers CONFIG_MAPPING so AutoConfig.from_pretrained()
# can instantiate our config class when loading models with model_type="lfm2_moe"
try:
    CONFIG_MAPPING.register("lfm2_moe", Lfm2MoeConfig)
except Exception:
    # Already registered or registration failed - use direct assignment
    CONFIG_MAPPING._extra_content["lfm2_moe"] = Lfm2MoeConfig
