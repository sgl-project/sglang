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
# ==============================================================================
"""Jamba model configuration

Jamba is AI21's hybrid Transformer-Mamba1-MoE model.
Unlike NemotronH which uses Mamba2, Jamba uses Mamba1 for SSM layers.

Key differences from NemotronH/Mamba2:
- Mamba1: 2D temporal state (intermediate_size/tp, state_size), no groups/heads
- Mamba2: 3D temporal state (num_heads/tp, head_dim, state_size), uses n_groups

Layer pattern (default Jamba 1.5):
- Attention layer every 8 layers (attn_layer_period=8, offset=4)
- MoE layer every 2 layers (expert_layer_period=2, offset=1)
- Rest are Mamba1 + dense MLP layers
"""

from typing import List, Union

from transformers.configuration_utils import PretrainedConfig
from transformers.utils import logging

from sglang.srt.configs.mamba_utils import (
    Mamba1CacheParams,
    Mamba1StateShape,
    mamba_state_dtype,
)

logger = logging.get_logger(__name__)

# Layer type constants
MAMBA = "mamba"
ATTENTION = "attention"
MLP = "mlp"
MOE = "moe"


class JambaConfig(PretrainedConfig):
    r"""
    Configuration class for Jamba model (AI21's hybrid Transformer-Mamba1-MoE).

    Args:
        vocab_size (`int`, *optional*, defaults to 65536):
            Vocabulary size of the Jamba model.
        tie_word_embeddings (`bool`, *optional*, defaults to `False`):
            Whether to tie input and output embeddings.
        hidden_size (`int`, *optional*, defaults to 4096):
            Dimension of the hidden representations.
        intermediate_size (`int`, *optional*, defaults to 14336):
            Dimension of the MLP representations.
        num_hidden_layers (`int`, *optional*, defaults to 32):
            Number of hidden layers in the model.
        num_attention_heads (`int`, *optional*, defaults to 32):
            Number of attention heads for each attention layer.
        num_key_value_heads (`int`, *optional*, defaults to 8):
            Number of key-value heads for GQA.
        hidden_act (`str`, *optional*, defaults to "silu"):
            Activation function for MLP layers.
        initializer_range (`float`, *optional*, defaults to 0.02):
            Standard deviation for weight initialization.
        rms_norm_eps (`float`, *optional*, defaults to 1e-6):
            Epsilon for RMS normalization.
        use_cache (`bool`, *optional*, defaults to `True`):
            Whether to use KV cache.
        pad_token_id (`int`, *optional*, defaults to 0):
            Padding token ID.
        bos_token_id (`int`, *optional*, defaults to 1):
            Beginning of sequence token ID.
        eos_token_id (`int`, *optional*, defaults to 2):
            End of sequence token ID.
        max_position_embeddings (`int`, *optional*, defaults to 262144):
            Maximum sequence length.
        attention_dropout (`float`, *optional*, defaults to 0.0):
            Dropout ratio for attention.
        attn_layer_period (`int`, *optional*, defaults to 8):
            Insert attention layer every N layers.
        attn_layer_offset (`int`, *optional*, defaults to 4):
            Offset for first attention layer.
        expert_layer_period (`int`, *optional*, defaults to 2):
            Insert MoE layer every N layers.
        expert_layer_offset (`int`, *optional*, defaults to 1):
            Offset for first MoE layer.
        use_mamba_kernels (`bool`, *optional*, defaults to `True`):
            Whether to use fast Mamba kernels.
        mamba_d_state (`int`, *optional*, defaults to 16):
            SSM state dimension for Mamba1.
        mamba_d_conv (`int`, *optional*, defaults to 4):
            Convolution kernel size for Mamba1.
        mamba_expand (`int`, *optional*, defaults to 2):
            Expansion factor for Mamba1 intermediate size.
        mamba_dt_rank (`int` or `str`, *optional*, defaults to "auto"):
            Rank of dt projection. "auto" sets it to ceil(hidden_size / 16).
        mamba_conv_bias (`bool`, *optional*, defaults to `True`):
            Whether to use bias in Mamba convolution.
        mamba_proj_bias (`bool`, *optional*, defaults to `False`):
            Whether to use bias in Mamba projections.
        num_experts (`int`, *optional*, defaults to 16):
            Number of experts in MoE layers.
        num_experts_per_tok (`int`, *optional*, defaults to 2):
            Number of experts activated per token (top-k).
        router_aux_loss_coef (`float`, *optional*, defaults to 0.001):
            Auxiliary loss coefficient for load balancing.
    """

    model_type = "jamba"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        vocab_size: int = 65536,
        tie_word_embeddings: bool = False,
        hidden_size: int = 4096,
        intermediate_size: int = 14336,
        num_hidden_layers: int = 32,
        num_attention_heads: int = 32,
        num_key_value_heads: int = 8,
        hidden_act: str = "silu",
        initializer_range: float = 0.02,
        rms_norm_eps: float = 1e-6,
        use_cache: bool = True,
        pad_token_id: int = 0,
        bos_token_id: int = 1,
        eos_token_id: int = 2,
        max_position_embeddings: int = 262144,
        attention_dropout: float = 0.0,
        # Layer pattern configuration
        attn_layer_period: int = 8,
        attn_layer_offset: int = 4,
        expert_layer_period: int = 2,
        expert_layer_offset: int = 1,
        # Mamba1 configuration
        use_mamba_kernels: bool = True,
        mamba_d_state: int = 16,
        mamba_d_conv: int = 4,
        mamba_expand: int = 2,
        mamba_dt_rank: Union[int, str] = "auto",
        mamba_conv_bias: bool = True,
        mamba_proj_bias: bool = False,
        # MoE configuration
        num_experts: int = 16,
        num_experts_per_tok: int = 2,
        router_aux_loss_coef: float = 0.001,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.tie_word_embeddings = tie_word_embeddings
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.max_position_embeddings = max_position_embeddings
        self.attention_dropout = attention_dropout

        # For backward compatibility
        if num_key_value_heads is None:
            num_key_value_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads

        self.hidden_act = hidden_act
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.use_cache = use_cache

        # Layer pattern configuration
        self.attn_layer_period = attn_layer_period
        self.attn_layer_offset = attn_layer_offset
        self.expert_layer_period = expert_layer_period
        self.expert_layer_offset = expert_layer_offset

        # Mamba1 configuration
        self.use_mamba_kernels = use_mamba_kernels
        self.mamba_d_state = mamba_d_state
        self.mamba_d_conv = mamba_d_conv
        self.mamba_expand = mamba_expand
        self.mamba_dt_rank = mamba_dt_rank
        self.mamba_conv_bias = mamba_conv_bias
        self.mamba_proj_bias = mamba_proj_bias

        # MoE configuration
        self.num_experts = num_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.router_aux_loss_coef = router_aux_loss_coef

        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )

    @property
    def mamba_intermediate_size(self) -> int:
        """Compute Mamba1 intermediate size from hidden_size and expand factor."""
        return self.hidden_size * self.mamba_expand

    @property
    def mamba_dt_rank_value(self) -> int:
        """Get the actual dt_rank value, computing if 'auto'."""
        if self.mamba_dt_rank == "auto":
            import math

            return math.ceil(self.hidden_size / 16)
        return self.mamba_dt_rank

    def _get_layer_block_type(self, layer_idx: int) -> str:
        """Determine if layer is attention or mamba."""
        if (layer_idx - self.attn_layer_offset) % self.attn_layer_period == 0:
            return ATTENTION
        return MAMBA

    def _get_layer_ffn_type(self, layer_idx: int) -> str:
        """Determine if layer uses MoE or dense MLP."""
        if (layer_idx - self.expert_layer_offset) % self.expert_layer_period == 0:
            return MOE
        return MLP

    def get_layer_types(self, layer_idx: int) -> tuple:
        """Get (block_type, ffn_type) for a given layer index."""
        return self._get_layer_block_type(layer_idx), self._get_layer_ffn_type(layer_idx)

    @property
    def mamba_layer_ids(self) -> List[int]:
        """Get list of layer indices that use Mamba1."""
        return [
            i
            for i in range(self.num_hidden_layers)
            if self._get_layer_block_type(i) == MAMBA
        ]

    @property
    def attention_layer_ids(self) -> List[int]:
        """Get list of layer indices that use attention."""
        return [
            i
            for i in range(self.num_hidden_layers)
            if self._get_layer_block_type(i) == ATTENTION
        ]

    @property
    def moe_layer_ids(self) -> List[int]:
        """Get list of layer indices that use MoE."""
        return [
            i
            for i in range(self.num_hidden_layers)
            if self._get_layer_ffn_type(i) == MOE
        ]

    @property
    def full_attention_layer_ids(self) -> List[int]:
        """Alias for attention_layer_ids (for compatibility with HybridLinearAttnBackend)."""
        return self.attention_layer_ids

    @property
    def mamba1_cache_params(self) -> Mamba1CacheParams:
        """Create Mamba1 cache parameters for this config."""
        from sglang.srt.layers.dp_attention import get_attention_tp_size

        intermediate_size = self.mamba_intermediate_size
        shape = Mamba1StateShape.create(
            tp_world_size=get_attention_tp_size(),
            intermediate_size=intermediate_size,
            state_size=self.mamba_d_state,
            conv_kernel=self.mamba_d_conv,
        )

        return Mamba1CacheParams(
            shape=shape,
            layers=self.mamba_layer_ids,
            dtype=mamba_state_dtype(self),
        )

    @property
    def mamba2_cache_params(self) -> Mamba1CacheParams:
        """Alias for mamba1_cache_params (for compatibility with HybridReqToTokenPool).

        Note: Jamba uses Mamba1, not Mamba2. This alias exists because the existing
        memory pool infrastructure uses `mamba2_cache_params` as the property name.
        """
        return self.mamba1_cache_params
