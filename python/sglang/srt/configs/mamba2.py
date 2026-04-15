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
"""Pure Mamba2 model configuration for standalone SSM models (e.g., Codestral Mamba 7B)."""

from typing import Optional

from transformers.configuration_utils import PretrainedConfig
from transformers.utils import logging

from sglang.srt.configs.mamba_utils import (
    Mamba2CacheParams,
    Mamba2StateShape,
    mamba2_state_dtype,
)

logger = logging.get_logger(__name__)


class Mamba2Config(PretrainedConfig):
    """
    Configuration for pure Mamba2 models (no attention layers).

    This config class provides the interface expected by SGLang's hybrid model
    infrastructure (``full_attention_layer_ids``, ``mamba_layer_ids``,
    ``mamba2_cache_params``) for models that are 100% SSM with zero attention.

    Target models: ``mistralai/Mamba-Codestral-7B-v0.1`` and other
    ``Mamba2ForCausalLM`` architectures.
    """

    model_type = "mamba2"

    def __init__(
        self,
        vocab_size: int = 32768,
        hidden_size: int = 4096,
        num_hidden_layers: int = 64,
        num_heads: int = 128,
        head_dim: int = 64,
        state_size: int = 128,
        conv_kernel: int = 4,
        expand: int = 2,
        n_groups: int = 8,
        intermediate_size: Optional[int] = None,
        hidden_act: str = "silu",
        initializer_range: float = 0.1,
        layer_norm_epsilon: float = 1e-5,
        rms_norm: bool = True,
        use_bias: bool = False,
        use_conv_bias: bool = True,
        use_cache: bool = True,
        time_step_rank: int = 256,
        time_step_scale: float = 1.0,
        time_step_min: float = 0.001,
        time_step_max: float = 0.1,
        time_step_floor: float = 0.0001,
        time_step_init_scheme: str = "random",
        time_step_limit=None,
        residual_in_fp32: bool = True,
        rescale_prenorm_residual: bool = False,
        norm_before_gate: bool = True,
        chunk_size: int = 256,
        tie_word_embeddings: bool = False,
        pad_token_id: int = 0,
        bos_token_id: int = 0,
        eos_token_id: int = 0,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.state_size = state_size
        self.conv_kernel = conv_kernel
        self.expand = expand
        self.n_groups = n_groups
        self.intermediate_size = (
            intermediate_size if intermediate_size is not None else hidden_size * expand
        )
        self.hidden_act = hidden_act
        self.initializer_range = initializer_range
        self.layer_norm_epsilon = layer_norm_epsilon
        self.rms_norm = rms_norm
        self.use_bias = use_bias
        self.use_conv_bias = use_conv_bias
        self.use_cache = use_cache
        self.time_step_rank = time_step_rank
        self.time_step_scale = time_step_scale
        self.time_step_min = time_step_min
        self.time_step_max = time_step_max
        self.time_step_floor = time_step_floor
        self.time_step_init_scheme = time_step_init_scheme
        self.time_step_limit = time_step_limit
        self.residual_in_fp32 = residual_in_fp32
        self.rescale_prenorm_residual = rescale_prenorm_residual
        self.norm_before_gate = norm_before_gate
        self.chunk_size = chunk_size

        # Compatibility aliases for ModelConfig.get_num_kv_heads() and
        # other code paths expecting attention-related attributes
        self.num_attention_heads = self.num_heads
        self.num_kv_heads = 0  # Pure SSM has no KV heads

        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )

        # Pure SSM: num_key_value_heads == 0 so that cell_size computes to 0
        self.num_key_value_heads = 0

    @property
    def mamba_chunk_size(self):
        return self.chunk_size

    @property
    def layers_block_type(self):
        return ["mamba"] * self.num_hidden_layers

    @property
    def full_attention_layer_ids(self):
        return []

    @property
    def mamba_layer_ids(self):
        return list(range(self.num_hidden_layers))

    @property
    def mamba2_cache_params(self) -> Mamba2CacheParams:
        from sglang.srt.layers.dp_attention import get_attention_tp_size

        shape = Mamba2StateShape.create(
            tp_world_size=get_attention_tp_size(),
            intermediate_size=self.intermediate_size,
            n_groups=self.n_groups,
            num_heads=self.num_heads,
            head_dim=self.head_dim,
            state_size=self.state_size,
            conv_kernel=self.conv_kernel,
        )
        return Mamba2CacheParams(
            shape=shape,
            layers=self.mamba_layer_ids,
            dtype=mamba2_state_dtype(self),
        )
