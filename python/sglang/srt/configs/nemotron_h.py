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
# Adapted from https://github.com/vllm-project/vllm/blob/main/vllm/transformers_utils/configs/nemotron_h.py

"""NemotronH model configuration"""

from transformers.configuration_utils import PretrainedConfig
from transformers.utils import logging

from sglang.srt.configs.mamba_utils import (
    Mamba2CacheParams,
    Mamba2StateShape,
    mamba2_state_dtype,
)

logger = logging.get_logger(__name__)

MAMBA = "M"
ATTENTION = "*"
MLP = "-"
MOE = "E"
DEFAULT_LAYERS_BLOCK_TYPE = ["mamba", "moe", "attention", "moe"]
DEFAULT_MTP_LAYERS_BLOCK_TYPE = ["attention", "moe"]


class NemotronHConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a
    [`NemotronHModel`]. It is used to instantiate a NemotronH model according
    to the specified arguments, defining the model architecture. Instantiating
    a configuration with the defaults will yield a similar configuration to
    that of the NemotronH-v0.1 model.
    Args:
        vocab_size (`int`, *optional*, defaults to 131072):
            Vocabulary size of the NemotronH model. Defines the number of
            different tokens that can be represented by the `inputs_ids`
            passed when calling [`NemotronHModel`]
        tie_word_embeddings (`bool`, *optional*, defaults to `False`):
            Whether the model's input and output word embeddings should be
            tied. Note that this is only relevant if the model has an output
            word embedding layer.
        hidden_size (`int`, *optional*, defaults to 4096):
            Dimension of the hidden representations.
        intermediate_size (`int`, *optional*, defaults to 21504):
            Dimension of the MLP representations.
        num_hidden_layers (`int`, *optional*):
            Deprecated. Kept only for backward compatibility. The effective
            layer count is derived from `layers_block_type`.
        hybrid_override_pattern (`str`, *optional*, defaults to
            `"M-M-M-M*-M-M-M-M-M*-M-M-M-M-M*-M-M-M-M-M*-M-M-M-M-M-"`):
            Deprecated compatibility field. Pattern string where each
            character represents Mamba2 (`M`), Attention (`*`), MLP (`-`),
            or MoE (`E`).
        layers_block_type (`list[str]`, *optional*):
            Canonical layer layout. Each entry is one of:
            `"mamba"`, `"attention"`, `"mlp"`, `"moe"`.
        num_attention_heads (`int`, *optional*, defaults to 32):
            Number of attention heads for each attention layer in the
            Transformer encoder.
        attention_head_dim (`int`, *optional*, defaults to 128):
            Dimension of each attention head.
        num_key_value_heads (`int`, *optional*, defaults to 8):
            This is the number of key_value heads that should be used to
            implement Grouped Query Attention. If
            `num_key_value_heads=num_attention_heads`, the model will use
            Multi Head Attention (MHA), if `num_key_value_heads=1` the model
            will use Multi Query Attention (MQA) otherwise GQA is used.
        mlp_hidden_act (`str`, *optional*, defaults to "relu2"):
            The non-linear activation function in the MLP layers.
        attention_bias (`bool`, *optional*, defaults to `False`):
            Whether to use bias in attention layers.
        mlp_bias (`bool`, *optional*, defaults to `False`):
            Whether to use bias in MLP layers.
        use_bias (`bool`, *optional*, defaults to `False`):
            Whether to use bias in the model.
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for
            initializing all weight matrices.
        layer_norm_epsilon (`float`, *optional*, defaults to 1e-5):
            The epsilon used by the layer normalization layers.
        residual_in_fp32 (`bool`, *optional*, defaults to `False`):
            Whether or not residuals should be in `float32`. If set to `False`
            residuals will keep the same `dtype` as the rest of the model.
        use_cache (`bool`, *optional*, defaults to `True`):
            Whether or not the model should return the last key/values
            attentions (not used by all models). Only relevant if
            `config.is_decoder=True`.
        num_logits_to_keep (`int` or `None`, *optional*, defaults to 1):
            Number of prompt logits to calculate during generation. If `None`,
            all logits will be calculated. If an integer value, only last
            `num_logits_to_keep` logits will be calculated.
        pad_token_id (`int`, *optional*, defaults to 0):
            The id of the padding token.
        bos_token_id (`int`, *optional*, defaults to 1):
            The id of the "beginning-of-sequence" token.
        eos_token_id (`int`, *optional*, defaults to 2):
            The id of the "end-of-sequence" token.
        sliding_window (`int`, *optional*, defaults to None):
            Sliding window attention window size.
        max_position_embeddings (`int`, *optional*, defaults to 4096):
            The maximum sequence length that this model might ever be used
            with.
        attention_dropout (`float`, *optional*, defaults to 0.0):
            The dropout ratio for the attention probabilities.
        hidden_dropout (`float`, *optional*, defaults to 0.0):
            The dropout ratio for the hidden states.
        use_mamba_kernels (`bool`, *optional*, defaults to `True`):
            Flag indicating whether or not to use the fast mamba kernels.
            These are available only if `mamba-ssm` and `causal-conv1d`
            are installed, and the mamba modules are running on a CUDA device.
        ssm_state_size (`int`, *optional*, defaults to 128):
            The dimension of the mamba state space latents.
        mamba_num_heads (`int`, *optional*, defaults to 128):
            Number of heads in Mamba layers.
        mamba_n_groups (`int`, *optional*, defaults to 8):
            Number of groups in Mamba layers.
        mamba_head_dim (`int`, *optional*, defaults to 64):
            Dimension of each Mamba head.
        mamba_d_conv (`int`, *optional*, defaults to 4):
            The size of the mamba convolution kernel.
        mamba_expand (`int`, *optional*, defaults to 2):
            Expanding factor used to determine the mamba intermediate size.
        mamba_hidden_act (`str`, *optional*, defaults to "silu"):
            The non-linear activation function in the Mamba layers.
        mamba_dt_min (`float`, *optional*, defaults to 0.001):
            Minimum value for the time step in Mamba.
        mamba_dt_max (`float`, *optional*, defaults to 0.1):
            Maximum value for the time step in Mamba.
        mamba_dt_limit (`tuple`, *optional*, defaults to (0.0, float("inf"))):
            Limits for the time step in Mamba.
        mamba_dt_init_floor (`float`, *optional*, defaults to 1e-4):
            Floor value for time step initialization in Mamba.
        mamba_conv_bias (`bool`, *optional*, defaults to `True`):
            Whether to use bias in the convolution layer of the mamba mixer
            block.
        mamba_proj_bias (`bool`, *optional*, defaults to `False`):
            Whether to use bias in the input and output projections of the
            mamba mixer block.
        mamba_chunk_size (`int`, *optional*, defaults to 256):
            Size of chunks for Mamba processing.
        rescale_prenorm_residual (`bool`, *optional*, defaults to `True`):
            Whether to rescale the pre-normalization residual connections.
    """

    model_type = "nemotron_h"
    keys_to_ignore_at_inference = ["past_key_values"]

    @staticmethod
    def _validate_layers_block_type(
        layers_block_type, expected_length=None, param_name="layers_block_type"
    ):
        """
        Validate layers_block_type list.
        Args:
            layers_block_type: List of layer types to validate.
            expected_length: If provided, validate the list has this length.
            param_name: Parameter name for error messages.
        Raises:
            ValueError: If validation fails.
        """
        if not isinstance(layers_block_type, list):
            raise ValueError(
                f"{param_name} must be a list of strings. Got type: {type(layers_block_type)}"
            )
        if expected_length is not None and len(layers_block_type) != expected_length:
            raise ValueError(
                f"{param_name} must have length {expected_length}. Got length {len(layers_block_type)}."
            )
        valid_types = {"mamba", "attention", "mlp", "moe"}
        if not all(block_type in valid_types for block_type in layers_block_type):
            invalid = set(layers_block_type) - valid_types
            raise ValueError(
                f"{param_name} contains invalid types: {invalid}. Must be one of: {valid_types}"
            )

    @staticmethod
    def _resolve_layers_block_type(
        layers_block_type, hybrid_override_pattern, kwargs
    ) -> list[str]:
        """Resolve canonical layers_block_type from new and legacy config fields."""
        # Prefer explicit kwargs override first (legacy HF path), otherwise use
        # the function argument value from config fields.
        pattern = kwargs.pop("hybrid_override_pattern", hybrid_override_pattern)
        if layers_block_type is None:
            if pattern is not None:
                layers_block_type = NemotronHConfig._pattern_to_list(pattern)
            else:
                # Last-resort fallback to preserve compatibility when neither
                # canonical nor legacy pattern fields are provided.
                layers_block_type = DEFAULT_LAYERS_BLOCK_TYPE
        return layers_block_type

    @staticmethod
    def _resolve_mtp_layers_block_type(mtp_layers_block_type, kwargs) -> list[str]:
        """Resolve canonical mtp_layers_block_type from new and legacy config fields."""
        if "mtp_hybrid_override_pattern" in kwargs:
            pattern = kwargs.pop("mtp_hybrid_override_pattern")
            if mtp_layers_block_type is None or mtp_layers_block_type == [
                "attention",
                "moe",
            ]:
                mtp_layers_block_type = NemotronHConfig._pattern_to_list(pattern)
        return mtp_layers_block_type

    def __init__(
        self,
        vocab_size=131072,
        tie_word_embeddings=False,
        hidden_size=4096,
        intermediate_size=21504,
        num_hidden_layers=None,  # Deprecated, only for backward compatibility
        hybrid_override_pattern="M-M-M-M*-M-M-M-M-M*-M-M-M-M-M*-M-M-M-M-M*-M-M-M-M-M-",
        layers_block_type=None,
        num_attention_heads=32,
        head_dim=128,
        num_key_value_heads=8,  # nemo: num_query_groups
        mlp_hidden_act="relu2",
        attention_bias=False,
        mlp_bias=False,
        use_bias=False,
        initializer_range=0.02,  # nemo: init_method_std
        layer_norm_epsilon=1e-5,  # nemo: layernorm_epsilon
        residual_in_fp32=False,  #  Megatron Core default value
        use_cache=True,
        num_logits_to_keep=1,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
        sliding_window=None,
        max_position_embeddings=4096,
        attention_dropout=0.0,
        hidden_dropout=0.0,  # * ADDED
        use_mamba_kernels=True,
        ssm_state_size=128,  # mamba_state_size
        mamba_num_heads=128,
        mamba_n_groups=8,  # nemo: mamba_ssm_ngroups = num_heads
        mamba_head_dim=64,
        mamba_d_conv=4,
        mamba_expand=2,
        mamba_hidden_act="silu",
        mamba_dt_min=0.001,
        mamba_dt_max=0.1,
        mamba_dt_limit=(0.0, float("inf")),
        mamba_dt_init_floor=1e-4,
        mamba_conv_bias=True,
        mamba_proj_bias=False,
        mamba_chunk_size=256,
        rescale_prenorm_residual=True,
        n_routed_experts=8,
        n_shared_experts=1,
        moe_intermediate_size=7688,
        moe_shared_expert_intermediate_size=7688,
        moe_latent_size=None,
        num_experts_per_tok=2,
        routed_scaling_factor=1.0,
        n_group=1,
        topk_group=1,
        norm_topk_prob=True,
        num_nextn_predict_layers=0,
        mtp_layers_block_type=DEFAULT_MTP_LAYERS_BLOCK_TYPE,
        **kwargs,
    ):
        # Compatibility parsing: normalize legacy pattern fields into canonical list fields.
        layers_block_type = self._resolve_layers_block_type(
            layers_block_type, hybrid_override_pattern, kwargs
        )
        mtp_layers_block_type = self._resolve_mtp_layers_block_type(
            mtp_layers_block_type, kwargs
        )

        # num_hidden_layers is deprecated and ignored as a source of truth.
        if (
            num_hidden_layers is not None
            and len(layers_block_type) != num_hidden_layers
        ):
            logger.warning(
                f"num_hidden_layers ({num_hidden_layers}) is deprecated and doesn't match "
                f"layers_block_type length ({len(layers_block_type)}). Using layers_block_type length."
            )

        # Core model attributes.
        self.vocab_size = vocab_size
        self.tie_word_embeddings = tie_word_embeddings
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_attention_heads = num_attention_heads
        self.head_dim = head_dim
        self.sliding_window = sliding_window
        self.max_position_embeddings = max_position_embeddings
        self.attention_dropout = attention_dropout
        self.hidden_dropout = hidden_dropout

        self._validate_layers_block_type(
            layers_block_type, expected_length=None, param_name="layers_block_type"
        )
        self.layers_block_type = layers_block_type

        # for backward compatibility
        if num_key_value_heads is None:
            num_key_value_heads = num_attention_heads

        self.num_key_value_heads = num_key_value_heads
        self.mlp_hidden_act = mlp_hidden_act
        self.attention_bias = attention_bias
        self.mlp_bias = mlp_bias
        self.use_bias = use_bias
        self.initializer_range = initializer_range
        self.layer_norm_epsilon = layer_norm_epsilon
        self.residual_in_fp32 = residual_in_fp32

        self.use_cache = use_cache
        self.num_logits_to_keep = num_logits_to_keep

        # Mamba attributes.
        self.use_mamba_kernels = use_mamba_kernels
        self.mamba_n_groups = mamba_n_groups
        self.mamba_head_dim = mamba_head_dim
        self.ssm_state_size = ssm_state_size
        self.mamba_num_heads = mamba_num_heads
        self.conv_kernel = mamba_d_conv
        self.expand = mamba_expand
        self.mamba_hidden_act = mamba_hidden_act
        self.time_step_min = mamba_dt_min
        self.time_step_max = mamba_dt_max
        self.time_step_limit = mamba_dt_limit
        self.time_step_floor = mamba_dt_init_floor
        self.use_conv_bias = mamba_conv_bias
        self.mamba_proj_bias = mamba_proj_bias
        self.mamba_chunk_size = mamba_chunk_size
        self.rescale_prenorm_residual = rescale_prenorm_residual
        # MoE attributes.
        self.n_routed_experts = n_routed_experts
        self.n_shared_experts = n_shared_experts
        self.moe_intermediate_size = moe_intermediate_size
        self.moe_shared_expert_intermediate_size = moe_shared_expert_intermediate_size
        self.moe_latent_size = moe_latent_size
        self.num_experts_per_tok = num_experts_per_tok
        self.routed_scaling_factor = routed_scaling_factor
        self.n_group = n_group
        self.topk_group = topk_group
        self.norm_topk_prob = norm_topk_prob
        # MTP attributes.
        self.num_nextn_predict_layers = num_nextn_predict_layers

        if self.num_nextn_predict_layers > 0:
            if mtp_layers_block_type is None:
                raise ValueError(
                    "mtp_layers_block_type is required when num_nextn_predict_layers > 0. "
                    "Please provide an explicit list of layer types for MTP layers. "
                    "Example: mtp_layers_block_type=['attention', 'moe']"
                )
            self._validate_layers_block_type(
                mtp_layers_block_type, None, "mtp_layers_block_type"
            )
        self.mtp_layers_block_type = mtp_layers_block_type

        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )

    @property
    def mamba_layer_ids(self):
        return [
            i
            for i in range(self.num_hidden_layers)
            if self.hybrid_override_pattern[i] == MAMBA
        ]

    @property
    def full_attention_layer_ids(self):
        return [
            i
            for i in range(self.num_hidden_layers)
            if self.hybrid_override_pattern[i] == ATTENTION
        ]

    @property
    def mamba2_cache_params(self) -> Mamba2CacheParams:
        from sglang.srt.layers.dp_attention import get_attention_tp_size

        shape = Mamba2StateShape.create(
            tp_world_size=get_attention_tp_size(),
            intermediate_size=self.mamba_num_heads * self.mamba_head_dim,
            n_groups=self.n_groups,
            num_heads=self.mamba_num_heads,
            head_dim=self.mamba_head_dim,
            state_size=self.ssm_state_size,
            conv_kernel=self.conv_kernel,
        )

        return Mamba2CacheParams(
            shape=shape, layers=self.mamba_layer_ids, dtype=mamba2_state_dtype(self)
        )

    @property
    def num_hidden_layers(self) -> int:
        """
        Number of hidden layers derived from the length of layers_block_type.
        This property replaces the deprecated num_hidden_layers parameter.
        """
        return len(self.layers_block_type)

    @num_hidden_layers.setter
    def num_hidden_layers(self, value):
        """
        Setter for backward compatibility when loading configs.
        The value is ignored since num_hidden_layers is computed from layers_block_type.
        """
        pass

    @property
    def hybrid_override_pattern(self) -> str:
        """
        Backward compatibility property.
        Returns the pattern string representation of layers_block_type.
        """
        return self._list_to_pattern(self.layers_block_type)

    @hybrid_override_pattern.setter
    def hybrid_override_pattern(self, value):
        """
        Setter for backward compatibility when loading configs.
        """
        self.layers_block_type = self._pattern_to_list(value)

    @property
    def mtp_hybrid_override_pattern(self) -> str:
        """
        Backward compatibility property.
        Returns the pattern string representation of mtp_layers_block_type.
        """
        return self._list_to_pattern(self.mtp_layers_block_type)

    @mtp_hybrid_override_pattern.setter
    def mtp_hybrid_override_pattern(self, value):
        """Setter for backward compatibility when loading configs."""
        self.mtp_layers_block_type = self._pattern_to_list(value)

    @staticmethod
    def _list_to_pattern(layers_list: list[str]) -> str:
        """Convert list of layer types back to pattern string (for backward compatibility)."""
        reverse_mapping = {
            "mamba": MAMBA,
            "moe": MOE,
            "attention": ATTENTION,
            "mlp": MLP,
        }
        return "".join(reverse_mapping[layer_type] for layer_type in layers_list)

    @staticmethod
    def _pattern_to_list(pattern: str) -> list[str]:
        """Convert pattern string to list of layer types (for backward compatibility)."""
        if any(char not in {MAMBA, MOE, ATTENTION, MLP} for char in pattern):
            raise ValueError(
                "Pattern must only contain characters 'M', '*', '-' or 'E'. "
                f"Got: {pattern}"
            )
        pattern_mapping = {
            MAMBA: "mamba",
            MOE: "moe",
            ATTENTION: "attention",
            MLP: "mlp",
        }
        return [pattern_mapping[char] for char in pattern]
