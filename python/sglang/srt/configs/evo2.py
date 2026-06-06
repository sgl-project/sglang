# SPDX-License-Identifier: Apache-2.0
"""Evo2 model configuration for SGLang.

Evo2 (StripedHyena 2) is a hybrid DNA foundation model that interleaves
standard attention layers with Hyena convolution operators (HCL, HCM, HCS).
"""

from transformers.configuration_utils import PretrainedConfig
from transformers.utils import logging

logger = logging.get_logger(__name__)


class Evo2Config(PretrainedConfig):
    """Configuration for the Evo2 StripedHyena 2 model.

    Supports the following variants:
    - evo2-1b-8k
    - evo2-7b-8k / evo2-7b-1m
    - evo2-40b-8k / evo2-40b-1m
    """

    model_type = "evo2"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        vocab_size: int = 512,
        hidden_size: int = 4096,
        num_filters: int = 4096,
        num_layers: int = 32,
        num_attention_heads: int = 32,
        num_key_value_heads: int = 32,
        intermediate_size: int = 11008,
        hidden_act: str = "gelu",
        rms_norm_eps: float = 1e-6,
        initializer_range: float = 0.02,
        max_position_embeddings: int = 32768,
        tie_word_embeddings: bool = True,
        use_cache: bool = True,
        pad_token_id: int = 0,
        bos_token_id: int = 1,
        eos_token_id: int = 2,
        # Hyena-specific parameters
        state_size: int = 16,
        short_filter_length: int = 3,
        hcm_filter_length: int = 128,
        hcs_filter_length: int = 7,
        hcl_filter_groups: int = 4096,
        hcm_filter_groups: int = 256,
        hcs_filter_groups: int = 256,
        # Layer indices
        attn_layer_idxs: list = None,
        hcl_layer_idxs: list = None,
        hcm_layer_idxs: list = None,
        hcs_layer_idxs: list = None,
        # RoPE
        rotary_emb_base: float = 10000.0,
        rotary_emb_scaling: float = None,
        # Evo2-specific
        evo2_style_activations: bool = True,
        mlp_activation: str = "gelu",
        interleave: bool = True,
        column_split: bool = True,
        column_split_hyena: bool = False,
        # Bias flags
        mha_out_proj_bias: bool = True,
        hyena_out_proj_bias: bool = True,
        qkv_proj_bias: bool = False,
        short_filter_bias: bool = False,
        # Tokenizer
        tokenizer_type: str = "CharLevelTokenizer",
        make_vocab_size_divisible_by: int = 8,
        inner_size_multiple_of: int = 16,
        # FP8
        use_fp8_input_projections: bool = False,
        # Flash attention / FFT
        use_flash_attn: bool = True,
        use_flashfft: bool = False,
        **kwargs,
    ):
        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )

        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_filters = num_filters
        self.num_layers = num_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act
        self.rms_norm_eps = rms_norm_eps
        self.initializer_range = initializer_range
        self.max_position_embeddings = max_position_embeddings
        self.use_cache = use_cache

        # Hyena parameters
        self.state_size = state_size
        self.short_filter_length = short_filter_length
        self.hcm_filter_length = hcm_filter_length
        self.hcs_filter_length = hcs_filter_length
        self.hcl_filter_groups = hcl_filter_groups
        self.hcm_filter_groups = hcm_filter_groups
        self.hcs_filter_groups = hcs_filter_groups

        # Default layer indices for evo2-7b-8k (32 layers)
        if attn_layer_idxs is None:
            attn_layer_idxs = [3, 10, 17, 24, 31]
        if hcl_layer_idxs is None:
            hcl_layer_idxs = [2, 6, 9, 13, 16, 20, 23, 27, 30]
        if hcm_layer_idxs is None:
            hcm_layer_idxs = [1, 5, 8, 12, 15, 19, 22, 26, 29]
        if hcs_layer_idxs is None:
            hcs_layer_idxs = [0, 4, 7, 11, 14, 18, 21, 25, 28]

        self.attn_layer_idxs = attn_layer_idxs
        self.hcl_layer_idxs = hcl_layer_idxs
        self.hcm_layer_idxs = hcm_layer_idxs
        self.hcs_layer_idxs = hcs_layer_idxs

        # RoPE
        self.rotary_emb_base = rotary_emb_base
        self.rotary_emb_scaling = rotary_emb_scaling

        # Evo2 flags
        self.evo2_style_activations = evo2_style_activations
        self.mlp_activation = mlp_activation
        self.interleave = interleave
        self.column_split = column_split
        self.column_split_hyena = column_split_hyena

        # Bias flags
        self.mha_out_proj_bias = mha_out_proj_bias
        self.hyena_out_proj_bias = hyena_out_proj_bias
        self.qkv_proj_bias = qkv_proj_bias
        self.short_filter_bias = short_filter_bias

        # Other
        self.tokenizer_type = tokenizer_type
        self.make_vocab_size_divisible_by = make_vocab_size_divisible_by
        self.inner_size_multiple_of = inner_size_multiple_of
        self.use_fp8_input_projections = use_fp8_input_projections
        self.use_flash_attn = use_flash_attn
        self.use_flashfft = use_flashfft

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_attention_heads

    @property
    def num_heads(self) -> int:
        return self.num_attention_heads

    @property
    def num_kv_heads(self) -> int:
        return self.num_key_value_heads
