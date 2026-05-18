from transformers.configuration_utils import PretrainedConfig
from transformers.utils import logging

from sglang.srt.configs.mamba_utils import (
    Mamba2CacheParams,
    Mamba2StateShape,
    mamba2_state_dtype,
)

logger = logging.get_logger(__name__)

MAMBA = "mamba"
ATTENTION = "attention"


class BambaConfig(PretrainedConfig):
    """
    This is the configuration class to store the configuration of a [`BambaModel`]. It is used to instantiate a
    Bamba model according to the specified arguments, defining the model architecture. Bamba is a hybrid
    architecture combining Mamba-2 SSM layers with full attention layers, jointly developed by IBM, Princeton,
    and UIUC.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        vocab_size (`int`, *optional*, defaults to 128000):
            Vocabulary size of the Bamba model. Defines the number of different tokens that can be represented by the
            `inputs_ids` passed when calling [`BambaModel`].
        tie_word_embeddings (`bool`, *optional*, defaults to `False`):
            Whether the model's input and output word embeddings should be tied.
        hidden_size (`int`, *optional*, defaults to 4096):
            Dimension of the hidden representations.
        intermediate_size (`int`, *optional*, defaults to 14336):
            Dimension of the MLP representations.
        num_hidden_layers (`int`, *optional*, defaults to 32):
            Number of hidden layers in the model.
        num_attention_heads (`int`, *optional*, defaults to 32):
            Number of attention heads for each attention layer.
        num_key_value_heads (`int`, *optional*, defaults to 8):
            Number of key/value heads used to implement Grouped Query Attention. If equal to
            `num_attention_heads` the model uses MHA, if set to 1 it uses MQA, otherwise GQA is used.
        hidden_act (`str` or `function`, *optional*, defaults to `"silu"`):
            The non-linear activation function in the decoder. Only `"silu"` is supported.
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated normal initializer for weight matrices.
        rms_norm_eps (`float`, *optional*, defaults to 1e-05):
            The epsilon used by the RMS normalization layers.
        use_cache (`bool`, *optional*, defaults to `True`):
            Whether or not the model should return the last key/value attentions. Only relevant when
            `config.is_decoder=True`.
        num_logits_to_keep (`int`, *optional*, defaults to 1):
            Number of prompt logits to compute during generation. Set to `None` to compute all logits.
        pad_token_id (`int`, *optional*, defaults to 0):
            The id of the padding token.
        bos_token_id (`int`, *optional*, defaults to 1):
            The id of the beginning-of-sequence token.
        eos_token_id (`int`, *optional*, defaults to 2):
            The id of the end-of-sequence token.
        max_position_embeddings (`int`, *optional*, defaults to 4096):
            Maximum sequence length the model can handle.
        attention_dropout (`float`, *optional*, defaults to 0.0):
            Dropout ratio applied to attention probabilities.
        attn_layer_indices (`list[int]`, *optional*):
            Specifies which layer indices use full attention. All other layers use the Mamba-2 mixer.
        attn_rotary_emb (`int`, *optional*, defaults to 64):
            Number of rotary embedding dimensions applied in attention layers (partial RoPE).
        rope_theta (`float`, *optional*, defaults to 10000.0):
            Base period for the RoPE embeddings. The Bamba-9B config.json omits this
            field so the HF default (``RotaryEmbeddingConfigMixin.default_theta = 10000.0``)
            applies. Do **not** change this to the LLaMA-3 value (500000).
        rope_scaling (`dict`, *optional*):
            Scaling configuration for RoPE. If `None`, no scaling is applied.
        rope_parameters (`dict`, *optional*):
            Full RoPE parameter dict as produced by newer transformers versions. When provided,
            `rope_theta` and `rope_scaling` are read from this dict.
        mamba_expand (`int`, *optional*, defaults to 2):
            Expansion factor relative to `hidden_size` used to derive the Mamba intermediate size.
        mamba_n_heads (`int`, *optional*, defaults to 128):
            Number of heads in the Mamba-2 SSM.
        mamba_d_head (`int` or `"auto"`, *optional*, defaults to `"auto"`):
            Head dimension for the Mamba-2 SSM. When set to `"auto"` it is computed as
            `mamba_expand * hidden_size // mamba_n_heads`.
        mamba_n_groups (`int`, *optional*, defaults to 1):
            Number of groups in the Mamba-2 SSM.
        mamba_d_state (`int`, *optional*, defaults to 128):
            Dimension of the Mamba-2 state space latents.
        mamba_d_conv (`int`, *optional*, defaults to 4):
            Size of the Mamba-2 convolution kernel.
        mamba_chunk_size (`int`, *optional*, defaults to 256):
            Chunk size used during prefill/training for the Mamba-2 scan.
        mamba_conv_bias (`bool`, *optional*, defaults to `True`):
            Whether to use bias in the Mamba-2 convolution layer.
        mamba_proj_bias (`bool`, *optional*, defaults to `False`):
            Whether to use bias in the Mamba-2 input and output projections.
        attention_bias (`bool`, *optional*, defaults to `False`):
            Whether to use bias in the attention projection layers.
        mlp_bias (`bool`, *optional*, defaults to `False`):
            Whether to use bias in the MLP projection layers.
        z_loss_coefficient (`float`, *optional*, defaults to 0.0):
            Coefficient for the auxiliary z-loss used to control logit growth during training.
        use_mamba_kernels (`bool`, *optional*, defaults to `True`):
            Whether to use optimized Mamba kernels during inference.
    """

    model_type = "bamba"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        vocab_size: int = 128000,
        tie_word_embeddings: bool = False,
        hidden_size: int = 4096,
        intermediate_size: int = 14336,
        num_hidden_layers: int = 32,
        num_attention_heads: int = 32,
        num_key_value_heads: int = 8,
        hidden_act: str = "silu",
        initializer_range: float = 0.02,
        rms_norm_eps: float = 1e-5,
        use_cache: bool = True,
        num_logits_to_keep: int = 1,
        pad_token_id: int = 0,
        bos_token_id: int = 1,
        eos_token_id: int = 2,
        max_position_embeddings: int = 4096,
        attention_dropout: float = 0.0,
        # Bamba-specific: which layers are attention vs mamba
        attn_layer_indices=None,
        # Rotary embedding
        attn_rotary_emb: int = 64,
        rope_theta: float = 10000.0,
        rope_scaling=None,
        rope_parameters=None,
        # Mamba-2 parameters
        mamba_expand: int = 2,
        mamba_n_heads: int = 128,
        mamba_d_head: str = "auto",
        mamba_n_groups: int = 1,
        mamba_d_state: int = 128,
        mamba_d_conv: int = 4,
        mamba_chunk_size: int = 256,
        mamba_conv_bias: bool = True,
        mamba_proj_bias: bool = False,
        # Optional / auxiliary
        attention_bias: bool = False,
        mlp_bias: bool = False,
        z_loss_coefficient: float = 0.0,
        mamba_dt_rank: int = None,
        use_mamba_kernels: bool = True,
        time_step_limit=None,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = (
            num_key_value_heads
            if num_key_value_heads is not None
            else num_attention_heads
        )
        self.hidden_act = hidden_act
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.use_cache = use_cache
        self.num_logits_to_keep = num_logits_to_keep
        self.max_position_embeddings = max_position_embeddings
        self.attention_dropout = attention_dropout

        # Attention layer indices
        self.attn_layer_indices = attn_layer_indices or []

        # Rotary embedding
        self.attn_rotary_emb = attn_rotary_emb
        self.rope_theta = rope_theta
        self.rope_scaling = rope_scaling
        self.rope_parameters = rope_parameters

        # Mamba-2 parameters
        self.mamba_expand = mamba_expand
        self.mamba_n_heads = mamba_n_heads
        # Auto-compute head dim if not specified
        mamba_intermediate = mamba_expand * hidden_size
        if mamba_d_head == "auto":
            mamba_d_head = mamba_intermediate // mamba_n_heads
        self.mamba_d_head = mamba_d_head
        self.mamba_n_groups = mamba_n_groups
        self.mamba_d_state = mamba_d_state
        self.mamba_d_conv = mamba_d_conv
        self.mamba_chunk_size = mamba_chunk_size
        self.mamba_conv_bias = mamba_conv_bias
        self.mamba_proj_bias = mamba_proj_bias

        # Auxiliary
        self.attention_bias = attention_bias
        self.mlp_bias = mlp_bias
        self.z_loss_coefficient = z_loss_coefficient
        self.mamba_dt_rank = mamba_dt_rank
        self.use_mamba_kernels = use_mamba_kernels
        self.time_step_limit = (
            tuple(time_step_limit) if time_step_limit is not None else None
        )

        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )

    @property
    def layers_block_type(self):
        """Return a list of 'attention' or 'mamba' for each layer."""
        return [
            (
                ATTENTION
                if (self.attn_layer_indices and i in self.attn_layer_indices)
                else MAMBA
            )
            for i in range(self.num_hidden_layers)
        ]

    @property
    def mamba_layer_ids(self):
        """Indices of Mamba mixer layers."""
        return [
            i
            for i in range(self.num_hidden_layers)
            if self.layers_block_type[i] == MAMBA
        ]

    @property
    def attention_layer_ids(self):
        """Indices of full-attention layers."""
        return [
            i
            for i in range(self.num_hidden_layers)
            if self.layers_block_type[i] == ATTENTION
        ]

    @property
    def full_attention_layer_ids(self):
        return self.attention_layer_ids

    @property
    def linear_layer_ids(self):
        """Alias of mamba_layer_ids (used by cache helpers)."""
        return self.mamba_layer_ids

    @property
    def mamba2_cache_params(self) -> Mamba2CacheParams:
        from sglang.srt.layers.dp_attention import get_attention_tp_size

        intermediate_size = self.mamba_expand * self.hidden_size
        shape = Mamba2StateShape.create(
            tp_world_size=get_attention_tp_size(),
            intermediate_size=intermediate_size,
            n_groups=self.mamba_n_groups,
            num_heads=self.mamba_n_heads,
            head_dim=self.mamba_d_head,
            state_size=self.mamba_d_state,
            conv_kernel=self.mamba_d_conv,
        )
        return Mamba2CacheParams(
            shape=shape,
            layers=self.mamba_layer_ids,
            dtype=mamba2_state_dtype(self),
        )
