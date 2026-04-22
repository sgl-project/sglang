from typing import List, Optional

from transformers import PretrainedConfig


class AfmoeConfig(PretrainedConfig):
    model_type = "afmoe"

    def __init__(
        self,
        vocab_size: int = 32000,
        hidden_size: int = 4096,
        intermediate_size: int = 11008,
        moe_intermediate_size: int = 256,
        num_hidden_layers: int = 32,
        num_attention_heads: int = 32,
        num_key_value_heads: Optional[int] = None,
        head_dim: Optional[int] = None,
        hidden_act: str = "silu",
        max_position_embeddings: int = 131072,
        initializer_range: float = 0.02,
        rms_norm_eps: float = 1e-5,
        use_cache: bool = True,
        pad_token_id: Optional[int] = None,
        bos_token_id: int = 1,
        eos_token_id: int = 2,
        tie_word_embeddings: bool = False,
        rope_theta: float = 10000.0,
        rope_scaling: Optional[dict] = None,
        attention_bias: bool = False,
        attention_dropout: float = 0.0,
        # MoE parameters
        num_experts: Optional[int] = None,
        num_experts_per_tok: Optional[int] = None,
        num_shared_experts: int = 0,
        num_dense_layers: int = 0,
        # Routing parameters
        score_func: str = "sigmoid",
        route_norm: bool = True,
        route_scale: float = 1.0,
        n_group: int = 1,
        topk_group: int = 1,
        # Attention parameters
        sliding_window: Optional[int] = None,
        layer_types: Optional[List[str]] = None,
        global_attn_every_n_layers: int = 4,
        # muP scaling
        mup_enabled: bool = False,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.moe_intermediate_size = moe_intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads

        if num_key_value_heads is None:
            num_key_value_heads = num_attention_heads

        self.num_key_value_heads = num_key_value_heads
        self.head_dim = (
            head_dim if head_dim is not None else hidden_size // num_attention_heads
        )
        self.hidden_act = hidden_act
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.use_cache = use_cache
        self.rope_theta = rope_theta
        self.rope_scaling = rope_scaling
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout

        # MoE parameters
        self.num_experts = num_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.num_shared_experts = num_shared_experts
        self.num_dense_layers = num_dense_layers

        # Routing parameters
        self.score_func = score_func
        self.route_norm = route_norm
        self.route_scale = route_scale
        self.n_group = n_group
        self.topk_group = topk_group

        # Attention parameters
        self.sliding_window = sliding_window
        self.layer_types = layer_types
        self.global_attn_every_n_layers = global_attn_every_n_layers

        # muP scaling
        self.mup_enabled = mup_enabled

        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )
