from typing import List, Optional, Union

from transformers import PretrainedConfig

SLIDING_ATTENTION = "sliding_attention"


class FlintConfig(PretrainedConfig):
    model_type = "flint"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        vocab_size: int = 150144,
        hidden_size: int = 2048,
        intermediate_size: int = 6144,
        num_hidden_layers: int = 49,
        num_attention_heads: int = 32,
        num_key_value_heads: int = 4,
        head_dim: int = 128,
        hidden_act: str = "silu",
        max_position_embeddings: int = 262144,
        initializer_range: float = 0.02,
        rms_norm_eps: float = 1e-6,
        use_cache: bool = True,
        tie_word_embeddings: bool = False,
        rope_theta: float = 10000.0,
        rope_parameters: Optional[dict] = None,
        attention_bias: bool = False,
        attention_dropout: float = 0.0,
        use_sliding_window: bool = True,
        sliding_window: Optional[int] = 513,
        layer_types: Optional[List[str]] = None,
        no_rope_layer_indices: Optional[List[int]] = None,
        softmax_type: str = "learnable",
        decoder_sparse_step: int = 1,
        moe_intermediate_size: int = 512,
        shared_expert_intermediate_size: int = 1024,
        num_experts_per_tok: int = 6,
        num_experts: int = 256,
        norm_topk_prob: bool = True,
        router_score_function: str = "sigmoid",
        output_router_logits: bool = False,
        router_aux_loss_coef: float = 0.0,
        mlp_only_layers: Optional[List[int]] = None,
        pad_token_id: Optional[int] = None,
        bos_token_id: Optional[int] = None,
        eos_token_id: Optional[Union[int, List[int]]] = None,
        **kwargs,
    ):
        if layer_types is None:
            layer_types = ["full_attention"] * num_hidden_layers
        if len(layer_types) != num_hidden_layers:
            raise ValueError(
                f"layer_types has {len(layer_types)} entries but "
                f"num_hidden_layers is {num_hidden_layers}"
            )
        if softmax_type != "learnable":
            raise ValueError(
                f"Flint requires softmax_type='learnable', got {softmax_type!r}."
            )
        if router_score_function != "sigmoid":
            raise ValueError(
                f"Flint requires router_score_function='sigmoid', got {router_score_function!r}."
            )
        if decoder_sparse_step != 1:
            raise ValueError(
                f"Flint requires decoder_sparse_step=1, got {decoder_sparse_step}."
            )
        kwargs.setdefault("architectures", ["FlintForCausalLM"])
        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim
        self.hidden_act = hidden_act
        self.max_position_embeddings = max_position_embeddings
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.use_cache = use_cache
        self.rope_parameters = dict(rope_parameters or kwargs.get("rope_scaling") or {})
        self.rope_parameters.setdefault("rope_type", "default")
        self.rope_parameters.setdefault("rope_theta", rope_theta)
        self.rope_theta = self.rope_parameters["rope_theta"]
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout
        self.use_sliding_window = use_sliding_window
        self.sliding_window = sliding_window if use_sliding_window else None
        self.layer_types = list(layer_types)
        self.no_rope_layer_indices = list(no_rope_layer_indices or [])
        self.softmax_type = softmax_type
        self.decoder_sparse_step = decoder_sparse_step
        self.moe_intermediate_size = moe_intermediate_size
        self.shared_expert_intermediate_size = shared_expert_intermediate_size
        self.num_experts_per_tok = num_experts_per_tok
        self.num_experts = num_experts
        self.norm_topk_prob = norm_topk_prob
        self.router_score_function = router_score_function
        self.output_router_logits = output_router_logits
        self.router_aux_loss_coef = router_aux_loss_coef
        self.mlp_only_layers = list(mlp_only_layers or [])
        self.enable_lm_head_fp32 = True
        self.learnable_sink = True
        self.hybrid_layer_pattern = [
            int(layer_type == SLIDING_ATTENTION) for layer_type in self.layer_types
        ]
        self.is_hybrid_swa = self.sliding_window is not None and any(
            self.hybrid_layer_pattern
        )
