from typing import List, Optional, Union

from transformers import PretrainedConfig

SLIDING_ATTENTION = "sliding_attention"
FULL_ATTENTION = "full_attention"


class RicursiveConfig(PretrainedConfig):
    model_type = "ricursive"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        vocab_size: int = 100352,
        hidden_size: int = 5120,
        intermediate_size: int = 40960,
        num_hidden_layers: int = 80,
        num_attention_heads: int = 80,
        num_key_value_heads: int = 8,
        head_dim: int = 128,
        hidden_act: str = "silu",
        max_position_embeddings: int = 2097152,
        rms_norm_eps: float = 1e-6,
        rope_theta: float = 10000.0,
        sliding_window: int = 1024,
        layer_types: Optional[List[str]] = None,
        use_gated_attention: bool = True,
        gate_full_precision: bool = True,
        use_valres: bool = True,
        ffn_clamp_threshold: Optional[float] = 20.0,
        embed_scale: Optional[float] = None,
        tie_word_embeddings: bool = False,
        use_cache: bool = True,
        pad_token_id: Optional[int] = None,
        bos_token_id: Optional[int] = None,
        eos_token_id: Optional[Union[int, List[int]]] = None,
        **kwargs,
    ):
        if layer_types is None:
            layer_types = [
                FULL_ATTENTION if i % 5 == 4 else SLIDING_ATTENTION
                for i in range(num_hidden_layers)
            ]
        if len(layer_types) != num_hidden_layers:
            raise ValueError(
                f"layer_types has {len(layer_types)} entries but "
                f"num_hidden_layers is {num_hidden_layers}"
            )
        if not gate_full_precision and use_gated_attention:
            raise ValueError("Ricursive computes its attention gate in fp32.")
        # The checkpoint repeats these next to layer_types. Drop them so the KV pool
        # split comes from layer_types and is checked against the built attention.
        kwargs.pop("is_hybrid_swa", None)
        kwargs.pop("hybrid_layer_pattern", None)
        kwargs.setdefault("architectures", ["RicursiveForCausalLM"])
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
        self.rms_norm_eps = rms_norm_eps
        self.rope_theta = rope_theta
        self.sliding_window = sliding_window
        self.layer_types = list(layer_types)
        self.use_gated_attention = use_gated_attention
        self.gate_full_precision = gate_full_precision
        self.use_valres = use_valres
        self.ffn_clamp_threshold = ffn_clamp_threshold
        self.embed_scale = embed_scale
        self.use_cache = use_cache
