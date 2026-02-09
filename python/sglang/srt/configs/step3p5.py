from typing import Any, Optional

from transformers.configuration_utils import PretrainedConfig


class Step3p5Config(PretrainedConfig):
    model_type = "step3p5"
    architectures = ["Step3p5ForCausalLM"]

    def __init__(
        self,
        hidden_size: int = 4096,
        intermediate_size: int = 11264,
        num_attention_heads: int = 64,
        num_attention_groups: int = 8,
        num_hidden_layers: int = 45,
        max_seq_len: int = 128000,
        vocab_size: int = 128815,
        rms_norm_eps: float = 1e-5,
        moe_intermediate_size: int = 1280,
        moe_num_experts: int = 288,
        moe_top_k: int = 8,
        rope_theta: float = 10000,
        rope_scaling: Optional[dict[str, Any]] = None,
        max_position_embeddings: int = 128000,
        share_expert_dims: int = 1280,
        head_dim: int = 128,
        norm_expert_weight: bool = True,
        layer_types: list[str] = None,
        sliding_window: Optional[int] = None,
        moe_layers_enum: tuple[int] = (
            3,
            4,
            5,
            6,
            7,
            8,
            9,
            10,
            11,
            12,
            13,
            14,
            15,
            16,
            17,
            18,
            19,
            20,
            21,
            22,
            23,
            24,
            25,
            26,
            27,
            28,
            29,
            30,
            31,
            32,
            33,
            34,
            35,
            36,
            37,
            38,
            39,
            40,
            41,
            42,
            43,
            44,
        ),
        **kwargs,
    ) -> None:
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_attention_heads = num_attention_heads
        self.num_attention_groups = num_attention_groups
        self.num_hidden_layers = num_hidden_layers
        self.max_seq_len = max_seq_len
        self.vocab_size = vocab_size
        self.rms_norm_eps = rms_norm_eps
        self.moe_intermediate_size = moe_intermediate_size
        self.moe_num_experts = moe_num_experts
        self.moe_top_k = moe_top_k
        self.rope_theta = rope_theta
        self.rope_scaling = rope_scaling
        self.max_position_embeddings = max_position_embeddings
        self.share_expert_dim = share_expert_dims
        self.head_dim = head_dim
        self.norm_expert_weight = norm_expert_weight
        self.moe_layers_enum = moe_layers_enum
        self.layer_types = layer_types
        self.sliding_window = sliding_window
        super().__init__(**kwargs)
