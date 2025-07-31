from typing import Optional

from transformers.configuration_utils import PretrainedConfig


class StepVisionEncoderConfig(PretrainedConfig):
    model_type = "step_vision_encoder"

    def __init__(
        self,
        hidden_size=1152,
        intermediate_size=4304,
        num_hidden_layers=27,
        num_attention_heads=16,
        image_size=560,
        patch_size=14,
        hidden_act="quick_gelu",
        num_channels=3,
        qkv_bias=True,
        drop_path_rate=0.0,
        attention_dropout=0.0,
        rope_theta=10000.0,
        output_hidden_size=2304,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.image_size = image_size
        self.patch_size = patch_size
        self.hidden_act = hidden_act
        self.num_channels = num_channels
        self.qkv_bias = qkv_bias
        self.drop_path_rate = drop_path_rate
        self.attention_dropout = attention_dropout
        self.rope_theta = rope_theta
        self.output_hidden_size = output_hidden_size


class Step3VTextConfig(PretrainedConfig):
    model_type = "step3v_text"

    def __init__(
        self,
        vocab_size=135200,
        hidden_size=5120,
        intermediate_size=13824,
        num_hidden_layers=60,
        num_attention_heads=40,
        num_key_value_heads=1,
        hidden_act="silu",
        max_position_embeddings=65536,
        rms_norm_eps=1e-5,
        rope_theta=10000.0,
        rope_scaling=None,
        tie_word_embeddings=False,
        share_q_dim=3072,
        head_dim=128,
        pad_token_id=151649,
        bos_token_id=1,
        eos_token_id=2,
        # MoE parameters
        moe_layers_enum=None,
        moe_num_experts=16,
        moe_top_k=2,
        moe_intermediate_size=8704,
        share_expert_dim=8704,
        norm_expert_weight=False,
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
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.hidden_act = hidden_act
        self.max_position_embeddings = max_position_embeddings
        self.rms_norm_eps = rms_norm_eps
        self.rope_theta = rope_theta
        self.rope_scaling = rope_scaling
        self.share_q_dim = share_q_dim
        self.head_dim = head_dim
        # MoE parameters
        self.moe_layers_enum = moe_layers_enum
        self.moe_num_experts = moe_num_experts
        self.moe_top_k = moe_top_k
        self.moe_intermediate_size = moe_intermediate_size
        self.share_expert_dim = share_expert_dim
        self.norm_expert_weight = norm_expert_weight


class Step3VConfig(PretrainedConfig):
    model_type = "step3v"
    is_composition = True

    def __init__(
        self,
        vision_config=None,
        text_config=None,
        projector_bias=True,
        understand_projector_stride=3,
        vocab_size=135200,
        hidden_size=5120,
        intermediate_size=13824,
        num_hidden_layers=60,
        num_attention_heads=40,
        num_key_value_heads=1,
        hidden_act="silu",
        max_position_embeddings=65536,
        rms_norm_eps=1e-5,
        rope_theta=10000.0,
        rope_scaling=None,
        tie_word_embeddings=False,
        share_q_dim=3072,
        head_dim=128,
        pad_token_id=151649,
        bos_token_id=1,
        eos_token_id=2,
        # MoE parameters
        moe_layers_enum=None,
        moe_num_experts=16,
        moe_top_k=2,
        moe_intermediate_size=8704,
        share_expert_dim=8704,
        norm_expert_weight=False,
        **kwargs,
    ):
        if vision_config is None:
            vision_config = StepVisionEncoderConfig()
        elif isinstance(vision_config, dict):
            vision_config = StepVisionEncoderConfig(**vision_config)

        if text_config is None:
            text_config = Step3VTextConfig(
                vocab_size=vocab_size,
                hidden_size=hidden_size,
                intermediate_size=intermediate_size,
                num_hidden_layers=num_hidden_layers,
                num_attention_heads=num_attention_heads,
                num_key_value_heads=num_key_value_heads,
                hidden_act=hidden_act,
                max_position_embeddings=max_position_embeddings,
                rms_norm_eps=rms_norm_eps,
                rope_theta=rope_theta,
                rope_scaling=rope_scaling,
                tie_word_embeddings=tie_word_embeddings,
                share_q_dim=share_q_dim,
                head_dim=head_dim,
                pad_token_id=pad_token_id,
                bos_token_id=bos_token_id,
                eos_token_id=eos_token_id,
                moe_layers_enum=moe_layers_enum,
                moe_num_experts=moe_num_experts,
                moe_top_k=moe_top_k,
                moe_intermediate_size=moe_intermediate_size,
                share_expert_dim=share_expert_dim,
                norm_expert_weight=norm_expert_weight,
            )
        elif isinstance(text_config, dict):
            text_config = Step3VTextConfig(**text_config)

        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )
        
        self.vision_config = vision_config
        self.text_config = text_config
        self.projector_bias = projector_bias
        self.understand_projector_stride = understand_projector_stride
        
        # Copy text config attributes to top level for compatibility
        self.vocab_size = text_config.vocab_size
        self.hidden_size = text_config.hidden_size
        self.intermediate_size = text_config.intermediate_size
        self.num_hidden_layers = text_config.num_hidden_layers
        self.num_attention_heads = text_config.num_attention_heads
        self.num_key_value_heads = text_config.num_key_value_heads
        self.hidden_act = text_config.hidden_act
        self.max_position_embeddings = text_config.max_position_embeddings
        self.rms_norm_eps = text_config.rms_norm_eps
        self.rope_theta = text_config.rope_theta
        self.rope_scaling = text_config.rope_scaling
        self.share_q_dim = text_config.share_q_dim
        self.head_dim = text_config.head_dim
        # MoE parameters
        self.moe_layers_enum = text_config.moe_layers_enum
        self.moe_num_experts = text_config.moe_num_experts
        self.moe_top_k = text_config.moe_top_k
        self.moe_intermediate_size = text_config.moe_intermediate_size
        self.share_expert_dim = text_config.share_expert_dim
        self.norm_expert_weight = text_config.norm_expert_weight