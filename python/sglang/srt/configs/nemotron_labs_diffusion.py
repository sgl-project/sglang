# Copyright 2026 SGLang Team
# Copyright contributors to the vLLM project
# Licensed under the Apache License, Version 2.0.
"""Configuration for Nemotron Labs Diffusion checkpoints."""

from transformers import PretrainedConfig


class NemotronLabsDiffusionConfig(PretrainedConfig):
    model_type = "nemotron_labs_diffusion"

    def __init__(
        self,
        vocab_size=131072,
        hidden_size=4096,
        intermediate_size=14336,
        num_hidden_layers=34,
        num_attention_heads=32,
        num_key_value_heads=8,
        head_dim=128,
        hidden_act="silu",
        max_position_embeddings=262144,
        rms_norm_eps=1e-5,
        rope_theta=1000000.0,
        rope_parameters=None,
        attention_bias=False,
        mlp_bias=False,
        sliding_window=None,
        mask_token_id=100,
        block_size=32,
        **kwargs,
    ):
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
        self.attention_bias = attention_bias
        self.mlp_bias = mlp_bias
        self.sliding_window = sliding_window
        self.mask_token_id = mask_token_id
        self.block_size = block_size
        rope_scaling = kwargs.pop("rope_scaling", None)
        self.rope_parameters = dict(rope_parameters or rope_scaling or {})
        self.rope_parameters.setdefault("rope_theta", rope_theta)
        self.rope_parameters.setdefault(
            "rope_type", self.rope_parameters.pop("type", "default")
        )
        self.rope_theta = rope_theta
        super().__init__(
            pad_token_id=kwargs.pop("pad_token_id", None),
            tie_word_embeddings=kwargs.pop("tie_word_embeddings", False),
            **kwargs,
        )
