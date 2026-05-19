# Copyright 2025 SGLang Team
# Licensed under the Apache License, Version 2.0
# ==============================================================================
"""HuggingFace-compatible config for pi0.5 VLA model.

Registered with AutoConfig so that ``model_type: "pi05"`` in config.json
is recognized before the model module is imported.
"""

from typing import Tuple

from transformers import PretrainedConfig


class Pi05Config(PretrainedConfig):
    """HuggingFace-compatible config for pi0.5 VLA model."""

    model_type = "pi05"

    def __init__(
        self,
        paligemma_variant: str = "gemma_2b",
        action_expert_variant: str = "gemma_300m",
        chunk_size: int = 50,
        max_action_dim: int = 32,
        max_state_dim: int = 32,
        num_inference_steps: int = 10,
        image_resolution: Tuple[int, int] = (224, 224),
        dtype: str = "float32",
        vocab_size: int = 257152,
        image_token_index: int | None = None,
        hidden_size: int = 2048,
        num_hidden_layers: int = 18,
        num_attention_heads: int = 8,
        num_key_value_heads: int = 1,
        head_dim: int = 256,
        intermediate_size: int = 16384,
        max_position_embeddings: int = 8192,
        tokenizer_max_length: int = 200,
        time_sampling_beta_alpha: float = 1.5,
        time_sampling_beta_beta: float = 1.0,
        time_sampling_scale: float = 0.999,
        time_sampling_offset: float = 0.001,
        min_period: float = 4e-3,
        max_period: float = 4.0,
        **kwargs,
    ):
        if "architectures" not in kwargs:
            kwargs["architectures"] = ["Pi05ForActionPrediction"]
        super().__init__(**kwargs)

        self.paligemma_variant = paligemma_variant
        self.action_expert_variant = action_expert_variant
        self.chunk_size = chunk_size
        self.max_action_dim = max_action_dim
        self.max_state_dim = max_state_dim
        self.num_inference_steps = num_inference_steps
        if (
            not isinstance(image_resolution, (tuple, list))
            or len(image_resolution) != 2
            or image_resolution[0] != image_resolution[1]
        ):
            raise ValueError(
                f"pi0.5 expects a square image_resolution (H == W); got {image_resolution!r}."
            )
        self.image_resolution = tuple(image_resolution)
        self.image_size = self.image_resolution[0]
        self.dtype = dtype

        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim
        self.intermediate_size = intermediate_size
        self.max_position_embeddings = max_position_embeddings

        self.tokenizer_max_length = tokenizer_max_length
        self.time_sampling_beta_alpha = time_sampling_beta_alpha
        self.time_sampling_beta_beta = time_sampling_beta_beta
        self.time_sampling_scale = time_sampling_scale
        self.time_sampling_offset = time_sampling_offset
        self.min_period = min_period
        self.max_period = max_period

        self.image_token_index = vocab_size if image_token_index is None else image_token_index
