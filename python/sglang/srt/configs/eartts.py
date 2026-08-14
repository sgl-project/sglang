# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Configuration for NVIDIA VoiceChat's EarTTS generation stage."""

from transformers import AutoConfig, PretrainedConfig


class EarTTSConfig(PretrainedConfig):
    model_type = "eartts"

    def __init__(
        self,
        hidden_size=1152,
        context_hidden_size=1536,
        intermediate_size=4608,
        num_hidden_layers=28,
        num_attention_heads=16,
        num_key_value_heads=16,
        head_dim=72,
        vocab_size=2,
        max_position_embeddings=131072,
        num_quantizers=31,
        codebook_size=1024,
        num_iter=8,
        top_p_or_k=0.8,
        noise_scale=0.8,
        exponent=3.0,
        latent_size=512,
        mog_low_rank=64,
        mog_num_layers=3,
        mog_num_predictions=1024,
        mog_min_log_std=-4.0,
        mog_eps=1e-6,
        query_pre_attn_scalar=256.0,
        attention_bias=False,
        rms_norm_eps=1e-6,
        layer_types=None,
        sliding_window=4096,
        rope_local_base_freq=10000.0,
        rope_theta=1000000.0,
        rope_scaling=None,
        rope_parameters=None,
        hidden_activation="gelu_pytorch_tanh",
        final_logit_softcapping=None,
        attn_logits_soft_cap=None,
        use_bidirectional_attention=False,
        is_causal=True,
        emb_vocab_size=151936,
        use_gated_fusion_for_text_audio=True,
        use_audio_prompt_frozen_projection=False,
        pad_token_id=0,
        tie_word_embeddings=True,
        dtype="float32",
        **kwargs,
    ):
        self.hidden_size = hidden_size
        self.context_hidden_size = context_hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.num_quantizers = num_quantizers
        self.codebook_size = codebook_size
        self.num_iter = num_iter
        self.top_p_or_k = top_p_or_k
        self.noise_scale = noise_scale
        self.exponent = exponent
        self.latent_size = latent_size
        self.mog_low_rank = mog_low_rank
        self.mog_num_layers = mog_num_layers
        self.mog_num_predictions = mog_num_predictions
        self.mog_min_log_std = mog_min_log_std
        self.mog_eps = mog_eps
        self.query_pre_attn_scalar = query_pre_attn_scalar
        self.attention_bias = attention_bias
        self.rms_norm_eps = rms_norm_eps
        self.layer_types = layer_types or ["full_attention"] * num_hidden_layers
        self.sliding_window = sliding_window
        self.rope_local_base_freq = rope_local_base_freq
        self.rope_theta = rope_theta
        self.hidden_activation = hidden_activation
        self.final_logit_softcapping = final_logit_softcapping
        self.attn_logits_soft_cap = attn_logits_soft_cap
        self.use_bidirectional_attention = use_bidirectional_attention
        self.is_causal = is_causal
        self.emb_vocab_size = emb_vocab_size
        self.use_gated_fusion_for_text_audio = use_gated_fusion_for_text_audio
        self.use_audio_prompt_frozen_projection = use_audio_prompt_frozen_projection
        super().__init__(
            pad_token_id=pad_token_id,
            tie_word_embeddings=tie_word_embeddings,
            dtype=dtype,
            **kwargs,
        )
        # Gemma3 uses a per-attention-type RoPE map. Set it after the base
        # constructor so Transformers does not validate it as a flat schema.
        self.rope_scaling = rope_scaling
        self.rope_parameters = rope_parameters or {
            "sliding_attention": {
                "rope_type": "default",
                "rope_theta": rope_local_base_freq,
            },
            "full_attention": {"rope_type": "default", "rope_theta": rope_theta},
        }


try:
    AutoConfig.register(EarTTSConfig.model_type, EarTTSConfig)
except ValueError:
    pass
