# coding=utf-8
# Copyright 2023-2026 SGLang Team
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Dory (NVIDIA looped normalized-GPT) model configuration."""

from __future__ import annotations

from typing import Dict, List, Optional

from transformers import CONFIG_MAPPING, PretrainedConfig

_ROPE_THETA = {"sliding_attention": 10_000, "full_attention": 10_000_000}


class DoryConfig(PretrainedConfig):
    """Looped latent-recurrent normalized-GPT: input block, looped recurrent block, output block.

    Attention layers repeat per loop and need one KV slot each, so the slot layout lives here
    and is the single source of truth for both the model and the KV pool sizing.
    """

    model_type = "dory"

    def __init__(
        self,
        vocab_size: int = 131072,
        hidden_size: int = 2560,
        num_hidden_layers: int = 72,
        intermediate_size: int = 7168,
        num_attention_heads: int = 16,
        num_key_value_heads: int = 4,
        head_dim: int = 256,
        max_position_embeddings: int = 262144,
        hidden_act: str = "silu",
        initializer_range: float = 0.02,
        sliding_window: Optional[int] = 1024,
        layer_types: Optional[List[str]] = None,
        rope_parameters: Optional[Dict[str, Dict]] = None,
        ngpt_alpha_init: float = 0.05,
        attention_output_gate_logit_scale_init: Optional[float] = None,
        n_input_layers: int = 0,
        n_recurrent_layers: Optional[int] = None,
        n_output_layers: int = 0,
        n_recurrent_loops: int = 1,
        recurrent_kv_cache_mode: str = "per_loop",
        tie_word_embeddings: bool = False,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.intermediate_size = intermediate_size
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim
        self.max_position_embeddings = max_position_embeddings
        self.hidden_act = hidden_act
        self.initializer_range = initializer_range
        self.sliding_window = sliding_window
        self.ngpt_alpha_init = ngpt_alpha_init
        self.attention_output_gate_logit_scale_init = (
            attention_output_gate_logit_scale_init
        )
        self.n_input_layers = n_input_layers
        self.n_output_layers = n_output_layers
        self.n_recurrent_loops = n_recurrent_loops
        self.recurrent_kv_cache_mode = recurrent_kv_cache_mode

        if layer_types is None:
            layer_types = [
                (
                    "mlp"
                    if i % 2
                    else ("full_attention" if i % 8 == 6 else "sliding_attention")
                )
                for i in range(num_hidden_layers)
            ]
        # transformers validates layer_types against an attention-kind whitelist with no "mlp";
        # Dory's layout is per physical layer (attention OR mlp), so it is carried separately.
        self.block_types = list(layer_types)
        self.layer_types = None
        if n_recurrent_layers is None:
            n_recurrent_layers = num_hidden_layers - n_input_layers - n_output_layers
        self.n_recurrent_layers = n_recurrent_layers
        if rope_parameters is None:
            rope_parameters = {
                kind: {
                    "rope_type": "default",
                    "rope_theta": theta,
                    "partial_rotary_factor": 1.0,
                }
                for kind, theta in _ROPE_THETA.items()
            }
        self.rope_parameters = rope_parameters

        self._validate()
        self._init_kv_slots()
        super().__init__(tie_word_embeddings=tie_word_embeddings, **kwargs)

    def _validate(self):
        types = self.block_types
        if len(types) != self.num_hidden_layers:
            raise ValueError("layer_types must have num_hidden_layers entries")
        unknown = set(types) - {"sliding_attention", "full_attention", "mlp"}
        if unknown:
            raise ValueError(f"Unsupported layer types: {sorted(unknown)}")
        partitions = (
            self.n_input_layers,
            self.n_recurrent_layers,
            self.n_output_layers,
        )
        if any(n < 0 for n in partitions) or sum(partitions) != self.num_hidden_layers:
            raise ValueError(
                "input, recurrent and output layers must sum to num_hidden_layers"
            )
        if self.n_recurrent_loops < 1:
            raise ValueError("n_recurrent_loops must be a positive integer")
        if self.recurrent_kv_cache_mode not in ("per_loop", "last_loop"):
            raise ValueError("recurrent_kv_cache_mode must be per_loop or last_loop")
        if "sliding_attention" in types and not self.sliding_window:
            raise ValueError("sliding_attention layers need sliding_window")
        for kind in set(types) - {"mlp"}:
            if kind not in self.rope_parameters:
                raise ValueError(f"rope_parameters is missing {kind}")

    def _init_kv_slots(self):
        rec_start = self.n_input_layers
        rec_end = rec_start + self.n_recurrent_layers
        is_attn = [kind != "mlp" for kind in self.block_types]
        rec_layers = [i for i in range(rec_start, rec_end) if is_attn[i]]
        static_layers = [i for i in range(self.num_hidden_layers) if is_attn[i]]
        static_layers = [i for i in static_layers if i not in set(rec_layers)]
        # last_loop keeps one slot per recurrent layer, overwritten by each loop.
        self._kv_loops = (
            self.n_recurrent_loops if self.recurrent_kv_cache_mode == "per_loop" else 1
        )
        self._static_kv_slots = {layer: i for i, layer in enumerate(static_layers)}
        self._recurrent_kv_slots = {layer: i for i, layer in enumerate(rec_layers)}
        self._num_recurrent_attn = len(rec_layers)

    def num_kv_slots(self) -> int:
        return len(self._static_kv_slots) + self._kv_loops * self._num_recurrent_attn

    def kv_slot_id(self, layer_idx: int, loop_idx: int = 0) -> int:
        static_slot = self._static_kv_slots.get(layer_idx)
        if static_slot is not None:
            return static_slot
        return (
            len(self._static_kv_slots)
            + loop_idx * self._num_recurrent_attn
            + self._recurrent_kv_slots[layer_idx]
        )

    def kv_loops(self, layer_idx: int) -> int:
        """Number of distinct KV slots this layer needs: one per loop when recurrent."""
        return self._kv_loops if layer_idx in self._recurrent_kv_slots else 1


CONFIG_MAPPING.register(DoryConfig.model_type, DoryConfig, exist_ok=True)
