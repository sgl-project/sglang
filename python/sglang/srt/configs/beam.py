# Copyright 2026 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Configuration for native Beam inference."""

from typing import List, Optional, Union

from transformers import PretrainedConfig

_DEFAULT_ROPE_THETA = 10000.0


def _normalize_rope_parameters(raw: Optional[dict]) -> dict:
    """Return the per-layer-type RoPE dict: default RoPE on sliding layers, NoPE on full."""
    raw = raw or {}
    if raw.keys() - {"sliding_attention", "full_attention"}:
        raise ValueError("rope_parameters must use per-layer-type keys")
    if raw.get("full_attention") is not None:
        raise ValueError("full_attention must run without rotary embeddings")
    sliding = raw.get("sliding_attention") or {}
    if sliding.get("rope_type", "default") != "default":
        raise ValueError("Beam supports only default RoPE")
    return {
        "sliding_attention": {
            "rope_type": "default",
            "rope_theta": float(sliding.get("rope_theta", _DEFAULT_ROPE_THETA)),
        },
        "full_attention": None,
    }


class BeamConfig(PretrainedConfig):
    model_type = "beam"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        vocab_size: int = 151936,
        hidden_size: int = 5120,
        intermediate_size: int = 20480,
        num_hidden_layers: int = 52,
        num_attention_heads: int = 48,
        num_key_value_heads: int = 8,
        head_dim: int = 128,
        hidden_act: str = "silu",
        max_position_embeddings: int = 1_048_576,
        initializer_range: float = 0.02,
        rms_norm_eps: float = 1e-6,
        use_cache: bool = True,
        tie_word_embeddings: bool = False,
        rope_parameters: Optional[dict] = None,
        attention_bias: bool = False,
        attention_dropout: float = 0.0,
        sliding_window: Optional[int] = 512,
        layer_types: Optional[List[str]] = None,
        mlp_layer_types: Optional[List[str]] = None,
        moe_intermediate_size: int = 2560,
        num_experts: int = 256,
        num_experts_per_tok: int = 8,
        norm_topk_prob: bool = True,
        n_group: int = 8,
        topk_group: int = 4,
        routed_scaling_factor: float = 2.5,
        final_logit_softcapping: Optional[float] = 30.0,
        pad_token_id: Optional[int] = None,
        bos_token_id: Optional[int] = None,
        eos_token_id: Optional[Union[int, List[int]]] = None,
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
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.use_cache = use_cache
        self.tie_word_embeddings = tie_word_embeddings
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout
        self.sliding_window = sliding_window
        self.moe_intermediate_size = moe_intermediate_size
        self.num_experts = num_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.norm_topk_prob = norm_topk_prob
        self.n_group = n_group
        self.topk_group = topk_group
        self.routed_scaling_factor = routed_scaling_factor
        self.final_logit_softcapping = final_logit_softcapping

        if num_hidden_layers <= 0:
            raise ValueError("num_hidden_layers must be positive")
        if layer_types is None:
            layer_types = [
                "sliding_attention" if (i + 1) % 6 else "full_attention"
                for i in range(num_hidden_layers)
            ]
            layer_types[-1] = "full_attention"
        self.layer_types = list(layer_types)
        if mlp_layer_types is None:
            num_dense_layers = min(3, num_hidden_layers)
            mlp_layer_types = ["dense"] * num_dense_layers + ["sparse"] * (
                num_hidden_layers - num_dense_layers
            )
        self.mlp_layer_types = list(mlp_layer_types)
        # Validate the schedules before the Transformers constructor, whose own
        # layer-type check raises a non-ValueError with a generic message.
        self._validate_layer_schedules()

        kwargs.setdefault("architectures", ["BeamForCausalLM"])
        if kwargs.get("dtype") is None and kwargs.get("torch_dtype") is None:
            kwargs["dtype"] = "bfloat16"
        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )

        # Assign after the Transformers constructor: its RoPE compatibility
        # pass would flatten a per-layer-type dictionary it finds early.
        self.rope_parameters = _normalize_rope_parameters(rope_parameters)

        self._validate_beam_config()

    def _validate_layer_schedules(self):
        for name in ("layer_types", "mlp_layer_types"):
            if len(getattr(self, name)) != self.num_hidden_layers:
                raise ValueError(f"{name} length must equal num_hidden_layers")
        if any(
            kind not in ("sliding_attention", "full_attention")
            for kind in self.layer_types
        ):
            raise ValueError(
                "layer_types must contain sliding_attention or full_attention"
            )
        if any(kind not in ("dense", "sparse") for kind in self.mlp_layer_types):
            raise ValueError("mlp_layer_types must contain dense or sparse")

    def _validate_beam_config(self):
        for name in (
            "vocab_size",
            "hidden_size",
            "intermediate_size",
            "num_attention_heads",
            "head_dim",
            "max_position_embeddings",
        ):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.head_dim % 2:
            raise ValueError("head_dim must be even for split-half RoPE")
        if "sliding_attention" in self.layer_types and (
            self.sliding_window is None or self.sliding_window < 2
        ):
            raise ValueError(
                "sliding_attention layers require sliding_window >= 2; "
                "native attention cannot represent a one-token window"
            )
        if (
            self.num_key_value_heads <= 0
            or self.num_attention_heads % self.num_key_value_heads
        ):
            raise ValueError(
                "num_attention_heads must be divisible by num_key_value_heads"
            )
        if "sparse" in self.mlp_layer_types:
            if (
                self.n_group <= 0
                or self.num_experts % self.n_group
                or self.num_experts // self.n_group < 2
            ):
                raise ValueError(
                    "num_experts must form n_group equal groups with at least two experts"
                )
            if not 1 <= self.topk_group <= self.n_group:
                raise ValueError("topk_group must be between 1 and n_group")
            if (
                not 1
                <= self.num_experts_per_tok
                <= self.topk_group * (self.num_experts // self.n_group)
            ):
                raise ValueError(
                    "num_experts_per_tok exceeds the selected expert groups"
                )

        # These fields describe the numerical contract of native Beam inference.
        # Fail early for variants that the model does not implement.
        required = {
            "hidden_act": "silu",
            "tie_word_embeddings": False,
            "attention_bias": False,
            "norm_topk_prob": True,
        }
        for name, expected in required.items():
            if getattr(self, name) != expected:
                raise ValueError(f"Native Beam requires {name}={expected!r}")


__all__ = ["BeamConfig"]
