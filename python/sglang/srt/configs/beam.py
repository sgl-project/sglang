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


def _default_rope_parameters() -> dict:
    return {
        "sliding_attention": {
            "rope_type": "default",
            "rope_theta": _DEFAULT_ROPE_THETA,
        },
        "full_attention": None,
    }


def _normalize_rope_parameters(raw) -> dict:
    """Normalize Beam's fixed sliding-RoPE and full-NoPE configuration."""
    if raw is None:
        return _default_rope_parameters()
    if not isinstance(raw, dict):
        raise TypeError(f"rope_parameters must be a dictionary: {raw!r}")
    expected = {"sliding_attention", "full_attention"}
    if set(raw) != expected:
        raise ValueError(
            f"rope_parameters must contain exactly {sorted(expected)}: {raw!r}"
        )
    if raw["full_attention"] is not None:
        raise ValueError("full_attention must run without rotary embeddings")
    sliding = raw["sliding_attention"]
    if not isinstance(sliding, dict):
        raise TypeError("sliding_attention rope parameters must be a dictionary")
    if sliding.get("rope_type", "default") != "default":
        raise ValueError("Beam supports only default RoPE")
    theta = sliding.get("rope_theta", _DEFAULT_ROPE_THETA)
    if isinstance(theta, bool) or not isinstance(theta, (int, float)):
        raise TypeError(f"rope_theta must be a number: {theta!r}")
    return {
        "sliding_attention": {"rope_type": "default", "rope_theta": float(theta)},
        "full_attention": None,
    }


class BeamConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`BeamForCausalLM`].
    It is used to instantiate a Beam model according to the specified arguments,
    defining the model architecture. Instantiating a configuration with the defaults
    will yield a configuration similar to that of Beam-1.0-500B-A24B.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control
    the model outputs. Read the documentation from [`PretrainedConfig`] for more
    information.

    Args:
        vocab_size (`int`, *optional*, defaults to 151936):
            Vocabulary size of the Beam model.
        hidden_size (`int`, *optional*, defaults to 5120):
            Dimension of the hidden representations.
        intermediate_size (`int`, *optional*, defaults to 20480):
            Dimension of the dense SwiGLU MLP representations.
        num_hidden_layers (`int`, *optional*, defaults to 52):
            Number of decoder layers.
        num_attention_heads (`int`, *optional*, defaults to 48):
            Number of query heads for each attention layer.
        num_key_value_heads (`int`, *optional*, defaults to 8):
            Number of key/value heads for grouped-query attention.
        head_dim (`int`, *optional*, defaults to 128):
            Dimension of each attention head, independent of
            `hidden_size // num_attention_heads`.
        hidden_act (`str`, *optional*, defaults to `"silu"`):
            Activation of the SwiGLU MLPs. Native Beam supports only `"silu"`.
        max_position_embeddings (`int`, *optional*, defaults to 1048576):
            The maximum sequence length that this model might ever be used with.
        initializer_range (`float`, *optional*, defaults to 0.02):
            Standard deviation of the truncated normal initializer.
        rms_norm_eps (`float`, *optional*, defaults to 1e-06):
            Epsilon used by the RMS normalization layers.
        use_cache (`bool`, *optional*, defaults to `True`):
            Whether the model returns key/value caches.
        tie_word_embeddings (`bool`, *optional*, defaults to `False`):
            Whether to tie the input and output embeddings. Native Beam requires `False`.
        rope_parameters (`dict`, *optional*):
            RoPE configuration keyed by attention layer type. The
            `"sliding_attention"` entry holds `rope_type` (only `"default"`) and
            `rope_theta` (defaults to 10000.0); `"full_attention"` must be `None`
            because full-attention layers run without rotary embeddings.
        attention_bias (`bool`, *optional*, defaults to `False`):
            Whether the attention projections use a bias. Native Beam requires `False`.
        attention_dropout (`float`, *optional*, defaults to 0.0):
            Dropout ratio for the attention probabilities.
        sliding_window (`int`, *optional*, defaults to 512):
            Number of visible positions in sliding-attention layers.
        layer_types (`list[str]`, *optional*):
            Per-layer attention type, each `"sliding_attention"` (RoPE, sliding
            window) or `"full_attention"` (NoPE, SSMax query scaling). Defaults to
            full attention on every sixth layer and on the final layer.
        mlp_layer_types (`list[str]`, *optional*):
            Per-layer feed-forward type, each `"dense"` or `"sparse"`. Defaults to
            the first three layers dense and the rest sparse.
        moe_intermediate_size (`int`, *optional*, defaults to 2560):
            Intermediate size of each routed expert.
        num_experts (`int`, *optional*, defaults to 256):
            Total number of routed experts.
        num_experts_per_tok (`int`, *optional*, defaults to 8):
            Number of experts selected for each token.
        norm_topk_prob (`bool`, *optional*, defaults to `True`):
            Whether to normalize the selected routing scores. Native Beam requires `True`.
        n_group (`int`, *optional*, defaults to 8):
            Number of expert groups used for routing.
        topk_group (`int`, *optional*, defaults to 4):
            Number of expert groups retained for each token.
        routed_scaling_factor (`float`, *optional*, defaults to 2.5):
            Multiplier applied to the combined routed-expert output.
        final_logit_softcapping (`float`, *optional*, defaults to 30.0):
            Tanh soft cap applied to the output logits in FP32.
    """

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

    def validate(self):
        # Workaround for transformers < 5.16: its save-time validate_rope() cannot
        # handle the None full_attention entry; drop once the pin moves past it.
        rope_parameters = self.rope_parameters
        self.rope_parameters = {
            "sliding_attention": dict(rope_parameters["sliding_attention"])
        }
        try:
            super().validate()
        finally:
            self.rope_parameters = rope_parameters

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
