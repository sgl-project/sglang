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


class BeamConfig(PretrainedConfig):
    """Beam configuration with explicit attention and dense/MoE layer schedules.

    ``no_rope_layers`` retains the checkpoint convention: 1 applies RoPE and
    0 disables it. Attention width is ``num_attention_heads * head_dim`` and
    need not equal ``hidden_size``. The default schedule includes a final
    full-attention layer; explicitly supplied layer arrays are preserved.
    Native BF16 serving uses BF16 residual arithmetic and, by default, BF16
    LM-head projection output followed by FP32 softcap and logprobs. Legacy
    ``lm_head_dtype`` and ``residual_in_fp32`` metadata is ignored.
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
        rope_theta: Optional[float] = None,
        rope_parameters: Optional[dict] = None,
        attention_bias: bool = False,
        attention_dropout: float = 0.0,
        sliding_window: Optional[int] = 512,
        layer_types: Optional[List[str]] = None,
        no_rope_layers: Optional[List[int]] = None,
        no_rope_layer_interval: int = 6,
        decoder_sparse_step: int = 1,
        moe_intermediate_size: int = 2560,
        num_experts: int = 256,
        num_experts_per_tok: int = 8,
        norm_topk_prob: bool = True,
        mlp_only_layers: Optional[List[int]] = None,
        scoring_func: str = "sigmoid",
        n_group: int = 8,
        topk_group: int = 4,
        routed_scaling_factor: float = 2.5,
        topk_method: str = "noaux_tc",
        ssmax: bool = True,
        ssmax_offset: int = 1,
        ssmax_divide_by_log_offset: bool = False,
        ssmax_scale_mode: str = "root10",
        enable_sandwich_norm: bool = True,
        self_attn_depth_scaled_residual: bool = True,
        mlp_depth_scaled_residual: bool = True,
        use_paramfree_qk_norm: bool = False,
        attention_output_gate: bool = True,
        attention_output_gate_bias: bool = True,
        attention_output_gate_floor: float = 0.0,
        attention_output_gate_temperature: float = 1.0,
        router_dtype: str = "fp32",
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
        self.no_rope_layer_interval = no_rope_layer_interval
        self.decoder_sparse_step = decoder_sparse_step
        self.moe_intermediate_size = moe_intermediate_size
        self.num_experts = num_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.norm_topk_prob = norm_topk_prob
        self.scoring_func = scoring_func
        self.n_group = n_group
        self.topk_group = topk_group
        self.routed_scaling_factor = routed_scaling_factor
        self.topk_method = topk_method
        self.ssmax = ssmax
        self.ssmax_offset = ssmax_offset
        self.ssmax_divide_by_log_offset = ssmax_divide_by_log_offset
        self.ssmax_scale_mode = ssmax_scale_mode
        self.enable_sandwich_norm = enable_sandwich_norm
        self.self_attn_depth_scaled_residual = self_attn_depth_scaled_residual
        self.mlp_depth_scaled_residual = mlp_depth_scaled_residual
        self.use_paramfree_qk_norm = use_paramfree_qk_norm
        self.attention_output_gate = attention_output_gate
        self.attention_output_gate_bias = attention_output_gate_bias
        self.attention_output_gate_floor = attention_output_gate_floor
        self.attention_output_gate_temperature = attention_output_gate_temperature
        self.router_dtype = "fp32" if router_dtype == "float32" else router_dtype
        self.final_logit_softcapping = final_logit_softcapping

        if num_hidden_layers <= 0 or no_rope_layer_interval <= 0:
            raise ValueError(
                "num_hidden_layers and no_rope_layer_interval must be positive"
            )
        if no_rope_layers is None:
            if layer_types is not None:
                no_rope_layers = [int(t == "sliding_attention") for t in layer_types]
            else:
                no_rope_layers = [
                    int((i + 1) % no_rope_layer_interval != 0)
                    for i in range(num_hidden_layers)
                ]
                no_rope_layers[-1] = 0
        self.no_rope_layers = list(no_rope_layers)
        self.layer_types = (
            list(layer_types)
            if layer_types is not None
            else [
                "sliding_attention" if flag else "full_attention"
                for flag in self.no_rope_layers
            ]
        )
        self.mlp_only_layers = (
            list(mlp_only_layers)
            if mlp_only_layers is not None
            else list(range(min(3, num_hidden_layers)))
        )

        # Older checkpoints use rope_scaling. Consume that alias before the
        # Transformers constructor can override explicit rope_parameters.
        legacy_rope_scaling = kwargs.pop("rope_scaling", None)
        if rope_parameters is None:
            rope_parameters = legacy_rope_scaling
        # Ignore obsolete precision pins without retaining them as HF extras.
        kwargs.pop("lm_head_dtype", None)
        kwargs.pop("residual_in_fp32", None)
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

        # Normalize and validate the effective configuration after Transformers
        # has processed its compatibility fields.
        self.rope_parameters = dict(rope_parameters or {})
        legacy_rope_type = self.rope_parameters.pop("type", "default")
        self.rope_parameters.setdefault("rope_type", legacy_rope_type)
        nested_theta = self.rope_parameters.get("rope_theta", rope_theta)
        if rope_theta is not None and nested_theta != rope_theta:
            raise ValueError("rope_theta disagrees with rope_parameters.rope_theta")
        self.rope_theta = nested_theta if nested_theta is not None else 10000.0
        self.rope_parameters["rope_theta"] = self.rope_theta
        rope_type = self.rope_parameters["rope_type"]
        if rope_type != "default":
            raise ValueError("Beam supports only default RoPE")
        # Default RoPE has no scaling factor; keep context derivation unscaled.
        self.rope_parameters = {
            "rope_type": "default",
            "rope_theta": self.rope_theta,
        }
        self.rope_scaling = dict(self.rope_parameters)

        self._validate_beam_config()

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
        for name in ("no_rope_layers", "layer_types"):
            if len(getattr(self, name)) != self.num_hidden_layers:
                raise ValueError(f"{name} length must equal num_hidden_layers")
        if any(flag not in (0, 1) for flag in self.no_rope_layers):
            raise ValueError("no_rope_layers must contain only 0 (NoPE) or 1 (RoPE)")
        if any(
            kind not in ("sliding_attention", "full_attention")
            for kind in self.layer_types
        ):
            raise ValueError(
                "layer_types must contain sliding_attention or full_attention"
            )
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
        if len(set(self.mlp_only_layers)) != len(self.mlp_only_layers) or any(
            i < 0 or i >= self.num_hidden_layers for i in self.mlp_only_layers
        ):
            raise ValueError("mlp_only_layers must contain unique valid layer indices")
        if len(self.mlp_only_layers) < self.num_hidden_layers:
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
            "decoder_sparse_step": 1,
            "norm_topk_prob": True,
            "scoring_func": "sigmoid",
            "topk_method": "noaux_tc",
            "ssmax": True,
            "ssmax_offset": 1,
            "ssmax_divide_by_log_offset": False,
            "ssmax_scale_mode": "root10",
            "enable_sandwich_norm": True,
            "self_attn_depth_scaled_residual": True,
            "mlp_depth_scaled_residual": True,
            "use_paramfree_qk_norm": False,
            "attention_output_gate": True,
            "attention_output_gate_bias": True,
            "attention_output_gate_floor": 0.0,
            "attention_output_gate_temperature": 1.0,
            "router_dtype": "fp32",
        }
        for name, expected in required.items():
            if getattr(self, name) != expected:
                raise ValueError(f"Native Beam requires {name}={expected!r}")


__all__ = ["BeamConfig"]
