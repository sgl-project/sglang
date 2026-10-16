# Copyright 2025-2026 SGLang Team
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
"""BerryLM-OS model configuration (text-only hybrid MoE decoder: gated delta-net
linear attention with a per-channel KDA forget gate, full attention every
``full_attention_interval`` layers, sparse MoE + shared expert, and a Gated
Block AttnRes mixer on the residual stream)."""

from transformers import PretrainedConfig

from sglang.srt.configs.linear_attn_model_registry import (
    LinearAttnModelSpec,
    register_linear_attn_model,
)
from sglang.srt.configs.mamba_utils import (
    KimiLinearCacheParams,
    KimiLinearStateShape,
    mamba2_state_dtype,
)


class BerryLMConfig(PretrainedConfig):
    model_type = "berrylm"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        vocab_size=180224,
        hidden_size=2048,
        num_hidden_layers=40,
        num_attention_heads=16,
        num_key_value_heads=2,
        hidden_act="silu",
        max_position_embeddings=262144,
        initializer_range=0.02,
        rms_norm_eps=1e-6,
        use_cache=True,
        tie_word_embeddings=False,
        rope_parameters=None,
        rope_scaling=None,
        partial_rotary_factor=0.25,
        attention_bias=False,
        attention_dropout=0.0,
        attn_output_gate=True,
        head_dim=256,
        linear_conv_kernel_dim=4,
        linear_key_head_dim=128,
        linear_value_head_dim=128,
        linear_num_key_heads=16,
        linear_num_value_heads=32,
        moe_intermediate_size=512,
        shared_expert_intermediate_size=512,
        num_experts_per_tok=8,
        num_experts=128,
        norm_topk_prob=True,
        output_router_logits=False,
        router_aux_loss_coef=0.001,
        layer_types=None,
        full_attention_interval=4,
        attn_res_block_size=8,
        attn_res_gated=True,
        attn_res_eps=1e-6,
        kda_gate_bottleneck=128,
        kda_safe_gate=False,
        kda_gate_lower_bound=None,
        **kwargs,
    ):
        super().__init__(tie_word_embeddings=tie_word_embeddings, **kwargs)
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.hidden_act = hidden_act
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.use_cache = use_cache
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout
        self.attn_output_gate = attn_output_gate
        self.head_dim = head_dim
        self.partial_rotary_factor = partial_rotary_factor
        # transformers v5 stores RoPE settings in ``rope_parameters``; keep both names populated.
        rope = dict(rope_parameters or rope_scaling or {})
        rope.setdefault("rope_type", "default")
        rope.setdefault("rope_theta", 10000000.0)
        rope.setdefault("partial_rotary_factor", partial_rotary_factor)
        self.rope_parameters = rope
        self.rope_scaling = rope
        self.rope_theta = rope["rope_theta"]

        self.full_attention_interval = full_attention_interval
        self.layer_types = layer_types
        if self.layer_types is None:
            self.layer_types = [
                (
                    "linear_attention"
                    if bool((i + 1) % full_attention_interval)
                    else "full_attention"
                )
                for i in range(self.num_hidden_layers)
            ]
        if len(self.layer_types) != self.num_hidden_layers:
            raise ValueError("layer_types must have num_hidden_layers entries")

        self.linear_conv_kernel_dim = linear_conv_kernel_dim
        self.linear_key_head_dim = linear_key_head_dim
        self.linear_value_head_dim = linear_value_head_dim
        self.linear_num_key_heads = linear_num_key_heads
        self.linear_num_value_heads = linear_num_value_heads
        self.moe_intermediate_size = moe_intermediate_size
        self.shared_expert_intermediate_size = shared_expert_intermediate_size
        self.num_experts_per_tok = num_experts_per_tok
        self.num_experts = num_experts
        self.norm_topk_prob = norm_topk_prob
        self.output_router_logits = output_router_logits
        self.router_aux_loss_coef = router_aux_loss_coef

        self.attn_res_block_size = int(attn_res_block_size)
        self.attn_res_gated = bool(attn_res_gated)
        self.attn_res_eps = float(attn_res_eps)
        self.kda_gate_bottleneck = int(kda_gate_bottleneck)
        self.kda_safe_gate = bool(kda_safe_gate)
        self.kda_gate_lower_bound = kda_gate_lower_bound

    @property
    def layers_block_type(self):
        # SGLang HybridLayerType values: "attention" / "linear_attention".
        return [
            "attention" if t == "full_attention" else "linear_attention"
            for t in self.layer_types
        ]

    @property
    def linear_layer_ids(self):
        return [
            i for i, t in enumerate(self.layers_block_type) if t == "linear_attention"
        ]

    @property
    def full_attention_layer_ids(self):
        return [i for i, t in enumerate(self.layers_block_type) if t == "attention"]

    @property
    def mamba2_cache_params(self) -> KimiLinearCacheParams:
        # KDA backend state layout (same as Kimi Linear): one fused q|k|v conv window
        # [kernel-1, 2*H*K + HV*V] and a [HV, V, K] recurrent state per linear layer.
        from sglang.srt.runtime_context import get_parallel

        if self.linear_key_head_dim != self.linear_value_head_dim:
            raise ValueError(
                "BerryLM KDA cache needs linear_key_head_dim == linear_value_head_dim"
            )
        shape = KimiLinearStateShape.create(
            tp_world_size=get_parallel().attn_tp_size,
            num_heads=self.linear_num_value_heads,
            head_dim=self.linear_value_head_dim,
            num_k_heads=self.linear_num_key_heads,
            head_k_dim=self.linear_key_head_dim,
            conv_kernel_size=self.linear_conv_kernel_dim,
        )
        return KimiLinearCacheParams(
            shape=shape, layers=self.linear_layer_ids, dtype=mamba2_state_dtype(self)
        )


# Hybrid plumbing (mamba pool sizing, linear-attention backend, radix-cache args) is
# driven by the linear-attention model registry: BerryLM shares the KDA backend with
# Kimi Linear (same fused q|k|v conv state layout; grouped value heads are handled in
# KDAAttnBackend.forward_extend).
register_linear_attn_model(
    LinearAttnModelSpec(
        config_class=BerryLMConfig,
        backend_class_name="sglang.srt.layers.attention.linear.kda_backend.KDAAttnBackend",
        arch_names=["BerryLMForCausalLM"],
        uses_mamba_radix_cache=True,
        support_mamba_cache=True,
        support_mamba_cache_extra_buffer=True,
    )
)
