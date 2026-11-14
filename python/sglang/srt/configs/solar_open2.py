# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SGLang project

from dataclasses import replace

from sglang.srt.configs.kimi_linear import KimiLinearConfig
from sglang.srt.configs.linear_attn_model_registry import (
    LinearAttnModelSpec,
    register_linear_attn_model,
)
from sglang.srt.configs.mamba_utils import mamba2_state_dtype


class SolarOpen2Config(KimiLinearConfig):
    model_type = "solar_open2"

    def __init__(
        self,
        num_hidden_layers=32,
        gqa_layers=None,
        gqa_interval=3,
        linear_attn_config=None,
        n_routed_experts=None,
        n_shared_experts=None,
        num_experts_per_tok=None,
        n_group=1,
        topk_group=1,
        norm_topk_prob=True,
        use_qk_norm=False,
        use_gqa_gate=False,
        use_gqa_gate_bias=False,
        attention_bias=False,
        use_rope=False,
        kda_use_full_proj=False,
        kda_allow_neg_eigval=False,
        max_position_embeddings=131072,
        **kwargs,
    ):
        if num_experts_per_tok is None:
            num_experts_per_tok = kwargs.pop("num_experts_per_token", 1)
        if gqa_layers is None:
            if gqa_interval <= 0:
                raise ValueError("gqa_interval must be positive")
            # One GQA layer followed by gqa_interval KDA layers, starting at 0.
            gqa_layers = [
                i for i in range(num_hidden_layers) if i % (gqa_interval + 1) == 0
            ]
        if any(i < 0 or i >= num_hidden_layers for i in gqa_layers):
            raise ValueError("gqa_layers must contain valid zero-based layer indices")
        self.gqa_layers = sorted(set(gqa_layers))
        self.gqa_interval = gqa_interval

        linear_attn_config = dict(linear_attn_config or {})
        linear_attn_config.setdefault("head_dim", 128)
        linear_attn_config.setdefault(
            "num_heads", kwargs.get("num_attention_heads", 32)
        )
        linear_attn_config.setdefault("short_conv_kernel_size", 4)
        # Kimi's config uses one-based layer IDs; Solar's gqa_layers is zero-based.
        linear_attn_config["full_attn_layers"] = [i + 1 for i in self.gqa_layers]
        linear_attn_config["kda_layers"] = [
            i + 1 for i in range(num_hidden_layers) if i not in self.gqa_layers
        ]
        kwargs.update(
            model_type=self.model_type,
            num_experts=n_routed_experts,
            num_shared_experts=n_shared_experts,
            num_experts_per_token=num_experts_per_tok,
            num_expert_group=n_group,
            moe_renormalize=norm_topk_prob,
        )
        super().__init__(
            num_hidden_layers=num_hidden_layers,
            linear_attn_config=linear_attn_config,
            topk_group=topk_group,
            max_position_embeddings=max_position_embeddings,
            **kwargs,
        )
        self.n_shared_experts = n_shared_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.n_group = n_group
        self.norm_topk_prob = norm_topk_prob
        self.use_qk_norm = use_qk_norm
        self.use_gqa_gate = use_gqa_gate
        self.use_gqa_gate_bias = use_gqa_gate_bias
        self.attention_bias = attention_bias
        self.use_rope = use_rope
        self.kda_use_full_proj = kda_use_full_proj
        self.kda_allow_neg_eigval = kda_allow_neg_eigval

    @property
    def mamba2_cache_params(self):
        params = super().mamba2_cache_params
        return replace(params, dtype=mamba2_state_dtype(self))


register_linear_attn_model(
    LinearAttnModelSpec(
        config_class=SolarOpen2Config,
        backend_class_name="sglang.srt.models.solar_open2.SolarOpen2KDAAttnBackend",
        arch_names=["SolarOpen2ForCausalLM"],
    )
)
