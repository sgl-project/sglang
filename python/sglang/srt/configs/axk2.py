"""Configuration for SKT A.X-K2 models."""

from typing import Any

from transformers.configuration_utils import PretrainedConfig


class AXK2Config(PretrainedConfig):
    """DeepSeek-V3-style MLA/MoE config with A.X-K2 gating attributes."""

    model_type = "axk2"
    keys_to_ignore_at_inference = ["past_key_values"]
    # The dense shortcut does not implement this architecture's fused
    # query/output-gate projection, so prefill must stay on native MLA.
    supports_mha_one_shot = False

    def __init__(
        self,
        vocab_size: int = 163840,
        hidden_size: int = 7168,
        intermediate_size: int = 18432,
        moe_intermediate_size: int = 2048,
        num_hidden_layers: int = 61,
        num_attention_heads: int = 64,
        num_key_value_heads: int | None = None,
        n_shared_experts: int | None = 1,
        n_routed_experts: int | None = 256,
        num_experts_per_tok: int = 8,
        first_k_dense_replace: int = 1,
        moe_layer_freq: int = 1,
        q_lora_rank: int | None = 1536,
        kv_lora_rank: int = 512,
        qk_nope_head_dim: int = 128,
        qk_rope_head_dim: int = 64,
        v_head_dim: int = 128,
        rms_norm_eps: float = 1e-6,
        rope_theta: float = 1_000_000.0,
        rope_scaling: dict[str, Any] | None = None,
        rope_parameters: dict[str, Any] | None = None,
        gated_norm: bool = True,
        gated_norm_rank: int = 16,
        attention_output_gate: bool = True,
        attn_gate_fused: bool = True,
        index_n_heads: int | None = None,
        index_head_dim: int | None = None,
        index_topk: int | None = None,
        use_cache: bool = True,
        **kwargs,
    ):
        for name, value in vars().items():
            if name not in {"self", "kwargs", "num_key_value_heads", "rope_parameters"}:
                setattr(self, name, value)
        self.num_key_value_heads = num_key_value_heads or num_attention_heads
        self.rope_parameters = (
            rope_parameters
            or rope_scaling
            or {"rope_type": "default", "rope_theta": rope_theta}
        )
        if index_n_heads is not None:
            self.index_n_heads = index_n_heads
        if index_head_dim is not None:
            self.index_head_dim = index_head_dim
        if index_topk is not None:
            self.index_topk = index_topk
        super().__init__(**kwargs)
