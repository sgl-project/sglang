import torch

from sglang.srt.layers.quantization.modelslim.modelslim import ModelSlimConfig
from sglang.srt.layers.quantization.modelslim.schemes import (
    ModelSlimW4A8Int8MoE,
    ModelSlimW8A8Int8,
)


def _kimi_k3_quant_description():
    expert_prefix = (
        "language_model.model.layers.1.block_sparse_moe.experts.0"
    )
    return {
        f"{expert_prefix}.w1.weight": "W4A8_DYNAMIC",
        f"{expert_prefix}.w2.weight": "W4A8_DYNAMIC",
        f"{expert_prefix}.w3.weight": "W4A8_DYNAMIC",
        (
            "language_model.model.layers.1.block_sparse_moe.shared_experts."
            "gate_proj.weight"
        ): "W8A8_DYNAMIC",
    }


def test_kimi_k3_modelslim_resolves_wrapper_and_moe_prefixes():
    config = ModelSlimConfig(_kimi_k3_quant_description())

    resolved = config._resolve_quant_prefix(
        "model.layers.1.mlp.shared_experts.gate_proj"
    )
    assert resolved == (
        "language_model.model.layers.1.block_sparse_moe.shared_experts."
        "gate_proj"
    )

    scheme = config.get_linear_scheme(
        torch.nn.Module(),
        "model.layers.1.mlp.shared_experts.gate_proj",
    )
    assert isinstance(scheme, ModelSlimW8A8Int8)
    assert scheme.is_dynamic


def test_kimi_k3_modelslim_resolves_w1_w3_w2_experts():
    config = ModelSlimConfig(_kimi_k3_quant_description())

    w13_scheme, w2_scheme = config.get_moe_scheme(
        torch.nn.Module(), "model.layers.1.mlp.experts"
    )

    assert isinstance(w13_scheme, ModelSlimW4A8Int8MoE)
    assert isinstance(w2_scheme, ModelSlimW4A8Int8MoE)
    assert w13_scheme.weight_prefix == "w13"
    assert w2_scheme.weight_prefix == "w2"
