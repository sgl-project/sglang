# SPDX-License-Identifier: Apache-2.0
"""Reject native K3 config variants before fixed-kernel provider allocation.

The actual SGLang KimiLinearConfig class body is used, with a minimal
PretrainedConfig base shim because the isolated CPU runner omits transformers.
This preserves the native constructor, aliases and layer-map properties; no
weights, CUDA providers or complete server import are validated here.
"""

import ast
from functools import cache
from pathlib import Path
from types import SimpleNamespace

import pytest

import sglang
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


@cache
def _native_config_type():
    path = Path(sglang.__file__).parent / "srt/configs/kimi_linear.py"
    tree = ast.parse(path.read_text())
    config = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "KimiLinearConfig"
    )

    class PretrainedConfig:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    namespace = {
        "__name__": "native_kimi_linear_config",
        "PretrainedConfig": PretrainedConfig,
        "KimiLinearCacheParams": object,
    }
    exec(  # noqa: S102 - execute only the reviewed native configuration class
        compile(ast.Module(body=[config], type_ignores=[]), str(path), "exec"),
        namespace,
    )
    return namespace["KimiLinearConfig"]


def _native_config(**updates):
    # Canonical dimensions from moonshotai/Kimi-K3's text_config. Unlike vLLM's
    # config names, native SGLang also materializes n_routed_experts as an alias.
    full = [*range(4, 93, 4), 93]
    config = {
        "hidden_size": 7168,
        "num_hidden_layers": 93,
        "num_attention_heads": 96,
        "q_lora_rank": 1536,
        "kv_lora_rank": 512,
        "qk_nope_head_dim": 128,
        "qk_rope_head_dim": 64,
        "v_head_dim": 128,
        "mla_use_nope": True,
        "mla_use_output_gate": True,
        "hidden_act": "situ",
        "num_experts": 896,
        "num_experts_per_token": 16,
        "num_expert_group": 1,
        "topk_group": 1,
        "moe_router_activation_func": "sigmoid",
        "routed_expert_hidden_size": 3584,
        "moe_intermediate_size": 3072,
        "num_shared_experts": 2,
        "latent_moe_use_norm": True,
        "attn_res_block_size": 12,
        "activation_situ_beta": 4.0,
        "activation_situ_linear_beta": 25.0,
        "linear_attn_config": {
            "num_heads": 96,
            "head_dim": 128,
            "short_conv_kernel_size": 4,
            "use_full_rank_gate": True,
            "full_attn_layers": full,
            "kda_layers": [i for i in range(1, 94) if i not in full],
        },
    }
    return _native_config_type()(**(config | updates))


def _model(config):
    class FlashInferMXFP4Method:
        pass

    layer = SimpleNamespace(
        _is_moe_layer=True,
        mlp=SimpleNamespace(
            experts=SimpleNamespace(quant_method=FlashInferMXFP4Method())
        ),
    )
    return SimpleNamespace(
        language_model=SimpleNamespace(
            config=config, model=SimpleNamespace(layers=(layer,) * 93)
        )
    )


def test_canonical_native_config_and_dynamic_situ_options_are_admitted():
    from sglang.srt.kernel_plugins.nvidia.kimi_k3.spec import _model_disable_reasons

    config = _native_config(
        activation_situ_beta=3.0,
        activation_situ_linear_beta=11.0,
        routed_scaling_factor=1.25,
        moe_renormalize=False,
    )
    assert config.n_routed_experts == config.num_experts == 896
    assert len(config.full_attention_layer_ids) == 24
    assert len(config.linear_layer_ids) == 69
    assert _model_disable_reasons(_model(config), None) == []


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("num_experts_per_token", 8),
        ("moe_router_activation_func", "softmax"),
        ("num_expert_group", 2),
        ("topk_group", 2),
        ("hidden_act", "silu"),
        ("hidden_size", 8192),
        ("num_attention_heads", 128),
        ("moe_intermediate_size", 2048),
        ("num_shared_experts", 1),
        ("routed_expert_hidden_size", 4096),
        ("mla_use_nope", False),
        ("latent_moe_use_norm", False),
    ],
)
def test_valid_native_variants_cannot_silently_change_prepared_semantics(name, value):
    from sglang.srt.kernel_plugins.nvidia.kimi_k3.spec import _model_disable_reasons

    config = _native_config(**{name: value})
    assert any(
        name in reason for reason in _model_disable_reasons(_model(config), None)
    )


def test_canonical_expert_alias_and_missing_configuration_fail_closed():
    from sglang.srt.kernel_plugins.nvidia.kimi_k3.spec import _model_disable_reasons

    config = _native_config()
    config.n_routed_experts = 448
    assert any(
        "n_routed_experts" in reason
        for reason in _model_disable_reasons(_model(config), None)
    )
    del config.n_routed_experts
    assert any(
        "n_routed_experts" in reason
        for reason in _model_disable_reasons(_model(config), None)
    )
    assert _model_disable_reasons(_model(None), None)


def test_kda_geometry_and_attnres_capacity_are_checked_before_build():
    from sglang.srt.kernel_plugins.nvidia.kimi_k3.spec import _model_disable_reasons

    config = _native_config(attn_res_block_size=11)
    config.linear_attn_config["short_conv_kernel_size"] = 3
    reasons = _model_disable_reasons(_model(config), None)
    assert any("short_conv_kernel_size" in reason for reason in reasons)
    assert any("8 AttnRes blocks" in reason for reason in reasons)


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__]))
