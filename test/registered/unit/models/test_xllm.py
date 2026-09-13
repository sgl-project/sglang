# Copyright 2023-2026 SGLang Team
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

import math
from types import SimpleNamespace

import pytest
import torch

from sglang.srt.configs.k2_horizon import K2HorizonConfig, XllmConfig
from sglang.srt.models.xllm import (
    EntryClass,
    K2HorizonForCausalLM,
    XllmAttention,
    XllmForCausalLM,
    XllmGroupRMSNorm,
    _normalize_k2_horizon_config,
    _validate_mova_config,
    _xllm_router_gemm,
    _xllm_stacked_params_mapping,
    permute_to_hf,
    permute_to_xllm,
)
from sglang.srt.runtime_context import get_context
from sglang.srt.utils.hf_transformers.common import _CONFIG_REGISTRY
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=9, suite="base-a-test-cpu")


class _IdentityRotary:
    def __call__(self, positions, q, k):
        self.positions_shape = tuple(positions.shape)
        self.q_shape = tuple(q.shape)
        self.k_shape = tuple(k.shape)
        return q, k


def test_native_config_and_model_registration():
    assert _CONFIG_REGISTRY["xllm"] is XllmConfig
    assert _CONFIG_REGISTRY["k2_horizon"] is K2HorizonConfig
    assert XllmForCausalLM in EntryClass
    assert K2HorizonForCausalLM in EntryClass


def test_dense_horizon_yarn_schema_normalization():
    config = K2HorizonConfig.from_dict(
        {
            "architectures": ["K2HorizonForCausalLM"],
            "model_type": "k2_horizon",
            "hidden_size": 16,
            "num_hidden_layers": 2,
            "mlp_only_layers": [0, 1],
            "rope_theta": 1_000_000.0,
            "max_position_embeddings": 32,
            "rope_parameters": {
                "rope_type": "yarn",
                "factor": 4.0,
                "original_max_position_embeddings": 8,
                "attention_factor": 1.1,
                "beta_fast": 32.0,
                "beta_slow": 1.0,
                "truncate": True,
            },
        }
    )

    _normalize_k2_horizon_config(config)

    assert config.num_values == 0
    assert config.num_values_per_tok == 0
    assert config.num_experts == 0
    assert config.num_experts_per_tok == 0
    assert config.num_shared_experts == 0
    assert config.num_dense_layers == 2
    assert config.rope_scaling["rope_type"] == "yarn"
    assert config.rope_scaling["original_max_position_embeddings"] == 8
    assert config.rope_scaling["attn_factor"] == pytest.approx(
        1.1 / (0.1 * math.log(4.0) + 1.0)
    )
    assert config._sglang_xllm_checkpoint_format == "k2_horizon_hf"


def test_mova_horizon_schema_requires_and_applies_source_router_contract():
    config = K2HorizonConfig.from_dict(
        {
            "architectures": ["K2HorizonForCausalLM"],
            "model_type": "k2_horizon",
            "hidden_size": 16,
            "num_hidden_layers": 4,
            "num_experts": 8,
            "num_experts_per_tok": 2,
            "num_shared_experts": 1,
            "mova_num_experts": 4,
            "mova_num_experts_per_tok": 2,
            "mlp_only_layers": [0],
            "attention_gate_func": "softplus",
            "rope_parameters": {
                "rope_type": "default",
                "rope_theta": 10_000_000.0,
            },
            "xllm_source_router_gemm_partitions": 2,
        }
    )

    _normalize_k2_horizon_config(config)

    assert config.num_values == 4
    assert config.num_values_per_tok == 2
    assert config.num_dense_layers == 1
    assert config.apply_attn_gate is True
    assert config.attn_gate_func == "softplus"
    assert config.rope_theta == 10_000_000.0


def test_mova_horizon_schema_rejects_missing_source_router_contract():
    config = K2HorizonConfig.from_dict(
        {
            "hidden_size": 16,
            "num_hidden_layers": 4,
            "mova_num_experts": 4,
            "mova_num_experts_per_tok": 2,
        }
    )

    with pytest.raises(ValueError, match="source router GEMM provenance"):
        _normalize_k2_horizon_config(config)


def test_group_rms_norm_matches_groupwise_reference_and_residual_contract():
    norm = XllmGroupRMSNorm(hidden_size=4, n_groups=2, eps=0.0)
    with torch.no_grad():
        norm.weight.copy_(torch.tensor([1.0, 2.0, 3.0, 4.0]))

    hidden = torch.tensor([[3.0, 4.0, 0.0, 5.0]])
    residual = torch.ones_like(hidden)
    combined = hidden + residual
    grouped = combined.reshape(1, 2, 2)
    expected = grouped * torch.rsqrt(grouped.square().mean(-1, keepdim=True))
    expected = expected.reshape(1, 4) * norm.weight

    output, returned_residual = norm(hidden, residual=residual)
    torch.testing.assert_close(output, expected)
    torch.testing.assert_close(returned_residual, combined)


def test_partial_rope_round_trips_non_rotary_dimensions():
    attention = object.__new__(XllmAttention)
    torch.nn.Module.__init__(attention)
    attention.num_heads = 2
    attention.num_kv_heads = 1
    attention.head_dim = 8
    attention.rope_head_dim = 4
    attention.rotary_emb = _IdentityRotary()

    positions = torch.arange(3)
    q = torch.randn(3, attention.num_heads * attention.head_dim)
    k = torch.randn(3, attention.num_kv_heads * attention.head_dim)
    q_out, k_out = XllmAttention._apply_partial_rope(attention, positions, q, k)

    torch.testing.assert_close(q_out, q)
    torch.testing.assert_close(k_out, k)
    assert attention.rotary_emb.positions_shape == (3,)
    assert attention.rotary_emb.q_shape == (3, 8)
    assert attention.rotary_emb.k_shape == (3, 4)


def test_permutation_helpers_use_expected_interleave_order():
    hf_value = torch.arange(8, dtype=torch.float32).reshape(1, 1, 8)
    xllm_value = torch.tensor([[[0.0, 4.0, 1.0, 5.0, 2.0, 6.0, 3.0, 7.0]]])

    torch.testing.assert_close(permute_to_xllm(hf_value), xllm_value)
    torch.testing.assert_close(permute_to_hf(xllm_value), hf_value)


def test_mp2_router_gemm_preserves_source_rounding_order():
    hidden = torch.tensor([[1.25, -0.75, 0.5, 2.0]], dtype=torch.bfloat16)
    weight = torch.tensor(
        [[0.25, 1.5, -1.0, 0.75], [2.0, -0.5, 1.25, 0.5]],
        dtype=torch.bfloat16,
    )
    hidden_parts = hidden.chunk(2, dim=-1)
    weight_parts = weight.chunk(2, dim=-1)
    expected = (
        torch.nn.functional.linear(
            hidden_parts[0].contiguous(), weight_parts[0].contiguous()
        ).float()
        + torch.nn.functional.linear(
            hidden_parts[1].contiguous(), weight_parts[1].contiguous()
        ).float()
    )

    torch.testing.assert_close(_xllm_router_gemm(hidden, weight, 2), expected)
    with pytest.raises(ValueError, match="requires BF16"):
        _xllm_router_gemm(hidden.float(), weight.float(), 2)


def test_mova_weight_mapping_uses_checkpoint_shaped_attention_projections():
    config = SimpleNamespace(num_values=4)
    mapping = _xllm_stacked_params_mapping(config)

    assert mapping == [
        (".gate_up_proj", ".gate_proj", 0),
        (".gate_up_proj", ".up_proj", 1),
        (".v_experts.weight", ".v_experts.0.weight", 0),
        (".v_experts.weight", ".v_experts.1.weight", 1),
        (".v_experts.weight", ".v_experts.2.weight", 2),
        (".v_experts.weight", ".v_experts.3.weight", 3),
    ]


def test_dense_weight_mapping_packs_qkv_and_gate_up():
    config = SimpleNamespace(num_values=0)

    assert _xllm_stacked_params_mapping(config) == [
        (".qkv_proj", ".q_proj", "q"),
        (".qkv_proj", ".k_proj", "k"),
        (".qkv_proj", ".v_proj", "v"),
        (".gate_up_proj", ".gate_proj", 0),
        (".gate_up_proj", ".up_proj", 1),
    ]


def test_strict_loader_rejects_unknown_checkpoint_weight():
    model = object.__new__(XllmForCausalLM)
    torch.nn.Module.__init__(model)
    model.config = SimpleNamespace(model_type="xllm", tie_word_embeddings=False)
    model.model = SimpleNamespace(start_layer=0, end_layer=1)
    model.pp_group = SimpleNamespace(is_first_rank=True, is_last_rank=True)
    model.stacked_params_mapping = []
    model.expert_params_mapping = []

    with pytest.raises(RuntimeError, match="did not resolve"):
        model.load_weights([("unexpected.weight", torch.ones(1))])


def _native_runtime_config(*, enable_two_batch_overlap=False, **overrides):
    values = {
        "enable_eplb": False,
        "init_expert_location": "trivial",
        "ep_num_redundant_experts": 0,
        "enable_two_batch_overlap": enable_two_batch_overlap,
    }
    values.update(overrides)
    return values


def test_native_xllm_requires_bfloat16(monkeypatch):
    config = XllmConfig(num_values=0, num_experts=192)
    monkeypatch.setattr(torch, "get_default_dtype", lambda: torch.float16)

    with (
        get_context().override_server_args(**_native_runtime_config()),
        pytest.raises(ValueError, match="requires --dtype bfloat16"),
    ):
        _validate_mova_config(config, quant_config=None)


def test_native_xllm_accepts_compressed_tensors_quantization(monkeypatch):
    config = XllmConfig(num_values=0, num_experts=0)
    quant_config = SimpleNamespace(get_name=lambda: "compressed_tensors")
    monkeypatch.setattr(torch, "get_default_dtype", lambda: torch.bfloat16)

    with get_context().override_server_args(**_native_runtime_config()):
        _validate_mova_config(config, quant_config=quant_config)


def test_native_xllm_rejects_other_quantization(monkeypatch):
    config = XllmConfig(num_values=0, num_experts=0)
    quant_config = SimpleNamespace(get_name=lambda: "awq")
    monkeypatch.setattr(torch, "get_default_dtype", lambda: torch.bfloat16)

    with (
        get_context().override_server_args(**_native_runtime_config()),
        pytest.raises(ValueError, match="supports only compressed-tensors"),
    ):
        _validate_mova_config(config, quant_config=quant_config)


def test_native_xllm_declares_quantized_fused_module_mapping():
    assert XllmForCausalLM.packed_modules_mapping == {
        "qkv_proj": ["q_proj", "k_proj", "v_proj"],
        "gate_up_proj": ["gate_proj", "up_proj"],
    }
    assert K2HorizonForCausalLM.packed_modules_mapping == (
        XllmForCausalLM.packed_modules_mapping
    )


def test_native_xllm_accepts_bfloat16_without_expert_remapping(monkeypatch):
    config = XllmConfig(num_values=0, num_experts=192)
    monkeypatch.setattr(torch, "get_default_dtype", lambda: torch.bfloat16)

    with get_context().override_server_args(**_native_runtime_config()):
        _validate_mova_config(config, quant_config=None)


@pytest.mark.parametrize(
    ("config_override", "error"),
    [
        ({"query_key_norm": True}, "query/key normalization"),
        ({"sliding_window": 4096}, "full causal attention only"),
        ({"use_sliding_window": True}, "full causal attention only"),
        ({"apply_attn_gate": True}, "gated attention"),
    ],
    ids=["qk-norm", "sliding-window", "use-sliding-window", "attention-gate"],
)
def test_native_dense_xllm_rejects_unimplemented_attention_features(
    monkeypatch, config_override, error
):
    config = XllmConfig(num_values=0, num_experts=192, **config_override)
    monkeypatch.setattr(torch, "get_default_dtype", lambda: torch.bfloat16)

    with (
        get_context().override_server_args(**_native_runtime_config()),
        pytest.raises(ValueError, match=error),
    ):
        _validate_mova_config(config, quant_config=None)


def test_native_xllm_rejects_two_batch_overlap(monkeypatch):
    config = XllmConfig(num_values=0, num_experts=192)
    monkeypatch.setattr(torch, "get_default_dtype", lambda: torch.bfloat16)

    with (
        get_context().override_server_args(
            **_native_runtime_config(enable_two_batch_overlap=True)
        ),
        pytest.raises(ValueError, match="does not yet support.*two-batch-overlap"),
    ):
        _validate_mova_config(config, quant_config=None)


@pytest.mark.parametrize(
    "runtime_override",
    [
        {"enable_eplb": True},
        {"init_expert_location": "random"},
        {"ep_num_redundant_experts": 1},
    ],
    ids=["eplb", "initial-placement", "redundant-expert"],
)
def test_native_xllm_rejects_unmapped_expert_modes(monkeypatch, runtime_override):
    config = XllmConfig(num_values=0, num_experts=192)
    monkeypatch.setattr(torch, "get_default_dtype", lambda: torch.bfloat16)

    with (
        get_context().override_server_args(
            **_native_runtime_config(**runtime_override)
        ),
        pytest.raises(ValueError, match="does not yet support EPLB"),
    ):
        _validate_mova_config(config, quant_config=None)


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v"]))
