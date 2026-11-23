import json
import sys
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from torch import nn

from sglang.srt.configs.model_config import ModelConfig
from sglang.srt.layers.quantization.fp8 import Fp8Config
from sglang.srt.model_loader.utils import set_default_torch_dtype
from sglang.srt.models.bailing_moe_nextn import BailingMoeForCausalLMNextN
from sglang.srt.runtime_context import get_parallel
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import published_topology

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


@pytest.fixture
def checkpoint(tmp_path):
    config = {
        "model_type": "bailing_moe_v3_vl",
        "architectures": ["BailingMoeV3VLForConditionalGeneration"],
        "torch_dtype": "bfloat16",
        "mrope_section": [8, 12, 12],
        "text_config": {
            "num_hidden_layers": 42,
            "num_nextn_predict_layers": 1,
            "hidden_size": 2560,
            "num_attention_heads": 32,
            "q_lora_rank": None,
            "layer_group_size": 6,
            "first_k_dense_replace": 2,
            "num_experts": 512,
            "moe_intermediate_size": 768,
            "moe_shared_expert_intermediate_size": 768,
            "moe_router_enable_expert_bias": True,
            "score_function": "sigmoid",
            "short_conv_kernel_size": 4,
            "mtp_use_kda": False,
            "gated_attention_proj_granularity_type": "head_wise",
            "expert_swiglu_limit_list": [4] * 42,
            "share_expert_swiglu_limit_list": [7] * 42,
        },
        "quantization_config": {
            "quant_method": "fp8",
            "activation_scheme": "dynamic",
            "weight_block_size": [128, 128],
            "modules_to_not_convert": ["model.visual", "lm_head"],
        },
    }
    (tmp_path / "config.json").write_text(json.dumps(config))
    return str(tmp_path)


def test_nextn_projection_isolates_target_quantization_and_layer_schedule(checkpoint):
    with published_topology(device="cpu"):
        target = ModelConfig(checkpoint, trust_remote_code=False)
        before = target.hf_config.to_dict()
        draft = ModelConfig(
            checkpoint,
            trust_remote_code=False,
            is_draft_model=True,
            speculative_algorithm="NEXTN",
        )

    assert target.hf_config.to_dict() == before
    assert draft.hf_config.architectures == ["BailingMoeForCausalLMNextN"]
    assert draft.hf_config is draft.hf_text_config
    assert draft.hf_text_config.full_attention_layer_ids == [0]
    assert draft.hf_text_config.linear_layer_ids == []
    assert not draft.is_multimodal
    assert draft.model_is_mrope
    assert draft.dtype == torch.bfloat16
    assert "model.layers.42.eh_proj" in draft.hf_quant_config["modules_to_not_convert"]
    assert (
        "model.layers.42.eh_proj"
        not in target.hf_quant_config["modules_to_not_convert"]
    )


def test_nextn_loads_fp8_gate_scale_and_selects_local_heads(checkpoint):
    group = SimpleNamespace(world_size=2, rank_in_group=1, cpu_group=None)
    pp = SimpleNamespace(
        world_size=1, rank_in_group=0, is_first_rank=True, is_last_rank=True
    )
    with (
        published_topology(
            device="cpu",
            tp_size=2,
            moe_runner_backend="triton",
            disable_shared_experts_fusion=True,
        ),
        get_parallel().override(
            tp_rank=1,
            attn_tp_rank=1,
            moe_tp_rank=1,
            pp_group=pp,
            tp_group=group,
            attn_tp_group=group,
            moe_tp_group=group,
        ),
        set_default_torch_dtype(torch.bfloat16),
    ):
        config = ModelConfig(
            checkpoint,
            trust_remote_code=False,
            is_draft_model=True,
            speculative_algorithm="NEXTN",
        ).hf_config
        with (
            torch.device("meta"),
            patch(
                "sglang.srt.models.deepseek_v2.get_rope_wrapper",
                return_value=nn.Identity(),
            ),
            patch(
                "sglang.srt.models.bailing_moe_v3.get_rope", return_value=nn.Identity()
            ),
        ):
            draft = BailingMoeForCausalLMNextN(
                config, Fp8Config.from_config(config.quantization_config)
            )
        assert draft.model.eh_proj.weight.dtype == torch.bfloat16
        assert not hasattr(draft.model.eh_proj, "weight_scale_inv")
        assert draft.model.decoder.mlp.expert_swiglu_limit is None
        assert draft.model.decoder.mlp.share_expert_swiglu_limit is None

        attention = draft.model.decoder.attention
        gate = attention.g_proj
        assert gate.weight.dtype == torch.float8_e4m3fn
        assert gate.weight.shape == (32, 2560)
        assert gate.weight_scale_inv.shape == (1, 20)
        for name in ("weight", "weight_scale_inv"):
            param = nn.Parameter(
                torch.zeros_like(getattr(gate, name), device="cpu"), requires_grad=False
            )
            param.weight_loader = gate.weight_loader
            setattr(gate, name, param)
        weight = (torch.arange(32).float()[:, None].expand(32, 2560) / 16).to(
            torch.float8_e4m3fn
        )
        scale = torch.arange(1, 21, dtype=torch.float32).reshape(1, 20) / 20
        with patch.object(draft, "post_load_weights"):
            draft.load_weights(
                iter(
                    [
                        ("model.layers.42.attention.g_proj.weight_scale_inv", scale),
                        ("model.layers.42.attention.g_proj.weight", weight),
                    ]
                )
            )
        torch.testing.assert_close(gate.weight.float(), weight.float())
        torch.testing.assert_close(gate.weight_scale_inv, scale)
        inputs = torch.full((2, 2560), 1 / 2560, dtype=torch.bfloat16)

        def cpu_gate(layer, x, bias=None):
            dequantized = (
                layer.weight.float() * layer.weight_scale_inv.repeat_interleave(128, 1)
            )
            return (x.float() @ dequantized.T).to(x.dtype)

        # CPU reference replaces only the FP8 GEMM; head selection and sigmoid run normally.
        with patch.object(gate.quant_method, "apply", side_effect=cpu_gate):
            result = attention._forward_gated(inputs)
        full_gate = cpu_gate(gate, inputs)
        torch.testing.assert_close(
            result, full_gate[:, 16:].float().sigmoid().to(inputs.dtype)
        )


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
