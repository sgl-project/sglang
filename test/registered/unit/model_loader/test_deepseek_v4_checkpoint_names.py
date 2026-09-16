"""V4 checkpoint names must agree with registered FFNs and quantization prefixes."""

from types import SimpleNamespace

from sglang.srt.layers.quantization.modelslim.modelslim import ModelSlimConfig
from sglang.srt.models.deepseek_v4 import DeepseekV4ForCausalLM
from sglang.srt.models.deepseek_v4_dspark import DeepseekV4ForCausalLMDSpark
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def test_native_v4_mapper_outputs_registered_ffn_weights_and_scales():
    remap = DeepseekV4ForCausalLM.remap_weight_name_to_dpsk_hf_format
    for suffix, target in (
        ("experts.3.w1.weight", "experts.3.gate_proj.weight"),
        ("experts.3.w2.scale", "experts.3.down_proj.weight_scale_inv"),
        ("shared_experts.w3.weight", "shared_experts.up_proj.weight"),
        ("gate.bias", "gate.e_score_correction_bias"),
    ):
        expected = f"model.layers.2.ffn.{target}"
        assert remap(f"layers.2.ffn.{suffix}") == expected
        assert remap(expected) == expected
        assert (
            remap(f"mtp.0.ffn.{suffix}", is_nextn=True, num_hidden_layers=2) == expected
        )


def test_dspark_cpu_cuda_and_npu_mappers_agree_on_ffn_targets():
    model = SimpleNamespace(confidence_head=None, uses_own_vocab_modules=False)
    for mapper in (
        DeepseekV4ForCausalLMDSpark._remap_dspark_weight_name,
        DeepseekV4ForCausalLMDSpark._remap_dspark_weight_name_npu,
    ):
        assert mapper(model, "mtp.0.ffn.experts.1.w1.weight") == (
            "stages.0.ffn.experts.1.gate_proj.weight"
        )
        assert mapper(model, "mtp.0.ffn.experts.1.w2.scale") == (
            "stages.0.ffn.experts.1.down_proj.weight_scale_inv"
        )


def test_modelslim_uses_the_same_v4_and_dspark_ffn_prefixes():
    config = ModelSlimConfig(
        {"hc_head_fn.weight": "FLOAT", "layers.2.ffn.experts.3.w1.weight": "W8A8"}
    )
    assert (
        config.quant_description["model.layers.2.ffn.experts.3.gate_proj.weight"]
        == "W8A8"
    )
    assert not any(".mlp." in name for name in config.quant_description)

    config = ModelSlimConfig({"mtp.0.ffn.experts.1.w1.weight": "W8A8"})
    assert config.quant_description["stages.0.ffn.experts.1.gate_proj.weight"] == "W8A8"
    assert not any(".mlp." in name for name in config.quant_description)


if __name__ == "__main__":
    import sys

    import pytest

    args = ["-x" if arg == "-f" else arg for arg in sys.argv[1:]]
    sys.exit(pytest.main([__file__, "-v", *args]))
