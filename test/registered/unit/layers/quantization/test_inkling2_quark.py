"""The layout of the AMD Quark exports of Inkling-2, parsed from their quantization_config
(CPU only)."""

import copy
import re
import sys
from types import SimpleNamespace

import pytest
import torch

from sglang.srt.layers import linear
from sglang.srt.layers.quantization.fp8 import Fp8Config
from sglang.srt.layers.quantization.inkling2 import Inkling2QuantConfig
from sglang.srt.layers.quantization.inkling2_quark import Inkling2QuarkLayout
from sglang.srt.layers.quantization.quark.quark import QuarkConfig
from sglang.srt.models import inkling2 as inkling2_model
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.inkling2_quark_test_utils import (
    EXPERT_FORMAT,
    VARIANTS,
    has_fp8_attention,
    native_quant_dict,
    quantization_config,
)

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

NUM_LAYERS = 24
EXPERTS = "model.llm.layers.0.mlp.experts."
ATTENTION = "model.llm.layers.0.attn."
BF16, F32, U8, I32 = torch.bfloat16, torch.float32, torch.uint8, torch.int32
E4M3 = torch.float8_e4m3fn


def _layout(config):
    if isinstance(config, str):
        config = quantization_config(config)
    return Inkling2QuarkLayout.from_quantization_dict(config, NUM_LAYERS)


def _set(*path_and_value):
    *path, value = path_and_value

    def edit(config):
        for key in path[:-1]:
            config = config[key]
        config[path[-1]] = value

    return edit


@pytest.mark.parametrize("variant", VARIANTS)
def test_real_export_configs_are_accepted(variant):
    layout = _layout(variant)
    assert isinstance(layout, Inkling2QuantConfig)
    assert layout.expert_format == EXPERT_FORMAT[variant]
    assert layout.num_layers == NUM_LAYERS
    assert not (layout.tensor_dtypes or layout.mxfp8_weights or layout.nvfp4_weights)
    assert layout.unembed_output_dtype == torch.float32  # as in the native release
    if has_fp8_attention(variant):
        assert layout.attention_fp8.weight_config["dtype"] == "fp8_e4m3"
        assert layout.attention_fp8.input_config["per_token"] is True
    else:
        assert layout.attention_fp8 is None


@pytest.mark.parametrize(
    "variant,edit,key",
    [
        ("UINT4-AWQ", _set("global_quant_config", "weight", "symmetric", True), "symmetric"),
        ("UINT4-AWQ", _set("global_quant_config", "weight", "group_size", 64), "group_size"),
        ("MXFP4-RTN", _set("global_quant_config", "weight", "scale_calculation_mode", "floor"), "scale_calculation_mode"),
        ("MXFP4-RTN", _set("global_quant_config", "input_tensors", None), "input_tensors"),
        ("UINT4-AWQ", _set("quant_method", "fp8"), "quant_method"),
        ("UINT4-AWQ", _set("export", "weight_format", "fake_quantized"), "export.weight_format"),
        ("MXFP4-RTN", _set("export", "kv_cache_group", ["*k_proj"]), "export.kv_cache_group"),
        ("MXFP4-AttnFP8", lambda c: c["layer_quant_config"].popitem(), "layer_quant_config"),
        ("UINT4-AWQ", lambda c: c["exclude"].remove("model.llm.layers.*.attn.*"), "layers.0.attn.wq_du"),
        ("UINT4-AWQ", lambda c: c["exclude"].remove("model.llm.layers.*.mlp.gate"), "layers.0.mlp.gate"),
        ("MXFP4-RTN", lambda c: c["exclude"].append("model.llm.layers.7.mlp.experts"), "layers.7.mlp.experts"),
    ],
)  # fmt: skip
def test_unsupported_configs_are_rejected_with_the_offending_key(variant, edit, key):
    config = copy.deepcopy(quantization_config(variant))
    edit(config)
    with pytest.raises(ValueError, match=re.escape(key)):
        _layout(config)


def test_linear_quant_kill_switch_refuses_fp8_attention_only(monkeypatch):
    """With SGLANG_ROCM_DISABLE_LINEARQUANT=1 RowParallelLinear drops its quant config,
    so wo_ud would be built in BF16 for an FP8 checkpoint tensor."""
    monkeypatch.setattr(linear, "_disable_hip_linear_quant", True)
    with pytest.raises(ValueError, match="SGLANG_ROCM_DISABLE_LINEARQUANT"):
        _layout("MXFP4-AttnFP8")
    assert _layout("MXFP4-RTN").attention_fp8 is None


@pytest.mark.parametrize(
    "variant,name,dtype,expected",
    [
        ("MXFP4-RTN", EXPERTS + "w13_weight", U8, U8),
        ("MXFP4-RTN", EXPERTS + "w13_weight", I32, None),
        ("MXFP4-RTN", EXPERTS + "w13_weight_scale", BF16, None),
        ("MXFP4-RTN", EXPERTS + "w13_weight_zero_point", I32, None),
        ("UINT4-AWQ", EXPERTS + "w13_weight", I32, I32),
        ("UINT4-AWQ", EXPERTS + "w13_weight", U8, None),
        ("UINT4-AWQ", EXPERTS + "w2_weight_scale", BF16, F32),
        ("UINT4-AWQ", EXPERTS + "w2_weight_zero_point", I32, I32),
        ("UINT4-AWQ-AttnFP8", ATTENTION + "wq_du.weight", E4M3, E4M3),
        ("UINT4-AWQ-AttnFP8", ATTENTION + "wq_du.weight", torch.float8_e4m3fnuz, None),
        ("UINT4-AWQ-AttnFP8", ATTENTION + "wq_du.weight_scale", BF16, F32),
        ("UINT4-AWQ", ATTENTION + "wq_du.weight", BF16, BF16),
        ("UINT4-AWQ", "model.llm.layers.0.mlp.gate.bias", F32, F32),
    ],
)
def test_checkpoint_tensor_dtype_policy(variant, name, dtype, expected):
    """Packed tensors keep their dtype and BF16 scales become FP32; other dtypes are
    refused. Tensors the export does not quantize are left to the loader."""
    layout, tensor = _layout(variant), torch.ones(4).to(dtype)
    if expected is None:
        with pytest.raises(ValueError, match=re.escape(name)):
            layout.checkpoint_tensor(name, tensor)
        return
    loaded = layout.checkpoint_tensor(name, tensor)
    assert loaded.dtype == expected
    if expected == dtype:
        assert loaded is tensor
    else:
        assert torch.equal(loaded, tensor.float())


def _model_gate(monkeypatch, quant_config):
    """Construct the model up to its pipeline-parallel check, which follows the check of
    the quant config and fails for every config that passes it."""
    monkeypatch.setattr(
        inkling2_model,
        "get_parallel",
        lambda: SimpleNamespace(pp_group=SimpleNamespace(world_size=2)),
    )
    config = SimpleNamespace(text_config=SimpleNamespace(num_hidden_layers=NUM_LAYERS))
    inkling2_model.Inkling2ForConditionalGeneration(config, quant_config)


def test_the_model_gate_takes_inkling2_configs_only(monkeypatch):
    for config in (
        None,
        Inkling2QuantConfig(native_quant_dict()),
        _layout("UINT4-AWQ"),
    ):
        with pytest.raises(ValueError, match="pipeline parallelism"):
            _model_gate(monkeypatch, config)
    with pytest.raises(ValueError, match="checkpoint precision metadata"):
        _model_gate(monkeypatch, Fp8Config())


@pytest.mark.parametrize("variant", VARIANTS)
def test_generic_quark_config_converts_to_the_layout(variant):
    generic = QuarkConfig.from_config(
        {**quantization_config(variant), "packed_modules_mapping": {}}
    )
    layout = Inkling2QuarkLayout.from_quark_config(
        generic, SimpleNamespace(num_hidden_layers=NUM_LAYERS)
    )
    assert layout.expert_format == EXPERT_FORMAT[variant]
    assert (layout.attention_fp8 is not None) is has_fp8_attention(variant)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
