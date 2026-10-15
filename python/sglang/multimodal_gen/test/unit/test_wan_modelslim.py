# SPDX-License-Identifier: Apache-2.0
"""CPU checks of ModelSlim checkpoint loading and NPU operator contracts."""

import json
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from safetensors.torch import load_file, save_file

from sglang.multimodal_gen.configs.models.dits.wanvideo import WanVideoArchConfig
from sglang.multimodal_gen.runtime.layers.linear import (
    LinearBase,
    UnquantizedLinearMethod,
)
from sglang.multimodal_gen.runtime.layers.quantization import modelslim_mxfp4_scheme
from sglang.multimodal_gen.runtime.layers.quantization.modelslim import ModelSlimConfig
from sglang.multimodal_gen.runtime.layers.quantization.modelslim_mxfp4_scheme import (
    ModelSlimMXFP4Scheme,
)
from sglang.multimodal_gen.runtime.managers.forward_context import set_forward_context
from sglang.multimodal_gen.tools.wan_repack import convert_transformer, pack_fp4_weight
from sglang.srt.environ import envs
from sglang.srt.hardware_backend.npu.quantization import linear_method_npu


@pytest.mark.parametrize("quant_type", ["FLOAT", "W4A4_MXFP4", "W8A8_MXFP8"])
def test_remapped_linear_scheme(quant_type):
    config = ModelSlimConfig(
        {"blocks.0.attn1.to_q.weight": quant_type},
        WanVideoArchConfig().reverse_param_names_mapping,
    )
    layer = LinearBase(64, 64, quant_config=config, prefix="blocks.0.to_q")
    if quant_type == "FLOAT":
        assert isinstance(layer.quant_method, UnquantizedLinearMethod)
    else:
        assert layer.scheme.__class__.__name__ == (
            "ModelSlimMXFP4Scheme"
            if quant_type == "W4A4_MXFP4"
            else "ModelSlimMXFP8Scheme"
        )


@pytest.mark.parametrize(
    "types,expected",
    [
        (("FLOAT", "FLOAT"), True),
        (("W4A4_MXFP4",) * 2, False),
        (("FLOAT", "W4A4_MXFP4"), None),
    ],
)
def test_remapped_fused_shards(types, expected):
    config = ModelSlimConfig(
        {
            f"blocks.0.attn1.to_{name}.weight": kind
            for name, kind in zip(("q", "k"), types)
        },
        WanVideoArchConfig().reverse_param_names_mapping,
    )
    mapping = {"qkv": ["to_q", "to_k"]}
    if expected is None:
        with pytest.raises(ValueError, match="some but not all shards"):
            config.is_layer_skipped("blocks.0.qkv", mapping)
    else:
        assert config.is_layer_skipped("blocks.0.qkv", mapping) is expected


def test_i2v_repack_preserves_tensors_and_descriptors(tmp_path):
    names = {"k_img": "add_k_proj", "v_img": "add_v_proj", "norm_k_img": "norm_added_k"}
    weights = {
        f"blocks.0.cross_attn.{name}.weight": torch.ones(2, 64) for name in names
    }
    values = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])
    fp4 = torch.cat((values, -values)).unsqueeze(0).to(torch.float8_e4m3fn)
    weights["blocks.0.self_attn.q.linear.weight"] = fp4
    weights["blocks.0.self_attn.q_rot"] = torch.eye(2)
    description = {key: "FLOAT" for key in weights}
    description.update(
        {
            "blocks.0.self_attn.q.linear.weight": "W4A4_MXFP4",
            "blocks.0.self_attn.quant_type": "MXFP4_DYNAMIC",
            "timestep_policy": {"w4a4_linear": {"default": "W4A4"}},
        }
    )
    save_file(weights, tmp_path / "quant_model_weight.safetensors")
    (tmp_path / "quant_model_description.json").write_text(json.dumps(description))
    out = tmp_path / "out"
    convert_transformer("Wan2.2-I2V-A14B", tmp_path, out)
    tensors = load_file(out / "diffusion_pytorch_model.safetensors")
    metadata = json.loads((out / "quant_model_description.json").read_text())
    for name, target in names.items():
        key = f"blocks.0.attn2.{target}.weight"
        assert torch.equal(tensors[key], weights[f"blocks.0.cross_attn.{name}.weight"])
        assert metadata[key] == "FLOAT"
    assert tensors["blocks.0.attn1.to_q.weight"].tolist() == [
        [16, 50, 84, 118, 152, 186, 220, 254]
    ]
    assert torch.equal(tensors["blocks.0.attn1.q_rot"], torch.eye(2))
    assert metadata["blocks.0.attn1.quant_type"] == "MXFP4_DYNAMIC"
    assert metadata["timestep_policy"] == description["timestep_policy"]
    assert (
        pack_fp4_weight(tensors["blocks.0.attn1.to_q.weight"])
        is tensors["blocks.0.attn1.to_q.weight"]
    )
    save_file(
        {
            "blocks.0.self_attn.q_rot": torch.eye(2),
            "blocks.0.attn1.q_rot": torch.eye(2),
        },
        tmp_path / "quant_model_weight.safetensors",
    )
    with pytest.raises(ValueError, match="collide"):
        convert_transformer("Wan2.2-I2V-A14B", tmp_path, out)


@pytest.mark.parametrize("dual", [False, True])
@pytest.mark.parametrize("nz", [False, True])
def test_packed_weight_and_scale_preparation(monkeypatch, dual, nz):
    # Simulate device residency only; this verifies bytes/shapes, not physical NZ.
    monkeypatch.setattr(torch.Tensor, "is_npu", property(lambda _: True), raising=False)
    monkeypatch.setenv("SGLANG_NPU_ENABLE_MXFP4_NZ", "1" if nz else "0")
    monkeypatch.setenv("SGLANG_NPU_DISABLE_ACL_FORMAT_WEIGHT", "0")
    formats = {}

    def format_cast(weight, *args, **kwargs):
        converted = weight.clone()
        formats[converted.data_ptr()] = 50 if "input_dtype" in kwargs else 29
        return converted

    cast = Mock(side_effect=format_cast)
    monkeypatch.setattr(
        modelslim_mxfp4_scheme,
        "torch_npu",
        SimpleNamespace(
            npu_format_cast=cast,
            float4_e2m1fn_x2=296,
            get_npu_format=lambda weight: formats[weight.data_ptr()],
        ),
        raising=False,
    )
    config = {"linear.mul_scale": "FLOAT", "linear.weight_dual_scale": "FLOAT"}
    scheme = ModelSlimMXFP4Scheme(
        config, "linear", "W4A4_MXFP4_DUALSCALE" if dual else "W4A4_MXFP4"
    )
    layer = torch.nn.Module()
    scheme.create_weights(layer, 512, [2], 512, 2, torch.bfloat16)
    packed = torch.arange(256, dtype=torch.uint8).repeat(2, 1)
    scales = torch.arange(32, dtype=torch.uint8).reshape(2, 16)
    layer.weight.data.copy_(packed)
    layer.weight_scale.data.copy_(scales)
    layer.mul_scale.data.fill_(2)
    if dual:
        layer.weight_dual_scale.data.fill_(3)
    scheme.process_weights_after_loading(layer)
    assert layer.use_mul_scale and torch.equal(layer.mul_scale, torch.full((512,), 2.0))
    if dual:
        assert torch.equal(layer.weight.view(torch.uint8), packed)
        assert torch.equal(layer.weight_scale, scales.reshape(2, 8, 2))
        assert layer.weight_dual_scale.shape == (1, 2)
        assert cast.call_args.args[1] == 29
        assert cast.call_args.kwargs == {"customize_dtype": torch.int8}
        quant = Mock(return_value=(torch.ones(1, 256), torch.ones(1), torch.ones(1)))
        matmul = Mock(return_value=torch.zeros(1, 2, dtype=torch.bfloat16))
        monkeypatch.setattr(
            modelslim_mxfp4_scheme,
            "torch_npu",
            SimpleNamespace(
                npu_dynamic_dual_level_mx_quant=quant,
                npu_dual_level_quant_matmul=matmul,
            ),
        )
        x = torch.ones(1, 512, dtype=torch.bfloat16)
        scheme.apply_weights(layer, x)
        assert torch.equal(quant.call_args.args[0], x)
        assert torch.equal(quant.call_args.kwargs["smooth_scale"], x[0] * 2)
    else:
        assert torch.equal(layer.weight.view(torch.uint8), packed.T)
        assert torch.equal(layer.weight_scale, scales.reshape(2, 8, 2).transpose(0, 1))
        if nz:
            assert cast.call_count == 2
            assert cast.call_args_list[0].kwargs == {
                "customize_dtype": torch.float8_e4m3fn,
                "input_dtype": 296,
            }
            assert cast.call_args_list[1].kwargs == {}
            assert torch.equal(layer.weight_w4a8.view(torch.uint8), packed.T)
            assert layer.weight_w4a8.data_ptr() != layer.weight.data_ptr()
            assert "weight_w4a8" not in layer.state_dict()
            runtime = modelslim_mxfp4_scheme.torch_npu
            for actual in (2, 29):
                layer.weight = torch.nn.Parameter(packed.clone(), requires_grad=False)
                layer.weight_scale = torch.nn.Parameter(
                    scales.clone(), requires_grad=False
                )
                monkeypatch.setattr(runtime, "get_npu_format", lambda _: actual)
                with pytest.raises(RuntimeError, match="expected"):
                    scheme.process_weights_after_loading(layer)
        else:
            cast.assert_not_called()


def test_nz_rejects_disabled_weight_formats(monkeypatch):
    scheme = ModelSlimMXFP4Scheme({}, "linear", "W4A4_MXFP4")
    with (
        envs.SGLANG_NPU_ENABLE_MXFP4_NZ.override(True),
        envs.SGLANG_NPU_DISABLE_ACL_FORMAT_WEIGHT.override(True),
    ):
        with pytest.raises(ValueError, match="conflicts"):
            scheme.process_weights_after_loading(torch.nn.Module())


@pytest.mark.parametrize("wan,nz", [(False, False), (True, False), (True, True)])
@pytest.mark.parametrize("activation", ["W4A4", "W4A8"])
def test_linear_operator_contract(monkeypatch, wan, nz, activation):
    monkeypatch.setattr(linear_method_npu, "_get_float4_e2m1fn_x2_dtype", lambda: 296)
    quant = Mock(return_value=(torch.ones(6, 32), torch.ones(6, 1, 2)))
    matmul = Mock(return_value=torch.zeros(6, 2, dtype=torch.bfloat16))
    monkeypatch.setattr(torch.ops.npu, "npu_dynamic_mx_quant", quant, raising=False)
    monkeypatch.setattr(torch.ops.npu, "npu_quant_matmul", matmul, raising=False)
    layer = SimpleNamespace(
        weight=torch.ones(32, 2, dtype=torch.uint8),
        weight_scale=torch.ones(1, 2, 2, dtype=torch.uint8),
        use_mul_scale=True,
        mul_scale=torch.full((64,), 2.0),
    )
    x, bias = torch.ones(2, 3, 64, dtype=torch.bfloat16), torch.ones(2)
    if nz:
        layer.weight_w4a8 = layer.weight + 1
    if wan:
        scheme = ModelSlimMXFP4Scheme({}, "linear", "W4A4_MXFP4")
        layer.mxfp4_quant_kwargs = {"scale_alg": 2, "dst_type_max": 7.25}
        with set_forward_context(
            2 if activation == "W4A4" else 0,
            None,
            SimpleNamespace(num_inference_steps=40),
        ):
            output = scheme.apply_weights(layer, x, bias)
    else:
        kernel = (
            linear_method_npu.NPUSingleLevelMXFP4OfflineLinearMethod
            if activation == "W4A4"
            else linear_method_npu.NPUMXFP4W4A8OfflineLinearMethod
        )()
        output = kernel.apply(layer, x, bias)
    assert output.shape == (2, 3, 2)
    assert torch.equal(quant.call_args.args[0], x.reshape(6, 64) * (2 if wan else 1))
    options = quant.call_args.kwargs
    args = matmul.call_args.kwargs
    assert matmul.call_args.args[1] is (
        layer.weight_w4a8 if nz and activation == "W4A8" else layer.weight
    )
    assert args["x2_dtype"] == 296
    assert args["scale_dtype"] == args["pertoken_scale_dtype"] == torch.float8_e8m0fnu
    if activation == "W4A4":
        assert options == {
            "dst_type": 296,
            "round_mode": "rint" if wan else "round",
            **({"scale_alg": 2, "dst_type_max": 7.25} if wan else {}),
        }
        assert args["x1_dtype"] == 296 and args["group_sizes"] == [1, 1, 32]
    else:
        assert options == {"dst_type": torch.float8_e4m3fn}
        assert args["group_sizes"] == [0, 0, 32]
    wan_bias = wan and activation == "W4A8"
    assert args["bias"].dtype == (torch.bfloat16 if wan_bias else torch.float32)
    assert args["bias"].shape == ((1, 2) if wan_bias else (2,))
