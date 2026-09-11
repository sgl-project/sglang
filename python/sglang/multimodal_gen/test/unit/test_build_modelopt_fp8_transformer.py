# SPDX-License-Identifier: Apache-2.0

import json

import pytest
import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file

from sglang.multimodal_gen.tools.build_modelopt_fp8_transformer import (
    build_fp8_scale_map,
    build_modelopt_fp8_transformer,
)


def _write_llada_modelopt_export(
    tmp_path,
    source_tensors,
    backbone_state,
    *,
    quantization_config=None,
    class_name="LLaDAImageTransformer2DModel",
):
    source_dir = tmp_path / "source"
    output_dir = tmp_path / "output"
    source_dir.mkdir()
    source_config = {
        "_class_name": class_name,
        "quantization_config": {
            "quant_method": "modelopt",
            "quant_algo": "FP8",
            **(quantization_config or {}),
        },
    }
    (source_dir / "config.json").write_text(json.dumps(source_config))
    save_file(source_tensors, source_dir / "model.safetensors")

    backbone_path = tmp_path / "backbone.pt"
    torch.save({"model_state_dict": backbone_state}, backbone_path)
    return source_dir, output_dir, backbone_path


def test_dynamic_activation_scale_map_only_requires_weight_amax():
    state_dict = {
        "layers.0.attention.to_qkv.weight_quantizer._amax": torch.tensor(224.0),
    }

    scale_map = build_fp8_scale_map(
        state_dict,
        maxbound=448.0,
        require_input_scale=False,
    )

    assert set(scale_map) == {"layers.0.attention.to_qkv.weight"}
    assert set(scale_map["layers.0.attention.to_qkv.weight"]) == {"weight_scale"}
    torch.testing.assert_close(
        scale_map["layers.0.attention.to_qkv.weight"]["weight_scale"],
        torch.tensor([0.5]),
    )


def test_static_activation_scale_map_still_requires_input_amax():
    weight_amax_only = {
        "layers.0.attention.to_qkv.weight_quantizer._amax": torch.tensor(224.0),
    }
    assert build_fp8_scale_map(weight_amax_only, maxbound=448.0) == {}

    state_dict = {
        **weight_amax_only,
        "layers.0.attention.to_qkv.input_quantizer._amax": torch.tensor(112.0),
    }
    scale_map = build_fp8_scale_map(state_dict, maxbound=448.0)

    assert set(scale_map["layers.0.attention.to_qkv.weight"]) == {
        "weight_scale",
        "input_scale",
    }
    torch.testing.assert_close(
        scale_map["layers.0.attention.to_qkv.weight"]["input_scale"],
        torch.tensor([0.25]),
    )


def test_llada_conversion_writes_dynamic_fp8_without_input_scales(tmp_path):
    source_dir, output_dir, backbone_path = _write_llada_modelopt_export(
        tmp_path,
        {
            "layers.0.attention.to_q.weight": torch.ones(2, 2),
            "layers.0.attention.to_q.input_scale": torch.tensor([0.25]),
        },
        {
            "layers.0.attention.to_qkv.weight_quantizer._amax": torch.tensor(224.0),
        },
    )

    stats = build_modelopt_fp8_transformer(
        modelopt_hf_dir=str(source_dir),
        modelopt_backbone_ckpt=str(backbone_path),
        output_dir=str(output_dir),
        model_type="llada-image",
    )

    output_config = json.loads((output_dir / "config.json").read_text())
    assert output_config["quantization_config"] == {
        "quant_method": "fp8",
        "activation_scheme": "dynamic",
        "ignored_layers": [],
    }
    output_tensors = load_file(output_dir / "model.safetensors")
    assert output_tensors["layers.0.attention.to_q.weight"].dtype == torch.float8_e4m3fn
    assert "layers.0.attention.to_q.weight_scale" in output_tensors
    assert "layers.0.attention.to_q.input_scale" not in output_tensors
    assert stats["added_scale_tensors"] == 1


@pytest.mark.parametrize(
    ("module_prefix", "source_modules", "fused_module"),
    [
        ("layers.0.attention", ("to_q", "to_k", "to_v"), "to_qkv"),
        ("layers.0.feed_forward", ("w1", "w3"), "w13"),
    ],
)
def test_llada_conversion_rejects_mismatched_fused_weight_scales(
    tmp_path, module_prefix, source_modules, fused_module
):
    source_dir, output_dir, backbone_path = _write_llada_modelopt_export(
        tmp_path,
        {
            f"{module_prefix}.{module}.weight": torch.ones(2, 2)
            for module in source_modules
        },
        {
            f"{module_prefix}.{module}.weight_quantizer._amax": torch.tensor(
                224.0 if index == 0 else 112.0
            )
            for index, module in enumerate(source_modules)
        },
    )

    with pytest.raises(
        ValueError,
        match=(
            "LLaDA-Image FP8 conversion currently requires identical weight scales"
            f".*{fused_module}"
        ),
    ):
        build_modelopt_fp8_transformer(
            modelopt_hf_dir=str(source_dir),
            modelopt_backbone_ckpt=str(backbone_path),
            output_dir=str(output_dir),
            model_type="llada-image",
        )


def test_llada_conversion_rejects_static_activation_quantization(tmp_path):
    source_dir, output_dir, backbone_path = _write_llada_modelopt_export(
        tmp_path,
        {"layers.0.attention.to_q.weight": torch.ones(2, 2)},
        {
            "layers.0.attention.to_qkv.weight_quantizer._amax": torch.tensor(224.0),
            "layers.0.attention.to_qkv.input_quantizer._amax": torch.tensor(112.0),
        },
    )

    with pytest.raises(
        ValueError,
        match=(
            "LLaDA-Image FP8 conversion currently supports only dynamic "
            "activation quantization"
        ),
    ):
        build_modelopt_fp8_transformer(
            modelopt_hf_dir=str(source_dir),
            modelopt_backbone_ckpt=str(backbone_path),
            output_dir=str(output_dir),
            model_type="llada-image",
            activation_scheme="static",
        )


@pytest.mark.parametrize(
    ("metadata", "requested", "expected"),
    [
        ({}, "auto", "static"),
        ({"activation_scheme": "dynamic"}, "auto", "dynamic"),
        ({"activation_scheme": "static"}, "auto", "static"),
        (
            {"config_groups": {"group_0": {"input_activations": {"dynamic": True}}}},
            "auto",
            "dynamic",
        ),
        (
            {"config_groups": {"group_0": {"input_activations": {"dynamic": False}}}},
            "auto",
            "static",
        ),
        (
            {"config_groups": {"group_0": {"weights": {"dynamic": True}}}},
            "auto",
            "static",
        ),
        (
            {
                "config_groups": {
                    "group_0": {"input_activations": {"dynamic": False}},
                    "group_1": {"input_activations": {"dynamic": True}},
                }
            },
            "dynamic",
            "dynamic",
        ),
        (
            {
                "config_groups": {
                    "group_0": {"input_activations": {"dynamic": False}},
                    "group_1": {"input_activations": {"dynamic": True}},
                }
            },
            "static",
            "static",
        ),
        ({"activation_scheme": "static"}, "dynamic", "dynamic"),
        ({"activation_scheme": "dynamic"}, "static", "static"),
        (
            {"config_groups": {"group_0": {"input_activations": {"dynamic": True}}}},
            "static",
            "static",
        ),
    ],
)
def test_conversion_resolves_activation_metadata(
    tmp_path, metadata, requested, expected
):
    source_dir, output_dir, backbone_path = _write_llada_modelopt_export(
        tmp_path,
        {"layers.0.attention.to_q.weight": torch.ones(2, 2)},
        {
            "layers.0.attention.to_q.weight_quantizer._amax": torch.tensor(224.0),
            "layers.0.attention.to_q.input_quantizer._amax": torch.tensor(112.0),
        },
        quantization_config=metadata,
        class_name="ExampleTransformer2DModel",
    )
    source_config = (source_dir / "config.json").read_bytes()
    build_modelopt_fp8_transformer(
        modelopt_hf_dir=str(source_dir),
        modelopt_backbone_ckpt=str(backbone_path),
        output_dir=str(output_dir),
        model_type="none",
        activation_scheme=requested,
    )
    quant_config = json.loads((output_dir / "config.json").read_text())[
        "quantization_config"
    ]
    assert quant_config.get("activation_scheme", "static") == expected
    for group in quant_config.get("config_groups", {}).values():
        if group.get("input_activations") is not None:
            assert group["input_activations"]["dynamic"] is (expected == "dynamic")
    tensors = load_file(output_dir / "model.safetensors")
    assert ("layers.0.attention.to_q.input_scale" in tensors) is (expected == "static")
    with safe_open(output_dir / "model.safetensors", framework="pt") as checkpoint:
        assert json.loads(checkpoint.metadata()["quantization_config"]) == quant_config
    assert (source_dir / "config.json").read_bytes() == source_config


@pytest.mark.parametrize("requested", ["auto", "dynamic"])
def test_llada_static_source_requires_explicit_dynamic_output(tmp_path, requested):
    source_dir, output_dir, backbone_path = _write_llada_modelopt_export(
        tmp_path,
        {"layers.0.attention.to_q.weight": torch.ones(2, 2)},
        {"layers.0.attention.to_qkv.weight_quantizer._amax": torch.tensor(224.0)},
        quantization_config={
            "config_groups": {"group_0": {"input_activations": {"dynamic": False}}}
        },
    )
    kwargs = dict(
        modelopt_hf_dir=str(source_dir),
        modelopt_backbone_ckpt=str(backbone_path),
        output_dir=str(output_dir),
        model_type="llada-image",
        activation_scheme=requested,
    )
    if requested == "auto":
        with pytest.raises(ValueError, match="set activation_scheme='dynamic'"):
            build_modelopt_fp8_transformer(**kwargs)
        assert not output_dir.exists()
    else:
        build_modelopt_fp8_transformer(**kwargs)
        quant_config = json.loads((output_dir / "config.json").read_text())[
            "quantization_config"
        ]
        assert quant_config["activation_scheme"] == "dynamic"
        assert "layers.0.attention.to_q.input_scale" not in load_file(
            output_dir / "model.safetensors"
        )


@pytest.mark.parametrize(
    "metadata",
    [
        {"activation_scheme": "invalid"},
        {"config_groups": {"group_0": {"input_activations": {"dynamic": "false"}}}},
        {
            "config_groups": {
                "group_0": {"input_activations": {"dynamic": False}},
                "group_1": {"input_activations": {"dynamic": True}},
            }
        },
        {
            "activation_scheme": "dynamic",
            "config_groups": {"group_0": {"input_activations": {"dynamic": False}}},
        },
    ],
)
def test_auto_rejects_invalid_or_conflicting_activation_metadata(tmp_path, metadata):
    source_dir, output_dir, backbone_path = _write_llada_modelopt_export(
        tmp_path,
        {"layers.0.attention.to_q.weight": torch.ones(2, 2)},
        {"layers.0.attention.to_qkv.weight_quantizer._amax": torch.tensor(224.0)},
        quantization_config=metadata,
    )
    with pytest.raises(ValueError, match="activation"):
        build_modelopt_fp8_transformer(
            modelopt_hf_dir=str(source_dir),
            modelopt_backbone_ckpt=str(backbone_path),
            output_dir=str(output_dir),
            model_type="llada-image",
        )
    assert not output_dir.exists()


@pytest.mark.parametrize("requested", ["static", "dynamic"])
@pytest.mark.parametrize(
    "metadata",
    [
        {"config_groups": None},
        {"config_groups": {"group_0": None}},
        {"config_groups": {"group_0": {"input_activations": "invalid"}}},
    ],
)
def test_explicit_output_rejects_malformed_metadata_before_writing(
    tmp_path, requested, metadata
):
    source_dir, output_dir, backbone_path = _write_llada_modelopt_export(
        tmp_path,
        {"layers.0.attention.to_q.weight": torch.ones(2, 2)},
        {"layers.0.attention.to_q.weight_quantizer._amax": torch.tensor(224.0)},
        quantization_config=metadata,
        class_name="ExampleTransformer2DModel",
    )
    with pytest.raises(ValueError, match="ModelOpt.*dictionary"):
        build_modelopt_fp8_transformer(
            modelopt_hf_dir=str(source_dir),
            modelopt_backbone_ckpt=str(backbone_path),
            output_dir=str(output_dir),
            model_type="none",
            activation_scheme=requested,
        )
    assert not output_dir.exists()
