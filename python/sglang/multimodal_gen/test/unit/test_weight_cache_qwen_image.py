# SPDX-License-Identifier: Apache-2.0
"""Qwen adapter decisions without loading checkpoint tensors."""

import json
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch
from safetensors.torch import save_file

from sglang.multimodal_gen.configs.pipeline_configs.qwen_image import (
    QwenImagePipelineConfig,
)
from sglang.multimodal_gen.runtime.pipelines.qwen_image import QwenImagePipeline
from sglang.multimodal_gen.runtime.pipelines_core.prepare import prepare_pipeline
from sglang.multimodal_gen.runtime.server_args import ServerArgs
from sglang.multimodal_gen.runtime.weight_cache import adapters
from sglang.multimodal_gen.runtime.weight_cache.adapters import (
    common,
    dit_qwen_image,
    dit_wan,
)


@pytest.fixture
def prepared_qwen(tmp_path):
    from sglang.multimodal_gen.runtime.models.registry import ModelRegistry

    index = {
        "_class_name": "QwenImagePipeline",
        "transformer": ["diffusers", "QwenImageTransformer2DModel"],
        "text_encoder": ["transformers", "Qwen2_5_VLForConditionalGeneration"],
        "tokenizer": ["transformers", "Qwen2Tokenizer"],
        "vae": ["diffusers", "AutoencoderKLQwenImage"],
        "scheduler": ["diffusers", "FlowMatchEulerDiscreteScheduler"],
    }
    (tmp_path / "model_index.json").write_text(json.dumps(index))
    component = tmp_path / "transformer"
    component.mkdir()
    (component / "config.json").write_text(
        json.dumps(
            {
                "_class_name": "QwenImageTransformer2DModel",
                **dit_qwen_image.EXPECTED_CONFIG,
            }
        )
    )
    save_file({}, component / "diffusion_pytorch_model.safetensors")
    ModelRegistry.resolve_model_cls("QwenImageTransformer2DModel")
    with patch.object(ServerArgs, "_adjust_network_ports"):
        args = ServerArgs(
            model_path=str(tmp_path),
            pipeline_config=QwenImagePipelineConfig(),
            weight_cache_mode="client",
            performance_mode="manual",
        )
    with (
        patch(
            "sglang.multimodal_gen.runtime.pipelines_core.prepare.maybe_download_model",
            return_value=str(tmp_path),
        ),
        patch(
            "sglang.multimodal_gen.runtime.platforms.current_platform.is_cuda",
            return_value=True,
        ),
    ):
        yield args


def test_both_adapters_use_the_same_ordinary_and_meta_loaders():
    assert dit_qwen_image.load_ordinary is dit_wan.load_ordinary is common.load_ordinary
    assert dit_qwen_image.build_meta is dit_wan.build_meta is common.build_meta


def test_qwen_preparation_is_pure_and_freezes_adapter(prepared_qwen):
    from sglang.multimodal_gen.runtime.loader.component_loaders import (
        transformer_loader,
    )

    with (
        patch.object(torch.nn.Module, "__init__", side_effect=AssertionError("module")),
        patch.object(
            torch.cuda,
            "_lazy_init",
            side_effect=AssertionError("CUDA"),
        ),
        patch.object(
            transformer_loader,
            "get_local_torch_device",
            side_effect=AssertionError("rank"),
        ),
    ):
        prepared = prepare_pipeline(QwenImagePipeline, prepared_qwen, required=True)
    assert prepared.adapter_id == dit_qwen_image.ADAPTER_ID
    assert prepared_qwen.model_paths == {}
    with patch.object(
        adapters, "for_pipeline", side_effect=AssertionError("rediscovery")
    ):
        assert prepared.adapter is dit_qwen_image
    ordinary = prepare_pipeline(
        QwenImagePipeline, prepared_qwen.resolve_variant(weight_cache_mode="off")
    )
    assert prepared.specs == ordinary.specs
    assert prepared.adapter.fingerprint_fields(
        prepared.transformer
    ) == ordinary.adapter.fingerprint_fields(ordinary.transformer)


@pytest.mark.parametrize(
    "variant",
    [
        "class",
        "config",
        "resolved_config",
        "conditioning",
        "quant",
        "dtype",
        "attention",
        "cpu",
        "tp",
        "compile",
    ],
)
def test_qwen_adapter_rejects_unverified_representations(prepared_qwen, variant):
    recipe = prepare_pipeline(
        QwenImagePipeline, prepared_qwen, required=True
    ).transformer.thaw()
    attention = "fa"
    if variant == "class":
        recipe.model_cls = type("QwenImageTransformer2DModel", (), {})
    elif variant == "config":
        recipe.init_params["hf_config"]["unverified_extension"] = True
    elif variant == "resolved_config":
        recipe.init_params["config"].arch_config.joint_attention_dim = 1024
    elif variant == "conditioning":
        recipe.init_params["config"].arch_config.zero_cond_t = True
    elif variant == "quant":
        recipe.quant_spec.post_load_hooks.append(object())
    elif variant == "dtype":
        recipe.quant_spec = replace(recipe.quant_spec, param_dtype=torch.float16)
    elif variant == "attention":
        attention = "torch_sdpa"
    elif variant == "cpu":
        recipe.component_starts_on_cpu = True
    elif variant == "tp":
        recipe.server_args.tp_size = 2
    elif variant == "compile":
        recipe.server_args.enable_torch_compile = True
    frozen = Mock()
    frozen.thaw.return_value = recipe
    with pytest.raises(ValueError):
        dit_qwen_image.validate_supported(
            frozen, pipeline_name="QwenImagePipeline", attention=attention
        )


def test_forced_pipeline_does_not_admit_edit_checkpoint(prepared_qwen):
    from pathlib import Path

    path = Path(prepared_qwen.model_path) / "model_index.json"
    index = json.loads(path.read_text())
    index["_class_name"] = "QwenImageEditPipeline"
    path.write_text(json.dumps(index))
    with pytest.raises(ValueError, match="admitted single-transformer"):
        prepare_pipeline(QwenImagePipeline, prepared_qwen, required=True)
    assert (
        prepare_pipeline(
            QwenImagePipeline, prepared_qwen.resolve_variant(weight_cache_mode="off")
        )
        is None
    )


def test_unknown_prepared_adapter_fails_closed():
    with pytest.raises(ValueError, match="Unknown prepared"):
        adapters.by_id("unverified")


@pytest.mark.parametrize("enabled", [None, *range(5)])
def test_import_finalization_rejects_quantized_derived_state(enabled):
    names = (
        "_enable_nvfp4_resnorm_quant",
        "_fp8_img_attn_norm_quant",
        "_fp8_txt_attn_norm_quant",
        "_fp8_img_mlp_norm_quant",
        "_fp8_txt_mlp_norm_quant",
    )
    block = SimpleNamespace(
        **{name: index == enabled for index, name in enumerate(names)}
    )
    model = SimpleNamespace(
        transformer_blocks=[block],
        post_load_weights=Mock(side_effect=AssertionError("Repeated weight transform")),
    )
    with patch.object(
        dit_qwen_image, "finalize_loaded_model", return_value=model
    ) as finalize:
        if enabled is None:
            assert dit_qwen_image.finalize_after_import(model) is model
            finalize.assert_called_once_with(model)
        else:
            with pytest.raises(ValueError, match="derived state"):
                dit_qwen_image.finalize_after_import(model)
            finalize.assert_not_called()
    model.post_load_weights.assert_not_called()
