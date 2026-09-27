# SPDX-License-Identifier: Apache-2.0
"""Qwen state-contract decisions without loading checkpoint tensors."""

import json
import multiprocessing as mp
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch
from safetensors.torch import save_file

from sglang.multimodal_gen.configs.pipeline_configs.qwen_image import (
    QwenImagePipelineConfig,
)
from sglang.multimodal_gen.runtime.loader import native_dit_state
from sglang.multimodal_gen.runtime.loader.native_dit_state import QWEN_IMAGE
from sglang.multimodal_gen.runtime.pipelines.qwen_image import QwenImagePipeline
from sglang.multimodal_gen.runtime.pipelines_core.prepare import prepare_pipeline
from sglang.multimodal_gen.runtime.server_args import ServerArgs
from sglang.multimodal_gen.runtime.weight_cache import policy


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
                **QWEN_IMAGE.expected_config,
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


def _auto_plan_in_fresh_process(model_path, free_gb, connection):
    from sglang.multimodal_gen.runtime.platforms import current_platform
    from sglang.multimodal_gen.runtime.weight_cache.identity import compatibility_plan

    # Inject the hardware observation, not the tuning result or residency plan.
    with (
        patch.object(current_platform, "is_cuda", return_value=True),
        patch.object(current_platform, "is_cpu", return_value=False),
        patch.object(
            current_platform, "get_available_gpu_memory", return_value=free_gb
        ) as probe,
        patch.object(current_platform, "get_device_uuid", return_value="GPU-planned"),
        patch.object(ServerArgs, "_adjust_network_ports"),
        patch(
            "sglang.multimodal_gen.runtime.weight_cache.identity.environment_identity",
            return_value={"test_build": "fixed"},
        ),
        patch(
            "sglang.multimodal_gen.runtime.pipelines_core.prepare.maybe_download_model",
            return_value=model_path,
        ),
    ):
        args = ServerArgs(
            model_path=model_path,
            pipeline_config=QwenImagePipelineConfig(),
            performance_mode="auto",
            weight_cache_mode="client",
            weight_cache_allow_weak_checkpoint_identity=True,
        )
        prepared = prepare_pipeline(QwenImagePipeline, args, required=True)
        connection.send(
            {
                "compatibility": compatibility_plan(prepared, args).to_dict(),
                "execution": prepared.execution_plan,
                "dit": args.residency_mode("transformer"),
                "fsdp": args.should_use_fsdp_for_component("transformer"),
                "probe_count": probe.call_count,
            }
        )
    connection.close()


def test_auto_tuning_across_bare_owner_worker_memory_states(prepared_qwen):
    context = mp.get_context("spawn")
    results = []
    # These model three separate process observations before/after resident
    # owner allocations. Actual tuner runs in every process, unchanged.
    for free_gb in (80, 40, 30):
        parent, child = context.Pipe(duplex=False)
        process = context.Process(
            target=_auto_plan_in_fresh_process,
            args=(prepared_qwen.model_path, free_gb, child),
        )
        process.start()
        child.close()
        try:
            assert parent.poll(90), "planning child timed out"
            results.append(parent.recv())
            process.join(10)
            assert process.exitcode == 0
        finally:
            if process.is_alive():
                process.kill()
                process.join(10)
            parent.close()
    assert all(result["probe_count"] > 0 for result in results)
    assert all(result["dit"] == "resident" and not result["fsdp"] for result in results)
    assert (
        results[0]["compatibility"]
        == results[1]["compatibility"]
        == results[2]["compatibility"]
    )
    assert results[0]["execution"] != results[1]["execution"]


def test_qwen_preparation_is_pure_and_freezes_contract(prepared_qwen):
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
    assert prepared.component("transformer").contract is QWEN_IMAGE
    assert prepared_qwen.model_paths == {}
    with patch.object(
        policy, "for_pipeline", side_effect=AssertionError("rediscovery")
    ):
        assert prepared.component("transformer").contract is QWEN_IMAGE
    ordinary = prepare_pipeline(
        QwenImagePipeline, prepared_qwen.resolve_variant(weight_cache_mode="off")
    )
    assert prepared.specs == ordinary.specs
    assert (
        prepared.component("transformer").fingerprint_fields()
        == ordinary.component("transformer").fingerprint_fields()
    )


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
def test_qwen_contract_rejects_unverified_representations(prepared_qwen, variant):
    recipe = (
        prepare_pipeline(QwenImagePipeline, prepared_qwen, required=True)
        .component("transformer")
        .recipe.thaw()
    )
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
        QWEN_IMAGE.validate_supported(frozen, attention=attention)


def test_forced_pipeline_does_not_admit_edit_checkpoint(prepared_qwen):
    from pathlib import Path

    path = Path(prepared_qwen.model_path) / "model_index.json"
    index = json.loads(path.read_text())
    index["_class_name"] = "QwenImageEditPipeline"
    path.write_text(json.dumps(index))
    with pytest.raises(ValueError, match="audited pipeline component layout"):
        prepare_pipeline(QwenImagePipeline, prepared_qwen, required=True)
    assert (
        prepare_pipeline(
            QwenImagePipeline, prepared_qwen.resolve_variant(weight_cache_mode="off")
        )
        is None
    )


def test_unbound_pipeline_with_same_model_is_not_admitted():
    class UnverifiedPipeline(QwenImagePipeline):
        pass

    assert policy.for_pipeline(UnverifiedPipeline) is None


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
        native_dit_state, "finalize_loaded_model", return_value=model
    ) as finalize:
        if enabled is None:
            assert QWEN_IMAGE.finalize_after_import(model) is model
            finalize.assert_called_once_with(model)
        else:
            with pytest.raises(ValueError, match="derived state"):
                QWEN_IMAGE.finalize_after_import(model)
            finalize.assert_not_called()
    model.post_load_weights.assert_not_called()
