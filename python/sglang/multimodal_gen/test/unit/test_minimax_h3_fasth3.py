# SPDX-License-Identifier: Apache-2.0
"""FastH3 (4-step VSA-distilled MiniMax-H3) registration and admission contracts."""

from __future__ import annotations

import re
from types import SimpleNamespace

import pytest
import torch

from sglang.multimodal_gen.configs.pipeline_configs.minimax_h3 import (
    FastH3PipelineConfig,
    MiniMaxH3PipelineConfig,
)
from sglang.multimodal_gen.configs.sample.minimax_h3 import FastH3SamplingParams
from sglang.multimodal_gen.registry import (
    get_model_info,
    get_non_diffusers_pipeline_name,
)
from sglang.multimodal_gen.runtime.distributed.parallel_state import (
    maybe_init_distributed_environment_and_model_parallel,
    model_parallel_is_initialized,
)
from sglang.multimodal_gen.runtime.layers.linear import UnquantizedLinearMethod
from sglang.multimodal_gen.runtime.layers.quantization.fp8 import Fp8Config
from sglang.multimodal_gen.runtime.models.dits.minimax_h3 import MiniMaxH3DiTModel
from sglang.multimodal_gen.test.single_test_file.component_accuracy.utils import (
    ensure_distributed_env_defaults,
)

FASTH3_MODEL_ID = "FastVideo/FastVideo-FastH3-4-step-Preview-v1-VSA-DataFree"


def _ensure_single_process_parallel_runtime() -> None:
    if model_parallel_is_initialized():
        return
    ensure_distributed_env_defaults()
    maybe_init_distributed_environment_and_model_parallel(tp_size=1, sp_size=1)


def test_registry_resolves_fasth3_configs() -> None:
    info = get_model_info(FASTH3_MODEL_ID)
    assert info.sampling_param_cls is FastH3SamplingParams
    assert info.pipeline_config_cls is FastH3PipelineConfig
    assert get_non_diffusers_pipeline_name(FASTH3_MODEL_ID) == "FastH3Pipeline"
    materialized = (
        "/cache/materialized_models/"
        "FastVideo__FastVideo-FastH3-4-step-Preview-v1-VSA-DataFree-0123abcd"
    )
    assert get_model_info(materialized).sampling_param_cls is FastH3SamplingParams


def test_fasth3_sampling_defaults_and_task_rejection() -> None:
    params = FastH3SamplingParams(prompt="p")
    assert params.num_inference_steps == 5
    assert params.guidance_scale == 1.0

    with pytest.raises(ValueError, match="exactly five sigma grid points"):
        FastH3SamplingParams(prompt="p", num_inference_steps=50)

    with pytest.raises(ValueError, match="distilled for t2va only"):
        FastH3SamplingParams(
            prompt="p",
            task="fl2va",
            conditions=[{"type": "image", "uri": "x.png", "role": "first_frame"}],
            target={
                "short_edge": 768,
                "aspect_ratio": "16:9",
                "duration_seconds": 5.0,
            },
        )


def test_fasth3_pipeline_config_gates_and_rejections() -> None:
    config = FastH3PipelineConfig()
    assert config.dit_config.arch_config.has_gate_compress
    assert not MiniMaxH3PipelineConfig().dit_config.arch_config.has_gate_compress
    mapping = config.dit_config.arch_config.param_names_mapping
    source = "transformer_blocks.7.attn.to_gate_compress.weight"
    targets = [
        re.sub(pattern, target if isinstance(target, str) else target[0], source)
        for pattern, target in mapping.items()
        if re.match(pattern, source)
    ]
    assert targets == ["blocks.7.attn.to_gate_compress.weight"]

    with pytest.raises(ValueError, match="--model-variant does not apply"):
        config.validate_server_args(SimpleNamespace(model_variant="ref2va"))
    with pytest.raises(ValueError, match="no.*audited high-quality deployment"):
        config.validate_quality_deployment(server_args=None)


def test_fasth3_lora_bundle_is_rejected_loudly() -> None:
    model = SimpleNamespace(arch=SimpleNamespace(adaln_affine_input_dim=None))
    plain = {
        "blocks.0.attn.qkv_proj.lora_A": torch.zeros(3, 64, 8),
        "blocks.0.attn.qkv_proj.lora_B": torch.zeros(3, 8, 64),
    }
    assert MiniMaxH3DiTModel.prepare_lora_adapter(model, dict(plain)) == plain

    bundle = dict(plain)
    bundle["blocks.0.attn.qkv_proj.diff"] = torch.zeros(3, 64, 64)
    bundle["audio_patch_proj.diff_b"] = torch.zeros(64)
    bundle["blocks.0.attn.to_gate_compress.set_weight"] = torch.zeros(64, 64)
    with pytest.raises(ValueError, match="3 non-LoRA tensors.*set_weight"):
        MiniMaxH3DiTModel.prepare_lora_adapter(model, bundle)


def test_fasth3_gates_stay_bf16_under_runtime_quantization() -> None:
    _ensure_single_process_parallel_runtime()
    with torch.device("meta"):
        model = MiniMaxH3DiTModel(
            config=FastH3PipelineConfig().dit_config,
            hf_config={},
            quant_config=Fp8Config(),
        )

    attn = model.blocks[0].attn
    assert not isinstance(attn.qkv_proj.quant_method, UnquantizedLinearMethod)
    assert isinstance(attn.to_gate_compress.quant_method, UnquantizedLinearMethod)
    assert attn.to_gate_compress.weight.dtype == torch.bfloat16
    assert attn.to_gate_compress.weight.missing_param_init == "error"
    assert model.token_refiner.blocks[0].attn.to_gate_compress is None
