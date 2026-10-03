# SPDX-License-Identifier: Apache-2.0
"""FastH3 8-Step V2 (VSA-distilled MiniMax-H3) registration and admission contracts."""

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
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.release_metadata import (
    MiniMaxH3ReleaseMetadata,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.time_request import (
    minimax_h3_time_shift_sigmas,
)
from sglang.multimodal_gen.test.single_test_file.component_accuracy.utils import (
    ensure_distributed_env_defaults,
)

FASTH3_MODEL_ID = "FastVideo/FastVideo-FastH3-8-Step-V2"
DMD_STEPS = (999, 874, 749, 624, 500, 375, 250, 125)


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
        "/cache/materialized_models/FastVideo__FastVideo-FastH3-8-Step-V2-0123abcd"
    )
    assert get_model_info(materialized).sampling_param_cls is FastH3SamplingParams


def test_fasth3_sampling_defaults_and_task_rejection() -> None:
    params = FastH3SamplingParams(prompt="p")
    assert params.num_inference_steps == 9
    assert params.guidance_scale == 1.0

    with pytest.raises(ValueError, match="exactly nine sigma grid points"):
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


def test_fasth3_dmd_schedule_matches_trained_rungs() -> None:
    """The distilled rungs are shifted exactly once, as FastVideo serves them, and
    a wrong step count or non-decreasing rungs are rejected."""
    metadata = MiniMaxH3ReleaseMetadata.from_model_index(
        {
            "_minimax_h3": {
                "schema_version": 1,
                "partition": "fl2va",
                "tasks": ["t2va"],
                "sigma_shift_scales": {"video": 10.0, "audio": 3.0},
                "dmd_denoising_steps": list(DMD_STEPS),
            }
        }
    )
    assert metadata.dmd_denoising_steps == DMD_STEPS
    # FastVideo's _set_dmd_schedule output for the video / audio shifts 10 / 3
    served = {
        10.0: [
            *(0.9999, 0.985788, 0.967575, 0.943168, 0.909091),
            *(0.857143, 0.769231, 0.588235, 0.0),
        ],
        3.0: [
            *(0.999666, 0.954148, 0.89952, 0.83274, 0.75),
            *(0.642857, 0.5, 0.3, 0.0),
        ],
    }
    for shift, expected in served.items():
        sigmas = minimax_h3_time_shift_sigmas(
            num_steps=9, shift_scale=shift, dmd_steps=DMD_STEPS
        )
        assert sigmas == pytest.approx(expected, abs=1e-6)
    with pytest.raises(ValueError, match="8 DiT forwards"):
        minimax_h3_time_shift_sigmas(num_steps=5, shift_scale=10.0, dmd_steps=DMD_STEPS)
    with pytest.raises(ValueError, match="strictly decreasing"):
        MiniMaxH3ReleaseMetadata.from_model_index(
            {
                "_minimax_h3": {
                    "schema_version": 1,
                    "partition": "fl2va",
                    "tasks": ["t2va"],
                    "sigma_shift_scales": {"video": 10.0, "audio": 3.0},
                    "dmd_denoising_steps": [999, 999],
                }
            }
        )


def test_fasth3_pipeline_config_gates_and_rejections() -> None:
    config = FastH3PipelineConfig()
    assert config.vsa_sparsity == 0.8
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
    model = SimpleNamespace(
        arch=SimpleNamespace(adaln_affine_input_dim=None),
        _adaln_precomputed=False,
    )
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
