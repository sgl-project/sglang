# SPDX-License-Identifier: Apache-2.0
"""FastH3 (4-step DMD2-distilled MiniMax-H3) registration and admission
contracts: t2va-only task coverage, five-point schedule defaults, trained
VSA gate module, and the bundled overlay release metadata."""

from __future__ import annotations

import json
import os
import re

import pytest
import torch

from sglang.multimodal_gen.configs.pipeline_configs.minimax_h3 import (
    FastH3PipelineConfig,
    MiniMaxH3PipelineConfig,
)
from sglang.multimodal_gen.configs.sample.minimax_h3 import (
    FastH3SamplingParams,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.release_metadata import (
    MiniMaxH3ReleaseMetadata,
)

FASTH3_MODEL_ID = "FastVideo/FastVideo-FastH3-4-step-Preview-v1-VSA-DataFree"

_OVERLAY_DIR = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        "..",
        "..",
        "model_overlays",
        "fasth3_4step_preview_v1_vsa_datafree",
    )
)


def test_registry_resolves_fasth3_configs() -> None:
    from sglang.multimodal_gen.registry import (
        get_model_info,
        get_non_diffusers_pipeline_name,
    )

    info = get_model_info(FASTH3_MODEL_ID)
    assert info.sampling_param_cls is FastH3SamplingParams
    assert info.pipeline_config_cls is FastH3PipelineConfig
    assert get_non_diffusers_pipeline_name(FASTH3_MODEL_ID) == "FastH3Pipeline"
    # The materialized overlay directory name must resolve identically.
    materialized = (
        "/cache/materialized_models/"
        "FastVideo__FastVideo-FastH3-4-step-Preview-v1-VSA-DataFree-0123abcd"
    )
    info = get_model_info(materialized)
    assert info.sampling_param_cls is FastH3SamplingParams


def test_fasth3_sampling_defaults_and_task_rejection() -> None:
    params = FastH3SamplingParams(prompt="p")
    assert params.num_inference_steps == 5
    assert params.guidance_scale == 1.0

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


def test_fasth3_pipeline_config_gates_and_quality() -> None:
    config = FastH3PipelineConfig()
    assert config.dit_config.arch_config.has_gate_compress
    assert not MiniMaxH3PipelineConfig().dit_config.arch_config.has_gate_compress

    with pytest.raises(ValueError, match="no.*audited high-quality deployment"):
        config.validate_quality_deployment(server_args=None)


def test_fasth3_rejects_model_variant() -> None:
    from types import SimpleNamespace

    config = FastH3PipelineConfig()
    with pytest.raises(ValueError, match="--model-variant does not apply"):
        config.validate_server_args(SimpleNamespace(model_variant="ref2va"))


def test_fasth3_gate_param_mapping() -> None:
    mapping = FastH3PipelineConfig().dit_config.arch_config.param_names_mapping
    source = "transformer_blocks.7.attn.to_gate_compress.weight"
    targets = [
        re.sub(pattern, target if isinstance(target, str) else target[0], source)
        for pattern, target in mapping.items()
        if re.match(pattern, source)
    ]
    assert targets == ["blocks.7.attn.to_gate_compress.weight"]


def test_fasth3_bundled_overlay_release_metadata() -> None:
    with open(os.path.join(_OVERLAY_DIR, "model_index.json"), encoding="utf-8") as f:
        model_index = json.load(f)
    metadata = MiniMaxH3ReleaseMetadata.from_model_index(model_index)
    assert metadata.partition == "fl2va"
    assert metadata.tasks == ("t2va",)
    assert metadata.sigma_shift_scales == {"video": 12.0, "audio": 3.0}

    with open(
        os.path.join(_OVERLAY_DIR, "_overlay", "overlay_manifest.json"),
        encoding="utf-8",
    ) as f:
        manifest = json.load(f)
    assert manifest["source_model_id"] == FASTH3_MODEL_ID
    assert manifest["custom_materializer"] == "_overlay/materialize.py"
    trees = {
        m["src"]: m["dst_dir"]
        for m in manifest["file_mappings"]
        if m.get("type") == "tree"
    }
    assert trees["transformer"] == "transformer"
    # The video VAE is re-serialized (source form) and both VAE configs are
    # class-name-patched by the materializer, so no vae/ mapping exists.
    assert "vae" not in trees


def test_fasth3_distilled_schedule_is_uniform_five_points() -> None:
    """The distilled grid is the base shift-12 schedule at 5 sigma points;
    dmd_denoising_steps [999, 749, 500, 250] are the unshifted grid labels."""
    from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.time_request import (
        minimax_h3_time_shift_sigmas,
    )

    sigmas = minimax_h3_time_shift_sigmas(num_steps=5, shift_scale=12.0)
    assert len(sigmas) == 5
    assert sigmas[0] == 1.0 and sigmas[-1] == 0.0
    base = torch.linspace(1.0, 0.0, 5)
    expected = (12.0 * base / (1 + 11.0 * base)).tolist()
    assert sigmas == pytest.approx(expected)


def test_fasth3_lora_bundle_is_rejected_loudly() -> None:
    """The sibling FastH3 LoRA bundle carries .diff/.diff_b deltas and
    to_gate_compress.set_weight tensors; --lora-path must fail with a readable
    error instead of silently dropping them."""
    from types import SimpleNamespace

    from sglang.multimodal_gen.runtime.models.dits.minimax_h3 import (
        MiniMaxH3DiTModel,
    )

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
    with pytest.raises(ValueError) as excinfo:
        MiniMaxH3DiTModel.prepare_lora_adapter(model, bundle)
    message = str(excinfo.value)
    assert "not a plain LoRA" in message
    assert "set_weight" in message
    assert "FastVideo-FastH3-4-step-Preview-v1-LoRA" in message
    assert "FastVideo-FastH3-4-step-Preview-v1-VSA-DataFree" in message


@pytest.mark.parametrize("quant_name", ["fp8", "kitchen_int8"])
def test_fasth3_gates_stay_bf16_under_runtime_quantization(quant_name: str) -> None:
    from sglang.multimodal_gen.runtime.layers.linear import UnquantizedLinearMethod
    from sglang.multimodal_gen.runtime.layers.quantization.configs.kitchen_int8_config import (
        KitchenInt8Config,
    )
    from sglang.multimodal_gen.runtime.layers.quantization.fp8 import Fp8Config
    from sglang.multimodal_gen.runtime.models.dits.minimax_h3 import (
        MiniMaxH3DiTModel,
    )
    from sglang.multimodal_gen.test.unit.test_minimax_h3_dit_contract import (
        _ensure_single_process_parallel_runtime,
    )

    if quant_name == "kitchen_int8":
        pytest.importorskip("comfy_kitchen")
    _ensure_single_process_parallel_runtime()
    quant_config = Fp8Config() if quant_name == "fp8" else KitchenInt8Config()
    with torch.device("meta"):
        model = MiniMaxH3DiTModel(
            config=FastH3PipelineConfig().dit_config,
            hf_config={},
            quant_config=quant_config,
        )

    attn = model.blocks[0].attn
    assert not isinstance(attn.qkv_proj.quant_method, UnquantizedLinearMethod)
    assert isinstance(attn.to_gate_compress.quant_method, UnquantizedLinearMethod)
    assert attn.to_gate_compress.weight.dtype == torch.bfloat16
    assert model.token_refiner.blocks[0].attn.to_gate_compress is None


def test_fasth3_adaln_cache_variant_is_the_distilled_partition() -> None:
    from types import SimpleNamespace

    no_variant = SimpleNamespace(model_variant=None)
    assert FastH3PipelineConfig().adaln_cache_model_variant(no_variant) == "fl2va"
    assert MiniMaxH3PipelineConfig().adaln_cache_model_variant(no_variant) is None
    assert (
        MiniMaxH3PipelineConfig().adaln_cache_model_variant(
            SimpleNamespace(model_variant="ref2va")
        )
        == "ref2va"
    )


def test_cache_dit_is_a_declared_noop_at_four_forwards() -> None:
    from types import SimpleNamespace

    from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.stages.denoising import (
        _cache_dit_noop_reason,
    )

    batch = SimpleNamespace(sampling_params=SimpleNamespace(cache_dit_params=None))
    reason = _cache_dit_noop_reason(4, batch)
    assert reason is not None and "4 DiT forwards" in reason
    assert _cache_dit_noop_reason(49, batch) is None
    lowered = SimpleNamespace(
        sampling_params=SimpleNamespace(cache_dit_params={"max_warmup_steps": 1})
    )
    assert _cache_dit_noop_reason(4, lowered) is None
