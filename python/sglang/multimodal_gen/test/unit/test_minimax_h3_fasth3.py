# SPDX-License-Identifier: Apache-2.0
"""FastH3 (VSA-distilled MiniMax-H3) registration, schedule, and admission contracts."""

from __future__ import annotations

import json
import re
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from sglang.multimodal_gen.configs.pipeline_configs.minimax_h3 import (
    FastH3PipelineConfig,
    FastH3V2PipelineConfig,
    MiniMaxH3PipelineConfig,
)
from sglang.multimodal_gen.configs.sample.minimax_h3 import (
    FastH3SamplingParams,
    FastH3V2SamplingParams,
)
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
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.constants import (
    MINIMAX_H3_SIGMAS_EXTRA_KEY,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.fasth3_contract import (
    FASTH3_INFERENCE_CONTRACT_FILE,
    FastH3InferenceContract,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.release_metadata import (
    MiniMaxH3ReleaseMetadata,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.stages.denoising import (
    _maybe_prepare_vsa_h3_step_metadata,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.stages.timestep_preparation import (
    MiniMaxH3TimestepPreparationStage,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.time_request import (
    minimax_h3_rung_sigmas,
    minimax_h3_time_shift_sigmas,
)
from sglang.multimodal_gen.runtime.platforms import AttentionBackendEnum
from sglang.multimodal_gen.runtime.utils.model_overlay import (
    download_overlay_metadata,
    load_model_index_from_dir,
    load_overlay_manifest_if_present,
    resolve_model_overlay,
)
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

    with pytest.raises(ValueError, match="exactly 5 sigma grid points"):
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


# ===== FastH3 8-Step V2 =====

V2_MODEL_ID = "FastVideo/FastVideo-FastH3-8-Step-V2"
V2_RUNGS = (999, 874, 749, 624, 500, 375, 250, 125)
V2_SHIFTS = {"video": 10.0, "audio": 3.0}
# The schedule fields of the checkpoint's published fastvideo_inference.json.
V2_CONTRACT = {
    "attention_backend": "VIDEO_SPARSE_ATTN_H3",
    "audio_scheduler_shift": 3.0,
    "dmd_denoising_steps": list(V2_RUNGS),
    "guidance_scale": 1.0,
    "num_inference_steps": 9,
    "schema_version": "fasth3-inference-contract-v1",
    "task": "t2av",
    "transformer_forwards": 8,
    "video_scheduler_shift": 10.0,
    "vsa_sparsity": 0.8,
    "vsa_tile_size": 64,
}
# Exact fp32 sigmas FastVideo's 8-step recipe steps through (rung / 1000, shifted).
V2_VIDEO_SIGMAS = [
    0.9998998641967773,
    0.9857883453369141,
    0.9675752520561218,
    0.9431681036949158,
    0.9090909361839294,
    0.8571428656578064,
    0.7692307829856873,
    0.5882353186607361,
    0.0,
]
V2_AUDIO_SIGMAS = [
    0.9996663928031921,
    0.9541484117507935,
    0.8995195627212524,
    0.8327401280403137,
    0.75,
    0.6428571343421936,
    0.5,
    0.30000001192092896,
    0.0,
]


def _v2_bundled_overlay_dir() -> str:
    return download_overlay_metadata(
        V2_MODEL_ID, resolve_model_overlay(V2_MODEL_ID), snapshot_download_fn=None
    )


def _v2_release() -> MiniMaxH3ReleaseMetadata:
    return MiniMaxH3ReleaseMetadata.from_model_index(
        load_model_index_from_dir(_v2_bundled_overlay_dir())
    )


def _write_v2_checkpoint(tmp_path, contract=V2_CONTRACT, shifts=V2_SHIFTS) -> str:
    (tmp_path / FASTH3_INFERENCE_CONTRACT_FILE).write_text(json.dumps(contract))
    for modality, subdir in (("video", "scheduler"), ("audio", "audio_scheduler")):
        (tmp_path / subdir).mkdir(exist_ok=True)
        (tmp_path / subdir / "scheduler_config.json").write_text(
            json.dumps({"_class_name": "MiniMaxH3Scheduler", "shift": shifts[modality]})
        )
    return str(tmp_path)


def _prepare_v2_sigmas(*, num_steps=9, is_warmup=False, **plan_overrides):
    stage = MiniMaxH3TimestepPreparationStage(
        sigma_shift_scales=dict(V2_SHIFTS), sigma_rungs=V2_RUNGS
    )
    batch = SimpleNamespace(
        extra={}, num_inference_steps=num_steps, is_warmup=is_warmup
    )
    plan = SimpleNamespace(
        flow_shift=None,
        audio_flow_shift=None,
        default_flow_shift=12.0,
        default_audio_flow_shift=3.0,
    )
    for name, value in plan_overrides.items():
        setattr(plan, name, value)
    stage._generate_sigmas_from_plan(batch, plan)
    stage._publish_native_timestep_state(batch)
    return batch


def test_registry_resolves_fasth3_v2_apart_from_the_preview() -> None:
    info = get_model_info(V2_MODEL_ID)
    assert info.sampling_param_cls is FastH3V2SamplingParams
    assert info.pipeline_config_cls is FastH3V2PipelineConfig
    assert get_non_diffusers_pipeline_name(V2_MODEL_ID) == "FastH3V2Pipeline"
    materialized = (
        "/cache/materialized_models/FastVideo__FastVideo-FastH3-8-Step-V2-0123abcd"
    )
    assert get_model_info(materialized).sampling_param_cls is FastH3V2SamplingParams
    assert get_model_info(FASTH3_MODEL_ID).sampling_param_cls is FastH3SamplingParams


def test_bundled_v2_overlay_materializes_the_contract_files() -> None:
    overlay_dir = _v2_bundled_overlay_dir()
    manifest = load_overlay_manifest_if_present(overlay_dir)
    assert manifest["source_model_id"] == V2_MODEL_ID
    mapped = {mapping["src"] for mapping in manifest["file_mappings"]}
    assert {FASTH3_INFERENCE_CONTRACT_FILE, "scheduler", "audio_scheduler"} <= mapped
    assert load_model_index_from_dir(overlay_dir)["_class_name"] == "FastH3V2Pipeline"
    release = _v2_release()
    assert release.tasks == ("t2va",)
    assert release.sigma_shift_scales == V2_SHIFTS


def test_fasth3_v2_sampling_pins_nine_grid_points() -> None:
    assert FastH3V2SamplingParams(prompt="p").num_inference_steps == 9
    with pytest.raises(ValueError, match="exactly 9 sigma grid points"):
        FastH3V2SamplingParams(prompt="p", num_inference_steps=5)


def test_fasth3_v2_defaults_the_transformer_to_vsa_h3() -> None:
    def server_args(attention_backend=None):
        return SimpleNamespace(
            model_variant=None,
            attention_backend=attention_backend,
            component_attention_backends={},
            resolve_component_attention_backend=lambda component: (None, None),
            ring_degree=1,
            enable_torch_compile=False,
            enable_breakable_cuda_graph=False,
        )

    unset = server_args()
    FastH3V2PipelineConfig().validate_server_args(unset)
    assert unset.attention_backend == "video_sparse_attn_h3"
    explicit = server_args(attention_backend="fa")
    FastH3V2PipelineConfig().validate_server_args(explicit)
    assert explicit.attention_backend == "fa"


def test_rung_sigmas_are_the_trained_fp32_ladder() -> None:
    for modality, expected in (("video", V2_VIDEO_SIGMAS), ("audio", V2_AUDIO_SIGMAS)):
        shift = V2_SHIFTS[modality]
        sigmas = minimax_h3_rung_sigmas(rungs=V2_RUNGS, shift_scale=shift)
        assert sigmas == expected
        base = np.array([rung / 1000.0 for rung in V2_RUNGS] + [0.0], np.float32)
        oracle = np.float32(shift) * base / (1 + np.float32(shift - 1) * base)
        assert sigmas == oracle.astype(np.float32).tolist()
        # The rungs are not the uniform nine-point grid: they differ at 999..624.
        uniform = minimax_h3_time_shift_sigmas(num_steps=9, shift_scale=shift)
        assert sigmas[:4] != uniform[:4] and sigmas[4:] == uniform[4:]


def test_timestep_stage_serves_the_rung_ladder() -> None:
    batch = _prepare_v2_sigmas()
    assert batch.extra[MINIMAX_H3_SIGMAS_EXTRA_KEY] == {
        "video": V2_VIDEO_SIGMAS,
        "audio": V2_AUDIO_SIGMAS,
    }
    expected_t = 1.0 - torch.tensor(V2_VIDEO_SIGMAS[:-1], dtype=torch.float32)
    torch.testing.assert_close(batch.timesteps, expected_t, rtol=0, atol=0)

    # Warmup runs a prefix of the served grid instead of a rescaled one.
    warmup = _prepare_v2_sigmas(num_steps=2, is_warmup=True)
    assert warmup.extra[MINIMAX_H3_SIGMAS_EXTRA_KEY]["video"] == V2_VIDEO_SIGMAS[:2]


@pytest.mark.parametrize(
    "kwargs, match",
    [
        ({"num_steps": 5}, "must be 9 sigma grid points"),
        ({"flow_shift": 12.0}, "trained shifts"),
        ({"audio_flow_shift": 6.0}, "trained shifts"),
    ],
)
def test_timestep_stage_rejects_off_ladder_requests(kwargs, match) -> None:
    with pytest.raises(ValueError, match=match):
        _prepare_v2_sigmas(**kwargs)


def test_contract_binds_the_trained_ladder_and_sparsity(tmp_path) -> None:
    contract = FastH3InferenceContract.load(_write_v2_checkpoint(tmp_path))
    release = contract.bind(_v2_release())
    assert release.sigma_rungs == V2_RUNGS
    assert release.vsa_sparsity == 0.8
    assert release.sigma_shift_scales == V2_SHIFTS


@pytest.mark.parametrize(
    "overrides, match",
    [
        ({"schema_version": "fasth3-inference-contract-v0"}, "schema_version"),
        ({"task": "fl2av"}, "task must be 't2av'"),
        ({"guidance_scale": 3.0}, "CFG"),
        ({"dmd_denoising_steps": [999, 999, 500]}, "strictly decreasing"),
        ({"dmd_denoising_steps": [1000.0, 500]}, "strictly decreasing"),
        ({"num_inference_steps": 8}, "num_inference_steps=9"),
        ({"video_scheduler_shift": 12.0}, "disagrees with"),
        ({"attention_backend": "FLASH_ATTN"}, "attention_backend"),
        ({"vsa_tile_size": 256}, "64-token tiles"),
        ({"vsa_sparsity": 1.0}, r"vsa_sparsity must be in \[0, 1\)"),
    ],
)
def test_contract_rejects_an_inconsistent_checkpoint(
    tmp_path, overrides, match
) -> None:
    with pytest.raises(ValueError, match=match):
        FastH3InferenceContract.load(
            _write_v2_checkpoint(tmp_path, contract={**V2_CONTRACT, **overrides})
        )


def test_contract_fails_loudly_on_missing_files_and_release_drift(tmp_path) -> None:
    with pytest.raises(ValueError, match="missing fastvideo_inference.json"):
        FastH3InferenceContract.load(str(tmp_path))
    with pytest.raises(ValueError, match="disagrees with"):
        FastH3InferenceContract.load(
            _write_v2_checkpoint(tmp_path, shifts={"video": 12.0, "audio": 3.0})
        )

    contract = FastH3InferenceContract.load(_write_v2_checkpoint(tmp_path))
    release = _v2_release()
    with pytest.raises(ValueError, match="disagree with the checkpoint's trained"):
        contract.bind(replace(release, video_sigma_shift=12.0))
    with pytest.raises(ValueError, match="serves t2va only"):
        contract.bind(replace(release, tasks=("t2va", "fl2va")))


@pytest.mark.parametrize(
    "checkpoint_sparsity, config, expected",
    [
        (0.8, {}, 0.8),
        (0.8, {"VSA_sparsity": 0.5}, 0.5),
        (None, {}, 0.9),
    ],
)
def test_vsa_sparsity_defaults_to_the_checkpoint(
    checkpoint_sparsity, config, expected
) -> None:
    model = SimpleNamespace(
        _resolve_attention_backend_once=lambda: None,
        _resolved_attention_backend=AttentionBackendEnum.VIDEO_SPARSE_ATTN_H3,
    )
    video_rows = 4 * 4 * 4
    packed = {
        "text_pos": torch.arange(10),
        "update_mask": torch.ones(video_rows, dtype=torch.bool),
        "img_pos": torch.arange(video_rows),
        "audio_pos": torch.arange(8),
    }
    build = _maybe_prepare_vsa_h3_step_metadata(
        model=model,
        packed=packed,
        ctx=SimpleNamespace(is_ref2va=False, latent_t=4, latent_h=8, latent_w=8),
        server_args=SimpleNamespace(
            attention_backend_config=config,
            pipeline_config=FastH3V2PipelineConfig(),
        ),
        device=torch.device("cpu"),
        checkpoint_sparsity=checkpoint_sparsity,
    )
    assert build(0).VSA_sparsity == expected
