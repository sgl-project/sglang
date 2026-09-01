# SPDX-License-Identifier: Apache-2.0
"""High-value task, partition, and public request admission contracts."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from sglang.multimodal_gen.configs.pipeline_configs.minimax_h3 import (
    MiniMaxH3PipelineConfig,
)
from sglang.multimodal_gen.configs.sample.minimax_h3 import MiniMaxH3SamplingParams
from sglang.multimodal_gen.runtime.entrypoints.openai.protocol import (
    VideoGenerationsRequest,
)
from sglang.multimodal_gen.runtime.layers.attention.backends.attention_backend import (
    AttentionRequirements,
)
from sglang.multimodal_gen.runtime.managers.memory_managers.component_residency import (
    LAYERWISE_OFFLOAD,
    RESIDENT,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.release_metadata import (
    MiniMaxH3PartitionAdmissionStage,
    MiniMaxH3ReleaseMetadata,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.request_validation import (
    minimax_h3_validate_canonical_request,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.resolved_plan import (
    minimax_h3_resolve_plan,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.stages.denoising import (
    MiniMaxH3DenoisingStage,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.task_profiles import (
    partition_for_task,
)
from sglang.multimodal_gen.runtime.platforms import (
    AttentionBackendEnum,
    current_platform,
)
from sglang.multimodal_gen.runtime.server_args.server_args import Backend

TARGET = {
    "short_edge": 768,
    "aspect_ratio": "16:9",
    "duration_seconds": 5.0,
}


@pytest.mark.parametrize(
    ("task", "conditions", "partition", "visual", "audio", "chains"),
    [
        ("t2va", [], "fl2va", [], [], []),
        (
            "fl2va",
            [
                {
                    "type": "image",
                    "uri": "file:///first.png",
                    "role": "keyframe",
                    "frame_index": 0,
                },
                {
                    "type": "image",
                    "uri": "file:///last.png",
                    "role": "keyframe",
                    "frame_index": -1,
                },
            ],
            "fl2va",
            [0, 1],
            [],
            ["image.target_canvas", "image.target_canvas"],
        ),
        (
            "ref2va",
            [
                {
                    "type": "image",
                    "uri": "file:///image.png",
                    "role": "reference",
                },
                {
                    "type": "video",
                    "uri": "file:///video.mp4",
                    "role": "reference",
                    "start_time_seconds": 12.5,
                },
                {
                    "type": "audio",
                    "uri": "file:///audio.wav",
                    "role": "reference",
                },
                {
                    "type": "video_audio",
                    "uri": "file:///av.mp4",
                    "role": "reference",
                },
            ],
            "ref2va",
            [0, 1, 3],
            [1, 2, 3],
            [
                "image.reference_preserve",
                "video.reference_preserve",
                "audio",
                "video_audio.reference_preserve",
            ],
        ),
        (
            "ref2va",
            [
                {
                    "type": "image",
                    "uri": "file:///first.png",
                    "role": "keyframe",
                    "frame_index": 0,
                },
                {
                    "type": "image",
                    "uri": "file:///subject.png",
                    "role": "reference",
                },
            ],
            "ref2va",
            [0, 1],
            [],
            ["image.target_canvas", "image.reference_preserve"],
        ),
    ],
)
def test_public_tasks_resolve_to_exact_partition_and_encoder_plan(
    task, conditions, partition, visual, audio, chains
):
    canonical = minimax_h3_validate_canonical_request(
        task=task,
        prompt="contract",
        conditions=conditions,
        target=TARGET,
        seed=0,
    )
    plan = minimax_h3_resolve_plan(canonical)

    assert partition_for_task(task) == partition
    assert plan.task == task
    assert plan.encoders["visual"] == visual
    assert plan.encoders["audio"] == audio
    assert [material.material_chain for material in plan.materials] == chains
    if task == "ref2va":
        assert plan.encoders["qwen"]["ordered_condition_indices"] == [
            index
            for index, condition in enumerate(conditions)
            if condition["role"] == "reference"
        ]
    for index, condition in enumerate(conditions):
        if condition.get("start_time_seconds") is not None:
            assert plan.materials[index].start_time_seconds == 12.5
    assert plan.shape["frame_count"] == 124
    assert plan.shape["video_latent_t"] == 37


def test_ref2va_rejects_keyframes_without_a_reference():
    with pytest.raises(ValueError, match="at least one reference"):
        minimax_h3_validate_canonical_request(
            task="ref2va",
            prompt="contract",
            conditions=[
                {
                    "type": "image",
                    "uri": "file:///first.png",
                    "role": "keyframe",
                    "frame_index": 0,
                }
            ],
            target=TARGET,
        )


@pytest.mark.parametrize(
    ("partition", "tasks"),
    [("fl2va", ["t2va", "fl2va"]), ("ref2va", ["ref2va"])],
)
def test_loaded_weight_partition_admits_only_its_declared_tasks(partition, tasks):
    metadata = MiniMaxH3ReleaseMetadata.from_model_index(
        {
            "_minimax_h3": {
                "schema_version": 1,
                "partition": partition,
                "tasks": tasks,
                "task_aliases": {},
                "sigma_shift_scales": {"video": 12.0, "audio": 3.0},
            }
        }
    )

    assert [metadata.canonical_task(task) for task in tasks] == tasks
    rejected = "ref2va" if partition == "fl2va" else "t2va"
    with pytest.raises(ValueError):
        metadata.canonical_task(rejected)


def test_duration_admission_accepts_released_4_to_15_second_range():
    for duration in (4.0, 15.0):
        target = {**TARGET, "duration_seconds": duration}
        canonical = minimax_h3_validate_canonical_request(
            task="t2va",
            prompt="duration contract",
            conditions=[],
            target=target,
            seed=0,
        )
        assert canonical["target"]["duration_seconds"] == duration

    for duration in (3.9, 15.1):
        target = {**TARGET, "duration_seconds": duration}
        with pytest.raises(ValueError, match=r"\[4, 15\]"):
            minimax_h3_validate_canonical_request(
                task="t2va",
                prompt="duration contract",
                conditions=[],
                target=target,
                seed=0,
            )


def test_video_adapter_lowers_only_native_fields_and_rejects_cfg():
    request = VideoGenerationsRequest(
        prompt="contract",
        task="t2va",
        conditions=[],
        target=TARGET,
        flow_shift=8.0,
        audio_flow_shift=2.0,
        quality="high",
        imgvid_cond_noise_aug_for_inference=0.75,
        audio_cond_noise_aug_for_inference=0.5,
    )
    generic = {
        "prompt": request.prompt,
        "seed": request.seed,
        "flow_shift": request.flow_shift,
    }

    lowered = MiniMaxH3SamplingParams.lower_video_request_kwargs(request, generic)
    assert lowered == {
        "prompt": "contract",
        "seed": request.seed,
        "task": "t2va",
        "conditions": [],
        "target": TARGET,
        "flow_shift": 8.0,
        "audio_flow_shift": 2.0,
        "quality": "high",
        "imgvid_cond_noise_aug_for_inference": 0.75,
        "audio_cond_noise_aug_for_inference": 0.5,
    }

    with pytest.raises(ValueError):
        MiniMaxH3SamplingParams.lower_video_request_kwargs(
            request, {**generic, "guidance_scale": 7.5}
        )


@pytest.mark.parametrize("bad_quality", ["ultra", "draft", "", 1])
def test_video_adapter_rejects_invalid_quality(bad_quality):
    request = VideoGenerationsRequest(
        prompt="contract",
        task="t2va",
        conditions=[],
        target=TARGET,
        quality=bad_quality,
    )
    with pytest.raises(ValueError, match="quality must be one of"):
        MiniMaxH3SamplingParams.lower_video_request_kwargs(
            request, {"prompt": request.prompt, "seed": request.seed}
        )


class _HopperCapability:
    def to_int(self) -> int:
        return 90


def _quality_server_args():
    return SimpleNamespace(
        attention_backend=None,
        model_variant="fl2va",
        num_gpus=4,
        backend=Backend.AUTO,
        component_attention_backends={},
        enable_breakable_cuda_graph=False,
        enable_torch_compile=False,
        is_dit_layerwise_offload_selected=False,
        performance_mode="speed",
        quantization=None,
        transformer_weights_path=None,
        regional_compile=False,
        ring_degree=1,
        sp_degree=4,
        tp_size=1,
        ulysses_degree=4,
        use_fsdp_inference=False,
    )


def test_high_quality_deployment_rejects_transformer_weight_override():
    config = MiniMaxH3PipelineConfig()
    server_args = _quality_server_args()
    server_args.transformer_weights_path = "model.gguf"

    with (
        patch.object(current_platform, "is_cuda", return_value=True),
        patch.object(current_platform, "get_device_name", return_value="NVIDIA H200"),
        patch.object(
            current_platform,
            "get_device_capability",
            return_value=_HopperCapability(),
        ),
        pytest.raises(ValueError, match="transformer_weights_path"),
    ):
        config.validate_quality_deployment(server_args)


def test_high_quality_request_warns_when_bcg_suppresses_cache_dit():
    stage = MiniMaxH3DenoisingStage.__new__(MiniMaxH3DenoisingStage)
    stage.server_args = SimpleNamespace(enable_breakable_cuda_graph=True)
    stage._cache_dit_enabled = False
    batch = SimpleNamespace(
        sampling_params=SimpleNamespace(
            quality="high",
            _explicit_fields={"quality"},
            enable_cache_dit=None,
            cache_dit_params=None,
        )
    )

    with patch(
        "sglang.multimodal_gen.runtime.pipelines_core.stages.denoising."
        "logger.warning_once"
    ) as warning_once:
        stage._maybe_enable_cache_dit(50, batch)

    warning_once.assert_called_once_with(
        "Cache-DiT was requested but is disabled because breakable CUDA graphs "
        "are enabled."
    )


def test_quality_admission_fails_closed_outside_validated_request():
    metadata = MiniMaxH3ReleaseMetadata.from_model_index(
        {
            "_minimax_h3": {
                "schema_version": 1,
                "partition": "fl2va",
                "tasks": ["t2va", "fl2va"],
                "task_aliases": {},
                "sigma_shift_scales": {"video": 12.0, "audio": 3.0},
            }
        }
    )
    canonical = minimax_h3_validate_canonical_request(
        task="t2va",
        prompt="quality",
        conditions=[],
        target=TARGET,
        seed=0,
    )
    plan = minimax_h3_resolve_plan(canonical)
    batch = SimpleNamespace(
        sampling_params=SimpleNamespace(task="t2va", quality="high"),
        num_inference_steps=50,
        is_warmup=False,
    )
    stage = MiniMaxH3PartitionAdmissionStage(metadata)
    config = MiniMaxH3PipelineConfig()
    server_args = _quality_server_args()
    server_args.pipeline_config = config

    with (
        patch.object(current_platform, "is_cuda", return_value=True),
        patch.object(current_platform, "get_device_name", return_value="NVIDIA H200"),
        patch.object(
            current_platform,
            "get_device_capability",
            return_value=_HopperCapability(),
        ),
        patch(
            "sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages."
            "minimax_h3.release_metadata.minimax_h3_plan_from_batch",
            return_value=plan,
        ),
    ):
        assert stage.forward(batch, server_args) is batch
        batch.num_inference_steps = 40
        with pytest.raises(ValueError, match="validated only"):
            stage.forward(batch, server_args)

    batch.sampling_params.quality = "lossless"
    batch.num_inference_steps = 50
    server_args.attention_backend = "sage_attn"
    assert stage.forward(batch, server_args) is batch

    batch.sampling_params.quality = "ultra"
    server_args.attention_backend = None
    with pytest.raises(ValueError, match="quality must be one of"):
        stage.forward(batch, server_args)


def test_validate_server_args_requires_packed_varlen_backend():
    config = MiniMaxH3PipelineConfig()
    server_args = SimpleNamespace(
        component_attention_backends={},
        attention_backend="sage_attn",
        ring_degree=1,
        resolve_component_attention_backend=lambda *_names: (None, None),
    )
    with patch(
        "sglang.multimodal_gen.configs.pipeline_configs.minimax_h3.get_attn_backend"
    ) as get_attn_backend:
        MiniMaxH3PipelineConfig.validate_server_args(config, server_args)
    get_attn_backend.assert_called_once_with(
        128,
        torch.bfloat16,
        selected_attention_backend=AttentionBackendEnum.SAGE_ATTN,
        attention_requirements=AttentionRequirements(packed_varlen=True),
    )
    with patch(
        "sglang.multimodal_gen.configs.pipeline_configs.minimax_h3.get_attn_backend",
        side_effect=ValueError("does not implement packed varlen attention"),
    ):
        with pytest.raises(ValueError, match="does not implement packed varlen"):
            MiniMaxH3PipelineConfig.validate_server_args(config, server_args)


def test_validate_server_args_accepts_transformer_backend_override():
    config = MiniMaxH3PipelineConfig()
    server_args = SimpleNamespace(
        component_attention_backends={"transformer": "subblock_sparse_attn"},
        attention_backend="fa",
        ring_degree=1,
        resolve_component_attention_backend=lambda *_names: (
            AttentionBackendEnum.SUBBLOCK_SPARSE_ATTN,
            "transformer",
        ),
    )

    with patch(
        "sglang.multimodal_gen.configs.pipeline_configs.minimax_h3.get_attn_backend"
    ) as get_attn_backend:
        MiniMaxH3PipelineConfig.validate_server_args(config, server_args)
    get_attn_backend.assert_called_once_with(
        128,
        torch.bfloat16,
        selected_attention_backend=AttentionBackendEnum.SUBBLOCK_SPARSE_ATTN,
        attention_requirements=AttentionRequirements(packed_varlen=True),
    )


def test_resolve_transformer_attention_backend_uses_selector_precedence():
    config = MiniMaxH3PipelineConfig()
    subblock = AttentionBackendEnum.SUBBLOCK_SPARSE_ATTN
    fa = AttentionBackendEnum.FA
    sdpa = AttentionBackendEnum.TORCH_SDPA
    cases = (
        ("fa", subblock, None, subblock),
        ("subblock_sparse_attn", fa, None, fa),
        (subblock, None, None, subblock),
        ("fa", subblock, sdpa, sdpa),
    )
    for global_backend, component_backend, forced_backend, expected in cases:
        server_args = SimpleNamespace(
            attention_backend=global_backend,
            resolve_component_attention_backend=lambda *_names: (
                component_backend,
                "transformer" if component_backend is not None else None,
            ),
        )
        with patch(
            "sglang.multimodal_gen.configs.pipeline_configs.minimax_h3."
            "get_global_forced_attn_backend",
            return_value=forced_backend,
        ):
            resolved = config.resolve_transformer_attention_backend(server_args)
            assert resolved is expected
            assert config.uses_subblock_attention(server_args) is (
                expected is AttentionBackendEnum.SUBBLOCK_SPARSE_ATTN
            )


def test_mps_admission_requires_layerwise_residency_for_every_h3_component():
    config = MiniMaxH3PipelineConfig()
    modes = {
        "transformer": LAYERWISE_OFFLOAD,
        "text_encoder": LAYERWISE_OFFLOAD,
        "video_vae": LAYERWISE_OFFLOAD,
        "audio_vae": LAYERWISE_OFFLOAD,
    }
    server_args = SimpleNamespace(
        component_attention_backends={},
        attention_backend=None,
        enable_torch_compile=False,
        ring_degree=1,
        residency_mode=modes.get,
        resolve_component_attention_backend=lambda *_names: (None, None),
    )

    with patch.object(current_platform, "is_mps", return_value=True):
        MiniMaxH3PipelineConfig.validate_server_args(config, server_args)

        modes["audio_vae"] = RESIDENT
        with pytest.raises(ValueError, match="audio_vae"):
            MiniMaxH3PipelineConfig.validate_server_args(config, server_args)
