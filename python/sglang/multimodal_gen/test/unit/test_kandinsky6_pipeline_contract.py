# SPDX-License-Identifier: Apache-2.0
"""Pipeline wiring, residency precision and joint video/audio denoising."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

from sglang.multimodal_gen.configs.pipeline_configs.kandinsky6 import (
    Kandinsky6TI2VAPipelineConfig,
)
from sglang.multimodal_gen.configs.sample.kandinsky6 import (
    Kandinsky6TI2VASamplingParams,
)
from sglang.multimodal_gen.runtime.disaggregation.roles import RoleType
from sglang.multimodal_gen.runtime.distributed.cfg_policy import CFGPolicy
from sglang.multimodal_gen.runtime.managers.memory_managers import (
    component_residency_strategies,
)
from sglang.multimodal_gen.runtime.models.dits.kandinsky6 import (
    Kandinsky6RoPE1D,
    Kandinsky6RoPE3D,
)
from sglang.multimodal_gen.runtime.models.schedulers.kandinsky6_piflow import (
    PiflowScheduler,
)
from sglang.multimodal_gen.runtime.pipelines.kandinsky6_pipeline import (
    Kandinsky6TI2VAPipeline,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages import InputValidationStage
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.kandinsky6 import (
    Kandinsky6AudioDecodingStage,
    Kandinsky6DecodingStage,
    Kandinsky6DenoisingStage,
    Kandinsky6ImageEncodingStage,
    Kandinsky6LatentPreparationStage,
    denoising,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.text_encoding import (
    TextEncodingStage,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.timestep_preparation import (
    TimestepPreparationStage,
)
from sglang.multimodal_gen.runtime.server_args import get_global_server_args

EXPECTED_REQUIRED_MODULES = [
    "scheduler",
    "text_encoder",
    "text_encoder_2",
    "tokenizer",
    "tokenizer_2",
    "transformer",
    "vae",
    "audio_vae",
]

EXPECTED_STAGE_ORDER = [
    ("input_validation_stage", InputValidationStage),
    ("text_encoding_stage", TextEncodingStage),
    ("timestep_preparation_stage", TimestepPreparationStage),
    ("latent_preparation_stage", Kandinsky6LatentPreparationStage),
    ("image_encoding_stage", Kandinsky6ImageEncodingStage),
    ("denoising_stage", Kandinsky6DenoisingStage),
    # Audio decodes before video, matching the diffusers reference -- see
    # Kandinsky6TI2VAPipeline.create_pipeline_stages's comment.
    ("audio_decoding_stage", Kandinsky6AudioDecodingStage),
    ("decoding_stage", Kandinsky6DecodingStage),
]


def _make_pipeline() -> Kandinsky6TI2VAPipeline:
    pipeline = object.__new__(Kandinsky6TI2VAPipeline)
    pipeline.modules = {
        name: MagicMock(name=name)
        for name in Kandinsky6TI2VAPipeline._required_config_modules
    }
    pipeline._stages = []
    pipeline._stage_name_mapping = {}
    pipeline._disagg_role = RoleType.MONOLITHIC
    return pipeline


def test_disagg_role_rejects_non_monolithic_deployment():
    pipeline = object.__new__(Kandinsky6TI2VAPipeline)

    with pytest.raises(ValueError, match="monolithic"):
        pipeline.validate_disagg_role(RoleType.ENCODER)
    with pytest.raises(ValueError, match="monolithic"):
        pipeline.validate_disagg_role(RoleType.DENOISER)

    # MONOLITHIC is the only accepted role -- no exception raised.
    pipeline.validate_disagg_role(RoleType.MONOLITHIC)


def test_create_pipeline_stages_produces_documented_order():
    pipeline = _make_pipeline()
    assert pipeline.pipeline_name == "Kandinsky6TI2VAPipeline"
    assert pipeline.is_video_pipeline is True
    assert pipeline._required_config_modules == EXPECTED_REQUIRED_MODULES
    pipeline.create_pipeline_stages(MagicMock())
    assert [type(stage) for stage in pipeline.stages] == [
        stage_cls for _, stage_cls in EXPECTED_STAGE_ORDER
    ]
    assert [
        (name, type(stage)) for name, stage in pipeline._stage_name_mapping.items()
    ] == EXPECTED_STAGE_ORDER

    # get_module wiring: each stage received the mocked module it asked for.
    denoising_stage = pipeline._stage_name_mapping["denoising_stage"]
    assert denoising_stage.transformer is pipeline.modules["transformer"]

    audio_decoding_stage = pipeline._stage_name_mapping["audio_decoding_stage"]
    assert audio_decoding_stage.audio_vae is pipeline.modules["audio_vae"]

    decoding_stage = pipeline._stage_name_mapping["decoding_stage"]
    assert decoding_stage.vae is pipeline.modules["vae"]

    image_encoding_stage = pipeline._stage_name_mapping["image_encoding_stage"]
    assert image_encoding_stage.vae is pipeline.modules["vae"]


def _pro_distill_scheduler() -> PiflowScheduler:
    """The scheduler of kandinskylab/Kandinsky-6.0-Pro-distill-5s-Diffusers."""
    scheduler = PiflowScheduler(
        nfe=None,
        n_grid=10,
        shift=5.0,
        eps=1e-6,
        final_step_size_scale=0.5,
        num_policy_substeps=128,
    )
    scheduler.set_timesteps(10)
    return scheduler


@pytest.mark.parametrize("stage_name", ["denoising_stage", "audio_decoding_stage"])
@pytest.mark.parametrize("precision", ["bf16", "fp32"])
def test_component_residency_preserves_loaded_precision(stage_name, precision):
    pipeline = _make_pipeline()
    args = MagicMock()
    args.pipeline_config.dit_precision = precision
    args.pipeline_config.audio_vae_precision = precision
    args.component_precisions = {"transformer": precision, "audio_vae": precision}
    pipeline.create_pipeline_stages(args)
    use = pipeline._stage_name_mapping[stage_name].component_uses(args)[0]

    # loaders own precision; movement must not downcast RoPE or audio buffers
    assert use.target_dtype is None


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA transfers")
def test_denoising_offload_roundtrip_preserves_fp32_rope(monkeypatch):
    device = torch.device("cuda", torch.cuda.current_device())
    monkeypatch.setattr(
        component_residency_strategies, "get_local_torch_device", lambda: device
    )
    transformer = torch.nn.ModuleDict(
        {
            "projection": torch.nn.Linear(8, 8, dtype=torch.bfloat16),
            "rope_1d": Kandinsky6RoPE1D(8),
            "rope_3d": Kandinsky6RoPE3D((4, 4, 4)),
        }
    ).to(device)
    buffers = {name: value.clone() for name, value in transformer.named_buffers()}
    stage = Kandinsky6DenoisingStage(transformer, scheduler=MagicMock())
    use = stage.component_uses(get_global_server_args())[0]

    for _ in range(2):
        transformer.cpu()
        component_residency_strategies._module_to_local_device(
            transformer, dtype=use.target_dtype
        )
        assert next(transformer.parameters()).dtype == torch.bfloat16
        for name, value in transformer.named_buffers():
            assert value.device == device
            torch.testing.assert_close(value, buffers[name], rtol=0, atol=0)


def test_parallel_cfg_uses_serial_arithmetic_for_video_and_audio(monkeypatch):
    config = Kandinsky6TI2VAPipelineConfig()
    batch = SimpleNamespace(
        do_classifier_free_guidance=True,
        guidance_scale=5.0,
        cfg_normalization=0.0,
        guidance_rescale=0.0,
    )
    policy = config.cfg_policy.build(batch, {}, {}, {})
    positive = torch.linspace(-1, 1, 256, dtype=torch.bfloat16)
    negative = torch.linspace(-0.8, 0.9, 256, dtype=torch.bfloat16)
    predictions = [(positive, negative), (negative, positive)]
    expected = policy.combine(predictions, batch, 5.0, config)
    legacy = CFGPolicy().combine(predictions, batch, 5.0, config, cfg_parallel=True)
    assert any(not torch.equal(a, b) for a, b in zip(expected, legacy, strict=True))
    monkeypatch.setattr(denoising, "get_classifier_free_guidance_world_size", lambda: 2)
    monkeypatch.setattr(denoising, "run_cfg_parallel", lambda *_args: predictions)
    monkeypatch.setattr(
        denoising,
        "run_two_branch_cfg_parallel",
        lambda *_args: pytest.fail("legacy all-reduce changes BF16 rounding"),
    )
    stage = Kandinsky6DenoisingStage(MagicMock(), scheduler=MagicMock())
    actual = stage._predict_joint_velocity(
        transformer=stage.transformer,
        cfg_policy=policy,
        batch=batch,
        server_args=SimpleNamespace(pipeline_config=config, enable_cfg_parallel=True),
        step_index=0,
        video_input=positive,
        audio_input=negative,
        t_expand=torch.zeros(1),
        visual_rope_pos=[],
        scale_factor=(1.0, 1.0, 1.0),
        sparse_params=None,
        visual_token_type_ids=None,
    )
    for result, reference in zip(actual, expected, strict=True):
        torch.testing.assert_close(result, reference, rtol=0, atol=0)


@pytest.mark.parametrize(
    "guidance_scale", [5.0, 0.5, 1.001, float("nan"), float("inf")]
)
def test_denoising_rejects_guidance_other_than_one_with_a_piflow_scheduler(
    guidance_scale,
):
    """The distilled checkpoint must reject unsupported CFG, not silently ignore it."""
    scheduler = _pro_distill_scheduler()
    batch = SimpleNamespace(
        timesteps=scheduler.timesteps,
        latents=torch.zeros(1, 1, 1, 1, 1),
        audio_latents=torch.zeros(1, 1, 1),
        scheduler=scheduler,
        guidance_scale=guidance_scale,
    )
    stage = Kandinsky6DenoisingStage(transformer=MagicMock(), scheduler=scheduler)

    with pytest.raises(ValueError, match=r"requires guidance_scale=1\.0"):
        stage.forward(batch, MagicMock())


@pytest.mark.parametrize("height,width", [(32, 32), (512, 768), (768, 512)])
def test_distilled_denoising_calls_the_dit_once_per_step_without_guidance(
    monkeypatch, height, width
):
    """Pro-distill: 10 steps at guidance 1.0. The DiT head is ``n_grid`` (= 10) times as
    wide as the latents, the pi-Flow step folds it back, and the returned latents keep
    the latent width. The negative prompt is never fed to the DiT."""
    monkeypatch.setattr(
        denoising, "get_local_torch_device", lambda: torch.device("cpu")
    )
    monkeypatch.setattr(
        get_global_server_args().pipeline_config, "dit_precision", "fp32"
    )
    video_channels, audio_channels, n_grid, text_len = 4, 6, 10, 7
    text_lens = []

    class _FakeDiT(torch.nn.Module):
        def forward(self, **kwargs):
            assert tuple(kwargs["scale_factor"]) == (1.0, 2.0, 2.0)
            text_lens.append(kwargs["encoder_hidden_states"].shape[1])
            video, audio = kwargs["hidden_states"], kwargs["hidden_states_audio"]
            video_velocity = torch.randn(*video.shape[:-1], n_grid * video_channels)
            audio_velocity = torch.randn(*audio.shape[:-1], n_grid * audio_channels)
            assert kwargs["return_dict"] is False
            return video_velocity, audio_velocity

    fake_dit = _FakeDiT()

    scheduler = _pro_distill_scheduler()
    # visual_cond: the DiT input carries the latents, a condition copy and a mask
    video = torch.randn(1, 2, 4, 4, 2 * video_channels + 1)
    batch = SimpleNamespace(
        timesteps=scheduler.timesteps,
        latents=video,
        audio_latents=torch.randn(1, 5, audio_channels),
        scheduler=scheduler,
        guidance_scale=1.0,
        do_classifier_free_guidance=False,
        extra={},
        prompt_embeds=[torch.randn(1, text_len, 8)],
        negative_prompt_embeds=[torch.randn(1, text_len + 2, 8)],
        pooled_embeds=[torch.randn(1, 3)],
        prompt_attention_mask=[torch.ones(1, text_len)],
        height=height,
        width=width,
        image_latent=None,
        is_warmup=True,
        sampling_params=Kandinsky6TI2VASamplingParams(quality="lossless"),
    )
    pipeline_config = Kandinsky6TI2VAPipelineConfig(dit_precision="fp32")
    pipeline_config.dit_config.update_model_arch(
        dict(
            in_visual_dim=video_channels,
            patch_size=(1, 2, 2),
            scale_factor=(1.0, 2.0, 2.0),
        )
    )
    server_args = SimpleNamespace(
        pipeline_config=pipeline_config, enable_cfg_parallel=False
    )
    stage = Kandinsky6DenoisingStage(transformer=fake_dit, scheduler=scheduler)

    result = stage.forward(batch, server_args)

    assert text_lens == [text_len] * 10
    assert tuple(result.latents.shape) == (1, 2, 4, 4, video_channels)
    assert tuple(result.audio_latents.shape) == (1, 5, audio_channels)
    assert torch.isfinite(result.latents).all()
    assert torch.isfinite(result.audio_latents).all()
