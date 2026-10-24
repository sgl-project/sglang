import pytest
import torch
from diffusers import SCMScheduler as DiffusersSCMScheduler

from sglang.multimodal_gen.configs.models.dits.sana import SanaConfig
from sglang.multimodal_gen.configs.pipeline_configs.sana import SanaPipelineConfig
from sglang.multimodal_gen.configs.pipeline_configs.sana_sprint import (
    ASPECT_RATIO_1024_BIN,
    SanaSprintPipelineConfig,
    classify_height_width_bin,
)
from sglang.multimodal_gen.configs.sample.sana import SanaSamplingParams
from sglang.multimodal_gen.configs.sample.sana_sprint import SanaSprintSamplingParams
from sglang.multimodal_gen.registry import _get_config_info, get_model_info
from sglang.multimodal_gen.runtime.models.schedulers.scheduling_scm import (
    SCMScheduler,
)
from sglang.multimodal_gen.runtime.pipelines.sana_sprint import _resize_and_crop

_SIGMA_DATA = 0.5


def _fake_dit(latent_model_input: torch.Tensor, scm_timestep) -> torch.Tensor:
    """A deterministic stand-in for the bf16 DiT."""
    weight = torch.linspace(-1.0, 1.0, 64).view(8, 8)
    mixed = torch.einsum("bchw,cd->bdhw", latent_model_input.float(), weight)
    return (torch.tanh(mixed) * (1 + scm_timestep)).to(torch.bfloat16)


def _diffusers_sprint_loop(steps: int, schedule: dict) -> torch.Tensor:
    """The SanaSprintPipeline.__call__ denoising loop, reduced to the scheduler math."""
    scheduler = DiffusersSCMScheduler()
    if "timesteps" in schedule:
        scheduler.set_timesteps(
            steps,
            timesteps=schedule["timesteps"],
            max_timesteps=None,
            intermediate_timesteps=None,
        )
    else:
        scheduler.set_timesteps(
            steps,
            max_timesteps=1.57080,
            intermediate_timesteps=(
                schedule.get("intermediate_timesteps", 1.3) if steps == 2 else None
            ),
        )
    scheduler.set_begin_index(0)
    generator = torch.Generator().manual_seed(0)
    latents = torch.randn(1, 8, 4, 4, generator=generator) * _SIGMA_DATA
    for t in scheduler.timesteps[:-1]:
        timestep = t.expand(1)
        scm_t = torch.sin(timestep) / (torch.cos(timestep) + torch.sin(timestep))
        scm_t4 = scm_t.view(-1, 1, 1, 1)
        norm = torch.sqrt(scm_t4**2 + (1 - scm_t4) ** 2)
        model_input = latents / _SIGMA_DATA * norm
        model_output = _fake_dit(model_input.to(torch.bfloat16), scm_t)
        pred = (
            (1 - 2 * scm_t4) * model_input
            + (1 - 2 * scm_t4 + 2 * scm_t4**2) * model_output
        ) / norm
        latents, denoised = scheduler.step(
            pred.float() * _SIGMA_DATA,
            timestep,
            latents,
            generator=generator,
            return_dict=False,
        )
    return denoised / _SIGMA_DATA


def _shared_denoising_loop(steps: int, schedule: dict) -> torch.Tensor:
    """How LatentPreparationStage and DenoisingStage drive the scheduler."""
    scheduler = SCMScheduler()
    scheduler.set_timesteps(steps, **schedule)
    # DenoisingStage._reset_scheduler_loop_state runs before every request.
    scheduler._step_index = None
    scheduler._begin_index = None
    generator = torch.Generator().manual_seed(0)
    latents = torch.randn(1, 8, 4, 4, generator=generator) * scheduler.init_noise_sigma
    for t in scheduler.timesteps:
        model_input = scheduler.scale_model_input(latents, t).to(torch.bfloat16)
        model_output = _fake_dit(model_input, t)
        latents = scheduler.step(
            model_output, t, latents, generator=generator, return_dict=False
        )[0]
    return latents


@pytest.mark.parametrize(
    "steps, schedule",
    [
        (1, {}),
        (2, {}),
        (4, {}),
        (2, {"intermediate_timesteps": 1.1}),
        (4, {"timesteps": [1.5708, 1.3, 1.0, 0.6, 0.0]}),
    ],
)
def test_scm_scheduler_reproduces_diffusers_sprint_loop(steps, schedule):
    """The SCM scheduler must reproduce Diffusers' SanaSprintPipeline loop exactly:
    the sCM timestep the DiT sees, its input scaling, the fp32 flow-to-TrigFlow
    conversion of a bf16 DiT output, the step noise and the final /sigma_data,
    for default, intermediate-angle and explicit-angle schedules."""
    torch.testing.assert_close(
        _shared_denoising_loop(steps, schedule),
        _diffusers_sprint_loop(steps, schedule),
        rtol=0,
        atol=0,
    )


def test_sana_sprint_registry_and_guidance_distilled_defaults(monkeypatch):
    """Sprint checkpoints must not fall into the generic SANA detector, and the
    guidance-distilled defaults must never run an unconditional CFG pass."""
    cases = {
        "Efficient-Large-Model/Sana_Sprint_0.6B_1024px_diffusers": (
            "SanaSprintPipeline",
            SanaSprintPipelineConfig,
            SanaSprintSamplingParams,
        ),
        "Efficient-Large-Model/Sana_Sprint_1.6B_1024px_diffusers": (
            "SanaSprintPipeline",
            SanaSprintPipelineConfig,
            SanaSprintSamplingParams,
        ),
        # Unregistered mirrors resolve through detectors, where "sana" also matches.
        "my-org/Sana_Sprint_0.6B_1024px_mirror": (
            "SanaSprintPipeline",
            SanaSprintPipelineConfig,
            SanaSprintSamplingParams,
        ),
        "Efficient-Large-Model/Sana_600M_1024px_diffusers": (
            "SanaPipeline",
            SanaPipelineConfig,
            SanaSamplingParams,
        ),
    }
    for model_id, (class_name, config_cls, sampling_cls) in cases.items():
        monkeypatch.setattr(
            "sglang.multimodal_gen.registry.maybe_download_model_index",
            lambda _, name=class_name: {"_class_name": name},
        )
        get_model_info.cache_clear()
        model_info = get_model_info(model_id)
        assert model_info.pipeline_config_cls is config_cls
        assert model_info.sampling_param_cls is sampling_cls
    get_model_info.cache_clear()

    sampling = SanaSprintSamplingParams(negative_prompt="blurry")
    assert (sampling.num_inference_steps, sampling.guidance_scale) == (2, 4.5)
    assert not sampling.should_use_classifier_free_guidance()


@pytest.mark.parametrize(
    "model_id",
    [
        "Efficient-Large-Model/Sana_Sprint_0.6B_1024px_diffusers",
        "Efficient-Large-Model/Sana_Sprint_1.6B_1024px_diffusers",
    ],
)
def test_sana_sprint_checkpoint_ids_are_registered_exactly(monkeypatch, model_id):
    """A registered id resolves from the path table alone, without the
    model_index.json fetch the detector fallback needs, and `--model-id` finds it."""

    def _no_model_index(_):
        raise AssertionError("registered id fell through to the detector fallback")

    monkeypatch.setattr(
        "sglang.multimodal_gen.registry.maybe_download_model_index", _no_model_index
    )
    for kwargs in ({}, {"model_id": model_id.split("/")[-1]}):
        config_info = _get_config_info(model_id, **kwargs)
        assert config_info.pipeline_config_cls is SanaSprintPipelineConfig
        assert config_info.sampling_param_cls is SanaSprintSamplingParams


def test_sana_qk_norm_follows_checkpoint_config():
    """SANA 1.0 configs omit `qk_norm` and ship no norm_q/norm_k weights, so the
    default must stay off; SANA 1.5 and Sprint configs turn it on explicitly."""
    sana_1_0 = SanaConfig()
    sana_1_0.update_model_arch({"num_layers": 28, "cross_attention_dim": 1152})
    assert sana_1_0.arch_config.qk_norm is None

    sprint = SanaConfig()
    sprint.update_model_arch(
        {"qk_norm": "rms_norm_across_heads", "guidance_embeds": True}
    )
    assert sprint.arch_config.qk_norm == "rms_norm_across_heads"
    assert sprint.arch_config.guidance_embeds_scale == 0.1


def test_sana_sprint_schedule_fields_reach_the_image_api():
    """The image API forwards only declared model fields; an undeclared Sprint
    field would be dropped from HTTP requests without an error."""
    from sglang.multimodal_gen.runtime.entrypoints.openai.protocol import (
        ImageGenerationsRequest,
    )
    from sglang.multimodal_gen.runtime.entrypoints.openai.utils import (
        request_model_kwargs,
    )

    fields = {
        "max_timesteps": 1.5,
        "intermediate_timesteps": 1.1,
        "trigflow_timesteps": [1.5, 1.1, 0.0],
        "use_resolution_binning": False,
    }
    request = ImageGenerationsRequest(prompt="a fox", **fields)
    assert request_model_kwargs(request, SanaSprintSamplingParams, "image") == fields


def test_sana_sprint_rejects_unusable_schedules():
    """Bad sCM schedules must fail request validation, not the worker."""
    config = SanaSprintPipelineConfig()
    bad = [
        {"trigflow_timesteps": [1.5708, 0.0]},  # needs num_inference_steps + 1
        {"trigflow_timesteps": [1.0, 1.3, 0.0]},  # not decreasing
        {"intermediate_timesteps": 1.6},  # outside (0, max_timesteps)
    ]
    for overrides in bad:
        with pytest.raises(ValueError):
            SanaSprintSamplingParams(**overrides)._validate_with_pipeline_config(config)
    SanaSprintSamplingParams(
        trigflow_timesteps=[1.5708, 1.3, 0.0]
    )._validate_with_pipeline_config(config)


def test_sana_sprint_resolution_binning_matches_diffusers():
    """Binning must pick Diffusers' 1024px aspect-ratio bin for the latent grid and
    resize/crop the decoded frames back exactly as Diffusers' image processor does."""
    from diffusers.image_processor import PixArtImageProcessor
    from diffusers.pipelines.pixart_alpha.pipeline_pixart_alpha import (
        ASPECT_RATIO_1024_BIN as DIFFUSERS_BIN,
    )

    assert ASPECT_RATIO_1024_BIN == {
        ratio: (int(h), int(w)) for ratio, (h, w) in DIFFUSERS_BIN.items()
    }
    for height, width in [(1000, 700), (800, 1200), (1024, 1024), (512, 2100)]:
        assert classify_height_width_bin(
            height, width
        ) == PixArtImageProcessor.classify_height_width_bin(
            height, width, ratios=DIFFUSERS_BIN
        )

    config = SanaSprintPipelineConfig()
    batch = SanaSprintSamplingParams(height=1000, width=700)
    # 1000 / 700 = 1.43 is closest to the 1.46 bin, (1216, 832).
    assert config.prepare_latent_shape(batch, 1, 1) == (1, 32, 1216 // 32, 832 // 32)
    batch.use_resolution_binning = False
    assert config.prepare_latent_shape(batch, 1, 1) == (1, 32, 1000 // 32, 700 // 32)

    images = torch.rand(2, 3, 1216, 832)
    expected = PixArtImageProcessor.resize_and_crop_tensor(images, 700, 1000)
    torch.testing.assert_close(_resize_and_crop(images, 700, 1000), expected)
    frames = images.unsqueeze(2)  # [B, C, F=1, H, W] as DecodingStage returns
    torch.testing.assert_close(
        _resize_and_crop(frames, 700, 1000), expected.unsqueeze(2)
    )
