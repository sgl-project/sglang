import pytest
import torch
from diffusers import SCMScheduler as DiffusersSCMScheduler

from sglang.multimodal_gen.configs.models.dits.sana import SanaConfig
from sglang.multimodal_gen.configs.pipeline_configs.sana import SanaPipelineConfig
from sglang.multimodal_gen.configs.pipeline_configs.sana_sprint import (
    SanaSprintPipelineConfig,
)
from sglang.multimodal_gen.configs.sample.sana import SanaSamplingParams
from sglang.multimodal_gen.configs.sample.sana_sprint import SanaSprintSamplingParams
from sglang.multimodal_gen.registry import get_model_info
from sglang.multimodal_gen.runtime.models.schedulers.scheduling_scm import (
    SCMScheduler,
)

_SIGMA_DATA = 0.5


def _fake_dit(latent_model_input: torch.Tensor, scm_timestep) -> torch.Tensor:
    """A deterministic stand-in for the bf16 DiT."""
    weight = torch.linspace(-1.0, 1.0, 64).view(8, 8)
    mixed = torch.einsum("bchw,cd->bdhw", latent_model_input.float(), weight)
    return (torch.tanh(mixed) * (1 + scm_timestep)).to(torch.bfloat16)


def _diffusers_sprint_loop(steps: int) -> torch.Tensor:
    """The SanaSprintPipeline.__call__ denoising loop, reduced to the scheduler math."""
    scheduler = DiffusersSCMScheduler()
    scheduler.set_timesteps(
        steps,
        max_timesteps=1.57080,
        intermediate_timesteps=1.3 if steps == 2 else None,
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


def _shared_denoising_loop(steps: int) -> torch.Tensor:
    """How LatentPreparationStage and DenoisingStage drive the scheduler."""
    scheduler = SCMScheduler()
    scheduler.set_timesteps(steps)
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


@pytest.mark.parametrize("steps", [1, 2, 4])
def test_scm_scheduler_reproduces_diffusers_sprint_loop(steps):
    """The SCM scheduler must reproduce Diffusers' SanaSprintPipeline loop exactly:
    the sCM timestep the DiT sees, its input scaling, the fp32 flow-to-TrigFlow
    conversion of a bf16 DiT output, the step noise and the final /sigma_data."""
    torch.testing.assert_close(
        _shared_denoising_loop(steps), _diffusers_sprint_loop(steps), rtol=0, atol=0
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
