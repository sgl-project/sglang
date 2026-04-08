from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn as nn

import sglang.multimodal_gen.runtime.pipelines_core.stages.denoising as denoising_module
from sglang.multimodal_gen.configs.pipeline_configs.base import ModelTaskType
from sglang.multimodal_gen.runtime import server_args as server_args_module
from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import Req
from sglang.multimodal_gen.runtime.pipelines_core.stages.denoising import (
    DenoisingStage,
)
from sglang.multimodal_gen.runtime.server_args import set_global_server_args


class DummyTransformer(nn.Module):
    def forward(self, hidden_states, timestep, **kwargs):
        return torch.zeros_like(hidden_states)


class DummyAttnBackend:
    def get_enum(self):
        return "DUMMY"

    def get_builder_cls(self):
        return None


class DummyScheduler:
    def __init__(self, num_steps: int) -> None:
        self.num_inference_steps = num_steps
        self.timesteps = torch.arange(num_steps, 0, -1, dtype=torch.float32)
        self.order = 1

    def set_timesteps(self, num_steps, device=None):
        self.timesteps = torch.arange(
            num_steps, 0, -1, dtype=torch.float32, device=device
        )

    def set_begin_index(self, index: int) -> None:
        return None

    def scale_model_input(self, sample, timestep):
        return sample

    def step(self, model_output, timestep, sample, **kwargs):
        return (sample,)


@dataclass
class DummyDiTConfig:
    hidden_size: int = 64
    num_attention_heads: int = 8


@dataclass
class DummyPipelineConfig:
    task_type: ModelTaskType = ModelTaskType.T2I
    embedded_cfg_scale: float = 1.0
    should_use_guidance: bool = False
    dit_config: DummyDiTConfig = field(default_factory=DummyDiTConfig)

    def slice_noise_pred(self, noise_pred, latents):
        return noise_pred

    def get_classifier_free_guidance_scale(self, batch, guidance_scale):
        return guidance_scale

    def postprocess_cfg_noise(self, batch, noise_pred, noise_pred_cond):
        return noise_pred

    def post_denoising_loop(self, latents, batch):
        return latents


@dataclass
class DummyServerArgs:
    pipeline_config: DummyPipelineConfig = field(default_factory=DummyPipelineConfig)
    enable_cfg_parallel: bool = False
    comfyui_mode: bool = False
    disable_autocast: bool = True
    enable_torch_compile: bool = False
    device: str = "cpu"


class MinimalDenoisingStage(DenoisingStage):
    def _prepare_denoising_loop(self, batch: Req, server_args: DummyServerArgs):
        timesteps = batch.timesteps
        num_inference_steps = batch.num_inference_steps
        return {
            "extra_step_kwargs": {},
            "target_dtype": batch.latents.dtype,
            "autocast_enabled": False,
            "timesteps": timesteps,
            "num_inference_steps": num_inference_steps,
            "num_warmup_steps": 0,
            "image_kwargs": {},
            "pos_cond_kwargs": {},
            "neg_cond_kwargs": {},
            "latents": batch.latents,
            "boundary_timestep": None,
            "z": None,
            "reserved_frames_mask": None,
            "seq_len": None,
            "guidance": None,
            "workspace": None,
        }

    def _build_attn_metadata(self, *args, **kwargs):
        return None

    def _select_and_manage_model(self, t_int, boundary_timestep, server_args, batch):
        return self.transformer, batch.guidance_scale

    def _predict_noise_with_cfg(
        self,
        current_model,
        latent_model_input,
        timestep,
        **kwargs,
    ):
        return current_model(hidden_states=latent_model_input, timestep=timestep)

    def progress_bar(self, iterable=None, total=None):
        class _Bar:
            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

            def update(self, *args, **kwargs):
                return None

        return _Bar()

    def step_profile(self):
        return None


def test_dummy_denoising_loop_accuracy():
    device = torch.device("cpu")
    batch_size = 1
    channels = 4
    spatial_size = 8
    num_steps = 4

    transformer = DummyTransformer().to(device)
    scheduler = DummyScheduler(num_steps=num_steps)
    server_args = DummyServerArgs()
    previous_server_args = server_args_module._global_server_args
    set_global_server_args(server_args)
    previous_get_attn_backend = denoising_module.get_attn_backend
    previous_get_sp_world_size = denoising_module.get_sp_world_size
    try:
        denoising_module.get_attn_backend = lambda *args, **kwargs: DummyAttnBackend()
        denoising_module.get_sp_world_size = lambda: 1
        stage = MinimalDenoisingStage(transformer=transformer, scheduler=scheduler)

        latents = torch.randn(
            batch_size, channels, spatial_size, spatial_size, device=device
        )
        batch = Req(
            prompt="dummy",
            raw_latent_shape=[batch_size, channels, spatial_size, spatial_size],
            latents=latents.clone(),
            timesteps=scheduler.timesteps.to(device),
            num_inference_steps=num_steps,
            guidance_scale=1.0,
            do_classifier_free_guidance=False,
            return_trajectory_latents=False,
        )
        batch.extra = {}
        result = stage.forward(batch=batch, server_args=server_args)

        assert torch.allclose(result.latents, latents)
    finally:
        denoising_module.get_attn_backend = previous_get_attn_backend
        denoising_module.get_sp_world_size = previous_get_sp_world_size
        server_args_module._global_server_args = previous_server_args
