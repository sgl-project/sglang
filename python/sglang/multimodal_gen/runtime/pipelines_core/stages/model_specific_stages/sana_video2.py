# Copyright 2024-2026 NVIDIA CORPORATION & AFFILIATES
# SPDX-License-Identifier: Apache-2.0
"""Prompt conditioning and request-local flow solvers for SANA-Video 2.0."""

import numpy as np
import torch
from diffusers import FlowMatchEulerDiscreteScheduler
from diffusers.utils.torch_utils import randn_tensor
from PIL import Image

from sglang.multimodal_gen.runtime.disaggregation.roles import RoleType
from sglang.multimodal_gen.runtime.managers.memory_managers.component_manager import (
    ComponentUse,
)
from sglang.multimodal_gen.runtime.pipelines.sana_video import (
    select_sana_video_prompt_window,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.base import PipelineStage
from sglang.multimodal_gen.runtime.pipelines_core.stages.text_encoding import (
    TextEncodingStage,
)


def sample_flow_dpm(predict_noise, latents, steps, shift, callback=None):
    """Second-order multistep DPM-Solver++ with first-order endpoint steps."""
    if steps < 2:
        raise ValueError("Flow DPM-Solver requires at least two inference steps")
    betas = torch.linspace(1.0, 0.001, steps + 1, device="cpu").to(latents.device)
    sigmas = 1.0 - betas
    times = (shift * sigmas / (1 + (shift - 1) * sigmas)).flip(0)
    previous_time = previous_prediction = None
    x = latents
    for index in range(steps):
        s = times[index].reshape((1,) * x.ndim)
        t = times[index + 1].reshape((1,) * x.ndim)
        noise = predict_noise(x, times[index])
        prediction = (x - s * noise) / (1 - s)
        lambda_s = torch.log(1 - s) - torch.log(s)
        lambda_t = torch.log(1 - t) - torch.log(t)
        h = lambda_t - lambda_s
        alpha_t = torch.exp(torch.log(1 - t))
        phi = torch.expm1(-h)
        if index == 0 or index == steps - 1:
            x = t / s * x - alpha_t * phi * prediction
        else:
            lambda_previous = torch.log(1 - previous_time) - torch.log(previous_time)
            ratio = (lambda_s - lambda_previous) / h
            derivative = (1.0 / ratio) * (prediction - previous_prediction)
            x = (
                t / s * x
                - alpha_t * phi * prediction
                - 0.5 * (alpha_t * phi) * derivative
            )
        previous_time, previous_prediction = s, prediction
        if callback is not None:
            callback(index, times[index + 1], x)
    return x


def sample_ltx_euler(predict_flow, latents, steps, shift, callback=None):
    """Flow Euler with a clean, fixed first latent frame."""
    with torch.device("cpu"):
        scheduler = FlowMatchEulerDiscreteScheduler(shift=shift)
        scheduler.set_timesteps(steps, device=latents.device)
    condition_mask = torch.zeros_like(latents, dtype=torch.float32)
    condition_mask[:, :, 0] = 1
    for index, time in enumerate(scheduler.timesteps):
        timestep = torch.minimum(
            time.expand(latents.shape).float(), (1 - condition_mask) * 1000.0
        )
        prediction = predict_flow(latents, timestep[:, :1, :, :1, :1])
        batch, channels = latents.shape[:2]
        updated = (
            scheduler.step(
                -prediction.reshape(batch, channels, -1).transpose(1, 2),
                time,
                latents.reshape(batch, channels, -1).transpose(1, 2),
                per_token_timesteps=timestep.reshape(batch, channels, -1)[:, 0],
                return_dict=False,
            )[0]
            .transpose(1, 2)
            .reshape(latents.shape)
        )
        denoise_mask = time / 1000 - 1e-6 < (1.0 - condition_mask)
        latents = torch.where(denoise_mask, updated, latents).to(latents.dtype)
        if callback is not None:
            callback(index, time, latents)
    return latents


class SanaVideo2TextEncodingStage(TextEncodingStage):
    def __init__(self, text_encoders, tokenizers, instruction):
        super().__init__(text_encoders, tokenizers)
        self.instruction = instruction

    @torch.no_grad()
    def forward(self, batch, server_args):
        self.tokenizers[0].padding_side = "right"
        length = batch.max_sequence_length or 300
        prompts = [batch.prompt] if isinstance(batch.prompt, str) else batch.prompt
        motion = batch.extra.get("motion_score", 10)
        suffix = (
            f" motion score: {int(motion)}."
            if motion > 0
            else ""
            if motion < 0
            else " high motion"
            if batch.extra.get("high_motion", False)
            else " low motion"
        )
        prompts = [self.instruction + prompt.strip() + suffix for prompt in prompts]
        encoded_length = (
            len(self.tokenizers[0].encode(self.instruction)) + length - 2
            if self.instruction
            else length
        )
        outputs = list(
            self.encode_text(
                prompts,
                server_args,
                return_attention_mask=True,
                max_length=encoded_length,
            )
        )
        for index in (0, 1, 3):
            outputs[index] = [
                select_sana_video_prompt_window(value, length)
                for value in outputs[index]
            ]
        outputs[4] = [
            [int(value) for value in mask.sum(dim=1).tolist()] for mask in outputs[1]
        ]
        self._append_positive_text_outputs(batch, *outputs)
        if batch.do_classifier_free_guidance:
            negative = self.encode_text(
                batch.negative_prompt,
                server_args,
                return_attention_mask=True,
                max_length=length,
            )
            self._append_negative_text_outputs(batch, outputs[0], *negative)
        return batch


def prepare_image(image, height, width):
    image = image.convert("RGB")
    w, h = image.size
    scale = max(height / h, width / w)
    resized = image.resize(
        (round(w * scale), round(h * scale)), Image.Resampling.BICUBIC
    )
    top = round((resized.height - height) / 2.0)
    left = round((resized.width - width) / 2.0)
    pixels = np.array(resized)[top : top + height, left : left + width].copy()
    return (
        torch.from_numpy(pixels).permute(2, 0, 1).float().div_(255).sub_(0.5).div_(0.5)
    )


class SanaVideo2LatentPreparationStage(PipelineStage):
    def __init__(self, vae):
        super().__init__()
        self.vae = vae

    def component_uses(self, server_args, stage_name=None):
        return [ComponentUse(self._component_stage_name(stage_name), "vae")]

    @torch.no_grad()
    def forward(self, batch, server_args):
        config = server_args.pipeline_config
        if batch.height % 32 or batch.width % 32:
            raise ValueError("SANA-Video 2.0 height and width must be divisible by 32")
        batch.num_frames = config.adjust_num_frames(batch.num_frames)
        device = batch.prompt_embeds[0].device
        batch_size = batch.prompt_embeds[0].shape[0]
        shape = config.prepare_latent_shape(
            batch, batch_size, (batch.num_frames - 1) // 8 + 1
        )
        if batch.latents is None:
            batch.latents = randn_tensor(
                shape, generator=batch.generator, device=device, dtype=torch.float32
            )
        else:
            if tuple(batch.latents.shape) != shape:
                raise ValueError(
                    f"Expected latents with shape {shape}, got {tuple(batch.latents.shape)}"
                )
            batch.latents = batch.latents.to(device=device, dtype=torch.float32).clone()
        if batch.condition_image is not None:
            images = (
                batch.condition_image
                if isinstance(batch.condition_image, list)
                else [batch.condition_image]
            )
            if len(images) == 1:
                images = images * batch_size
            if len(images) != batch_size:
                raise ValueError("TI2V requires one conditioning image per prompt")
            self.begin_declared_component_use(component_name="vae", module=self.vae)
            pixels = torch.stack(
                [prepare_image(image, batch.height, batch.width) for image in images]
            )[:, :, None]
            pixels = pixels.to(device=device, dtype=next(self.vae.parameters()).dtype)
            image_latents = self.vae.encode(pixels).latent_dist.mode()
            scale, mean = config.get_decode_scale_and_shift(
                device, image_latents.dtype, self.vae
            )
            batch.latents[:, :, :1] = (image_latents - mean) * scale
        return batch


class SanaVideo2DenoisingStage(PipelineStage):
    def __init__(self, transformer):
        super().__init__()
        self.transformer = transformer
        self._bcg_runner = None

    @property
    def role_affinity(self) -> RoleType:
        return RoleType.DENOISER

    def component_uses(self, server_args, stage_name=None):
        return [ComponentUse(self._component_stage_name(stage_name), "transformer")]

    def default_workload_iterations(self, batch, num_inference_steps):
        return num_inference_steps

    @torch.no_grad()
    def forward(self, batch, server_args):
        self.begin_declared_component_use(
            component_name="transformer", module=self.transformer
        )
        cfg = batch.do_classifier_free_guidance
        embeds = batch.prompt_embeds[0]
        mask = batch.prompt_attention_mask[0]
        if cfg:
            embeds = torch.cat([batch.negative_prompt_embeds[0], embeds])
            mask = torch.cat([batch.negative_attention_mask[0], mask])
        is_ti2v = batch.condition_image is not None
        runner = None
        if server_args.enable_breakable_cuda_graph:
            runner = self._bcg_runner
            if runner is None:
                from sglang.multimodal_gen.runtime.breakable_cuda_graph.runner import (
                    DiffusionBreakableCudaGraphRunner,
                )

                runner = DiffusionBreakableCudaGraphRunner(
                    self.transformer, batch.latents.device
                )
                self._bcg_runner = runner
        captured_warmup = False

        def predict(x, time):
            nonlocal captured_warmup
            inputs = torch.cat([x, x]) if cfg else x
            if time.ndim == 0:
                timestep = (time * 1000).expand(inputs.shape[0])
            else:
                timestep = torch.cat([time, time]) if cfg else time
            call_kwargs = dict(
                hidden_states=inputs,
                timestep=timestep,
                encoder_hidden_states=embeds,
                encoder_attention_mask=mask,
            )
            if runner is not None:
                if batch.is_warmup and not captured_warmup:
                    runner.capture(**call_kwargs)
                    if is_ti2v:
                        other_timestep = timestep.flatten(1)[:, -1]
                    else:
                        other_timestep = (
                            timestep[:, None, None, None, None]
                            .expand(-1, 1, inputs.shape[2], 1, 1)
                            .clone()
                        )
                        other_timestep[:, :, 0] = 0
                    runner.capture(**{**call_kwargs, "timestep": other_timestep})
                    captured_warmup = True
                prediction = runner(**call_kwargs)
            else:
                prediction = self.transformer(**call_kwargs)
            if not is_ti2v:
                # Keep flow-to-noise conversion before CFG to preserve FP32 rounding.
                sigma = time.reshape((1,) * x.ndim).to(inputs)
                prediction = (1 - sigma) * prediction + inputs
            if cfg:
                uncond, cond = prediction.chunk(2)
                prediction = uncond + batch.guidance_scale * (cond - uncond)
            return prediction

        steps = (
            max(2, batch.num_inference_steps)
            if batch.is_warmup
            else batch.num_inference_steps
        )
        sampler = sample_ltx_euler if is_ti2v else sample_flow_dpm
        flow_shift = (
            batch.flow_shift
            if batch.flow_shift is not None
            else server_args.pipeline_config.flow_shift
        )
        with self.progress_bar(total=steps, batch=batch) as progress:
            batch.latents = sampler(
                predict,
                batch.latents,
                steps,
                flow_shift,
                callback=lambda *_: progress.update(),
            )
        return batch
