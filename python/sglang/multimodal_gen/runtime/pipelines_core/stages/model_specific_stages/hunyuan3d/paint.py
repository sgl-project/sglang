# SPDX-License-Identifier: Apache-2.0
"""Hunyuan3D texture-generation stages."""

from __future__ import annotations

import concurrent.futures
import inspect
import os
from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np
import torch
from diffusers.image_processor import VaeImageProcessor
from diffusers.utils.torch_utils import randn_tensor
from einops import rearrange
from PIL import Image
from torch import nn
from transformers import PreTrainedTokenizerBase

from sglang.multimodal_gen.configs.models.encoders import BaseEncoderOutput
from sglang.multimodal_gen.configs.pipeline_configs.hunyuan3d import (
    Hunyuan3D2PipelineConfig,
)
from sglang.multimodal_gen.runtime.managers.forward_context import set_forward_context
from sglang.multimodal_gen.runtime.managers.memory_managers.component_manager import (
    ComponentUse,
)
from sglang.multimodal_gen.runtime.models.dits.hunyuan3d_paint import (
    Hunyuan3DPaintUNet,
    compute_multi_resolution_mask,
)
from sglang.multimodal_gen.runtime.models.dits.stable_diffusion import (
    StableDiffusionUNet2DConditionModel,
)
from sglang.multimodal_gen.runtime.models.encoders.clip import CLIPTextModel
from sglang.multimodal_gen.runtime.models.vaes.autoencoder import AutoencoderKL
from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import OutputBatch, Req
from sglang.multimodal_gen.runtime.pipelines_core.stages.base import (
    PipelineStage,
    StageParallelismType,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.validators import (
    StageValidators as V,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.validators import (
    VerificationResult,
)
from sglang.multimodal_gen.runtime.server_args import ServerArgs


def _module_dtype(module: nn.Module) -> torch.dtype:
    return next(module.parameters()).dtype


def _to_rgb_image(image: Image.Image, background: int = 127) -> Image.Image:
    if image.mode == "RGB":
        return image
    if image.mode != "RGBA":
        raise ValueError(f"Unsupported image mode: {image.mode}")
    background_image = Image.new("RGB", image.size, (background,) * 3)
    background_image.paste(image, mask=image.getchannel("A"))
    return background_image


def _recorrect_rgb(
    source: torch.Tensor,
    target: torch.Tensor,
    alpha: torch.Tensor,
    scale: float = 0.95,
) -> torch.Tensor:
    mask = alpha[..., 0] > 0.5
    if not torch.any(mask):
        return torch.cat([target, alpha], dim=-1)

    source_masked = source[mask]
    target_masked = target[mask]
    corrected = torch.empty_like(source)
    epsilon = torch.finfo(source.dtype).eps
    for channel in range(3):
        source_mean = source_masked[:, channel].mean()
        source_std = source_masked[:, channel].std().clamp_min(epsilon)
        target_mean = target_masked[:, channel].mean()
        target_std = target_masked[:, channel].std()
        corrected[..., channel] = (
            (source[..., channel] - scale * source_mean) * (target_std / source_std)
            + scale * target_mean
        ).clamp(0, 1)

    if torch.mean((source - target) ** 2) < torch.mean((corrected - target) ** 2):
        corrected = source
    return torch.cat([corrected, alpha], dim=-1)


def _scheduler_step_kwargs(
    scheduler: Any, generator: torch.Generator
) -> dict[str, Any]:
    parameters = inspect.signature(scheduler.step).parameters
    kwargs: dict[str, Any] = {}
    if "eta" in parameters:
        kwargs["eta"] = 0.0
    if "generator" in parameters:
        kwargs["generator"] = generator
    return kwargs


@dataclass(slots=True)
class PaintDenoisingInputs:
    timesteps: torch.Tensor
    latents: torch.Tensor
    model_kwargs: dict[str, Any]
    num_views: int
    guidance_scale: float
    use_cfg: bool
    generator: torch.Generator
    latent_channels: int


class Hunyuan3DPaintPreprocessStage(PipelineStage):
    """Unwrap the mesh, remove image lighting, and render geometry controls."""

    CAMERA_AZIMS = [0, 90, 180, 270, 0, 180]
    CAMERA_ELEVS = [0, 0, 0, 0, 90, -90]
    VIEW_WEIGHTS = [1, 0.1, 0.5, 0.1, 0.05, 0.05]

    def __init__(
        self,
        config: Hunyuan3D2PipelineConfig,
        delight_transformer: StableDiffusionUNet2DConditionModel | None,
        delight_vae: AutoencoderKL | None,
        delight_text_encoder: CLIPTextModel | None,
        delight_tokenizer: PreTrainedTokenizerBase | None,
        delight_scheduler: Any,
    ) -> None:
        super().__init__()
        self.config = config
        self.delight_transformer = delight_transformer
        self.delight_vae = delight_vae
        self.delight_text_encoder = delight_text_encoder
        self.delight_tokenizer = delight_tokenizer
        self.delight_scheduler = delight_scheduler
        self._renderer: Any = None
        self._delight_image_processor = VaeImageProcessor(vae_scale_factor=8)

        if config.delight_enable and any(
            component is None
            for component in (
                delight_transformer,
                delight_vae,
                delight_text_encoder,
                delight_tokenizer,
                delight_scheduler,
            )
        ):
            raise ValueError(
                "Delight is enabled, but its model components are missing."
            )

    @property
    def parallelism_type(self) -> StageParallelismType:
        return StageParallelismType.MAIN_RANK_ONLY

    def component_uses(
        self, server_args: ServerArgs, stage_name: str | None = None
    ) -> list[ComponentUse]:
        del server_args
        if not self.config.delight_enable:
            return []
        stage_name = self._component_stage_name(stage_name)
        return [
            ComponentUse(stage_name, "delight_text_encoder", phase="prompt"),
            ComponentUse(stage_name, "delight_vae", phase="encode"),
            ComponentUse(
                stage_name,
                "delight_transformer",
                phase="denoise",
                memory_intensive=True,
            ),
            ComponentUse(stage_name, "delight_vae", phase="decode"),
        ]

    @staticmethod
    def _unwrap_mesh(mesh: Any) -> Any:
        from sglang.multimodal_gen.runtime.utils.mesh3d_utils import mesh_uv_wrap

        return mesh_uv_wrap(mesh)

    @staticmethod
    def _load_input_image(image_path: str) -> Image.Image:
        from sglang.multimodal_gen.runtime.utils.mesh3d_utils import recenter_image

        with Image.open(image_path) as input_image:
            return recenter_image(input_image.copy())

    @staticmethod
    def _prepare_delight_target(
        image: Image.Image, device: torch.device
    ) -> tuple[Image.Image, torch.Tensor, torch.Tensor]:
        image = image.resize((512, 512), Image.Resampling.BICUBIC)
        if image.mode == "RGBA":
            pixels = np.asarray(image).copy()
            kernel = np.ones((3, 3), np.uint8)
            pixels[..., 3] = cv2.erode(pixels[..., 3], kernel, iterations=1)
            pixels[pixels[..., 3] == 0, :3] = 255
            image = Image.fromarray(pixels, mode="RGBA")
            target = torch.from_numpy(pixels.astype(np.float32) / 255).to(device)
            return image.convert("RGB"), target[..., :3], target[..., 3:]

        image = image.convert("RGB")
        target = torch.from_numpy(np.asarray(image, dtype=np.float32) / 255).to(device)
        return image, target, torch.ones_like(target[..., :1])

    def _encode_delight_prompts(
        self, use_cfg: bool, device: torch.device
    ) -> torch.Tensor:
        assert self.delight_tokenizer is not None
        assert self.delight_text_encoder is not None
        prompts = [self.config.delight_prompt]
        if use_cfg:
            prompts.append(self.config.delight_negative_prompt)
        text_inputs = self.delight_tokenizer(
            prompts,
            padding="max_length",
            max_length=self.delight_tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        with self.use_declared_component(
            component_name="delight_text_encoder",
            module=self.delight_text_encoder,
            phase="prompt",
        ) as text_encoder:
            assert isinstance(text_encoder, CLIPTextModel)
            self.delight_text_encoder = text_encoder
            output: BaseEncoderOutput = text_encoder(
                input_ids=text_inputs.input_ids.to(device),
                attention_mask=None,
            )
        prompt_embeds = output.last_hidden_state
        if prompt_embeds is None:
            raise RuntimeError("The delight text encoder returned no hidden states.")
        if not use_cfg:
            return prompt_embeds
        positive, negative = prompt_embeds.chunk(2)
        return torch.cat([positive, negative, negative])

    @torch.no_grad()
    def _run_delight(self, image: Image.Image) -> Image.Image:
        assert self.delight_transformer is not None
        assert self.delight_vae is not None
        assert self.delight_scheduler is not None

        device = self.device
        image, target_rgb, alpha = self._prepare_delight_target(image, device)
        use_cfg = (
            self.config.delight_guidance_scale > 1
            and self.config.delight_cfg_image >= 1
        )
        prompt_embeds = self._encode_delight_prompts(use_cfg, device)

        processed_image = self._delight_image_processor.preprocess(image)
        with self.use_declared_component(
            component_name="delight_vae",
            module=self.delight_vae,
            phase="encode",
        ) as vae:
            assert isinstance(vae, AutoencoderKL)
            self.delight_vae = vae
            vae_dtype = _module_dtype(vae)
            image_latents = vae.encode(
                processed_image.to(device=device, dtype=vae_dtype)
            ).latent_dist.mode()

        if use_cfg:
            image_latents = torch.cat(
                [image_latents, image_latents, torch.zeros_like(image_latents)]
            )

        scheduler = self.delight_scheduler
        scheduler.set_timesteps(self.config.delight_num_inference_steps, device=device)
        generator = torch.Generator(device="cpu").manual_seed(42)
        latent_channels = self.delight_transformer.config.out_channels
        latents = randn_tensor(
            (1, latent_channels, image_latents.shape[-2], image_latents.shape[-1]),
            generator=generator,
            device=device,
            dtype=prompt_embeds.dtype,
        )
        latents *= scheduler.init_noise_sigma
        step_kwargs = _scheduler_step_kwargs(scheduler, generator)

        with self.use_declared_component(
            component_name="delight_transformer",
            module=self.delight_transformer,
            phase="denoise",
        ) as transformer:
            assert isinstance(transformer, StableDiffusionUNet2DConditionModel)
            self.delight_transformer = transformer
            for step_index, timestep in enumerate(scheduler.timesteps):
                latent_input = torch.cat([latents] * 3) if use_cfg else latents
                latent_input = scheduler.scale_model_input(latent_input, timestep)
                latent_input = torch.cat([latent_input, image_latents], dim=1)
                with set_forward_context(
                    current_timestep=step_index,
                    attn_metadata=None,
                ):
                    noise_prediction = transformer(
                        latent_input,
                        timestep,
                        encoder_hidden_states=prompt_embeds,
                        return_dict=False,
                    )[0]
                if use_cfg:
                    text, image_only, unconditioned = noise_prediction.chunk(3)
                    noise_prediction = (
                        unconditioned
                        + self.config.delight_guidance_scale * (text - image_only)
                        + self.config.delight_cfg_image * (image_only - unconditioned)
                    )
                latents = scheduler.step(
                    noise_prediction,
                    timestep,
                    latents,
                    **step_kwargs,
                    return_dict=False,
                )[0]

        with self.use_declared_component(
            component_name="delight_vae",
            module=self.delight_vae,
            phase="decode",
        ) as vae:
            assert isinstance(vae, AutoencoderKL)
            self.delight_vae = vae
            scaling_factor = vae.config.arch_config.scaling_factor
            decoded = vae.decode(latents / scaling_factor)
        result = self._delight_image_processor.postprocess(decoded, output_type="pil")[
            0
        ]

        source_rgb = torch.from_numpy(np.asarray(result, dtype=np.float32) / 255).to(
            device
        )
        corrected = _recorrect_rgb(source_rgb, target_rgb, alpha)
        composited = corrected[..., :3] * corrected[..., 3:] + 1.0 * (
            1.0 - corrected[..., 3:]
        )
        return Image.fromarray(
            (composited.clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)
        )

    def _prepare_reference_image(self, image_path: str) -> Image.Image:
        image = self._load_input_image(image_path)
        if not self.config.delight_enable:
            return image
        return self._run_delight(image)

    def _render_multiview(
        self, mesh: Any
    ) -> tuple[list[Image.Image], list[Image.Image]]:
        if self._renderer is None:
            from sglang.multimodal_gen.runtime.utils.mesh3d_utils import MeshRender

            self._renderer = MeshRender(
                default_resolution=self.config.paint_render_size,
                texture_size=self.config.paint_texture_size,
                device=self.device,
            )
        self._renderer.load_mesh(mesh)
        normal_maps = self._renderer.render_normal_multiview(
            self.CAMERA_ELEVS, self.CAMERA_AZIMS, use_abs_coor=True
        )
        position_maps = self._renderer.render_position_multiview(
            self.CAMERA_ELEVS, self.CAMERA_AZIMS
        )
        return normal_maps, position_maps

    def forward(self, batch: Req, server_args: ServerArgs) -> Req:
        del server_args
        if batch.extra.get("_mesh_failed"):
            batch.extra.update(
                {
                    "paint_mesh": None,
                    "delighted_image": None,
                    "normal_maps": [],
                    "position_maps": [],
                    "camera_azims": self.CAMERA_AZIMS,
                    "camera_elevs": self.CAMERA_ELEVS,
                    "view_weights": self.VIEW_WEIGHTS,
                    "renderer": None,
                }
            )
            return batch

        mesh = batch.extra["shape_meshes"]
        if isinstance(mesh, list):
            mesh = mesh[0]
        if isinstance(mesh, list):
            mesh = mesh[0]
        image_path = batch.image_path
        if not isinstance(image_path, str):
            raise TypeError("Hunyuan3D Paint expects one image path per request.")

        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            mesh_future = executor.submit(self._unwrap_mesh, mesh)
            image_future = executor.submit(self._prepare_reference_image, image_path)
            paint_mesh = mesh_future.result()
            delighted_image = image_future.result()

        normal_maps, position_maps = self._render_multiview(paint_mesh)
        batch.extra.update(
            {
                "paint_mesh": paint_mesh,
                "delighted_image": delighted_image,
                "normal_maps": normal_maps,
                "position_maps": position_maps,
                "camera_azims": self.CAMERA_AZIMS,
                "camera_elevs": self.CAMERA_ELEVS,
                "view_weights": self.VIEW_WEIGHTS,
                "renderer": self._renderer,
            }
        )
        return batch

    def verify_input(self, batch: Req, server_args: ServerArgs) -> VerificationResult:
        del server_args
        result = VerificationResult()
        result.add_check("shape_meshes", batch.extra.get("shape_meshes"), V.not_none)
        result.add_check("image_path", batch.image_path, V.not_none)
        return result

    def verify_output(self, batch: Req, server_args: ServerArgs) -> VerificationResult:
        del server_args
        if batch.extra.get("_mesh_failed"):
            return VerificationResult()
        result = VerificationResult()
        result.add_check("paint_mesh", batch.extra.get("paint_mesh"), V.not_none)
        result.add_check(
            "delighted_image", batch.extra.get("delighted_image"), V.not_none
        )
        result.add_check("normal_maps", batch.extra.get("normal_maps"), V.is_list)
        result.add_check("position_maps", batch.extra.get("position_maps"), V.is_list)
        result.add_check("renderer", batch.extra.get("renderer"), V.not_none)
        return result


class Hunyuan3DPaintTexGenStage(PipelineStage):
    """Generate consistent multi-view textures from geometry controls."""

    def __init__(
        self,
        config: Hunyuan3D2PipelineConfig,
        transformer: Hunyuan3DPaintUNet,
        scheduler: Any,
        vae: AutoencoderKL,
    ) -> None:
        super().__init__()
        self.config = config
        self.transformer = transformer
        self.scheduler = scheduler
        self.vae = vae
        block_channels = vae.config.arch_config.block_out_channels
        self.vae_scale_factor = 2 ** (len(block_channels) - 1)
        self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor)

    @property
    def parallelism_type(self) -> StageParallelismType:
        return StageParallelismType.MAIN_RANK_ONLY

    def component_uses(
        self, server_args: ServerArgs, stage_name: str | None = None
    ) -> list[ComponentUse]:
        del server_args
        stage_name = self._component_stage_name(stage_name)
        return [
            ComponentUse(stage_name, "paint_vae", phase="encode"),
            ComponentUse(
                stage_name,
                "paint_transformer",
                phase="denoise",
                memory_intensive=True,
            ),
            ComponentUse(stage_name, "paint_vae", phase="decode"),
        ]

    @staticmethod
    def _pil_views_to_tensor(
        images: list[Image.Image],
        size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        tensors = []
        for image in images:
            image = image.resize((size, size), Image.Resampling.BICUBIC)
            if image.mode == "L":
                image = image.point(lambda value: 255 if value > 1 else 0).convert(
                    "RGB"
                )
            pixels = np.asarray(image, dtype=np.float32) / 255
            if pixels.shape[-1] == 4:
                alpha = pixels[..., 3:]
                pixels = pixels[..., :3] * alpha + (1 - alpha)
            tensor = torch.from_numpy(pixels).permute(2, 0, 1).contiguous()
            tensors.append(tensor)
        return torch.stack(tensors).unsqueeze(0).to(device=device, dtype=dtype)

    @staticmethod
    def _encode_images(
        vae: AutoencoderKL,
        images: torch.Tensor,
        generator: torch.Generator,
    ) -> torch.Tensor:
        batch_size, num_images = images.shape[:2]
        images = rearrange(images, "b n c h w -> (b n) c h w")
        images = images.mul(2).sub(1)
        posterior = vae.encode(images).latent_dist
        scaling_factor = vae.config.arch_config.scaling_factor
        latents = posterior.sample(generator=generator) * scaling_factor
        return rearrange(
            latents,
            "(b n) c h w -> b n c h w",
            b=batch_size,
            n=num_images,
        )

    @staticmethod
    def _camera_index(azimuth: float, elevation: float) -> int:
        base_index = int(((azimuth // 30) + 9) % 12)
        if elevation == 0:
            base, divisor = 12, 1
        elif elevation == 20:
            base, divisor = 24, 1
        elif elevation == -20:
            base, divisor = 0, 1
        elif elevation == 90:
            base, divisor = 40, 3
        elif elevation == -90:
            base, divisor = 36, 3
        else:
            base, divisor = 12, 1
        return base + base_index // divisor

    def _timesteps(self, device: torch.device) -> torch.Tensor:
        if not self.config.paint_turbo_mode:
            self.scheduler.set_timesteps(
                self.config.paint_num_inference_steps, device=device
            )
            return self.scheduler.timesteps

        self.scheduler.set_timesteps(
            num_inference_steps=10,
            original_inference_steps=30,
            device=device,
        )
        return self.scheduler.timesteps

    def _prepare_denoising_inputs(self, batch: Req) -> PaintDenoisingInputs:
        device = self.device
        render_size = self.config.paint_resolution
        normal_maps = batch.extra["normal_maps"]
        position_maps = batch.extra["position_maps"]
        if not isinstance(normal_maps, list) or not isinstance(position_maps, list):
            raise TypeError("Hunyuan3D Paint geometry controls must be image lists.")

        reference = batch.extra["delighted_image"]
        references = reference if isinstance(reference, list) else [reference]
        reference_images = [
            _to_rgb_image(image).resize(
                (render_size, render_size), Image.Resampling.BICUBIC
            )
            for image in references
        ]
        vae_generator = torch.Generator(device=device).manual_seed(0)
        latent_generator = torch.Generator(device=device).manual_seed(0)

        with self.use_declared_component(
            component_name="paint_vae", module=self.vae, phase="encode"
        ) as vae:
            assert isinstance(vae, AutoencoderKL)
            self.vae = vae
            vae_dtype = _module_dtype(vae)
            reference_tensor = self._pil_views_to_tensor(
                reference_images, render_size, device, vae_dtype
            )
            normal_tensor = self._pil_views_to_tensor(
                normal_maps, render_size, device, vae_dtype
            )
            position_tensor = self._pil_views_to_tensor(
                position_maps, render_size, device, vae_dtype
            )
            reference_latents = self._encode_images(
                vae, reference_tensor, vae_generator
            )
            normal_latents = self._encode_images(vae, normal_tensor, vae_generator)
            position_latents = self._encode_images(vae, position_tensor, vae_generator)

        camera_info = [
            self._camera_index(azimuth, elevation)
            for azimuth, elevation in zip(
                batch.extra["camera_azims"], batch.extra["camera_elevs"]
            )
        ]
        camera_info_gen = torch.tensor([camera_info], device=device, dtype=torch.int64)
        camera_info_ref = torch.zeros((1, 1), device=device, dtype=torch.int64)
        use_cfg = (
            self.config.paint_guidance_scale > 1 and not self.config.paint_turbo_mode
        )

        position_attention_mask = None
        if self.config.paint_turbo_mode:
            position_attention_mask = compute_multi_resolution_mask(position_tensor)

        reference_scale: torch.Tensor | float = 1.0
        if use_cfg:
            reference_latents = torch.cat(
                [torch.zeros_like(reference_latents), reference_latents]
            )
            reference_scale = torch.as_tensor(
                [0.0, 1.0], device=device, dtype=reference_latents.dtype
            )
            normal_latents = torch.cat([normal_latents, normal_latents])
            position_latents = torch.cat([position_latents, position_latents])
            camera_info_gen = torch.cat([camera_info_gen, camera_info_gen])
            camera_info_ref = torch.cat([camera_info_ref, camera_info_ref])

        num_views = len(normal_maps)
        model_kwargs: dict[str, Any] = {
            "ref_latents": reference_latents,
            "num_in_batch": num_views,
            "condition_embed_dict": {},
            "normal_imgs": normal_latents,
            "position_imgs": position_latents,
            "camera_info_gen": camera_info_gen,
            "camera_info_ref": camera_info_ref,
            "ref_scale": reference_scale,
        }
        if position_attention_mask is not None:
            model_kwargs["position_attn_mask"] = position_attention_mask

        timesteps = self._timesteps(device)
        latent_channels = self.transformer.config.in_channels
        latent_size = render_size // self.vae_scale_factor
        latents = randn_tensor(
            (num_views, latent_channels, latent_size, latent_size),
            generator=latent_generator,
            device=device,
            dtype=_module_dtype(self.transformer),
        )
        latents *= self.scheduler.init_noise_sigma
        return PaintDenoisingInputs(
            timesteps=timesteps,
            latents=latents,
            model_kwargs=model_kwargs,
            num_views=num_views,
            guidance_scale=self.config.paint_guidance_scale,
            use_cfg=use_cfg,
            generator=latent_generator,
            latent_channels=latent_channels,
        )

    @torch.no_grad()
    def _denoise(self, inputs: PaintDenoisingInputs) -> torch.Tensor:
        scheduler = self.scheduler
        latents = inputs.latents
        step_kwargs = _scheduler_step_kwargs(scheduler, inputs.generator)
        with self.use_declared_component(
            component_name="paint_transformer",
            module=self.transformer,
            phase="denoise",
        ) as transformer:
            assert isinstance(transformer, Hunyuan3DPaintUNet)
            self.transformer = transformer
            prompt_embeds = transformer.learned_text_clip_gen
            if inputs.use_cfg:
                prompt_embeds = torch.cat(
                    [torch.zeros_like(prompt_embeds), prompt_embeds]
                )

            for step_index, timestep in enumerate(inputs.timesteps):
                latents = rearrange(
                    latents, "(b n) c h w -> b n c h w", n=inputs.num_views
                )
                latent_input = (
                    torch.cat([latents, latents]) if inputs.use_cfg else latents
                )
                latent_input = rearrange(latent_input, "b n c h w -> (b n) c h w")
                latent_input = scheduler.scale_model_input(latent_input, timestep)
                latent_input = rearrange(
                    latent_input,
                    "(b n) c h w -> b n c h w",
                    n=inputs.num_views,
                )
                with set_forward_context(
                    current_timestep=step_index,
                    attn_metadata=None,
                ):
                    noise_prediction = transformer(
                        latent_input,
                        timestep,
                        encoder_hidden_states=prompt_embeds,
                        return_dict=False,
                        **inputs.model_kwargs,
                    )[0]
                latents = rearrange(latents, "b n c h w -> (b n) c h w")
                if inputs.use_cfg:
                    unconditioned, conditioned = noise_prediction.chunk(2)
                    noise_prediction = unconditioned + inputs.guidance_scale * (
                        conditioned - unconditioned
                    )
                latents = scheduler.step(
                    noise_prediction,
                    timestep,
                    latents[:, : inputs.latent_channels],
                    **step_kwargs,
                    return_dict=False,
                )[0]
        return latents

    @torch.no_grad()
    def _decode(self, latents: torch.Tensor) -> list[Image.Image]:
        with self.use_declared_component(
            component_name="paint_vae", module=self.vae, phase="decode"
        ) as vae:
            assert isinstance(vae, AutoencoderKL)
            self.vae = vae
            scaling_factor = vae.config.arch_config.scaling_factor
            decoded = vae.decode(latents / scaling_factor)
        return self.image_processor.postprocess(decoded, output_type="pil")

    def forward(self, batch: Req, server_args: ServerArgs) -> Req:
        del server_args
        if batch.extra.get("_mesh_failed"):
            batch.extra["multiview_textures"] = []
            return batch
        inputs = self._prepare_denoising_inputs(batch)
        batch.extra["multiview_textures"] = self._decode(self._denoise(inputs))
        return batch

    def verify_input(self, batch: Req, server_args: ServerArgs) -> VerificationResult:
        del server_args
        if batch.extra.get("_mesh_failed"):
            return VerificationResult()
        result = VerificationResult()
        result.add_check(
            "delighted_image", batch.extra.get("delighted_image"), V.not_none
        )
        result.add_check("normal_maps", batch.extra.get("normal_maps"), V.is_list)
        result.add_check("position_maps", batch.extra.get("position_maps"), V.is_list)
        result.add_check("camera_azims", batch.extra.get("camera_azims"), V.is_list)
        result.add_check("camera_elevs", batch.extra.get("camera_elevs"), V.is_list)
        return result

    def verify_output(self, batch: Req, server_args: ServerArgs) -> VerificationResult:
        del server_args
        result = VerificationResult()
        result.add_check(
            "multiview_textures", batch.extra.get("multiview_textures"), V.is_list
        )
        return result


class Hunyuan3DPaintPostprocessStage(PipelineStage):
    """Bake generated views into a texture and export the final mesh."""

    def __init__(self, config: Hunyuan3D2PipelineConfig) -> None:
        super().__init__()
        self.config = config

    @property
    def parallelism_type(self) -> StageParallelismType:
        return StageParallelismType.MAIN_RANK_ONLY

    @staticmethod
    def _cleanup_obj_artifacts(obj_path: str, files_before_export: set[str]) -> None:
        obj_dir = os.path.dirname(obj_path) or "."
        generated_files = set(os.listdir(obj_dir)) - files_before_export
        cleanup_paths = {obj_path}
        cleanup_paths.update(
            os.path.join(obj_dir, filename)
            for filename in generated_files
            if filename.endswith(".mtl") or filename.endswith(".png")
        )
        for path in cleanup_paths:
            try:
                os.remove(path)
            except FileNotFoundError:
                continue

    def forward(self, batch: Req, server_args: ServerArgs) -> OutputBatch:
        del server_args
        if batch.is_warmup or batch.extra.get("_mesh_failed"):
            return OutputBatch(output_file_paths=[], metrics=batch.metrics)

        renderer = batch.extra["renderer"]
        textures = [
            image.resize(
                (self.config.paint_render_size, self.config.paint_render_size),
                Image.Resampling.BICUBIC,
            )
            for image in batch.extra["multiview_textures"]
        ]
        texture, mask = renderer.bake_from_multiview(
            textures,
            batch.extra["camera_elevs"],
            batch.extra["camera_azims"],
            batch.extra["view_weights"],
            method="fast",
        )
        mask_array = (mask.squeeze(-1).cpu().numpy() * 255).astype(np.uint8)
        texture = renderer.texture_inpaint(texture, mask_array)
        renderer.set_texture(texture)
        textured_mesh = renderer.save_mesh()

        obj_path = batch.extra["shape_obj_path"]
        return_path = batch.extra["shape_return_path"]
        obj_dir = os.path.dirname(obj_path) or "."
        files_before_export = set(os.listdir(obj_dir))
        textured_mesh.export(obj_path)
        if self.config.paint_save_glb:
            return_path = os.path.splitext(obj_path)[0] + ".glb"
            textured_mesh.export(return_path)
            self._cleanup_obj_artifacts(obj_path, files_before_export)

        return OutputBatch(output_file_paths=[return_path], metrics=batch.metrics)

    def verify_input(self, batch: Req, server_args: ServerArgs) -> VerificationResult:
        del server_args
        if batch.extra.get("_mesh_failed"):
            return VerificationResult()
        result = VerificationResult()
        result.add_check("renderer", batch.extra.get("renderer"), V.not_none)
        result.add_check(
            "multiview_textures", batch.extra.get("multiview_textures"), V.is_list
        )
        result.add_check("camera_elevs", batch.extra.get("camera_elevs"), V.is_list)
        result.add_check("camera_azims", batch.extra.get("camera_azims"), V.is_list)
        result.add_check("view_weights", batch.extra.get("view_weights"), V.is_list)
        return result


__all__ = [
    "Hunyuan3DPaintPreprocessStage",
    "Hunyuan3DPaintTexGenStage",
    "Hunyuan3DPaintPostprocessStage",
]
