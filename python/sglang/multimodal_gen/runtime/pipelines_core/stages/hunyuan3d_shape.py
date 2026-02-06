# SPDX-License-Identifier: Apache-2.0
"""
Hunyuan3D shape generation stages.

This module contains the pipeline stages for Hunyuan3D 3D shape generation,
including preprocessing, conditioning, denoising, and export stages.
"""

from __future__ import annotations

import os
from typing import Any

import numpy as np
import torch

from sglang.multimodal_gen.configs.pipeline_configs.hunyuan3d import (
    Hunyuan3D2PipelineConfig,
)
from sglang.multimodal_gen.runtime.loader.component_loader import TransformerLoader
from sglang.multimodal_gen.runtime.managers.forward_context import set_forward_context
from sglang.multimodal_gen.runtime.models.mesh3d_utils import export_to_trimesh
from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import OutputBatch, Req
from sglang.multimodal_gen.runtime.pipelines_core.stages.base import PipelineStage
from sglang.multimodal_gen.runtime.pipelines_core.stages.denoising import DenoisingStage
from sglang.multimodal_gen.runtime.pipelines_core.stages.validators import (
    StageValidators as V,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.validators import (
    VerificationResult,
)
from sglang.multimodal_gen.runtime.server_args import ServerArgs
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

logger = init_logger(__name__)


def retrieve_timesteps(
    scheduler,
    num_inference_steps=None,
    device=None,
    timesteps=None,
    sigmas=None,
    **kwargs,
):
    """Retrieve timesteps from scheduler.

    Calls the scheduler's set_timesteps method and retrieves timesteps.
    Handles custom timesteps and sigmas.
    """
    import inspect

    if timesteps is not None and sigmas is not None:
        raise ValueError("Only one of timesteps or sigmas can be passed.")

    if timesteps is not None:
        accepts_timesteps = "timesteps" in set(
            inspect.signature(scheduler.set_timesteps).parameters.keys()
        )
        if not accepts_timesteps:
            raise ValueError(
                f"Scheduler {scheduler.__class__} doesn't support custom timesteps."
            )
        scheduler.set_timesteps(timesteps=timesteps, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)

    elif sigmas is not None:
        accepts_sigmas = "sigmas" in set(
            inspect.signature(scheduler.set_timesteps).parameters.keys()
        )
        if not accepts_sigmas:
            raise ValueError(
                f"Scheduler {scheduler.__class__} doesn't support custom sigmas."
            )
        scheduler.set_timesteps(sigmas=sigmas, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)

    else:
        scheduler.set_timesteps(num_inference_steps, device=device, **kwargs)
        timesteps = scheduler.timesteps

    return timesteps, num_inference_steps


def _prepare_shape_image(image_processor, image, mask=None) -> dict:
    """Prepare shape image for conditioning."""
    if isinstance(image, torch.Tensor) and isinstance(mask, torch.Tensor):
        return {"image": image, "mask": mask}

    if isinstance(image, str) and not os.path.exists(image):
        raise FileNotFoundError(f"Couldn't find image at path {image}")

    if not isinstance(image, list):
        image = [image]

    outputs = [image_processor(img) for img in image]
    cond_input = {k: [] for k in outputs[0].keys()}
    for output in outputs:
        for key, value in output.items():
            cond_input[key].append(value)
    for key, value in cond_input.items():
        if isinstance(value[0], torch.Tensor):
            cond_input[key] = torch.cat(value, dim=0)
    return cond_input


def _move_to_device(payload, device, dtype):
    """Recursively move tensors in payload to specified device and dtype."""
    if isinstance(payload, torch.Tensor):
        return payload.to(device=device, dtype=dtype)
    if isinstance(payload, dict):
        return {k: _move_to_device(v, device, dtype) for k, v in payload.items()}
    if isinstance(payload, list):
        return [_move_to_device(v, device, dtype) for v in payload]
    return payload


class Hunyuan3DInputStage(PipelineStage):
    """Input validation stage for Hunyuan3D pipeline."""

    def __init__(self, config: Hunyuan3D2PipelineConfig) -> None:
        super().__init__()
        self.config = config

    def forward(self, batch: Req, server_args: ServerArgs) -> Req:
        if server_args.num_gpus != 1:
            raise ValueError("Hunyuan3D pipeline only supports num_gpus=1.")
        if batch.image_path is None:
            raise ValueError("Hunyuan3D requires 'image_path' input.")
        if isinstance(batch.image_path, list):
            if len(batch.image_path) != 1:
                raise ValueError("Hunyuan3D only supports a single image input.")
            batch.image_path = batch.image_path[0]
        if not isinstance(batch.image_path, str):
            raise ValueError(
                f"Hunyuan3D expects image_path as str, got {type(batch.image_path)}"
            )
        if not os.path.exists(batch.image_path):
            raise FileNotFoundError(f"Image path not found: {batch.image_path}")
        if batch.num_outputs_per_prompt != 1:
            raise ValueError("Hunyuan3D only supports num_outputs_per_prompt=1.")
        return batch


class Hunyuan3DShapePreprocessStage(PipelineStage):
    """Preprocess stage for shape generation."""

    def __init__(self, image_processor: Any) -> None:
        super().__init__()
        self.image_processor = image_processor

    def forward(self, batch: Req, server_args: ServerArgs) -> Req:
        cond_inputs = _prepare_shape_image(self.image_processor, batch.image_path)
        image = cond_inputs.pop("image")

        batch.extra["shape_cond_inputs"] = cond_inputs
        batch.extra["shape_image"] = image
        return batch


class Hunyuan3DShapeConditioningStage(PipelineStage):
    """Stage for computing shape conditioning embeddings with CFG support."""

    def __init__(self, conditioner: Any, model: Any) -> None:
        super().__init__()
        self.conditioner = conditioner
        self.model = model

    def forward(self, batch: Req, server_args: ServerArgs) -> Req:
        image = batch.extra["shape_image"]
        cond_inputs = batch.extra["shape_cond_inputs"]
        device = self.device
        dtype = next(self.model.parameters()).dtype

        image = _move_to_device(image, device, dtype)
        cond_inputs = _move_to_device(cond_inputs, device, dtype)

        do_cfg = batch.guidance_scale >= 0 and not (
            hasattr(self.model, "guidance_embed") and self.model.guidance_embed is True
        )

        cond = self.conditioner(image=image, **cond_inputs)
        if do_cfg:
            un_cond = self.conditioner.unconditional_embedding(
                image.shape[0], **cond_inputs
            )

            def cat_recursive(a, b):
                if isinstance(a, torch.Tensor):
                    return torch.cat([a, b], dim=0).to(dtype)
                out = {}
                for key in a.keys():
                    out[key] = cat_recursive(a[key], b[key])
                return out

            cond = cat_recursive(cond, un_cond)

        batch.prompt_embeds = [cond]
        batch.do_classifier_free_guidance = do_cfg
        batch.extra["shape_image"] = image
        return batch


class Hunyuan3DShapeLatentStage(PipelineStage):
    """Latent preparation stage for shape generation."""

    def __init__(self, scheduler: Any, vae: Any, model: Any) -> None:
        super().__init__()
        self.scheduler = scheduler
        self.vae = vae
        self.model = model

    def _prepare_latents(self, batch_size, dtype, device, generator):
        from diffusers.utils.torch_utils import randn_tensor

        shape = (batch_size, *self.vae.latent_shape)
        latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        return latents * getattr(self.scheduler, "init_noise_sigma", 1.0)

    def forward(self, batch: Req, server_args: ServerArgs) -> Req:
        image = batch.extra["shape_image"]
        batch_size = image.shape[0]
        device = self.device
        dtype = next(self.model.parameters()).dtype

        sigmas = np.linspace(0, 1, batch.num_inference_steps)
        timesteps, _ = retrieve_timesteps(
            self.scheduler,
            batch.num_inference_steps,
            device,
            sigmas=sigmas,
        )

        generator = batch.generator
        if generator is None and batch.seed is not None:
            generator = torch.Generator(device=device).manual_seed(batch.seed)

        latents = self._prepare_latents(batch_size, dtype, device, generator)

        guidance = None
        if hasattr(self.model, "guidance_embed") and self.model.guidance_embed is True:
            guidance = torch.tensor(
                [batch.guidance_scale] * batch_size, device=device, dtype=dtype
            )

        batch.timesteps = timesteps
        batch.latents = latents
        batch.extra["shape_guidance"] = guidance
        return batch


class Hunyuan3DShapeDenoisingStage(DenoisingStage):
    """Denoising stage for Hunyuan3D shape generation.

    Inherits from DenoisingStage and adapts methods for Hunyuan3D-specific:
    - Conditioning structure (dict-based)
    - Timestep normalization
    - CFG handling (concat latents pattern)
    """

    def __init__(self, transformer: Any, scheduler: Any, **kwargs) -> None:
        super().__init__(transformer=transformer, scheduler=scheduler, **kwargs)

    def _prepare_denoising_loop(self, batch: Req, server_args: ServerArgs):
        """Prepare Hunyuan3D-specific variables for the base denoising loop."""
        assert self.transformer is not None
        pipeline = self.pipeline() if self.pipeline else None
        cache_dit_num_inference_steps = batch.extra.get(
            "cache_dit_num_inference_steps", batch.num_inference_steps
        )
        if not server_args.model_loaded["transformer"]:
            loader = TransformerLoader()
            self.transformer = loader.load(
                server_args.model_paths["transformer"], server_args, "transformer"
            )
            self._maybe_enable_cache_dit(cache_dit_num_inference_steps, batch)
            self._maybe_enable_torch_compile(self.transformer)
            if pipeline:
                pipeline.add_module("transformer", self.transformer)
            server_args.model_loaded["transformer"] = True
        else:
            self._maybe_enable_cache_dit(cache_dit_num_inference_steps, batch)

        timesteps = batch.timesteps
        if timesteps is None:
            raise ValueError("Timesteps must be provided")

        latents = batch.latents
        if latents is None:
            raise ValueError("Latents must be provided")

        cond = batch.prompt_embeds[0] if batch.prompt_embeds else None
        if cond is None:
            raise ValueError("Conditioning (prompt_embeds) must be provided")

        if batch.raw_latent_shape is None:
            batch.raw_latent_shape = latents.shape

        guidance = batch.extra.get("shape_guidance")
        num_inference_steps = batch.num_inference_steps
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order

        extra_step_kwargs = self.prepare_extra_func_kwargs(
            self.scheduler.step,
            {"generator": batch.generator, "eta": batch.eta},
        )

        target_dtype = next(self.transformer.parameters()).dtype
        autocast_enabled = (
            target_dtype != torch.float32
        ) and not server_args.disable_autocast

        pos_cond_kwargs = {"encoder_hidden_states": cond}
        neg_cond_kwargs = {}

        return {
            "extra_step_kwargs": extra_step_kwargs,
            "target_dtype": target_dtype,
            "autocast_enabled": autocast_enabled,
            "timesteps": timesteps,
            "num_inference_steps": num_inference_steps,
            "num_warmup_steps": num_warmup_steps,
            "image_kwargs": {},
            "pos_cond_kwargs": pos_cond_kwargs,
            "neg_cond_kwargs": neg_cond_kwargs,
            "latents": latents,
            "prompt_embeds": batch.prompt_embeds,
            "neg_prompt_embeds": None,
            "boundary_timestep": None,
            "z": None,
            "reserved_frames_mask": None,
            "seq_len": None,
            "guidance": guidance,
        }

    def _predict_noise(
        self,
        current_model,
        latent_model_input,
        timestep,
        target_dtype,
        guidance: torch.Tensor,
        **kwargs,
    ):
        """Hunyuan3D-specific noise prediction with normalized timestep."""
        cond = kwargs.get("encoder_hidden_states")
        timestep_norm = timestep / self.scheduler.config.num_train_timesteps
        return current_model(latent_model_input, timestep_norm, cond, guidance=guidance)

    def _predict_noise_with_cfg(
        self,
        current_model,
        latent_model_input: torch.Tensor,
        timestep,
        batch: Req,
        timestep_index: int,
        attn_metadata,
        target_dtype,
        current_guidance_scale,
        image_kwargs: dict[str, Any],
        pos_cond_kwargs: dict[str, Any],
        neg_cond_kwargs: dict[str, Any],
        server_args,
        guidance,
        latents,
    ):
        """Hunyuan3D-specific CFG: concat latents, single forward, then split."""
        cond = pos_cond_kwargs.get("encoder_hidden_states")
        do_cfg = batch.do_classifier_free_guidance

        if do_cfg:
            latent_input = torch.cat([latent_model_input] * 2)
        else:
            latent_input = latent_model_input

        timestep_expanded = timestep.expand(latent_input.shape[0]).to(latents.dtype)

        with set_forward_context(
            current_timestep=timestep_index,
            attn_metadata=attn_metadata,
            forward_batch=batch,
        ):
            noise_pred = self._predict_noise(
                current_model=current_model,
                latent_model_input=latent_input,
                timestep=timestep_expanded,
                target_dtype=target_dtype,
                guidance=guidance,
                encoder_hidden_states=cond,
            )

        if do_cfg:
            noise_pred_cond, noise_pred_uncond = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + current_guidance_scale * (
                noise_pred_cond - noise_pred_uncond
            )

        return noise_pred

    def verify_input(self, batch: Req, server_args: ServerArgs) -> VerificationResult:
        """Verify Hunyuan3D denoising stage inputs."""
        result = VerificationResult()
        result.add_check("timesteps", batch.timesteps, [V.is_tensor, V.min_dims(1)])
        result.add_check("latents", batch.latents, V.is_tensor)
        result.add_check("prompt_embeds", batch.prompt_embeds, V.list_not_empty)
        result.add_check(
            "num_inference_steps", batch.num_inference_steps, V.positive_int
        )
        result.add_check("guidance_scale", batch.guidance_scale, V.non_negative_float)
        return result

    def verify_output(self, batch: Req, server_args: ServerArgs) -> VerificationResult:
        """Verify Hunyuan3D denoising stage outputs."""
        result = VerificationResult()
        result.add_check("latents", batch.latents, V.is_tensor)
        return result


class Hunyuan3DShapeExportStage(PipelineStage):
    """Export stage for shape generation (VAE decoding and mesh extraction)."""

    def __init__(self, vae: Any, config: Hunyuan3D2PipelineConfig) -> None:
        super().__init__()
        self.vae = vae
        self.config = config

    def forward(self, batch: Req, server_args: ServerArgs) -> Req:
        if self.config.shape_mc_algo is not None:
            try:
                from sglang.multimodal_gen.runtime.models.vaes.hunyuan3d_vae import (
                    SurfaceExtractors,
                )

                self.vae.surface_extractor = SurfaceExtractors[
                    self.config.shape_mc_algo
                ]()
            except ImportError:
                logger.warning(
                    f"Could not load SurfaceExtractors for mc_algo={self.config.shape_mc_algo}"
                )

        latents = batch.latents

        if self.config.shape_output_type != "latent":
            latents = 1.0 / self.vae.scale_factor * latents
            latents = self.vae(latents)

            outputs = self.vae.latents2mesh(
                latents,
                bounds=self.config.shape_box_v,
                mc_level=self.config.shape_mc_level,
                num_chunks=self.config.shape_num_chunks,
                octree_resolution=self.config.shape_octree_resolution,
                mc_algo=self.config.shape_mc_algo,
                enable_pbar=False,
            )
        else:
            outputs = latents

        if self.config.shape_output_type == "trimesh":
            outputs = export_to_trimesh(outputs)

        batch.extra["shape_meshes"] = outputs
        return batch


class Hunyuan3DShapeSaveStage(PipelineStage):
    """Save stage for shape generation (mesh export to file)."""

    def __init__(self, config: Hunyuan3D2PipelineConfig) -> None:
        super().__init__()
        self.config = config

    def _get_output_paths(self, batch: Req) -> tuple[str, str]:
        output_path = batch.output_file_path() or os.path.join(
            batch.output_path, "output.obj"
        )
        if output_path.endswith(".glb"):
            obj_path = output_path[:-4] + ".obj"
            return obj_path, output_path
        if output_path.endswith(".obj"):
            return output_path, output_path
        return output_path + ".obj", output_path + ".obj"

    def forward(self, batch: Req, server_args: ServerArgs) -> Req:
        mesh_outputs = batch.extra["shape_meshes"]
        mesh = mesh_outputs[0] if isinstance(mesh_outputs, list) else mesh_outputs
        if isinstance(mesh, list):
            mesh = mesh[0]

        obj_path, return_path = self._get_output_paths(batch)
        output_dir = os.path.dirname(obj_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        mesh.export(obj_path)

        batch.extra["shape_obj_path"] = obj_path
        batch.extra["shape_return_path"] = return_path
        return batch


class Hunyuan3DShapeOnlyOutputStage(PipelineStage):
    """Output stage when paint is disabled (shape only)."""

    def __init__(self, config: Hunyuan3D2PipelineConfig) -> None:
        super().__init__()
        self.config = config

    def forward(self, batch: Req, server_args: ServerArgs) -> OutputBatch:
        obj_path = batch.extra["shape_obj_path"]
        return_path = batch.extra["shape_return_path"]
        if return_path.endswith(".glb"):
            return_path = obj_path
        return OutputBatch(output=[return_path], timings=batch.timings)


__all__ = [
    "retrieve_timesteps",
    "Hunyuan3DInputStage",
    "Hunyuan3DShapePreprocessStage",
    "Hunyuan3DShapeConditioningStage",
    "Hunyuan3DShapeLatentStage",
    "Hunyuan3DShapeDenoisingStage",
    "Hunyuan3DShapeExportStage",
    "Hunyuan3DShapeSaveStage",
    "Hunyuan3DShapeOnlyOutputStage",
]
