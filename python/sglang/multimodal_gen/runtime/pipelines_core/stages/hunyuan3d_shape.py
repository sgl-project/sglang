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
from sglang.multimodal_gen.runtime.models.mesh_utils import export_to_trimesh
from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import OutputBatch, Req
from sglang.multimodal_gen.runtime.pipelines_core.stages.base import PipelineStage
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

        print(f"[SGLANG] Stage1 - image shape: {image.shape}")
        print(
            f"[SGLANG] Stage1 - image stats: min={image.min():.4f}, max={image.max():.4f}, mean={image.mean():.4f}"
        )
        if "mask" in cond_inputs:
            print(f"[SGLANG] Stage1 - mask shape: {cond_inputs['mask'].shape}")

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

        batch.extra["shape_cond"] = cond
        batch.extra["shape_do_cfg"] = do_cfg
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

        print(f"[DEBUG] sigmas: {sigmas}")
        print(f"[DEBUG] timesteps: {timesteps}")

        generator = batch.generator
        if generator is None and batch.seed is not None:
            generator = torch.Generator(device=device).manual_seed(batch.seed)

        latents = self._prepare_latents(batch_size, dtype, device, generator)

        print(f"[SGLANG] Stage3 - latents shape: {latents.shape}")
        print(
            f"[SGLANG] Stage3 - latents stats: min={latents.min():.4f}, max={latents.max():.4f}, mean={latents.mean():.4f}"
        )
        print(f"[SGLANG] Stage3 - latents[0,0,:5]: {latents[0,0,:5]}")
        print(f"[DEBUG] seed: {batch.seed}")
        print(f"[DEBUG] latents shape: {latents.shape}")
        print(
            f"[DEBUG] latents stats: min={latents.min():.4f}, max={latents.max():.4f}, mean={latents.mean():.4f}"
        )
        guidance = None
        if hasattr(self.model, "guidance_embed") and self.model.guidance_embed is True:
            guidance = torch.tensor(
                [batch.guidance_scale] * batch_size, device=device, dtype=dtype
            )

        batch.extra["shape_timesteps"] = timesteps
        batch.extra["shape_latents"] = latents
        batch.extra["shape_guidance"] = guidance
        return batch


class Hunyuan3DShapeDenoisingStage(PipelineStage):
    """Denoising stage for shape generation."""

    def __init__(self, scheduler: Any, model: Any) -> None:
        super().__init__()
        self.scheduler = scheduler
        self.model = model

    def forward(self, batch: Req, server_args: ServerArgs) -> Req:
        timesteps = batch.extra["shape_timesteps"]
        latents = batch.extra["shape_latents"]
        cond = batch.extra["shape_cond"]
        guidance = batch.extra["shape_guidance"]
        do_cfg = batch.extra["shape_do_cfg"]

        print(f"[DEBUG] guidance_scale: {batch.guidance_scale}")
        print(f"[DEBUG] num_inference_steps: {batch.num_inference_steps}")
        print(f"[DEBUG] do_cfg: {batch.extra['shape_do_cfg']}")
        for i, t in enumerate(timesteps):
            if do_cfg:
                latent_model_input = torch.cat([latents] * 2)
            else:
                latent_model_input = latents

            timestep = t.expand(latent_model_input.shape[0]).to(latents.dtype)
            timestep = timestep / self.scheduler.config.num_train_timesteps

            if i == 1:
                print(
                    f"[SGLANG] Step 1 INPUT - latents: min={latents.min():.6f}, max={latents.max():.6f}, mean={latents.mean():.6f}"
                )
                print(f"[SGLANG] Step 1 INPUT - latents[0,0,:5]: {latents[0,0,:5]}")
                print(f"[SGLANG] Step 1 INPUT - timestep: {timestep[0].item():.6f}")
                print(f"[SGLANG] Step 1 INPUT - t (raw): {t.item():.6f}")

            noise_pred = self.model(
                latent_model_input, timestep, cond, guidance=guidance
            )

            if i in [0, 1, 2, 3, 4, 5]:
                print(
                    f"[SGLANG] Step {i} - noise_pred: min={noise_pred.min():.4f}, max={noise_pred.max():.4f}, mean={noise_pred.mean():.4f}"
                )
            if do_cfg:
                noise_pred_cond, noise_pred_uncond = noise_pred.chunk(2)
                if i == 0:
                    print(
                        f"[SGLANG] Step 0 CFG - noise_pred_cond[0,0,0]: {noise_pred_cond[0,0,0].item():.6f}"
                    )
                    print(
                        f"[SGLANG] Step 0 CFG - noise_pred_uncond[0,0,0]: {noise_pred_uncond[0,0,0].item():.6f}"
                    )
                    print(
                        f"[SGLANG] Step 0 CFG - cond stats: min={noise_pred_cond.min():.4f}, max={noise_pred_cond.max():.4f}, mean={noise_pred_cond.mean():.6f}"
                    )
                    print(
                        f"[SGLANG] Step 0 CFG - uncond stats: min={noise_pred_uncond.min():.4f}, max={noise_pred_uncond.max():.4f}, mean={noise_pred_uncond.mean():.6f}"
                    )
                    print(
                        f"[SGLANG] Step 0 - cond_tensor[0,0,:5] (conditional): {cond['main'][0,0,:5]}"
                    )
                    print(
                        f"[SGLANG] Step 0 - cond_tensor[1,0,:5] (unconditional): {cond['main'][1,0,:5]}"
                    )
                noise_pred = noise_pred_uncond + batch.guidance_scale * (
                    noise_pred_cond - noise_pred_uncond
                )

            if i == 0:
                print(
                    f"[SGLANG] Stage4 step0 - noise_pred (after CFG) stats: min={noise_pred.min():.4f}, max={noise_pred.max():.4f}, mean={noise_pred.mean():.4f}"
                )
                print(
                    f"[SGLANG] Step 0 CALC - latents[0,0,0] (input, fp16): {latents[0,0,0].item():.6f}"
                )
                print(
                    f"[SGLANG] Step 0 CALC - latents[0,0,0] (as fp32): {latents[0,0,0].float().item():.6f}"
                )
                print(
                    f"[SGLANG] Step 0 CALC - noise_pred[0,0,0] (after CFG): {noise_pred[0,0,0].item():.6f}"
                )
                sigma_0 = self.scheduler.sigmas[0].item()
                sigma_1 = self.scheduler.sigmas[1].item()
                print(
                    f"[SGLANG] Step 0 CALC - sigma={sigma_0:.8f}, sigma_next={sigma_1:.8f}"
                )
                print(
                    f"[SGLANG] Step 0 CALC - (sigma_next - sigma) = {(sigma_1 - sigma_0):.8f}"
                )
                expected = (
                    latents[0, 0, 0].float().item()
                    + (sigma_1 - sigma_0) * noise_pred[0, 0, 0].float().item()
                )
                print(
                    f"[SGLANG] Step 0 CALC - expected prev_sample[0,0,0] = {expected:.6f}"
                )
            if i in [0, 10, 20, 30, 40]:
                sigma = self.scheduler.sigmas[i]
                sigma_next = self.scheduler.sigmas[i + 1]
                print(
                    f"[SGLANG] Step {i} - sigma={sigma:.6f}, sigma_next={sigma_next:.6f}"
                )
            outputs = self.scheduler.step(noise_pred, t, latents)
            latents = outputs.prev_sample
            if i in [0, 10, 20, 30, 40, 49]:
                print(
                    f"[SGLANG] Stage4 step{i} - latents: min={latents.min():.4f}, max={latents.max():.4f}, mean={latents.mean():.4f}"
                )
            if i == 0:
                print(
                    f"[SGLANG] Stage4 step0 - latents after step: min={latents.min():.4f}, max={latents.max():.4f}, mean={latents.mean():.4f}"
                )
                print(
                    f"[SGLANG] Step 0 RESULT - actual latents[0,0,0]: {latents[0,0,0].item():.6f}"
                )
        print(
            f"[SGLANG] Stage4 final - latents stats: min={latents.min():.4f}, max={latents.max():.4f}, mean={latents.mean():.4f}"
        )
        print(f"[SGLANG] Stage4 final - latents[0,0,:5]: {latents[0,0,:5]}")
        batch.extra["shape_latents"] = latents
        return batch


class Hunyuan3DShapeExportStage(PipelineStage):
    """Export stage for shape generation (VAE decoding and mesh extraction)."""

    def __init__(self, vae: Any, config: Hunyuan3D2PipelineConfig) -> None:
        super().__init__()
        self.vae = vae
        self.config = config

    def forward(self, batch: Req, server_args: ServerArgs) -> Req:
        # Update surface extractor if specified
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

        latents = batch.extra["shape_latents"]
        print(
            f"[SGLANG] Stage5 - input latents: min={latents.min():.4f}, max={latents.max():.4f}"
        )
        print(
            f"[DEBUG] Before VAE - latents stats: "
            f"min={latents.min().item():.4f}, max={latents.max().item():.4f}, "
            f"mean={latents.mean().item():.4f}, std={latents.std().item():.4f}"
        )

        if self.config.shape_output_type != "latent":
            latents = 1.0 / self.vae.scale_factor * latents
            print(f"[SGLANG] Stage5 - scale_factor: {self.vae.scale_factor}")
            print(
                f"[SGLANG] Stage5 - scaled latents: min={latents.min():.4f}, max={latents.max():.4f}"
            )
            print(
                f"[DEBUG] After scale - latents: min={latents.min():.4f}, max={latents.max():.4f}"
            )
            latents = self.vae(latents)
            print(f"[SGLANG] Stage5 - decoded latents shape: {latents.shape}")
            print(
                f"[SGLANG] Stage5 - decoded latents: min={latents.min():.4f}, max={latents.max():.4f}"
            )
            print(
                f"[SGLANG] Stage5 - output latents: min={latents.min():.4f}, max={latents.max():.4f}"
            )
            print(
                f"[DEBUG] After VAE forward - latents: min={latents.min():.4f}, max={latents.max():.4f}"
            )
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
