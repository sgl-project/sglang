# SPDX-License-Identifier: Apache-2.0
"""
Hunyuan3D image-to-mesh pipeline implementation.
"""

from __future__ import annotations

import os
import sys
from typing import Any

import numpy as np
import torch

from sglang.multimodal_gen.configs.pipeline_configs.hunyuan3d import (
    Hunyuan3DPipelineConfig,
)
from sglang.multimodal_gen.runtime.pipelines_core.composed_pipeline_base import (
    ComposedPipelineBase,
)
from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import OutputBatch, Req
from sglang.multimodal_gen.runtime.pipelines_core.stages.base import (
    PipelineStage,
    StageParallelismType,
)
from sglang.multimodal_gen.runtime.platforms import current_platform
from sglang.multimodal_gen.runtime.server_args import ServerArgs
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

logger = init_logger(__name__)


def _ensure_hunyuan3d_paths(repo_path: str) -> None:
    repo_root = os.path.abspath(repo_path)
    shape_path = os.path.join(repo_root, "hy3dshape")
    paint_path = os.path.join(repo_root, "hy3dpaint")

    if not os.path.isdir(shape_path) or not os.path.isdir(paint_path):
        raise FileNotFoundError(
            "Hunyuan3D repo path must contain 'hy3dshape' and 'hy3dpaint': "
            f"{repo_root}"
        )

    for path in (shape_path, paint_path):
        if path not in sys.path:
            sys.path.insert(0, path)


def _prepare_shape_image(image_processor, image, mask=None) -> dict:
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
    if isinstance(payload, torch.Tensor):
        return payload.to(device=device, dtype=dtype)
    if isinstance(payload, dict):
        return {k: _move_to_device(v, device, dtype) for k, v in payload.items()}
    if isinstance(payload, list):
        return [_move_to_device(v, device, dtype) for v in payload]
    return payload


class Hunyuan3DInputStage(PipelineStage):
    def __init__(self, config: Hunyuan3DPipelineConfig) -> None:
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
        if batch.num_inference_steps is None:
            batch.num_inference_steps = self.config.shape_num_inference_steps
        if batch.guidance_scale is None:
            batch.guidance_scale = self.config.shape_guidance_scale
        return batch


class ShapePreprocessStage(PipelineStage):
    def __init__(self, image_processor: Any) -> None:
        super().__init__()
        self.image_processor = image_processor

    def forward(self, batch: Req, server_args: ServerArgs) -> Req:
        cond_inputs = _prepare_shape_image(self.image_processor, batch.image_path)
        image = cond_inputs.pop("image")
        batch.extra["shape_cond_inputs"] = cond_inputs
        batch.extra["shape_image"] = image
        return batch


class ShapeConditioningStage(PipelineStage):
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


class ShapeLatentStage(PipelineStage):
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
        from sglang.multimodal_gen.runtime.models.hunyuan3d.hy3dshape.hy3dshape.pipelines import (
            retrieve_timesteps,
        )

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

        batch.extra["shape_timesteps"] = timesteps
        batch.extra["shape_latents"] = latents
        batch.extra["shape_guidance"] = guidance
        return batch


class ShapeDenoisingStage(PipelineStage):
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

        for t in timesteps:
            if do_cfg:
                latent_model_input = torch.cat([latents] * 2)
            else:
                latent_model_input = latents

            timestep = t.expand(latent_model_input.shape[0]).to(latents.dtype)
            timestep = timestep / self.scheduler.config.num_train_timesteps
            noise_pred = self.model(
                latent_model_input, timestep, cond, guidance=guidance
            )

            if do_cfg:
                noise_pred_cond, noise_pred_uncond = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + batch.guidance_scale * (
                    noise_pred_cond - noise_pred_uncond
                )

            outputs = self.scheduler.step(noise_pred, t, latents)
            latents = outputs.prev_sample

        batch.extra["shape_latents"] = latents
        return batch


class ShapeExportStage(PipelineStage):
    def __init__(self, vae: Any, config: Hunyuan3DPipelineConfig) -> None:
        super().__init__()
        self.vae = vae
        self.config = config

    def forward(self, batch: Req, server_args: ServerArgs) -> Req:
        from sglang.multimodal_gen.runtime.models.hunyuan3d.hy3dshape.hy3dshape.models.autoencoders import (
            SurfaceExtractors,
        )
        from sglang.multimodal_gen.runtime.models.hunyuan3d.hy3dshape.hy3dshape.pipelines import (
            export_to_trimesh,
        )

        if self.config.shape_mc_algo is not None:
            self.vae.surface_extractor = SurfaceExtractors[
                self.config.shape_mc_algo
            ]()

        latents = batch.extra["shape_latents"]
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


class ShapeSaveStage(PipelineStage):
    def __init__(self, config: Hunyuan3DPipelineConfig) -> None:
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


class PaintStage(PipelineStage):
    def __init__(self, paint_pipeline: Any, config: Hunyuan3DPipelineConfig) -> None:
        super().__init__()
        self.paint_pipeline = paint_pipeline
        self.config = config

    @property
    def parallelism_type(self) -> StageParallelismType:
        return StageParallelismType.MAIN_RANK_ONLY

    def forward(self, batch: Req, server_args: ServerArgs) -> OutputBatch:
        obj_path = batch.extra["shape_obj_path"]
        return_path = batch.extra["shape_return_path"]

        self.paint_pipeline(
            mesh_path=obj_path,
            image_path=batch.image_path,
            output_mesh_path=obj_path,
            use_remesh=self.config.paint_use_remesh,
            save_glb=self.config.paint_save_glb,
        )

        if return_path.endswith(".glb"):
            if self.config.paint_save_glb:
                return_path = obj_path[:-4] + ".glb"
            else:
                return_path = obj_path

        return OutputBatch(output=[return_path], timings=batch.timings)


class ShapeOnlyOutputStage(PipelineStage):
    def __init__(self, config: Hunyuan3DPipelineConfig) -> None:
        super().__init__()
        self.config = config

    def forward(self, batch: Req, server_args: ServerArgs) -> OutputBatch:
        obj_path = batch.extra["shape_obj_path"]
        return_path = batch.extra["shape_return_path"]
        if return_path.endswith(".glb"):
            return_path = obj_path
        return OutputBatch(output=[return_path], timings=batch.timings)


class Hunyuan3DPipeline(ComposedPipelineBase):
    pipeline_name = "Hunyuan3DPipeline"
    _required_config_modules = [
        "hy3dshape_model",
        "hy3dshape_vae",
        "hy3dshape_scheduler",
        "hy3dshape_conditioner",
        "hy3dshape_image_processor",
    ]

    def _load_config(self) -> dict[str, Any]:
        return {
            "_class_name": self.pipeline_name,
            "_diffusers_version": "0.0.0",
            "hy3dshape_model": ["diffusers", "Hunyuan3DShapeModel"],
            "hy3dshape_vae": ["diffusers", "Hunyuan3DShapeVAE"],
            "hy3dshape_scheduler": ["diffusers", "Hunyuan3DShapeScheduler"],
            "hy3dshape_conditioner": ["diffusers", "Hunyuan3DShapeConditioner"],
            "hy3dshape_image_processor": ["diffusers", "Hunyuan3DShapeImageProcessor"],
        }

    def initialize_pipeline(self, server_args: ServerArgs):
        config = server_args.pipeline_config
        if not isinstance(config, Hunyuan3DPipelineConfig):
            raise TypeError(
                "Hunyuan3DPipeline requires Hunyuan3DPipelineConfig, "
                f"got {type(config)}"
            )

        _ensure_hunyuan3d_paths(config.hunyuan3d_repo_path)
        repo_root = os.path.abspath(config.hunyuan3d_repo_path)

        if config.paint_enable:
            from textureGenPipeline import Hunyuan3DPaintConfig, Hunyuan3DPaintPipeline

            paint_model_path = config.paint_model_path or self.model_path

            paint_config = Hunyuan3DPaintConfig(
                max_num_view=config.paint_max_num_view,
                resolution=config.paint_resolution,
            )
            device_str = str(torch.device(current_platform.device_type))
            paint_config.device = device_str
            paint_config.multiview_pretrained_path = paint_model_path
            paint_config.multiview_cfg_path = os.path.join(
                repo_root, "hy3dpaint", "cfgs", "hunyuan-paint-pbr.yaml"
            )
            paint_config.realesrgan_ckpt_path = os.path.join(
                repo_root, "hy3dpaint", "ckpt", "RealESRGAN_x4plus.pth"
            )

            paint_pipeline = Hunyuan3DPaintPipeline(paint_config)
            self.add_module("paint_pipeline", paint_pipeline)

    def create_pipeline_stages(self, server_args: ServerArgs):
        config = server_args.pipeline_config
        assert isinstance(config, Hunyuan3DPipelineConfig)

        self.add_stage(
            stage_name="input_stage", stage=Hunyuan3DInputStage(config=config)
        )
        self.add_stage(
            stage_name="shape_preprocess_stage",
            stage=ShapePreprocessStage(
                image_processor=self.get_module("hy3dshape_image_processor"),
            ),
        )
        self.add_stage(
            stage_name="shape_conditioning_stage",
            stage=ShapeConditioningStage(
                conditioner=self.get_module("hy3dshape_conditioner"),
                model=self.get_module("hy3dshape_model"),
            ),
        )
        self.add_stage(
            stage_name="shape_latent_stage",
            stage=ShapeLatentStage(
                scheduler=self.get_module("hy3dshape_scheduler"),
                vae=self.get_module("hy3dshape_vae"),
                model=self.get_module("hy3dshape_model"),
            ),
        )
        self.add_stage(
            stage_name="shape_denoising_stage",
            stage=ShapeDenoisingStage(
                scheduler=self.get_module("hy3dshape_scheduler"),
                model=self.get_module("hy3dshape_model"),
            ),
        )
        self.add_stage(
            stage_name="shape_export_stage",
            stage=ShapeExportStage(
                vae=self.get_module("hy3dshape_vae"),
                config=config,
            ),
        )
        self.add_stage(stage_name="shape_save_stage", stage=ShapeSaveStage(config))
        if config.paint_enable:
            self.add_stage(
                stage_name="paint_stage",
                stage=PaintStage(
                    paint_pipeline=self.get_module("paint_pipeline"), config=config
                ),
            )
        else:
            self.add_stage(
                stage_name="paint_stage",
                stage=ShapeOnlyOutputStage(config=config),
            )


EntryClass = Hunyuan3DPipeline
