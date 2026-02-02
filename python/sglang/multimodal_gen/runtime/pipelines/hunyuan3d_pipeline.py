# SPDX-License-Identifier: Apache-2.0
"""
Hunyuan3D image-to-mesh pipeline implementation.
"""

from __future__ import annotations

import os
from typing import Any

import numpy as np
import torch
from einops import rearrange

from sglang.multimodal_gen.configs.pipeline_configs.hunyuan3d import (
    Hunyuan3D2PipelineConfig,
)

# Import internal utilities
from sglang.multimodal_gen.runtime.models.mesh_utils import export_to_trimesh
from sglang.multimodal_gen.runtime.models.model_stages.hunyuan_shape_conditioning import (
    ShapeConditioningStage,
)
from sglang.multimodal_gen.runtime.pipelines_core.composed_pipeline_base import (
    ComposedPipelineBase,
)
from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import OutputBatch, Req
from sglang.multimodal_gen.runtime.pipelines_core.stages import InputValidationStage
from sglang.multimodal_gen.runtime.pipelines_core.stages.base import (
    PipelineStage,
    StageParallelismType,
)
from sglang.multimodal_gen.runtime.server_args import ServerArgs
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger


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


logger = init_logger(__name__)


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


class Hunyuan3DInputStage(PipelineStage):
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


class ShapePreprocessStage(PipelineStage):
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
        image = batch.extra["shape_image"]
        batch_size = image.shape[0]
        device = self.device
        dtype = next(self.model.parameters()).dtype
        # dtype = torch.float32

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
        print(f"[SGLANG] Stage3 - latents[0,0,:5]: {latents[0,0,:5]}")  # 具体数值对比
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

        print(f"[DEBUG] guidance_scale: {batch.guidance_scale}")
        print(f"[DEBUG] num_inference_steps: {batch.num_inference_steps}")
        print(f"[DEBUG] do_cfg: {batch.extra['shape_do_cfg']}")
        for i, t in enumerate(timesteps):
            if do_cfg:
                latent_model_input = torch.cat([latents] * 2)
            else:
                latent_model_input = latents

            timestep = t.expand(latent_model_input.shape[0]).to(latents.dtype)

            # timestep = t.expand(latent_model_input.shape[0]).to(torch.float32)
            timestep = timestep / self.scheduler.config.num_train_timesteps

            # 详细打印 Step 1 的输入，用于对比
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
                # 详细调试 scheduler.step 计算
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


class ShapeExportStage(PipelineStage):
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


class ShapeSaveStage(PipelineStage):
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


class PaintPreprocessStage(PipelineStage):
    """Stage 1: UV unwrap and image delight preprocessing.

    This stage prepares the mesh and reference image for texture generation:
    - Applies UV unwrapping to the mesh using xatlas
    - Removes lighting/shadows from the reference image using the delight model
    """

    def __init__(self, config: Hunyuan3D2PipelineConfig) -> None:
        super().__init__()
        self.config = config
        self.delight_model = None
        self._delight_loaded = False

    def _load_delight_model(self, server_args: ServerArgs):
        """Lazy load the delight model."""
        if self._delight_loaded:
            return

        from sglang.multimodal_gen.runtime.models.preprocessors.delight import (
            LightShadowRemover,
        )

        # Get model path from config
        model_path = server_args.model_path
        delight_subfolder = getattr(
            self.config, "delight_subfolder", "hunyuan3d-delight-v2-0"
        )

        # Try to load from HuggingFace or local path
        import os

        from huggingface_hub import snapshot_download

        base_dir = os.environ.get("HY3DGEN_MODELS", "~/.cache/hy3dgen")
        local_path = os.path.expanduser(
            os.path.join(base_dir, model_path, delight_subfolder)
        )

        if not os.path.exists(local_path):
            try:
                path = snapshot_download(
                    repo_id=model_path,
                    allow_patterns=[f"{delight_subfolder}/*"],
                    local_dir=os.path.expanduser(os.path.join(base_dir, model_path)),
                )
                local_path = os.path.join(path, delight_subfolder)
            except Exception as e:
                logger.warning(f"Could not download delight model: {e}")
                local_path = None

        if local_path and os.path.exists(local_path):
            self.delight_model = LightShadowRemover(local_path)
            self._delight_loaded = True
            logger.info("Delight model loaded successfully")
        else:
            logger.warning(
                "Delight model not available, skipping delight preprocessing"
            )

    def forward(self, batch: Req, server_args: ServerArgs) -> Req:
        from PIL import Image

        from sglang.multimodal_gen.runtime.models.mesh_utils import mesh_uv_wrap

        # Get mesh from shape generation
        mesh = batch.extra["shape_meshes"]
        if isinstance(mesh, list):
            mesh = mesh[0]

        # UV unwrap
        try:
            mesh = mesh_uv_wrap(mesh)
            logger.info("UV unwrapping completed")
        except Exception as e:
            logger.warning(f"UV unwrapping failed: {e}")

        batch.extra["paint_mesh"] = mesh

        # Load and process reference image
        image = Image.open(batch.image_path)

        # Apply delight if model is available
        self._load_delight_model(server_args)
        if self.delight_model is not None:
            try:
                image = self.delight_model(image)
                logger.info("Image delight completed")
            except Exception as e:
                logger.warning(f"Image delight failed: {e}")

        batch.extra["delighted_image"] = image
        return batch


class PaintRenderStage(PipelineStage):
    """Stage 2: Multi-view normal and position map rendering.

    This stage renders the mesh from multiple viewpoints to create
    conditioning inputs for the texture diffusion model.
    """

    def __init__(self, config: Hunyuan3D2PipelineConfig) -> None:
        super().__init__()
        self.config = config
        self.renderer = None

        # Camera configuration for 6 views
        self.camera_azims = [0, 90, 180, 270, 0, 180]
        self.camera_elevs = [0, 0, 0, 0, 90, -90]
        self.view_weights = [1, 0.1, 0.5, 0.1, 0.05, 0.05]

    def _init_renderer(self):
        """Initialize the mesh renderer."""
        if self.renderer is not None:
            return

        from sglang.multimodal_gen.runtime.models.renderers.mesh_render import (
            MeshRender,
        )

        render_size = getattr(self.config, "paint_resolution", 512)
        texture_size = getattr(self.config, "paint_texture_size", 2048)

        self.renderer = MeshRender(
            default_resolution=render_size,
            texture_size=texture_size,
        )
        logger.info("Mesh renderer initialized")

    def forward(self, batch: Req, server_args: ServerArgs) -> Req:
        self._init_renderer()

        mesh = batch.extra["paint_mesh"]

        # Load mesh into renderer
        self.renderer.load_mesh(mesh)

        # Render normal maps
        normal_maps = self.renderer.render_normal_multiview(
            self.camera_elevs, self.camera_azims, use_abs_coor=True
        )

        # Render position maps
        position_maps = self.renderer.render_position_multiview(
            self.camera_elevs, self.camera_azims
        )

        batch.extra["normal_maps"] = normal_maps
        batch.extra["position_maps"] = position_maps
        batch.extra["camera_azims"] = self.camera_azims
        batch.extra["camera_elevs"] = self.camera_elevs
        batch.extra["view_weights"] = self.view_weights
        batch.extra["renderer"] = self.renderer

        logger.info(f"Rendered {len(normal_maps)} views for texture generation")
        return batch


def guidance_scale_embedding(
    w: torch.Tensor, embedding_dim: int = 512, dtype: torch.dtype = torch.float32
) -> torch.Tensor:
    """Generate guidance scale embeddings.

    Source: Hunyuan3D-2/hy3dgen/texgen/hunyuanpaint/pipeline.py
    Reference: https://github.com/google-research/vdm
    """
    assert len(w.shape) == 1
    w = w * 1000.0

    half_dim = embedding_dim // 2
    emb = torch.log(torch.tensor(10000.0)) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, dtype=dtype) * -emb)
    emb = w.to(dtype)[:, None] * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    if embedding_dim % 2 == 1:
        emb = torch.nn.functional.pad(emb, (0, 1))
    assert emb.shape == (w.shape[0], embedding_dim)
    return emb


def extract_into_tensor(
    a: torch.Tensor, t: torch.Tensor, x_shape: tuple, n_gen: int
) -> torch.Tensor:
    """Extract values from tensor and reshape for multi-view generation.

    Source: Hunyuan3D-2/hy3dgen/texgen/hunyuanpaint/pipeline.py
    """
    out = a.gather(-1, t)
    out = out.repeat(n_gen)
    out = rearrange(out, "(b n) -> b n", n=n_gen)
    b, c, *_ = out.shape
    return out.reshape(b, c, *((1,) * (len(x_shape) - 2)))


def get_predicted_original_sample(
    model_output: torch.Tensor,
    timesteps: torch.Tensor,
    sample: torch.Tensor,
    prediction_type: str,
    alphas: torch.Tensor,
    sigmas: torch.Tensor,
    n_gen: int,
) -> torch.Tensor:
    """Get predicted original sample from model output.

    Source: Hunyuan3D-2/hy3dgen/texgen/hunyuanpaint/pipeline.py
    """
    alphas = extract_into_tensor(alphas, timesteps, sample.shape, n_gen)
    sigmas = extract_into_tensor(sigmas, timesteps, sample.shape, n_gen)
    model_output = rearrange(model_output, "(b n) c h w -> b n c h w", n=n_gen)

    if prediction_type == "epsilon":
        pred_x_0 = (sample - sigmas * model_output) / alphas
    elif prediction_type == "sample":
        pred_x_0 = model_output
    elif prediction_type == "v_prediction":
        pred_x_0 = alphas * sample - sigmas * model_output
    else:
        raise ValueError(
            f"Prediction type {prediction_type} is not supported; "
            "currently, `epsilon`, `sample`, and `v_prediction` are supported."
        )

    return pred_x_0


def get_predicted_noise(
    model_output: torch.Tensor,
    timesteps: torch.Tensor,
    sample: torch.Tensor,
    prediction_type: str,
    alphas: torch.Tensor,
    sigmas: torch.Tensor,
    n_gen: int,
) -> torch.Tensor:
    """Get predicted noise from model output.

    Source: Hunyuan3D-2/hy3dgen/texgen/hunyuanpaint/pipeline.py
    """
    alphas = extract_into_tensor(alphas, timesteps, sample.shape, n_gen)
    sigmas = extract_into_tensor(sigmas, timesteps, sample.shape, n_gen)
    model_output = rearrange(model_output, "(b n) c h w -> b n c h w", n=n_gen)

    if prediction_type == "epsilon":
        pred_epsilon = model_output
    elif prediction_type == "sample":
        pred_epsilon = (sample - alphas * model_output) / sigmas
    elif prediction_type == "v_prediction":
        pred_epsilon = alphas * model_output + sigmas * sample
    else:
        raise ValueError(
            f"Prediction type {prediction_type} is not supported; "
            "currently, `epsilon`, `sample`, and `v_prediction` are supported."
        )

    return pred_epsilon


def to_rgb_image(maybe_rgba):
    """Convert RGBA image to RGB.

    Source: Hunyuan3D-2/hy3dgen/texgen/hunyuanpaint/pipeline.py
    """
    from PIL import Image

    if maybe_rgba.mode == "RGB":
        return maybe_rgba
    if maybe_rgba.mode == "RGBA":
        rgba = maybe_rgba
        img = np.random.randint(
            127, 128, size=[rgba.size[1], rgba.size[0], 3], dtype=np.uint8
        )
        img = Image.fromarray(img, "RGB")
        img.paste(rgba, mask=rgba.getchannel("A"))
        return img
    raise ValueError(f"Unsupported image type: {maybe_rgba.mode}")


class DDIMSolver:
    """DDIM solver for fast sampling.

    Source: Hunyuan3D-2/hy3dgen/texgen/hunyuanpaint/pipeline.py
    """

    def __init__(
        self,
        alpha_cumprods: np.ndarray,
        timesteps: int = 1000,
        ddim_timesteps: int = 50,
    ):
        step_ratio = timesteps // ddim_timesteps
        self.ddim_timesteps = (
            np.arange(1, ddim_timesteps + 1) * step_ratio
        ).round().astype(np.int64) - 1
        self.ddim_alpha_cumprods = alpha_cumprods[self.ddim_timesteps]
        self.ddim_alpha_cumprods_prev = np.asarray(
            [alpha_cumprods[0]] + alpha_cumprods[self.ddim_timesteps[:-1]].tolist()
        )
        # Convert to torch tensors
        self.ddim_timesteps = torch.from_numpy(self.ddim_timesteps).long()
        self.ddim_alpha_cumprods = torch.from_numpy(self.ddim_alpha_cumprods)
        self.ddim_alpha_cumprods_prev = torch.from_numpy(self.ddim_alpha_cumprods_prev)

    def to(self, device: torch.device) -> "DDIMSolver":
        self.ddim_timesteps = self.ddim_timesteps.to(device)
        self.ddim_alpha_cumprods = self.ddim_alpha_cumprods.to(device)
        self.ddim_alpha_cumprods_prev = self.ddim_alpha_cumprods_prev.to(device)
        return self

    def ddim_step(
        self,
        pred_x0: torch.Tensor,
        pred_noise: torch.Tensor,
        timestep_index: torch.Tensor,
        n_gen: int,
    ) -> torch.Tensor:
        alpha_cumprod_prev = extract_into_tensor(
            self.ddim_alpha_cumprods_prev, timestep_index, pred_x0.shape, n_gen
        )
        dir_xt = (1.0 - alpha_cumprod_prev).sqrt() * pred_noise
        x_prev = alpha_cumprod_prev.sqrt() * pred_x0 + dir_xt
        return x_prev


class PaintDiffusionStage(PipelineStage):
    """Stage 3: Multi-view texture diffusion generation.

    This stage uses the HunyuanPaint model to generate textures
    for multiple views based on the reference image and rendered maps.
    """

    def __init__(self, config: Hunyuan3D2PipelineConfig) -> None:
        super().__init__()
        self.config = config
        self.paint_vae = None
        self.paint_text_encoder = None
        self.paint_tokenizer = None
        self.paint_unet = None
        self.paint_scheduler = None
        self.paint_feature_extractor = None
        self.paint_image_processor = None
        self.paint_solver = None
        self.paint_vae_scale_factor = None
        self.paint_is_turbo = False
        self._paint_loaded = False

    @property
    def parallelism_type(self) -> StageParallelismType:
        return StageParallelismType.MAIN_RANK_ONLY

    def _load_paint_pipeline(self, server_args: ServerArgs):
        """Lazy load paint diffusion components."""
        if self._paint_loaded:
            return

        import torch
        from huggingface_hub import snapshot_download

        model_path = server_args.model_path
        paint_subfolder = getattr(
            self.config, "paint_subfolder", "hunyuan3d-paint-v2-0"
        )

        # Determine local path
        base_dir = os.environ.get("HY3DGEN_MODELS", "~/.cache/hy3dgen")
        local_path = os.path.expanduser(
            os.path.join(base_dir, model_path, paint_subfolder)
        )

        # Download if not exists
        if not os.path.exists(local_path):
            try:
                logger.info(
                    f"Downloading paint model from {model_path}/{paint_subfolder}..."
                )
                path = snapshot_download(
                    repo_id=model_path,
                    allow_patterns=[f"{paint_subfolder}/*"],
                    local_dir=os.path.expanduser(os.path.join(base_dir, model_path)),
                )
                local_path = os.path.join(path, paint_subfolder)
            except Exception as e:
                logger.warning(f"Could not download paint model: {e}")
                self._paint_loaded = True
                return

        # Load paint pipeline components
        try:
            from diffusers import AutoencoderKL, EulerAncestralDiscreteScheduler
            from diffusers.image_processor import VaeImageProcessor
            from transformers import CLIPImageProcessor, CLIPTextModel, CLIPTokenizer

            from sglang.multimodal_gen.runtime.models.unets.hunyuan3d_paint import (
                UNet2p5DConditionModel,
            )

            logger.info(f"Loading paint model from {local_path}")

            # Load components
            self.paint_vae = AutoencoderKL.from_pretrained(
                os.path.join(local_path, "vae"),
                torch_dtype=torch.float16,
            ).to("cuda")

            self.paint_text_encoder = CLIPTextModel.from_pretrained(
                os.path.join(local_path, "text_encoder"),
                torch_dtype=torch.float16,
            ).to("cuda")

            self.paint_tokenizer = CLIPTokenizer.from_pretrained(
                os.path.join(local_path, "tokenizer"),
            )

            self.paint_unet = UNet2p5DConditionModel.from_pretrained(
                os.path.join(local_path, "unet"),
                torch_dtype=torch.float16,
            ).to("cuda")

            self.paint_scheduler = EulerAncestralDiscreteScheduler.from_pretrained(
                os.path.join(local_path, "scheduler"),
            )

            self.paint_feature_extractor = CLIPImageProcessor.from_pretrained(
                os.path.join(local_path, "feature_extractor"),
            )
            self.paint_vae_scale_factor = 2 ** (
                len(self.paint_vae.config.block_out_channels) - 1
            )
            self.paint_image_processor = VaeImageProcessor(
                vae_scale_factor=self.paint_vae_scale_factor
            )
            self.paint_solver = DDIMSolver(
                self.paint_scheduler.alphas_cumprod.cpu().numpy(),
                timesteps=self.paint_scheduler.config.num_train_timesteps,
                ddim_timesteps=30,
            ).to(torch.device("cuda"))
            self.paint_is_turbo = bool(getattr(self.config, "paint_turbo_mode", False))

            logger.info("Paint pipeline loaded successfully")

        except Exception as e:
            logger.warning(f"Failed to load paint pipeline: {e}")
            logger.warning("Will use fallback texture generation (reference image)")
            self.paint_vae = None
            self.paint_text_encoder = None
            self.paint_tokenizer = None
            self.paint_unet = None
            self.paint_scheduler = None
            self.paint_feature_extractor = None
            self.paint_image_processor = None
            self.paint_solver = None
            self.paint_vae_scale_factor = None
            self.paint_is_turbo = False

        self._paint_loaded = True

    def _convert_pil_list_to_tensor(self, images, device: torch.device) -> torch.Tensor:
        bg_c = [1.0, 1.0, 1.0]
        images_tensor = []
        for batch_imgs in images:
            view_imgs = []
            for pil_img in batch_imgs:
                img = np.asarray(pil_img, dtype=np.float32) / 255.0
                if img.shape[2] > 3:
                    alpha = img[:, :, 3:]
                    img = img[:, :, :3] * alpha + bg_c * (1 - alpha)
                img = (
                    torch.from_numpy(img)
                    .permute(2, 0, 1)
                    .unsqueeze(0)
                    .contiguous()
                    .to(device=device, dtype=self.paint_vae.dtype)
                )
                view_imgs.append(img)
            view_imgs = torch.cat(view_imgs, dim=0)
            images_tensor.append(view_imgs.unsqueeze(0))

        images_tensor = torch.cat(images_tensor, dim=0)
        return images_tensor

    @torch.no_grad()
    def _encode_images(self, images: torch.Tensor) -> torch.Tensor:
        batch_size = images.shape[0]
        images = rearrange(images, "b n c h w -> (b n) c h w")

        dtype = next(self.paint_vae.parameters()).dtype
        images = (images - 0.5) * 2.0
        posterior = self.paint_vae.encode(images.to(dtype)).latent_dist
        latents = posterior.sample() * self.paint_vae.config.scaling_factor

        latents = rearrange(latents, "(b n) c h w -> b n c h w", b=batch_size)
        return latents

    @staticmethod
    def _compute_camera_index(azim: float, elev: float) -> int:
        if elev == 0:
            base = 12
            idx = int(((azim // 30) + 9) % 12)
        elif elev == 20:
            base = 24
            idx = int(((azim // 30) + 9) % 12)
        elif elev == -20:
            base = 0
            idx = int(((azim // 30) + 9) % 12)
        elif elev == 90:
            base = 40
            idx = 0
        elif elev == -90:
            base = 36
            idx = 0
        else:
            base = 12
            idx = int(((azim // 30) + 9) % 12)
        return base + idx

    @torch.no_grad()
    def _prepare_conditions(
        self,
        batch: Req,
        device: torch.device,
        guidance_scale: float,
    ) -> None:
        image = batch.extra["delighted_image"]
        normal_maps = batch.extra["normal_maps"]
        position_maps = batch.extra["position_maps"]
        camera_azims = batch.extra["camera_azims"]
        camera_elevs = batch.extra["camera_elevs"]

        if not isinstance(image, list):
            image = [image]
        image = [to_rgb_image(img) for img in image]

        image_vae = [
            torch.tensor(np.array(img, dtype=np.float32) / 255.0) for img in image
        ]
        image_vae = [
            img_vae.unsqueeze(0).permute(0, 3, 1, 2).unsqueeze(0)
            for img_vae in image_vae
        ]
        image_vae = torch.cat(image_vae, dim=1).to(
            device=device, dtype=self.paint_vae.dtype
        )

        batch_size, _ = image_vae.shape[0], image_vae.shape[1]
        assert batch_size == 1

        batch.extra["paint_ref_latents"] = self._encode_images(image_vae)

        if isinstance(normal_maps, list):
            normal_maps = self._convert_pil_list_to_tensor([normal_maps], device)
        if isinstance(position_maps, list):
            position_maps = self._convert_pil_list_to_tensor([position_maps], device)

        if normal_maps is not None:
            batch.extra["paint_normal_imgs"] = self._encode_images(normal_maps)
        if position_maps is not None:
            batch.extra["paint_position_maps"] = position_maps
            batch.extra["paint_position_imgs"] = self._encode_images(position_maps)

        camera_info = [
            self._compute_camera_index(azim, elev)
            for azim, elev in zip(camera_azims, camera_elevs)
        ]
        batch.extra["paint_camera_info_gen"] = torch.tensor(
            [camera_info], device=device, dtype=torch.int64
        )
        batch.extra["paint_camera_info_ref"] = torch.tensor(
            [[0]], device=device, dtype=torch.int64
        )

        if self.paint_is_turbo and "paint_position_maps" in batch.extra:
            from sglang.multimodal_gen.runtime.models.unets.hunyuan3d_paint import (
                compute_multi_resolution_discrete_voxel_indice,
                compute_multi_resolution_mask,
            )

            position_maps = batch.extra["paint_position_maps"]
            batch.extra["paint_position_attn_mask"] = compute_multi_resolution_mask(
                position_maps
            )
            batch.extra["paint_position_voxel_indices"] = (
                compute_multi_resolution_discrete_voxel_indice(position_maps)
            )

        if guidance_scale > 1 and not self.paint_is_turbo:
            ref_latents = batch.extra["paint_ref_latents"]
            negative_ref_latents = torch.zeros_like(ref_latents)
            batch.extra["paint_ref_latents"] = torch.cat(
                [negative_ref_latents, ref_latents]
            )
            batch.extra["paint_ref_scale"] = torch.as_tensor([0.0, 1.0]).to(
                batch.extra["paint_ref_latents"]
            )

            if "paint_normal_imgs" in batch.extra:
                batch.extra["paint_normal_imgs"] = torch.cat(
                    (batch.extra["paint_normal_imgs"], batch.extra["paint_normal_imgs"])
                )

            if "paint_position_imgs" in batch.extra:
                batch.extra["paint_position_imgs"] = torch.cat(
                    (
                        batch.extra["paint_position_imgs"],
                        batch.extra["paint_position_imgs"],
                    )
                )

            if "paint_position_maps" in batch.extra:
                batch.extra["paint_position_maps"] = torch.cat(
                    (
                        batch.extra["paint_position_maps"],
                        batch.extra["paint_position_maps"],
                    )
                )

            if "paint_camera_info_gen" in batch.extra:
                batch.extra["paint_camera_info_gen"] = torch.cat(
                    (
                        batch.extra["paint_camera_info_gen"],
                        batch.extra["paint_camera_info_gen"],
                    )
                )
            if "paint_camera_info_ref" in batch.extra:
                batch.extra["paint_camera_info_ref"] = torch.cat(
                    (
                        batch.extra["paint_camera_info_ref"],
                        batch.extra["paint_camera_info_ref"],
                    )
                )

        model_kwargs = {
            "ref_latents": batch.extra["paint_ref_latents"],
            "num_in_batch": batch.extra["paint_num_in_batch"],
        }
        if "paint_ref_scale" in batch.extra:
            model_kwargs["ref_scale"] = batch.extra["paint_ref_scale"]
        if "paint_normal_imgs" in batch.extra:
            model_kwargs["normal_imgs"] = batch.extra["paint_normal_imgs"]
        if "paint_position_imgs" in batch.extra:
            model_kwargs["position_imgs"] = batch.extra["paint_position_imgs"]
        if "paint_position_maps" in batch.extra:
            model_kwargs["position_maps"] = batch.extra["paint_position_maps"]
        if "paint_camera_info_gen" in batch.extra:
            model_kwargs["camera_info_gen"] = batch.extra["paint_camera_info_gen"]
        if "paint_camera_info_ref" in batch.extra:
            model_kwargs["camera_info_ref"] = batch.extra["paint_camera_info_ref"]
        if "paint_position_attn_mask" in batch.extra:
            model_kwargs["position_attn_mask"] = batch.extra["paint_position_attn_mask"]
        if "paint_position_voxel_indices" in batch.extra:
            model_kwargs["position_voxel_indices"] = batch.extra[
                "paint_position_voxel_indices"
            ]
        batch.extra["paint_model_kwargs"] = model_kwargs

    def _prepare_prompt_embeds(self, batch: Req) -> None:
        prompt_embeds = self.paint_unet.learned_text_clip_gen.repeat(1, 1, 1)
        negative_prompt_embeds = torch.zeros_like(prompt_embeds)
        batch.extra["paint_prompt_embeds"] = prompt_embeds
        batch.extra["paint_negative_prompt_embeds"] = negative_prompt_embeds

    def _prepare_timesteps(
        self,
        batch: Req,
        device: torch.device,
        num_inference_steps: int,
        timesteps: list[int] | None = None,
        sigmas: list[float] | None = None,
    ) -> None:
        if self.paint_is_turbo:
            bsz = 3
            index = torch.arange(29, -1, -bsz, device=device).long()
            timesteps = self.paint_solver.ddim_timesteps[index]
            self.paint_scheduler.set_timesteps(timesteps=timesteps.cpu(), device=device)
        else:
            timesteps, num_inference_steps = retrieve_timesteps(
                self.paint_scheduler, num_inference_steps, device, timesteps, sigmas
            )

        batch.extra["paint_timesteps"] = timesteps
        batch.extra["paint_num_inference_steps"] = num_inference_steps

    def _prepare_latents(
        self,
        batch: Req,
        device: torch.device,
        height: int,
        width: int,
        generator: torch.Generator | None,
        latents: torch.Tensor | None = None,
    ) -> None:
        from diffusers.utils.torch_utils import randn_tensor

        if latents is None:
            num_channels_latents = self.paint_unet.config.in_channels
            shape = (
                batch.extra["paint_num_in_batch"],
                num_channels_latents,
                height // self.paint_vae_scale_factor,
                width // self.paint_vae_scale_factor,
            )
            latents = randn_tensor(
                shape,
                generator=generator,
                device=device,
                dtype=batch.extra["paint_prompt_embeds"].dtype,
            )
        else:
            latents = latents.to(device)

        if hasattr(self.paint_scheduler, "init_noise_sigma"):
            latents = latents * self.paint_scheduler.init_noise_sigma

        batch.extra["paint_latents"] = latents

    def _prepare_extra_step_kwargs(
        self, generator: torch.Generator | None, eta: float
    ) -> dict[str, Any]:
        import inspect

        extra_step_kwargs = {}
        accepts_eta = "eta" in set(
            inspect.signature(self.paint_scheduler.step).parameters.keys()
        )
        if accepts_eta:
            extra_step_kwargs["eta"] = eta

        accepts_generator = "generator" in set(
            inspect.signature(self.paint_scheduler.step).parameters.keys()
        )
        if accepts_generator:
            extra_step_kwargs["generator"] = generator
        return extra_step_kwargs

    @torch.no_grad()
    def _denoise_loop(
        self,
        batch: Req,
        guidance_scale: float,
        guidance_rescale: float = 0.0,
        eta: float = 0.0,
        generator: torch.Generator | None = None,
    ) -> None:
        from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion import (
            rescale_noise_cfg,
        )

        timesteps = batch.extra["paint_timesteps"]
        latents = batch.extra["paint_latents"]
        num_in_batch = batch.extra["paint_num_in_batch"]

        prompt_embeds = batch.extra["paint_prompt_embeds"]
        negative_prompt_embeds = batch.extra["paint_negative_prompt_embeds"]

        do_cfg = guidance_scale > 1 and not self.paint_is_turbo
        if do_cfg:
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])

        extra_step_kwargs = self._prepare_extra_step_kwargs(generator, eta)
        num_channels_latents = self.paint_unet.config.in_channels

        for t in timesteps:
            latents = rearrange(latents, "(b n) c h w -> b n c h w", n=num_in_batch)
            latent_model_input = torch.cat([latents] * 2) if do_cfg else latents
            latent_model_input = rearrange(
                latent_model_input, "b n c h w -> (b n) c h w"
            )
            latent_model_input = self.paint_scheduler.scale_model_input(
                latent_model_input, t
            )
            latent_model_input = rearrange(
                latent_model_input, "(b n) c h w -> b n c h w", n=num_in_batch
            )

            noise_pred = self.paint_unet(
                latent_model_input,
                t,
                encoder_hidden_states=prompt_embeds,
                timestep_cond=None,
                cross_attention_kwargs=None,
                added_cond_kwargs=None,
                return_dict=False,
                **batch.extra["paint_model_kwargs"],
            )[0]

            latents = rearrange(latents, "b n c h w -> (b n) c h w")

            if do_cfg:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (
                    noise_pred_text - noise_pred_uncond
                )

            if do_cfg and guidance_rescale > 0.0:
                noise_pred = rescale_noise_cfg(
                    noise_pred, noise_pred_text, guidance_rescale=guidance_rescale
                )

            latents = self.paint_scheduler.step(
                noise_pred,
                t,
                latents[:, :num_channels_latents, :, :],
                **extra_step_kwargs,
                return_dict=False,
            )[0]

        batch.extra["paint_latents"] = latents

    @torch.no_grad()
    def _decode_images(self, batch: Req, output_type: str) -> list:
        latents = batch.extra["paint_latents"]

        if output_type != "latent":
            image = self.paint_vae.decode(
                latents / self.paint_vae.config.scaling_factor, return_dict=False
            )[0]
        else:
            image = latents

        image = self.paint_image_processor.postprocess(image, output_type=output_type)
        batch.extra["paint_images"] = image
        return image

    def forward(self, batch: Req, server_args: ServerArgs) -> Req:
        self._load_paint_pipeline(server_args)

        delighted_image = batch.extra["delighted_image"]
        normal_maps = batch.extra["normal_maps"]

        if self.paint_unet is not None:
            try:
                num_steps = getattr(self.config, "paint_num_inference_steps", 15)
                guidance_scale = getattr(self.config, "paint_guidance_scale", 3.0)
                render_size = getattr(self.config, "paint_resolution", 512)
                device = self.device

                batch.extra["paint_num_in_batch"] = len(normal_maps)
                self._prepare_conditions(batch, device, guidance_scale)
                self._prepare_prompt_embeds(batch)
                self._prepare_timesteps(batch, device, num_steps)

                generator = batch.generator
                if generator is None and batch.seed is not None:
                    generator = torch.Generator(device=device).manual_seed(batch.seed)

                self._prepare_latents(
                    batch,
                    device,
                    render_size,
                    render_size,
                    generator,
                )
                self._denoise_loop(batch, guidance_scale, generator=generator)
                multiview_textures = self._decode_images(batch, output_type="pil")
                logger.info(
                    "Paint pipeline generated %d textures", len(multiview_textures)
                )
            except Exception as e:
                logger.error(f"Paint pipeline execution failed: {e}")
                # Fallback
                render_size = getattr(self.config, "paint_resolution", 512)
                multiview_textures = [
                    delighted_image.resize((render_size, render_size))
                    for _ in range(len(normal_maps))
                ]
        else:
            # Fallback: use delighted image for all views
            logger.warning(
                "Paint pipeline not available, using reference image for all views"
            )
            render_size = getattr(self.config, "paint_resolution", 512)
            multiview_textures = [
                delighted_image.resize((render_size, render_size))
                for _ in range(len(normal_maps))
            ]

        batch.extra["multiview_textures"] = multiview_textures
        logger.info(f"Generated {len(multiview_textures)} texture views")
        return batch


class PaintPostprocessStage(PipelineStage):
    """Stage 4: Texture baking and mesh export.

    This stage bakes the generated textures onto the mesh UV space
    and exports the final textured mesh.
    """

    def __init__(self, config: Hunyuan3D2PipelineConfig) -> None:
        super().__init__()
        self.config = config

    def forward(self, batch: Req, server_args: ServerArgs) -> OutputBatch:
        renderer = batch.extra["renderer"]
        multiview_textures = batch.extra["multiview_textures"]
        camera_elevs = batch.extra["camera_elevs"]
        camera_azims = batch.extra["camera_azims"]
        view_weights = batch.extra["view_weights"]

        # Resize textures if needed
        render_size = getattr(self.config, "paint_resolution", 512)
        resized_textures = []
        for tex in multiview_textures:
            if hasattr(tex, "resize"):
                resized_textures.append(tex.resize((render_size, render_size)))
            else:
                resized_textures.append(tex)

        # Bake textures from multiple views
        try:
            texture, mask = renderer.bake_from_multiview(
                resized_textures,
                camera_elevs,
                camera_azims,
                view_weights,
                method="fast",
            )

            # Inpaint missing regions
            mask_np = (mask.squeeze(-1).cpu().numpy() * 255).astype("uint8")
            texture = renderer.texture_inpaint(texture, mask_np)

            # Apply texture to mesh
            renderer.set_texture(texture)
            textured_mesh = renderer.save_mesh()
            logger.info("Texture baking completed")
        except Exception as e:
            logger.error(f"Texture baking failed: {e}")
            # Fallback to untextured mesh
            textured_mesh = batch.extra["paint_mesh"]

        # Export mesh
        obj_path = batch.extra["shape_obj_path"]
        return_path = batch.extra["shape_return_path"]

        # Save textured mesh
        try:
            textured_mesh.export(obj_path)

            if return_path.endswith(".glb") and self.config.paint_save_glb:
                glb_path = obj_path[:-4] + ".glb"
                textured_mesh.export(glb_path)
                return_path = glb_path
        except Exception as e:
            logger.error(f"Mesh export failed: {e}")

        return OutputBatch(output=[return_path], timings=batch.timings)


# Legacy PaintStage kept for backward compatibility
class PaintStage(PipelineStage):
    """Legacy paint stage - redirects to new multi-stage implementation."""

    def __init__(self, paint_pipeline: Any, config: Hunyuan3D2PipelineConfig) -> None:
        super().__init__()
        self.paint_pipeline = paint_pipeline
        self.config = config

    @property
    def parallelism_type(self) -> StageParallelismType:
        return StageParallelismType.MAIN_RANK_ONLY

    def forward(self, batch: Req, server_args: ServerArgs) -> OutputBatch:
        obj_path = batch.extra["shape_obj_path"]
        return_path = batch.extra["shape_return_path"]

        if self.paint_pipeline is not None:
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
    def __init__(self, config: Hunyuan3D2PipelineConfig) -> None:
        super().__init__()
        self.config = config

    def forward(self, batch: Req, server_args: ServerArgs) -> OutputBatch:
        obj_path = batch.extra["shape_obj_path"]
        return_path = batch.extra["shape_return_path"]
        if return_path.endswith(".glb"):
            return_path = obj_path
        return OutputBatch(output=[return_path], timings=batch.timings)


class Hunyuan3D2Pipeline(ComposedPipelineBase):
    pipeline_name = "Hunyuan3D2Pipeline"
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
        if not isinstance(config, Hunyuan3D2PipelineConfig):
            raise TypeError(
                "Hunyuan3D2Pipeline requires Hunyuan3D2PipelineConfig, "
                f"got {type(config)}"
            )

        if config.paint_enable:
            self._initialize_paint_pipeline(config)

    def _initialize_paint_pipeline(self, config: Hunyuan3D2PipelineConfig):
        """Initialize the paint pipeline for texture generation.

        This sets up the new 4-stage texture generation pipeline:
        1. PaintPreprocessStage: UV unwrap and image delight
        2. PaintRenderStage: Multi-view normal/position rendering
        3. PaintDiffusionStage: Texture diffusion generation
        4. PaintPostprocessStage: Texture baking and export
        """
        logger.info(
            "Paint pipeline (texture generation) initialized with 4-stage architecture."
        )

    def create_pipeline_stages(self, server_args: ServerArgs):
        config = server_args.pipeline_config
        assert isinstance(config, Hunyuan3D2PipelineConfig)

        self.add_stage(
            stage_name="input_validation_stage", stage=InputValidationStage()
        )

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
            # New 4-stage texture generation pipeline
            self.add_stage(
                stage_name="paint_preprocess_stage",
                stage=PaintPreprocessStage(config=config),
            )
            self.add_stage(
                stage_name="paint_render_stage",
                stage=PaintRenderStage(config=config),
            )
            self.add_stage(
                stage_name="paint_diffusion_stage",
                stage=PaintDiffusionStage(config=config),
            )
            self.add_stage(
                stage_name="paint_postprocess_stage",
                stage=PaintPostprocessStage(config=config),
            )
        else:
            self.add_stage(
                stage_name="paint_stage",
                stage=ShapeOnlyOutputStage(config=config),
            )


EntryClass = Hunyuan3D2Pipeline
