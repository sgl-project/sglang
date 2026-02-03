# SPDX-License-Identifier: Apache-2.0
"""
Hunyuan3D paint/texture generation stages.

This module contains the pipeline stages for Hunyuan3D texture generation,
including preprocessing, rendering, diffusion, and postprocessing stages.
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
from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import OutputBatch, Req
from sglang.multimodal_gen.runtime.pipelines_core.stages.base import (
    PipelineStage,
    StageParallelismType,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.hunyuan3d_shape import (
    retrieve_timesteps,
)
from sglang.multimodal_gen.runtime.server_args import ServerArgs
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

logger = init_logger(__name__)


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


def _recorrect_rgb(
    src_image: torch.Tensor,
    target_image: torch.Tensor,
    alpha_channel: torch.Tensor,
    scale: float = 0.95,
) -> torch.Tensor:
    """Correct RGB values to match target color distribution.

    Source: Hunyuan3D-2/hy3dgen/texgen/utils/dehighlight_utils.py
    """

    def flat_and_mask(bgr, a):
        mask = torch.where(a > 0.5, True, False)
        bgr_flat = bgr.reshape(-1, bgr.shape[-1])
        mask_flat = mask.reshape(-1)
        bgr_flat_masked = bgr_flat[mask_flat, :]
        return bgr_flat_masked

    src_flat = flat_and_mask(src_image, alpha_channel)
    target_flat = flat_and_mask(target_image, alpha_channel)
    corrected_bgr = torch.zeros_like(src_image)

    for i in range(3):
        src_mean, src_stddev = torch.mean(src_flat[:, i]), torch.std(src_flat[:, i])
        target_mean, target_stddev = torch.mean(target_flat[:, i]), torch.std(
            target_flat[:, i]
        )
        corrected_bgr[:, :, i] = torch.clamp(
            (src_image[:, :, i] - scale * src_mean) * (target_stddev / src_stddev)
            + scale * target_mean,
            0,
            1,
        )

    src_mse = torch.mean((src_image - target_image) ** 2)
    modify_mse = torch.mean((corrected_bgr - target_image) ** 2)
    if src_mse < modify_mse:
        corrected_bgr = torch.cat([src_image, alpha_channel], dim=-1)
    else:
        corrected_bgr = torch.cat([corrected_bgr, alpha_channel], dim=-1)

    return corrected_bgr


class Hunyuan3DPaintUVUnwrapStage(PipelineStage):
    """Stage 1a: UV unwrap preprocessing.

    This stage applies UV unwrapping to the mesh using xatlas.
    """

    def __init__(self, config: Hunyuan3D2PipelineConfig) -> None:
        super().__init__()
        self.config = config

    def forward(self, batch: Req, server_args: ServerArgs) -> Req:
        import time

        from sglang.multimodal_gen.runtime.models.mesh3d_utils import mesh_uv_wrap

        # Get mesh from shape generation
        mesh = batch.extra["shape_meshes"]
        if isinstance(mesh, list):
            mesh = mesh[0]

        try:
            start_time = time.time()
            mesh = mesh_uv_wrap(mesh)
            elapsed = time.time() - start_time
            logger.info(f"UV unwrapping completed in {elapsed:.2f}s")
        except Exception as e:
            logger.warning(f"UV unwrapping failed: {e}")

        batch.extra["paint_mesh"] = mesh
        return batch


class Hunyuan3DPaintDelightStage(PipelineStage):
    """Stage 1b: Image delight preprocessing.

    This stage removes lighting/shadows from the reference image
    using the StableDiffusionInstructPix2PixPipeline (delight model).
    """

    def __init__(self, config: Hunyuan3D2PipelineConfig) -> None:
        super().__init__()
        self.config = config
        self.pipeline = None
        self._loaded = False

    def _load_delight_model(self, server_args: ServerArgs):
        """Lazy load the delight model."""
        if self._loaded:
            return

        from diffusers import (
            EulerAncestralDiscreteScheduler,
            StableDiffusionInstructPix2PixPipeline,
        )
        from huggingface_hub import snapshot_download

        # Get model path from config
        model_path = server_args.model_path
        delight_subfolder = getattr(
            self.config, "delight_subfolder", "hunyuan3d-delight-v2-0"
        )

        # Try to load from HuggingFace or local path
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
            pipeline = StableDiffusionInstructPix2PixPipeline.from_pretrained(
                local_path,
                torch_dtype=torch.float16,
                safety_checker=None,
            )
            pipeline.scheduler = EulerAncestralDiscreteScheduler.from_config(
                pipeline.scheduler.config
            )
            pipeline.set_progress_bar_config(disable=True)
            self.pipeline = pipeline.to("cuda", torch.float16)
            self._loaded = True
            logger.info("Delight model loaded successfully")
        else:
            logger.warning(
                "Delight model not available, skipping delight preprocessing"
            )
            self._loaded = True

    @torch.no_grad()
    def _run_delight(self, image):
        """Run the delight pipeline on an image."""
        import cv2
        from PIL import Image as PILImage

        image = image.resize((512, 512))

        if image.mode == "RGBA":
            image_array = np.array(image)
            alpha_channel = image_array[:, :, 3]
            erosion_size = 3
            kernel = np.ones((erosion_size, erosion_size), np.uint8)
            alpha_channel = cv2.erode(alpha_channel, kernel, iterations=1)
            image_array[alpha_channel == 0, :3] = 255
            image_array[:, :, 3] = alpha_channel
            image = PILImage.fromarray(image_array)

            image_tensor = torch.tensor(np.array(image) / 255.0).to("cuda")
            alpha = image_tensor[:, :, 3:]
            rgb_target = image_tensor[:, :, :3]
        else:
            image_tensor = torch.tensor(np.array(image) / 255.0).to("cuda")
            alpha = torch.ones_like(image_tensor)[:, :, :1]
            rgb_target = image_tensor[:, :, :3]

        image = image.convert("RGB")

        image = self.pipeline(
            prompt=self.config.delight_prompt,
            negative_prompt=self.config.delight_negative_prompt,
            image=image,
            generator=torch.manual_seed(42),
            height=512,
            width=512,
            strength=self.config.delight_strength,
            num_inference_steps=self.config.delight_num_inference_steps,
            image_guidance_scale=self.config.delight_cfg_image,
            guidance_scale=self.config.delight_guidance_scale,
        ).images[0]

        image_tensor = torch.tensor(np.array(image) / 255.0).to("cuda")
        rgb_src = image_tensor[:, :, :3]
        image = _recorrect_rgb(rgb_src, rgb_target, alpha)
        image = image[:, :, :3] * image[:, :, 3:] + torch.ones_like(image[:, :, :3]) * (
            1.0 - image[:, :, 3:]
        )
        from PIL import Image as PILImage

        image = PILImage.fromarray((image.cpu().numpy() * 255).astype(np.uint8))

        return image

    def forward(self, batch: Req, server_args: ServerArgs) -> Req:
        from PIL import Image

        # Load reference image
        image = Image.open(batch.image_path)

        # Check if delight is enabled
        if not self.config.delight_enable:
            logger.info("Delight preprocessing disabled, using original image")
            batch.extra["delighted_image"] = image
            return batch

        # Apply delight if model is available
        self._load_delight_model(server_args)
        if self.pipeline is not None:
            try:
                image = self._run_delight(image)
                logger.info("Image delight completed")
            except Exception as e:
                logger.warning(f"Image delight failed: {e}")

        batch.extra["delighted_image"] = image
        return batch


class Hunyuan3DPaintRenderStage(PipelineStage):
    """Stage 2: Multi-view normal and position map rendering.

    This stage renders the mesh from multiple viewpoints to create
    conditioning inputs for the texture diffusion model.
    """

    # Camera configuration constants for 6 views
    CAMERA_AZIMS = [0, 90, 180, 270, 0, 180]
    CAMERA_ELEVS = [0, 0, 0, 0, 90, -90]
    VIEW_WEIGHTS = [1, 0.1, 0.5, 0.1, 0.05, 0.05]

    def __init__(self, config: Hunyuan3D2PipelineConfig) -> None:
        super().__init__()
        self.config = config
        self.renderer = None

    def _init_renderer(self):
        """Initialize the mesh renderer."""
        if self.renderer is not None:
            return

        from sglang.multimodal_gen.runtime.models.mesh3d_utils import MeshRender

        self.renderer = MeshRender(
            default_resolution=self.config.paint_render_size,
            texture_size=self.config.paint_texture_size,
        )
        logger.info("Mesh renderer initialized")

    def forward(self, batch: Req, server_args: ServerArgs) -> Req:
        self._init_renderer()

        mesh = batch.extra["paint_mesh"]

        # Load mesh into renderer
        self.renderer.load_mesh(mesh)

        # Render normal maps
        normal_maps = self.renderer.render_normal_multiview(
            self.CAMERA_ELEVS, self.CAMERA_AZIMS, use_abs_coor=True
        )

        # Render position maps
        position_maps = self.renderer.render_position_multiview(
            self.CAMERA_ELEVS, self.CAMERA_AZIMS
        )

        batch.extra["normal_maps"] = normal_maps
        batch.extra["position_maps"] = position_maps
        batch.extra["camera_azims"] = self.CAMERA_AZIMS
        batch.extra["camera_elevs"] = self.CAMERA_ELEVS
        batch.extra["view_weights"] = self.VIEW_WEIGHTS
        batch.extra["renderer"] = self.renderer

        logger.info(f"Rendered {len(normal_maps)} views for texture generation")
        return batch


class Hunyuan3DPaintDiffusionStage(PipelineStage):
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

            from sglang.multimodal_gen.runtime.models.dits.hunyuan3d import (
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

        # Resize normal/position maps to paint_resolution before encoding
        # This ensures latent dimensions match between noise and condition maps
        target_size = self.config.paint_resolution
        if isinstance(normal_maps, list):
            normal_maps = [
                (
                    img.resize((target_size, target_size))
                    if hasattr(img, "resize")
                    else img
                )
                for img in normal_maps
            ]
            normal_maps = self._convert_pil_list_to_tensor([normal_maps], device)
        if isinstance(position_maps, list):
            position_maps = [
                (
                    img.resize((target_size, target_size))
                    if hasattr(img, "resize")
                    else img
                )
                for img in position_maps
            ]
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
            from sglang.multimodal_gen.runtime.models.dits.hunyuan3d import (
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
                # Get parameters from batch (SamplingParams) or fallback to config
                num_steps = (
                    getattr(batch, "paint_num_inference_steps", None)
                    or self.config.paint_num_inference_steps
                )
                guidance_scale = (
                    getattr(batch, "paint_guidance_scale", None)
                    or self.config.paint_guidance_scale
                )
                render_size = self.config.paint_resolution
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
                render_size = self.config.paint_resolution
                multiview_textures = [
                    delighted_image.resize((render_size, render_size))
                    for _ in range(len(normal_maps))
                ]
        else:
            # Fallback: use delighted image for all views
            logger.warning(
                "Paint pipeline not available, using reference image for all views"
            )
            render_size = self.config.paint_resolution
            multiview_textures = [
                delighted_image.resize((render_size, render_size))
                for _ in range(len(normal_maps))
            ]

        batch.extra["multiview_textures"] = multiview_textures
        logger.info(f"Generated {len(multiview_textures)} texture views")
        return batch


class Hunyuan3DPaintPostprocessStage(PipelineStage):
    """Stage 4: Texture baking and mesh export.

    This stage bakes the generated textures onto the mesh UV space
    and exports the final textured mesh.
    """

    def __init__(self, config: Hunyuan3D2PipelineConfig) -> None:
        super().__init__()
        self.config = config

    def forward(self, batch: Req, server_args: ServerArgs) -> OutputBatch:
        import time

        renderer = batch.extra["renderer"]
        multiview_textures = batch.extra["multiview_textures"]
        camera_elevs = batch.extra["camera_elevs"]
        camera_azims = batch.extra["camera_azims"]
        view_weights = batch.extra["view_weights"]

        print(f"[DEBUG Postprocess] Starting with {len(multiview_textures)} textures")

        # Resize textures if needed
        render_size = getattr(self.config, "paint_resolution", 512)
        resized_textures = []
        for tex in multiview_textures:
            if hasattr(tex, "resize"):
                resized_textures.append(tex.resize((render_size, render_size)))
            else:
                resized_textures.append(tex)
        print(
            f"[DEBUG Postprocess] Resized {len(resized_textures)} textures to {render_size}x{render_size}"
        )

        # Bake textures from multiple views
        try:
            print("[DEBUG Postprocess] Starting bake_from_multiview()...")
            start_time = time.time()
            texture, mask = renderer.bake_from_multiview(
                resized_textures,
                camera_elevs,
                camera_azims,
                view_weights,
                method="fast",
            )
            elapsed = time.time() - start_time
            print(
                f"[DEBUG Postprocess] bake_from_multiview() completed in {elapsed:.2f} seconds"
            )
            print(
                f"[DEBUG Postprocess] texture shape: {texture.shape if hasattr(texture, 'shape') else 'N/A'}"
            )

            # Inpaint missing regions
            print("[DEBUG Postprocess] Starting texture_inpaint()...")
            start_time = time.time()
            mask_np = (mask.squeeze(-1).cpu().numpy() * 255).astype("uint8")
            texture = renderer.texture_inpaint(texture, mask_np)
            elapsed = time.time() - start_time
            print(
                f"[DEBUG Postprocess] texture_inpaint() completed in {elapsed:.2f} seconds"
            )

            # Apply texture to mesh
            print("[DEBUG Postprocess] Setting texture...")
            start_time = time.time()
            renderer.set_texture(texture)
            elapsed = time.time() - start_time
            print(
                f"[DEBUG Postprocess] set_texture() completed in {elapsed:.2f} seconds"
            )

            print("[DEBUG Postprocess] Saving mesh...")
            start_time = time.time()
            textured_mesh = renderer.save_mesh()
            elapsed = time.time() - start_time
            print(f"[DEBUG Postprocess] save_mesh() completed in {elapsed:.2f} seconds")
            logger.info("Texture baking completed")
        except Exception as e:
            print(f"[DEBUG Postprocess] Texture baking failed: {e}")
            logger.error(f"Texture baking failed: {e}")
            # Fallback to untextured mesh
            textured_mesh = batch.extra["paint_mesh"]

        # Export mesh
        obj_path = batch.extra["shape_obj_path"]
        return_path = batch.extra["shape_return_path"]
        print(f"[DEBUG Postprocess] Exporting mesh to {obj_path}")

        # Save textured mesh
        try:
            print("[DEBUG Postprocess] Starting mesh export...")
            start_time = time.time()
            textured_mesh.export(obj_path)
            elapsed = time.time() - start_time
            print(f"[DEBUG Postprocess] Mesh exported in {elapsed:.2f} seconds")

            if return_path.endswith(".glb") and self.config.paint_save_glb:
                print("[DEBUG Postprocess] Exporting GLB...")
                start_time = time.time()
                glb_path = obj_path[:-4] + ".glb"
                textured_mesh.export(glb_path)
                elapsed = time.time() - start_time
                print(f"[DEBUG Postprocess] GLB exported in {elapsed:.2f} seconds")
                return_path = glb_path
        except Exception as e:
            print(f"[DEBUG Postprocess] Mesh export failed: {e}")
            logger.error(f"Mesh export failed: {e}")

        return OutputBatch(output=[return_path], timings=batch.timings)


# Legacy PaintStage kept for backward compatibility
class Hunyuan3DPaintStage(PipelineStage):
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


__all__ = [
    "Hunyuan3DPaintUVUnwrapStage",
    "Hunyuan3DPaintDelightStage",
    "Hunyuan3DPaintRenderStage",
    "Hunyuan3DPaintDiffusionStage",
    "Hunyuan3DPaintPostprocessStage",
    "Hunyuan3DPaintStage",
]
