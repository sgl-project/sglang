"""
Hunyuan3D paint/texture generation stages.

Three-stage pipeline: Preprocess -> TexGen -> Postprocess.
"""

from __future__ import annotations

import os
from typing import Any

import numpy as np
import torch
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion import (
    retrieve_timesteps,
)
from einops import rearrange

from sglang.multimodal_gen.configs.pipeline_configs.hunyuan3d import (
    Hunyuan3D2PipelineConfig,
)
from sglang.multimodal_gen.runtime.managers.forward_context import set_forward_context
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
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

logger = init_logger(__name__)


# Utility functions
def guidance_scale_embedding(
    w: torch.Tensor, embedding_dim: int = 512, dtype: torch.dtype = torch.float32
) -> torch.Tensor:
    """Generate guidance scale embeddings."""
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
    """Extract values from tensor and reshape for multi-view generation."""
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
    """Get predicted original sample from model output."""
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
    """Get predicted noise from model output."""
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
    """Convert RGBA image to RGB."""
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
    """DDIM solver for fast sampling."""

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
    """Correct RGB values to match target color distribution."""

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


# Stage 1: Preprocess (UV unwrap + delight + multi-view rendering)
class Hunyuan3DPaintPreprocessStage(PipelineStage):
    """Preprocessing: UV unwrap + delight in parallel, then multi-view rendering."""

    CAMERA_AZIMS = [0, 90, 180, 270, 0, 180]
    CAMERA_ELEVS = [0, 0, 0, 0, 90, -90]
    VIEW_WEIGHTS = [1, 0.1, 0.5, 0.1, 0.05, 0.05]

    @property
    def parallelism_type(self) -> StageParallelismType:
        return StageParallelismType.MAIN_RANK_ONLY

    def __init__(self, config: Hunyuan3D2PipelineConfig) -> None:
        super().__init__()
        self.config = config
        self._delight_pipeline = None
        self._delight_loaded = False
        self._renderer = None
        self._renderer_loaded = False

    # --- UV unwrap ---

    def _do_uv_unwrap(self, batch: Req, server_args: ServerArgs) -> Req:
        import time

        from sglang.multimodal_gen.runtime.utils.mesh3d_utils import mesh_uv_wrap

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

    # --- Delight ---

    def _load_delight_model(self, server_args: ServerArgs):
        if self._delight_loaded:
            return

        from diffusers import (
            EulerAncestralDiscreteScheduler,
            StableDiffusionInstructPix2PixPipeline,
        )
        from huggingface_hub import snapshot_download

        model_path = server_args.model_path
        delight_subfolder = getattr(
            self.config, "delight_subfolder", "hunyuan3d-delight-v2-0"
        )

        local_path = os.path.join(model_path, delight_subfolder)
        if not os.path.exists(local_path):
            local_path = os.path.expanduser(local_path)

        if not os.path.exists(local_path):
            try:
                downloaded = snapshot_download(
                    repo_id=model_path,
                    allow_patterns=[f"{delight_subfolder}/*"],
                )
                local_path = os.path.join(downloaded, delight_subfolder)
            except Exception as e:
                logger.warning("Could not download delight model: %s", e)
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
            self._delight_pipeline = pipeline.to(self.device, torch.float16)
            logger.info("Delight model loaded successfully")
        else:
            logger.warning(
                "Delight model not available, skipping delight preprocessing"
            )

        self._delight_loaded = True

    @torch.no_grad()
    def _run_delight(self, image):
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

            image_tensor = torch.tensor(np.array(image) / 255.0).to(self.device)
            alpha = image_tensor[:, :, 3:]
            rgb_target = image_tensor[:, :, :3]
        else:
            image_tensor = torch.tensor(np.array(image) / 255.0).to(self.device)
            alpha = torch.ones_like(image_tensor)[:, :, :1]
            rgb_target = image_tensor[:, :, :3]

        image = image.convert("RGB")

        image = self._delight_pipeline(
            prompt=self.config.delight_prompt,
            image=image,
            generator=torch.manual_seed(42),
            height=512,
            width=512,
            num_inference_steps=self.config.delight_num_inference_steps,
            image_guidance_scale=self.config.delight_cfg_image,
            guidance_scale=self.config.delight_guidance_scale,
        ).images[0]

        image_tensor = torch.tensor(np.array(image) / 255.0).to(self.device)
        rgb_src = image_tensor[:, :, :3]
        image = _recorrect_rgb(rgb_src, rgb_target, alpha)
        image = image[:, :, :3] * image[:, :, 3:] + torch.ones_like(image[:, :, :3]) * (
            1.0 - image[:, :, 3:]
        )
        image = PILImage.fromarray((image.cpu().numpy() * 255).astype(np.uint8))

        return image

    def _do_delight(self, batch: Req, server_args: ServerArgs) -> Req:
        from PIL import Image

        from sglang.multimodal_gen.runtime.utils.mesh3d_utils import recenter_image

        image = Image.open(batch.image_path)
        image = recenter_image(image)

        if not self.config.delight_enable:
            logger.info("Delight preprocessing disabled, using original image")
            batch.extra["delighted_image"] = image
            return batch

        self._load_delight_model(server_args)
        if self._delight_pipeline is not None:
            try:
                image = self._run_delight(image)
                logger.info("Image delight completed")
            except Exception as e:
                logger.warning(f"Image delight failed: {e}")

        batch.extra["delighted_image"] = image
        return batch

    # --- Multi-view rendering ---

    def _init_renderer(self):
        if self._renderer_loaded:
            return

        from sglang.multimodal_gen.runtime.utils.mesh3d_utils import MeshRender

        self._renderer = MeshRender(
            default_resolution=self.config.paint_render_size,
            texture_size=self.config.paint_texture_size,
        )
        self._renderer_loaded = True
        logger.info("Mesh renderer initialized")

    def _render_multiview(self, mesh) -> tuple:
        self._init_renderer()
        self._renderer.load_mesh(mesh)

        normal_maps = self._renderer.render_normal_multiview(
            self.CAMERA_ELEVS, self.CAMERA_AZIMS, use_abs_coor=True
        )
        position_maps = self._renderer.render_position_multiview(
            self.CAMERA_ELEVS, self.CAMERA_AZIMS
        )

        logger.info(f"Rendered {len(normal_maps)} views for texture generation")
        return normal_maps, position_maps

    # --- Forward ---

    def forward(self, batch: Req, server_args: ServerArgs) -> Req:
        if batch.extra.get("_mesh_failed"):
            logger.warning("Mesh generation failed, skipping paint preprocessing")
            batch.extra["paint_mesh"] = None
            batch.extra["delighted_image"] = None
            batch.extra["normal_maps"] = []
            batch.extra["position_maps"] = []
            batch.extra["camera_azims"] = self.CAMERA_AZIMS
            batch.extra["camera_elevs"] = self.CAMERA_ELEVS
            batch.extra["view_weights"] = self.VIEW_WEIGHTS
            batch.extra["renderer"] = None
            return batch

        import concurrent.futures
        import copy

        # 1. UV unwrap + delight in parallel
        batch_for_uv = batch
        batch_for_delight = copy.copy(batch)
        batch_for_delight.extra = batch.extra.copy()

        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            uv_future = executor.submit(self._do_uv_unwrap, batch_for_uv, server_args)
            delight_future = executor.submit(
                self._do_delight, batch_for_delight, server_args
            )
            uv_future.result()
            delight_future.result()

        batch.extra["paint_mesh"] = batch_for_uv.extra.get("paint_mesh")
        batch.extra["delighted_image"] = batch_for_delight.extra.get("delighted_image")

        # 2. Multi-view rendering
        normal_maps, position_maps = self._render_multiview(batch.extra["paint_mesh"])
        batch.extra["normal_maps"] = normal_maps
        batch.extra["position_maps"] = position_maps
        batch.extra["camera_azims"] = self.CAMERA_AZIMS
        batch.extra["camera_elevs"] = self.CAMERA_ELEVS
        batch.extra["view_weights"] = self.VIEW_WEIGHTS
        batch.extra["renderer"] = self._renderer

        return batch

    def verify_input(self, batch: Req, server_args: ServerArgs) -> VerificationResult:
        result = VerificationResult()
        result.add_check("shape_meshes", batch.extra.get("shape_meshes"), V.not_none)
        result.add_check("image_path", batch.image_path, V.not_none)
        return result

    def verify_output(self, batch: Req, server_args: ServerArgs) -> VerificationResult:
        result = VerificationResult()
        result.add_check("paint_mesh", batch.extra.get("paint_mesh"), V.not_none)
        result.add_check(
            "delighted_image", batch.extra.get("delighted_image"), V.not_none
        )
        result.add_check("normal_maps", batch.extra.get("normal_maps"), V.is_list)
        result.add_check("position_maps", batch.extra.get("position_maps"), V.is_list)
        result.add_check("renderer", batch.extra.get("renderer"), V.not_none)
        return result


# Stage 2: TexGen (model loading + input prep + denoising + decode)
class Hunyuan3DPaintTexGenStage(PipelineStage):
    def __init__(
        self,
        config: Hunyuan3D2PipelineConfig,
        paint_dir: str | None = None,
        transformer: Any = None,
        scheduler: Any = None,
        vae: Any = None,
        vae_scale_factor: int = 8,
        image_processor: Any = None,
        solver: Any = None,
        is_turbo: bool = False,
    ) -> None:
        super().__init__()
        self.config = config
        self.paint_dir = paint_dir
        self.transformer = transformer
        self.scheduler = scheduler
        self.vae = vae
        self.vae_scale_factor = vae_scale_factor
        self.image_processor = image_processor
        self.solver = solver
        self.is_turbo = is_turbo
        self._loaded = transformer is not None

    @property
    def parallelism_type(self) -> StageParallelismType:
        return StageParallelismType.MAIN_RANK_ONLY

    def _load_paint_models(self, server_args: ServerArgs) -> None:
        """Load paint models from pre-resolved local path (no network)."""
        if self._loaded:
            return
        if self.paint_dir is None:
            logger.warning("No paint model directory resolved, skipping")
            self._loaded = True
            return
        try:
            self._do_load_paint(server_args)
            logger.info("Paint pipeline loaded successfully")
        except Exception as e:
            logger.warning("Failed to load paint pipeline: %s", e)
            self.vae = None
            self.transformer = None
            self.scheduler = None
        self._loaded = True

    def _do_load_paint(self, server_args: ServerArgs) -> None:
        import json

        from diffusers import AutoencoderKL
        from diffusers.image_processor import VaeImageProcessor

        from sglang.multimodal_gen.runtime.models.dits.hunyuan3d import (
            UNet2p5DConditionModel,
        )

        local_path = self.paint_dir
        logger.info("Loading paint model from %s", local_path)
        vae_dir = os.path.join(local_path, "vae")
        with open(os.path.join(vae_dir, "config.json"), "r") as f:
            vae_config = json.load(f)
        vae_config = {k: v for k, v in vae_config.items() if not k.startswith("_")}
        self.vae = AutoencoderKL(**vae_config)
        st_path = os.path.join(vae_dir, "diffusion_pytorch_model.safetensors")
        bin_path = os.path.join(vae_dir, "diffusion_pytorch_model.bin")
        if os.path.exists(st_path):
            from safetensors.torch import load_file

            state_dict = load_file(st_path)
        elif os.path.exists(bin_path):
            state_dict = torch.load(bin_path, map_location="cpu", weights_only=True)
        else:
            raise FileNotFoundError(f"No VAE weights in {vae_dir}")
        self.vae.load_state_dict(state_dict)
        self.vae = self.vae.to(device=self.device, dtype=torch.float16).eval()
        self.transformer = UNet2p5DConditionModel.from_pretrained(
            os.path.join(local_path, "unet"),
            torch_dtype=torch.float16,
        ).to(self.device)
        self.is_turbo = bool(getattr(self.config, "paint_turbo_mode", False))
        sched_path = os.path.join(local_path, "scheduler", "scheduler_config.json")
        with open(sched_path, "r") as f:
            sched_cfg = json.load(f)
        if self.is_turbo:
            from diffusers import LCMScheduler

            self.scheduler = LCMScheduler.from_config(sched_cfg)
        else:
            from diffusers import EulerAncestralDiscreteScheduler

            self.scheduler = EulerAncestralDiscreteScheduler.from_config(
                sched_cfg, timestep_spacing="trailing"
            )
        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
        self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor)
        self.solver = DDIMSolver(
            self.scheduler.alphas_cumprod.cpu().numpy(),
            timesteps=self.scheduler.config.num_train_timesteps,
            ddim_timesteps=30,
        ).to(self.device)
        if server_args.enable_torch_compile:
            compile_mode = os.environ.get(
                "SGLANG_TORCH_COMPILE_MODE", "max-autotune-no-cudagraphs"
            )
            logger.info("Compiling paint transformer with mode: %s", compile_mode)
            self.transformer.compile(mode=compile_mode, fullgraph=False, dynamic=None)

    def _convert_pil_list_to_tensor(
        self, images: list, device: torch.device
    ) -> torch.Tensor:
        bg_c = [1.0, 1.0, 1.0]
        images_tensor = []
        for batch_imgs in images:
            view_imgs = []
            for pil_img in batch_imgs:
                if pil_img.mode == "L":
                    pil_img = pil_img.point(
                        lambda x: 255 if x > 1 else 0, mode="1"
                    ).convert("RGB")
                img = np.asarray(pil_img, dtype=np.float32) / 255.0
                if img.shape[2] > 3:
                    alpha = img[:, :, 3:]
                    img = img[:, :, :3] * alpha + bg_c * (1 - alpha)
                img = (
                    torch.from_numpy(img)
                    .permute(2, 0, 1)
                    .unsqueeze(0)
                    .contiguous()
                    .to(device=device, dtype=self.vae.dtype)
                )
                view_imgs.append(img)
            view_imgs = torch.cat(view_imgs, dim=0)
            images_tensor.append(view_imgs.unsqueeze(0))
        return torch.cat(images_tensor, dim=0)

    @torch.no_grad()
    def _encode_images(self, images: torch.Tensor) -> torch.Tensor:
        batch_size = images.shape[0]
        images = rearrange(images, "b n c h w -> (b n) c h w")
        dtype = next(self.vae.parameters()).dtype
        images = (images - 0.5) * 2.0
        posterior = self.vae.encode(images.to(dtype)).latent_dist
        latents = posterior.sample() * self.vae.config.scaling_factor
        return rearrange(latents, "(b n) c h w -> b n c h w", b=batch_size)

    @staticmethod
    def _compute_camera_index(azim: float, elev: float) -> int:
        base_idx = int(((azim // 30) + 9) % 12)
        if elev == 0:
            base, divisor = 12, 1
        elif elev == 20:
            base, divisor = 24, 1
        elif elev == -20:
            base, divisor = 0, 1
        elif elev == 90:
            base, divisor = 40, 3
        elif elev == -90:
            base, divisor = 36, 3
        else:
            base, divisor = 12, 1
        return base + (base_idx // divisor)

    def _prepare_denoising_inputs(
        self,
        batch: Req,
        server_args: ServerArgs,
    ) -> dict[str, Any]:
        import random

        from diffusers.utils.torch_utils import randn_tensor

        device = self.device
        normal_maps = batch.extra["normal_maps"]
        position_maps = batch.extra["position_maps"]
        camera_azims = batch.extra["camera_azims"]
        camera_elevs = batch.extra["camera_elevs"]

        num_steps = self.config.paint_num_inference_steps
        guidance_scale = self.config.paint_guidance_scale
        render_size = self.config.paint_resolution
        num_in_batch = len(normal_maps)

        seed = 0
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        generator = torch.Generator(device=device).manual_seed(seed)

        image = batch.extra["delighted_image"]
        if not isinstance(image, list):
            image = [image]
        image = [to_rgb_image(img) for img in image]

        image_vae = [
            torch.tensor(np.array(img, dtype=np.float32) / 255.0) for img in image
        ]
        image_vae = [
            iv.unsqueeze(0).permute(0, 3, 1, 2).unsqueeze(0) for iv in image_vae
        ]
        image_vae = torch.cat(image_vae, dim=1).to(device=device, dtype=self.vae.dtype)
        ref_latents = self._encode_images(image_vae)

        target_size = render_size
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

        normal_imgs = (
            self._encode_images(normal_maps) if normal_maps is not None else None
        )
        position_imgs = (
            self._encode_images(position_maps) if position_maps is not None else None
        )

        camera_info = [
            self._compute_camera_index(azim, elev)
            for azim, elev in zip(camera_azims, camera_elevs)
        ]
        camera_info_gen = torch.tensor([camera_info], device=device, dtype=torch.int64)
        camera_info_ref = torch.tensor([[0]], device=device, dtype=torch.int64)

        do_cfg = guidance_scale > 1 and not self.is_turbo

        if self.is_turbo and position_maps is not None:
            from sglang.multimodal_gen.runtime.models.dits.hunyuan3d import (
                compute_multi_resolution_discrete_voxel_indice,
                compute_multi_resolution_mask,
            )

            position_attn_mask = compute_multi_resolution_mask(position_maps)
            position_voxel_indices = compute_multi_resolution_discrete_voxel_indice(
                position_maps
            )
        else:
            position_attn_mask = None
            position_voxel_indices = None

        if do_cfg:
            negative_ref_latents = torch.zeros_like(ref_latents)
            ref_latents = torch.cat([negative_ref_latents, ref_latents])
            ref_scale = torch.as_tensor([0.0, 1.0]).to(ref_latents)
            if normal_imgs is not None:
                normal_imgs = torch.cat((normal_imgs, normal_imgs))
            if position_imgs is not None:
                position_imgs = torch.cat((position_imgs, position_imgs))
            if position_maps is not None:
                position_maps = torch.cat((position_maps, position_maps))
            camera_info_gen = torch.cat((camera_info_gen, camera_info_gen))
            camera_info_ref = torch.cat((camera_info_ref, camera_info_ref))
        else:
            ref_scale = None

        model_kwargs = {
            "ref_latents": ref_latents,
            "num_in_batch": num_in_batch,
        }
        if ref_scale is not None:
            model_kwargs["ref_scale"] = ref_scale
        if normal_imgs is not None:
            model_kwargs["normal_imgs"] = normal_imgs
        if position_imgs is not None:
            model_kwargs["position_imgs"] = position_imgs
        if position_maps is not None:
            model_kwargs["position_maps"] = position_maps
        model_kwargs["camera_info_gen"] = camera_info_gen
        model_kwargs["camera_info_ref"] = camera_info_ref
        if position_attn_mask is not None:
            model_kwargs["position_attn_mask"] = position_attn_mask
        if position_voxel_indices is not None:
            model_kwargs["position_voxel_indices"] = position_voxel_indices

        prompt_embeds = self.transformer.learned_text_clip_gen.repeat(1, 1, 1)
        negative_prompt_embeds = torch.zeros_like(prompt_embeds)

        if self.is_turbo:
            bsz = 3
            index = torch.arange(29, -1, -bsz, device=device).long()
            timesteps = self.solver.ddim_timesteps[index]
            self.scheduler.set_timesteps(timesteps=timesteps.cpu(), device=device)
            timesteps = self.scheduler.timesteps
        else:
            timesteps, num_steps = retrieve_timesteps(
                self.scheduler, num_steps, device, None, None
            )

        num_channels_latents = self.transformer.config.in_channels
        latent_shape = (
            num_in_batch,
            num_channels_latents,
            render_size // self.vae_scale_factor,
            render_size // self.vae_scale_factor,
        )
        latents = randn_tensor(
            latent_shape, generator=generator, device=device, dtype=prompt_embeds.dtype
        )
        latents = latents * self.scheduler.init_noise_sigma

        return {
            "timesteps": timesteps,
            "latents": latents,
            "prompt_embeds": prompt_embeds,
            "negative_prompt_embeds": negative_prompt_embeds,
            "model_kwargs": model_kwargs,
            "num_in_batch": num_in_batch,
            "num_inference_steps": num_steps,
            "guidance_scale": guidance_scale,
            "do_cfg": do_cfg,
            "generator": generator,
            "num_channels_latents": num_channels_latents,
        }

    @torch.no_grad()
    def _denoise_loop(
        self,
        timesteps: torch.Tensor,
        latents: torch.Tensor,
        prompt_embeds: torch.Tensor,
        negative_prompt_embeds: torch.Tensor,
        model_kwargs: dict[str, Any],
        num_in_batch: int,
        guidance_scale: float,
        do_cfg: bool,
        generator: torch.Generator,
        num_channels_latents: int,
    ) -> torch.Tensor:
        import inspect

        if do_cfg:
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])

        extra_step_kwargs = {}
        if "eta" in inspect.signature(self.scheduler.step).parameters:
            extra_step_kwargs["eta"] = 0.0
        if "generator" in inspect.signature(self.scheduler.step).parameters:
            extra_step_kwargs["generator"] = generator

        for step_idx, t in enumerate(timesteps):
            latents = rearrange(latents, "(b n) c h w -> b n c h w", n=num_in_batch)
            latent_model_input = torch.cat([latents] * 2) if do_cfg else latents
            latent_model_input = rearrange(
                latent_model_input, "b n c h w -> (b n) c h w"
            )
            latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)
            latent_model_input = rearrange(
                latent_model_input, "(b n) c h w -> b n c h w", n=num_in_batch
            )

            with set_forward_context(
                current_timestep=step_idx,
                attn_metadata=None,
            ):
                noise_pred = self.transformer(
                    latent_model_input,
                    t,
                    encoder_hidden_states=prompt_embeds,
                    timestep_cond=None,
                    cross_attention_kwargs=None,
                    added_cond_kwargs=None,
                    return_dict=False,
                    **model_kwargs,
                )[0]

            latents = rearrange(latents, "b n c h w -> (b n) c h w")

            if do_cfg:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (
                    noise_pred_text - noise_pred_uncond
                )

            latents = self.scheduler.step(
                noise_pred,
                t,
                latents[:, :num_channels_latents, :, :],
                **extra_step_kwargs,
                return_dict=False,
            )[0]

        return latents

    @torch.no_grad()
    def _decode_latents(self, latents: torch.Tensor) -> list:
        image = self.vae.decode(
            latents / self.vae.config.scaling_factor, return_dict=False
        )[0]
        return self.image_processor.postprocess(image, output_type="pil")

    def forward(self, batch: Req, server_args: ServerArgs) -> Req:
        if batch.extra.get("_mesh_failed"):
            logger.warning("Mesh generation failed, skipping paint texgen")
            batch.extra["multiview_textures"] = []
            return batch

        self._load_paint_models(server_args)

        delighted_image = batch.extra["delighted_image"]
        normal_maps = batch.extra["normal_maps"]

        if self.transformer is not None:
            try:
                prepared = self._prepare_denoising_inputs(batch, server_args)

                latents = self._denoise_loop(
                    timesteps=prepared["timesteps"],
                    latents=prepared["latents"],
                    prompt_embeds=prepared["prompt_embeds"],
                    negative_prompt_embeds=prepared["negative_prompt_embeds"],
                    model_kwargs=prepared["model_kwargs"],
                    num_in_batch=prepared["num_in_batch"],
                    guidance_scale=prepared["guidance_scale"],
                    do_cfg=prepared["do_cfg"],
                    generator=prepared["generator"],
                    num_channels_latents=prepared["num_channels_latents"],
                )

                multiview_textures = self._decode_latents(latents)
                logger.info(
                    "Paint pipeline generated %d textures", len(multiview_textures)
                )

            except Exception as e:
                logger.error(f"Paint pipeline execution failed: {e}")
                import traceback

                traceback.print_exc()
                render_size = self.config.paint_resolution
                multiview_textures = [
                    delighted_image.resize((render_size, render_size))
                    for _ in range(len(normal_maps))
                ]
        else:
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

    def verify_input(self, batch: Req, server_args: ServerArgs) -> VerificationResult:
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
        result = VerificationResult()
        result.add_check(
            "multiview_textures", batch.extra.get("multiview_textures"), V.is_list
        )
        return result


# Stage 3: Postprocess (texture baking + mesh export)
class Hunyuan3DPaintPostprocessStage(PipelineStage):
    """Texture baking from multi-view images and final mesh export."""

    @property
    def parallelism_type(self) -> StageParallelismType:
        return StageParallelismType.MAIN_RANK_ONLY

    def __init__(self, config: Hunyuan3D2PipelineConfig) -> None:
        super().__init__()
        self.config = config

    def forward(self, batch: Req, server_args: ServerArgs) -> OutputBatch:
        if batch.extra.get("_mesh_failed"):
            logger.warning("Mesh generation failed, skipping paint postprocess")
            return OutputBatch(output_file_paths=[], metrics=batch.metrics)

        renderer = batch.extra["renderer"]
        multiview_textures = batch.extra["multiview_textures"]
        camera_elevs = batch.extra["camera_elevs"]
        camera_azims = batch.extra["camera_azims"]
        view_weights = batch.extra["view_weights"]

        render_size = getattr(self.config, "paint_render_size", 2048)
        resized_textures = []
        for tex in multiview_textures:
            if hasattr(tex, "resize"):
                resized_textures.append(tex.resize((render_size, render_size)))
            else:
                resized_textures.append(tex)

        try:
            texture, mask = renderer.bake_from_multiview(
                resized_textures,
                camera_elevs,
                camera_azims,
                view_weights,
                method="fast",
            )

            mask_np = (mask.squeeze(-1).cpu().numpy() * 255).astype("uint8")
            texture = renderer.texture_inpaint(texture, mask_np)

            renderer.set_texture(texture)
            textured_mesh = renderer.save_mesh()
            logger.info("Texture baking completed")
        except Exception as e:
            logger.error(f"Texture baking failed: {e}")
            textured_mesh = batch.extra["paint_mesh"]

        obj_path = batch.extra["shape_obj_path"]
        return_path = batch.extra["shape_return_path"]

        try:
            textured_mesh.export(obj_path)
            if self.config.paint_save_glb:
                glb_path = obj_path[:-4] + ".glb"
                textured_mesh.export(glb_path)
                return_path = glb_path
                self._cleanup_obj_artifacts(obj_path)
        except Exception as e:
            logger.error(f"Mesh export failed: {e}")

        return OutputBatch(output_file_paths=[return_path], metrics=batch.metrics)

    @staticmethod
    def _cleanup_obj_artifacts(obj_path: str) -> None:
        """Remove OBJ file and trimesh-generated material artifacts."""
        obj_dir = os.path.dirname(obj_path) or "."
        targets = [obj_path]
        for f in os.listdir(obj_dir):
            if f.endswith(".mtl") or (f.startswith("material") and f.endswith(".png")):
                targets.append(os.path.join(obj_dir, f))
        for path in targets:
            try:
                os.remove(path)
            except OSError:
                pass

    def verify_input(self, batch: Req, server_args: ServerArgs) -> VerificationResult:
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
