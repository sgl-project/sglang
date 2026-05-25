# SPDX-License-Identifier: Apache-2.0
#
# SANA-WM BeforeDenoisingStage.
#
# Consolidates all model-specific pre-processing for SANA-WM TI2V inference:
#   1. Adjust num_frames to satisfy (F-1) % temporal_stride == 0
#   2. Initialize random noise latents (5D: B, 128, T_latent, H_sp, W_sp)
#   3. VAE-encode the first-frame conditioning image and splice into noisy latents
#      (replaces latent[:, :, 0] with the encoded first-frame latent)
#   4. Build the (B, F_orig, 20) ``camera_conditions`` tensor consumed by the
#      UCPE camera branch (16 c2w flat + 4 intrinsics ``fx,fy,cx,cy``).
#   5. Compute the 48-channel packed Plücker raymap consumed by the
#      ``plucker_embedder`` (one chunk = vae_temporal_stride original frames).
#   6. Prepare FlowMatch timesteps and sigmas (uses flow_shift=9.95).
#
# Text encoding is handled upstream by the standard TextEncodingStage (Gemma-2).
# After this stage, the Req batch contains all fields needed by DenoisingStage.

from typing import Optional

import numpy as np
import torch
from diffusers.utils.torch_utils import randn_tensor

from sglang.multimodal_gen.configs.pipeline_configs.sana_wm import SanaWMPipelineConfig
from sglang.multimodal_gen.runtime.distributed import get_local_torch_device
from sglang.multimodal_gen.runtime.pipelines_core.diffusion_scheduler_utils import (
    get_or_create_request_scheduler,
)
from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import Req
from sglang.multimodal_gen.runtime.pipelines_core.stages.base import PipelineStage
from sglang.multimodal_gen.runtime.server_args import ServerArgs
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

logger = init_logger(__name__)


class SanaWMBeforeDenoisingStage(PipelineStage):
    """
    Monolithic pre-processing stage for SANA-WM TI2V inference.

    Must run AFTER TextEncodingStage (which populates batch.prompt_embeds).
    """

    def __init__(
        self,
        vae,
        transformer,
        scheduler,
        pipeline_config: SanaWMPipelineConfig,
    ):
        super().__init__()
        self.vae = vae
        self.transformer = transformer
        self.scheduler = scheduler
        self.pipeline_config = pipeline_config

    # -----------------------------------------------------------------------
    # Helper: VAE-encode and scale an image tensor
    # -----------------------------------------------------------------------

    @torch.no_grad()
    def _vae_encode_image(
        self,
        image: torch.Tensor,     # (1, C, H, W) or (1, C, 1, H, W) in [0, 1] float
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor:
        """Encode a single image frame through the VAE encoder."""
        vae = self.vae
        vae_dtype_map = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}
        vae_dtype = vae_dtype_map.get(self.pipeline_config.vae_precision, torch.bfloat16)

        # Normalize image to [-1, 1] range expected by the VAE
        if image.max() > 1.01:
            image = image / 255.0
        image = (image * 2.0 - 1.0).to(device=device, dtype=vae_dtype)

        # Add temporal dim if not present: (B, C, H, W) → (B, C, 1, H, W)
        if image.dim() == 4:
            image = image.unsqueeze(2)

        z = vae.encode(image).mean.float()

        # Apply shift and scaling factors if present
        if hasattr(vae, "shift_factor") and vae.shift_factor is not None:
            sf = vae.shift_factor
            z = z - (sf.to(z.device, z.dtype) if isinstance(sf, torch.Tensor) else sf)

        scale = vae.scaling_factor if hasattr(vae, "scaling_factor") else 1.0
        if isinstance(scale, torch.Tensor):
            z = z * scale.to(z.device, z.dtype)
        else:
            z = z * scale

        return z.to(dtype=dtype)  # (1, 128, 1, H_sp, W_sp)

    # -----------------------------------------------------------------------
    # Helper: initialize noise latents
    # -----------------------------------------------------------------------

    def _prepare_noise_latents(
        self,
        shape: tuple,
        dtype: torch.dtype,
        device: torch.device,
        generator: torch.Generator,
    ) -> torch.Tensor:
        return randn_tensor(shape, generator=generator, device=device, dtype=dtype)

    # -----------------------------------------------------------------------
    # Helper: splice first-frame image latent into noise latents
    # -----------------------------------------------------------------------

    @torch.no_grad()
    def _splice_first_frame(
        self,
        latents: torch.Tensor,  # (B, 128, T_lat, H_sp, W_sp)
        condition_image,        # PIL Image or torch.Tensor
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor:
        """Replace latents[:, :, 0] with VAE-encoded first frame."""
        import PIL.Image

        # Convert PIL to tensor if needed
        if isinstance(condition_image, PIL.Image.Image):
            import torchvision.transforms.functional as TF
            img_tensor = TF.to_tensor(condition_image.convert("RGB")).unsqueeze(0)
        elif isinstance(condition_image, torch.Tensor):
            img_tensor = condition_image.float()
            if img_tensor.dim() == 3:
                img_tensor = img_tensor.unsqueeze(0)
        else:
            logger.warning("condition_image type unsupported; skipping first-frame splice.")
            return latents

        # Resize if needed to match latent spatial dims
        B, C, T_lat, H_sp, W_sp = latents.shape
        target_h = H_sp * self.pipeline_config.vae_stride[1]  # 32
        target_w = W_sp * self.pipeline_config.vae_stride[2]  # 32
        if img_tensor.shape[-2] != target_h or img_tensor.shape[-1] != target_w:
            import torch.nn.functional as F
            img_tensor = F.interpolate(
                img_tensor,
                size=(target_h, target_w),
                mode="bilinear",
                align_corners=False,
            )

        first_frame_z = self._vae_encode_image(img_tensor, dtype, device)
        # first_frame_z: (1, 128, 1, H_sp, W_sp) — expand to batch
        first_frame_z = first_frame_z.expand(B, -1, -1, -1, -1)

        # Splice: replace the first temporal latent frame
        latents = latents.clone()
        latents[:, :, 0:1] = first_frame_z
        return latents

    # -----------------------------------------------------------------------
    # Helper: compute timesteps and sigmas for FlowMatch scheduling
    # -----------------------------------------------------------------------

    def _prepare_timesteps(
        self,
        batch: Req,
        server_args: ServerArgs,
        device: torch.device,
    ):
        """Set up scheduler timesteps and populate batch.timesteps, .sigmas."""
        scheduler = get_or_create_request_scheduler(batch, self.scheduler)
        num_inference_steps = batch.num_inference_steps

        # Use flow_shift from pipeline config
        flow_shift = getattr(self.pipeline_config, "flow_shift", 9.95)
        kwargs = {}

        # diffusers FlowMatchEulerDiscreteScheduler supports mu/shift
        import inspect
        sig_params = inspect.signature(scheduler.set_timesteps).parameters
        if "shift" in sig_params:
            kwargs["shift"] = flow_shift
        elif "mu" in sig_params:
            # Convert flow_shift to mu: mu ≈ log(shift)
            import math
            kwargs["mu"] = math.log(flow_shift)

        scheduler.set_timesteps(num_inference_steps, device=device, **kwargs)
        timesteps = scheduler.timesteps
        sigmas = scheduler.sigmas.tolist()

        batch.timesteps = timesteps
        batch.sigmas = sigmas
        batch.scheduler = scheduler
        return batch

    # -----------------------------------------------------------------------
    # Main forward
    # -----------------------------------------------------------------------

    @torch.no_grad()
    def forward(self, batch: Req, server_args: ServerArgs) -> Req:
        """
        Pre-process everything needed by DenoisingStage for SANA-WM.

        Expects batch to already have prompt_embeds set by TextEncodingStage.
        """
        device = get_local_torch_device()
        dtype = torch.bfloat16  # SANA-WM runs in bf16

        # --- 0. Adjust num_frames to be compatible with VAE temporal stride ---
        num_frames = batch.num_frames or 49
        num_frames = self.pipeline_config.adjust_num_frames(num_frames)
        batch.num_frames = num_frames

        # --- 1. Generator for reproducibility ---
        seed = batch.seed if hasattr(batch, "seed") and batch.seed is not None else 0
        generator = torch.Generator(device=device).manual_seed(seed)
        batch.generator = generator

        # --- 2. Compute latent shape and initialize noise ---
        batch_size = batch.batch_size or 1
        latent_shape = self.pipeline_config.prepare_latent_shape(batch, batch_size, num_frames)
        # latent_shape: (B, 128, T_latent, H_sp, W_sp)
        latents = self._prepare_noise_latents(latent_shape, dtype, device, generator)

        # Store raw shape for DecodingStage
        batch.raw_latent_shape = latent_shape

        # --- 3. VAE-encode first frame and splice into noise latents ---
        condition_image = getattr(batch, "condition_image", None)
        if condition_image is not None:
            try:
                latents = self._splice_first_frame(latents, condition_image, dtype, device)
                self.log_info("First-frame spliced into noise latents.")
            except Exception as e:
                logger.warning(f"First-frame splice failed: {e}. Using pure noise latents.")
        else:
            self.log_info("No condition_image provided; using pure noise latents (T2V mode).")

        batch.latents = latents

        # --- 4. Camera conditioning ---
        # The DiT consumes two tensors:
        #   * ``camera_conditions``: (B, F_orig, 20) -- 16 c2w flat + (fx, fy, cx, cy).
        #     This drives the UCPE camera-branch attention.
        #   * ``chunk_plucker``:      (B, 48, T_lat, H_sp, W_sp) -- packed Plücker
        #     coords for the post-attention plucker_embedder.
        #
        # Inputs from the user (via batch.extra) may be:
        #   - camera_conditions: pre-built (B, F_orig, 20) tensor
        #   - OR camera_to_world (B, F_orig, 4, 4) + intrinsics (B, F_orig, 3, 3)
        if not hasattr(batch, "extra") or batch.extra is None:
            batch.extra = {}

        camera_conditions = batch.extra.get("camera_conditions", None)
        camera_to_world = batch.extra.get("camera_to_world", None)
        intrinsics = batch.extra.get("intrinsics", None)

        # Build ``camera_conditions`` from c2w + intrinsics if not already provided.
        if (
            camera_conditions is None
            and camera_to_world is not None
            and intrinsics is not None
        ):
            try:
                if not isinstance(camera_to_world, torch.Tensor):
                    camera_to_world = torch.as_tensor(camera_to_world)
                if not isinstance(intrinsics, torch.Tensor):
                    intrinsics = torch.as_tensor(intrinsics)
                if camera_to_world.dim() == 3:        # (F, 4, 4) -> (1, F, 4, 4)
                    camera_to_world = camera_to_world.unsqueeze(0)
                if intrinsics.dim() == 2:             # (3, 3) -> (1, 1, 3, 3)
                    intrinsics = intrinsics.unsqueeze(0).unsqueeze(0)
                elif intrinsics.dim() == 3:           # (F, 3, 3) or (B, 3, 3)?
                    # if matches F, treat as per-frame; else broadcast
                    F_dim = camera_to_world.shape[1]
                    if intrinsics.shape[0] == F_dim:
                        intrinsics = intrinsics.unsqueeze(0)
                    else:
                        intrinsics = intrinsics.unsqueeze(1)
                # Broadcast intrinsics across frames if single per batch
                if intrinsics.shape[1] == 1 and camera_to_world.shape[1] > 1:
                    intrinsics = intrinsics.expand(-1, camera_to_world.shape[1], -1, -1)

                camera_to_world = camera_to_world.to(device=device, dtype=dtype)
                intrinsics = intrinsics.to(device=device, dtype=dtype)

                B = camera_to_world.shape[0]
                F_dim = camera_to_world.shape[1]
                c2w_flat = camera_to_world.reshape(B, F_dim, 16)
                fx = intrinsics[..., 0, 0]
                fy = intrinsics[..., 1, 1]
                cx = intrinsics[..., 0, 2]
                cy = intrinsics[..., 1, 2]
                camera_conditions = torch.cat([
                    c2w_flat,
                    torch.stack([fx, fy, cx, cy], dim=-1),
                ], dim=-1)
            except Exception as e:
                logger.warning(f"Failed to build camera_conditions from c2w+intrinsics: {e}.")
                camera_conditions = None

        # Optionally compute the chunk-packed Plücker raymap. This is what
        # the plucker_embedder consumes -- shape (B, 48, T_lat, H_sp, W_sp).
        chunk_plucker = batch.extra.get("chunk_plucker", None)
        if (
            chunk_plucker is None
            and camera_conditions is not None
            and self.pipeline_config.camera_conditioning
        ):
            try:
                from sglang.multimodal_gen.runtime.models.dits.sana_wm import (
                    compute_chunk_plucker,
                )
                T_lat = latent_shape[2]
                sp_h = latent_shape[3]
                sp_w = latent_shape[4]
                vae_temporal_stride = self.pipeline_config.vae_stride[0]
                F_orig = camera_conditions.shape[1]
                if F_orig != T_lat * vae_temporal_stride:
                    logger.warning(
                        f"camera_conditions has {F_orig} frames but expected "
                        f"T_lat*stride = {T_lat}*{vae_temporal_stride} = "
                        f"{T_lat * vae_temporal_stride}. Disabling Plücker."
                    )
                    chunk_plucker = None
                else:
                    chunk_plucker = compute_chunk_plucker(
                        camera_conditions=camera_conditions.to(dtype),
                        HW=(T_lat, sp_h, sp_w),
                        vae_temporal_stride=vae_temporal_stride,
                        patch_size=(1, 1, 1),
                    )
                    self.log_info("Plücker raymaps computed: shape %s", str(chunk_plucker.shape))
            except Exception as e:
                logger.warning(f"chunk_plucker computation failed: {e}. Disabling Plücker mixing.")
                chunk_plucker = None

        if camera_conditions is not None:
            batch.extra["camera_conditions"] = camera_conditions.to(device=device, dtype=dtype)
        if chunk_plucker is not None:
            batch.extra["chunk_plucker"] = chunk_plucker.to(device=device, dtype=dtype)

        # --- 5. Prepare timesteps and sigmas ---
        batch = self._prepare_timesteps(batch, server_args, device)

        # --- 6. Ensure prompt_embeds is a list (DenoisingStage expects list[Tensor]) ---
        if isinstance(batch.prompt_embeds, torch.Tensor):
            batch.prompt_embeds = [batch.prompt_embeds]
        if (
            batch.negative_prompt_embeds is not None
            and isinstance(batch.negative_prompt_embeds, torch.Tensor)
        ):
            batch.negative_prompt_embeds = [batch.negative_prompt_embeds]

        # --- 7. CFG setup ---
        batch.do_classifier_free_guidance = (
            getattr(batch, "guidance_scale", 1.0) > 1.0
        )

        self.log_info(
            "BeforeDenoisingStage done: latent=%s, T_lat=%d, H_sp=%d, W_sp=%d, "
            "num_inference_steps=%d, camera=%s",
            str(latent_shape),
            latent_shape[2],
            latent_shape[3],
            latent_shape[4],
            batch.num_inference_steps,
            "yes" if batch.extra.get("camera_conditions") is not None else "no",
        )
        return batch
