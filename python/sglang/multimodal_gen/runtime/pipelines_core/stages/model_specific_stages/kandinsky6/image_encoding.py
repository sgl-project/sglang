# SPDX-License-Identifier: Apache-2.0
"""Optional image conditioning, adapted from FastVideo's Kandinsky6ImageEncodingStage.

Append a clean reference latent frame and mark its token type. Denoising re-pins
that frame every step and removes it before decoding."""

from __future__ import annotations

import PIL.Image
import torch

from sglang.multimodal_gen.runtime.distributed import get_local_torch_device
from sglang.multimodal_gen.runtime.managers.memory_managers.component_manager import (
    ComponentUse,
)
from sglang.multimodal_gen.runtime.models.vaes.common import ParallelTiledVAE
from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import Req
from sglang.multimodal_gen.runtime.pipelines_core.stages.base import PipelineStage
from sglang.multimodal_gen.runtime.pipelines_core.stages.validators import (
    StageValidators as V,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.validators import (
    VerificationResult,
)
from sglang.multimodal_gen.runtime.server_args import ServerArgs
from sglang.multimodal_gen.runtime.utils.condition_expansion import (
    PromptToSampleBatchExpander,
)
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger
from sglang.multimodal_gen.runtime.utils.precision import (
    autocast_context,
    autocast_enabled,
    resolve_precision,
    temporary_module_dtype,
)
from sglang.multimodal_gen.runtime.utils.vision import (
    normalize,
    numpy_to_pt,
    pil_to_numpy,
)

logger = init_logger(__name__)

# model-specific reference-frame state shared through Req.extra
TAIL_COND_ACTIVE_EXTRA_KEY = "kandinsky6_tail_cond_active"
VISUAL_TOKEN_TYPE_IDS_EXTRA_KEY = "kandinsky6_visual_token_type_ids"


class Kandinsky6ImageEncodingStage(PipelineStage):
    """Optional IT2VA conditioning: no-op for a pure T2VA call.

    See module docstring for the tail-cond scheme.
    """

    def __init__(self, vae: ParallelTiledVAE) -> None:
        super().__init__()
        self.vae = vae

    def component_uses(
        self, server_args: ServerArgs, stage_name: str | None = None
    ) -> list[ComponentUse]:
        vae_dtype = resolve_precision(
            server_args, "vae", precision_attr="vae_precision"
        )
        stage_name = self._component_stage_name(stage_name)
        return [ComponentUse(stage_name, "vae", target_dtype=vae_dtype)]

    @staticmethod
    def _cover_resize_dims(
        src_h: int, src_w: int, height: int, width: int
    ) -> tuple[int, int]:
        """Resize to cover (height, width), preserving aspect before the centre crop."""
        scale = min(src_h / height, src_w / width)
        # clamp float round-trip undershoot on the constraining axis before centre crop
        new_h = max(height, int(src_h / scale))
        new_w = max(width, int(src_w / scale))
        return new_h, new_w

    @classmethod
    def _preprocess(cls, image, height: int, width: int) -> torch.Tensor:
        if isinstance(image, PIL.Image.Image):
            src_w, src_h = image.size
            new_h, new_w = cls._cover_resize_dims(src_h, src_w, height, width)
            image = image.resize((new_w, new_h), resample=PIL.Image.BILINEAR)
            top, left = (new_h - height) // 2, (new_w - width) // 2
            image = image.crop((left, top, left + width, top + height))
            image = numpy_to_pt(pil_to_numpy(image))
            return normalize(image)
        if image.min() >= 0:
            logger.warning(
                "Kandinsky6 conditioning image tensor has no negative values; "
                "assuming range [0, 1] and normalizing to [-1, 1]. Pass a "
                "[-1, 1] tensor with negative values to skip normalization."
            )
            image = normalize(image)
        if image.ndim == 3:
            image = image.unsqueeze(0)
        src_h, src_w = image.shape[-2], image.shape[-1]
        if (src_h, src_w) != (height, width):
            new_h, new_w = cls._cover_resize_dims(src_h, src_w, height, width)
            image = torch.nn.functional.interpolate(
                image.float(), size=(new_h, new_w), mode="bilinear", antialias=True
            )
            top, left = (new_h - height) // 2, (new_w - width) // 2
            image = image[..., top : top + height, left : left + width]
        return image

    @staticmethod
    def _encode_scale_and_shift(
        latents: torch.Tensor, scaling_factor, shift_factor
    ) -> torch.Tensor:
        if shift_factor is not None:
            if isinstance(shift_factor, torch.Tensor):
                shift_factor = shift_factor.to(latents.device, latents.dtype)
            latents = latents - shift_factor
        if isinstance(scaling_factor, torch.Tensor):
            scaling_factor = scaling_factor.to(latents.device, latents.dtype)
        return latents * scaling_factor

    @torch.no_grad()
    def forward(self, batch: Req, server_args: ServerArgs) -> Req:
        if batch.condition_image is None:
            return batch
        if batch.latents is None:
            raise ValueError(
                "Kandinsky6 video latents must be prepared before image conditioning."
            )

        image_source = batch.condition_image
        if isinstance(image_source, list):
            image_source = image_source[0]

        device = get_local_torch_device()
        height = int(batch.height)
        width = int(batch.width)
        vae_dtype = resolve_precision(
            server_args, "vae", precision_attr="vae_precision"
        )
        vae_autocast_enabled = autocast_enabled(vae_dtype, server_args.disable_autocast)

        image = self._preprocess(image_source, height, width)
        image = image.to(device=device, dtype=torch.float32).unsqueeze(2)  # [B,C,1,H,W]

        generator = batch.generator
        if isinstance(generator, list) and len(generator) != image.shape[0]:
            generator = generator[0]

        with self.use_declared_component(component_name="vae", module=self.vae) as vae:
            assert vae is not None
            self.vae = vae
            prev_use_tiling = vae.use_tiling
            vae.use_tiling = False
            try:
                with autocast_context(vae_dtype, server_args.disable_autocast):
                    should_cast_vae = not vae_autocast_enabled
                    if not vae_autocast_enabled:
                        image = image.to(vae_dtype)
                    with temporary_module_dtype(
                        vae, vae_dtype, enabled=should_cast_vae
                    ) as vae_cast:
                        image_latent = vae_cast.encode(image).sample(
                            generator=generator
                        )
            finally:
                vae.use_tiling = prev_use_tiling

            scaling_factor, shift_factor = (
                server_args.pipeline_config.get_decode_scale_and_shift(
                    image_latent.device, image_latent.dtype, vae
                )
            )

        image_latent = self._encode_scale_and_shift(
            image_latent, scaling_factor, shift_factor
        )
        # [B,C,1,H,W] -> [B,1,H,W,C] channel-last, matching the video latent.
        image_latent = image_latent.permute(0, 2, 3, 4, 1).contiguous()
        batch.image_latent = image_latent

        latents = batch.latents
        if batch.image_latent.shape[0] != latents.shape[0]:
            # encode one reference image, then expand it to the request's output batch
            expander = PromptToSampleBatchExpander(
                prompt_batch_size=batch.image_latent.shape[0],
                sample_batch_size=latents.shape[0],
            )
            expander.expand_field(batch, "image_latent")

        image_latent = batch.image_latent.to(device=latents.device, dtype=latents.dtype)
        num_channels = image_latent.shape[-1]

        ref_frame = image_latent
        if latents.shape[-1] > num_channels:
            # [real, cond, mask]: write the reference only into real; cond stays zero
            cond_block = torch.zeros_like(image_latent)
            mask_block = torch.ones_like(image_latent[..., :1])
            ref_frame = torch.cat([image_latent, cond_block, mask_block], dim=-1)

        batch.latents = torch.cat([latents, ref_frame], dim=1)
        batch.extra[TAIL_COND_ACTIVE_EXTRA_KEY] = True

        num_video_frames = latents.shape[1]
        token_type_ids = torch.zeros(
            (latents.shape[0], num_video_frames + 1), dtype=torch.long, device=device
        )
        token_type_ids[:, -1] = 1
        batch.extra[VISUAL_TOKEN_TYPE_IDS_EXTRA_KEY] = token_type_ids

        return batch

    def verify_input(self, batch: Req, server_args: ServerArgs) -> VerificationResult:
        result = VerificationResult()
        result.add_check("latents", batch.latents, [V.is_tensor, V.with_dims(5)])
        return result
