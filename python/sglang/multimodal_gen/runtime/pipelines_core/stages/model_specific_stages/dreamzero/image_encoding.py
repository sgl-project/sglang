# SPDX-License-Identifier: Apache-2.0
"""DreamZero observation preparation and visual encoding stages.

The action endpoint sends normalized observation tensors through msgpack. These
stages materialize tensor inputs, encode the CLIP anchor frame, encode Wan VAE
conditioning latents, and keep visual state in the DreamZero session cache.
"""
from __future__ import annotations

import time
import types
from collections.abc import Mapping
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from sglang.multimodal_gen.runtime.loader.component_loaders.image_encoder_loader import (
    ImageEncoderLoader,
)
from sglang.multimodal_gen.runtime.managers.forward_context import set_forward_context
from sglang.multimodal_gen.runtime.managers.memory_managers.component_manager import (
    ComponentUse,
)
from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import Req
from sglang.multimodal_gen.runtime.pipelines_core.stages.base import PipelineStage
from sglang.multimodal_gen.runtime.pipelines_core.stages.image_encoding import (
    ImageEncodingStage,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.dreamzero.session_cache import (
    DreamZeroCachePool,
    DreamZeroCachePoolManager,
    DreamZeroRequestCache,
    normalize_batched_session_fields,
    record_session_timing,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.validators import (
    VerificationResult,
)
from sglang.multimodal_gen.runtime.server_args import ServerArgs
from sglang.multimodal_gen.utils import PRECISION_TO_TYPE


def _dreamzero_non_causal_clip_attention_forward(
    self,
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor | None = None,
):
    """Run native CLIP attention with a bidirectional vision mask."""
    if attention_mask is None:
        attention_mask = hidden_states.new_ones(hidden_states.shape[:2])
    return self._dreamzero_original_forward(
        hidden_states,
        attention_mask=attention_mask,
    )


def _patch_dreamzero_clip_vision_attention(model: torch.nn.Module) -> None:
    """Patch only DreamZero's CLIP instance instead of changing shared CLIP code."""
    for layer in model.vision_model.encoder.layers:
        attention = layer.self_attn
        attention.attn.attn_impl.causal = False
        attention._dreamzero_original_forward = attention.forward
        attention.forward = types.MethodType(
            _dreamzero_non_causal_clip_attention_forward,
            attention,
        )


def load_dreamzero_image_encoder(
    server_args: ServerArgs,
    component_model_path: str,
) -> torch.nn.Module:
    """Load the image encoder through SGLang's component loader and patch attention."""
    image_encoder = ImageEncoderLoader().load_customized(
        component_model_path,
        server_args,
        "image_encoder",
    )
    _patch_dreamzero_clip_vision_attention(image_encoder)
    return image_encoder


def _module_device(module: torch.nn.Module) -> torch.device:
    return next(module.parameters()).device


def _dit_dtype(server_args: ServerArgs) -> torch.dtype:
    return PRECISION_TO_TYPE[server_args.pipeline_config.dit_precision]


def _as_bcthw(videos: torch.Tensor) -> torch.Tensor:
    if videos.ndim != 5:
        raise ValueError(f"DreamZero images must be 5D, got {tuple(videos.shape)}")
    if videos.shape[-1] in (1, 3):
        videos = videos.permute(0, 4, 1, 2, 3)
    elif videos.shape[2] in (1, 3) and videos.shape[1] != 3:
        videos = videos.permute(0, 2, 1, 3, 4)
    if videos.dtype == torch.uint8:
        videos = videos.float() / 255.0
    return videos


def _normalize_video_range(videos: torch.Tensor) -> torch.Tensor:
    if videos.numel() == 0:
        return videos
    if videos.amin() >= 0 and videos.amax() <= 1:
        return videos.mul(2).sub(1)
    return videos


def _select_image_context(videos_bcthw: torch.Tensor) -> torch.Tensor:
    if videos_bcthw.shape[2] in (4, 9):
        return videos_bcthw[:, :, -1:].transpose(1, 2)
    return videos_bcthw[:, :, :1].transpose(1, 2)


def _clip_pixel_values(image: torch.Tensor, image_size: int) -> torch.Tensor:
    pixel_values = torch.cat(
        [
            F.interpolate(
                frame_group,
                size=(image_size, image_size),
                mode="bicubic",
                align_corners=False,
            )
            for frame_group in image
        ]
    )
    pixel_values = pixel_values.mul(0.5).add(0.5)
    mean = torch.tensor(
        [0.48145466, 0.4578275, 0.40821073],
        device=pixel_values.device,
        dtype=pixel_values.dtype,
    ).view(1, 3, 1, 1)
    std = torch.tensor(
        [0.26862954, 0.26130258, 0.27577711],
        device=pixel_values.device,
        dtype=pixel_values.dtype,
    ).view(1, 3, 1, 1)
    return (pixel_values - mean) / std


def _dreamzero_videos(batch: Req) -> torch.Tensor:
    inputs: dict[str, Any] = batch.dreamzero_inputs
    videos_input = inputs.get("images")
    if videos_input is None:
        videos_input = inputs.get("videos")
    if videos_input is None:
        raise ValueError("DreamZero image encoding requires 'images' or 'videos'")
    return _as_bcthw(videos_input)


class DreamZeroObsPrepStage(PipelineStage):
    """Materialize normalized DreamZero action observations.

    Writes ``batch.dreamzero_inputs`` plus normalized session IDs/reset masks for
    downstream text, visual, and denoising stages.
    """

    def verify_output(self, batch: Req, server_args: ServerArgs) -> VerificationResult:
        result = VerificationResult()
        result.add_check(
            "dreamzero_inputs",
            batch.dreamzero_inputs,
            lambda value: isinstance(value, dict),
        )
        return result

    @staticmethod
    def _to_tensor_tree(value: Any) -> Any:
        """Convert msgpack-decoded numpy leaves to tensors."""
        if isinstance(value, np.ndarray):
            return torch.from_numpy(value)
        if isinstance(value, Mapping):
            return {
                key: DreamZeroObsPrepStage._to_tensor_tree(item)
                for key, item in value.items()
            }
        if isinstance(value, list):
            return [DreamZeroObsPrepStage._to_tensor_tree(item) for item in value]
        if isinstance(value, tuple):
            return tuple(DreamZeroObsPrepStage._to_tensor_tree(item) for item in value)
        return value

    def forward(self, batch: Req, server_args: ServerArgs) -> Req:
        normalized_input = batch.extra.get("dreamzero_normalized_input")
        if normalized_input is None:
            raise ValueError(
                "DreamZero request requires 'dreamzero_normalized_input' in Req.extra"
            )
        if not isinstance(normalized_input, Mapping):
            raise TypeError("DreamZero normalized input must be a mapping")

        model_inputs = self._to_tensor_tree(dict(normalized_input))
        normalize_start = time.perf_counter()
        if not server_args.pipeline_config.disable_autocast:
            for key, value in list(model_inputs.items()):
                if torch.is_tensor(value) and value.dtype == torch.float32:
                    model_inputs[key] = value.to(dtype=torch.bfloat16)

        batch.dreamzero_inputs = model_inputs
        batch_size = next(
            int(value.shape[0])
            for value in (
                model_inputs.get("images"),
                model_inputs.get("videos"),
                model_inputs.get("state"),
            )
            if torch.is_tensor(value) and value.ndim > 0
        )
        session_ids, reset_mask = normalize_batched_session_fields(
            session_ids=batch.extra.get("dreamzero_session_ids"),
            reset_mask=batch.extra.get("dreamzero_reset_mask"),
            batch_size=batch_size,
        )
        batch.dreamzero_session_ids = session_ids
        batch.dreamzero_reset_mask = reset_mask
        record_session_timing(
            batch,
            "session_normalize_ms",
            (time.perf_counter() - normalize_start) * 1000,
        )
        return batch


class DreamZeroVisualEncodingStage(ImageEncodingStage):
    """Encode DreamZero visual conditioning and maintain visual session state.

    Inherits SGLang ``ImageEncodingStage`` for declared component placement, then
    writes ``dreamzero_clip_feature``, ``dreamzero_y``, and
    ``dreamzero_latent_video``.
    """

    deduplicated_output_fields = ()

    def __init__(
        self,
        image_encoder: torch.nn.Module | None = None,
        vae: torch.nn.Module | None = None,
        cache_manager: DreamZeroCachePoolManager | None = None,
    ) -> None:
        super().__init__(image_processor=None, image_encoder=image_encoder)
        self.vae = vae
        self.cache_manager = cache_manager

    def component_uses(
        self, server_args: ServerArgs, stage_name: str | None = None
    ) -> list[ComponentUse]:
        stage_name = self._component_stage_name(stage_name)
        return [
            ComponentUse(
                stage_name,
                "image_encoder",
                target_dtype=PRECISION_TO_TYPE[
                    server_args.pipeline_config.image_encoder_precision
                ],
            ),
            ComponentUse(
                stage_name,
                "vae",
                target_dtype=PRECISION_TO_TYPE[
                    server_args.pipeline_config.vae_precision
                ],
            ),
        ]

    def verify_input(self, batch: Req, server_args: ServerArgs) -> VerificationResult:
        result = VerificationResult()
        result.add_check(
            "dreamzero_inputs",
            getattr(batch, "dreamzero_inputs", None),
            lambda value: isinstance(value, dict),
        )
        return result

    def verify_output(self, batch: Req, server_args: ServerArgs) -> VerificationResult:
        result = VerificationResult()
        result.add_check(
            "dreamzero_clip_feature",
            getattr(batch, "dreamzero_clip_feature", None),
            torch.is_tensor,
        )
        result.add_check(
            "dreamzero_y",
            getattr(batch, "dreamzero_y", None),
            torch.is_tensor,
        )
        result.add_check(
            "dreamzero_latent_video",
            getattr(batch, "dreamzero_latent_video", None),
            torch.is_tensor,
        )
        return result

    @staticmethod
    def _normalize_sglang_wan_latent(vae: torch.nn.Module, posterior) -> torch.Tensor:
        """Normalize Wan VAE posterior outputs to the latent scale expected by DiT."""
        if torch.is_tensor(posterior):
            return posterior
        mean_tensor = posterior.mean
        mean = torch.tensor(
            vae.latents_mean,
            device=mean_tensor.device,
            dtype=mean_tensor.dtype,
        ).view(1, mean_tensor.shape[1], 1, 1, 1)
        std = torch.tensor(
            vae.latents_std,
            device=mean_tensor.device,
            dtype=mean_tensor.dtype,
        ).view(1, mean_tensor.shape[1], 1, 1, 1)
        return (mean_tensor - mean) / std

    @staticmethod
    def _write_vae_outputs(batch: Req, server_args: ServerArgs, y: torch.Tensor) -> Req:
        """Write VAE conditioning tensor and the current-frame latent to ``batch``."""
        dit_arch = server_args.pipeline_config.dit_config.arch_config
        vae_arch = server_args.pipeline_config.vae_config.arch_config
        latent_channels = int(vae_arch.z_dim)
        in_dim = int(dit_arch.in_dim)
        batch_size = y.shape[0]
        num_t = y.shape[2]
        h_latent, w_latent = y.shape[3], y.shape[4]
        if y.shape[1] != latent_channels:
            raise ValueError(
                "DreamZero VAE latent channel mismatch: "
                f"expected {latent_channels}, got {y.shape[1]}"
            )
        mask = torch.zeros(
            batch_size,
            4,
            num_t,
            h_latent,
            w_latent,
            dtype=y.dtype,
            device=y.device,
        )
        mask[:, :, 0:1] = 1
        conditioning_y = torch.cat([mask, y], dim=1)
        batch.dreamzero_latent_video = y[:, :, 0:1]
        if not dit_arch.concat_first_frame_latent:
            if in_dim != latent_channels:
                raise ValueError(
                    "DreamZero TI2V in_dim must match VAE latent channels when "
                    "concat_first_frame_latent=False: "
                    f"in_dim={in_dim}, latent_channels={latent_channels}"
                )
            # The original image path always returns ys=[mask, latent]. TI2V
            # does not concatenate ys into the DiT video input, but tests and
            # session state still compare this reference-shaped tensor.
            batch.dreamzero_y = conditioning_y
            return batch

        expected_in_dim = latent_channels * 2 + 4
        if in_dim != expected_in_dim:
            raise ValueError(
                "DreamZero I2V in_dim mismatch for [x; mask; y] concat: "
                f"expected {expected_in_dim}, got {in_dim}"
            )
        batch.dreamzero_y = conditioning_y
        return batch

    def _videos_for_visual_context(
        self,
        batch: Req,
        server_args: ServerArgs,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Return normalized ``[B, C, T, H, W]`` video for CLIP/VAE encoding."""
        videos = _normalize_video_range(
            _dreamzero_videos(batch).to(device=device, dtype=dtype)
        )
        target_h = int(server_args.pipeline_config.synthetic_height or 0)
        target_w = int(server_args.pipeline_config.synthetic_width or 0)
        if (
            target_h > 0
            and target_w > 0
            and tuple(videos.shape[-2:])
            != (
                target_h,
                target_w,
            )
        ):
            batch_size, channels, num_frames, height, width = videos.shape
            videos = torch.nn.functional.interpolate(
                videos.reshape(batch_size * num_frames, channels, height, width),
                size=(target_h, target_w),
                mode="bilinear",
                align_corners=False,
            ).reshape(batch_size, channels, num_frames, target_h, target_w)
        return videos

    def _encode_clip_feature(
        self,
        batch: Req,
        server_args: ServerArgs,
        *,
        image: torch.Tensor | None,
    ) -> torch.Tensor:
        """Encode the anchor image with the declared DreamZero image encoder."""
        with self.use_declared_component(
            component_name="image_encoder", module=self.image_encoder
        ) as image_encoder:
            if image_encoder is None:
                raise ValueError("DreamZero image encoder module is not loaded")
            self.image_encoder = image_encoder
            dtype = _dit_dtype(server_args)
            device = _module_device(image_encoder)
            if image is None or image.device != device or image.dtype != dtype:
                videos = self._videos_for_visual_context(
                    batch,
                    server_args,
                    device=device,
                    dtype=dtype,
                )
                image = _select_image_context(videos)
            with torch.amp.autocast(
                dtype=torch.bfloat16,
                device_type=device.type,
                enabled=device.type == "cuda",
            ):
                with set_forward_context(current_timestep=0, attn_metadata=None):
                    image_size = image_encoder.config.image_size
                    pixel_values = _clip_pixel_values(image, image_size)
                    pixel_values = pixel_values.to(
                        dtype=next(image_encoder.parameters()).dtype
                    )
                    return (
                        image_encoder(pixel_values=pixel_values)
                        .last_hidden_state.clone()
                        .to(dtype=dtype)
                    )

    def _encode_vae_context(
        self,
        batch: Req,
        server_args: ServerArgs,
        *,
        image: torch.Tensor | None,
        videos: torch.Tensor | None,
    ) -> Req:
        """Encode first-frame conditioning latents used by the causal DiT rollout."""
        with self.use_declared_component(component_name="vae", module=self.vae) as vae:
            if vae is None:
                raise ValueError("DreamZero VAE module is not loaded")
            self.vae = vae
            dtype = _dit_dtype(server_args)
            device = _module_device(vae)
            if videos is None or videos.device != device or videos.dtype != dtype:
                videos = self._videos_for_visual_context(
                    batch,
                    server_args,
                    device=device,
                    dtype=dtype,
                )
            if image is None or image.device != device or image.dtype != dtype:
                image = _select_image_context(videos)
            image_input = image.transpose(1, 2).contiguous()
            batch_size = image_input.shape[0]
            num_frames = server_args.pipeline_config.num_frames
            height, width = videos.shape[-2:]
            image_zeros = torch.zeros(
                batch_size,
                3,
                num_frames - 1,
                height,
                width,
                dtype=dtype,
                device=device,
            )
            vae_input = torch.cat([image_input, image_zeros], dim=2)
            with torch.amp.autocast(
                dtype=torch.bfloat16,
                device_type=device.type,
                enabled=device.type == "cuda",
            ):
                posterior = vae.encode(vae_input)
                y = self._normalize_sglang_wan_latent(vae, posterior).to(dtype=dtype)
            return self._write_vae_outputs(batch, server_args, y)

    def _encode_current_video(
        self,
        batch: Req,
        server_args: ServerArgs,
        *,
        videos: torch.Tensor | None,
    ) -> torch.Tensor:
        """Encode the streaming video block appended after the first anchor frame."""
        with self.use_declared_component(component_name="vae", module=self.vae) as vae:
            if vae is None:
                raise ValueError("DreamZero VAE module is not loaded")
            self.vae = vae
            dtype = _dit_dtype(server_args)
            device = _module_device(vae)
            if videos is None or videos.device != device or videos.dtype != dtype:
                videos = self._videos_for_visual_context(
                    batch,
                    server_args,
                    device=device,
                    dtype=dtype,
                )

            num_frame_per_block = int(
                server_args.pipeline_config.dit_config.arch_config.num_frame_per_block
            )
            num_frames = int(videos.shape[2])
            if (num_frames - 1) // 4 != num_frame_per_block:
                latent_blocks = num_frames // 4
                if latent_blocks == num_frame_per_block:
                    videos = torch.cat([videos[:, :, :1], videos], dim=2)
                elif latent_blocks > 0:
                    repeat_factor = num_frame_per_block // latent_blocks
                    if repeat_factor < 1:
                        raise ValueError(
                            "DreamZero observation contains more VAE blocks than "
                            "num_frame_per_block"
                        )
                    videos = torch.repeat_interleave(videos, repeat_factor, dim=2)
                    videos = torch.cat([videos[:, :, :1], videos], dim=2)
                else:
                    raise ValueError(
                        "DreamZero streaming VAE input must contain at least four "
                        "frames; single-frame inputs start a new anchor window"
                    )

            with torch.amp.autocast(
                dtype=torch.bfloat16,
                device_type=device.type,
                enabled=device.type == "cuda",
            ):
                posterior = vae.encode(videos)
                return self._normalize_sglang_wan_latent(vae, posterior).to(dtype=dtype)

    def forward(self, batch: Req, server_args: ServerArgs) -> Req:
        request_cache: DreamZeroRequestCache = batch.dreamzero_cache
        return self._forward_cache_manager(batch, server_args, request_cache)

    def _forward_cache_manager(
        self,
        batch: Req,
        server_args: ServerArgs,
        request_cache: DreamZeroRequestCache,
    ):
        """Reuse first-frame visual conditioning and update current video latents."""
        if self.cache_manager is None:
            raise RuntimeError("DreamZero visual stage requires a cache manager")
        state: DreamZeroCachePool = self.cache_manager.pool
        slots = request_cache.slot_indices
        dtype = _dit_dtype(server_args)
        device: torch.device | None = None
        if self.image_encoder is not None:
            device = _module_device(self.image_encoder)
        elif self.vae is not None:
            device = _module_device(self.vae)

        videos = None
        image = None
        if device is not None:
            videos = self._videos_for_visual_context(
                batch,
                server_args,
                device=device,
                dtype=dtype,
            )
            image = _select_image_context(videos)

        current_start_frame = request_cache.uniform_current_start_frame(
            self.cache_manager
        )
        if current_start_frame == 0:
            batch.dreamzero_clip_feature = self._encode_clip_feature(
                batch,
                server_args,
                image=image,
            )
            batch = self._encode_vae_context(
                batch,
                server_args,
                image=image,
                videos=videos,
            )
            state.scatter_visual(
                slots,
                clip_feas=batch.dreamzero_clip_feature,
                ys=batch.dreamzero_y,
                latent_video=batch.dreamzero_latent_video,
            )
            return batch

        clip_feas, ys, _ = state.gather_visual(slots)
        latent_video = self._encode_current_video(
            batch,
            server_args,
            videos=videos,
        )
        state.scatter_visual(slots, latent_video=latent_video)
        batch.dreamzero_clip_feature = clip_feas
        batch.dreamzero_y = ys
        batch.dreamzero_latent_video = latent_video
        return batch
