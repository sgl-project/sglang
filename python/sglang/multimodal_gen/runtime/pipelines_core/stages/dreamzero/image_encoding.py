# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from typing import Any

import torch

from sglang.multimodal_gen.runtime.managers.dreamzero_session_cache import (
    DreamZeroCachePool,
    DreamZeroCachePoolManager,
    DreamZeroRequestCache,
    enter_request_cache,
)
from sglang.multimodal_gen.runtime.managers.forward_context import set_forward_context
from sglang.multimodal_gen.runtime.managers.memory_managers.component_manager import (
    ComponentUse,
)
from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import Req
from sglang.multimodal_gen.runtime.pipelines_core.stages.base import PipelineStage
from sglang.multimodal_gen.runtime.pipelines_core.stages.validators import (
    VerificationResult,
)
from sglang.multimodal_gen.runtime.platforms import current_platform
from sglang.multimodal_gen.runtime.server_args import ServerArgs
from sglang.multimodal_gen.utils import PRECISION_TO_TYPE


def _module_device(module: torch.nn.Module) -> torch.device:
    try:
        return next(module.parameters()).device
    except StopIteration:
        return torch.device(current_platform.device_type)


def _dit_dtype(server_args: ServerArgs) -> torch.dtype:
    return PRECISION_TO_TYPE[server_args.pipeline_config.dit_precision]


def _as_bcthw(videos: torch.Tensor) -> torch.Tensor:
    if videos.ndim != 5:
        raise ValueError(f"DreamZero images must be 5D, got {tuple(videos.shape)}")
    if videos.shape[-1] in (1, 3):
        videos = videos.permute(0, 4, 1, 2, 3)
    elif videos.shape[2] in (1, 3) and videos.shape[1] != 3:
        videos = videos.permute(0, 2, 1, 3, 4)
    elif videos.shape[1] in (1, 3):
        pass
    if videos.dtype == torch.uint8:
        videos = videos.float() / 255.0
    return videos


def _normalize_video_range(videos: torch.Tensor) -> torch.Tensor:
    if videos.numel() == 0:
        return videos
    if videos.amin() >= 0 and videos.amax() <= 1:
        return videos.mul(2).sub(1)
    return videos


def _select_image_context(
    videos_bcthw: torch.Tensor,
    *,
    first_frame: bool = False,
) -> torch.Tensor:
    if videos_bcthw.shape[2] in (4, 9):
        return videos_bcthw[:, :, -1:].transpose(1, 2)
    if first_frame:
        return videos_bcthw[:, :, :1].transpose(1, 2)
    return videos_bcthw[:, :, :1].transpose(1, 2)


def _dreamzero_videos(batch: Req) -> torch.Tensor:
    inputs: dict[str, Any] = batch.dreamzero_inputs
    videos_input = inputs.get("images")
    if videos_input is None:
        videos_input = inputs.get("videos")
    if videos_input is None:
        raise ValueError("DreamZero image encoding requires 'images' or 'videos'")
    return _as_bcthw(videos_input)


class DreamZeroVisualEncodingStage(PipelineStage):
    """Encode DreamZero visual context into CLIP and VAE conditioning tensors."""

    def __init__(
        self,
        image_encoder: torch.nn.Module | None = None,
        vae: torch.nn.Module | None = None,
        cache_manager: DreamZeroCachePoolManager | None = None,
    ) -> None:
        super().__init__()
        self.image_encoder = image_encoder
        self.vae = vae
        self.cache_manager = cache_manager

    @property
    def role_affinity(self):
        from sglang.multimodal_gen.runtime.disaggregation.roles import RoleType

        return RoleType.ENCODER

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
        if torch.is_tensor(posterior):
            return posterior
        mean_tensor = posterior.mean
        latents_mean = getattr(vae, "latents_mean", None)
        latents_std = getattr(vae, "latents_std", None)
        if latents_mean is None or latents_std is None:
            return mean_tensor
        mean = torch.tensor(
            latents_mean,
            device=mean_tensor.device,
            dtype=mean_tensor.dtype,
        ).view(1, mean_tensor.shape[1], 1, 1, 1)
        std = torch.tensor(
            latents_std,
            device=mean_tensor.device,
            dtype=mean_tensor.dtype,
        ).view(1, mean_tensor.shape[1], 1, 1, 1)
        return (mean_tensor - mean) / std

    @staticmethod
    def _write_vae_outputs(batch: Req, server_args: ServerArgs, y: torch.Tensor) -> Req:
        dit_arch = server_args.pipeline_config.dit_config.arch_config
        vae_arch = server_args.pipeline_config.vae_config.arch_config
        latent_channels = int(getattr(vae_arch, "z_dim", y.shape[1]))
        in_dim = int(getattr(dit_arch, "in_dim", y.shape[1]))
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
        if not bool(getattr(dit_arch, "concat_first_frame_latent", True)):
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
        videos = _normalize_video_range(
            _dreamzero_videos(batch).to(device=device, dtype=dtype)
        )
        target_h = int(getattr(server_args.pipeline_config, "synthetic_height", 0) or 0)
        target_w = int(getattr(server_args.pipeline_config, "synthetic_width", 0) or 0)
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
        inputs: dict[str, Any] = batch.dreamzero_inputs
        precomputed = inputs.get("clip_feature")
        if precomputed is not None:
            device = (
                _module_device(self.image_encoder)
                if self.image_encoder is not None
                else precomputed.device
            )
            return precomputed.to(device=device, dtype=_dit_dtype(server_args))

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
                arch = server_args.pipeline_config.dit_config.arch_config
                image = _select_image_context(
                    videos,
                    first_frame=not bool(
                        getattr(arch, "concat_first_frame_latent", True)
                    ),
                )
            batch.dreamzero_image_context_input = image
            with torch.amp.autocast(
                dtype=torch.bfloat16,
                device_type=device.type,
                enabled=device.type == "cuda",
            ):
                with set_forward_context(current_timestep=0, attn_metadata=None):
                    return image_encoder.encode_image(image).to(dtype=dtype)

    def _encode_vae_context(
        self,
        batch: Req,
        server_args: ServerArgs,
        *,
        image: torch.Tensor | None,
        videos: torch.Tensor | None,
    ) -> Req:
        inputs: dict[str, Any] = batch.dreamzero_inputs
        if all(key in inputs for key in ("y", "latent_video")):
            y = inputs["y"]
            latent_video = inputs["latent_video"]
            device = _module_device(self.vae) if self.vae is not None else y.device
            dtype = _dit_dtype(server_args)
            y = y.to(device=device, dtype=dtype)
            batch.dreamzero_latent_video = latent_video.to(device=device, dtype=dtype)
            latent_channels = int(
                getattr(
                    server_args.pipeline_config.vae_config.arch_config,
                    "z_dim",
                    y.shape[1],
                )
            )
            if y.shape[1] == latent_channels:
                batch = self._write_vae_outputs(batch, server_args, y)
                batch.dreamzero_latent_video = latent_video.to(
                    device=device,
                    dtype=dtype,
                )
                return batch
            if y.shape[1] != latent_channels + 4:
                raise ValueError(
                    "DreamZero precomputed y must be either VAE latent channels "
                    "or DreamZero ys=[mask, latent]: "
                    f"got {y.shape[1]}, latent_channels={latent_channels}"
                )
            batch.dreamzero_y = y
            return batch

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
                arch = server_args.pipeline_config.dit_config.arch_config
                image = _select_image_context(
                    videos,
                    first_frame=not bool(
                        getattr(arch, "concat_first_frame_latent", True)
                    ),
                )
            batch.dreamzero_image_context_input = image
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
        inputs: dict[str, Any] = batch.dreamzero_inputs
        precomputed = inputs.get("latent_video")
        if precomputed is not None:
            device = (
                _module_device(self.vae) if self.vae is not None else precomputed.device
            )
            return precomputed.to(device=device, dtype=_dit_dtype(server_args))

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

    @staticmethod
    def _infer_batch_size(inputs: dict[str, Any]) -> int:
        for value in inputs.values():
            if torch.is_tensor(value):
                return int(value.shape[0])
        raise ValueError("DreamZero visual stage cannot infer batch size")

    def forward(self, batch: Req, server_args: ServerArgs) -> Req:
        arch = server_args.pipeline_config.dit_config.arch_config
        max_chunk_size = int(getattr(arch, "max_chunk_size", -1))
        local_attn_size = (
            -1
            if max_chunk_size == -1
            else max_chunk_size * int(arch.num_frame_per_block) + 1
        )
        inputs: dict[str, Any] = batch.dreamzero_inputs
        request_cache, _ = enter_request_cache(
            batch,
            self.cache_manager,
            local_attn_size=local_attn_size,
            batch_size=self._infer_batch_size(inputs),
        )
        return self._forward_cache_manager(batch, server_args, request_cache)

    def _forward_cache_manager(
        self,
        batch: Req,
        server_args: ServerArgs,
        request_cache: DreamZeroRequestCache,
    ):
        state: DreamZeroCachePool = request_cache.pool(self.cache_manager)
        slots = request_cache.slot_indices
        dtype = _dit_dtype(server_args)
        device: torch.device | None = None
        if self.image_encoder is not None:
            device = _module_device(self.image_encoder)
        elif self.vae is not None:
            device = _module_device(self.vae)

        videos = None
        image = None
        inputs: dict[str, Any] = batch.dreamzero_inputs
        arch = server_args.pipeline_config.dit_config.arch_config
        needs_image = "clip_feature" not in inputs or not all(
            key in inputs for key in ("y", "latent_video")
        )
        if needs_image and device is not None:
            videos = self._videos_for_visual_context(
                batch,
                server_args,
                device=device,
                dtype=dtype,
            )
            image = _select_image_context(
                videos,
                first_frame=not bool(getattr(arch, "concat_first_frame_latent", True)),
            )
            batch.dreamzero_image_context_input = image

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
