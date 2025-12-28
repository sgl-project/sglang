from typing import Any, Union

import torch
import torch.nn.functional as F
from diffusers.models.autoencoders.vae import DiagonalGaussianDistribution

from sglang.multimodal_gen.runtime.distributed import get_local_torch_device
from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import Req
from sglang.multimodal_gen.runtime.pipelines_core.stages.base import PipelineStage
from sglang.multimodal_gen.runtime.server_args import ServerArgs
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger
from sglang.multimodal_gen.utils import PRECISION_TO_TYPE

logger = init_logger(__name__)


class WanAnimateConditioningStage(PipelineStage):
    def __init__(self, vae: Any):
        super().__init__()
        self.vae = vae

    def encode(
        self,
        video_condition: torch.Tensor,
        batch: Req,
        server_args: ServerArgs,
    ) -> torch.Tensor:
        # Setup VAE precision
        vae_dtype = PRECISION_TO_TYPE[server_args.pipeline_config.vae_precision]
        vae_autocast_enabled = (
            vae_dtype != torch.float32
        ) and not server_args.disable_autocast

        # Encode Image
        with torch.autocast(
            device_type="cuda", dtype=vae_dtype, enabled=vae_autocast_enabled
        ):
            if server_args.pipeline_config.vae_tiling:
                self.vae.enable_tiling()
            if not vae_autocast_enabled:
                video_condition = video_condition.to(vae_dtype)
            encoder_output: DiagonalGaussianDistribution = self.vae.encode(
                video_condition
            )

        generator = batch.generator

        sample_mode = server_args.pipeline_config.vae_config.encode_sample_mode()

        latent_condition = self.retrieve_latents(
            encoder_output, generator, sample_mode=sample_mode
        )
        latent_condition = server_args.pipeline_config.postprocess_vae_encode(
            latent_condition, self.vae
        )

        scaling_factor, shift_factor = (
            server_args.pipeline_config.get_decode_scale_and_shift(
                device=latent_condition.device,
                dtype=latent_condition.dtype,
                vae=self.vae,
            )
        )

        # apply shift & scale if needed
        if isinstance(shift_factor, torch.Tensor):
            shift_factor = shift_factor.to(latent_condition.device)

        if isinstance(scaling_factor, torch.Tensor):
            scaling_factor = scaling_factor.to(latent_condition.device)

        latent_condition -= shift_factor
        latent_condition = latent_condition * scaling_factor

        # output = server_args.pipeline_config.postprocess_image_latent(
        #     latent_condition, batch
        # )
        return latent_condition

    def retrieve_latents(
        self,
        encoder_output: DiagonalGaussianDistribution,
        generator: torch.Generator | None = None,
        sample_mode: str = "sample",
    ):
        if sample_mode == "sample":
            return encoder_output.sample(generator)
        elif sample_mode == "argmax":
            return encoder_output.mode()
        else:
            raise AttributeError("Could not access latents of provided encoder_output")

    def get_i2v_mask(
        self,
        batch_size: int,
        latent_t: int,
        latent_h: int,
        latent_w: int,
        mask_len: int = 1,
        dtype: torch.dtype = None,
        device: Union[str, torch.device] = "cuda",
    ) -> torch.Tensor:
        # mask_pixel_values shape (if supplied): [B, C = 1, T, latent_h, latent_w]
        mask_lat_size = torch.zeros(
            batch_size,
            1,
            (latent_t - 1) * 4 + 1,
            latent_h,
            latent_w,
            dtype=dtype,
            device=device,
        )
        mask_lat_size[:, :, :mask_len] = 1
        first_frame_mask = mask_lat_size[:, :, 0:1]
        first_frame_mask = torch.repeat_interleave(first_frame_mask, dim=2, repeats=4)
        mask_lat_size = torch.concat([first_frame_mask, mask_lat_size[:, :, 1:]], dim=2)
        mask_lat_size = mask_lat_size.view(
            batch_size, -1, 4, latent_h, latent_w
        ).transpose(
            1, 2
        )  # [B, C = 1, 4 * T_lat, H_lat, W_lat] --> [B, C = 4, T_lat, H_lat, W_lat]

        return mask_lat_size

    def prepare_prev_segment_cond_latents(
        self,
        batch,
        server_args,
        prev_segment_cond_video,
        batch_size: int = 1,
        segment_frame_length: int = 77,
        height: int = 720,
        width: int = 1280,
        prev_segment_cond_frames: int = 1,
        interpolation_mode: str = "bicubic",
        dtype=torch.float32,
        device="cuda",
    ) -> torch.Tensor:
        # prev_segment_cond_video shape: (B, C, T, H, W) in pixel space if supplied
        # background_video shape: (B, C, T, H, W) (same as prev_segment_cond_video shape)
        # mask_video shape: (B, 1, T, H, W) (same as prev_segment_cond_video, but with only 1 channel)
        first_frame = prev_segment_cond_video is None
        cond_frames_shape = (
            batch_size,
            3,
            prev_segment_cond_frames,
            height,
            width,
        )  # In pixel space
        if prev_segment_cond_video is None:
            prev_segment_cond_video = torch.zeros(
                cond_frames_shape, dtype=dtype, device=device
            )
        else:
            assert prev_segment_cond_video.shape == cond_frames_shape

        data_batch_size, channels, _, segment_height, segment_width = (
            prev_segment_cond_video.shape
        )
        num_latent_frames = (segment_frame_length - 1) // 4 + 1
        latent_height = height // 8
        latent_width = width // 8
        if segment_height != height or segment_width != width:
            print(
                f"Interpolating prev segment cond video from ({segment_width}, {segment_height}) to ({width}, {height})"
            )
            # Perform a 4D (spatial) rather than a 5D (spatiotemporal) reshape, following the original code
            prev_segment_cond_video = prev_segment_cond_video.transpose(1, 2).flatten(
                0, 1
            )  # [B * T, C, H, W]
            prev_segment_cond_video = F.interpolate(
                prev_segment_cond_video, size=(height, width), mode=interpolation_mode
            )
            prev_segment_cond_video = prev_segment_cond_video.unflatten(
                0, (batch_size, -1)
            ).transpose(1, 2)

        remaining_segment_frames = segment_frame_length - prev_segment_cond_frames
        remaining_segment = torch.zeros(
            batch_size,
            channels,
            remaining_segment_frames,
            height,
            width,
            dtype=dtype,
            device=device,
        )

        # Prepend the conditioning frames from the previous segment to the remaining segment video in the frame dim
        prev_segment_cond_video = prev_segment_cond_video.to(dtype=dtype)
        full_segment_cond_video = torch.cat(
            [prev_segment_cond_video, remaining_segment], dim=2
        )

        prev_segment_cond_latents = self.encode(
            full_segment_cond_video, batch, server_args
        )

        # Prepare I2V mask
        prev_segment_cond_mask = self.get_i2v_mask(
            batch_size,
            num_latent_frames,
            latent_height,
            latent_width,
            mask_len=prev_segment_cond_frames if not first_frame else 0,
            dtype=dtype,
            device=device,
        )

        # Prepend cond I2V mask to prev segment cond latents along channel dimension
        prev_segment_cond_latents = torch.cat(
            [prev_segment_cond_mask, prev_segment_cond_latents], dim=1
        )
        return prev_segment_cond_latents

    def forward(self, batch: Req, server_args: ServerArgs) -> Req:
        self.vae = self.vae.to(get_local_torch_device())

        clip_len = server_args.pipeline_config.clip_len
        refert_num = server_args.pipeline_config.refert_num
        cur_segment = batch.extra.get("cur_segment")
        start_frame = cur_segment * (clip_len - refert_num)
        end_frame = start_frame + clip_len

        pose_video_tensor = batch.extra.get("pose_video")[
            :, :, start_frame:end_frame, :, :
        ]
        face_video_tensor = batch.extra.get("face_video")[
            :, :, start_frame:end_frame, :, :
        ]
        if cur_segment == 0:
            prev_segment_cond_video = None
        else:
            prev_segment_cond_video = (
                batch.extra.get("all_frames")[:, :, -refert_num:].clone().detach()
            )
            prev_segment_cond_video = prev_segment_cond_video * 2 - 1

        pose_latents_no_ref = self.encode(pose_video_tensor, batch, server_args)

        batch.extra["pose_hidden_states"] = pose_latents_no_ref
        batch.extra["face_pixel_values"] = face_video_tensor

        batch.extra["prev_segment_cond_latents"] = (
            self.prepare_prev_segment_cond_latents(
                batch,
                server_args,
                prev_segment_cond_video,
                segment_frame_length=clip_len,
                height=batch.height,
                width=batch.width,
                prev_segment_cond_frames=refert_num,
                device=get_local_torch_device(),
                dtype=pose_latents_no_ref.dtype,
            )
        )

        self.maybe_free_model_hooks()
        return batch
