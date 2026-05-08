import torch
from diffusers.utils.torch_utils import randn_tensor

from sglang.multimodal_gen.configs.pipeline_configs.ltx_2 import (
    is_ltx23_native_variant,
)
from sglang.multimodal_gen.runtime.distributed import get_local_torch_device
from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import Req
from sglang.multimodal_gen.runtime.pipelines_core.stages.latent_preparation import (
    LatentPreparationStage,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.validators import (
    StageValidators as V,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.validators import (
    VerificationResult,
)
from sglang.multimodal_gen.runtime.server_args import (
    ServerArgs,
    is_ltx2_two_stage_pipeline_name,
)
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

logger = init_logger(__name__)


class LTX2AVLatentPreparationStage(LatentPreparationStage):
    """
    LTX-2 specific latent preparation stage that handles both video and audio latents.
    """

    def __init__(self, scheduler, transformer=None, audio_vae=None):
        super().__init__(scheduler, transformer)
        self.audio_vae = audio_vae

    def verify_input(self, batch: Req, server_args: ServerArgs) -> VerificationResult:
        """Verify latent preparation stage inputs."""
        result = VerificationResult()
        result.add_check(
            "prompt_or_embeds",
            None,
            lambda _: V.string_or_list_strings(batch.prompt)
            or V.list_not_empty(batch.prompt_embeds)
            or V.is_tensor(batch.prompt_embeds),
        )

        if isinstance(batch.prompt_embeds, list):
            result.add_check("prompt_embeds", batch.prompt_embeds, V.list_of_tensors)
        else:
            result.add_check("prompt_embeds", batch.prompt_embeds, V.is_tensor)

        result.add_check(
            "num_videos_per_prompt", batch.num_outputs_per_prompt, V.positive_int
        )
        result.add_check("generator", batch.generator, V.generator_or_list_generators)
        result.add_check("num_frames", batch.num_frames, V.positive_int)
        result.add_check("height", batch.height, V.positive_int)
        result.add_check("width", batch.width, V.positive_int)
        result.add_check("latents", batch.latents, V.none_or_tensor)
        return result

    def _get_latent_dtype(
        self,
        batch: Req,
        server_args: ServerArgs,
    ):
        if is_ltx23_native_variant(server_args.pipeline_config.vae_config.arch_config):
            if is_ltx2_two_stage_pipeline_name(server_args.pipeline_class_name):
                return server_args.pipeline_config.get_latent_dtype(
                    batch.prompt_embeds[0].dtype
                )
            return torch.float32
        return torch.float32

    @staticmethod
    def _packed_video_latent_shape(
        latent_shape: tuple[int, int, int, int, int],
        pipeline_config,
    ) -> tuple[int, int, int]:
        batch_size, channels, num_frames, height, width = latent_shape
        patch_size_t = int(pipeline_config.patch_size_t)
        patch_size = int(pipeline_config.patch_size)
        return (
            batch_size,
            (num_frames // patch_size_t)
            * (height // patch_size)
            * (width // patch_size),
            channels * patch_size_t * patch_size * patch_size,
        )

    @staticmethod
    def _packed_audio_latent_shape(
        latent_shape: tuple[int, int, int, int],
    ) -> tuple[int, int, int]:
        batch_size, channels, latent_length, mel_bins = latent_shape
        return (batch_size, latent_length, channels * mel_bins)

    def forward(self, batch: Req, server_args: ServerArgs) -> Req:
        if not is_ltx23_native_variant(
            server_args.pipeline_config.vae_config.arch_config
        ):
            batch = super().forward(batch, server_args)

            try:
                generate_audio = batch.generate_audio
            except AttributeError:
                generate_audio = True
            if not generate_audio:
                batch.audio_latents = None
                batch.raw_audio_latent_shape = None
                return batch

            device = get_local_torch_device()
            dtype = self._get_latent_dtype(batch, server_args)
            generator = batch.generator

            audio_latents = batch.audio_latents
            batch_size = batch.batch_size
            num_frames = batch.num_frames

            if audio_latents is None:
                shape = server_args.pipeline_config.prepare_audio_latent_shape(
                    batch, batch_size, num_frames
                )

                audio_latents = randn_tensor(
                    shape, generator=generator, device=device, dtype=dtype
                )
            else:
                audio_latents = audio_latents.to(device)

            audio_latents = server_args.pipeline_config.maybe_pack_audio_latents(
                audio_latents, batch_size, batch
            )

            batch.audio_latents = audio_latents
            batch.raw_audio_latent_shape = audio_latents.shape
            return batch

        # 1. Prepare video latents directly in packed token space.
        # Official LTX-2.3 pipelines sample noise after patchify; generating unpacked
        # [B, C, F, H, W] noise and packing afterwards changes token ordering.
        latent_num_frames = self.adjust_video_length(batch, server_args)
        batch_size = batch.batch_size
        dtype = self._get_latent_dtype(batch, server_args)
        device = get_local_torch_device()
        generator = batch.generator

        latents = batch.latents
        num_frames = (
            latent_num_frames if latent_num_frames is not None else batch.num_frames
        )

        if latents is None:
            latent_shape = server_args.pipeline_config.prepare_latent_shape(
                batch, batch_size, num_frames
            )
            packed_video_shape = self._packed_video_latent_shape(
                latent_shape, server_args.pipeline_config
            )
            latents = randn_tensor(
                packed_video_shape,
                generator=generator,
                device=device,
                dtype=dtype,
            )
            batch.extra["ltx2_stage1_packed_video_shape"] = tuple(packed_video_shape)

            latent_ids = server_args.pipeline_config.maybe_prepare_latent_ids(latents)
            if latent_ids is not None:
                batch.latent_ids = latent_ids.to(device=device)
        else:
            latents = latents.to(device)
            latents = server_args.pipeline_config.maybe_pack_latents(
                latents, batch_size, batch
            )

        if hasattr(self.scheduler, "init_noise_sigma"):
            latents = latents * self.scheduler.init_noise_sigma

        batch.latents = latents
        batch.raw_latent_shape = latents.shape

        # 2. Prepare Audio Latents (optional)
        # Default to True if not specified
        try:
            generate_audio = batch.generate_audio
        except AttributeError:
            generate_audio = True
        if not generate_audio:
            batch.audio_latents = None
            batch.raw_audio_latent_shape = None
            return batch

        audio_latents = batch.audio_latents

        if audio_latents is None:
            latent_shape = server_args.pipeline_config.prepare_audio_latent_shape(
                batch, batch_size, batch.num_frames
            )
            packed_audio_shape = self._packed_audio_latent_shape(latent_shape)
            audio_latents = randn_tensor(
                packed_audio_shape,
                generator=generator,
                device=device,
                dtype=dtype,
            )
            batch.extra["ltx2_stage1_packed_audio_shape"] = tuple(packed_audio_shape)
        else:
            audio_latents = audio_latents.to(device)
            audio_latents = server_args.pipeline_config.maybe_pack_audio_latents(
                audio_latents, batch_size, batch
            )

        # Store in batch
        batch.audio_latents = audio_latents
        batch.raw_audio_latent_shape = audio_latents.shape

        return batch
