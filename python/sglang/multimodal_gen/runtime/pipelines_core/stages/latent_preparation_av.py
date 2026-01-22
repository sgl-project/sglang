import torch
from diffusers.utils.torch_utils import randn_tensor

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
from sglang.multimodal_gen.runtime.server_args import ServerArgs
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

    def forward(self, batch: Req, server_args: ServerArgs) -> Req:
        # 1. Prepare Video Latents using base class logic
        # This sets batch.latents and batch.raw_latent_shape
        batch = super().forward(batch, server_args)

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

        device = get_local_torch_device()
        if isinstance(batch.prompt_embeds, list) and batch.prompt_embeds:
            dtype = batch.prompt_embeds[0].dtype
        elif isinstance(batch.prompt_embeds, torch.Tensor):
            dtype = batch.prompt_embeds.dtype
        else:
            dtype = torch.float16
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

        # Store in batch
        batch.audio_latents = audio_latents
        batch.raw_audio_latent_shape = audio_latents.shape

        return batch
