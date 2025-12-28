from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import OutputBatch, Req
from sglang.multimodal_gen.runtime.pipelines_core.stages.base import PipelineStage
from sglang.multimodal_gen.runtime.pipelines_core.stages.decoding import DecodingStage
from sglang.multimodal_gen.runtime.pipelines_core.stages.denoising import DenoisingStage
from sglang.multimodal_gen.runtime.pipelines_core.stages.latent_preparation import (
    LatentPreparationStage,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.timestep_preparation import (
    TimestepPreparationStage,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.wan_animate_conditioning import (
    WanAnimateConditioningStage,
)
from sglang.multimodal_gen.runtime.server_args import ServerArgs
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

logger = init_logger(__name__)


class SegmentLoopStage(PipelineStage):
    def __init__(self, vae, scheduler, transformer, transformer_2) -> None:
        super().__init__()
        self.animate_cond_stage = WanAnimateConditioningStage(vae=vae)
        self.timestep_prep_stage = TimestepPreparationStage(scheduler=scheduler)
        self.latent_stage = LatentPreparationStage(
            scheduler=scheduler, transformer=transformer
        )
        self.denoising_stage = DenoisingStage(
            transformer=transformer,
            transformer_2=transformer_2,
            scheduler=scheduler,
            vae=vae,
        )
        self.decoding_stage = DecodingStage(vae=vae)

    def forward(
        self,
        batch: Req,
        server_args: ServerArgs,
    ) -> OutputBatch:

        num_segments = batch.extra["num_segments"]
        refert_num = server_args.pipeline_config.refert_num
        final_video_segments = []

        # Clear any previous generated context
        batch.extra["all_frames"] = None
        output = None
        for segment_idx in range(num_segments):
            logger.debug(f"--> Processing segment {segment_idx + 1}/{num_segments}")

            # Update segment index in batch for downstream stages to read
            batch.extra["cur_segment"] = segment_idx
            # --- Pipeline Chain ---
            # 1. Conditioning (Encodes the specific segment of pose/face/image)
            batch = self.animate_cond_stage.forward(batch, server_args)

            # 2. Timestep Prep (Sets up noise schedule for this run)
            batch = self.timestep_prep_stage.forward(batch, server_args)

            # 3. Latent Prep (Prepares initial noise or input latents)
            batch = self.latent_stage.forward(batch, server_args)

            # 4. Denoising (Runs the Transformer/UNet loop)
            batch = self.denoising_stage.forward(batch, server_args)

            # 5. Decoding (VAE Decode for this segment)
            output = self.decoding_stage.forward(batch, server_args)
            # If decoding returned an OutputBatch, it's the final assembled output —
            # propagate it upward immediately instead of treating it as a Req.
            if isinstance(output, OutputBatch):
                break

        # 5. Final Concatenation
        # Concatenate all valid non-overlapping segments
        assert isinstance(output, OutputBatch)

        logger.info("WanAnimate generation complete.")

        return output
