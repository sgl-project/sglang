# SPDX-License-Identifier: Apache-2.0
"""Wan-Animate-2 pipeline.

Stages: WanAnimate2BeforeDenoisingStage (per-clip conditioning) -> WanAnimate2DenoisingStage
(per-clip denoise + in-loop VAE decode) -> WanAnimate2OutputStage (emit frames).
Routed by model_index.json ``_class_name == "WanAnimate2Pipeline"``; config comes from
the registry detector (Wan_Animate_2_14B_Config). Every component loads through the
standard component loaders. Text uses TextEncodingStage; CLIP and VAE adapters
preserve the reference preprocessing and latent scaling.
"""

from sglang.multimodal_gen.runtime.disaggregation.roles import RoleType
from sglang.multimodal_gen.runtime.distributed import get_local_torch_device
from sglang.multimodal_gen.runtime.loader.component_loaders.transformer_loader import (
    TransformerLoader,
)
from sglang.multimodal_gen.runtime.pipelines_core.composed_pipeline_base import (
    ComposedPipelineBase,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.wan_animate_2 import (
    WanAnimate2BeforeDenoisingStage,
    WanAnimate2DenoisingStage,
    WanAnimate2ImageEncoderAdapter,
    WanAnimate2OutputStage,
    WanAnimate2VaeAdapter,
)
from sglang.multimodal_gen.runtime.server_args import ServerArgs


class WanAnimate2TransformerLoader(TransformerLoader):
    """The checkpoint is the plain Wan2.2-I2V-14B backbone, so every key must map."""

    strict_checkpoint_keys = True


class WanAnimate2Pipeline(ComposedPipelineBase):
    """Wan-Animate-2 14B in-context reference image + reference video -> video pipeline."""

    pipeline_name = "WanAnimate2Pipeline"
    is_video_pipeline = True

    # All seven model_index.json components. The scheduler is built from the
    # checkpoint's scheduler/scheduler_config.json (DPM-Solver++, flow prediction,
    # flow sigmas) by the shared loader. Two of its settings are overridden at
    # runtime: the stages call set_timesteps(sigmas=...) with the official Wan
    # sigma grid at every clip (sigma_0 is exactly 1.0), shifted by
    # pipeline_config.flow_shift, so the JSON's flow_shift and timestep_spacing
    # are not consulted. The JSON's beta_schedule only shapes the beta-derived
    # tables, which the flow-sigma path does not read.
    _required_config_modules: list[str] = [
        "text_encoder",
        "tokenizer",
        "vae",
        "transformer",
        "image_encoder",
        "image_processor",
        "scheduler",
    ]
    component_loaders = {"transformer": WanAnimate2TransformerLoader}

    def create_pipeline_stages(self, server_args: ServerArgs) -> None:
        # Keep native modules visible to residency and layerwise-offload management;
        # adapters only translate the model-specific stage API.
        device = get_local_torch_device()
        image_encoder = WanAnimate2ImageEncoderAdapter.from_loaded(
            self.get_module("image_encoder"),
            image_processor=self.get_module("image_processor"),
            device=device,
        )
        vae = WanAnimate2VaeAdapter(self.get_module("vae"))
        self.add_stage(
            stage_name="wan_animate_2_before_denoising",
            stage=WanAnimate2BeforeDenoisingStage(
                vae=vae,
                image_encoder=image_encoder,
                text_encoder=self.get_module("text_encoder"),
                tokenizer=self.get_module("tokenizer"),
                pipeline_config=server_args.pipeline_config,
                scheduler=self.get_module("scheduler"),
            ),
        )
        # Decode each clip before constructing the following clip's conditions.
        self.add_stage(
            stage_name="wan_animate_2_denoising",
            stage=WanAnimate2DenoisingStage(
                transformer=self.get_module("transformer"),
                scheduler=self.get_module("scheduler"),
                pipeline=self,
                vae=vae,
                image_encoder=image_encoder,
            ),
        )
        # Replaces the standard DecodingStage: frames are already decoded in the loop.
        self.add_stage_factory(
            RoleType.DECODER,
            lambda: WanAnimate2OutputStage(),
            "decoding_stage",
        )


EntryClass = WanAnimate2Pipeline
