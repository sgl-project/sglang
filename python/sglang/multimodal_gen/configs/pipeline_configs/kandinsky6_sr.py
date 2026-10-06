# SPDX-License-Identifier: Apache-2.0
"""Kandinsky 6 video super-resolution (VSR) pipeline configuration.

The SR pipeline is video -> video: no text encoder and no audio model.  Its ``scheduler``
component selects the sampler (``PiflowScheduler`` with ``nfe``: pi-Flow; otherwise
flow-Euler).  ``ModelTaskType.TI2V`` is used (like Cosmos3's video-to-video) because the
task enum has no video-to-video member; the input video travels in
``SamplingParams.video_path`` and every SR-specific stage owns its own input handling, so
the generic image pre-processing is skipped.
"""

import re
from dataclasses import dataclass, field

from sglang.multimodal_gen.configs.models.dits.kandinsky6_sr import (
    Kandinsky6SRDitConfig,
)
from sglang.multimodal_gen.configs.models.vaes.kandinsky6_sr import (
    Kandinsky6SRVAEConfig,
)
from sglang.multimodal_gen.configs.pipeline_configs.base import (
    ModelTaskType,
    PipelineConfig,
)


@dataclass
class Kandinsky6SRPipelineConfig(PipelineConfig):
    """Configuration for ``Kandinsky6SRPipeline``."""

    task_type: ModelTaskType = ModelTaskType.TI2V
    skip_input_image_preprocess: bool = True

    # A missing / unexpected weight, a bad config or an unsupported flag must fail the
    # load loudly: the loaders only load strictly and re-raise their errors (instead of
    # falling back to a diffusers ``AutoModel``) for these components.
    native_only_components: tuple[str, ...] = ("transformer", "vae", "latent_upscaler")

    dit_config: Kandinsky6SRDitConfig = field(default_factory=Kandinsky6SRDitConfig)
    dit_precision: str = "bf16"
    vae_config: Kandinsky6SRVAEConfig = field(default_factory=Kandinsky6SRVAEConfig)
    vae_precision: str = "bf16"
    vae_tiling: bool = False
    vae_sp: bool = False

    # Text-free: no text encoders (all four per-encoder tuples must stay equally long).
    text_encoder_configs: tuple = ()
    text_encoder_precisions: tuple = ()
    preprocess_text_funcs: tuple = ()
    postprocess_text_funcs: tuple = ()

    should_use_guidance: bool = False
    enable_autocast: bool = False

    def supports_disaggregation(self) -> bool:
        # The tiled loop runs in one process over shared tensors.
        return False


__all__ = ["Kandinsky6SRPipelineConfig"]


def _is_kandinsky6_sr(model_id: str) -> bool:
    normalized = model_id.lower().replace("-", "").replace("_", "")
    return re.search(r"kandinsky6(?:\.\d+)?(?:vsr|sr|superres)", normalized) is not None


def register():
    from sglang.multimodal_gen.configs.sample.kandinsky6_sr import (
        Kandinsky6SRSamplingParams,
    )
    from sglang.multimodal_gen.registry import register_configs

    register_configs(
        sampling_param_cls=Kandinsky6SRSamplingParams,
        pipeline_config_cls=Kandinsky6SRPipelineConfig,
        hf_model_paths=[
            "kandinskylab/Kandinsky-6.0-VSR-5s-Diffusers",
            "kandinskylab/Kandinsky-6.0-VSR-distilled2steps-5s-Diffusers",
        ],
        model_detectors=[_is_kandinsky6_sr],
    )
