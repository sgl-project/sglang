# Copied and adapted from: https://github.com/hao-ai-lab/FastVideo

# SPDX-License-Identifier: Apache-2.0
from dataclasses import dataclass, field

from sglang.multimodal_gen.configs.sample.sampling_params import SamplingParams
from sglang.multimodal_gen.configs.sample.teacache import TeaCacheParams

@dataclass
class FluxSamplingParams(SamplingParams):
    num_frames: int = 1
    # Denoising stage
    guidance_scale: float = 1.0
    negative_prompt: str = None
    num_inference_steps: int = 50

    teacache_params: TeaCacheParams = field(
        default_factory=lambda: TeaCacheParams(
            teacache_thresh=0.6, # from https://github.com/ali-vilab/TeaCache/blob/7c10efc4702c6b619f47805f7abe4a7a08085aa0/TeaCache4FLUX/teacache_flux.py#L323
            coefficients=[4.98651651e+02, -2.83781631e+02,  5.58554382e+01, -3.82021401e+00, 2.64230861e-01] # from https://github.com/ali-vilab/TeaCache/blob/7c10efc4702c6b619f47805f7abe4a7a08085aa0/TeaCache4FLUX/teacache_flux.py#L113C32-L113C115
        )
    )

    def __post_init__(self):
        default_sample_size = 128
        vae_scale_factor = 8
        # FIXME
        # self.height = default_sample_size * vae_scale_factor
        # self.width = default_sample_size * vae_scale_factor


@dataclass
class Flux2KleinSamplingParams(FluxSamplingParams):
    # Klein is step-distilled, so default to 4 steps
    num_inference_steps: int = 4
