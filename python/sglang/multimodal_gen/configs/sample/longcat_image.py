from dataclasses import dataclass, field

from sglang.multimodal_gen.configs.sample.sampling_params import SamplingParams
from sglang.multimodal_gen.configs.sample.teacache import TeaCacheParams


@dataclass
class LongCatImageSamplingParams(SamplingParams):
    num_inference_steps: int = 50
    guidance_scale: float = 4.5
    height: int = 1024
    width: int = 1024
    # Override base class defaults to enable LongCat-specific features by default
    enable_cfg_renorm: bool = True
    enable_prompt_rewrite: bool = True

    # TeaCache is disabled by default until coefficients are validated for LongCat-Image.
    # Coefficients borrowed from ZImage (same Flux-like MMDiT architecture family).
    # Enable with: --enable-teacache
    enable_teacache: bool = False
    teacache_params: TeaCacheParams = field(
        default_factory=lambda: TeaCacheParams(
            teacache_thresh=0.15,
            coefficients=[
                7.33226126e02,
                -4.01131952e02,
                6.75869174e01,
                -3.14987800e00,
                9.61237896e-02,
            ],
        )
    )
