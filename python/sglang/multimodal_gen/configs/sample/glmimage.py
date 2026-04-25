from dataclasses import dataclass, field

from sglang.multimodal_gen.configs.sample.sampling_params import SamplingParams
from sglang.multimodal_gen.configs.sample.teacache import TeaCacheParams


@dataclass
class GlmImageTeaCacheParams(TeaCacheParams):
    """TeaCache parameters for GLM-Image model.

    GLM-Image uses the standard TeaCache approach with modulated input
    computed from the timestep embedding (temb).
    """

    # Default threshold tuned for GLM-Image quality/speed tradeoff
    teacache_thresh: float = 0.2

    # Default polynomial coefficients for L1 rescaling
    # These may need tuning specific to GLM-Image
    coefficients: list[float] = field(
        default_factory=lambda: [
            -6071.632298241158,
            1837.6579251847247,
            -172.12278847677337,
            7.159036598427308,
            -0.07853601464946189,
        ]
    )


@dataclass
class GlmImageSamplingParams(SamplingParams):
    negative_prompt = ""

    num_frames: int = 1
    guidance_scale: float = 1.5
    num_inference_steps: int = 30

    teacache_params: GlmImageTeaCacheParams = field(
        default_factory=GlmImageTeaCacheParams
    )
