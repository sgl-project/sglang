from dataclasses import dataclass

from sglang.multimodal_gen.configs.sample.sampling_params import SamplingParams


@dataclass
class LongCatImageSamplingParams(SamplingParams):
    num_inference_steps: int = 50
    guidance_scale: float = 4.5
    height: int = 1024
    width: int = 1024
    # Override base class defaults to enable LongCat-specific features by default
    enable_cfg_renorm: bool = True
    enable_prompt_rewrite: bool = True
