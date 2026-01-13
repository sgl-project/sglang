from dataclasses import dataclass

from sglang.multimodal_gen.configs.sample.sampling_params import SamplingParams


@dataclass
class GlmImageSamplingParams(SamplingParams):
    negative_prompt = ""

    num_frames: int = 1
    guidance_scale: float = 1.5
    num_inference_steps: int = 30
