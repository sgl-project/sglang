from dataclasses import dataclass

from sglang.multimodal_gen.configs.sample.sampling_params import SamplingParams


@dataclass
class JoyImageEditSamplingParams(SamplingParams):
    """Default sampling params for JoyImage Edit single-image I2I."""

    negative_prompt: str = ""
    num_frames: int = 1
    guidance_scale: float = 4.0
    num_inference_steps: int = 40
