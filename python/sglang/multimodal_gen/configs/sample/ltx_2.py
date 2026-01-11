import dataclasses
from sglang.multimodal_gen.configs.sample.sampling_params import SamplingParams


@dataclasses.dataclass
class LTX2SamplingParams(SamplingParams):
    """Sampling parameters for LTX-2."""

    # Video parameters
    height: int = 480
    width: int = 704
    num_frames: int = 121
    fps: int = 24
    
    # Audio specific
    generate_audio: bool = True
    
    # Denoising parameters
    guidance_scale: float = 3.0
    num_inference_steps: int = 50
