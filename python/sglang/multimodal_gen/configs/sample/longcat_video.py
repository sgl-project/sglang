from dataclasses import dataclass, field

from sglang.multimodal_gen.configs.sample.sampling_params import SamplingParams


@dataclass
class LongCatVideoT2VSamplingParams(SamplingParams):
    height: int = 480
    width: int = 832
    num_frames: int = 93
    fps: int = 16

    guidance_scale: float = 4.0
    negative_prompt: str = "低质量, 模糊, 噪声, 变形, 丑陋"
    num_inference_steps: int = 50

    supported_resolutions: list[tuple[int, int]] | None = field(
        default_factory=lambda: [
            (832, 480),
            (480, 832),
        ]
    )
