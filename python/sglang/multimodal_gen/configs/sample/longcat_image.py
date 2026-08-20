from dataclasses import dataclass

from sglang.multimodal_gen.configs.sample.sampling_params import SamplingParams


@dataclass
class LongCatImageSamplingParams(SamplingParams):
    num_inference_steps: int = 50
    guidance_scale: float = 4.5
    height: int = 1024
    width: int = 1024
    enable_cfg_renorm: bool = True
    cfg_renorm_min: float = 0.0
    enable_prompt_rewrite: bool = True

    @classmethod
    def image_request_extra_fields(cls) -> frozenset[str]:
        return frozenset(
            {
                "cfg_renorm_min",
                "enable_cfg_renorm",
                "enable_prompt_rewrite",
            }
        )
