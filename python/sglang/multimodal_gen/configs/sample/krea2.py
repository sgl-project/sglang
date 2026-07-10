# Krea-2 (K2) sampling defaults.
#
# `guidance_scale` is the SGLang classifier-free-guidance scale, which equals the
# K2 reference `cfg + 1` (SGLang combines `uncond + scale*(cond-uncond)`, the K2
# reference uses `cond + cfg*(cond-uncond)`). So K2 cfg=0 -> guidance_scale=1.0
# (no CFG), K2 cfg=3.5 -> guidance_scale=4.5.
from dataclasses import dataclass

from sglang.multimodal_gen.configs.sample.sampling_params import SamplingParams


@dataclass
class Krea2SamplingParams(SamplingParams):
    """Distilled `oss_turbo` defaults: 8 steps, CFG disabled."""

    negative_prompt: str = ""
    num_frames: int = 1
    height: int = 1024
    width: int = 1024
    guidance_scale: float = 1.0
    num_inference_steps: int = 8


@dataclass
class Krea2RawSamplingParams(Krea2SamplingParams):
    """Base `oss_raw` defaults: full sampler with CFG."""

    guidance_scale: float = 4.5
    num_inference_steps: int = 52
