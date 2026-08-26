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


@dataclass
class LongCatImageEditSamplingParams(SamplingParams):
    """Defaults for LongCat-Image-Edit (mirrors diffusers LongCatImageEditPipeline).

    Output height/width are derived from the condition image (~1MP), so no
    defaults are set here. The reference uses an empty negative prompt and
    plain CFG (no renorm, no prompt rewrite).
    """

    num_frames: int = 1
    num_inference_steps: int = 50
    guidance_scale: float = 4.5
    negative_prompt: str = ""
    enable_cfg_renorm: bool = False
    enable_prompt_rewrite: bool = False


@dataclass
class LongCatImageEditTurboSamplingParams(LongCatImageEditSamplingParams):
    """LongCat-Image-Edit-Turbo: distilled, 8 steps, CFG disabled."""

    num_inference_steps: int = 8
    guidance_scale: float = 1.0
