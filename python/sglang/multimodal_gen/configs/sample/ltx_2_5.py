import dataclasses

from sglang.multimodal_gen.configs.sample.ltx_2 import LTX2SamplingParams


@dataclasses.dataclass
class LTX25SamplingParams(LTX2SamplingParams):
    """Sampling defaults for the LTX-2.5 distilled transformer.

    `model_index.json` points at the distilled DiT, which runs **unguided** off an
    explicit sigma schedule (see `LTX25PipelineConfig.default_sigmas`) rather than
    a step count. `guidance_scale=1.0` disables CFG; STG and modality guidance
    stay off. Feeding it a generic linear schedule instead costs quality.

    Reference: the "Quick start — distilled, convolutional decode" recipe in the
    `Lightricks/LTX-2.5-Diffusers` model card.
    """

    seed: int = 42
    generator_device: str = "cuda"

    height: int = 544
    width: int = 960
    num_frames: int = 121
    fps: int = 24

    guidance_scale: float = 1.0

    # `auto_duration` on the base class has the duration head predict this
    # instead, overriding `num_frames`.
    # The schedule is pinned by the pipeline config; this only keeps the
    # reported step count honest.
    num_inference_steps: int = 8
