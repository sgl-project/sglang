import dataclasses
from typing import Any, ClassVar

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

    use_diffusion_decoder: bool = False
    auto_duration: bool = False
    auto_duration_min_seconds: float = 1.0
    auto_duration_max_seconds: float = 20.0

    # `auto_duration` has the duration head predict this instead, overriding
    # `num_frames`.
    # The schedule is pinned by the pipeline config; this only keeps the
    # reported step count honest.
    num_inference_steps: int = 8

    # The two-stage pipeline generates at half resolution and refines back up;
    # its default is the official two-stage canvas. 960x544 above is the
    # one-stage default, and two-stage cannot halve it: LTX2HalveResolutionStage
    # needs each side to be a multiple of 64. The warmup builder reads this too.
    two_stage_default_size: ClassVar[tuple[int, int]] = (1920, 1088)

    def _adjust(self, server_args: Any) -> None:
        from sglang.multimodal_gen.runtime.server_args import (
            is_ltx2_two_stage_pipeline_name,
        )

        explicit = getattr(self, "_explicit_fields", None) or set()
        if is_ltx2_two_stage_pipeline_name(
            getattr(server_args, "pipeline_class_name", None)
        ) and not explicit & {"width", "height"}:
            self.width, self.height = self.two_stage_default_size
        super()._adjust(server_args)

    @classmethod
    def video_request_extra_fields(cls) -> frozenset[str]:
        return frozenset(
            {
                "auto_duration",
                "auto_duration_max_seconds",
                "auto_duration_min_seconds",
                "use_diffusion_decoder",
            }
        )
