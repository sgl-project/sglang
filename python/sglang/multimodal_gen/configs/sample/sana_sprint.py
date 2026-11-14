# SPDX-License-Identifier: Apache-2.0
"""Sampling parameters for SANA-Sprint few-step image generation."""

from dataclasses import dataclass

from sglang.multimodal_gen.configs.sample.sana import SanaSamplingParams

# Diffusers' default max angle, 3.7e-6 above pi/2. sCM's sin(a) / (cos(a) + sin(a))
# leaves [0, 1] outside [0, pi/2] and has a pole at 3pi/4.
_MAX_TRIGFLOW_ANGLE = 1.57080


@dataclass
class SanaSprintSamplingParams(SanaSamplingParams):
    """Defaults follow Diffusers' ``SanaSprintPipeline``.

    SANA-Sprint is guidance-distilled: ``guidance_scale`` feeds the DiT's
    guidance embedding and no unconditional pass runs, so negative prompts
    are not used.

    The sCM schedule runs over TrigFlow angles in radians: ``max_timesteps``
    down to 0 in equal steps, or ``max_timesteps, intermediate_timesteps, 0``
    for 2 steps. ``trigflow_timesteps`` replaces that schedule with explicit
    angles (``num_inference_steps + 1`` of them, ending at the clean sample).

    ``use_resolution_binning`` generates at the closest trained 1024px aspect
    ratio and resizes the image back to the requested size.
    """

    num_inference_steps: int = 2
    guidance_scale: float = 4.5
    negative_prompt: str | None = None
    max_timesteps: float = _MAX_TRIGFLOW_ANGLE
    # None means Diffusers' 1.3 for 2 steps and no intermediate angle otherwise.
    intermediate_timesteps: float | None = None
    trigflow_timesteps: list[float] | None = None
    use_resolution_binning: bool = True

    @classmethod
    def image_request_extra_fields(cls) -> frozenset[str]:
        return frozenset(
            {
                "max_timesteps",
                "intermediate_timesteps",
                "trigflow_timesteps",
                "use_resolution_binning",
            }
        )

    def should_use_classifier_free_guidance(self) -> bool:
        return False

    def _validate_with_pipeline_config(self, pipeline_config):
        super()._validate_with_pipeline_config(pipeline_config)
        if self.trigflow_timesteps is not None:
            self._validate_trigflow_timesteps(list(self.trigflow_timesteps))
            return
        if not 0 < self.max_timesteps <= _MAX_TRIGFLOW_ANGLE:
            raise ValueError(
                f"max_timesteps must lie in (0, {_MAX_TRIGFLOW_ANGLE}], "
                f"got {self.max_timesteps}"
            )
        if self.intermediate_timesteps is None:
            return
        if self.num_inference_steps != 2:
            raise ValueError(
                "intermediate_timesteps only applies to num_inference_steps=2, got "
                f"num_inference_steps={self.num_inference_steps}"
            )
        if not 0 < self.intermediate_timesteps < self.max_timesteps:
            raise ValueError(
                "intermediate_timesteps must lie in (0, max_timesteps), got "
                f"{self.intermediate_timesteps} with max_timesteps={self.max_timesteps}"
            )

    def _validate_trigflow_timesteps(self, angles: list[float]) -> None:
        if len(angles) != self.num_inference_steps + 1:
            raise ValueError(
                "trigflow_timesteps must hold num_inference_steps + 1 angles, got "
                f"{len(angles)} for num_inference_steps={self.num_inference_steps}"
            )
        if any(not 0 <= a <= _MAX_TRIGFLOW_ANGLE for a in angles):
            raise ValueError(
                f"trigflow_timesteps must lie in [0, {_MAX_TRIGFLOW_ANGLE}], got {angles}"
            )
        if any(later >= earlier for earlier, later in zip(angles, angles[1:])):
            raise ValueError(f"trigflow_timesteps must be decreasing, got {angles}")
        if angles[-1] != 0:
            raise ValueError(
                f"trigflow_timesteps must end at 0, the clean sample, got {angles}"
            )
