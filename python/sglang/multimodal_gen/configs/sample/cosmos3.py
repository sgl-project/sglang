# SPDX-License-Identifier: Apache-2.0
"""Cosmos3 sampling parameters.

A single ``SamplingParams`` class serves T2V, I2V, and T2I — the per-request
mode is dispatched in the pipeline from ``num_frames`` (``== 1`` → T2I) and
``image_path`` (set → I2V). For ``num_frames == 1`` the output ``data_type``
flips to ``IMAGE`` so the file extension and decode path agree.
"""

from dataclasses import dataclass, field

from sglang.multimodal_gen.configs.sample.sampling_params import (
    DataType,
    SamplingParams,
)


@dataclass
class Cosmos3SamplingParams(SamplingParams):
    """Cosmos3 sampling parameters (T2V defaults; also used for I2V / T2I)."""

    height: int = 720
    width: int = 1280
    num_frames: int = 81
    fps: int = 24

    guidance_scale: float = 4.0
    num_inference_steps: int = 35

    negative_prompt: str = ""

    # Optional CFG window — T2I requests typically pass e.g. ``(400, 1000)`` to
    # skip guidance at low noise levels. T2V / I2V leave it unset.
    guidance_interval: tuple[float, float] | None = None

    supported_resolutions: list[tuple[int, int]] | None = field(
        default_factory=lambda: [
            (1280, 720),
            (720, 1280),
            (832, 480),
            (480, 832),
            (1024, 1024),
        ]
    )

    def _set_output_file_name(self) -> None:
        # The pipeline config's ``task_type=TI2V`` drives ``data_type`` to
        # VIDEO, but a single-frame request is a T2I and must pick the IMAGE
        # extension. Flip before the base derives the file name.
        if self.num_frames == 1:
            self.data_type = DataType.IMAGE
        super()._set_output_file_name()
