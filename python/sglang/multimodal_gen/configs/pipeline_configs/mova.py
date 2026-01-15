# SPDX-License-Identifier: Apache-2.0
"""
MoVA pipeline configuration.
"""

from dataclasses import dataclass

from sglang.multimodal_gen.configs.pipeline_configs.base import (
    ModelTaskType,
    PipelineConfig,
)
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

logger = init_logger(__name__)


@dataclass
class MovaPipelineConfig(PipelineConfig):
    """Configuration for MoVA (text+image -> video+audio) pipelines."""

    task_type: ModelTaskType = ModelTaskType.TI2V

    # MoVA specific
    audio_vae_type: str = "dac"
    boundary_ratio: float | None = 0.9

    # temporal alignment: MoVA expects (num_frames - 1) % 4 == 0
    time_division_factor: int = 4
    time_division_remainder: int = 1

    def adjust_num_frames(self, num_frames: int) -> int:
        if num_frames is None:
            return num_frames
        if num_frames % self.time_division_factor != self.time_division_remainder:
            adjusted = (
                (num_frames + self.time_division_factor - 1)
                // self.time_division_factor
                * self.time_division_factor
                + self.time_division_remainder
            )
            logger.warning(
                "`num_frames` (%s) is not compatible with MoVA temporal constraints. "
                "Rounding to %s.",
                num_frames,
                adjusted,
            )
            return adjusted
        return num_frames
