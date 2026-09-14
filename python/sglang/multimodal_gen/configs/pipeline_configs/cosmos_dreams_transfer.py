# SPDX-License-Identifier: Apache-2.0
"""Cosmos-Dreams-Transfer pipeline configuration.

Same distilled Cosmos3 Omni checkpoint layout as Cosmos-Dreams, with a
``control_video`` conditioning contract instead of an action contract.
"""

from dataclasses import dataclass

from sglang.multimodal_gen.configs.models.dits.cosmos_dreams import (
    CONTROL_VIDEO_CONDITIONING_MODE,
)
from sglang.multimodal_gen.configs.pipeline_configs.base import ModelTaskType
from sglang.multimodal_gen.configs.pipeline_configs.cosmos_dreams import (
    CosmosDreamsConfig,
)


@dataclass
class CosmosDreamsTransferConfig(CosmosDreamsConfig):
    conditioning_mode: str = CONTROL_VIDEO_CONDITIONING_MODE
    # The control clip arrives as control_path, never as an image; TI2V would
    # make the server warmup attach a synthetic image the request validator rejects.
    task_type: ModelTaskType = ModelTaskType.T2V

    def adjust_num_frames(self, num_frames: int, *, log_adjustment: bool = True) -> int:
        # The rollout consumes whole latent chunks after frame 0 and the control
        # stage trims clips to that partition; rounding here keeps warmup probes
        # on counts the pipeline accepts (17, 33, 49, ... pixel frames at 4x4).
        del log_adjustment
        chunk, factor = self.chunk_size, self.temporal_compression_factor
        latent = max(1 + chunk, (max(int(num_frames), 1) - 1) // factor + 1)
        aligned_latent = 1 + ((latent - 1) // chunk) * chunk
        return 1 + (aligned_latent - 1) * factor

    # The transfer stages tokenize with the Transfer system prompt themselves;
    # recorded here so generic prompt plumbing reports the truth.
    use_system_prompt: bool = True
