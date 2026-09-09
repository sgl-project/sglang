# SPDX-License-Identifier: Apache-2.0
"""Cosmos-Dreams-Transfer pipeline configuration.

Same distilled Cosmos3 Omni checkpoint layout as Cosmos-Dreams, with a
``control_video`` conditioning contract instead of an action contract.
"""

from dataclasses import dataclass

from sglang.multimodal_gen.configs.models.dits.cosmos_dreams import (
    CONTROL_VIDEO_CONDITIONING_MODE,
)
from sglang.multimodal_gen.configs.pipeline_configs.cosmos_dreams import (
    CosmosDreamsConfig,
)


@dataclass
class CosmosDreamsTransferConfig(CosmosDreamsConfig):
    conditioning_mode: str = CONTROL_VIDEO_CONDITIONING_MODE

    # The transfer stages tokenize with the Transfer system prompt themselves;
    # recorded here so generic prompt plumbing reports the truth.
    use_system_prompt: bool = True
