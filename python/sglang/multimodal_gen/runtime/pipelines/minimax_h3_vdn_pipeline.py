# SPDX-License-Identifier: Apache-2.0
from sglang.multimodal_gen.runtime.pipelines.minimax_h3_pipeline import (
    MiniMaxH3Pipeline,
)


class VDNH3Pipeline(MiniMaxH3Pipeline):
    """VDN-H3 on the MiniMax-H3 pipeline: the model overlay materializes a
    base-H3 layout (LoRAs prefused, linear branch attached), so only the DiT
    blocks' attention and its backend differ."""

    pipeline_name = "VDNH3Pipeline"
    default_model_subfolder = None


EntryClass = [VDNH3Pipeline]
