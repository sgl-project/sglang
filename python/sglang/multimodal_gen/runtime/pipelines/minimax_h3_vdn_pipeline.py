# SPDX-License-Identifier: Apache-2.0
from sglang.multimodal_gen.runtime.pipelines.minimax_h3_pipeline import (
    MiniMaxH3Pipeline,
)


class VDNH3Pipeline(MiniMaxH3Pipeline):
    """VDN-H3: hybrid window-softmax + Video Delta linear attention
    MiniMax-H3, 8-NFE DMD2 distill (t2va only).

    OpenVDN/vdn-minimax-h3 ships deltas (a linear branch plus two LoRAs) on
    the released H3. The registered model overlay materializes a base-H3
    layout with both LoRAs prefused and the branch attached, so every stage,
    loader and admission path below is exactly the MiniMax-H3 one; only the
    DiT blocks' attention (and its backend) differ.
    """

    pipeline_name = "VDNH3Pipeline"
    default_model_subfolder = None


EntryClass = [VDNH3Pipeline]
