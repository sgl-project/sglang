# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import Req
from sglang.multimodal_gen.runtime.server_args import ServerArgs
from sglang.srt.ug.context import UGContextBundle
from sglang.srt.ug.denoiser import UGMiddleBridge
from sglang.srt.ug.interleaved import UGGKind, UGGSegmentResult


class U1PixelFlowGSegmentExecutor:
    """SenseNova U1 pixel-flow G executor shell.

    Real patchify, timestep, flow update, guidance, and unpatchify logic belongs
    in this model-specific executor in later roadmap items.
    """

    required_g_kind: UGGKind = "pixel_flow"

    def __call__(
        self,
        *,
        bridge: UGMiddleBridge,
        contexts: UGContextBundle,
        batch: Req,
        server_args: ServerArgs,
    ) -> UGGSegmentResult:
        del bridge, contexts, batch, server_args
        raise NotImplementedError(
            "SenseNova U1 pixel-flow executor is not wired yet. U1 G mechanics "
            "will be implemented behind this model-specific executor."
        )
