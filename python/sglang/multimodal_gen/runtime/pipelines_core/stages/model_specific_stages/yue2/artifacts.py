# SPDX-License-Identifier: Apache-2.0
"""Optional per-request artifact export pipeline stage."""
from __future__ import annotations

from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import Req
from sglang.multimodal_gen.runtime.pipelines_core.stages.base import PipelineStage
from sglang.multimodal_gen.runtime.server_args import ServerArgs
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

from sglang.srt.models.yue2.protocol import CODEC_OFFSET, SongRequest
from sglang.srt.models.yue2.storage import export_song_artifacts

logger = init_logger(__name__)


class Yue2ArtifactExportStage(PipelineStage):
    """Export a request's NAR latents / token ids / plan when requested.

    Runs after NAR (latents exist) and before VAE, and is a no-op unless the
    request carries ``yue2_artifacts_dir``.
    """

    def forward(self, batch: Req, server_args: ServerArgs) -> Req:
        artifacts_dir = batch.extra.get("yue2_artifacts_dir") or ""
        if not artifacts_dir:
            return batch
        request: SongRequest = batch.extra["yue2_request"]
        export_song_artifacts(
            artifacts_dir,
            request=request,
            latents=batch.extra["yue2_latents"],
            semantic_ids=[
                int(token) - CODEC_OFFSET
                for token in batch.extra["yue2_semantic_ids"]
            ],
            prefix_ids=[int(token) for token in batch.extra["yue2_semantic_prefix"]],
            abc_ids=[int(token) for token in batch.extra.get("yue2_abc_ids") or []],
            timing=batch.extra.get("yue2_timing") or {},
            truncated=bool(batch.extra.get("yue2_truncated")),
        )
        return batch
