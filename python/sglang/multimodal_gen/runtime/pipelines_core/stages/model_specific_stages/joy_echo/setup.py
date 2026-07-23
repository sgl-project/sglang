# SPDX-License-Identifier: Apache-2.0
"""JoyEcho pre-denoising setup stages (multi-shot session + sigma schedule)."""

from __future__ import annotations

from typing import TYPE_CHECKING

from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import Req
from sglang.multimodal_gen.runtime.pipelines_core.stages.base import PipelineStage
from sglang.multimodal_gen.runtime.server_args import ServerArgs
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

if TYPE_CHECKING:
    from sglang.multimodal_gen.runtime.pipelines.joy_echo_pipeline import (
        JoyEchoPipeline,
    )

logger = init_logger(__name__)


class JoyEchoMultishotSetupStage(PipelineStage):
    """Apply official per-shot seeding before input validation and latent noise."""

    def __init__(self, pipeline: JoyEchoPipeline) -> None:
        super().__init__()
        self.pipeline = pipeline

    def _maybe_reset_multishot_session(self, batch: Req) -> None:
        """Reset shot index and memory bank at the start of a new ``generate()`` session."""
        if not batch.reset_memory_bank:
            return

        session_id = batch.request_id
        if session_id is not None:
            if session_id == self.pipeline._multishot_session_id:
                return
            self.pipeline._multishot_session_id = session_id
            self.pipeline.multishot_index = 0
            self.pipeline.reset_memory_bank()
            logger.info(
                "JoyEcho memory bank reset for new multi-shot session (request_id=%s)",
                session_id,
            )
            return

        if self.pipeline.multishot_index == 0:
            self.pipeline.reset_memory_bank()
            logger.info("JoyEcho memory bank reset for new multi-shot session")

    def forward(self, batch: Req, server_args: ServerArgs) -> Req:
        if not batch.enable_memory_bank:
            return batch

        self._maybe_reset_multishot_session(batch)

        shot_idx = self.pipeline.multishot_index
        self.pipeline.multishot_index += 1

        base_seed = batch.seed
        if isinstance(base_seed, list):
            if not base_seed:
                raise ValueError("seed list must not be empty for JoyEcho multi-shot")
            base_seed = base_seed[0]

        # Official inference.py: prompt_seed = int(cfg.seed) + shot_idx
        batch.seed = int(base_seed) + shot_idx

        logger.info(
            "JoyEcho multi-shot setup: shot_idx=%d seed=%d",
            shot_idx,
            batch.seed,
        )
        return batch


class JoyEchoSigmaPreparationStage(PipelineStage):
    """Prepare JoyEcho DMD sigma schedule without LTX-2 shift remapping."""

    def forward(self, batch: Req, server_args: ServerArgs) -> Req:
        batch.extra["ltx2_phase"] = "stage1"

        sigmas = batch.sigmas
        if sigmas is None:
            sampling_sigmas = batch.sampling_params.sigmas
            if sampling_sigmas is not None:
                sigmas = list(sampling_sigmas)
            else:
                sigmas = list(server_args.pipeline_config.default_sigmas)

        batch.sigmas = list(sigmas)
        batch.num_inference_steps = max(len(batch.sigmas) - 1, 1)
        return batch
