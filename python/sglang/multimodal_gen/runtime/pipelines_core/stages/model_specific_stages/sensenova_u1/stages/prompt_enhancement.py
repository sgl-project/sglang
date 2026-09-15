# SPDX-License-Identifier: Apache-2.0
"""Prompt enhancement stage for the native SenseNova-U1 pipeline.

SenseNova-U1 ships no local PE checkpoint; enhancement is always a remote
call through SenseNovaU1PEClient (OpenAI-compatible chat/completions).
"""

from __future__ import annotations

from sglang.multimodal_gen.configs.sensenova_u1 import (
    DEFAULT_PE_SYSTEM_PROMPT,
    SENSENOVA_U1_REQUEST_EXTRA_KEY,
)
from sglang.multimodal_gen.runtime.models.sensenova_u1.pe_client import (
    SenseNovaU1PEClient,
)
from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import Req
from sglang.multimodal_gen.runtime.pipelines_core.stages.base import PipelineStage
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.sensenova_u1.stages.generation import (
    SenseNovaU1GenerationOptions,
)
from sglang.multimodal_gen.runtime.server_args import ServerArgs
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

logger = init_logger(__name__)


class SenseNovaU1PromptEnhancementStage(PipelineStage):
    def __init__(self, pe_client: SenseNovaU1PEClient):
        super().__init__()
        self.pe_client = pe_client

    def forward(self, batch: Req, server_args: ServerArgs) -> Req:
        del server_args
        options = SenseNovaU1GenerationOptions.from_batch(batch)
        if not options.use_pe:
            return batch
        if self.pe_client is None:
            raise RuntimeError(
                "Prompt enhancement was requested but no PE backend is configured."
            )

        # batch.prompt is always a single str for SenseNova-U1, never list[str]:
        # SenseNovaU1PipelineConfig.supports_dynamic_batching() is False (the
        # only mechanism that would merge multiple prompts into one Req), and
        # SenseNovaU1GenerationStage hardcodes batch_size=1 with no list
        # handling of its own -- a list would already break there.
        batch.prompt = self.pe_client.enhance(
            system_prompt=DEFAULT_PE_SYSTEM_PROMPT,
            user_prompt=batch.prompt,
        )
        logger.debug("SenseNova-U1 PE enhanced prompt: %s", batch.prompt)
        # Threaded through to SenseNovaU1GenerationStage, which surfaces it in
        # OutputBatch.usage so API callers see it without scraping server logs.
        batch.extra.setdefault(SENSENOVA_U1_REQUEST_EXTRA_KEY, {})[
            "enhanced_prompt"
        ] = batch.prompt
        return batch
