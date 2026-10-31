# SPDX-License-Identifier: Apache-2.0
"""Text conditioning for Fibo 2.0, which shares one checkpoint between both paths.

`ImageEncodingStage` encodes a prompt the way an editing model needs: Qwen3-VL
sees the images to edit next to the prompt and returns the bundles the
transformer consumes. It returns early when a request carries no image, which is
correct for models whose checkpoints ship a separate generation pipeline but not
here: sglang resolves the pipeline class from `model_index.json`, and Fibo 2.0's
single checkpoint serves generation and editing alike.

So this stage keeps `ImageEncodingStage` for the editing path and falls back to
the standard text-only encoding when the request has no image. Both paths return
the same six bundles because the bundle assembly lives in
`BriaFibo2PipelineConfig.postprocess_text_funcs`.
"""

from __future__ import annotations

from typing import Optional

from sglang.multimodal_gen.runtime.managers.memory_managers.component_manager import (
    ComponentUse,
)
from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import Req
from sglang.multimodal_gen.runtime.pipelines_core.stages.image_encoding import (
    ImageEncodingStage,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.text_encoding import (
    TextEncodingStage,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.validators import (
    VerificationResult,
)
from sglang.multimodal_gen.runtime.server_args import ServerArgs
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

logger = init_logger(__name__)


class BriaFibo2ConditionEncodingStage(ImageEncodingStage):
    """Encode the prompt with the images to edit, or as plain text without them."""

    def __init__(self, image_processor, text_encoder, tokenizer) -> None:
        super().__init__(image_processor=image_processor, text_encoder=text_encoder)
        # Built lazily so a checkpoint that never generates pays nothing for it.
        self._text_stage: Optional[TextEncodingStage] = None
        self._tokenizer = tokenizer

    def _resolved_text_stage(self) -> TextEncodingStage:
        if self._text_stage is None:
            self._text_stage = TextEncodingStage(
                text_encoders=[self.text_encoder], tokenizers=[self._tokenizer]
            )
        return self._text_stage

    def component_uses(
        self, server_args: ServerArgs, stage_name: str | None = None
    ) -> list[ComponentUse]:
        uses = list(super().component_uses(server_args, stage_name))
        stage_name = self._component_stage_name(stage_name)
        if not any(use.component_name == "tokenizer" for use in uses):
            uses.append(ComponentUse(stage_name, "tokenizer"))
        return uses

    def forward(self, batch: Req, server_args: ServerArgs) -> Req:
        if batch.condition_image is None:
            return self._resolved_text_stage().forward(batch, server_args)
        return super().forward(batch, server_args)

    def verify_input(self, batch: Req, server_args: ServerArgs) -> VerificationResult:
        if batch.condition_image is None:
            return self._resolved_text_stage().verify_input(batch, server_args)
        return super().verify_input(batch, server_args)

    def verify_output(self, batch: Req, server_args: ServerArgs) -> VerificationResult:
        if batch.condition_image is None:
            return self._resolved_text_stage().verify_output(batch, server_args)
        return super().verify_output(batch, server_args)

    def build_dedup_fingerprint(self, batch: Req, server_args: ServerArgs):
        if batch.condition_image is None:
            return self._resolved_text_stage().build_dedup_fingerprint(
                batch, server_args
            )
        return super().build_dedup_fingerprint(batch, server_args)


__all__ = ["BriaFibo2ConditionEncodingStage"]
