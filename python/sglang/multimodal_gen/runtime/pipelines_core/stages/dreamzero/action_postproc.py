# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import torch

from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import OutputBatch, Req
from sglang.multimodal_gen.runtime.pipelines_core.stages.base import PipelineStage
from sglang.multimodal_gen.runtime.pipelines_core.stages.validators import (
    VerificationResult,
)
from sglang.multimodal_gen.runtime.server_args import ServerArgs


class DreamZeroActionOutputStage(PipelineStage):
    """Publish normalized DreamZero action chunks as pipeline output."""

    def verify_input(self, batch: Req, server_args: ServerArgs) -> VerificationResult:
        result = VerificationResult()
        result.add_check(
            "dreamzero_action_pred",
            batch.dreamzero_action_pred,
            torch.is_tensor,
        )
        return result

    def forward(self, batch: Req, server_args: ServerArgs) -> OutputBatch:
        normalized_action = batch.dreamzero_action_pred.float()
        action_payload = {
            "request_id": batch.request_id,
            "actions": normalized_action.detach().cpu().numpy(),
            "parameters": {
                "num_inference_steps": server_args.pipeline_config.default_num_inference_steps
            },
        }

        return OutputBatch(
            output=[action_payload],
            action_pred=normalized_action,
            metrics=batch.metrics,
        )
