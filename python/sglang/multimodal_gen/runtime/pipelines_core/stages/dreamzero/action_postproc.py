# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from typing import Any

import torch

from sglang.multimodal_gen.runtime.managers.dreamzero_session_cache import (
    session_metadata_from_batch,
)
from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import Req
from sglang.multimodal_gen.runtime.pipelines_core.stages.base import PipelineStage
from sglang.multimodal_gen.runtime.pipelines_core.stages.validators import (
    VerificationResult,
)
from sglang.multimodal_gen.runtime.server_args import ServerArgs


class DreamZeroActionUnnormalizeStage(PipelineStage):
    """Convert normalized DreamZero action chunks back to environment space."""

    def __init__(self, unapply: Any | None = None):
        super().__init__()
        self.unapply = unapply

    def verify_input(self, batch: Req, server_args: ServerArgs) -> VerificationResult:
        result = VerificationResult()
        result.add_check(
            "dreamzero_action_pred",
            getattr(batch, "dreamzero_action_pred", None),
            torch.is_tensor,
        )
        return result

    def forward(self, batch: Req, server_args: ServerArgs) -> Req:
        normalized_action = batch.dreamzero_action_pred.float()
        session_metadata_kind = "slot_pool"
        unapply = self.unapply or batch.extra.get("dreamzero_unapply")
        if unapply is None:
            batch.output = normalized_action
            batch.dreamzero_session_metadata = session_metadata_from_batch(batch)
            batch.dreamzero_session_metadata_kind = session_metadata_kind
            return batch

        try:
            from tianshou.data import Batch

            action_batch = Batch(normalized_action=normalized_action)
            obs = getattr(batch, "dreamzero_original_obs", None)
            result = unapply(action_batch, obs=obs)
            if isinstance(result, Batch) and hasattr(result, "action"):
                batch.output = result.action
            else:
                batch.output = result
        except ImportError:
            batch.output = unapply(normalized_action)
        batch.dreamzero_session_metadata = session_metadata_from_batch(batch)
        batch.dreamzero_session_metadata_kind = session_metadata_kind
        return batch
