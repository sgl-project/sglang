# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import time
from collections.abc import Mapping
from typing import Any

import numpy as np
import torch

from sglang.multimodal_gen.runtime.managers.dreamzero_session_cache import (
    normalize_batched_session_fields,
    record_session_timing,
)
from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import Req
from sglang.multimodal_gen.runtime.pipelines_core.stages.base import PipelineStage
from sglang.multimodal_gen.runtime.pipelines_core.stages.dreamzero.utils import (
    infer_dreamzero_model_input_batch_size,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.validators import (
    VerificationResult,
)
from sglang.multimodal_gen.runtime.server_args import ServerArgs


class DreamZeroObsPrepStage(PipelineStage):
    """Prepare DreamZero request-local observation inputs.

    The stage accepts already-normalized model inputs via
    ``Req.extra["dreamzero_normalized_input"]``.
    """

    def verify_output(self, batch: Req, server_args: ServerArgs) -> VerificationResult:
        result = VerificationResult()
        result.add_check(
            "dreamzero_inputs",
            batch.dreamzero_inputs,
            lambda value: isinstance(value, dict),
        )
        return result

    @staticmethod
    def _to_tensor_tree(value: Any) -> Any:
        if isinstance(value, np.ndarray):
            return torch.from_numpy(value)
        if isinstance(value, Mapping):
            return {
                key: DreamZeroObsPrepStage._to_tensor_tree(item)
                for key, item in value.items()
            }
        if isinstance(value, list):
            return [DreamZeroObsPrepStage._to_tensor_tree(item) for item in value]
        if isinstance(value, tuple):
            return tuple(DreamZeroObsPrepStage._to_tensor_tree(item) for item in value)
        return value

    def forward(self, batch: Req, server_args: ServerArgs) -> Req:
        normalized_input = batch.extra.get("dreamzero_normalized_input")

        if normalized_input is None:
            raise ValueError(
                "DreamZero request requires 'dreamzero_normalized_input' in Req.extra"
            )

        if not isinstance(normalized_input, Mapping):
            raise TypeError("DreamZero normalized input must be a mapping")

        model_inputs = self._to_tensor_tree(dict(normalized_input))
        normalize_start = time.perf_counter()
        target_dtype = torch.bfloat16
        if not server_args.pipeline_config.disable_autocast:
            for key, value in list(model_inputs.items()):
                if torch.is_tensor(value) and value.dtype == torch.float32:
                    model_inputs[key] = value.to(dtype=target_dtype)

        batch.dreamzero_inputs = model_inputs
        batch_size = infer_dreamzero_model_input_batch_size(model_inputs)
        session_ids, reset_mask = normalize_batched_session_fields(
            session_ids=batch.extra.get("dreamzero_session_ids"),
            reset_mask=batch.extra.get("dreamzero_reset_mask"),
            batch_size=batch_size,
        )
        batch.dreamzero_session_ids = session_ids
        batch.dreamzero_reset_mask = reset_mask
        record_session_timing(
            batch,
            "session_normalize_ms",
            (time.perf_counter() - normalize_start) * 1000,
        )
        return batch
