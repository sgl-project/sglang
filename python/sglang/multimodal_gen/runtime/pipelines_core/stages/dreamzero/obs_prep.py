# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import time
from collections.abc import Callable, Mapping
from typing import Any

import torch

from sglang.multimodal_gen.runtime.managers.dreamzero_session_cache import (
    normalize_batched_session_fields,
    record_session_timing,
)
from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import Req
from sglang.multimodal_gen.runtime.pipelines_core.stages.base import PipelineStage
from sglang.multimodal_gen.runtime.pipelines_core.stages.validators import (
    VerificationResult,
)
from sglang.multimodal_gen.runtime.server_args import ServerArgs


class DreamZeroObsPrepStage(PipelineStage):
    """Prepare DreamZero request-local observation inputs.

    The stage accepts already-normalized model inputs via
    ``Req.extra["dreamzero_normalized_input"]``. For integration probes that keep
    an external transform object around, ``obs_transform`` may be supplied and
    will be called with ``Req.extra["dreamzero_obs"]``.
    """

    def __init__(self, obs_transform: Callable[[Any], Mapping[str, Any]] | None = None):
        super().__init__()
        self.obs_transform = obs_transform

    def verify_output(self, batch: Req, server_args: ServerArgs) -> VerificationResult:
        result = VerificationResult()
        result.add_check(
            "dreamzero_inputs",
            getattr(batch, "dreamzero_inputs", None),
            lambda value: isinstance(value, dict),
        )
        return result

    @staticmethod
    def _infer_model_input_batch_size(model_inputs: Mapping[str, Any]) -> int:
        for key in (
            "images",
            "videos",
            "state",
            "text",
            "text_negative",
            "clip_feature",
            "y",
            "latent_video",
        ):
            value = model_inputs.get(key)
            if torch.is_tensor(value) and value.ndim > 0:
                return int(value.shape[0])
        prompt_embs = model_inputs.get("prompt_embs")
        if isinstance(prompt_embs, (list, tuple)):
            for value in prompt_embs:
                if torch.is_tensor(value) and value.ndim > 0:
                    return int(value.shape[0])
        raise ValueError("Cannot infer DreamZero batch size from normalized input")

    def forward(self, batch: Req, server_args: ServerArgs) -> Req:
        normalized_input = batch.extra.get("dreamzero_normalized_input")
        original_obs = batch.extra.get("dreamzero_original_obs")

        if normalized_input is None:
            obs = batch.extra.get("dreamzero_obs")
            if obs is None:
                raise ValueError(
                    "DreamZero request requires 'dreamzero_normalized_input' or "
                    "'dreamzero_obs' in Req.extra"
                )
            original_obs = obs
            if self.obs_transform is None:
                normalized_input = obs
            else:
                normalized_input = self.obs_transform(obs)

        if not isinstance(normalized_input, Mapping):
            if hasattr(normalized_input, "__getstate__"):
                normalized_input = normalized_input.__getstate__()
            else:
                raise TypeError(
                    "DreamZero normalized input must be a mapping or expose "
                    "__getstate__()"
                )

        model_inputs = dict(normalized_input)
        normalize_start = time.perf_counter()
        target_dtype = torch.bfloat16
        if not server_args.pipeline_config.disable_autocast:
            for key, value in list(model_inputs.items()):
                if torch.is_tensor(value) and value.dtype == torch.float32:
                    model_inputs[key] = value.to(dtype=target_dtype)

        batch.dreamzero_inputs = model_inputs
        batch.dreamzero_original_obs = original_obs
        batch_size = self._infer_model_input_batch_size(model_inputs)
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
