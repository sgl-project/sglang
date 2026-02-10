# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import inspect
from types import SimpleNamespace
from typing import Any

from diffusers.schedulers.scheduling_flow_match_euler_discrete import (
    FlowMatchEulerDiscreteScheduler,
)

from .base import AdapterResult
from .contracts import RolloutRequest
from .sd3_pipeline_with_logprob import (
    build_rollout_metadata_from_sd3_outputs,
    pipeline_with_logprob,
)


class SD3LogprobAdapter:
    name = "sd3"

    def can_handle(self, pipe: Any, request: RolloutRequest) -> bool:
        mode = str(request.get("mode", "")).lower()
        if mode and mode != "logprob_rollout":
            return False

        class_name = pipe.__class__.__name__.lower()
        if "stablediffusion3" not in class_name:
            return False

        scheduler = getattr(pipe, "scheduler", None)
        return isinstance(scheduler, FlowMatchEulerDiscreteScheduler)

    def run(
        self,
        pipe: Any,
        batch: Any,
        kwargs: dict[str, Any],
        request: RolloutRequest,
    ) -> AdapterResult:
        del batch

        params = request.get("params", {}) or {}
        noise_level = float(params.get("noise_level", 0.7))
        return_prev_sample_mean = bool(params.get("return_prev_latents_mean", False))

        signature = inspect.signature(pipeline_with_logprob)
        valid_keys = set(signature.parameters.keys()) - {"self"}
        call_kwargs = {k: v for k, v in kwargs.items() if k in valid_keys}
        call_kwargs["noise_level"] = noise_level
        call_kwargs["return_prev_sample_mean"] = return_prev_sample_mean

        output = pipeline_with_logprob(pipe, **call_kwargs)

        if return_prev_sample_mean:
            image, all_latents, all_log_probs, all_prev_latents_mean = output
        else:
            image, all_latents, all_log_probs = output
            all_prev_latents_mean = None

        rollout_metadata = build_rollout_metadata_from_sd3_outputs(
            pipe=pipe,
            all_latents=all_latents,
            all_log_probs=all_log_probs,
            all_prev_latents_mean=all_prev_latents_mean,
            num_inference_steps=int(call_kwargs.get("num_inference_steps", 28)),
        )

        # Mimic diffusers output object so existing extraction logic can be reused.
        return AdapterResult(
            output=SimpleNamespace(images=image),
            rollout_metadata=rollout_metadata,
        )
