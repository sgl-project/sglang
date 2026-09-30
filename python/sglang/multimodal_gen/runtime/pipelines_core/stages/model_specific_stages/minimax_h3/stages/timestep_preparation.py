# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import json
import math
from collections.abc import Mapping
from pathlib import Path

import torch

from sglang.multimodal_gen import envs
from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import Req
from sglang.multimodal_gen.runtime.pipelines_core.stages.base import PipelineStage
from sglang.multimodal_gen.runtime.pipelines_core.stages.validators import (
    StageValidators as V,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.validators import (
    VerificationResult,
)
from sglang.multimodal_gen.runtime.server_args import ServerArgs

from ..constants import MINIMAX_H3_SIGMAS_EXTRA_KEY


class MiniMaxH3TimestepPreparationStage(PipelineStage):
    deduplicated_tensor_tree_output_fields = ("timesteps", "sigmas")
    deduplicated_extra_tensor_tree_output_keys = (MINIMAX_H3_SIGMAS_EXTRA_KEY,)

    def __init__(
        self,
        sigma_shift_scales=None,
        sigma_rungs: tuple[int, ...] | None = None,
    ) -> None:
        super().__init__()
        # Per-model sigma shift override (model_index.json "_minimax_h3" release
        # block, sigma_shift_scales): the schedule constants are a MODEL
        # serving contract — fl2va and ref2va use video 12 / audio 3 by default.
        self.sigma_shift_scales = sigma_shift_scales
        # A distilled checkpoint's trained ladder replaces the uniform grid and
        # pins both the grid size and the model shifts.
        self.sigma_rungs = sigma_rungs
        self._pdd_config = None
        pdd_heads = envs.SGLANG_DIFFUSION_MINIMAX_H3_PDD_HEADS
        if pdd_heads:
            from safetensors import safe_open

            self._pdd_config = json.loads(
                Path(pdd_heads).with_name("pdd_config.json").read_text()
            )
            with safe_open(pdd_heads, "pt") as f:
                steps = f.get_slice("video_out.weight").get_shape()[0]
            if self._pdd_config["num_inference_steps"] != steps + 1:
                raise ValueError("MiniMax-H3 PDD config does not match the fused heads")

    def forward(self, batch: Req, server_args: ServerArgs) -> Req:
        from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.resolved_plan import (
            minimax_h3_plan_from_batch,
        )

        plan = minimax_h3_plan_from_batch(batch)
        if plan is None:
            raise NotImplementedError(
                f"{self.__class__.__name__} is a MiniMax H3 contract stage "
                "and has no implementation yet."
            )
        self._generate_sigmas_from_plan(batch, plan)
        self._apply_pdd_schedule(batch)
        self._publish_native_timestep_state(batch)
        return batch

    def build_dedup_fingerprint(self, batch: Req, server_args: ServerArgs):
        from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.resolved_plan import (
            minimax_h3_plan_from_batch,
        )

        plan = minimax_h3_plan_from_batch(batch)
        if plan is None:
            raise NotImplementedError(
                f"{self.__class__.__name__} is a MiniMax H3 contract stage "
                "and has no implementation yet."
            )
        return (
            batch.num_inference_steps,
            batch.is_warmup,
            plan.flow_shift,
            plan.audio_flow_shift,
            plan.default_flow_shift,
            plan.default_audio_flow_shift,
            self.freeze_for_dedup(self.sigma_shift_scales),
            self.sigma_rungs,
        )

    def _apply_pdd_schedule(self, batch: Req) -> None:
        if self._pdd_config is None:
            return
        from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.time_request import (
            minimax_h3_time_shift_sigmas,
        )

        config = self._pdd_config
        sigmas = batch.extra[MINIMAX_H3_SIGMAS_EXTRA_KEY]
        for modality in ("video", "audio"):
            expected = minimax_h3_time_shift_sigmas(
                num_steps=config["num_inference_steps"],
                shift_scale=config[f"{modality}_shift"],
            )
            if batch.is_warmup:
                # Warmup may run fewer steps, but must use the same intervals
                # as serving rather than rescaling a shorter grid to [1, 0].
                sigmas[modality] = expected[: max(2, batch.num_inference_steps)]
                continue
            actual = sigmas[modality]
            if len(actual) != len(expected) or any(
                not math.isclose(a, b, rel_tol=1e-6, abs_tol=1e-7)
                for a, b in zip(actual, expected)
            ):
                raise ValueError(
                    f"MiniMax-H3 PDD requires num_inference_steps="
                    f"{config['num_inference_steps']} and {modality} shift="
                    f"{config[f'{modality}_shift']} to match the fused heads"
                )

    @staticmethod
    def _publish_native_timestep_state(batch: Req) -> None:
        sigmas = batch.extra.get(MINIMAX_H3_SIGMAS_EXTRA_KEY)
        if not isinstance(sigmas, dict):
            raise ValueError("MiniMax H3 sigma schedules must be a mapping")
        video_sigmas = sigmas.get("video")
        audio_sigmas = sigmas.get("audio")
        if (
            not isinstance(video_sigmas, list)
            or not isinstance(audio_sigmas, list)
            or len(video_sigmas) != len(audio_sigmas)
            or len(video_sigmas) < 2
        ):
            raise ValueError(
                "MiniMax H3 video/audio sigma schedules must be equal-length lists"
            )
        batch.sigmas = list(video_sigmas)
        batch.timesteps = torch.tensor(
            [1.0 - float(sigma) for sigma in video_sigmas[:-1]],
            dtype=torch.float32,
        )

    def _generate_sigmas_from_plan(self, batch: Req, plan) -> None:
        """Generate the fixed per-modality float32 time-shift schedules."""
        from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.time_request import (
            minimax_h3_time_shift_sigmas,
        )

        if MINIMAX_H3_SIGMAS_EXTRA_KEY in batch.extra:
            return
        requested_num_steps = getattr(batch, "num_inference_steps", None)
        if requested_num_steps is None:
            sampling = getattr(batch, "sampling_params", None)
            requested_num_steps = getattr(sampling, "num_inference_steps", None)
        if requested_num_steps is None:
            requested_num_steps = 50
        if (
            isinstance(requested_num_steps, bool)
            or not isinstance(requested_num_steps, int)
            or requested_num_steps <= 0
        ):
            raise ValueError(
                "num_inference_steps must be a positive integer, got "
                f"{requested_num_steps!r}"
            )

        model_scales = self.sigma_shift_scales
        if model_scales is not None and not isinstance(model_scales, Mapping):
            raise ValueError("model sigma_shift_scales must be an object")

        def resolved_scale(
            *, modality: str, request_value, task_default: float
        ) -> float:
            value = request_value
            source = (
                f"request {'flow_shift' if modality == 'video' else 'audio_flow_shift'}"
            )
            if value is None and model_scales is not None:
                value = model_scales.get(modality)
                source = f"model sigma_shift_scales.{modality}"
            if value is None:
                value = task_default
                source = (
                    "task default "
                    f"{'flow_shift' if modality == 'video' else 'audio_flow_shift'}"
                )
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"{source} must be a positive finite number")
            scale = float(value)
            if not math.isfinite(scale) or scale <= 0.0:
                raise ValueError(f"{source} must be a positive finite number")
            return scale

        scales = {
            "video": resolved_scale(
                modality="video",
                request_value=plan.flow_shift,
                task_default=plan.default_flow_shift,
            ),
            "audio": resolved_scale(
                modality="audio",
                request_value=plan.audio_flow_shift,
                task_default=plan.default_audio_flow_shift,
            ),
        }
        if self.sigma_rungs is not None:
            batch.extra[MINIMAX_H3_SIGMAS_EXTRA_KEY] = self._rung_sigmas(
                num_steps=requested_num_steps,
                is_warmup=batch.is_warmup,
                scales=scales,
            )
            return
        sigmas: dict[str, list[float]] = {}
        for modality in ("video", "audio"):
            sigmas[modality] = minimax_h3_time_shift_sigmas(
                num_steps=requested_num_steps,
                shift_scale=scales[modality],
            )
        batch.extra[MINIMAX_H3_SIGMAS_EXTRA_KEY] = sigmas

    def _rung_sigmas(
        self, *, num_steps: int, is_warmup: bool, scales: dict[str, float]
    ) -> dict[str, list[float]]:
        """Shift the trained ladder; warmup keeps a prefix of the served grid."""
        from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.time_request import (
            minimax_h3_rung_sigmas,
        )

        grid_points = len(self.sigma_rungs) + 1
        if not is_warmup and num_steps != grid_points:
            raise ValueError(
                f"This checkpoint was distilled on {len(self.sigma_rungs)} rungs, "
                f"so num_inference_steps must be {grid_points} sigma grid points; "
                f"got {num_steps}"
            )
        trained = {
            modality: float(self.sigma_shift_scales[modality])
            for modality in ("video", "audio")
        }
        if scales != trained:
            raise ValueError(
                "This checkpoint's trained ladder only holds at its trained "
                f"shifts {trained}; the request resolved to {scales}. Leave "
                "flow_shift and audio_flow_shift unset."
            )
        keep = min(max(2, num_steps), grid_points)
        return {
            modality: minimax_h3_rung_sigmas(
                rungs=self.sigma_rungs, shift_scale=scales[modality]
            )[:keep]
            for modality in ("video", "audio")
        }

    def verify_input(self, batch: Req, server_args: ServerArgs) -> VerificationResult:
        result = VerificationResult()
        result.add_check(
            "num_inference_steps", batch.num_inference_steps, V.positive_int
        )
        result.add_check("timesteps", batch.timesteps, V.none_or_tensor)
        result.add_check("sigmas", batch.sigmas, V.none_or_list)
        return result

    def verify_output(self, batch: Req, server_args: ServerArgs) -> VerificationResult:
        result = VerificationResult()
        result.add_check("timesteps", batch.timesteps, [V.is_tensor, V.with_dims(1)])
        result.add_check("sigmas", batch.sigmas, V.list_not_empty)
        result.add_check(
            MINIMAX_H3_SIGMAS_EXTRA_KEY,
            batch.extra.get(MINIMAX_H3_SIGMAS_EXTRA_KEY),
            lambda value: isinstance(value, Mapping),
        )
        return result


__all__ = ["MiniMaxH3TimestepPreparationStage"]
