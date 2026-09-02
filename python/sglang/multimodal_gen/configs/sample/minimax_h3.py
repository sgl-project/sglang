# SPDX-License-Identifier: Apache-2.0
import math
import os
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import msgspec

from sglang.multimodal_gen.configs.sample.sampling_params import SamplingParams

_MINIMAX_H3_MAX_SIGNED_SEED = (1 << 63) - 1


def _optional_unit_float(value: Any, field_name: str) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field_name} must be a number")
    out = float(value)
    if out < 0.0 or out > 1.0:
        raise ValueError(f"{field_name} must be in [0, 1]")
    return out


def _optional_positive_finite_float(value: Any, field_name: str) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field_name} must be a number")
    out = float(value)
    if not math.isfinite(out) or out <= 0.0:
        raise ValueError(f"{field_name} must be a positive finite number")
    return out


@dataclass
class MiniMaxH3SamplingParams(SamplingParams):
    height: int = 512
    width: int = 896
    num_inference_steps: int = 50
    num_frames: int = field(default=1, init=False)
    fps: int = field(default=24, init=False)
    negative_prompt: None = field(default=None, init=False)
    guidance_scale: float = field(default=1.0, init=False)
    guidance_scale_2: None = field(default=None, init=False)
    true_cfg_scale: None = field(default=None, init=False)
    guidance_rescale: float = field(default=0.0, init=False)
    cfg_normalization: float = field(default=0.0, init=False)
    imgvid_cond_noise_aug_for_inference: float | None = None
    audio_cond_noise_aug_for_inference: float | None = None
    task: str | None = None
    conditions: list[dict[str, Any]] | None = None
    target: dict[str, Any] | None = None
    audio_flow_shift: float | None = None
    output_mode: str | None = field(
        default=None,
        metadata={"batch_sig_exclude": True},
    )

    @classmethod
    def video_request_extra_fields(cls) -> frozenset[str]:
        return frozenset(
            {
                "task",
                "conditions",
                "target",
                "audio_flow_shift",
                "audio_guidance_scale",
                "quality",
                "output_mode",
                "imgvid_cond_noise_aug_for_inference",
                "audio_cond_noise_aug_for_inference",
            }
        )

    @staticmethod
    def _video_hooks():
        from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.video_adapter import (
            MiniMaxH3VideoModelAdapter,
        )

        return MiniMaxH3VideoModelAdapter()

    @classmethod
    def lower_video_request_kwargs(
        cls,
        request: Any,
        kwargs: dict[str, Any],
    ) -> dict[str, Any]:
        return cls._video_hooks().lower_video_request_kwargs(request, kwargs)

    def prepare_video_request_for_queue(self, req: Any) -> None:
        hooks = self._video_hooks()
        hooks.validate_sampling_params(self)
        hooks.prepare_for_queue_sync(req)

    def expand_video_request_outputs_for_queue(self, req: Any) -> list[Any]:
        """Use the same independent-seed grouped path in serve and generate."""

        from sglang.multimodal_gen.runtime.entrypoints.utils import (
            expand_request_outputs,
        )

        return expand_request_outputs(req)

    def prepare_synthetic_warmup_request_for_queue(
        self, req: Any, server_args: Any
    ) -> None:
        """Lower generic warmup into one valid native partition request.

        This intentionally calls the existing pre-queue resolver directly
        instead of the public video admission hook: synthetic warmup disables
        file delivery, while the public H3 contract correctly requires it.
        """
        selected_variant = getattr(server_args, "model_variant", None)
        if selected_variant is not None:
            selected_partition = str(selected_variant).strip().lower()
        else:
            selected_path = server_args.model_subfolder or server_args.model_path
            selected_partition = os.path.basename(
                os.path.normpath(str(selected_path))
            ).lower()

        if selected_partition == "ref2va":
            image_path = req.image_path
            if isinstance(image_path, list):
                if not image_path:
                    raise ValueError(
                        "MiniMax H3 Ref2VA synthetic warmup requires an image"
                    )
                image_path = image_path[0]
            if not isinstance(image_path, str) or not image_path:
                raise ValueError("MiniMax H3 Ref2VA synthetic warmup requires an image")
            task = "ref2va"
            conditions = [
                {
                    "type": "image",
                    "uri": image_path,
                    "role": "reference",
                }
            ]
        else:
            task = "t2va"
            conditions = []

        self.task = task
        self.conditions = conditions
        self.target = {
            "short_edge": 768,
            "aspect_ratio": "16:9",
            "duration_seconds": 5.0,
        }
        selected_seed = req.seed if isinstance(req.seed, int) else int(req.seed[0])
        req.extra.update(self.build_request_extra(_seed_override=int(selected_seed)))
        self._video_hooks().prepare_for_queue_sync(req)

    def project_video_queued_job_fields(self, req: Any) -> dict[str, str]:
        return self._video_hooks().project_queued_job_fields(req)

    def validate_video_final_outputs(
        self,
        output_paths: list[str],
        req: Any,
    ) -> dict[str, str]:
        return self._video_hooks().validate_final_outputs_sync(output_paths, req)

    def cleanup_video_request(self, req: Any) -> None:
        self._video_hooks().cleanup_request_sync(req)

    def _adjust(self, server_args) -> None:
        """Apply generic path/output adjustments without deriving time shape.

        The generic helper normally rewrites ``num_frames`` for temporal VAE
        alignment and GPU sharding.  MiniMax H3 resolves that shape from the
        canonical target during pre-queue admission, so those rewrites must
        not leak into the offline parameter object.  Keep the transport
        metadata at its internal sentinel values until pre-queue populates the
        actual batch shape.
        """

        super()._adjust(server_args)
        self.fps = 24
        self.num_frames = 1

    def _validate(self) -> None:
        self.fps = 24
        self.num_frames = 1
        if isinstance(self.target, Mapping):
            self.target = {
                field_name: self.target[field_name]
                for field_name in (
                    "short_edge",
                    "aspect_ratio",
                    "duration_seconds",
                )
                if field_name in self.target
            }
        super()._validate()
        _optional_positive_finite_float(self.flow_shift, "flow_shift")
        _optional_positive_finite_float(self.audio_flow_shift, "audio_flow_shift")
        if self.enable_frame_interpolation:
            raise ValueError(
                "MiniMax H3 does not support enable_frame_interpolation: the "
                "accepted delivery contract is the canonical 24 fps output"
            )
        if self.enable_upscaling:
            raise ValueError(
                "MiniMax H3 does not support enable_upscaling: the accepted "
                "delivery contract is the resolved target canvas"
            )
        if self.enable_teacache:
            raise ValueError(
                "MiniMax H3 does not support enable_teacache: its packed "
                "video/audio denoise loop has no lossless TeaCache contract"
            )
        if self.rollout:
            raise ValueError(
                "MiniMax H3 does not support rollout: its coupled video/audio "
                "scheduler has no SchedulerRLMixin contract"
            )
        if self.return_trajectory_latents or self.return_trajectory_decoded:
            raise ValueError(
                "MiniMax H3 does not support trajectory output for its coupled "
                "video/audio denoise state"
            )
        seeds = self.seed if isinstance(self.seed, list) else [self.seed]
        for seed in seeds:
            if seed > _MINIMAX_H3_MAX_SIGNED_SEED:
                raise ValueError(
                    "MiniMax H3 seed must not exceed the signed int64 maximum, "
                    f"got {seed}"
                )
        if (
            isinstance(self.seed, int)
            and self.seed + self.num_outputs_per_prompt - 1
            > _MINIMAX_H3_MAX_SIGNED_SEED
        ):
            raise ValueError(
                "MiniMax H3 scalar seed plus output index must fit the "
                "signed int64 upper bound"
            )

    def build_request_extra(self, *, _seed_override: int | None = None) -> dict:
        _optional_unit_float(
            self.imgvid_cond_noise_aug_for_inference,
            "imgvid_cond_noise_aug_for_inference",
        )
        _optional_unit_float(
            self.audio_cond_noise_aug_for_inference,
            "audio_cond_noise_aug_for_inference",
        )
        extra = super().build_request_extra()
        if self.task is not None:
            from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.request_validation import (
                minimax_h3_validate_canonical_request,
            )

            extra["minimax_h3_canonical_request"] = (
                minimax_h3_validate_canonical_request(
                    task=self.task,
                    prompt=self.prompt,
                    conditions=self.conditions,
                    target=self.target,
                    flow_shift=self.flow_shift,
                    audio_flow_shift=self.audio_flow_shift,
                    seed=(
                        _seed_override
                        if _seed_override is not None
                        else self.seed if isinstance(self.seed, int) else None
                    ),
                )
            )
        elif (
            self.conditions is not None
            or self.target is not None
            or self.flow_shift is not None
            or self.audio_flow_shift is not None
        ):
            raise ValueError(
                "task is required when conditions/target/flow_shift/"
                "audio_flow_shift are provided"
            )
        return extra

    def refresh_request_extra_after_output_expansion(self, req: Any) -> None:
        """Copy validated canonical identity with the selected scalar Req seed."""

        selected_seed = getattr(req, "seed", None)
        if isinstance(selected_seed, int):
            canonical_key = "minimax_h3_canonical_request"
            canonical = dict(req.extra[canonical_key])
            canonical["seed"] = selected_seed
            req.extra[canonical_key] = canonical
            resolved_plan_key = "minimax_h3_resolved_plan"
            resolved_plan = req.extra.get(resolved_plan_key)
            if resolved_plan is not None:
                req.extra[resolved_plan_key] = msgspec.structs.replace(
                    resolved_plan, seed=selected_seed
                )
        else:
            req.extra.update(self.build_request_extra())


@dataclass
class FastH3SamplingParams(MiniMaxH3SamplingParams):
    """FastH3: five sigma grid points, i.e. the four distilled DiT forwards."""

    num_inference_steps: int = 5

    def _validate(self) -> None:
        super()._validate()
        if self.num_inference_steps != 5:
            raise ValueError(
                "FastH3 is distilled for exactly five sigma grid points (four DiT "
                f"forwards); got num_inference_steps={self.num_inference_steps}. "
                "Use MiniMaxAI/MiniMax-H3 for other schedules."
            )
        if self.task is not None and self.task.strip().lower() != "t2va":
            raise ValueError(
                "FastH3 is distilled for t2va only; fl2va and ref2va were not "
                f"distilled (got task={self.task!r}). Use MiniMaxAI/MiniMax-H3 "
                "for those tasks."
            )


__all__ = ["FastH3SamplingParams", "MiniMaxH3SamplingParams"]
