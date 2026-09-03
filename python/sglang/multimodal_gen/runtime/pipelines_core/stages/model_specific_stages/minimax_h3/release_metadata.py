# SPDX-License-Identifier: Apache-2.0
"""Public MiniMax H3 model-index admission contract."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping

from sglang.multimodal_gen.configs.sample.sampling_params import QUALITY_LEVELS
from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import Req
from sglang.multimodal_gen.runtime.pipelines_core.stages.base import PipelineStage
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.resolved_plan import (
    minimax_h3_plan_from_batch,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.task_profiles import (
    canonical_minimax_h3_task,
    partition_for_task,
)
from sglang.multimodal_gen.runtime.server_args import ServerArgs

_MINIMAX_H3_QUALITY_WORKLOAD = {
    "task": "t2va",
    "width": 1344,
    "height": 768,
    "fps": 24,
    "frame_count": 124,
    "num_inference_steps": 50,
    "flow_shift": 12.0,
    "audio_flow_shift": 3.0,
}


def _string_list(value: Any, path: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{path} must be a non-empty list")
    values = tuple(value)
    if any(not isinstance(item, str) or not item for item in values):
        raise ValueError(f"{path} must contain non-empty strings")
    if len(set(values)) != len(values):
        raise ValueError(f"{path} must not contain duplicates")
    return values


@dataclass(frozen=True)
class MiniMaxH3ReleaseMetadata:
    schema_version: int
    partition: str
    tasks: tuple[str, ...]
    task_aliases: Mapping[str, str]
    video_sigma_shift: float
    audio_sigma_shift: float

    @classmethod
    def from_model_index(
        cls, model_index: Mapping[str, Any]
    ) -> MiniMaxH3ReleaseMetadata:
        raw = model_index.get("_minimax_h3")
        if not isinstance(raw, Mapping):
            raise ValueError("model_index.json._minimax_h3 must be an object")
        if raw.get("schema_version") != 1:
            raise ValueError("model_index.json._minimax_h3.schema_version must be 1")
        partition = raw.get("partition")
        if partition not in {"fl2va", "ref2va"}:
            raise ValueError(
                "model_index.json._minimax_h3.partition must be one of fl2va, ref2va"
            )
        tasks = _string_list(raw.get("tasks"), "model_index.json._minimax_h3.tasks")
        aliases = raw.get("task_aliases", {})
        if not isinstance(aliases, Mapping) or any(
            not isinstance(key, str)
            or not key
            or not isinstance(value, str)
            or not value
            for key, value in aliases.items()
        ):
            raise ValueError(
                "model_index.json._minimax_h3.task_aliases must map strings to strings"
            )
        scales = raw.get("sigma_shift_scales")
        if not isinstance(scales, Mapping):
            raise ValueError(
                "model_index.json._minimax_h3.sigma_shift_scales must be an object"
            )
        try:
            video_sigma = float(scales["video"])
            audio_sigma = float(scales["audio"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                "model_index.json._minimax_h3.sigma_shift_scales requires numeric "
                "video and audio values"
            ) from exc
        metadata = cls(
            schema_version=1,
            partition=partition,
            tasks=tasks,
            task_aliases=dict(aliases),
            video_sigma_shift=video_sigma,
            audio_sigma_shift=audio_sigma,
        )
        for task in metadata.tasks:
            if canonical_minimax_h3_task(task) != task:
                raise ValueError(
                    f"tasks must contain canonical task names, got {task!r}"
                )
            if partition_for_task(task) != partition:
                raise ValueError(
                    f"task {task!r} does not belong to partition {partition!r}"
                )
        for alias, target in metadata.task_aliases.items():
            if target not in metadata.tasks:
                raise ValueError(
                    f"task alias {alias!r} targets undeclared task {target!r}"
                )
            if canonical_minimax_h3_task(alias) != target:
                raise ValueError(
                    f"unsupported task alias mapping {alias!r} -> {target!r}"
                )
        return metadata

    @property
    def sigma_shift_scales(self) -> dict[str, float]:
        return {"video": self.video_sigma_shift, "audio": self.audio_sigma_shift}

    def canonical_task(self, task: str) -> str:
        normalized = task.strip().lower()
        canonical = self.task_aliases.get(normalized, normalized)
        if canonical not in self.tasks:
            raise ValueError(
                f"task {task!r} is not served by MiniMax H3 partition {self.partition!r}; "
                f"supported tasks: {list(self.tasks)!r}"
            )
        if partition_for_task(canonical) != self.partition:
            raise ValueError(
                f"task {task!r} resolves outside partition {self.partition!r}"
            )
        return canonical


class MiniMaxH3PartitionAdmissionStage(PipelineStage):
    def __init__(self, metadata: MiniMaxH3ReleaseMetadata) -> None:
        super().__init__()
        self.metadata = metadata

    def forward(self, batch: Req, server_args: ServerArgs) -> Req:
        task = None if batch.sampling_params is None else batch.sampling_params.task
        if not isinstance(task, str) or not task.strip():
            raise ValueError("MiniMax H3 request task must be a non-empty string")
        self.metadata.canonical_task(task)
        if batch.num_inference_steps < 2:
            raise ValueError(
                "MiniMax H3 requires num_inference_steps >= 2 because its "
                "video/audio sigma schedules include both interval endpoints"
            )
        quality = getattr(batch.sampling_params, "quality", "lossless")
        if quality not in QUALITY_LEVELS:
            raise ValueError(
                f"quality must be one of {list(QUALITY_LEVELS)}, got {quality!r}"
            )
        high_quality = quality == "high"
        if high_quality and not batch.is_warmup:
            server_args.pipeline_config.validate_quality_deployment(server_args)
            plan = minimax_h3_plan_from_batch(batch)
            if plan is None:
                raise ValueError(
                    'MiniMax-H3 quality="high" requires a resolved request plan'
                )
            shape = plan.shape
            actual = {
                "task": plan.task,
                "width": int(shape["width"]),
                "height": int(shape["height"]),
                "fps": int(shape["fps"]),
                "frame_count": int(shape["frame_count"]),
                "num_inference_steps": int(batch.num_inference_steps),
                "flow_shift": float(
                    plan.flow_shift
                    if plan.flow_shift is not None
                    else plan.default_flow_shift
                ),
                "audio_flow_shift": float(
                    plan.audio_flow_shift
                    if plan.audio_flow_shift is not None
                    else plan.default_audio_flow_shift
                ),
            }
            exact_fields = (
                "task",
                "width",
                "height",
                "fps",
                "frame_count",
                "num_inference_steps",
            )
            exact = all(
                actual[name] == _MINIMAX_H3_QUALITY_WORKLOAD[name]
                for name in exact_fields
            )
            shifts = math.isclose(
                actual["flow_shift"],
                _MINIMAX_H3_QUALITY_WORKLOAD["flow_shift"],
                abs_tol=1e-9,
            ) and math.isclose(
                actual["audio_flow_shift"],
                _MINIMAX_H3_QUALITY_WORKLOAD["audio_flow_shift"],
                abs_tol=1e-9,
            )
            if not exact or not shifts:
                raise ValueError(
                    'MiniMax-H3 quality="high" is validated only for '
                    f"{_MINIMAX_H3_QUALITY_WORKLOAD}; got {actual}"
                )
        return batch


__all__ = ["MiniMaxH3PartitionAdmissionStage", "MiniMaxH3ReleaseMetadata"]
