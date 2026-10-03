# SPDX-License-Identifier: Apache-2.0
"""FastVideo's FastH3 inference contract (``fastvideo_inference.json``).

A distilled FastH3 export declares its trained noise ladder, modality shifts,
and VSA policy in this sidecar. Binding it onto the release metadata makes the
served computation the trained one; any disagreement with the checkpoint's
scheduler configs or with the release block fails the load.
"""

from __future__ import annotations

import dataclasses
import json
import math
import os
from typing import Any

import msgspec

from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.release_metadata import (
    MiniMaxH3ReleaseMetadata,
)

FASTH3_INFERENCE_CONTRACT_FILE = "fastvideo_inference.json"
_SCHEMA_VERSION = "fasth3-inference-contract-v1"
_SCHEDULER_CONFIG_FILES = {
    "video": os.path.join("scheduler", "scheduler_config.json"),
    "audio": os.path.join("audio_scheduler", "scheduler_config.json"),
}
_VSA_H3_BACKEND = "VIDEO_SPARSE_ATTN_H3"
# SGLang's VSA-H3 kernel serves the 64-token (4, 4, 4) tile geometry only.
_VSA_H3_TILE_SIZE = 64


class FastH3InferenceContract(msgspec.Struct, frozen=True):
    sigma_rungs: tuple[int, ...]
    video_shift: float
    audio_shift: float
    vsa_sparsity: float

    @classmethod
    def load(cls, model_dir: str) -> FastH3InferenceContract:
        path = os.path.join(model_dir, FASTH3_INFERENCE_CONTRACT_FILE)
        if not os.path.isfile(path):
            raise ValueError(
                f"FastH3 checkpoint at {model_dir} is missing "
                f"{FASTH3_INFERENCE_CONTRACT_FILE}, which defines its trained schedule"
            )
        with open(path, encoding="utf-8") as f:
            raw = json.load(f)
        return cls.from_dict(raw, scheduler_shifts=_read_scheduler_shifts(model_dir))

    @classmethod
    def from_dict(
        cls, raw: Any, *, scheduler_shifts: dict[str, float]
    ) -> FastH3InferenceContract:
        if not isinstance(raw, dict) or raw.get("schema_version") != _SCHEMA_VERSION:
            raise ValueError(
                f"FastH3 inference contract must declare schema_version "
                f"{_SCHEMA_VERSION!r}"
            )
        if raw.get("task") != "t2av":
            raise ValueError(
                f"FastH3 inference contract task must be 't2av', got {raw.get('task')!r}"
            )
        if raw.get("guidance_scale", 1.0) != 1.0:
            raise ValueError(
                "FastH3 inference contract declares CFG "
                f"(guidance_scale={raw['guidance_scale']!r}); H3 serves the "
                "CFG-distilled positive branch only"
            )
        return cls(
            sigma_rungs=_validated_rungs(raw),
            video_shift=_declared_shift(raw, "video", scheduler_shifts["video"]),
            audio_shift=_declared_shift(raw, "audio", scheduler_shifts["audio"]),
            vsa_sparsity=_validated_vsa_sparsity(raw),
        )

    def bind(self, metadata: MiniMaxH3ReleaseMetadata) -> MiniMaxH3ReleaseMetadata:
        if metadata.tasks != ("t2va",):
            raise ValueError(
                "FastH3 inference contract serves t2va only, but the release "
                f"declares tasks {list(metadata.tasks)!r}"
            )
        release_shifts = (metadata.video_sigma_shift, metadata.audio_sigma_shift)
        if release_shifts != (self.video_shift, self.audio_shift):
            raise ValueError(
                "model_index.json sigma_shift_scales (video, audio) = "
                f"{release_shifts} disagree with the checkpoint's trained shifts "
                f"{(self.video_shift, self.audio_shift)}"
            )
        return dataclasses.replace(
            metadata,
            sigma_rungs=self.sigma_rungs,
            vsa_sparsity=self.vsa_sparsity,
        )


def _read_scheduler_shifts(model_dir: str) -> dict[str, float]:
    shifts: dict[str, float] = {}
    for modality, rel_path in _SCHEDULER_CONFIG_FILES.items():
        path = os.path.join(model_dir, rel_path)
        if not os.path.isfile(path):
            raise ValueError(
                f"FastH3 checkpoint is missing {rel_path} under {model_dir}"
            )
        with open(path, encoding="utf-8") as f:
            shifts[modality] = _positive_finite(json.load(f).get("shift"), rel_path)
    return shifts


def _positive_finite(value: Any, source: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value <= 0
    ):
        raise ValueError(f"{source} must be a positive finite shift, got {value!r}")
    return float(value)


def _validated_rungs(raw: dict[str, Any]) -> tuple[int, ...]:
    rungs = raw.get("dmd_denoising_steps")
    if (
        not isinstance(rungs, list)
        or not rungs
        or any(type(rung) is not int or not 0 < rung <= 1000 for rung in rungs)
        or any(left <= right for left, right in zip(rungs, rungs[1:]))
    ):
        raise ValueError(
            "FastH3 inference contract dmd_denoising_steps must be strictly "
            f"decreasing integers in (0, 1000], got {rungs!r}"
        )
    num_forwards = len(rungs)
    if (
        raw.get("transformer_forwards") != num_forwards
        or raw.get("num_inference_steps") != num_forwards + 1
    ):
        raise ValueError(
            f"FastH3 inference contract has {num_forwards} rungs, so it needs "
            f"transformer_forwards={num_forwards} and num_inference_steps="
            f"{num_forwards + 1}; got {raw.get('transformer_forwards')!r} and "
            f"{raw.get('num_inference_steps')!r}"
        )
    return tuple(rungs)


def _declared_shift(
    raw: dict[str, Any], modality: str, scheduler_shift: float
) -> float:
    key = f"{modality}_scheduler_shift"
    if key not in raw:
        return scheduler_shift
    declared = _positive_finite(raw[key], f"FastH3 inference contract {key}")
    if declared != scheduler_shift:
        raise ValueError(
            f"FastH3 inference contract {key}={declared} disagrees with "
            f"{_SCHEDULER_CONFIG_FILES[modality]} shift={scheduler_shift}"
        )
    return declared


def _validated_vsa_sparsity(raw: dict[str, Any]) -> float:
    if raw.get("attention_backend") != _VSA_H3_BACKEND:
        raise ValueError(
            f"FastH3 inference contract attention_backend must be {_VSA_H3_BACKEND!r}, "
            f"got {raw.get('attention_backend')!r}"
        )
    if raw.get("vsa_tile_size") != _VSA_H3_TILE_SIZE:
        raise ValueError(
            f"FastH3 inference contract vsa_tile_size={raw.get('vsa_tile_size')!r}; "
            f"SGLang's VSA-H3 kernel serves {_VSA_H3_TILE_SIZE}-token tiles only"
        )
    sparsity = raw.get("vsa_sparsity")
    if (
        isinstance(sparsity, bool)
        or not isinstance(sparsity, (int, float))
        or not 0.0 <= sparsity < 1.0
    ):
        raise ValueError(
            f"FastH3 inference contract vsa_sparsity must be in [0, 1), got {sparsity!r}"
        )
    return float(sparsity)


__all__ = ["FASTH3_INFERENCE_CONTRACT_FILE", "FastH3InferenceContract"]
