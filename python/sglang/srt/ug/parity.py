# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Protocol, cast

UGParityTask = Literal["vlm", "text_to_image", "image_edit", "interleave"]
UG_PARITY_TASKS = frozenset(("vlm", "text_to_image", "image_edit", "interleave"))
_UG_PARITY_TASK_ALIASES = {"interleaved": "interleave"}


def normalize_ug_parity_task(task: Any) -> UGParityTask:
    normalized = str(task).strip().lower().replace("-", "_")
    normalized = _UG_PARITY_TASK_ALIASES.get(normalized, normalized)
    if normalized not in UG_PARITY_TASKS:
        raise ValueError(f"Unsupported UG parity task: {task!r}")
    return cast(UGParityTask, normalized)


@dataclass(frozen=True)
class UGParityCase:
    """A reproducible input case shared by official and SGLang UG runners."""

    case_id: str
    task: UGParityTask
    prompt: str | None = None
    messages: tuple[dict[str, Any], ...] = ()
    image_path: str | None = None
    seed: int | None = None
    sampling_params: dict[str, Any] = field(default_factory=dict)
    dump_points: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "messages", tuple(dict(message) for message in self.messages)
        )
        object.__setattr__(self, "task", normalize_ug_parity_task(self.task))
        object.__setattr__(self, "sampling_params", dict(self.sampling_params))
        object.__setattr__(self, "dump_points", tuple(self.dump_points))
        object.__setattr__(self, "metadata", dict(self.metadata))
        self.validate()

    def validate(self) -> None:
        if not self.case_id:
            raise ValueError("UG parity case_id must be non-empty")
        if self.task not in UG_PARITY_TASKS:
            raise ValueError(f"Unsupported UG parity task: {self.task!r}")
        if self.seed is not None and not isinstance(self.seed, int):
            raise ValueError("UG parity seed must be an int or None")

    def to_dict(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "task": self.task,
            "prompt": self.prompt,
            "messages": [dict(message) for message in self.messages],
            "image_path": self.image_path,
            "seed": self.seed,
            "sampling_params": dict(self.sampling_params),
            "dump_points": list(self.dump_points),
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "UGParityCase":
        return cls(
            case_id=str(payload.get("case_id", "")),
            task=payload.get("task"),
            prompt=payload.get("prompt"),
            messages=tuple(payload.get("messages") or ()),
            image_path=payload.get("image_path"),
            seed=payload.get("seed"),
            sampling_params=dict(payload.get("sampling_params") or {}),
            dump_points=tuple(payload.get("dump_points") or ()),
            metadata=dict(payload.get("metadata") or {}),
        )

    def to_json(self) -> str:
        return _json_dumps(self.to_dict())

    @classmethod
    def from_json(cls, payload: str) -> "UGParityCase":
        return cls.from_dict(json.loads(payload))

    def write_json(self, path: str | Path) -> None:
        _write_json(path, self.to_dict())

    @classmethod
    def read_json(cls, path: str | Path) -> "UGParityCase":
        return cls.from_dict(_read_json(path))


@dataclass(frozen=True)
class UGTensorSummary:
    shape: tuple[int, ...]
    dtype: str
    numel: int
    sha256: str
    min: float | None = None
    max: float | None = None
    mean: float | None = None
    std: float | None = None

    @classmethod
    def from_tensor(cls, tensor: Any) -> "UGTensorSummary":
        try:
            import torch
        except ImportError as exc:
            raise RuntimeError("UGTensorSummary.from_tensor requires torch") from exc

        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"Expected torch.Tensor, got {type(tensor).__name__}")
        cpu_tensor = tensor.detach().cpu().contiguous()
        numel = int(cpu_tensor.numel())
        if numel == 0:
            return cls(
                shape=tuple(int(dim) for dim in cpu_tensor.shape),
                dtype=str(cpu_tensor.dtype),
                numel=0,
                sha256=hashlib.sha256(b"").hexdigest(),
            )

        byte_tensor = cpu_tensor.reshape(-1).view(torch.uint8)
        digest = hashlib.sha256(byte_tensor.numpy().tobytes()).hexdigest()
        numeric = cpu_tensor.float()
        return cls(
            shape=tuple(int(dim) for dim in cpu_tensor.shape),
            dtype=str(cpu_tensor.dtype),
            numel=numel,
            sha256=digest,
            min=float(numeric.min().item()),
            max=float(numeric.max().item()),
            mean=float(numeric.mean().item()),
            std=float(numeric.std(unbiased=False).item()),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "shape": list(self.shape),
            "dtype": self.dtype,
            "numel": self.numel,
            "sha256": self.sha256,
            "min": self.min,
            "max": self.max,
            "mean": self.mean,
            "std": self.std,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "UGTensorSummary":
        return cls(
            shape=tuple(int(dim) for dim in payload.get("shape", ())),
            dtype=str(payload.get("dtype", "")),
            numel=int(payload.get("numel", 0)),
            sha256=str(payload.get("sha256", "")),
            min=_optional_float(payload.get("min")),
            max=_optional_float(payload.get("max")),
            mean=_optional_float(payload.get("mean")),
            std=_optional_float(payload.get("std")),
        )


@dataclass(frozen=True)
class UGImageSummary:
    sha256: str
    path: str | None = None
    width: int | None = None
    height: int | None = None
    mode: str | None = None

    @classmethod
    def from_path(cls, path: str | Path) -> "UGImageSummary":
        image_path = Path(path)
        data = image_path.read_bytes()
        width = height = None
        mode = None
        try:
            from PIL import Image

            with Image.open(image_path) as image:
                width, height = image.size
                mode = image.mode
        except Exception:
            pass
        return cls.from_bytes(
            data,
            path=str(image_path),
            width=width,
            height=height,
            mode=mode,
        )

    @classmethod
    def from_bytes(
        cls,
        data: bytes,
        *,
        path: str | None = None,
        width: int | None = None,
        height: int | None = None,
        mode: str | None = None,
    ) -> "UGImageSummary":
        return cls(
            sha256=hashlib.sha256(data).hexdigest(),
            path=path,
            width=width,
            height=height,
            mode=mode,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "sha256": self.sha256,
            "path": self.path,
            "width": self.width,
            "height": self.height,
            "mode": self.mode,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "UGImageSummary":
        return cls(
            sha256=str(payload.get("sha256", "")),
            path=payload.get("path"),
            width=_optional_int(payload.get("width")),
            height=_optional_int(payload.get("height")),
            mode=payload.get("mode"),
        )


@dataclass(frozen=True)
class UGParityArtifact:
    case_id: str
    runner: str
    task: str | None = None
    text: str | None = None
    token_ids: dict[str, tuple[int, ...]] = field(default_factory=dict)
    images: dict[str, UGImageSummary] = field(default_factory=dict)
    tensors: dict[str, UGTensorSummary] = field(default_factory=dict)
    debug_counters: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    error: str | None = None

    def __post_init__(self) -> None:
        if not self.case_id:
            raise ValueError("UG parity artifact case_id must be non-empty")
        if not self.runner:
            raise ValueError("UG parity artifact runner must be non-empty")
        object.__setattr__(
            self,
            "token_ids",
            {
                name: tuple(int(token_id) for token_id in ids)
                for name, ids in self.token_ids.items()
            },
        )
        object.__setattr__(self, "images", dict(self.images))
        object.__setattr__(self, "tensors", dict(self.tensors))
        object.__setattr__(self, "debug_counters", dict(self.debug_counters))
        object.__setattr__(self, "metadata", dict(self.metadata))

    def to_dict(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "runner": self.runner,
            "task": self.task,
            "text": self.text,
            "token_ids": {
                name: list(token_ids) for name, token_ids in self.token_ids.items()
            },
            "images": {
                name: summary.to_dict() for name, summary in self.images.items()
            },
            "tensors": {
                name: summary.to_dict() for name, summary in self.tensors.items()
            },
            "debug_counters": dict(self.debug_counters),
            "metadata": dict(self.metadata),
            "error": self.error,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "UGParityArtifact":
        return cls(
            case_id=str(payload.get("case_id", "")),
            runner=str(payload.get("runner", "")),
            task=payload.get("task"),
            text=payload.get("text"),
            token_ids={
                name: tuple(int(token_id) for token_id in token_ids)
                for name, token_ids in (payload.get("token_ids") or {}).items()
            },
            images={
                name: UGImageSummary.from_dict(summary)
                for name, summary in (payload.get("images") or {}).items()
            },
            tensors={
                name: UGTensorSummary.from_dict(summary)
                for name, summary in (payload.get("tensors") or {}).items()
            },
            debug_counters=dict(payload.get("debug_counters") or {}),
            metadata=dict(payload.get("metadata") or {}),
            error=payload.get("error"),
        )

    def to_json(self) -> str:
        return _json_dumps(self.to_dict())

    @classmethod
    def from_json(cls, payload: str) -> "UGParityArtifact":
        return cls.from_dict(json.loads(payload))

    def write_json(self, path: str | Path) -> None:
        _write_json(path, self.to_dict())

    @classmethod
    def read_json(cls, path: str | Path) -> "UGParityArtifact":
        return cls.from_dict(_read_json(path))


@dataclass(frozen=True)
class UGParityTolerance:
    text_exact: bool = True
    image_sha256_exact: bool = True
    compare_tensors: bool = True
    tensor_stat_atol: float = 1e-5
    tensor_stat_rtol: float = 1e-5
    require_tensor_sha256: bool = False


@dataclass(frozen=True)
class UGParityDifference:
    field: str
    reference: Any
    candidate: Any
    message: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "field": self.field,
            "reference": self.reference,
            "candidate": self.candidate,
            "message": self.message,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "UGParityDifference":
        return cls(
            field=str(payload.get("field", "")),
            reference=payload.get("reference"),
            candidate=payload.get("candidate"),
            message=str(payload.get("message", "")),
        )


@dataclass(frozen=True)
class UGParityReport:
    case_id: str
    passed: bool
    reference_runner: str
    candidate_runner: str
    differences: tuple[UGParityDifference, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "passed": self.passed,
            "reference_runner": self.reference_runner,
            "candidate_runner": self.candidate_runner,
            "differences": [diff.to_dict() for diff in self.differences],
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "UGParityReport":
        return cls(
            case_id=str(payload.get("case_id", "")),
            passed=bool(payload.get("passed", False)),
            reference_runner=str(payload.get("reference_runner", "")),
            candidate_runner=str(payload.get("candidate_runner", "")),
            differences=tuple(
                UGParityDifference.from_dict(diff)
                for diff in payload.get("differences", ())
            ),
            metadata=dict(payload.get("metadata") or {}),
        )

    def to_json(self) -> str:
        return _json_dumps(self.to_dict())

    @classmethod
    def from_json(cls, payload: str) -> "UGParityReport":
        return cls.from_dict(json.loads(payload))

    def write_json(self, path: str | Path) -> None:
        _write_json(path, self.to_dict())


class UGParityRunner(Protocol):
    runner_name: str

    def run(self, case: UGParityCase) -> UGParityArtifact: ...


def run_ug_parity_case(
    case: UGParityCase,
    *,
    reference_runner: UGParityRunner,
    candidate_runner: UGParityRunner,
    tolerance: UGParityTolerance | None = None,
) -> UGParityReport:
    reference = reference_runner.run(case)
    candidate = candidate_runner.run(case)
    return compare_ug_parity_artifacts(
        reference,
        candidate,
        tolerance=tolerance,
    )


def compare_ug_parity_artifacts(
    reference: UGParityArtifact,
    candidate: UGParityArtifact,
    *,
    tolerance: UGParityTolerance | None = None,
) -> UGParityReport:
    tolerance = tolerance or UGParityTolerance()
    differences: list[UGParityDifference] = []

    if reference.case_id != candidate.case_id:
        differences.append(
            _diff(
                "case_id",
                reference.case_id,
                candidate.case_id,
                "case_id mismatch",
            )
        )
    if reference.task is not None and candidate.task is not None:
        if reference.task != candidate.task:
            differences.append(
                _diff("task", reference.task, candidate.task, "task mismatch")
            )

    if reference.error or candidate.error:
        if reference.error != candidate.error:
            differences.append(
                _diff(
                    "error",
                    reference.error,
                    candidate.error,
                    "runner error mismatch",
                )
            )

    _compare_text(reference, candidate, tolerance, differences)
    _compare_token_ids(reference, candidate, differences)
    _compare_images(reference, candidate, tolerance, differences)
    if tolerance.compare_tensors:
        _compare_tensors(reference, candidate, tolerance, differences)

    return UGParityReport(
        case_id=reference.case_id,
        passed=not differences,
        reference_runner=reference.runner,
        candidate_runner=candidate.runner,
        differences=tuple(differences),
    )


def _compare_text(
    reference: UGParityArtifact,
    candidate: UGParityArtifact,
    tolerance: UGParityTolerance,
    differences: list[UGParityDifference],
) -> None:
    if reference.text is None and candidate.text is None:
        return
    if tolerance.text_exact and reference.text != candidate.text:
        differences.append(
            _diff("text", reference.text, candidate.text, "text output mismatch")
        )


def _compare_token_ids(
    reference: UGParityArtifact,
    candidate: UGParityArtifact,
    differences: list[UGParityDifference],
) -> None:
    for name in sorted(set(reference.token_ids) | set(candidate.token_ids)):
        ref = reference.token_ids.get(name)
        cand = candidate.token_ids.get(name)
        if ref != cand:
            differences.append(
                _diff(
                    f"token_ids.{name}",
                    list(ref) if ref is not None else None,
                    list(cand) if cand is not None else None,
                    "token id sequence mismatch",
                )
            )


def _compare_images(
    reference: UGParityArtifact,
    candidate: UGParityArtifact,
    tolerance: UGParityTolerance,
    differences: list[UGParityDifference],
) -> None:
    for name in sorted(set(reference.images) | set(candidate.images)):
        ref = reference.images.get(name)
        cand = candidate.images.get(name)
        if ref is None or cand is None:
            differences.append(
                _diff(
                    f"images.{name}",
                    ref.to_dict() if ref else None,
                    cand.to_dict() if cand else None,
                    "image summary missing",
                )
            )
            continue
        for field_name in ("width", "height", "mode"):
            ref_value = getattr(ref, field_name)
            cand_value = getattr(cand, field_name)
            if ref_value != cand_value:
                differences.append(
                    _diff(
                        f"images.{name}.{field_name}",
                        ref_value,
                        cand_value,
                        "image metadata mismatch",
                    )
                )
        if tolerance.image_sha256_exact and ref.sha256 != cand.sha256:
            differences.append(
                _diff(
                    f"images.{name}.sha256",
                    ref.sha256,
                    cand.sha256,
                    "image sha256 mismatch",
                )
            )


def _compare_tensors(
    reference: UGParityArtifact,
    candidate: UGParityArtifact,
    tolerance: UGParityTolerance,
    differences: list[UGParityDifference],
) -> None:
    for name in sorted(set(reference.tensors) | set(candidate.tensors)):
        ref = reference.tensors.get(name)
        cand = candidate.tensors.get(name)
        if ref is None or cand is None:
            differences.append(
                _diff(
                    f"tensors.{name}",
                    ref.to_dict() if ref else None,
                    cand.to_dict() if cand else None,
                    "tensor summary missing",
                )
            )
            continue
        if ref.shape != cand.shape:
            differences.append(
                _diff(
                    f"tensors.{name}.shape",
                    list(ref.shape),
                    list(cand.shape),
                    "tensor shape mismatch",
                )
            )
        if ref.dtype != cand.dtype:
            differences.append(
                _diff(
                    f"tensors.{name}.dtype",
                    ref.dtype,
                    cand.dtype,
                    "tensor dtype mismatch",
                )
            )
        if ref.numel != cand.numel:
            differences.append(
                _diff(
                    f"tensors.{name}.numel",
                    ref.numel,
                    cand.numel,
                    "tensor numel mismatch",
                )
            )
        if tolerance.require_tensor_sha256 and ref.sha256 != cand.sha256:
            differences.append(
                _diff(
                    f"tensors.{name}.sha256",
                    ref.sha256,
                    cand.sha256,
                    "tensor sha256 mismatch",
                )
            )
        for stat_name in ("min", "max", "mean", "std"):
            ref_value = getattr(ref, stat_name)
            cand_value = getattr(cand, stat_name)
            if not _close_optional_float(
                ref_value,
                cand_value,
                atol=tolerance.tensor_stat_atol,
                rtol=tolerance.tensor_stat_rtol,
            ):
                differences.append(
                    _diff(
                        f"tensors.{name}.{stat_name}",
                        ref_value,
                        cand_value,
                        "tensor stat mismatch",
                    )
                )


def _close_optional_float(
    reference: float | None,
    candidate: float | None,
    *,
    atol: float,
    rtol: float,
) -> bool:
    if reference is None or candidate is None:
        return reference is candidate
    return math.isclose(reference, candidate, abs_tol=atol, rel_tol=rtol)


def _diff(
    field: str,
    reference: Any,
    candidate: Any,
    message: str,
) -> UGParityDifference:
    return UGParityDifference(
        field=field,
        reference=reference,
        candidate=candidate,
        message=message,
    )


def _optional_float(value: Any) -> float | None:
    return None if value is None else float(value)


def _optional_int(value: Any) -> int | None:
    return None if value is None else int(value)


def _json_dumps(payload: dict[str, Any]) -> str:
    return json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True)


def _write_json(path: str | Path, payload: dict[str, Any]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(_json_dumps(payload) + "\n", encoding="utf-8")


def _read_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as f:
        return json.load(f)
