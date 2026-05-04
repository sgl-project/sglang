# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from io import BytesIO
from pathlib import Path
from typing import Any, Literal, Protocol

from PIL import Image

UGParityTask = Literal["vlm", "t2i", "edit", "interleave"]
_SUPPORTED_TASKS = {"vlm", "t2i", "edit", "interleave"}


@dataclass(frozen=True)
class UGParityCase:
    case_id: str
    model: str
    task: UGParityTask
    prompt: str | None = None
    image_path: str | None = None
    messages: tuple[dict[str, Any], ...] = ()
    seed: int | None = None
    sampling_params: dict[str, Any] = field(default_factory=dict)
    dump_points: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.case_id:
            raise ValueError("UG parity case_id must be non-empty")
        if not self.model:
            raise ValueError("UG parity model must be non-empty")
        if self.task not in _SUPPORTED_TASKS:
            raise ValueError(f"Unsupported UG parity task: {self.task!r}")
        object.__setattr__(self, "messages", tuple(self.messages or ()))
        object.__setattr__(self, "dump_points", tuple(self.dump_points or ()))

    def to_dict(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "model": self.model,
            "task": self.task,
            "prompt": self.prompt,
            "image_path": self.image_path,
            "messages": list(self.messages),
            "seed": self.seed,
            "sampling_params": dict(self.sampling_params),
            "dump_points": list(self.dump_points),
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "UGParityCase":
        return cls(
            case_id=data["case_id"],
            model=data["model"],
            task=data["task"],
            prompt=data.get("prompt"),
            image_path=data.get("image_path"),
            messages=tuple(data.get("messages") or ()),
            seed=data.get("seed"),
            sampling_params=dict(data.get("sampling_params") or {}),
            dump_points=tuple(data.get("dump_points") or ()),
            metadata=dict(data.get("metadata") or {}),
        )

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True)

    @classmethod
    def from_json(cls, data: str) -> "UGParityCase":
        return cls.from_dict(json.loads(data))


@dataclass(frozen=True)
class UGImageSummary:
    size: tuple[int, int]
    mode: str
    sha256: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "size": list(self.size),
            "mode": self.mode,
            "sha256": self.sha256,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "UGImageSummary":
        return cls(
            size=tuple(data["size"]),
            mode=data["mode"],
            sha256=data["sha256"],
        )


@dataclass(frozen=True)
class UGTensorSummary:
    shape: tuple[int, ...]
    dtype: str
    min: float | None
    max: float | None
    mean: float | None
    std: float | None
    sha256: str

    @classmethod
    def from_tensor(cls, tensor: Any) -> "UGTensorSummary":
        data = tensor.detach().cpu().contiguous()
        stats = data.float()
        if stats.numel() == 0:
            min_value = max_value = mean_value = std_value = None
        else:
            min_value = float(stats.min().item())
            max_value = float(stats.max().item())
            mean_value = float(stats.mean().item())
            std_value = float(stats.std(unbiased=False).item())
        return cls(
            shape=tuple(int(dim) for dim in data.shape),
            dtype=str(data.dtype),
            min=min_value,
            max=max_value,
            mean=mean_value,
            std=std_value,
            sha256=_sha256_tensor(data),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "shape": list(self.shape),
            "dtype": self.dtype,
            "min": self.min,
            "max": self.max,
            "mean": self.mean,
            "std": self.std,
            "sha256": self.sha256,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "UGTensorSummary":
        return cls(
            shape=tuple(data["shape"]),
            dtype=data["dtype"],
            min=data.get("min"),
            max=data.get("max"),
            mean=data.get("mean"),
            std=data.get("std"),
            sha256=data["sha256"],
        )


@dataclass(frozen=True)
class UGParityArtifact:
    case_id: str
    model: str
    task: UGParityTask
    runner: str
    text: str | None = None
    image: UGImageSummary | None = None
    tensors: dict[str, UGTensorSummary] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    debug_counters: dict[str, Any] = field(default_factory=dict)
    error: str | None = None

    def __post_init__(self) -> None:
        if not self.case_id:
            raise ValueError("UG parity artifact case_id must be non-empty")
        if not self.runner:
            raise ValueError("UG parity artifact runner must be non-empty")

    def to_dict(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "model": self.model,
            "task": self.task,
            "runner": self.runner,
            "text": self.text,
            "image": self.image.to_dict() if self.image is not None else None,
            "tensors": {
                name: summary.to_dict() for name, summary in self.tensors.items()
            },
            "metadata": dict(self.metadata),
            "debug_counters": dict(self.debug_counters),
            "error": self.error,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "UGParityArtifact":
        image = data.get("image")
        return cls(
            case_id=data["case_id"],
            model=data["model"],
            task=data["task"],
            runner=data["runner"],
            text=data.get("text"),
            image=UGImageSummary.from_dict(image) if image is not None else None,
            tensors={
                name: UGTensorSummary.from_dict(summary)
                for name, summary in (data.get("tensors") or {}).items()
            },
            metadata=dict(data.get("metadata") or {}),
            debug_counters=dict(data.get("debug_counters") or {}),
            error=data.get("error"),
        )

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True)

    @classmethod
    def from_json(cls, data: str) -> "UGParityArtifact":
        return cls.from_dict(json.loads(data))


@dataclass(frozen=True)
class UGParityDiff:
    field: str
    reference: Any
    candidate: Any
    reason: str


@dataclass(frozen=True)
class UGParityReport:
    case_id: str
    model: str
    passed: bool
    diffs: tuple[UGParityDiff, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "model": self.model,
            "passed": self.passed,
            "diffs": [asdict(diff) for diff in self.diffs],
            "metadata": dict(self.metadata),
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True)


class UGParityRunner(Protocol):
    def run(self, case: UGParityCase) -> UGParityArtifact: ...


def summarize_ug_image(image: Image.Image | str | Path) -> UGImageSummary:
    if isinstance(image, (str, Path)):
        path = Path(image)
        with Image.open(path) as opened:
            loaded = opened.convert("RGB")
    elif isinstance(image, Image.Image):
        loaded = image.convert("RGB")
    else:
        raise TypeError(f"UG image summary expects PIL image or path, got {type(image)}")

    buffer = BytesIO()
    loaded.save(buffer, format="PNG")
    return UGImageSummary(
        size=loaded.size,
        mode=loaded.mode,
        sha256=hashlib.sha256(buffer.getvalue()).hexdigest(),
    )


def compare_ug_parity_artifacts(
    reference: UGParityArtifact,
    candidate: UGParityArtifact,
) -> UGParityReport:
    diffs: list[UGParityDiff] = []
    _compare_scalar(diffs, "case_id", reference.case_id, candidate.case_id)
    _compare_scalar(diffs, "model", reference.model, candidate.model)
    _compare_scalar(diffs, "task", reference.task, candidate.task)
    _compare_scalar(diffs, "error", reference.error, candidate.error)
    _compare_scalar(diffs, "text", reference.text, candidate.text)
    _compare_image(diffs, reference.image, candidate.image)
    _compare_tensors(diffs, reference.tensors, candidate.tensors)
    return UGParityReport(
        case_id=reference.case_id,
        model=reference.model,
        passed=not diffs,
        diffs=tuple(diffs),
        metadata={
            "reference_runner": reference.runner,
            "candidate_runner": candidate.runner,
        },
    )


def write_ug_parity_bundle(
    *,
    output_dir: str | Path,
    case: UGParityCase,
    reference: UGParityArtifact,
    candidate: UGParityArtifact,
    report: UGParityReport,
) -> Path:
    bundle_dir = Path(output_dir) / case.case_id
    bundle_dir.mkdir(parents=True, exist_ok=True)
    _write_json(bundle_dir / "case.json", case.to_dict())
    _write_json(bundle_dir / "reference.json", reference.to_dict())
    _write_json(bundle_dir / "candidate.json", candidate.to_dict())
    _write_json(bundle_dir / "report.json", report.to_dict())
    return bundle_dir


def _compare_scalar(
    diffs: list[UGParityDiff],
    field: str,
    reference: Any,
    candidate: Any,
) -> None:
    if reference != candidate:
        diffs.append(
            UGParityDiff(
                field=field,
                reference=reference,
                candidate=candidate,
                reason="value mismatch",
            )
        )


def _compare_image(
    diffs: list[UGParityDiff],
    reference: UGImageSummary | None,
    candidate: UGImageSummary | None,
) -> None:
    if reference is None or candidate is None:
        _compare_scalar(diffs, "image", reference, candidate)
        return
    _compare_scalar(diffs, "image.size", reference.size, candidate.size)
    _compare_scalar(diffs, "image.mode", reference.mode, candidate.mode)
    _compare_scalar(diffs, "image.sha256", reference.sha256, candidate.sha256)


def _compare_tensors(
    diffs: list[UGParityDiff],
    reference: dict[str, UGTensorSummary],
    candidate: dict[str, UGTensorSummary],
) -> None:
    for name in sorted(set(reference) | set(candidate)):
        ref = reference.get(name)
        cand = candidate.get(name)
        if ref is None or cand is None:
            _compare_scalar(diffs, f"tensors.{name}", ref, cand)
            continue
        _compare_scalar(diffs, f"tensors.{name}.shape", ref.shape, cand.shape)
        _compare_scalar(diffs, f"tensors.{name}.dtype", ref.dtype, cand.dtype)
        _compare_scalar(diffs, f"tensors.{name}.sha256", ref.sha256, cand.sha256)


def _sha256_tensor(tensor: Any) -> str:
    try:
        payload = tensor.numpy().tobytes()
    except TypeError:
        payload = repr(tensor.tolist()).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")
