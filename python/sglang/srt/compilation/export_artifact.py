"""Export artifact metadata and path helpers for ``@sgl_compile``."""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

import torch

from sglang.srt.compilation.export_context import DistributedExportContext
from sglang.srt.environ import envs
from sglang.version import __version__ as sglang_version

ExportArtifactMode = Literal["build_if_missing", "export_only", "load_only"]
ExportShapePolicy = Literal["infer_dynamic", "static"]

_FORMAT_EXTENSIONS = {
    "torch": ".pt2",
    "onnx": ".onnx",
}


@dataclass(frozen=True)
class TensorSchema:
    dtype: str
    shape: tuple[int, ...]
    device_type: str

    @classmethod
    def from_tensor(cls, tensor: torch.Tensor) -> "TensorSchema":
        return cls(
            dtype=str(tensor.dtype),
            shape=tuple(int(dim) for dim in tensor.shape),
            device_type=tensor.device.type,
        )


@dataclass(frozen=True)
class ExportArtifactMetadata:
    key: str
    format: str
    mode: ExportArtifactMode
    shape_policy: ExportShapePolicy
    sglang_version: str
    torch_version: str
    input_schema: tuple[TensorSchema | None, ...] = field(default_factory=tuple)
    copy_output_to_arg_index: int | None = None
    distributed_context: dict[str, Any] = field(default_factory=dict)

    def to_json_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["input_schema"] = [
            asdict(schema) if schema is not None else None
            for schema in self.input_schema
        ]
        return data

    @classmethod
    def from_json_dict(cls, data: dict[str, Any]) -> "ExportArtifactMetadata":
        schema = tuple(
            TensorSchema(
                dtype=item["dtype"],
                shape=tuple(item["shape"]),
                device_type=item["device_type"],
            )
            if item is not None
            else None
            for item in data.get("input_schema", [])
        )
        return cls(
            key=data["key"],
            format=data["format"],
            mode=data.get("mode", "build_if_missing"),
            shape_policy=data.get("shape_policy", "infer_dynamic"),
            sglang_version=data.get("sglang_version", "unknown"),
            torch_version=data.get("torch_version", "unknown"),
            input_schema=schema,
            copy_output_to_arg_index=data.get("copy_output_to_arg_index"),
            distributed_context=data.get("distributed_context", {}),
        )


@dataclass(frozen=True)
class ExportArtifactSpec:
    key: str
    export_format: str
    mode: ExportArtifactMode
    shape_policy: ExportShapePolicy
    export_dir: Path | None
    copy_output_to_arg_index: int | None = None
    input_schema: tuple[TensorSchema | None, ...] = field(default_factory=tuple)
    distributed_context: DistributedExportContext = field(
        default_factory=DistributedExportContext
    )

    @property
    def safe_key(self) -> str:
        return safe_artifact_key(self.key)

    @property
    def torch_program_path(self) -> Path | None:
        if self.export_dir is None:
            return None
        return self.export_dir / f"{self.safe_key}.pt2"

    @property
    def runtime_artifact_path(self) -> Path | None:
        if self.export_dir is None:
            return None
        extension = _FORMAT_EXTENSIONS.get(self.export_format, f".{self.export_format}")
        return self.export_dir / f"{self.safe_key}{extension}"

    @property
    def metadata_path(self) -> Path | None:
        if self.export_dir is None:
            return None
        return self.export_dir / f"{self.safe_key}.metadata.json"

    @property
    def metadata(self) -> ExportArtifactMetadata:
        return ExportArtifactMetadata(
            key=self.key,
            format=self.export_format,
            mode=self.mode,
            shape_policy=self.shape_policy,
            sglang_version=sglang_version,
            torch_version=torch.__version__,
            input_schema=self.input_schema,
            copy_output_to_arg_index=self.copy_output_to_arg_index,
            distributed_context=self.distributed_context.to_metadata(),
        )

    def ensure_export_dir(self) -> None:
        if self.export_dir is not None:
            self.export_dir.mkdir(parents=True, exist_ok=True)

    def write_metadata(self) -> None:
        metadata_path = self.metadata_path
        if metadata_path is None:
            return
        self.ensure_export_dir()
        metadata_path.write_text(
            json.dumps(self.metadata.to_json_dict(), indent=2, sort_keys=True) + "\n"
        )

    def load_metadata(self) -> ExportArtifactMetadata | None:
        metadata_path = self.metadata_path
        if metadata_path is None or not metadata_path.exists():
            return None
        return ExportArtifactMetadata.from_json_dict(json.loads(metadata_path.read_text()))

    def validate_metadata(self) -> None:
        metadata = self.load_metadata()
        if metadata is None:
            return
        if metadata.key != self.key:
            raise RuntimeError(
                f"Export artifact key mismatch: expected {self.key!r}, "
                f"got {metadata.key!r}."
            )
        if metadata.format != self.export_format:
            raise RuntimeError(
                f"Export artifact format mismatch: expected {self.export_format!r}, "
                f"got {metadata.format!r}."
            )
        if metadata.copy_output_to_arg_index != self.copy_output_to_arg_index:
            raise RuntimeError(
                "Export artifact mutation contract mismatch: expected "
                f"{self.copy_output_to_arg_index!r}, got "
                f"{metadata.copy_output_to_arg_index!r}."
            )


def make_export_artifact_spec(
    *,
    key: str,
    export_format: str | None,
    mode: str | None,
    shape_policy: str | None,
    copy_output_to_arg_index: int | None,
    args: tuple[Any, ...],
    distributed_context: DistributedExportContext | None = None,
) -> ExportArtifactSpec:
    export_dir = envs.SGLANG_EXPORT_DIR.get()
    return ExportArtifactSpec(
        key=key,
        export_format=export_format or "torch",
        mode=normalize_artifact_mode(
            envs.SGLANG_EXPORT_ARTIFACT_MODE.get() or mode or "build_if_missing"
        ),
        shape_policy=normalize_shape_policy(shape_policy or "infer_dynamic"),
        export_dir=Path(export_dir) if export_dir else None,
        copy_output_to_arg_index=copy_output_to_arg_index,
        input_schema=tuple(
            TensorSchema.from_tensor(arg) if isinstance(arg, torch.Tensor) else None
            for arg in args
        ),
        distributed_context=distributed_context
        or DistributedExportContext.current(args),
    )


def normalize_artifact_mode(mode: str) -> ExportArtifactMode:
    if mode not in ("build_if_missing", "export_only", "load_only"):
        raise ValueError(
            "export_artifact_mode must be one of: build_if_missing, "
            f"export_only, load_only. Got {mode!r}."
        )
    return mode


def normalize_shape_policy(shape_policy: str) -> ExportShapePolicy:
    if shape_policy not in ("infer_dynamic", "static"):
        raise ValueError(
            "shape_policy must be one of: infer_dynamic, static. "
            f"Got {shape_policy!r}."
        )
    return shape_policy


def safe_artifact_key(key: str | None) -> str:
    if not key:
        raise ValueError("An export artifact key is required.")
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", key)


__all__ = [
    "ExportArtifactMetadata",
    "ExportArtifactMode",
    "ExportArtifactSpec",
    "ExportShapePolicy",
    "TensorSchema",
    "make_export_artifact_spec",
    "normalize_artifact_mode",
    "normalize_shape_policy",
    "safe_artifact_key",
]
