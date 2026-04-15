"""Request/response data structures for post-training APIs."""

from dataclasses import dataclass


@dataclass
class UpdateWeightFromDiskReqInput:
    """Request to update model weights from disk for diffusion models."""

    model_path: str
    flush_cache: bool = True
    target_modules: list[str] | None = None


@dataclass
class UpdateWeightFromTensorReqInput:
    """Request to update model weights from tensor payloads for diffusion models."""

    serialized_named_tensors: list[str | bytes]
    load_format: str | None = None
    target_modules: list[str] | None = None


@dataclass
class UpdateWeightFromTensorCheckerReqInput:
    """Request to verify live module weights against expected SHA-256 values."""

    target_module: str
    expected_named_tensors_sha256: dict[str, str]


@dataclass
class GetWeightsChecksumReqInput:
    """Compute SHA-256 checksum of loaded module weights for verification."""

    module_names: list[str] | None = None
