"""Request/response data structures for post-training APIs."""

from dataclasses import dataclass
from typing import Optional, Union


@dataclass
class UpdateWeightFromDiskReqInput:
    """Request to update model weights from disk for diffusion models."""

    model_path: str
    flush_cache: bool = True
    target_modules: list[str] | None = None


@dataclass
class UpdateWeightFromTensorReqInput:
    """Request to update model weights from tensor payloads for diffusion models."""

    serialized_named_tensors: list[Union[str, bytes]]
    load_format: Optional[str] = None
    target_modules: list[str] | None = None
    weight_version: Optional[str] = None


@dataclass
class UpdateWeightFromTensorReqOutput:
    """Response for update_weights_from_tensor request."""

    success: bool
    message: str


@dataclass
class UpdateWeightFromTensorCheckerReqInput:
    """Request to verify live transformer weights against expected SHA-256 values."""

    expected_transformer_sha256: list[dict[str, str]]


@dataclass
class UpdateWeightFromTensorCheckerReqOutput:
    """Response for update_weights_from_tensor_checker request."""

    success: bool
    message: str


@dataclass
class GetWeightsChecksumReqInput:
    """Compute SHA-256 checksum of loaded module weights for verification."""

    module_names: list[str] | None = None
