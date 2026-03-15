"""Request/response data structures for post-training APIs.

TODO(Shuwen, Chenyang): Split RL-oriented request types and serving-oriented
request types into dedicated files.
"""

from dataclasses import dataclass


@dataclass
class UpdateWeightFromDiskReqInput:
    """Request to update model weights from disk for diffusion models."""

    model_path: str
    flush_cache: bool = True
    target_modules: list[str] | None = None


@dataclass
class GetWeightsChecksumReqInput:
    """Compute SHA-256 checksum of loaded module weights for verification."""

    module_names: list[str] | None = None


@dataclass
class ReleaseMemoryOccupationReqInput:
    """Request to release (sleep) GPU memory occupation for the diffusion engine."""

    # TODO (Kun, Chenyang): We shall have rather dedicated
    # control of the Diffusion model's memory occupation.
    pass


@dataclass
class ResumeMemoryOccupationReqInput:
    """Request to resume (wake) GPU memory occupation for the diffusion engine."""

    pass
