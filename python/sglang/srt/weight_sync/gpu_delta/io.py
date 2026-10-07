"""Feature-owned HTTP and IPC schemas for GPU-delta controls."""

from typing import Any, Dict, List

from sglang.srt.managers.io_struct import BaseReq, hook_custom_types


class UpdateWeightsFromGpuDeltaReqInput(BaseReq, kw_only=True):
    manifest_path: str
    release_state: bool = True


class ClearGpuDeltaStateReqInput(BaseReq, kw_only=True):
    pass


class ReleaseGpuDeltaCacheReqInput(BaseReq, kw_only=True):
    # Internal second phase, sent only after every rank has closed its resources.
    owner_rank_ids: List[str]


class GetGpuDeltaInfoReqInput(BaseReq, kw_only=True):
    engine_id: str


class PrepareGpuDeltaReqInput(BaseReq, kw_only=True):
    session_id: str
    manifest_path: str
    manifest_sha256: str
    stream_id: str
    base_version: int
    target_version: int
    plan_digest: str
    participants: List[Dict[str, Any]]


class GetGpuDeltaStatusReqInput(BaseReq, kw_only=True):
    session_id: str


class ApplyGpuDeltaReqInput(BaseReq, kw_only=True):
    session_id: str


class AbortGpuDeltaReqInput(BaseReq, kw_only=True):
    session_id: str


class ResumeGpuDeltaReqInput(BaseReq, kw_only=True):
    session_id: str


class GpuDeltaReqOutput(BaseReq, kw_only=True):
    success: bool
    message: str
    participant: Dict[str, Any]


# Register before tokenizer/scheduler/DP-controller receive loops start.
hook_custom_types(
    UpdateWeightsFromGpuDeltaReqInput,
    ClearGpuDeltaStateReqInput,
    ReleaseGpuDeltaCacheReqInput,
    GetGpuDeltaInfoReqInput,
    PrepareGpuDeltaReqInput,
    GetGpuDeltaStatusReqInput,
    ApplyGpuDeltaReqInput,
    AbortGpuDeltaReqInput,
    ResumeGpuDeltaReqInput,
    GpuDeltaReqOutput,
)
