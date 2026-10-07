"""Feature-owned HTTP and IPC schemas for GPU-delta controls."""

from typing import Any, Dict, List

from sglang.srt.managers.io_struct import BaseReq, hook_custom_types


class LoadWeightsFromDeltaReqInput(BaseReq, kw_only=True):
    manifest_path: str
    release_state: bool = True


class ClearWeightsDeltaStateReqInput(BaseReq, kw_only=True):
    pass


class ReleaseWeightsDeltaCacheReqInput(BaseReq, kw_only=True):
    # Internal second phase, sent only after every rank has closed its resources.
    owner_rank_ids: List[str]


class GetWeightsDeltaInfoReqInput(BaseReq, kw_only=True):
    engine_id: str


class PrepareWeightsFromDeltaReqInput(BaseReq, kw_only=True):
    session_id: str
    manifest_path: str
    manifest_sha256: str
    stream_id: str
    base_version: int
    target_version: int
    plan_digest: str
    participants: List[Dict[str, Any]]


class GetWeightsDeltaStatusReqInput(BaseReq, kw_only=True):
    session_id: str


class UpdateWeightsFromDeltaReqInput(BaseReq, kw_only=True):
    session_id: str


class AbortWeightsFromDeltaReqInput(BaseReq, kw_only=True):
    session_id: str


class ResumeWeightsFromDeltaReqInput(BaseReq, kw_only=True):
    session_id: str


class DeltaWeightsReqOutput(BaseReq, kw_only=True):
    success: bool
    message: str
    participant: Dict[str, Any]


# Register before tokenizer/scheduler/DP-controller receive loops start.
hook_custom_types(
    LoadWeightsFromDeltaReqInput,
    ClearWeightsDeltaStateReqInput,
    ReleaseWeightsDeltaCacheReqInput,
    GetWeightsDeltaInfoReqInput,
    PrepareWeightsFromDeltaReqInput,
    GetWeightsDeltaStatusReqInput,
    UpdateWeightsFromDeltaReqInput,
    AbortWeightsFromDeltaReqInput,
    ResumeWeightsFromDeltaReqInput,
    DeltaWeightsReqOutput,
)
