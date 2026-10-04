"""Feature-owned HTTP and IPC schemas for GPU-delta controls."""

from typing import Any, Dict, List

from sglang.srt.managers.io_struct import BaseReq, hook_custom_types


class GetWeightsDeltaInfoReqInput(BaseReq, kw_only=True):
    engine_id: str


class PrepareWeightsFromDeltaReqInput(BaseReq, kw_only=True):
    session_id: str
    engine_id: str
    manifest_path: str
    manifest_sha256: str
    stream_id: str
    base_version: int
    target_version: int
    plan_digest: str
    participants: List[Dict[str, Any]]
    host_tensor_names: Dict[str, List[str]]


class GetWeightsDeltaStatusReqInput(BaseReq, kw_only=True):
    session_id: str


class UpdateWeightsFromDeltaReqInput(BaseReq, kw_only=True):
    session_id: str


class AbortWeightsFromDeltaReqInput(BaseReq, kw_only=True):
    session_id: str


class ResumeWeightsFromDeltaReqInput(BaseReq, kw_only=True):
    session_id: str
    receipts: List[Dict[str, Any]]


class DeltaWeightsReqOutput(BaseReq, kw_only=True):
    success: bool
    message: str
    participant: Dict[str, Any]


# Register before tokenizer/scheduler/DP-controller receive loops start.
hook_custom_types(
    GetWeightsDeltaInfoReqInput,
    PrepareWeightsFromDeltaReqInput,
    GetWeightsDeltaStatusReqInput,
    UpdateWeightsFromDeltaReqInput,
    AbortWeightsFromDeltaReqInput,
    ResumeWeightsFromDeltaReqInput,
    DeltaWeightsReqOutput,
)
