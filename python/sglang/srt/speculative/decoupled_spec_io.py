from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class DraftMeshMessageType(str, Enum):
    CONTROL_BATCH = "control_batch"
    TAIL_STREAM_OUTPUT_BATCH = "tail_stream_output_batch"


@dataclass(frozen=True)
class DraftReqKey:
    src_verifier_rank: int
    request_id: str


def build_draft_scheduler_rid(draft_key: DraftReqKey) -> str:
    return f"draft:{int(draft_key.src_verifier_rank)}:{draft_key.request_id}"


def parse_draft_scheduler_rid(rid: str) -> DraftReqKey:
    if rid.startswith("draft:"):
        encoded = rid[len("draft:") :]
        rank_text, sep, request_id = encoded.partition(":")
        if sep and request_id:
            return DraftReqKey(
                src_verifier_rank=int(rank_text),
                request_id=request_id,
            )

    raise ValueError(f"Invalid decoupled draft scheduler rid: {rid}")


@dataclass
class DraftSync:
    request_id: str
    src_verifier_rank: int
    dst_drafter_rank: int
    prompt_token_ids: list[int] = field(default_factory=list)
    committed_output_ids: list[int] = field(default_factory=list)

    @property
    def draft_key(self) -> DraftReqKey:
        return DraftReqKey(
            src_verifier_rank=int(self.src_verifier_rank),
            request_id=self.request_id,
        )


@dataclass
class VerifyCommit:
    """
    Sent from verifier to drafter to commit a portion of the draft outputs.

    Drafter relies on pre_verify_committed_len and bonus_token_pos to
    correctly push forward the verifier_committed_prefix_len.
    """

    request_id: str
    src_verifier_rank: int
    dst_drafter_rank: int
    pre_verify_committed_len: int
    bonus_token_id: int
    bonus_token_pos: int

    @property
    def draft_key(self) -> DraftReqKey:
        return DraftReqKey(
            src_verifier_rank=int(self.src_verifier_rank),
            request_id=self.request_id,
        )


@dataclass
class DraftClose:
    request_id: str
    src_verifier_rank: int
    dst_drafter_rank: int
    reason: str

    @property
    def draft_key(self) -> DraftReqKey:
        return DraftReqKey(
            src_verifier_rank=int(self.src_verifier_rank),
            request_id=self.request_id,
        )


@dataclass
class DraftTailStreamOutput:
    """
    Drafter sends a stream output to verifier whenever it decodes a new token
    """

    src_drafter_rank: int
    dst_verifier_rank: int
    request_id: str
    base_committed_len: int
    new_token_pos: int
    new_token_id: int


@dataclass
class DraftTailStreamOutputBatch:
    outputs: list[DraftTailStreamOutput] = field(default_factory=list)


@dataclass
class DraftControlBatch:
    dst_drafter_rank: int
    sync_messages: list[DraftSync] = field(default_factory=list)
    verify_commit_messages: list[VerifyCommit] = field(default_factory=list)
    close_messages: list[DraftClose] = field(default_factory=list)


DraftControlMessage = DraftSync | VerifyCommit | DraftClose


def iter_control_batch_messages(batch: DraftControlBatch) -> list[DraftControlMessage]:
    return [
        *batch.sync_messages,
        *batch.verify_commit_messages,
        *batch.close_messages,
    ]


@dataclass
class DraftMeshMessage:
    message_type: DraftMeshMessageType
    control_batch: Optional[DraftControlBatch] = None
    tail_stream_output_batch: Optional[DraftTailStreamOutputBatch] = None

    @staticmethod
    def from_control_batch(message: DraftControlBatch) -> "DraftMeshMessage":
        return DraftMeshMessage(
            message_type=DraftMeshMessageType.CONTROL_BATCH,
            control_batch=message,
        )

    @staticmethod
    def from_tail_stream_output_batch(
        message: DraftTailStreamOutputBatch,
    ) -> "DraftMeshMessage":
        return DraftMeshMessage(
            message_type=DraftMeshMessageType.TAIL_STREAM_OUTPUT_BATCH,
            tail_stream_output_batch=message,
        )


@dataclass(frozen=True)
class DecoupledSpecIpcConfig:
    bind_endpoint: str
    connect_endpoints: tuple[str, ...]
    rank: int
