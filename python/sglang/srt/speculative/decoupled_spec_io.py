from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class DraftMeshMessageType(str, Enum):
    CONTROL_BATCH = "control_batch"
    TAIL_STREAM_OUTPUT_BATCH = "tail_stream_output_batch"


@dataclass(frozen=True)
class DraftReqKey:
    """Request identity on the drafter side.

    The original request_id is only unique within the verifier that owns it.
    src_verifier_rank keeps the drafter-side request table unambiguous when
    multiple verifier ranks send work to the same drafter rank.
    """

    src_verifier_rank: int
    request_id: str


@dataclass
class DraftSync:
    """Open or re-open a drafter request from a verifier-owned prefix.

    The verifier is the source of truth for committed tokens. DraftSync gives
    the drafter the prompt and already committed output prefix that it must
    align to before it can emit draft tail tokens.
    """

    request_id: str
    src_verifier_rank: int
    dst_drafter_rank: int
    prompt_token_ids: list[int] = field(default_factory=list)
    committed_outputs: list[int] = field(default_factory=list)

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

    committed_tokens is the verifier-committed contiguous output segment:
    output_ids[
        pre_verify_committed_len:
        pre_verify_committed_len + len(committed_tokens)
    ].
    Drafter must align its reqs to these committed tokens,
    and sometimes needs to truncate tokens / reprefill.
    """

    request_id: str
    src_verifier_rank: int
    dst_drafter_rank: int
    pre_verify_committed_len: int
    committed_tokens: list[int]

    @property
    def draft_key(self) -> DraftReqKey:
        return DraftReqKey(
            src_verifier_rank=int(self.src_verifier_rank),
            request_id=self.request_id,
        )

    def validate_committed_tokens(self) -> None:
        if not self.committed_tokens:
            raise ValueError(
                "VerifyCommit committed_tokens must be non-empty: "
                f"request_id={self.request_id} "
                f"pre_verify_committed_len={self.pre_verify_committed_len}"
            )
        if int(self.pre_verify_committed_len) < 0:
            raise ValueError(
                "VerifyCommit pre_verify_committed_len must be non-negative: "
                f"request_id={self.request_id} "
                f"pre_verify_committed_len={self.pre_verify_committed_len}"
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
    Drafter sends one output token to the verifier-side DraftTailBuffer.

    base_committed_len records the verifier prefix length that the drafter used
    as the base when this token was emitted. The verifier compares it with its
    stale-base boundary before accepting the token as tail data or as
    pending-prefix confirmation.

    new_token_pos is the 0-based output token position for new_token. Normal
    decode streams send the latest generated token.
    """

    src_drafter_rank: int
    dst_verifier_rank: int
    request_id: str
    base_committed_len: int
    new_token_pos: int
    new_token: int


@dataclass
class DraftTailStreamOutputBatch:
    outputs: list[DraftTailStreamOutput] = field(default_factory=list)


@dataclass
class DraftControlBatch:
    dst_drafter_rank: int
    sync_messages: list[DraftSync] = field(default_factory=list)
    verify_commit_messages: list[VerifyCommit] = field(default_factory=list)
    close_messages: list[DraftClose] = field(default_factory=list)


@dataclass
class VerifierCommitSegment:
    """Contiguous VerifyCommit messages coalesced for one drafter request.

    When receiving contiguous VerifyCommit messages for the same draft req,
    the transport thread(TokenSync thread at drafter side) coalesces them into a single VerifierCommitSegment.

    VerifierCommitSegment represents a contiguous verifier-committed token segment for drafter,
    and drafter scheduler should align with these segments before emitting tail tokens
    """

    draft_key: DraftReqKey
    dst_drafter_rank: int
    pre_verify_committed_len: int
    committed_tokens: list[int] = field(default_factory=list)

    @property
    def end_committed_len(self) -> int:
        return int(self.pre_verify_committed_len) + len(self.committed_tokens)

    def append_message(self, message: VerifyCommit) -> None:
        """
        It runs on TokenSyncThread under _pending_lock. That loop only
        catches zmq.error.ContextTerminated, so a raise here escapes _run and
        silently kills the drafter control thread. It then stops applying
        ALL requests' controls while the verifier keeps pushing.

        TODO: 1. peer-data violations (non-contiguous / invalid len)
        should quarantine just that request (drop + add to close_keys), not
        crash the thread. 2. phase 5.c will handle the drafter failure by
        degrading the verifier into normal autoregressive decoding.
        """
        if message.draft_key != self.draft_key:
            raise RuntimeError(
                "Verifier commit segment received a commit for a different "
                f"request: segment_key={self.draft_key} message_key={message.draft_key}"
            )
        if int(message.dst_drafter_rank) != int(self.dst_drafter_rank):
            raise RuntimeError(
                "Verifier commit segment received a commit for a different "
                "drafter rank: "
                f"request_id={message.request_id} "
                f"segment_drafter_rank={self.dst_drafter_rank} "
                f"message_drafter_rank={message.dst_drafter_rank}"
            )
        message.validate_committed_tokens()
        pre_verify_committed_len = int(message.pre_verify_committed_len)
        if pre_verify_committed_len != self.end_committed_len:
            raise RuntimeError(
                "Verifier commit segment requires contiguous VerifyCommit "
                "messages: "
                f"request_id={message.request_id} "
                f"expected_pre_verify_committed_len={self.end_committed_len} "
                f"actual_pre_verify_committed_len={pre_verify_committed_len}"
            )

        token_ids = [int(token_id) for token_id in message.committed_tokens]
        self.committed_tokens.extend(token_ids)

    def extract_prefix(self, num_tokens: int) -> VerifierCommitSegment:
        num_tokens = int(num_tokens)
        if num_tokens <= 0:
            raise ValueError(
                "Verifier commit segment prefix length must be positive: "
                f"request_id={self.draft_key.request_id} num_tokens={num_tokens}"
            )
        if num_tokens > len(self.committed_tokens):
            raise ValueError(
                "Verifier commit segment prefix length exceeds segment length: "
                f"request_id={self.draft_key.request_id} "
                f"num_tokens={num_tokens} "
                f"segment_len={len(self.committed_tokens)}"
            )

        prefix_tokens = [
            int(token_id) for token_id in self.committed_tokens[:num_tokens]
        ]
        remaining_tokens = [
            int(token_id) for token_id in self.committed_tokens[num_tokens:]
        ]
        prefix_segment = VerifierCommitSegment(
            draft_key=self.draft_key,
            dst_drafter_rank=int(self.dst_drafter_rank),
            pre_verify_committed_len=int(self.pre_verify_committed_len),
            committed_tokens=prefix_tokens,
        )
        self.pre_verify_committed_len = int(self.pre_verify_committed_len) + num_tokens
        self.committed_tokens = remaining_tokens
        return prefix_segment


@dataclass
class DraftMeshMessage:
    message_type: DraftMeshMessageType
    control_batch: Optional[DraftControlBatch] = None
    tail_stream_output_batch: Optional[DraftTailStreamOutputBatch] = None

    @staticmethod
    def from_control_batch(message: DraftControlBatch) -> DraftMeshMessage:
        return DraftMeshMessage(
            message_type=DraftMeshMessageType.CONTROL_BATCH,
            control_batch=message,
        )

    @staticmethod
    def from_tail_stream_output_batch(
        message: DraftTailStreamOutputBatch,
    ) -> DraftMeshMessage:
        return DraftMeshMessage(
            message_type=DraftMeshMessageType.TAIL_STREAM_OUTPUT_BATCH,
            tail_stream_output_batch=message,
        )


@dataclass(frozen=True)
class DecoupledSpecIpcConfig:
    bind_endpoint: str
    connect_endpoints: tuple[str, ...]
    rank: int
