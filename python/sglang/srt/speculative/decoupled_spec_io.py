from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Optional


class DraftMeshMessageType(str, Enum):
    CONTROL_BATCH = "control_batch"
    ENUMERATION_BUFFER_BATCH = "enumeration_buffer_batch"


@dataclass(frozen=True)
class DraftReqKey:
    """Request identity on the drafter side.

    The original request_id is only unique within the verifier that owns it.
    src_verifier_rank keeps the drafter-side request table unambiguous when
    multiple verifier ranks send work to the same drafter rank.
    """

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

    Enumeration-side interpretation
    -------------------------------
    Under the enumeration data plane (DraftEnumerationBuffer) the verifier
    GPU-selects one enumerated chain per verify round, so the newly committed
    tokens are the accepted draft tokens followed by the bonus token:

        accept_len = len(committed_tokens) - 1
        bonus      = committed_tokens[-1]

    A fallback round (no enumerated chain was usable) commits exactly the bonus
    token, i.e. committed_tokens == [bonus] and accept_len == 0. Coalescing
    contiguous commits into a VerifierCommitSegment still works because the
    drafter only needs the latest committed prefix to base the next enumeration
    round on; it never reconstructs per-round accept/bonus boundaries.
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
class DraftEnumerationBuffer:
    """Drafter pushes a full enumeration block for one request, one round ahead.

    Instead of streaming draft tail tokens one at a time, the drafter
    pre-enumerates, from a single committed base, every chain the verifier could
    possibly select in the next verify round and pushes the whole block at once.
    The verifier then GPU-selects the matching chain; a branch that does not
    match is simply never selected, so a wrong guess is never committed.

    Enumeration dimensions
    ----------------------
    - num_steps (K): speculative_num_steps, the draft chain length per case.
    - fanout (F): speculative_fanout, the number of bonus-token guesses the
      drafter enumerates per accept case.
    - accept cases (K + 1): the possible accept lengths of the *previous* round
      that this block is drafted against, 0 .. K inclusive.

    tokens layout
    -------------
    tokens is a flat tuple of (K + 1) * F * K vocab ids laid out as
    [accept_case][guess][step]:

        flat_index = (accept_case * F + guess) * K + step

    with accept_case in [0, K], guess in [0, F), step in [0, K). The value at
    that index is the draft token the drafter proposes for chain
    (accept_case, guess) at position step.

    base_committed_len is the verifier committed length this block was drafted
    from; it is the staleness version. The verifier judges usability entirely on
    GPU by comparing base_committed_len against its current committed length
    together with the bonus-token match, so no host round-trip is needed to
    decide whether the block is fresh.
    """

    src_drafter_rank: int
    dst_verifier_rank: int
    request_id: str
    base_committed_len: int
    num_steps: int
    fanout: int
    tokens: tuple[int, ...] = ()

    @property
    def draft_key(self) -> DraftReqKey:
        return DraftReqKey(
            src_verifier_rank=int(self.dst_verifier_rank),
            request_id=self.request_id,
        )

    @property
    def num_tokens(self) -> int:
        return (int(self.num_steps) + 1) * int(self.fanout) * int(self.num_steps)

    def validate(self) -> None:
        if int(self.num_steps) < 1:
            raise ValueError(
                "DraftEnumerationBuffer num_steps must be >= 1: "
                f"request_id={self.request_id} num_steps={self.num_steps}"
            )
        if int(self.fanout) < 1:
            raise ValueError(
                "DraftEnumerationBuffer fanout must be >= 1: "
                f"request_id={self.request_id} fanout={self.fanout}"
            )
        if int(self.base_committed_len) < 0:
            raise ValueError(
                "DraftEnumerationBuffer base_committed_len must be non-negative: "
                f"request_id={self.request_id} "
                f"base_committed_len={self.base_committed_len}"
            )
        if len(self.tokens) != self.num_tokens:
            raise ValueError(
                "DraftEnumerationBuffer tokens length must equal "
                "(num_steps + 1) * fanout * num_steps: "
                f"request_id={self.request_id} "
                f"num_steps={self.num_steps} fanout={self.fanout} "
                f"expected={self.num_tokens} actual={len(self.tokens)}"
            )


@dataclass
class DraftEnumerationBufferBatch:
    buffers: list[DraftEnumerationBuffer] = field(default_factory=list)


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
class DraftControlInbox:
    """Drafter-side inbox for verifier control messages.

    The TokenSync thread temporarily stores incoming control messages here.
    The drafter scheduler extracts and consumes them each time it finishes a decoding step.
    """

    sync_messages: list[DraftSync] = field(default_factory=list)
    verifier_commit_segments: dict[DraftReqKey, VerifierCommitSegment] = field(
        default_factory=dict
    )
    close_keys: set[DraftReqKey] = field(default_factory=set)

    def is_empty(self) -> bool:
        return (
            not self.sync_messages
            and not self.verifier_commit_segments
            and not self.close_keys
        )

    def pending_control_count(self) -> int:
        return (
            len(self.sync_messages)
            + len(self.verifier_commit_segments)
            + len(self.close_keys)
        )

    def add_control_batch_locked(self, batch: DraftControlBatch) -> None:
        for message in batch.close_messages:
            self.add_close_key_locked(message.draft_key)
        for message in batch.sync_messages:
            if message.draft_key not in self.close_keys:
                self.sync_messages.append(message)
        for message in batch.verify_commit_messages:
            self.add_verify_commit_locked(message)

    def add_close_key_locked(self, draft_key: DraftReqKey) -> None:
        self.close_keys.add(draft_key)
        self.verifier_commit_segments.pop(draft_key, None)
        self.sync_messages = [
            message for message in self.sync_messages if message.draft_key != draft_key
        ]

    def add_verify_commit_locked(self, message: VerifyCommit) -> None:
        if message.draft_key in self.close_keys:
            return
        segment = self.verifier_commit_segments.get(message.draft_key)
        if segment is None:
            segment = VerifierCommitSegment(
                draft_key=message.draft_key,
                dst_drafter_rank=int(message.dst_drafter_rank),
                pre_verify_committed_len=int(message.pre_verify_committed_len),
            )
            segment.append_message(message)
            self.verifier_commit_segments[message.draft_key] = segment
            return
        segment.append_message(message)

    def extract_ready_controls_locked(
        self,
        consumable_commit_len: Callable[[VerifierCommitSegment], int],
    ) -> ReadyDraftControls:
        ready_controls = ReadyDraftControls()

        if self.close_keys:
            ready_controls.close_keys = self.close_keys
            self.close_keys = set()

        if self.sync_messages:
            ready_controls.sync_messages = self.sync_messages
            self.sync_messages = []

        for draft_key, segment in list(self.verifier_commit_segments.items()):
            consumable_len = consumable_commit_len(segment)
            if consumable_len <= 0:
                continue

            ready_controls.ready_commit_segments.append(
                segment.extract_prefix(consumable_len)
            )
            if not segment.committed_tokens:
                self.verifier_commit_segments.pop(draft_key, None)

        return ready_controls


@dataclass
class ReadyDraftControls:
    sync_messages: list[DraftSync] = field(default_factory=list)
    close_keys: set[DraftReqKey] = field(default_factory=set)
    ready_commit_segments: list[VerifierCommitSegment] = field(default_factory=list)

    def is_empty(self) -> bool:
        return (
            not self.sync_messages
            and not self.close_keys
            and not self.ready_commit_segments
        )

    def extracted_control_count(self) -> int:
        return (
            len(self.sync_messages)
            + len(self.close_keys)
            + len(self.ready_commit_segments)
        )


@dataclass
class DraftMeshMessage:
    message_type: DraftMeshMessageType
    control_batch: Optional[DraftControlBatch] = None
    enumeration_buffer_batch: Optional[DraftEnumerationBufferBatch] = None

    @staticmethod
    def from_control_batch(message: DraftControlBatch) -> DraftMeshMessage:
        return DraftMeshMessage(
            message_type=DraftMeshMessageType.CONTROL_BATCH,
            control_batch=message,
        )

    @staticmethod
    def from_enumeration_buffer_batch(
        message: DraftEnumerationBufferBatch,
    ) -> DraftMeshMessage:
        return DraftMeshMessage(
            message_type=DraftMeshMessageType.ENUMERATION_BUFFER_BATCH,
            enumeration_buffer_batch=message,
        )


@dataclass(frozen=True)
class DecoupledSpecIpcConfig:
    bind_endpoint: str
    connect_endpoints: tuple[str, ...]
    rank: int
