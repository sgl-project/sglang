"""AFD control plane: per-step sync messages between attn and ffn instances.

The control plane carries per-step metadata (ZMQ, CPU). Decode uses it with
the handshake-time bucket list to select fixed-capacity data-plane tensors;
the serial prefill fallback still uses exact-size tensors.
"""

from __future__ import annotations

import logging
from typing import List, Optional

import msgspec
import zmq

from sglang.srt.utils.network import get_zmq_socket

logger = logging.getLogger(__name__)


class AFDSyncMsg(msgspec.Struct, tag="sync", forbid_unknown_fields=True):
    """Per-step forward metadata, sent immediately before the attn forward.

    ``child_wire_capacities`` describes the ordered children of an opt-in
    microbatch-overlap parent. It is empty for the existing serial path.
    """

    step_id: int
    # Effective token slots per attn DP rank after native DP padding.
    tokens_per_rank: List[int]
    forward_mode: int  # ForwardMode value (extend/decode/idle/...)
    # Common decode bucket. Eager SCATTERED prefill derives each pair's exact
    # first dimension from tokens_per_rank instead.
    wire_capacity: int
    # Final A-side graph decision. NCCL requires both peers to either replay
    # their paired graphs or execute eagerly for a given step.
    communication_graph: bool = False
    # Ordered child wire capacities for the opt-in decode-overlap parent. Empty
    # keeps the existing single-work path.
    child_wire_capacities: tuple[int, ...] = ()

    @property
    def effective_child_wire_capacities(self) -> tuple[int, ...]:
        """Ordered child capacities, including serial work as one child."""
        return self.child_wire_capacities or (self.wire_capacity,)


class AFDDirectProposal(
    msgspec.Struct, tag="proposal", frozen=True, forbid_unknown_fields=True
):
    """One attn instance's candidate work for the next Direct union step."""

    proposal_id: int
    tokens_per_rank: tuple[int, ...]
    forward_mode: int
    wire_capacity: int
    can_run_graph: bool = False
    child_wire_capacities: tuple[int, ...] = ()


class AFDDirectCommonPlan(
    msgspec.Struct, tag="plan", frozen=True, forbid_unknown_fields=True
):
    """F-selected union-wide execution plan returned to every attn instance."""

    union_step_id: int
    observed_proposal_ids: tuple[int, ...]
    active_mask: tuple[bool, ...]
    forward_mode: int
    parent_wire_capacity: int
    wire_capacity: int
    communication_graph: bool = False
    child_wire_capacities: tuple[int, ...] = ()


def _check_direct_plan(
    plan: AFDDirectCommonPlan,
    session_id: int,
    proposal: AFDDirectProposal,
    union_step_id: int,
) -> bool:
    if plan.union_step_id != union_step_id:
        raise RuntimeError(
            "AFD Direct union_step_id out of order: "
            f"expected {union_step_id}, got {plan.union_step_id}"
        )
    if len(plan.observed_proposal_ids) != len(plan.active_mask):
        raise RuntimeError("AFD Direct proposal ids and active mask size mismatch")
    if session_id >= len(plan.active_mask):
        raise RuntimeError(f"AFD Direct plan does not contain session {session_id}")
    observed_proposal_id = plan.observed_proposal_ids[session_id]
    active = plan.active_mask[session_id]
    if observed_proposal_id > proposal.proposal_id:
        raise RuntimeError(
            "AFD Direct plan observed a future proposal for session "
            f"{session_id}: current={proposal.proposal_id}, "
            f"observed={observed_proposal_id}"
        )
    if active and observed_proposal_id != proposal.proposal_id:
        raise RuntimeError(
            "AFD Direct plan selected the wrong proposal for session "
            f"{session_id}: current={proposal.proposal_id}, "
            f"observed={observed_proposal_id}"
        )
    return observed_proposal_id == proposal.proposal_id


_sync_decoder = msgspec.msgpack.Decoder(AFDSyncMsg)
_proposal_decoder = msgspec.msgpack.Decoder(AFDDirectProposal)
_plan_decoder = msgspec.msgpack.Decoder(AFDDirectCommonPlan)


class AFDSyncMsgSender:
    """Attn side: PUSH socket owned by the single global sender rank."""

    def __init__(self, endpoint: str):
        self._socket = get_zmq_socket(
            zmq.Context.instance(), zmq.PUSH, endpoint, bind=False
        )

    def send(self, msg: AFDSyncMsg) -> None:
        self._socket.send(msgspec.msgpack.encode(msg))

    def close(self) -> None:
        self._socket.close(linger=0)


class AFDSyncMsgReceiver:
    """Ffn side: PULL socket bound by rank 0 of the ffn instance."""

    def __init__(self, endpoint: str):
        self._socket = get_zmq_socket(
            zmq.Context.instance(), zmq.PULL, endpoint, bind=True
        )
        self._last_step_id: Optional[int] = None
        logger.info("AFD control receiver bound at %s", endpoint)

    @property
    def socket(self) -> zmq.Socket:
        return self._socket

    def recv(self) -> AFDSyncMsg:
        """Blocking receive. Asserts step_id monotonicity (fail-fast on
        reordering/loss instead of silently desyncing with the attn side)."""
        msg = _sync_decoder.decode(self._socket.recv())
        expected = 0 if self._last_step_id is None else self._last_step_id + 1
        if msg.step_id != expected:
            raise RuntimeError(
                f"AFD control step_id out of order: expected {expected}, "
                f"got {msg.step_id}"
            )
        self._last_step_id = msg.step_id
        return msg

    def close(self) -> None:
        self._socket.close(linger=0)


class AFDDirectControlClient:
    """Attn-side PAIR channel for Direct union-step planning."""

    def __init__(self, endpoint: str, session_id: int):
        if session_id < 0:
            raise ValueError("AFD Direct session_id must be non-negative")
        self._session_id = session_id
        self._socket = get_zmq_socket(
            zmq.Context.instance(), zmq.PAIR, endpoint, bind=False
        )
        self._next_proposal_id = 0
        self._next_union_step_id = 0
        self._outstanding_proposal: Optional[AFDDirectProposal] = None

    @property
    def next_proposal_id(self) -> int:
        return self._next_proposal_id

    @property
    def has_outstanding_proposal(self) -> bool:
        return self._outstanding_proposal is not None

    def send_proposal(self, proposal: AFDDirectProposal) -> None:
        outstanding = self._outstanding_proposal
        if outstanding is not None and not (
            sum(outstanding.tokens_per_rank) == 0 and sum(proposal.tokens_per_rank) > 0
        ):
            raise RuntimeError("AFD Direct client already has an outstanding proposal")
        if proposal.proposal_id != self._next_proposal_id:
            raise RuntimeError(
                "AFD Direct proposal_id out of order: "
                f"expected {self._next_proposal_id}, got {proposal.proposal_id}"
            )
        self._socket.send(msgspec.msgpack.encode(proposal))
        self._outstanding_proposal = proposal
        self._next_proposal_id += 1

    def recv_plan(self) -> AFDDirectCommonPlan:
        outstanding = self._outstanding_proposal
        if outstanding is None:
            raise RuntimeError(
                "AFD Direct client cannot receive a plan without a proposal"
            )
        plan = _plan_decoder.decode(self._socket.recv())
        consumed_current = _check_direct_plan(
            plan,
            self._session_id,
            outstanding,
            self._next_union_step_id,
        )
        if consumed_current:
            self._outstanding_proposal = None
        self._next_union_step_id += 1
        return plan

    def poll_plan(self, timeout_ms: int = 0) -> bool:
        return bool(self._socket.poll(timeout_ms, zmq.POLLIN) & zmq.POLLIN)

    def close(self) -> None:
        self._socket.close(linger=0)


class AFDDirectControlServer:
    """F-side PAIR channel for one Direct attn session."""

    def __init__(self, endpoint: str, session_id: int):
        if session_id < 0:
            raise ValueError("AFD Direct session_id must be non-negative")
        self._session_id = session_id
        self._socket = get_zmq_socket(
            zmq.Context.instance(), zmq.PAIR, endpoint, bind=True
        )
        self._next_proposal_id = 0
        self._outstanding_proposal: Optional[AFDDirectProposal] = None
        logger.info(
            "AFD Direct control server bound at %s for session %d",
            endpoint,
            session_id,
        )

    @property
    def socket(self) -> zmq.Socket:
        return self._socket

    def recv_proposal(self) -> AFDDirectProposal:
        outstanding = self._outstanding_proposal
        if outstanding is not None and sum(outstanding.tokens_per_rank) != 0:
            raise RuntimeError("AFD Direct server already has an outstanding proposal")
        proposal = _proposal_decoder.decode(self._socket.recv())
        if proposal.proposal_id != self._next_proposal_id:
            raise RuntimeError(
                "AFD Direct proposal_id out of order for session "
                f"{self._session_id}: expected {self._next_proposal_id}, "
                f"got {proposal.proposal_id}"
            )
        if outstanding is not None and sum(proposal.tokens_per_rank) == 0:
            raise RuntimeError(
                "AFD Direct can only replace an idle proposal with active work"
            )
        self._outstanding_proposal = proposal
        self._next_proposal_id += 1
        return proposal

    def send_plan(self, plan: AFDDirectCommonPlan) -> None:
        if self._outstanding_proposal is None:
            raise RuntimeError(
                "AFD Direct server cannot send a plan without a proposal"
            )
        self._socket.send(msgspec.msgpack.encode(plan))
        self._outstanding_proposal = None

    def close(self) -> None:
        self._socket.close(linger=0)
