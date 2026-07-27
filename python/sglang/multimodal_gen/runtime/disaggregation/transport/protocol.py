# SPDX-License-Identifier: Apache-2.0
"""Transfer protocol messages for disaggregated diffusion.

All messages are sent as ZMQ multipart with a b"__transfer__" discriminator
in frame[0] and JSON payload in frame[1].
"""

import json
import logging
from dataclasses import asdict, dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

TRANSFER_MAGIC = b"__transfer__"


class TransferMsgType:
    # Instance → DiffusionServer
    STAGED = "transfer_staged"
    ALLOCATED = "transfer_allocated"
    PUSHED = "transfer_pushed"
    DONE = "transfer_done"

    # DiffusionServer → Instance
    ALLOC = "transfer_alloc"
    PUSH = "transfer_push"
    READY = "transfer_ready"

    # Registration
    REGISTER = "transfer_register"
    REGISTER_ACK = "transfer_register_ack"


@dataclass
class TransferStagedMsg:
    """A node finished computing and parked its output in the transfer pool.

    One staged buffer can feed several outgoing edges, so the orchestrator --
    not the producer -- decides how many times it gets pushed.
    """

    msg_type: str = TransferMsgType.STAGED
    request_id: str = ""
    data_size: int = 0
    manifest: dict = None
    session_id: str = ""
    pool_ptr: int = 0
    slot_offset: int = 0
    scalar_fields: dict = None
    # DAG node that produced this output.  Empty in classic three-role mode,
    # where the orchestrator infers the producer from the result socket.
    node: str = ""

    def __post_init__(self):
        if self.manifest is None:
            self.manifest = {}
        if self.scalar_fields is None:
            self.scalar_fields = {}


@dataclass
class TransferAllocMsg:
    msg_type: str = TransferMsgType.ALLOC
    request_id: str = ""
    data_size: int = 0
    source_role: str = ""
    edge_id: str = ""


@dataclass
class TransferAllocatedMsg:
    msg_type: str = TransferMsgType.ALLOCATED
    request_id: str = ""
    session_id: str = ""
    pool_ptr: int = 0
    slot_offset: int = 0
    slot_size: int = 0
    edge_id: str = ""


@dataclass
class TransferPushMsg:
    """Tells a producer to RDMA its staged buffer to one specific consumer.

    A fan-out produces one of these per outgoing edge; ``fanout_total`` lets
    the producer refcount its staged slot and free it only after the last
    push, since every edge reads the same bytes.
    """

    msg_type: str = TransferMsgType.PUSH
    request_id: str = ""
    dest_session_id: str = ""
    dest_addr: int = 0
    transfer_size: int = 0
    edge_id: str = ""
    fanout_total: int = 1


@dataclass
class TransferPushedMsg:
    msg_type: str = TransferMsgType.PUSHED
    request_id: str = ""
    edge_id: str = ""


@dataclass
class TransferReadyMsg:
    """Tells a consumer its data has landed.

    A join node receives one of these per incoming edge; it accumulates until
    it holds ``expected_inputs`` of them before building the request.
    """

    msg_type: str = TransferMsgType.READY
    request_id: str = ""
    manifest: dict = None
    slot_offset: int = 0
    scalar_fields: dict = None
    edge_id: str = ""
    input_index: int = 0
    expected_inputs: int = 1

    def __post_init__(self):
        if self.manifest is None:
            self.manifest = {}
        if self.scalar_fields is None:
            self.scalar_fields = {}


@dataclass
class TransferDoneMsg:
    msg_type: str = TransferMsgType.DONE
    request_id: str = ""
    error: str | None = None
    # DAG node reporting completion.  Replaces the old staged_for_decoder
    # flag: whether anything comes next is a property of the graph, which the
    # orchestrator knows and the worker does not.
    node: str = ""
    staged_for_next: bool = False


@dataclass
class TransferRegisterMsg:
    msg_type: str = TransferMsgType.REGISTER
    role: str = ""
    # DAG node name this instance serves.  Falls back to ``role`` in classic
    # three-role deployments.
    node: str = ""
    session_id: str = ""
    pool_ptr: int = 0
    pool_size: int = 0
    # The instance's own work endpoint (e.g. tcp://host:port). Used by the
    # DiffusionServer to key peer info by URL index (i.e. the same index used
    # to build the PUSH work-socket list), so the control plane and the RDMA
    # data plane cannot drift when instances register in a different order
    # than --*-urls.
    work_endpoint: str = ""
    # Pre-allocated receive slots: [{"offset": int, "size": int, "slot_id": int, "addr": int}]
    preallocated_slots: list = field(default_factory=list)


def encode_transfer_msg(msg: Any) -> list[bytes]:
    """Encode as [TRANSFER_MAGIC, json_payload_bytes]."""
    if hasattr(msg, "__dataclass_fields__"):
        d = asdict(msg)
    elif isinstance(msg, dict):
        d = msg
    else:
        raise TypeError(f"Cannot encode transfer message: {type(msg)}")

    return [TRANSFER_MAGIC, json.dumps(d, separators=(",", ":")).encode("utf-8")]


def decode_transfer_msg(frames: list[bytes]) -> dict:
    if len(frames) < 2 or frames[0] != TRANSFER_MAGIC:
        raise ValueError(f"Not a transfer message: frame[0]={frames[0]!r}")
    return json.loads(frames[1])


def is_transfer_message(frames: list) -> bool:
    return len(frames) >= 2 and (
        frames[0] == TRANSFER_MAGIC
        or (isinstance(frames[0], memoryview) and bytes(frames[0]) == TRANSFER_MAGIC)
        or (hasattr(frames[0], "bytes") and frames[0].bytes == TRANSFER_MAGIC)
    )
