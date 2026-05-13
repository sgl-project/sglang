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
    PUSHED = "transfer_pushed"
    DONE = "transfer_done"
    ALLOC_ACCEPTED = "transfer_alloc_accepted"
    ALLOC_REJECT = "transfer_alloc_reject"

    # DiffusionServer → Instance
    ALLOC = "transfer_alloc"
    PEER_INFO = "transfer_peer_info"
    READY = "transfer_ready"
    FAILED = "transfer_failed"
    ABORT = "transfer_abort"

    # Registration
    REGISTER = "transfer_register"
    # Reserved for a future registration handshake; currently unused.
    REGISTER_ACK = "transfer_register_ack"


@dataclass
class TransferStagedMsg:
    msg_type: str = TransferMsgType.STAGED
    request_id: str = ""
    data_size: int = 0
    meta_size: int = 0
    session_id: str = ""
    pool_ptr: int = 0
    slot_offset: int = 0
    meta_pool_ptr: int = 0
    meta_slot_offset: int = 0


@dataclass
class TransferAllocMsg:
    msg_type: str = TransferMsgType.ALLOC
    request_id: str = ""
    data_size: int = 0
    meta_size: int = 0
    receiver_session_id: str = ""
    source_role: str = ""
    source_instance: int = -1
    source_control_endpoint: str = ""
    source_host_id: str = ""
    preallocated_slot: dict | None = None


@dataclass
class TransferPushedMsg:
    msg_type: str = TransferMsgType.PUSHED
    request_id: str = ""
    success: bool = True
    error: str | None = None
    source_session_id: str = ""
    dest_session_id: str = ""
    receiver_role: str = ""
    receiver_instance: int = -1


@dataclass
class TransferAllocAcceptedMsg:
    msg_type: str = TransferMsgType.ALLOC_ACCEPTED
    request_id: str = ""
    receiver_role: str = ""
    receiver_instance: int = -1
    receiver_session_id: str = ""
    receiver_slot_offset: int = 0
    receiver_slot_size: int = 0
    receiver_meta_slot_offset: int = 0
    receiver_meta_slot_size: int = 0
    data_size: int = 0
    meta_size: int = 0
    prealloc_slot_id: int | None = None


@dataclass
class TransferAllocRejectMsg:
    msg_type: str = TransferMsgType.ALLOC_REJECT
    request_id: str = ""
    receiver_role: str = ""
    receiver_instance: int = -1
    receiver_session_id: str = ""
    retryable: bool = True
    reason: str = ""
    prealloc_slot_id: int | None = None


@dataclass
class TransferPeerInfoMsg:
    msg_type: str = TransferMsgType.PEER_INFO
    request_id: str = ""
    dest_session_id: str = ""
    dest_addr: int = 0
    transfer_size: int = 0
    meta_dest_addr: int = 0
    meta_transfer_size: int = 0
    receiver_role: str = ""
    receiver_instance: int = -1
    receiver_control_endpoint: str = ""
    receiver_host_id: str = ""
    receiver_supports_local_copy: bool = False
    dest_shm_name: str | None = None
    dest_shm_offset: int = 0
    meta_dest_shm_name: str | None = None
    meta_dest_shm_offset: int = 0
    prealloc_slot_id: int | None = None


@dataclass
class TransferReadyMsg:
    msg_type: str = TransferMsgType.READY
    request_id: str = ""
    source_session_id: str = ""
    dest_session_id: str = ""
    dest_slot_offset: int = 0
    dest_meta_slot_offset: int = 0
    data_size: int = 0
    meta_size: int = 0
    receiver_role: str = ""
    receiver_instance: int = -1
    prealloc_slot_id: int | None = None


@dataclass
class TransferFailedMsg:
    msg_type: str = TransferMsgType.FAILED
    request_id: str = ""
    error: str = ""
    receiver_role: str = ""
    receiver_instance: int = -1
    source_session_id: str = ""
    dest_session_id: str = ""
    prealloc_slot_id: int | None = None


@dataclass
class TransferAbortMsg:
    msg_type: str = TransferMsgType.ABORT
    request_id: str = ""
    reason: str = ""
    source: str = "server_discard"


@dataclass
class TransferDoneMsg:
    msg_type: str = TransferMsgType.DONE
    request_id: str = ""
    result_frames: list[bytes] | None = None
    error: str | None = None
    staged_for_decoder: bool = False
    session_id: str = ""
    pool_ptr: int = 0
    slot_offset: int = 0
    meta_pool_ptr: int = 0
    meta_slot_offset: int = 0
    data_size: int = 0
    meta_size: int = 0


@dataclass
class TransferRegisterMsg:
    msg_type: str = TransferMsgType.REGISTER
    role: str = ""
    instance_id: int = 0
    session_id: str = ""
    pool_ptr: int = 0
    pool_size: int = 0
    control_endpoint: str = ""
    work_endpoint: str = ""
    rank0_only: bool = True
    role_device: str = "auto"
    host_id: str = ""
    supports_local_copy: bool = False
    data_shm_name: str | None = None
    meta_pool_ptr: int = 0
    meta_pool_size: int = 0
    meta_shm_name: str | None = None
    capacity_slots: int = 0
    capacity_slot_size: int = 0
    # Pre-allocated receive slots:
    # [{"slot_id": int, "offset": int, "size": int, "addr": int,
    #   "meta_offset": int, "meta_size": int, "meta_addr": int}]
    preallocated_slots: list = field(default_factory=list)


def encode_transfer_msg(msg: Any) -> list[bytes]:
    """Encode as [TRANSFER_MAGIC, json_payload_bytes]."""
    if hasattr(msg, "__dataclass_fields__"):
        d = asdict(msg)
    elif isinstance(msg, dict):
        d = msg
    else:
        raise TypeError(f"Cannot encode transfer message: {type(msg)}")

    d.pop("result_frames", None)

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
