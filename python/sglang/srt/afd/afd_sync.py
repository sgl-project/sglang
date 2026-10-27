"""Typed per-step metadata for the AFD control plane."""

from __future__ import annotations

import logging
from typing import List, Optional

import msgspec
import zmq

from sglang.srt.utils.network import get_zmq_socket

logger = logging.getLogger(__name__)


class AFDSyncMsg(msgspec.Struct, tag="sync", forbid_unknown_fields=True):
    """Exact tensor sizes and forward mode sent before each attention step."""

    step_id: int
    tokens_per_rank: List[int]
    forward_mode: int
    wire_capacity: int


_sync_decoder = msgspec.msgpack.Decoder(AFDSyncMsg)


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
