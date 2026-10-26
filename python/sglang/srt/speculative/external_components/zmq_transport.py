"""ZMQ implementation of the external speculative decoding transport."""

import logging
from collections import deque

import msgspec
import msgspec.msgpack
import zmq

from sglang.srt.speculative.external_components.base_transport import (
    ExternalDraftTransport,
)
from sglang.srt.speculative.external_components.contract import (
    AllMessages,
    CancelInferenceRequest,
    CancelInferenceResponse,
    ConfigureDraftEngineRequest,
    ConfigureDraftEngineResponse,
    DraftContext,
    DraftTokensReply,
    DraftTokensRequest,
    PrefillCacheRequest,
    PrefillContext,
)

logger = logging.getLogger(__name__)


class ZmqTransport(ExternalDraftTransport):
    """ZMQ DEALER-to-ROUTER transport.

    Both control and data flow over a single DEALER socket connecting to
    the drafter's ROUTER endpoint.
    """

    def __init__(
        self,
        *,
        endpoint: str,
        draft_timeout_ms: int = 30_000,
        control_timeout_ms: int = 1_800_000,
    ):
        self._endpoint = endpoint
        self._draft_timeout_ms = draft_timeout_ms
        self._control_timeout_ms = control_timeout_ms
        self._socket = None
        self._zmq_context = None
        self._encoder = msgspec.msgpack.Encoder()
        self._decoder = msgspec.msgpack.Decoder(AllMessages)
        # A drafter may drop a draft in flight when its request is cancelled, and
        # sends no reply for that id after the cancel's ack; ids are never reused.
        self._cancelled_ids: set[int] = set()
        self._unacked_cancels: deque[list[int]] = deque()

    def open(self) -> None:
        self._zmq_context = zmq.Context()
        self._socket = self._zmq_context.socket(zmq.DEALER)
        self._socket.setsockopt(zmq.RCVTIMEO, self._draft_timeout_ms)
        self._socket.setsockopt(zmq.LINGER, 0)
        self._socket.connect(self._endpoint)
        logger.info("ZmqTransport connected to %s", self._endpoint)

    def close(self) -> None:
        if self._socket is not None:
            self._socket.close()
            self._socket = None
        if self._zmq_context is not None:
            self._zmq_context.term()
            self._zmq_context = None

    def _send(self, msg: msgspec.Struct) -> None:
        self._socket.send(self._encoder.encode(msg))

    def _recv(self, *, timeout_ms: int | None = None) -> AllMessages:
        if timeout_ms is not None:
            self._socket.setsockopt(zmq.RCVTIMEO, timeout_ms)
        try:
            return self._decoder.decode(self._socket.recv())
        finally:
            if timeout_ms is not None:
                self._socket.setsockopt(zmq.RCVTIMEO, self._draft_timeout_ms)

    # -- Public API ----------------------------------------------------------

    def configure_draft_engine(
        self, *, num_draft_tokens: int, protocol_version: int = 1
    ) -> None:
        req = ConfigureDraftEngineRequest(
            num_draft_tokens=num_draft_tokens,
            protocol_version=protocol_version,
        )
        self._send(req)
        resp = self._recv(timeout_ms=self._control_timeout_ms)
        if not isinstance(resp, ConfigureDraftEngineResponse):
            raise RuntimeError(
                f"Expected ConfigureDraftEngineResponse, got {type(resp).__name__}"
            )
        logger.info(
            "Drafter configured: num_draft_tokens=%d, protocol_version=%d",
            num_draft_tokens,
            protocol_version,
        )

    def prefill_cache(self, *, prefill_contexts: list[PrefillContext]) -> None:
        if not prefill_contexts:
            return
        self._send(PrefillCacheRequest(prefill_contexts=prefill_contexts))

    def request_drafts(self, *, draft_contexts: list[DraftContext]) -> None:
        if not draft_contexts:
            return
        self._send(DraftTokensRequest(draft_contexts=draft_contexts))

    def receive_drafts(
        self, *, request_ids: list[int], num_draft_tokens: int
    ) -> dict[int, list[int] | None]:
        result: dict[int, list[int] | None] = {}
        expected = set()
        for request_id in request_ids:
            if request_id in self._cancelled_ids:
                result[request_id] = None
            else:
                expected.add(request_id)
        # The drafter completes requests individually and sends one
        # DraftTokensReply per request, so a batch of N requests needs N recvs.
        while expected:
            try:
                reply = self._recv()
            except zmq.Again:
                logger.error(
                    "No draft reply within %d ms for request_ids=%s",
                    self._draft_timeout_ms,
                    sorted(expected),
                )
                # The worker aborts these requests and cancels them; a late
                # reply arrives before that cancel's ack and is dropped.
                self._cancelled_ids.update(expected)
                result.update(dict.fromkeys(expected))
                break
            if isinstance(reply, CancelInferenceResponse) and self._unacked_cancels:
                self._cancelled_ids.difference_update(self._unacked_cancels.popleft())
                continue
            if not isinstance(reply, DraftTokensReply):
                raise RuntimeError(
                    f"Expected DraftTokensReply, got {type(reply).__name__}"
                )
            if reply.request_id in self._cancelled_ids:
                # Late reply for a request cancelled while its draft was in flight.
                continue
            if reply.request_id not in expected:
                raise RuntimeError(
                    f"Unexpected request_id {reply.request_id} in draft reply "
                    f"(expected one of {request_ids})"
                )
            if len(reply.token_ids) == 0:
                logger.warning(
                    "Drafter reported error for request_id=%d "
                    "(prefill may have failed or timed out)",
                    reply.request_id,
                )
                result[reply.request_id] = None
                expected.discard(reply.request_id)
                continue
            if len(reply.token_ids) != num_draft_tokens:
                raise RuntimeError(
                    f"Draft reply for request_id={reply.request_id} has "
                    f"{len(reply.token_ids)} tokens, expected {num_draft_tokens}"
                )
            result[reply.request_id] = reply.token_ids
            expected.discard(reply.request_id)
        return result

    def cancel_inference(self, *, request_ids: list[int]) -> None:
        if not request_ids:
            return
        self._send(CancelInferenceRequest(request_ids=request_ids))
        self._cancelled_ids.update(request_ids)
        self._unacked_cancels.append(list(request_ids))
