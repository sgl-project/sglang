"""Async control client; model payloads still use the existing HTTP API."""

import asyncio
import math

import zmq
import zmq.asyncio
from sglang_simulator.compat import validate_coordinator_endpoint
from sglang_simulator.simulation.coordinator import (
    PROTOCOL_VERSION,
    Grant,
    ProtocolError,
    Ready,
    decode_grant,
    encode_message,
)


class AsyncCoordinatorClient:
    """One outstanding exchange, with wall-clock failure detection.

    A timeout makes the client unusable: retrying an ambiguous exchange could
    acknowledge results that the workload never consumed.
    """

    def __init__(self, endpoint: str, *, timeout_s: float = 30.0):
        self.endpoint = validate_coordinator_endpoint(endpoint)
        if not math.isfinite(timeout_s) or timeout_s <= 0:
            raise ValueError("timeout_s must be positive and finite")
        self.timeout_s = timeout_s
        self._context = zmq.asyncio.Context()
        self._socket = self._context.socket(zmq.REQ)
        self._socket.setsockopt(zmq.LINGER, 0)
        self._socket.setsockopt(zmq.IPV6, 1)
        self._socket.connect(self.endpoint)
        self._inflight = False
        self._closed = False
        self._now_ns = 0

    async def request(self, ready: Ready) -> Grant:
        if self._closed:
            raise ProtocolError("coordinator client is closed")
        if self._inflight:
            raise ProtocolError("only one Ready may be outstanding")
        self._inflight = True
        try:
            grant = await asyncio.wait_for(self._exchange(ready), self.timeout_s)
            if (
                grant.protocol_version != PROTOCOL_VERSION
                or grant.run_id != ready.run_id
                or grant.round != ready.round
                or grant.now_ns < self._now_ns
            ):
                raise ProtocolError("Grant does not match the outstanding Ready")
            if grant.status == "ERROR":
                raise ProtocolError(grant.error or "coordinator reported a run failure")
            self._now_ns = grant.now_ns
            if grant.status == "FINISHED":
                self.close()
            return grant
        except asyncio.TimeoutError as exc:
            self.close()
            raise ProtocolError(
                f"run={ready.run_id!r} round={ready.round}: "
                f"coordinator did not reply within {self.timeout_s} wall seconds"
            ) from exc
        except BaseException:
            self.close()
            raise
        finally:
            self._inflight = False

    async def _exchange(self, ready: Ready) -> Grant:
        await self._socket.send(encode_message(ready))
        return decode_grant(await self._socket.recv())

    def close(self) -> None:
        if not self._closed:
            self._closed = True
            self._socket.close(linger=0)
            self._context.term()

    async def __aenter__(self) -> "AsyncCoordinatorClient":  # noqa: PYI034
        return self

    async def __aexit__(self, exc_type, exc_value, traceback) -> None:
        self.close()
