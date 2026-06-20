"""Ringbuffer transport backend for the scheduler's request receiver.

This is the Rust alternative to the zmq `recv_from_tokenizer` socket. When the
multi-threaded Rust frontend (`sglang_server.Server`) is embedded in the
scheduler process, ingress requests are produced by the Rust TokenizerManager
into an in-process ring and drained here — no zmq hop, no extra serialization
beyond the msgpack bytes the payload already is.

GIL note: `Server.recv_requests` releases the GIL while popping from the Rust
ring (the producer threads never touch a Python object), so this drain never
holds the GIL across a wait — same non-blocking contract as `zmq.NOBLOCK`.

The bytes handed back are the exact msgpack frames the zmq path carries, so the
scheduler decodes them with the same `msgpack_decode` — this backend tracks the
IPC schema (incl. the msgpack migration) automatically.
"""

from __future__ import annotations

from typing import Any, List

from sglang.srt.managers.io_struct import msgpack_decode


class RingRequestReceiver:
    """Duck-typed stand-in for `recv_from_tokenizer`.

    `SchedulerRequestReceiver._pull_raw_reqs` checks for this marker class and
    drains via `drain()` instead of looping `sock_recv` on a zmq socket.
    """

    def __init__(self, server: Any, max_per_poll: int = 256):
        # `server` is a `sglang_server.Server` (the embedded Rust runtime).
        self._server = server
        self._max_per_poll = max_per_poll

    def drain(self, max_recv: int) -> List[Any]:
        """Non-blocking drain → list of decoded ingress request objects.

        Mirrors `sock_recv`: each raw frame is run through `msgpack_decode`, so
        the result is the same `TokenizedGenerateReqInput` / control objects the
        zmq path yields.
        """
        limit = max_recv if max_recv > 0 else self._max_per_poll
        raw_frames = self._server.recv_requests(limit)
        return [msgpack_decode(bytes(frame)) for frame in raw_frames]

    def push_chunk(self, chunk: bytes) -> bool:
        """Egress side: hand one msgpack-encoded ChunkEvent to the Rust egress
        ring (detokenizer → client stream). Returns False on backpressure."""
        return self._server.push_chunk(chunk)
