"""Fake SGLang worker that publishes KV-cache events, for kind E2E testing.

Unlike ``fake_worker.py`` this one advertises a ``kv_events`` block on
``/server_info`` and runs a real ZMQ PUB socket speaking SGLang's wire format
(3-frame multipart: topic, big-endian i64 seq, msgpack ``KVEventBatch``), so the
router's real subscriber path is exercised end to end.

Events are emitted **only** when the test asks for them, via ``/control/store``.
That is deliberate: a worker that emitted on a timer would make every
view-comparison assertion a race against the next event. The test drives the
event stream, then quiesces, then compares.

Endpoints beyond the usual worker surface:

  POST /control/store   {"chains": [[h1, h2, ...], ...], "dp_rank": 0}
                        -> publish one BlockStored batch per chain; returns the
                           seq numbers used, so the test can wait for routers to
                           catch up to a known watermark instead of sleeping.
  POST /control/remove  {"hashes": [h, ...], "dp_rank": 0}
  GET  /control/state   -> {"last_seq": {dp_rank: seq}}
"""

from __future__ import annotations

import asyncio
import os
import threading
import time

import msgspec
import uvicorn
import zmq
from fastapi import FastAPI, Request

app = FastAPI()

MODEL_ID = os.environ.get("MODEL_ID", "tiny")
BLOCK_SIZE = int(os.environ.get("BLOCK_SIZE", "4"))
DP_SIZE = int(os.environ.get("DP_SIZE", "1"))
KV_PORT_BASE = int(os.environ.get("KV_PORT_BASE", "5557"))
POD_IP = os.environ.get("POD_IP", "0.0.0.0")


class _Publisher:
    """One PUB socket per DP rank, mirroring ``ZmqEventPublisher``.

    The seq counter starts at 1 and is monotonic per rank, matching the engine.
    A lock serialises publishes so concurrent control calls cannot interleave a
    frame or hand out a duplicate seq.
    """

    def __init__(self, dp_rank: int) -> None:
        self._ctx = zmq.Context.instance()
        self._sock = self._ctx.socket(zmq.PUB)
        self._sock.set_hwm(100_000)
        self._sock.bind(f"tcp://0.0.0.0:{KV_PORT_BASE + dp_rank}")
        self._seq = 0
        self._lock = threading.Lock()
        self._encoder = msgspec.msgpack.Encoder()

    def publish(self, events: list[list]) -> int:
        # `EventBatch` is msgspec `array_like`: [ts, events, attn_dp_rank].
        batch = [time.time(), events, None]
        with self._lock:
            self._seq += 1
            seq = self._seq
            self._sock.send_multipart(
                (b"", seq.to_bytes(8, "big"), self._encoder.encode(batch))
            )
        return seq

    @property
    def last_seq(self) -> int:
        with self._lock:
            return self._seq


_publishers: dict[int, _Publisher] = {}


@app.on_event("startup")
async def _bind_publishers() -> None:
    for rank in range(DP_SIZE):
        _publishers[rank] = _Publisher(rank)
    # ZMQ PUB drops messages published before a subscriber has finished
    # connecting ("slow joiner"). The router connects on discovery; the test
    # additionally waits for a non-zero tree before asserting, so this short
    # settle only reduces noise in the logs.
    await asyncio.sleep(0.2)


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.get("/server_info")
async def server_info():
    return {
        "served_model_name": MODEL_ID,
        # No speculative decoding -> unigram block hashing.
        "speculative_algorithm": None,
        "kv_events": {
            "publisher": "zmq",
            # The router replaces a wildcard host with the host from the worker
            # URL; report the pod IP explicitly so it is unambiguous.
            "endpoint_host": POD_IP,
            "endpoint_port_base": KV_PORT_BASE,
            "topic": "",
            "block_size": BLOCK_SIZE,
            "dp_size": DP_SIZE,
        },
    }


@app.get("/v1/models")
async def models():
    return {
        "object": "list",
        "data": [
            {"id": MODEL_ID, "object": "model", "created": 0, "owned_by": "sglang"}
        ],
    }


@app.post("/control/store")
async def control_store(request: Request):
    payload = await request.json()
    dp_rank = int(payload.get("dp_rank", 0))
    chains: list[list[int]] = payload["chains"]
    seqs = []
    for chain in chains:
        # One BlockStored per block, parent-chained, exactly as
        # `_record_store_event` emits them. Each event is a TAGGED ARRAY —
        # msgspec `tag=True` puts the class name in element 0 — not a map:
        # ["BlockStored", block_hashes, parent_block_hash, token_ids,
        #  block_size, lora_id, medium].
        events = []
        parent = None
        for h in chain:
            events.append(["BlockStored", [h], parent, [], BLOCK_SIZE, None, None])
            parent = h
        seqs.append(_publishers[dp_rank].publish(events))
    return {"seqs": seqs, "last_seq": _publishers[dp_rank].last_seq}


@app.post("/control/remove")
async def control_remove(request: Request):
    payload = await request.json()
    dp_rank = int(payload.get("dp_rank", 0))
    # ["BlockRemoved", block_hashes, medium]
    seq = _publishers[dp_rank].publish([["BlockRemoved", payload["hashes"], None]])
    return {"seq": seq, "last_seq": _publishers[dp_rank].last_seq}


@app.get("/control/state")
async def control_state():
    return {"last_seq": {rank: p.last_seq for rank, p in _publishers.items()}}


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    payload = await request.json()
    messages = payload.get("messages", [])
    last = messages[-1]["content"] if messages else ""
    return {
        "id": "chatcmpl-mock",
        "object": "chat.completion",
        "model": payload.get("model", MODEL_ID),
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": last},
                "finish_reason": "stop",
            }
        ],
    }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
