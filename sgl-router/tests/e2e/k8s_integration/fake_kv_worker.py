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
                        -> publish one BlockStored batch per chain; returns
                           {"last_seq": N}, so the test can wait for routers to
                           catch up to a known watermark instead of sleeping.
"""

from __future__ import annotations

import os
import threading
import time
from typing import Optional

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


# Mirrors of the engine's event structs (`EventBatch`, `KVCacheEvent`,
# `BlockStored` in python/sglang/srt/disaggregation/kv_events.py),
# declared with the same msgspec options so msgspec itself produces the engine's
# encoding: the batch is an array `[ts, events, attn_dp_rank]`, and each event
# is a tagged MAP whose `type` key carries the class name. The router rejects
# the older tagged-array event shape outright, so hand-building events here
# would test nothing but a decode failure.
class _KVCacheEvent(msgspec.Struct, omit_defaults=True, gc=False, tag=True):
    pass


class BlockStored(_KVCacheEvent):
    block_hashes: list[int]
    parent_block_hash: Optional[int]
    token_ids: list[int]
    block_size: int
    lora_id: Optional[int]
    medium: Optional[str] = None


class _EventBatch(msgspec.Struct, array_like=True, gc=False):
    ts: float
    events: list[BlockStored]
    attn_dp_rank: Optional[int] = None


class _Publisher:
    """One PUB socket per DP rank, mirroring ``ZmqEventPublisher``.

    The seq counter starts at 0 and is monotonic per rank, matching the
    engine's ``itertools.count()``; the router resolves a rank whose stream it
    received from batch 0 without a peer snapshot. ``last_seq`` is -1 until the
    first publish.
    A lock serialises publishes so concurrent control calls cannot interleave a
    frame or hand out a duplicate seq.
    """

    def __init__(self, dp_rank: int) -> None:
        self._sock = zmq.Context.instance().socket(zmq.PUB)
        self._sock.set_hwm(100_000)
        self._sock.bind(f"tcp://0.0.0.0:{KV_PORT_BASE + dp_rank}")
        self._seq = -1
        self._lock = threading.Lock()
        self._encoder = msgspec.msgpack.Encoder()

    def publish(self, events: list[BlockStored]) -> None:
        batch = _EventBatch(ts=time.time(), events=events)
        with self._lock:
            self._seq += 1
            self._sock.send_multipart(
                (b"", self._seq.to_bytes(8, "big"), self._encoder.encode(batch))
            )

    @property
    def last_seq(self) -> int:
        with self._lock:
            return self._seq


# Bound before the HTTP server starts, so the sockets exist before the readiness
# probe can pass and the router can discover this worker and connect its SUB.
_publishers: dict[int, _Publisher] = {rank: _Publisher(rank) for rank in range(DP_SIZE)}


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
    for chain in chains:
        # One BlockStored per block, parent-chained, exactly as
        # `_record_store_event` emits them.
        events = []
        parent = None
        for h in chain:
            events.append(
                BlockStored(
                    block_hashes=[h],
                    parent_block_hash=parent,
                    token_ids=[],
                    block_size=BLOCK_SIZE,
                    lora_id=None,
                )
            )
            parent = h
        _publishers[dp_rank].publish(events)
    return {"last_seq": _publishers[dp_rank].last_seq}


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
