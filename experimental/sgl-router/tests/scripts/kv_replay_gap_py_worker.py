#!/usr/bin/env python3
# Copyright 2026 The SGLang Authors
# Licensed under the Apache License, Version 2.0
"""Small Python fake worker for KV replay gap validation.

This is a manual/automation harness, not a model server. It exposes the three
interfaces the Rust router needs:

* HTTP `/server_info` with live and replay KV event metadata.
* ZMQ PUB for live KV events.
* ZMQ ROUTER for replay requests, using `ZmqEventPublisher`'s wire format.

Example command stream after startup prints `READY {...}`:

    publish 1 10
    buffer 2 20
    live 3 30
    stop

`publish` sends a live event and stores it in the replay buffer. `buffer`
stores an event without publishing it, which deterministically creates a live
sequence gap while keeping replay able to recover it. `live` sends without
storing, useful when the replay socket should return only the missing batch.
"""

from __future__ import annotations

import argparse
import json
import struct
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Dict

END_SEQ = (-1).to_bytes(8, "big", signed=True)


def pack_array_len(n: int) -> bytes:
    if n < 16:
        return bytes([0x90 | n])
    if n <= 0xFFFF:
        return b"\xdc" + n.to_bytes(2, "big")
    raise ValueError(f"array too large: {n}")


def pack_str(value: str) -> bytes:
    raw = value.encode("utf-8")
    if len(raw) < 32:
        return bytes([0xA0 | len(raw)]) + raw
    if len(raw) <= 0xFF:
        return b"\xd9" + bytes([len(raw)]) + raw
    raise ValueError(f"string too large: {value!r}")


def pack_uint(value: int) -> bytes:
    if value < 0:
        raise ValueError("expected unsigned integer")
    if value < 128:
        return bytes([value])
    if value <= 0xFF:
        return b"\xcc" + bytes([value])
    if value <= 0xFFFF:
        return b"\xcd" + value.to_bytes(2, "big")
    if value <= 0xFFFFFFFF:
        return b"\xce" + value.to_bytes(4, "big")
    return b"\xcf" + value.to_bytes(8, "big")


def pack_i64(value: int) -> bytes:
    return b"\xd3" + struct.pack(">q", value)


def encode_block_stored_payload(block_hash: int, block_size: int) -> bytes:
    # KVEventBatch = [ts, [event], attn_dp_rank]
    event = b"".join(
        [
            pack_array_len(7),
            pack_str("BlockStored"),
            pack_array_len(1),
            pack_i64(block_hash),
            b"\xc0",  # parent_block_hash = None
            pack_array_len(0),  # token_ids
            pack_uint(block_size),
            b"\xc0",  # lora_id
            pack_str("GPU"),
        ]
    )
    return b"".join(
        [
            pack_array_len(3),
            b"\xcb" + struct.pack(">d", 0.0),
            pack_array_len(1),
            event,
            pack_uint(0),  # attn_dp_rank
        ]
    )


def make_handler(body: Dict[str, object]):
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            if self.path != "/server_info":
                self.send_error(404)
                return
            payload = json.dumps(body).encode("utf-8")
            self.send_response(200)
            self.send_header("content-type", "application/json")
            self.send_header("content-length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, _format: str, *_args: object) -> None:
            return

    return Handler


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--block-size", type=int, default=64)
    parser.add_argument("--buffer-steps", type=int, default=16)
    parser.add_argument("--topic", default="")
    args = parser.parse_args()

    try:
        import zmq
    except ModuleNotFoundError as exc:
        print(f"ERROR missing dependency: {exc}", file=sys.stderr, flush=True)
        return 2

    ctx = zmq.Context.instance()
    pub = ctx.socket(zmq.PUB)
    replay = ctx.socket(zmq.ROUTER)
    live_port = pub.bind_to_random_port("tcp://127.0.0.1")
    replay_port = replay.bind_to_random_port("tcp://127.0.0.1")

    body = {
        "kv_events": {
            "publisher": "zmq",
            "endpoint_host": "127.0.0.1",
            "endpoint_port_base": live_port,
            "topic": args.topic,
            "block_size": args.block_size,
            "dp_size": 1,
            "replay_endpoint_host": "127.0.0.1",
            "replay_endpoint_port_base": replay_port,
            "replay_buffer_steps": args.buffer_steps,
        }
    }
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(body))
    http_thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    http_thread.start()

    buffer: Dict[int, bytes] = {}
    running = True

    def replay_loop() -> None:
        poller = zmq.Poller()
        poller.register(replay, zmq.POLLIN)
        while running:
            events = dict(poller.poll(50))
            if replay not in events:
                continue
            frames = replay.recv_multipart()
            if len(frames) != 3:
                continue
            client_id, _empty, start_seq_bytes = frames
            start_seq = int.from_bytes(start_seq_bytes, "big")
            for seq in sorted(buffer):
                if seq >= start_seq:
                    replay.send_multipart(
                        [client_id, b"", seq.to_bytes(8, "big"), buffer[seq]]
                    )
            replay.send_multipart([client_id, b"", END_SEQ, b""])

    replay_thread = threading.Thread(target=replay_loop, daemon=True)
    replay_thread.start()

    ready = {
        "worker_url": f"http://127.0.0.1:{httpd.server_port}",
        "live_endpoint": f"tcp://127.0.0.1:{live_port}",
        "replay_endpoint": f"tcp://127.0.0.1:{replay_port}",
    }
    print("READY " + json.dumps(ready, sort_keys=True), flush=True)

    topic = args.topic.encode("utf-8")
    try:
        for raw in sys.stdin:
            parts = raw.strip().split()
            if not parts:
                continue
            cmd = parts[0]
            if cmd == "stop":
                print("OK stop", flush=True)
                break
            if cmd not in {"publish", "buffer", "live"} or len(parts) != 3:
                print(f"ERR bad command: {raw.strip()}", flush=True)
                continue
            seq = int(parts[1])
            block_hash = int(parts[2])
            payload = encode_block_stored_payload(block_hash, args.block_size)
            if cmd != "live":
                buffer[seq] = payload
            if cmd == "publish":
                pub.send_multipart([topic, seq.to_bytes(8, "big"), payload])
            elif cmd == "live":
                pub.send_multipart([topic, seq.to_bytes(8, "big"), payload])
            print(f"OK {cmd} {seq} {block_hash}", flush=True)
    finally:
        running = False
        time.sleep(0.1)
        httpd.shutdown()
        pub.close(linger=0)
        replay.close(linger=0)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
