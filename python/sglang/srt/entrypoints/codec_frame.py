"""
Codec binary transport utilities for SGLang.

Codec streams raw uint32 token IDs as MessagePack or Protobuf frames instead
of UTF-8 text wrapped in JSON SSE envelopes. For agent-to-agent workloads this
eliminates the text round-trip entirely: token IDs out of model A flow directly
into model B without detokenization or re-tokenization.

Wire format
-----------
MessagePack  (Content-Type: application/x-msgpack)
  {"ids": [u32, ...], "done": bool, "finish_reason": str | null}

Protobuf     (Content-Type: application/x-protobuf)
  message CodecFrame {
    repeated uint32 ids           = 1 [packed = true];
    bool            done          = 2;
    optional string finish_reason = 3;
  }
  Each frame is prefixed by a 4-byte big-endian length.

No new dependencies: msgspec is already in SGLang's requirements.
"""

from __future__ import annotations

import struct
from typing import List, Optional

import msgspec.msgpack

# ---------------------------------------------------------------------------
# Proto schema (returned by GET /codec/schema for client code generation)
# ---------------------------------------------------------------------------

PROTO_SCHEMA = """\
syntax = "proto3";

// One output chunk from the server.
message CodecFrame {
  repeated uint32 ids           = 1 [packed = true];
  bool            done          = 2;
  optional string finish_reason = 3;
  // Server-side tool-call detection (opt-in via request.tool_watcher).
  // When the model emits a complete <start>..</end> region in this
  // chunk, the parsed result rides along on the same frame whose `ids`
  // come from immediately after the region. Multiple tool calls in
  // one frame are emitted as a list.
  repeated ToolCall tool_calls  = 4;
}

message ToolCall {
  optional string name           = 1; // parsed from JSON body when shape matches
  string          arguments_json = 2; // raw JSON body between markers
  optional string id             = 3; // server-generated, e.g. "tc_<hex>"
}

// Input to POST /v1/completions/codec (bidirectional binary endpoint).
message CodecRequest {
  repeated uint32 prompt_ids    = 1 [packed = true];
  uint32          max_tokens    = 2;
  float           temperature   = 3;
  repeated string stop          = 4;
  string          stream_format = 5;
}
"""

# ---------------------------------------------------------------------------
# MessagePack
# ---------------------------------------------------------------------------

_encoder = msgspec.msgpack.Encoder()
_decoder = msgspec.msgpack.Decoder()


def encode_msgpack(
    ids: List[int],
    *,
    done: bool,
    finish_reason: Optional[str] = None,
    tool_calls: Optional[List[dict]] = None,
) -> bytes:
    frame: dict = {"ids": ids, "done": done}
    if finish_reason is not None:
        frame["finish_reason"] = finish_reason
    if tool_calls:
        # List of dicts shaped like
        # {"name": ..., "arguments_json": ..., "id": ...}
        frame["tool_calls"] = tool_calls
    return _encoder.encode(frame)


def decode_msgpack(data: bytes) -> dict:
    return _decoder.decode(data)


# ---------------------------------------------------------------------------
# Protobuf (hand-rolled; avoids a code-generation build step)
# ---------------------------------------------------------------------------


def _varint(n: int) -> bytes:
    parts: list[int] = []
    while True:
        bits = n & 0x7F
        n >>= 7
        if n == 0:
            parts.append(bits)
            break
        parts.append(bits | 0x80)
    return bytes(parts)


def _encode_tool_call_msg(call: dict) -> bytes:
    """Encode a single ToolCall sub-message (no length prefix; that's
    added by the caller as a length-delimited field in CodecFrame)."""
    parts: list[bytes] = []
    name = call.get("name")
    if name:
        b = name.encode()
        parts.append(b"\x0a" + _varint(len(b)) + b)  # field 1: string name
    args = call.get("arguments_json", "")
    bargs = args.encode()
    parts.append(
        b"\x12" + _varint(len(bargs)) + bargs
    )  # field 2: string arguments_json
    cid = call.get("id")
    if cid:
        b = cid.encode()
        parts.append(b"\x1a" + _varint(len(b)) + b)  # field 3: string id
    return b"".join(parts)


def encode_protobuf(
    ids: List[int],
    *,
    done: bool,
    finish_reason: Optional[str] = None,
    tool_calls: Optional[List[dict]] = None,
) -> bytes:
    """Encode a CodecFrame as protobuf with a 4-byte big-endian length prefix."""
    parts: list[bytes] = []

    # Field 1: repeated uint32 ids [packed]
    if ids:
        packed = b"".join(_varint(i) for i in ids)
        parts.append(b"\x0a" + _varint(len(packed)) + packed)

    # Field 2: bool done
    parts.append(b"\x10" + (b"\x01" if done else b"\x00"))

    # Field 3: optional string finish_reason
    if finish_reason:
        enc = finish_reason.encode()
        parts.append(b"\x1a" + _varint(len(enc)) + enc)

    # Field 4: repeated ToolCall tool_calls (length-delimited messages)
    if tool_calls:
        for call in tool_calls:
            sub = _encode_tool_call_msg(call)
            # Tag for field 4, wire type 2 (length-delimited) = (4 << 3) | 2 = 0x22
            parts.append(b"\x22" + _varint(len(sub)) + sub)

    payload = b"".join(parts)
    return struct.pack(">I", len(payload)) + payload


def decode_protobuf_request(data: bytes) -> dict:
    """
    Decode a CodecRequest protobuf payload (no length prefix).

    Wire-type handling:
      0 (varint)           — uint32 max_tokens (field 2)
      1 (64-bit)           — skip 8 bytes
      2 (length-delimited) — packed uint32 prompt_ids (1), stop strings (4),
                             stream_format string (5)
      5 (32-bit / float)   — temperature (field 3)
    """
    result: dict = {}
    pos = 0

    def read_varint() -> int:
        nonlocal pos
        n = shift = 0
        while True:
            if pos >= len(data):
                raise ValueError("Codec: truncated varint in CodecRequest")
            b = data[pos]
            pos += 1
            n |= (b & 0x7F) << shift
            if not (b & 0x80):
                return n
            shift += 7
            if shift > 35:
                raise ValueError("Codec: varint overflow in CodecRequest")

    while pos < len(data):
        tag = read_varint()
        field = tag >> 3
        wt = tag & 0x07

        if wt == 0:  # varint
            val = read_varint()
            if field == 2:
                result["max_tokens"] = val

        elif wt == 1:  # 64-bit fixed — skip
            pos += 8

        elif wt == 2:  # length-delimited
            length = read_varint()
            chunk = data[pos : pos + length]
            pos += length

            if field == 1:  # packed repeated uint32 prompt_ids
                ids: list[int] = []
                p = 0
                while p < len(chunk):
                    n = shift = 0
                    while True:
                        b = chunk[p]
                        p += 1
                        n |= (b & 0x7F) << shift
                        if not (b & 0x80):
                            break
                        shift += 7
                    ids.append(n)
                result["prompt_ids"] = ids
            elif field == 4:  # repeated string stop
                result.setdefault("stop", []).append(chunk.decode())
            elif field == 5:  # string stream_format
                result["stream_format"] = chunk.decode()

        elif wt == 5:  # 32-bit (float)
            (val,) = struct.unpack_from("<f", data, pos)
            pos += 4
            if field == 3:
                result["temperature"] = val

        else:
            raise ValueError(
                f"Codec: unrecognised wire type {wt} in CodecRequest field {field}"
            )

    return result


def encode_frame(
    stream_format: str,
    ids: List[int],
    *,
    done: bool,
    finish_reason: Optional[str] = None,
    tool_calls: Optional[List[dict]] = None,
) -> bytes:
    """Dispatch to the correct encoder based on stream_format."""
    if stream_format == "protobuf":
        return encode_protobuf(
            ids, done=done, finish_reason=finish_reason, tool_calls=tool_calls
        )
    return encode_msgpack(
        ids, done=done, finish_reason=finish_reason, tool_calls=tool_calls
    )
