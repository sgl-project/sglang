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

import array
import os
import struct
from typing import List, Optional, Union

import msgspec.msgpack

# v0.5 #76 (T1.4 OpenAI-bypass): when CODEC_OPENAI_BYPASS=1, the encoder
# accepts buffer-shaped ids (numpy array, array.array, bytes-of-uint32-LE)
# directly without converting to a Python list first. Saves ~25-40% codec-
# encode CPU per token in benchmarks. Default off so the JSON-SSE path is
# unchanged byte-for-byte.
_OPENAI_BYPASS = os.environ.get("CODEC_OPENAI_BYPASS", "0") == "1"

# Lazy import — numpy isn't a hard dependency of the encoder, only an
# accelerator when CODEC_OPENAI_BYPASS=1 + the caller hands us an ndarray.
try:
    import numpy as _np  # type: ignore[import-untyped]

    _HAVE_NUMPY = True
except ImportError:  # pragma: no cover - numpy is in sglang's reqs
    _np = None
    _HAVE_NUMPY = False

IdsLike = Union[List[int], "array.array", bytes, "_np.ndarray"]


def _normalise_ids_to_list(ids: IdsLike) -> List[int]:
    """Coerce IdsLike to a plain List[int].

    Always-correct fallback path: used by every encoder regardless of the
    CODEC_OPENAI_BYPASS gate. When the gate is OFF, the encoders take a
    List[int] arg directly and skip this function entirely.

    Order matters: List[int] check first since that's the common path.
    """
    if isinstance(ids, list):
        return ids
    if _HAVE_NUMPY and isinstance(ids, _np.ndarray):
        # tolist() on a uint32 ndarray is the only path that avoids the
        # per-element PyLong boxing — msgspec doesn't accept an ndarray
        # directly. The ID we save here is upstream (tokenizer_manager
        # never having created the List[int] in the first place); see
        # docs/engine-fork-tasks/v0.5-rollout.md § Task #76 for the
        # upstream half.
        return ids.tolist()
    if isinstance(ids, array.array):
        return ids.tolist()
    if isinstance(ids, (bytes, bytearray, memoryview)):
        if _HAVE_NUMPY:
            return _np.frombuffer(bytes(ids), dtype="<u4").tolist()
        # Stdlib little-endian uint32 unpack as a numpy-free fallback.
        # struct.unpack is ~10× faster than a Python list comprehension for
        # this — same shape, no Python-level loop.
        b = bytes(ids)
        if len(b) % 4 != 0:
            raise ValueError(
                f"codec_frame: bytes length {len(b)} is not a multiple of 4 (uint32 LE)"
            )
        return list(struct.unpack(f"<{len(b) // 4}I", b))
    raise TypeError(f"codec_frame: unsupported ids type {type(ids).__name__}")


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
    ids: IdsLike,
    *,
    done: bool,
    finish_reason: Optional[str] = None,
    tool_calls: Optional[List[dict]] = None,
) -> bytes:
    # The wire bytes are identical regardless of input type — msgspec
    # encodes the resulting List[int] to msgpack the same way either way.
    # The bypass path saves CPU upstream (no PyLong-list creation in
    # tokenizer_manager when output_ids surfaces as a numpy buffer).
    ids_list = ids if isinstance(ids, list) else _normalise_ids_to_list(ids)
    frame: dict = {"ids": ids_list, "done": done}
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
    ids: IdsLike,
    *,
    done: bool,
    finish_reason: Optional[str] = None,
    tool_calls: Optional[List[dict]] = None,
) -> bytes:
    """Encode a CodecFrame as protobuf with a 4-byte big-endian length prefix."""
    parts: list[bytes] = []

    # Field 1: repeated uint32 ids [packed]
    ids_list = ids if isinstance(ids, list) else _normalise_ids_to_list(ids)
    if ids_list:
        packed = b"".join(_varint(i) for i in ids_list)
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
                # Inline varint loop with the same overflow + truncation
                # protection as the outer `read_varint` helper (we can't
                # call the helper here because it works on `data` + `pos`,
                # not the sub-buffer `chunk` + `p`). Malformed input is
                # rejected with a ValueError instead of looping infinitely
                # or silently producing wrong ids.
                ids: list[int] = []
                p = 0
                while p < len(chunk):
                    n = shift = 0
                    while True:
                        if p >= len(chunk):
                            raise ValueError("Codec: truncated varint in prompt_ids")
                        b = chunk[p]
                        p += 1
                        n |= (b & 0x7F) << shift
                        if not (b & 0x80):
                            break
                        shift += 7
                        if shift > 35:
                            raise ValueError("Codec: varint overflow in prompt_ids")
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
