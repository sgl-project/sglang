"""
Smoke test for SGLang PR #24483 — Codec binary streaming modules.

Validates the wire format produced by sglang's codec_frame.py is bit-compatible
with the @codecai/web / codecai / Codec.Net clients. Runs purely in Python on
Windows — no torch, no GPU, no full sglang server required.

Three checks:
  1. msgpack round-trip: encode CodecFrame in sglang → decode with codecai
  2. protobuf round-trip: same, with the length-prefixed protobuf form
  3. compression negotiation: verify accept-encoding parsing matches spec

Run:
    py -3.13 H:/dev/sglang/smoke_codec.py
"""
from __future__ import annotations

import struct
import sys
from pathlib import Path

# Bypass the sglang package __init__ (which imports Linux-only `resource`).
# We only need codec_frame.py and codec_compression.py — both pure-Python.
import importlib.util as _ilu

_HERE = Path(__file__).parent / "python" / "sglang" / "srt" / "entrypoints"

def _load(modname: str, path: Path):
    spec = _ilu.spec_from_file_location(modname, path)
    mod = _ilu.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

codec_frame = _load("codec_frame", _HERE / "codec_frame.py")
codec_compression = _load("codec_compression", _HERE / "codec_compression.py")

encode_msgpack         = codec_frame.encode_msgpack
decode_msgpack         = codec_frame.decode_msgpack
encode_protobuf        = codec_frame.encode_protobuf
decode_protobuf_request = codec_frame.decode_protobuf_request
PROTO_SCHEMA           = codec_frame.PROTO_SCHEMA
negotiate_encoding     = codec_compression.negotiate_encoding
_parse_accept_encoding = codec_compression._parse_accept_encoding


# ── 1. msgpack round-trip ───────────────────────────────────────────────────


def test_msgpack_round_trip():
    print("test_msgpack_round_trip")
    cases = [
        # (ids, done, finish_reason)
        ([1, 2, 3], False, None),
        ([42], True, "eos_token"),
        ([], True, "stop_sequence"),
        (list(range(1024)), False, None),
        ([0xFFFF_FFFF, 0xDEAD_BEEF, 0xCAFE_BABE], False, None),  # max-ish uint32
    ]
    for ids, done, fr in cases:
        encoded = encode_msgpack(ids, done=done, finish_reason=fr)
        decoded = decode_msgpack(encoded)
        assert decoded["ids"] == ids, f"ids mismatch: {decoded['ids']!r} != {ids!r}"
        assert decoded["done"] == done, f"done mismatch: {decoded['done']} != {done}"
        if fr is None:
            assert "finish_reason" not in decoded, (
                f"finish_reason should be absent: {decoded!r}")
        else:
            assert decoded["finish_reason"] == fr
        print(f"  ✓ ids_len={len(ids):4d}  done={done!s:5s}  fr={fr!r:18s}  bytes={len(encoded)}")


# ── 2. protobuf round-trip ──────────────────────────────────────────────────


def _decode_protobuf_frame(data: bytes) -> dict:
    """Decode a length-prefixed CodecFrame produced by encode_protobuf.
    Mirrors what @codecai/web's decodeProtobufFrame does."""
    if len(data) < 4:
        raise ValueError("truncated length prefix")
    (length,) = struct.unpack(">I", data[:4])
    if len(data) - 4 < length:
        raise ValueError(f"truncated payload: declared {length}, have {len(data) - 4}")
    body = data[4:4 + length]

    result = {"ids": [], "done": False}
    pos = 0

    def read_varint():
        nonlocal pos
        n = shift = 0
        while True:
            b = body[pos]; pos += 1
            n |= (b & 0x7F) << shift
            if not (b & 0x80):
                return n
            shift += 7

    while pos < len(body):
        tag = read_varint()
        field = tag >> 3
        wt = tag & 0x07
        if field == 1 and wt == 2:           # repeated packed uint32 ids
            ln = read_varint()
            end = pos + ln
            while pos < end:
                result["ids"].append(read_varint())
        elif field == 2 and wt == 0:         # bool done
            v = read_varint()
            result["done"] = bool(v)
        elif field == 3 and wt == 2:         # string finish_reason
            ln = read_varint()
            result["finish_reason"] = body[pos:pos + ln].decode("utf-8")
            pos += ln
        else:
            # Skip unknown field
            if wt == 0:
                read_varint()
            elif wt == 2:
                ln = read_varint()
                pos += ln
            elif wt == 5:
                pos += 4
            elif wt == 1:
                pos += 8
            else:
                raise ValueError(f"unknown wire type {wt}")
    return result


def test_protobuf_round_trip():
    print("test_protobuf_round_trip")
    cases = [
        ([1, 2, 3], False, None),
        ([42], True, "eos_token"),
        ([], True, "error"),
        (list(range(1024)), False, None),
    ]
    for ids, done, fr in cases:
        encoded = encode_protobuf(ids, done=done, finish_reason=fr)
        # Length prefix sanity
        (declared,) = struct.unpack(">I", encoded[:4])
        assert declared == len(encoded) - 4, (
            f"length prefix mismatch: declared {declared}, body {len(encoded) - 4}")
        decoded = _decode_protobuf_frame(encoded)
        assert decoded["ids"] == ids
        assert decoded["done"] == done
        if fr is not None:
            assert decoded["finish_reason"] == fr
        print(f"  ✓ ids_len={len(ids):4d}  done={done!s:5s}  fr={fr!r:18s}  bytes={len(encoded)}")


def test_protobuf_request_decode():
    """Decode a CodecRequest protobuf produced by hand."""
    print("test_protobuf_request_decode")

    # Build a CodecRequest with prompt_ids=[1,2,3], max_tokens=128, stream_format="msgpack"
    def varint(n):
        out = b""
        while n >= 0x80:
            out += bytes([(n & 0x7F) | 0x80]); n >>= 7
        out += bytes([n])
        return out

    parts = []
    # field 1: packed uint32 prompt_ids
    packed = b"".join(varint(i) for i in [1, 2, 3])
    parts.append(b"\x0a" + varint(len(packed)) + packed)
    # field 2: max_tokens varint
    parts.append(b"\x10" + varint(128))
    # field 5: stream_format string
    fmt = b"msgpack"
    parts.append(b"\x2a" + varint(len(fmt)) + fmt)

    payload = b"".join(parts)
    decoded = decode_protobuf_request(payload)

    assert decoded.get("prompt_ids") == [1, 2, 3], decoded
    assert decoded.get("max_tokens") == 128, decoded
    assert decoded.get("stream_format") == "msgpack", decoded
    print(f"  ✓ {decoded}")


# ── 3. compression negotiation ──────────────────────────────────────────────


def test_compression_negotiation():
    print("test_compression_negotiation")
    cases = [
        # (Accept-Encoding header, expected encoding)
        ("",                                None),
        ("identity",                        None),       # only identity → no compression
        ("gzip",                            "gzip"),
        ("br, gzip",                        "br"),       # prefer brotli over gzip
        ("zstd, br, gzip",                  "zstd"),     # zstd first if available
        ("gzip, deflate",                   "gzip"),
        ("br;q=0.9, gzip;q=0.5",            "br"),
        ("gzip;q=0, br;q=0.5",              "br"),       # q=0 means "not acceptable"
        ("*",                               "zstd"),     # wildcard → server's first preference
        ("identity;q=0",                    None),       # identity refused; pick a real one
    ]
    for header, expected in cases:
        actual = negotiate_encoding(header)
        # Special case: zstd may be skipped if zstandard package not installed
        # (but we did install it earlier). Brotli same.
        marker = "✓" if actual == expected else "✗"
        print(f"  {marker} {header!r:35s} → {actual!r}")
        assert actual == expected, (
            f"mismatch on {header!r}: got {actual!r}, expected {expected!r}")


def test_accept_encoding_parser():
    print("test_accept_encoding_parser")
    parts = _parse_accept_encoding("br;q=0.9, gzip;q=0.5, zstd")
    # Order is preserved (parser doesn't sort by q); negotiate sorts.
    assert "br" in parts and "gzip" in parts and "zstd" in parts, parts
    print(f"  ✓ {parts}")


# ── 4. PROTO_SCHEMA matches spec ────────────────────────────────────────────


def test_proto_schema_present():
    print("test_proto_schema_present")
    assert "syntax = \"proto3\"" in PROTO_SCHEMA
    assert "CodecFrame" in PROTO_SCHEMA
    assert "uint32 ids" in PROTO_SCHEMA
    assert "[packed = true]" in PROTO_SCHEMA
    print(f"  ✓ schema is well-formed (~{len(PROTO_SCHEMA)} bytes)")


# ── 5. Cross-check: sglang msgpack ⇄ codecai client (if available) ──────────


def test_cross_client_msgpack():
    print("test_cross_client_msgpack")
    try:
        import asyncio

        async def go():
            # Build a synthetic stream of 5 sglang-encoded msgpack frames.
            stream_bytes = b""
            for i in range(5):
                done = (i == 4)
                fr = "eos_token" if done else None
                stream_bytes += encode_msgpack(
                    [1000 + i, 2000 + i], done=done, finish_reason=fr)

            # Decode through @codecai/web's Python twin.
            from codecai.stream import decode_msgpack_stream

            async def fake_aiter_raw():
                yield stream_bytes

            frames = []
            async for frame in decode_msgpack_stream(fake_aiter_raw()):
                frames.append(frame)
            return frames

        frames = asyncio.run(go())
        assert len(frames) == 5, f"expected 5 frames, got {len(frames)}"
        for i, f in enumerate(frames):
            assert list(f.ids) == [1000 + i, 2000 + i], f
            assert f.done == (i == 4)
            if i == 4:
                assert f.finish_reason == "eos_token"
        print(f"  ✓ 5 sglang frames decoded by codecai client; "
              f"all ids/done/finish_reason match")
    except ImportError as e:
        print(f"  (skipped — codecai not installed in this env: {e})")


def test_msgpack_tool_calls_field():
    """The new server-side ToolWatcher emits parsed tool calls on
    msgpack frames via the `tool_calls` field. Round-trip a frame
    that carries one and verify the shape decodes cleanly."""
    print("test_msgpack_tool_calls_field")
    tc = [{"name": "search", "arguments_json": '{"q":"hi"}', "id": "tc_00000001"}]
    encoded = encode_msgpack(
        [42, 99],
        done=False,
        finish_reason=None,
        tool_calls=tc,
    )
    decoded = decode_msgpack(encoded)
    assert decoded["ids"] == [42, 99]
    assert decoded["done"] is False
    assert "finish_reason" not in decoded
    assert decoded["tool_calls"] == tc, decoded
    print(f"  ✓ tool_calls round-trip ok ({len(encoded)} bytes)")


def test_protobuf_tool_calls_field():
    print("test_protobuf_tool_calls_field")
    tc = [{"name": "search", "arguments_json": '{"q":"hi"}', "id": "tc_00000001"}]
    encoded = encode_protobuf(
        [42, 99],
        done=False,
        finish_reason=None,
        tool_calls=tc,
    )
    # Decode field 4 (length-delimited ToolCall messages). Reuse the
    # existing protobuf-frame decoder helper and extend it.
    (length,) = struct.unpack(">I", encoded[:4])
    body = encoded[4:4 + length]
    pos = 0
    seen_tcs = []
    def read_varint():
        nonlocal pos
        n = shift = 0
        while True:
            b = body[pos]; pos += 1
            n |= (b & 0x7F) << shift
            if not (b & 0x80): return n
            shift += 7
    while pos < len(body):
        tag = read_varint()
        field = tag >> 3; wt = tag & 0x07
        if field == 4 and wt == 2:
            ln = read_varint(); end = pos + ln
            sub = body[pos:end]; pos = end
            sp = 0; obj = {}
            def rv():
                nonlocal sp
                n = shift = 0
                while True:
                    b = sub[sp]; sp += 1
                    n |= (b & 0x7F) << shift
                    if not (b & 0x80): return n
                    shift += 7
            while sp < len(sub):
                stag = rv(); sf = stag >> 3; swt = stag & 0x07
                if swt == 2:
                    sln = rv(); s = sub[sp:sp+sln].decode(); sp += sln
                    if sf == 1: obj["name"] = s
                    elif sf == 2: obj["arguments_json"] = s
                    elif sf == 3: obj["id"] = s
                else:
                    rv()
            seen_tcs.append(obj)
        elif wt == 0: read_varint()
        elif wt == 2: ln = read_varint(); pos += ln
        elif wt == 5: pos += 4
        elif wt == 1: pos += 8
    assert seen_tcs == tc, seen_tcs
    print(f"  ✓ protobuf tool_calls round-trip ok ({len(encoded)} bytes)")


def main():
    print("=" * 72)
    print(" SGLang PR #24483 — Codec module smoke test")
    print("=" * 72)

    tests = [
        test_msgpack_round_trip,
        test_protobuf_round_trip,
        test_protobuf_request_decode,
        test_compression_negotiation,
        test_accept_encoding_parser,
        test_proto_schema_present,
        test_cross_client_msgpack,
        test_msgpack_tool_calls_field,
        test_protobuf_tool_calls_field,
    ]
    fails = 0
    for t in tests:
        try:
            t()
        except AssertionError as e:
            print(f"  ✗ FAIL: {e}")
            fails += 1
        except Exception as e:
            print(f"  ✗ ERROR: {type(e).__name__}: {e}")
            fails += 1
        print()

    print("=" * 72)
    if fails:
        print(f"  {fails} test(s) failed")
        sys.exit(1)
    print("  all tests passed")


if __name__ == "__main__":
    main()
