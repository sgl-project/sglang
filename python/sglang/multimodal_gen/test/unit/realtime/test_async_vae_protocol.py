# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pytest

from sglang.multimodal_gen.runtime.realtime.async_vae_protocol import (
    AcceptDisposition,
    ChunkSequenceTracker,
    LatentChunkHeader,
    ProtocolViolation,
    checksum_payload,
    decode_message,
    encode_message,
)


def _header(**overrides) -> LatentChunkHeader:
    values = {
        "session_id": "s1",
        "generation_id": "g2",
        "request_id": "r0",
        "chunk_index": 0,
        "dtype": "bfloat16",
        "shape": (1, 48, 1, 30, 52),
        "byte_length": 149_760,
        "checksum": "abc",
    }
    values.update(overrides)
    return LatentChunkHeader(**values)


def test_latent_header_rejects_stale_generation():
    tracker = ChunkSequenceTracker("s1", "g2")

    with pytest.raises(ProtocolViolation, match="stale generation"):
        tracker.accept(_header(generation_id="g1"))


def test_latent_header_accepts_next_chunk_and_deduplicates_retry():
    tracker = ChunkSequenceTracker("s1", "g2")

    assert tracker.accept(_header()) is AcceptDisposition.ACCEPT
    assert tracker.accept(_header()) is AcceptDisposition.DUPLICATE
    assert tracker.accept(_header(chunk_index=1)) is AcceptDisposition.ACCEPT


def test_latent_header_rejects_gap_and_wrong_session():
    tracker = ChunkSequenceTracker("s1", "g2")

    with pytest.raises(ProtocolViolation, match="out-of-order chunk"):
        tracker.accept(_header(chunk_index=2))
    with pytest.raises(ProtocolViolation, match="wrong session"):
        tracker.accept(_header(session_id="s2"))


def test_message_round_trip_keeps_binary_payload_and_checksum():
    payload = b"\x00\x01latent"
    wire = encode_message(
        "latent_chunk",
        header=_header(
            byte_length=len(payload),
            checksum=checksum_payload(payload),
        ),
        payload=payload,
    )

    message = decode_message(wire)

    assert message["type"] == "latent_chunk"
    assert message["header"]["shape"] == [1, 48, 1, 30, 52]
    assert message["payload"] == payload


def test_decode_rejects_oversized_wire_message():
    with pytest.raises(ProtocolViolation, match="message exceeds"):
        decode_message(b"x" * 33, max_message_bytes=32)
