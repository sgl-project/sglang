# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from sglang.multimodal_gen.runtime.utils.realtime_trace import (
    calculate_overlap_ms,
    calculate_overlap_ratio,
    log_realtime_trace,
    realtime_trace_payload,
    register_realtime_trace_sink,
    unregister_realtime_trace_sink,
)


def test_trace_identity_is_correlated_and_media_is_redacted():
    session = SimpleNamespace(
        id="s1",
        generation_id="g1",
        trace_id="t1",
        trace_started_at=0,
    )
    event = realtime_trace_payload(
        session,
        "server.latent_transfer_accepted",
        request_id="r1",
        chunk_index=2,
        prompt="prompt text",
        latent_payload=b"secret latent",
    )

    assert event["generation_id"] == "g1"
    assert event["request_id"] == "r1"
    assert event["chunk_index"] == 2
    serialized = json.dumps(event)
    assert "prompt text" not in serialized
    assert "secret latent" not in serialized
    assert event["prompt_length"] == 11
    assert event["latent_payload_redacted"] is True


def test_overlap_ratio_uses_actual_interval_intersection():
    denoise = (100.0, 600.0)
    vae = (400.0, 700.0)

    assert calculate_overlap_ratio(denoise, vae) == pytest.approx(0.4)
    assert calculate_overlap_ms(denoise, vae) == pytest.approx(200_000)


def test_trace_sinks_are_scoped_by_session_even_when_client_trace_ids_collide():
    first_events = []
    second_events = []
    first = SimpleNamespace(
        id="session-1",
        generation_id="generation-1",
        trace_id="client-trace",
        trace_started_at=0,
    )
    second = SimpleNamespace(
        id="session-2",
        generation_id="generation-2",
        trace_id="client-trace",
        trace_started_at=0,
    )
    register_realtime_trace_sink(
        first.trace_id,
        first_events.append,
        session_id=first.id,
        generation_id=first.generation_id,
    )
    register_realtime_trace_sink(
        second.trace_id,
        second_events.append,
        session_id=second.id,
        generation_id=second.generation_id,
    )
    try:
        log_realtime_trace(SimpleNamespace(info=lambda *args: None), first, "first")
        log_realtime_trace(SimpleNamespace(info=lambda *args: None), second, "second")
    finally:
        unregister_realtime_trace_sink(
            first.trace_id,
            first_events.append,
            session_id=first.id,
            generation_id=first.generation_id,
        )
        unregister_realtime_trace_sink(
            second.trace_id,
            second_events.append,
            session_id=second.id,
            generation_id=second.generation_id,
        )

    assert [event["event"] for event in first_events] == ["first"]
    assert [event["event"] for event in second_events] == ["second"]
