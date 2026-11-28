# SPDX-License-Identifier: Apache-2.0

import asyncio

import msgspec.msgpack
import pytest
import websockets

from sglang.multimodal_gen.benchmarks.bench_realtime import (
    SessionTimeline,
    run_session,
    summarize_session,
)


def test_summary_counts_first_frame_events_gaps_and_stalls():
    timeline = SessionTimeline(
        frame_batches=[
            (
                1.0,
                0,
                None,
                True,
            ),  # an empty first chunk does not count as the first frame
            (2.0, 12, None, True),
            (3.0, 12, None, True),
            (4.0, 12, 1, True),
            (5.0, 12, 1, True),  # the same event echoed again is not a second answer
            (9.0, 12, 2, True),
        ],
        events_sent={1: 3.5, 2: 4.2, 3: 8.0},
        chunk_stats=[{"chunk_total_ms": 900}, {"chunk_total_ms": 1100}],
    )

    summary = summarize_session(timeline, stall_factor=2.0)

    assert summary["time_to_first_frame_ms"] == pytest.approx(2000)
    assert summary["frames"] == 60
    assert summary["chunks"] == 6
    assert summary["delivered_fps"] == pytest.approx(48 / 7)
    assert summary["event_latency_ms"] == pytest.approx([500, 4800])
    assert summary["events_unanswered"] == 1
    assert summary["chunk_gap_ms_median"] == pytest.approx(1000)
    assert summary["stalls"] == 1
    assert summary["server_chunk_total_ms_median"] == pytest.approx(1000)


def test_summary_of_a_session_without_frames():
    summary = summarize_session(
        SessionTimeline(errors=["another realtime session is already active"]),
        stall_factor=2.0,
    )

    assert summary["time_to_first_frame_ms"] is None
    assert summary["delivered_fps"] is None
    assert summary["chunks"] == 0
    assert summary["errors"] == ["another realtime session is already active"]


async def _fake_realtime_server(ws):
    """Two batches per chunk; a received event is echoed from the next chunk on."""
    init = msgspec.msgpack.decode(await ws.recv())
    assert init["type"] == "init"
    current_event = None

    async def drain_events():
        nonlocal current_event
        while True:
            try:
                message = await asyncio.wait_for(ws.recv(), timeout=0.001)
            except asyncio.TimeoutError:
                return
            current_event = msgspec.msgpack.decode(message)["event_id"]

    for chunk_index in range(init["max_chunks"]):
        await asyncio.sleep(0.02)
        await drain_events()
        for batch_index in range(2):
            header = {
                "type": "frame_batch_header",
                "chunk_index": chunk_index,
                "num_frames": 2,
                "total_size": 2 * 2 * 2 * 3,
                "frame_batch_index": batch_index,
                "num_frame_batches": 2,
                "is_final_frame_batch": batch_index == 1,
            }
            if current_event is not None:
                header["event_id"] = current_event
            await ws.send(msgspec.msgpack.encode(header))
            await ws.send(bytes(header["total_size"]))
        await ws.send(
            msgspec.msgpack.encode(
                {
                    "type": "chunk_stats",
                    "chunk_index": chunk_index,
                    "chunk_total_ms": 20,
                }
            )
        )
    await ws.close()


def test_run_session_against_a_fake_server():
    async def scenario():
        async with websockets.serve(_fake_realtime_server, "127.0.0.1", 0) as server:
            port = server.sockets[0].getsockname()[1]
            return await run_session(
                f"ws://127.0.0.1:{port}",
                {"type": "init", "max_chunks": 6},
                {"kind": "camera_actions", "payload": [["d"]]},
                event_every=2,
                recv_timeout_s=5.0,
            )

    timeline = asyncio.run(scenario())
    summary = summarize_session(timeline, stall_factor=10.0)

    assert summary["chunks"] == 6
    assert summary["frames"] == 24
    # events sent after chunks 2 and 4 are answered; the one after the last chunk
    # is either refused by the closing socket or left unanswered
    assert summary["events_sent"] in (2, 3)
    assert len(summary["event_latency_ms"]) == 2
    assert all(0 < latency < 1000 for latency in summary["event_latency_ms"])
    assert summary["events_unanswered"] == summary["events_sent"] - 2
    assert summary["server_chunk_total_ms_median"] == 20
    assert summary["errors"] == []
