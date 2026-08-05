from argparse import Namespace

import asyncio

from load_test import (
    aggregate_measurement_seconds,
    collect_trace_events,
    derive_trace_http_url,
    init_request,
    record_frame_batch,
    record_action_latency,
    server_action_latencies,
    stage_values,
    trace_contract_summary,
)


def test_init_request_keeps_t2v_frame_count_aligned_with_chunk_count():
    args = Namespace(
        model="/work/model",
        prompt="test prompt",
        size="832x480",
        fps=24,
    )

    request = init_request(args, total_chunks=5, trace_id="trace-1")

    assert request["num_frames"] == 65
    assert request["max_chunks"] == 5


def test_record_frame_batch_counts_all_batches_in_the_same_chunk():
    frame_counts = {}

    record_frame_batch(
        {"chunk_index": 3, "num_frames": 8}, frame_counts=frame_counts
    )
    record_frame_batch(
        {"chunk_index": 3, "num_frames": 8}, frame_counts=frame_counts
    )

    assert frame_counts == {3: 16}


def test_stage_values_excludes_warmup_and_records_local_vae():
    events = [
        {
            "event": "server.model_denoise_complete",
            "chunk_index": 0,
            "cuda_ms": 900.0,
        },
        {
            "event": "server.model_denoise_complete",
            "chunk_index": 2,
            "cuda_ms": 310.0,
        },
        {
            "event": "server.vae_decode_complete",
            "chunk_index": 2,
            "cuda_ms": 16.0,
        },
    ]

    assert stage_values(events, min_chunk_index=2) == {
        "denoise_ms": [310.0],
        "vae_decode_ms": [16.0],
    }


def test_trace_http_url_is_derived_from_the_public_websocket_origin():
    assert derive_trace_http_url(
        "wss://realtime.example.com/v1/realtime_video/generate?mode=t2v"
    ) == "https://realtime.example.com"
    assert derive_trace_http_url(
        "ws://127.0.0.1:18080/v1/realtime_video/generate"
    ) == "http://127.0.0.1:18080"


def test_collect_trace_events_polls_incrementally_and_deduplicates():
    class Response:
        def __init__(self, payload):
            self.payload = payload

        def raise_for_status(self):
            return None

        def json(self):
            return self.payload

    class Client:
        def __init__(self):
            self.calls = []

        async def get(self, url, *, params):
            self.calls.append((url, dict(params)))
            if len(self.calls) == 1:
                return Response(
                    {
                        "events": [
                            {"event": "gateway.ws_accepted", "trace_seq": 1},
                            {"event": "server.chunk_complete", "trace_seq": 2},
                        ],
                        "next_cursor": 2,
                    }
                )
            return Response(
                {
                    "events": [
                        {"event": "server.chunk_complete", "trace_seq": 2},
                        {"event": "server.vae_decode_complete", "trace_seq": 3},
                    ],
                    "next_cursor": 3,
                }
            )

    async def run():
        client = Client()
        events = await collect_trace_events(
            "http://gateway",
            "trace-a",
            client=client,
            timeout_s=0.1,
            poll_interval_s=0,
            stable_polls=1,
        )
        assert [event["trace_seq"] for event in events] == [1, 2, 3]
        assert client.calls[0][1]["after"] == 0
        assert client.calls[1][1]["after"] == 2

    asyncio.run(run())


def test_collect_trace_events_retries_a_transient_transport_timeout():
    class Response:
        def raise_for_status(self):
            return None

        def json(self):
            return {
                "events": [{"event": "server.chunk_complete", "trace_seq": 1}],
                "next_cursor": 1,
            }

    class Client:
        def __init__(self):
            self.calls = 0

        async def get(self, _url, *, params):
            self.calls += 1
            if self.calls == 1:
                raise TimeoutError("transient trace read timeout")
            return Response()

    async def run():
        client = Client()
        events = await collect_trace_events(
            "http://gateway",
            "trace-retry",
            client=client,
            timeout_s=0.1,
            poll_interval_s=0,
            stable_polls=1,
        )

        assert [event["trace_seq"] for event in events] == [1]
        assert client.calls == 3

    asyncio.run(run())


def test_stage_values_backfills_overlap_after_next_denoise_completes():
    events = [
        {
            "event": "server.remote_vae_complete",
            "chunk_index": 3,
            "overlap_with_next_denoise_ms": 0,
            "overlap_ratio": 0,
        },
        {
            "event": "server.vae_denoise_overlap_complete",
            "chunk_index": 3,
            "next_chunk_index": 4,
            "overlap_with_next_denoise_ms": 71.5,
            "overlap_ratio": 0.82,
        },
    ]

    result = stage_values(events)

    assert result["overlap_with_next_denoise_ms"] == [71.5]
    assert result["overlap_ratio"] == [0.82]


def test_action_latency_uses_chunk_stats_sampled_event_id():
    first_frame_at = {3: 12.5}
    action_sent_at = {1: 11.8, 2: 12.0}
    action_latencies = []

    record_action_latency(
        {"chunk_index": 3, "event_id": 2},
        first_frame_at=first_frame_at,
        action_sent_at=action_sent_at,
        action_latencies=action_latencies,
        min_chunk_index=2,
    )

    assert action_latencies == [500.0]
    assert action_sent_at == {}


def test_action_latency_discards_warmup_samples_without_recording_them():
    action_sent_at = {1: 1.0}
    action_latencies = []

    record_action_latency(
        {"chunk_index": 1, "event_id": 1},
        first_frame_at={1: 1.2},
        action_sent_at=action_sent_at,
        action_latencies=action_latencies,
        min_chunk_index=2,
    )

    assert action_latencies == []
    assert action_sent_at == {}


def test_aggregate_measurement_seconds_uses_real_overlapping_wall_window():
    sessions = [
        {"measured_started_at": 10.0, "measured_completed_at": 12.0},
        {"measured_started_at": 10.5, "measured_completed_at": 13.0},
    ]

    assert aggregate_measurement_seconds(sessions) == 3.0


def test_server_action_latencies_use_sampled_event_and_first_frame_marker():
    events = [
        {
            "event": "server.event_received",
            "event_id": 7,
            "client_sent_epoch_ms": 10_000.0,
            "server_epoch_ms": 10_020.0,
            "server_elapsed_ms": 20.0,
        },
        {
            "event": "server.remote_first_frame_received",
            "chunk_index": 3,
            "event_id": 7,
            "server_epoch_ms": 10_540.0,
            "server_elapsed_ms": 540.0,
        },
    ]

    assert server_action_latencies(events, min_chunk_index=2) == {
        "action_to_server_first_frame_ms": [540.0],
        "action_ingress_to_server_first_frame_ms": [520.0],
    }


def test_server_action_latencies_support_sync_marker_and_latest_prior_event():
    events = [
        {
            "event": "server.event_received",
            "event_id": 3,
            "client_sent_epoch_ms": 20_000.0,
            "server_epoch_ms": 20_010.0,
            "server_elapsed_ms": 10.0,
        },
        {
            "event": "server.output_send_start",
            "chunk_index": 4,
            "event_id": 4,
            "server_epoch_ms": 20_430.0,
            "server_elapsed_ms": 430.0,
        },
    ]

    assert server_action_latencies(events) == {
        "action_to_server_first_frame_ms": [430.0],
        "action_ingress_to_server_first_frame_ms": [420.0],
    }


def test_trace_contract_summary_proves_direct_vae_media_route():
    summary = trace_contract_summary(
        [
            {"event": "gateway.ws_accepted"},
            {"event": "coordinator.admit_complete"},
            {"event": "server.model_denoise_complete"},
            {"event": "server.vae_decode_complete"},
            {
                "event": "server.vae_frame_batch_sent",
                "output_direct": True,
            },
        ]
    )

    assert summary["event_names"] == [
        "coordinator.admit_complete",
        "gateway.ws_accepted",
        "server.model_denoise_complete",
        "server.vae_decode_complete",
        "server.vae_frame_batch_sent",
    ]
    assert summary["direct_vae_frame_batches"] == 1
