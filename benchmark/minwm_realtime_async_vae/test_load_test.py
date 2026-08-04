from argparse import Namespace

from load_test import init_request, record_action_latency, stage_values


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
