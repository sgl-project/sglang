"""Unit tests for the router Responses benchmark adapter."""

from __future__ import annotations

from argparse import Namespace

import pytest

from benchmark.multi_turn_chat import bench_router_responses as bench


def test_validate_args_rejects_http_previous_response_without_store():
    args = Namespace(
        parallel=1,
        client_transport="http_sse",
        chain_mode="previous_response_id",
        store_mode="false",
    )

    with pytest.raises(
        ValueError,
        match="HTTP SSE with --chain-mode previous_response_id requires --store-mode true",
    ):
        bench._validate_args(args)


def test_benchmark_contract_carries_router_axes():
    args = Namespace(
        model="Qwen/Qwen2.5-72B-Instruct",
        worker_transport="grpc",
        router_topology="regular_grpc_worker",
        store_mode="true",
        chain_mode="full_replay",
    )

    contract = bench._benchmark_contract(args, "websocket")

    assert contract == {
        "benchmark_family": "long_context_multiturn_qos",
        "run_class": "router_multiturn_adapter",
        "client_transport": "websocket",
        "worker_transport": "grpc",
        "router_topology": "regular_grpc_worker",
        "model_id": "Qwen/Qwen2.5-72B-Instruct",
        "topology_overlay": "none",
        "store_mode": "store_true",
        "workload_kind": "multi_turn_chat_full_replay",
    }


def test_prepare_turn_input_modes():
    conversation = [bench._user_message("first"), bench._assistant_message("reply")]

    assert (
        bench._prepare_turn_input("previous_response_id", conversation, "next")
        == "next"
    )
    assert bench._prepare_turn_input("full_replay", conversation, "next") == [
        *conversation,
        bench._user_message("next"),
    ]


def test_parse_sse_event_preserves_event_name_and_multiline_payload():
    event = bench._parse_sse_event(
        [
            "event: response.completed",
            'data: {"type":"response.completed",',
            'data: "response":{"id":"resp_123"}}',
        ]
    )

    assert event == {
        "type": "response.completed",
        "response": {"id": "resp_123"},
        "_sse_event_name": "response.completed",
    }


def test_response_timing_tracker_records_phase_gaps_and_trace():
    tracker = bench.ResponseTimingTracker(
        request_started_at=10.0,
        capture_event_trace=True,
        event_trace_limit=2,
    )

    tracker.observe({"type": "response.created"}, 10.010)
    tracker.observe({"type": "response.output_text.delta", "delta": "hello"}, 10.040)
    tracker.observe({"type": "response.completed"}, 10.070)

    metrics = tracker.metrics()

    assert metrics["request_to_first_event_ms"] == pytest.approx(10.0)
    assert metrics["request_to_first_content_ms"] == pytest.approx(40.0)
    assert metrics["request_to_completed_ms"] == pytest.approx(70.0)
    assert metrics["first_event_to_first_content_ms"] == pytest.approx(30.0)
    assert metrics["first_content_to_completed_ms"] == pytest.approx(30.0)
    assert metrics["event_count"] == 3
    assert metrics["output_text_delta_count"] == 1
    assert metrics["event_trace"] == [
        {
            "event_type": "response.created",
            "t_ms": pytest.approx(10.0),
            "delta_chars": 0,
        },
        {
            "event_type": "response.output_text.delta",
            "t_ms": pytest.approx(40.0),
            "delta_chars": 5,
        },
    ]


def test_validate_args_rejects_non_positive_event_trace_limit():
    args = Namespace(
        parallel=1,
        client_transport="websocket",
        chain_mode="full_replay",
        store_mode="false",
        event_trace_limit=0,
    )

    with pytest.raises(ValueError, match="--event-trace-limit must be >= 1"):
        bench._validate_args(args)
