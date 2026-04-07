"""Unit tests for WS benchmark helper functions.

These tests validate the benchmark contract mapping and context construction
logic without requiring GPU or model access.
"""

from __future__ import annotations

import pytest

from benchmarks.test_ws_microbench import (
    _benchmark_context,
    _benchmark_contract,
)


@pytest.mark.parametrize(
    "backend_name, expected_worker_transport, expected_router_topology",
    [
        ("http", "http", "regular_http_worker"),
        ("grpc", "grpc", "regular_grpc_worker"),
        ("pd", "http", "pd_http_workers"),
    ],
)
def test_benchmark_contract_maps_backend_axes(
    backend_name: str,
    expected_worker_transport: str,
    expected_router_topology: str,
):
    contract = _benchmark_contract(
        **_benchmark_context(
            benchmark_family="transport_qos",
            run_class="test_contract",
            backend_name=backend_name,
            model="Qwen/Qwen2.5-72B-Instruct",
            store_mode="store_false",
            workload_kind="single_turn_text",
        ),
        client_transport="websocket",
    )

    assert contract["worker_transport"] == expected_worker_transport
    assert contract["router_topology"] == expected_router_topology
    assert contract["client_transport"] == "websocket"
    assert contract["model_id"] == "Qwen/Qwen2.5-72B-Instruct"


def test_benchmark_context_rejects_unknown_backend():
    with pytest.raises(ValueError, match="Unsupported benchmark backend"):
        _benchmark_context(
            benchmark_family="transport_qos",
            run_class="test_contract",
            backend_name="ws_worker",
            model="Qwen/Qwen2.5-72B-Instruct",
            store_mode="store_false",
            workload_kind="single_turn_text",
        )
