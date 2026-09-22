"""Unit tests for virtual-model resource-aware routing."""

import pytest

from sglang_router.multi_model import ModelResolverConfig, RoutingProfile
from sglang_router.resource_aware_router import (
    ResourceAwareModelResolver,
    ResourceUnavailableError,
    RuntimeEndpoint,
    RuntimeLoadState,
    parse_runtime_load,
)


def _resolver():
    return ResourceAwareModelResolver(
        ModelResolverConfig(
            host="127.0.0.1",
            port=30000,
            refresh_interval_secs=1.0,
            stale_after_secs=3.0,
            request_timeout_secs=30.0,
            profiles=(
                RoutingProfile(
                    model_id="general-chat",
                    candidates=("qwen-awq", "qwen-fp16"),
                    max_kv_utilization=0.9,
                    max_waiting_requests=4,
                    min_free_tokens=32,
                ),
            ),
        )
    )


def _state(
    model_id,
    *,
    observed_at=100.0,
    token_usage=0.1,
    waiting=0,
    total_tokens=100,
    capacity=1000,
    running=0,
    max_running=8,
    healthy=True,
):
    return RuntimeLoadState(
        model_id=model_id,
        url=f"http://{model_id}",
        observed_at=observed_at,
        healthy=healthy,
        num_running_reqs=running,
        num_waiting_reqs=waiting,
        num_total_tokens=total_tokens,
        max_total_num_tokens=capacity,
        max_running_requests=max_running,
        token_usage=token_usage,
    )


def test_virtual_model_selects_the_least_constrained_concrete_model():
    resolver = _resolver()
    resolver.update(_state("qwen-awq", token_usage=0.78, waiting=2, running=3))
    resolver.update(_state("qwen-fp16", token_usage=0.12, waiting=0, running=0))

    assert resolver.resolve("general-chat", max_tokens=64, now=101.0) == "qwen-fp16"


def test_virtual_model_excludes_kv_constrained_or_queued_runtimes():
    resolver = _resolver()
    resolver.update(_state("qwen-awq", token_usage=0.95))
    resolver.update(_state("qwen-fp16", waiting=4))

    with pytest.raises(ResourceUnavailableError, match="No eligible Runtime"):
        resolver.resolve("general-chat", now=101.0)


def test_virtual_model_excludes_stale_state_and_preserves_explicit_model():
    resolver = _resolver()
    resolver.update(_state("qwen-awq", observed_at=90.0))
    resolver.update(_state("qwen-fp16", observed_at=100.0))

    assert resolver.resolve("general-chat", now=101.0) == "qwen-fp16"
    assert resolver.resolve("qwen-awq", now=101.0) == "qwen-awq"


def test_virtual_model_requires_declared_free_token_capacity():
    resolver = _resolver()
    resolver.update(_state("qwen-awq", total_tokens=980, capacity=1000))
    resolver.update(_state("qwen-fp16", total_tokens=800, capacity=1000))

    assert resolver.resolve("general-chat", max_tokens=128, now=101.0) == "qwen-fp16"


def test_parse_runtime_load_aggregates_per_dp_rank_metrics():
    state = parse_runtime_load(
        RuntimeEndpoint(model_id="qwen-awq", url="http://worker-a"),
        {
            "loads": [
                {
                    "num_running_reqs": 1,
                    "num_waiting_reqs": 2,
                    "num_total_tokens": 100,
                    "max_total_num_tokens": 1000,
                    "max_running_requests": 8,
                    "utilization": 0.4,
                    "memory": {
                        "weight_gb": 1.0,
                        "kv_cache_gb": 2.0,
                        "graph_gb": 0.5,
                        "token_capacity": 1000,
                    },
                },
                {
                    "num_running_reqs": 2,
                    "num_waiting_reqs": 1,
                    "num_total_tokens": 200,
                    "max_total_num_tokens": 1000,
                    "max_running_requests": 8,
                    "utilization": 0.6,
                    "memory": {
                        "weight_gb": 1.0,
                        "kv_cache_gb": 2.0,
                        "graph_gb": 0.5,
                        "token_capacity": 1000,
                    },
                },
            ]
        },
        observed_at=100.0,
    )

    assert state.num_running_reqs == 3
    assert state.num_waiting_reqs == 3
    assert state.num_total_tokens == 300
    assert state.max_total_num_tokens == 2000
    assert state.token_usage == 0.15
    assert state.memory == {
        "weight_gb": 2.0,
        "kv_cache_gb": 4.0,
        "graph_gb": 1.0,
        "token_capacity": 2000,
    }
