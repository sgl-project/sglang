"""
Unit tests for sglang_router.mini_lb (MiniLoadBalancer).

Covers the pure helper functions and validation logic in isolation, plus a
regression test for https://github.com/sgl-project/sglang/issues/30955:
the mini LB had no /abort_request route, so a client aborting a
disaggregated PD request got a 404 instead of the abort reaching the
prefill/decode workers.
"""

import asyncio
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient
from sglang_router import mini_lb
from sglang_router.mini_lb import (
    MiniLoadBalancer,
    _get_request_batch_size,
    app,
    maybe_wrap_ipv6_address,
)
from sglang_router.router_args import RouterArgs


def _pd_router_args(**overrides):
    defaults = dict(
        pd_disaggregation=True,
        prefill_urls=[("http://prefill1:8000", 9000)],
        decode_urls=["http://decode1:8001"],
    )
    defaults.update(overrides)
    return RouterArgs(**defaults)


class _FakeSession:
    """Minimal aiohttp.ClientSession stand-in that records every request()
    call instead of hitting the network."""

    def __init__(self, calls):
        self._calls = calls

    def __call__(self, *args, **kwargs):
        return self

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    def request(self, method, url, json=None):
        self._calls.append((method, url, json))
        future = asyncio.get_event_loop().create_future()
        future.set_result(None)
        return future


class TestMaybeWrapIpv6Address:
    def test_wraps_ipv6_in_brackets(self):
        assert maybe_wrap_ipv6_address("::1") == "[::1]"
        assert maybe_wrap_ipv6_address("2001:db8::1") == "[2001:db8::1]"

    def test_leaves_ipv4_unwrapped(self):
        assert maybe_wrap_ipv6_address("127.0.0.1") == "127.0.0.1"

    def test_leaves_hostname_unwrapped(self):
        assert maybe_wrap_ipv6_address("prefill1.internal") == "prefill1.internal"


class TestGetRequestBatchSize:
    def test_none_for_single_text(self):
        assert _get_request_batch_size({"text": "hello"}) is None

    def test_batch_size_from_text_list(self):
        assert _get_request_batch_size({"text": ["a", "b", "c"]}) == 3

    def test_none_for_single_input_ids(self):
        assert _get_request_batch_size({"input_ids": [1, 2, 3]}) is None

    def test_batch_size_from_input_ids_list(self):
        assert _get_request_batch_size({"input_ids": [[1, 2], [3, 4]]}) == 2

    def test_none_when_neither_field_present(self):
        assert _get_request_batch_size({}) is None


class TestValidateRouterArgs:
    def test_requires_pd_disaggregation(self):
        with pytest.raises(ValueError, match="PD disaggregation"):
            MiniLoadBalancer(RouterArgs(pd_disaggregation=False))

    def test_requires_nonempty_prefill_urls(self):
        with pytest.raises(ValueError, match="at least one prefill and one decode"):
            MiniLoadBalancer(_pd_router_args(prefill_urls=[]))

    def test_requires_nonempty_decode_urls(self):
        with pytest.raises(ValueError, match="at least one prefill and one decode"):
            MiniLoadBalancer(_pd_router_args(decode_urls=[]))

    def test_overrides_non_random_policy(self):
        router_args = _pd_router_args(policy="cache_aware")
        MiniLoadBalancer(router_args)
        # MiniLB only supports random policy, so a non-random policy must be
        # forced to "random" rather than silently ignored.
        assert router_args.policy == "random"


class TestForkDpRequests:
    def _lb_with_dp_sizes(self, prefill_dp_size=1, decode_dp_size=1):
        lb = MiniLoadBalancer(_pd_router_args())
        lb.prefill_dp_size = prefill_dp_size
        lb.decode_dp_size = decode_dp_size
        return lb

    def test_does_not_mutate_original_request(self):
        lb = self._lb_with_dp_sizes()
        original = {"text": "hello"}

        lb._fork_dp_requests(original)

        assert original == {"text": "hello"}

    def test_assigns_expected_keys(self):
        lb = self._lb_with_dp_sizes()

        prefill_req, decode_req, d_rank = lb._fork_dp_requests({"text": "hi"})

        assert prefill_req["routed_dp_rank"] == 0
        assert decode_req["routed_dp_rank"] == 0
        assert decode_req["disagg_prefill_dp_rank"] == 0
        assert d_rank == 0
        assert "disagg_prefill_dp_rank" not in prefill_req


class TestBroadcastRoutes:
    """Regression coverage for issue #30955 (missing /abort_request proxy),
    plus its sibling broadcast routes that share `_broadcast_to_backends`."""

    def setup_method(self):
        mini_lb.lb = MiniLoadBalancer(
            _pd_router_args(
                prefill_urls=[
                    ("http://prefill1:8000", 9000),
                    ("http://prefill2:8000", 9001),
                ],
                decode_urls=["http://decode1:8001"],
            )
        )
        self.client = TestClient(app)

    def teardown_method(self):
        mini_lb.lb = None

    def test_abort_request_returns_200_instead_of_404(self):
        calls = []
        with patch("sglang_router.mini_lb.aiohttp.ClientSession", _FakeSession(calls)):
            resp = self.client.post(
                "/abort_request", json={"rid": "abc", "abort_all": False}
            )

        assert resp.status_code == 200

    def test_abort_request_forwards_payload_to_every_backend(self):
        calls = []
        payload = {"rid": "abc", "abort_all": False}
        with patch("sglang_router.mini_lb.aiohttp.ClientSession", _FakeSession(calls)):
            self.client.post("/abort_request", json=payload)

        urls = {url for _, url, _ in calls}
        assert urls == {
            "http://prefill1:8000/abort_request",
            "http://prefill2:8000/abort_request",
            "http://decode1:8001/abort_request",
        }
        assert all(sent_json == payload for _, _, sent_json in calls)

    def test_flush_cache_forwards_to_every_backend(self):
        calls = []
        with patch("sglang_router.mini_lb.aiohttp.ClientSession", _FakeSession(calls)):
            resp = self.client.post("/flush_cache")

        assert resp.status_code == 200
        urls = {url for _, url, _ in calls}
        assert urls == {
            "http://prefill1:8000/flush_cache",
            "http://prefill2:8000/flush_cache",
            "http://decode1:8001/flush_cache",
        }

    def test_health_generate_forwards_to_every_backend(self):
        calls = []
        with patch("sglang_router.mini_lb.aiohttp.ClientSession", _FakeSession(calls)):
            resp = self.client.get("/health_generate")

        assert resp.status_code == 200
        methods_and_urls = {(method, url) for method, url, _ in calls}
        assert methods_and_urls == {
            ("GET", "http://prefill1:8000/health_generate"),
            ("GET", "http://prefill2:8000/health_generate"),
            ("GET", "http://decode1:8001/health_generate"),
        }
