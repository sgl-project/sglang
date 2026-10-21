"""MiniLB /health and /health_generate must report the backends, not only the
LB process: MiniLB routes to every registered server without health tracking,
so one dead server already fails part of the traffic."""

import asyncio
import json
import socket

import pytest
from aiohttp import web
from sglang_router import mini_lb

# Backend behaviors: an HTTP status, refused connection, or no answer at all.
REFUSED = "refused"
HANGS = "hangs"


def _handler(behavior, release):
    async def handler(request):
        if behavior == HANGS:
            # Released at teardown; the runner's cleanup waits for handlers.
            await release.wait()
            return web.Response(status=503)
        return web.Response(status=behavior)

    return handler


async def _serve(behavior, release):
    app = web.Application()
    app.router.add_get("/health", _handler(behavior, release))
    app.router.add_get("/health_generate", _handler(behavior, release))
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    port = site._server.sockets[0].getsockname()[1]
    return runner, f"http://127.0.0.1:{port}"


def _refused_url():
    """URL of a local port that nothing listens on."""
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    return f"http://127.0.0.1:{port}"


def _run(monkeypatch, handler, prefill_behaviors, decode_behaviors):
    """Serve one backend per behavior, point MiniLB at them, and call
    `handler`. Returns the response and the prefill and decode URLs."""

    async def run():
        runners = []
        release = asyncio.Event()

        async def backend_url(behavior):
            if behavior == REFUSED:
                return _refused_url()
            runner, url = await _serve(behavior, release)
            runners.append(runner)
            return url

        try:
            prefill_urls = [await backend_url(b) for b in prefill_behaviors]
            decode_urls = [await backend_url(b) for b in decode_behaviors]
            load_balancer = mini_lb.MiniLoadBalancer.__new__(mini_lb.MiniLoadBalancer)
            load_balancer.timeout = 3600
            load_balancer.prefill_urls = prefill_urls
            load_balancer.prefill_bootstrap_ports = [8998] * len(prefill_urls)
            load_balancer.decode_urls = decode_urls
            monkeypatch.setattr(mini_lb, "lb", load_balancer)
            return await handler(), prefill_urls, decode_urls
        finally:
            release.set()
            for runner in runners:
                await runner.cleanup()

    return asyncio.run(run())


@pytest.mark.unit
@pytest.mark.parametrize("handler", [mini_lb.health_check, mini_lb.health_generate])
@pytest.mark.parametrize(
    ("prefill_behaviors", "decode_behaviors", "expected_status", "unhealthy"),
    [
        pytest.param([200], [200], 200, [], id="all_healthy"),
        pytest.param([503], [200], 503, [("prefill", 0, 503)], id="prefill_503"),
        pytest.param([200], [REFUSED], 503, [("decode", 0, None)], id="decode_down"),
        pytest.param(
            [REFUSED],
            [REFUSED],
            503,
            [("prefill", 0, None), ("decode", 0, None)],
            id="both_down",
        ),
        pytest.param(
            [200], [200, 500], 503, [("decode", 1, 500)], id="one_of_two_decodes"
        ),
        pytest.param([HANGS], [200], 503, [("prefill", 0, None)], id="prefill_hangs"),
        pytest.param(
            [200], [], 503, [("decode", None, None)], id="no_decode_registered"
        ),
    ],
)
def test_health_reports_backends(
    monkeypatch,
    handler,
    prefill_behaviors,
    decode_behaviors,
    expected_status,
    unhealthy,
):
    monkeypatch.setattr(mini_lb, "HEALTH_CHECK_TIMEOUT_SECS", 1)

    response, prefill_urls, decode_urls = _run(
        monkeypatch, handler, prefill_behaviors, decode_behaviors
    )

    assert response.status_code == expected_status
    if expected_status == 200:
        return
    urls = {"prefill": prefill_urls, "decode": decode_urls}
    reported = json.loads(response.body)["unhealthy"]
    assert [(b["role"], b["url"], b["status"]) for b in reported] == [
        (role, None if index is None else urls[role][index], status)
        for role, index, status in unhealthy
    ]


@pytest.mark.unit
def test_hung_backend_reports_timeout(monkeypatch):
    monkeypatch.setattr(mini_lb, "HEALTH_CHECK_TIMEOUT_SECS", 1)

    response, _, _ = _run(monkeypatch, mini_lb.health_check, [200], [HANGS])

    assert response.status_code == 503
    assert json.loads(response.body)["unhealthy"][0]["error"] == "no answer in 1 s"


@pytest.mark.unit
def test_liveness_ignores_backends(monkeypatch):
    response, _, _ = _run(monkeypatch, mini_lb.liveness, [REFUSED], [REFUSED])
    assert response.status_code == 200


@pytest.mark.unit
@pytest.mark.parametrize(
    ("path", "endpoint"),
    [
        ("/health", mini_lb.health_check),
        ("/health_generate", mini_lb.health_generate),
        ("/liveness", mini_lb.liveness),
    ],
)
def test_health_routes_are_registered(path, endpoint):
    routes = {route.path: route.endpoint for route in mini_lb.app.routes}
    assert routes[path] is endpoint
