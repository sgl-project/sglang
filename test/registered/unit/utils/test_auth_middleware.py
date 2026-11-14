"""Auth keys may be published after Starlette builds its middleware stack."""

import asyncio
import sys
from contextlib import asynccontextmanager

import pytest
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from fastapi.testclient import TestClient

from sglang.srt.utils.auth import AuthLevel, add_api_key_middleware, auth_level
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


@pytest.mark.parametrize(
    "api_key,admin_key,normal,optional,forced",
    [
        (None, None, [200] * 4, [200] * 4, [403] * 4),
        ("user", None, [401, 401, 200, 401], [401, 401, 200, 401], [403] * 4),
        (None, "admin", [200] * 4, [401, 401, 401, 200], [401, 401, 401, 200]),
        (
            "user",
            "admin",
            [401, 401, 200, 401],
            [401, 401, 401, 200],
            [401, 401, 401, 200],
        ),
    ],
)
def test_auth_policy_after_lifespan_publication(
    api_key, admin_key, normal, optional, forced
):
    keys = {}

    @asynccontextmanager
    async def lifespan(app):
        assert app.middleware_stack is not None
        keys.update(api=api_key, admin=admin_key)
        yield

    app = FastAPI(lifespan=lifespan)

    @app.get("/normal")
    def normal_route():
        return {"ok": True}

    @app.get("/optional")
    @auth_level(AuthLevel.ADMIN_OPTIONAL)
    def optional_route():
        return {"ok": True}

    @app.get("/forced")
    @auth_level(AuthLevel.ADMIN_FORCE)
    def forced_route():
        return {"ok": True}

    for path in ("/health", "/ready", "/metrics"):
        app.add_api_route(path, normal_route)

    add_api_key_middleware(app, get_keys=lambda: (keys["api"], keys["admin"]))
    with TestClient(app) as client:
        for path, statuses in (
            ("/normal", normal),
            ("/optional", optional),
            ("/forced", forced),
        ):
            for credential, expected in zip((None, "wrong", "user", "admin"), statuses):
                headers = (
                    {}
                    if credential is None
                    else {"Authorization": f"Bearer {credential}"}
                )
                response = client.get(path, headers=headers)
                assert response.status_code == expected
        for path in ("/health", "/ready", "/metrics"):
            assert client.get(path).status_code == 200
        # Auth allows OPTIONS to reach the router (which has no OPTIONS route).
        assert client.options("/forced").status_code == 405


def test_deferred_keys_are_read_for_each_request():
    app = FastAPI()
    keys = {"api": "before"}

    @app.get("/normal")
    def normal():
        return {"ok": True}

    add_api_key_middleware(app, get_keys=lambda: (keys["api"], None))
    with TestClient(app) as client:
        assert (
            client.get(
                "/normal", headers={"Authorization": "Bearer before"}
            ).status_code
            == 200
        )
        keys["api"] = "after"
        assert (
            client.get(
                "/normal", headers={"Authorization": "Bearer before"}
            ).status_code
            == 401
        )
        assert (
            client.get("/normal", headers={"Authorization": "Bearer after"}).status_code
            == 200
        )


def test_key_resolution_failure_does_not_allow_request():
    def unavailable():
        raise RuntimeError("configuration unavailable")

    app = FastAPI()

    @app.get("/normal")
    def normal():
        pytest.fail("An unresolved key must not reach the endpoint")

    add_api_key_middleware(app, get_keys=unavailable)
    with TestClient(app, raise_server_exceptions=False) as client:
        assert client.get("/normal").status_code == 500


def test_auth_preserves_streaming_and_client_disconnect():
    app = FastAPI()
    receives = []

    @app.post("/stream")
    async def stream():
        async def events():
            yield "data: first\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(events(), media_type="text/event-stream")

    add_api_key_middleware(app, get_keys=lambda: ("user", None))
    with TestClient(app) as client:
        response = client.post("/stream", headers={"Authorization": "Bearer user"})
        assert response.status_code == 200
        assert response.text == "data: first\n\ndata: [DONE]\n\n"

    # Auth must forward receive itself, including disconnect notifications.
    async def downstream(scope, receive, send):
        receives.append(await receive())

    entry = app.user_middleware[0]
    middleware = entry.cls(downstream, **entry.kwargs)

    async def disconnected():
        return {"type": "http.disconnect"}

    async def unexpected_send(message):
        pytest.fail(f"Unexpected response: {message}")

    asyncio.run(
        middleware(
            {
                "type": "http",
                "method": "POST",
                "path": "/stream",
                "headers": [(b"authorization", b"Bearer user")],
            },
            disconnected,
            unexpected_send,
        )
    )
    assert receives == [{"type": "http.disconnect"}]


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, *sys.argv[1:]]))
