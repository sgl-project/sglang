"""Exercise auth on the imported HTTP app without model or scheduler startup."""

import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
from fastapi.responses import StreamingResponse
from fastapi.testclient import TestClient

from sglang.srt.entrypoints import http_server
from sglang.srt.runtime_context import (
    ParallelContext,
    RuntimeContext,
    get_serving,
    publish_role,
    use_context,
)
from sglang.srt.server_args import PortArgs, ServerArgs
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=12, suite="base-a-test-cpu")


@pytest.fixture
def worker_app(monkeypatch, request):
    app = http_server.app
    api_key, admin_key = getattr(request, "param", ("user", "admin"))
    args = ServerArgs(
        model_path="dummy",
        device="cpu",
        api_key=api_key,
        admin_api_key=admin_key,
        tokenizer_worker_num=4,
    )
    # Launchers resolve configuration before sending it to tokenizer workers.
    args.resolve_once()
    ports = PortArgs(
        tokenizer_ipc_name="unused",
        scheduler_input_ipc_name="unused",
        detokenizer_ipc_name="unused",
        nccl_port=0,
        rpc_ipc_name="unused",
        metrics_ipc_name="unused",
        tokenizer_worker_ipc_name="unused",
        decoupled_spec_ipc_config=None,
    )
    info = {"max_req_input_len": 4096, "startup_time": {}}
    context = RuntimeContext(parallel=ParallelContext())
    worker_ipcs = []

    @asynccontextmanager
    async def lifespan(app):
        assert app.middleware_stack is not None
        assert not context.is_config_namespace_published("serving")
        loaded_args = await http_server.init_multi_tokenizer()
        # The worker must deserialize its own ServerArgs, then publish those
        # values through the real runtime accessors used by auth middleware.
        assert isinstance(loaded_args, ServerArgs)
        assert loaded_args is not args
        assert publish_role() == "tokenizer"
        assert get_serving().api_key == api_key
        assert get_serving().admin_api_key == admin_key
        yield

    manager = SimpleNamespace(
        served_model_name="auth-fixture",
        model_config=SimpleNamespace(context_len=4096),
        flush_cache=AsyncMock(return_value=SimpleNamespace(success=True)),
        set_startup_time=Mock(),
    )

    def create_worker(loaded_args, loaded_ports):
        manager.server_args = loaded_args
        worker_ipcs.append(loaded_ports.tokenizer_ipc_name)
        return manager

    # Only replace model/tokenizer construction and process-local side effects.
    # Shared-memory serialization, reads, publish() and get_serving() are real.
    monkeypatch.setattr(http_server, "get_main_process_id", os.getpid)
    monkeypatch.setattr(
        http_server, "get_tokenizer_worker_class", lambda args: create_worker
    )
    monkeypatch.setattr(http_server, "TemplateManager", Mock)
    monkeypatch.setattr(http_server.setproctitle, "setproctitle", lambda title: None)
    monkeypatch.setattr(http_server.torch, "set_num_threads", lambda count: None)
    monkeypatch.setattr(http_server, "_global_state", None)
    monkeypatch.setattr(app.router, "lifespan_context", lifespan)
    monkeypatch.setattr(app, "middleware_stack", None)
    # The native name is scoped to this test process, just as in the launcher.
    shm = http_server.write_data_for_multi_tokenizer(ports, args, info)
    try:
        with use_context(context):
            yield app, manager
    finally:
        shm.unlink()
        for ipc in worker_ipcs:
            Path(ipc.removeprefix("ipc://")).unlink(missing_ok=True)


def test_reimported_app_authenticates_models_and_admin_routes(worker_app):
    app, manager = worker_app
    with TestClient(app) as client:
        assert client.get("/v1/models").status_code == 401
        assert (
            client.get(
                "/v1/models", headers={"Authorization": "Bearer wrong"}
            ).status_code
            == 401
        )
        result = client.get("/v1/models", headers={"Authorization": "Bearer user"})
        assert result.status_code == 200
        assert result.json()["data"][0]["id"] == "auth-fixture"
        assert (
            client.post(
                "/flush_cache", headers={"Authorization": "Bearer user"}
            ).status_code
            == 401
        )
        manager.flush_cache.assert_not_awaited()
        assert (
            client.post(
                "/flush_cache", headers={"Authorization": "Bearer admin"}
            ).status_code
            == 200
        )


@pytest.mark.parametrize("path", ["/v1/chat/completions", "/v1/responses"])
def test_reimported_app_authenticates_streaming_requests(worker_app, monkeypatch, path):
    app, _ = worker_app

    async def events():
        yield "data: first\n\n"
        yield "data: [DONE]\n\n"

    async def chat(request, raw_request):
        return StreamingResponse(events(), media_type="text/event-stream")

    async def responses(request, raw_request):
        return events()

    handler = AsyncMock(side_effect=chat if path.endswith("completions") else responses)
    state_name = (
        "openai_serving_chat"
        if path.endswith("completions")
        else "openai_serving_responses"
    )
    serving = SimpleNamespace(handle_request=handler, create_responses=handler)
    monkeypatch.setattr(app.state, state_name, serving, raising=False)
    payload = {"model": "auth-fixture", "stream": True}
    if path.endswith("completions"):
        payload["messages"] = [{"role": "user", "content": "hello"}]
    else:
        payload["input"] = "hello"

    with TestClient(app) as client:
        for token in (None, "wrong"):
            headers = {} if token is None else {"Authorization": f"Bearer {token}"}
            assert client.post(path, json=payload, headers=headers).status_code == 401
        handler.assert_not_awaited()
        response = client.post(
            path, json=payload, headers={"Authorization": "Bearer user"}
        )
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/event-stream")
        assert response.text == "data: first\n\ndata: [DONE]\n\n"


@pytest.mark.parametrize(
    "worker_app,normal,admin",
    [
        ((None, None), [200] * 4, [200] * 4),
        (("user", None), [401, 401, 200, 401], [401, 401, 200, 401]),
        ((None, "admin"), [200] * 4, [401, 401, 401, 200]),
        (("user", "admin"), [401, 401, 200, 401], [401, 401, 401, 200]),
    ],
    indirect=["worker_app"],
)
def test_multi_tokenizer_published_keys_authenticate_requests(
    worker_app, normal, admin
):
    app, manager = worker_app
    with TestClient(app) as client:
        assert manager.server_args.tokenizer_worker_num == 4
        assert manager.max_req_input_len == 4096
        for path, statuses in (("/v1/models", normal), ("/flush_cache", admin)):
            for key, expected in zip((None, "wrong", "user", "admin"), statuses):
                headers = {} if key is None else {"Authorization": f"Bearer {key}"}
                response = client.get(path, headers=headers)
                assert response.status_code == expected


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, *sys.argv[1:]]))
