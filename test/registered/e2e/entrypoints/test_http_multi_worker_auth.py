"""CPU HTTP acceptance: real worker configuration, model work doubled.

Four fresh Uvicorn workers must authenticate every route before and after HUP.
The supervisor writes native shared memory; each worker runs init_multi_tokenizer.
"""

import os
import signal
import socket
import subprocess
import sys
import time
from contextlib import asynccontextmanager
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=150, suite="base-a-test-cpu")


def _http_server():
    from sglang.test.test_utils import maybe_stub_sgl_kernel

    maybe_stub_sgl_kernel()
    from sglang.srt.entrypoints import http_server

    return http_server


class _ModelDouble:
    def __init__(self, args, ports):
        self.server_args = args
        self.ipc = ports.tokenizer_ipc_name
        self.served_model_name = "auth-fixture"
        self.model_config = SimpleNamespace(context_len=4096)

    def set_startup_time(self, value):
        self.startup_time = value

    async def flush_cache(self, **kwargs):
        return SimpleNamespace(success=True)


class _TemplateDouble:
    def initialize_templates(self, **kwargs):
        pass


class _WorkerHeader:
    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        async def with_worker(message):
            if message["type"] == "http.response.start":
                message["headers"].append((b"x-test-worker", str(os.getpid()).encode()))
            await send(message)

        await self.app(scope, receive, with_worker)


def create_app():
    from fastapi.responses import StreamingResponse

    from sglang.srt.runtime_context import get_context, get_serving, publish_role

    server = _http_server()
    server.get_tokenizer_worker_class = lambda args: _ModelDouble
    server.TemplateManager = _TemplateDouble

    async def events():
        yield "data: first\n\n"
        yield "data: [DONE]\n\n"

    async def chat(request, raw_request):
        return StreamingResponse(events(), media_type="text/event-stream")

    async def responses(request, raw_request):
        return events()

    @asynccontextmanager
    async def lifespan(app):
        assert app.middleware_stack is not None
        assert not get_context().is_config_namespace_published("serving")
        args = await server.init_multi_tokenizer()
        assert publish_role() == "tokenizer"
        assert get_serving().api_key == "test-user"
        assert get_serving().admin_api_key == "test-admin"
        assert args.tokenizer_worker_num == 4
        app.state.openai_serving_chat = SimpleNamespace(handle_request=chat)
        app.state.openai_serving_responses = SimpleNamespace(create_responses=responses)
        ready = Path(os.environ["SGLANG_AUTH_TEST_READY"])
        (ready / f"{os.getpid()}.ready").write_text(str(os.getpid()))
        try:
            yield
        finally:
            manager = server.get_global_state().tokenizer_manager
            Path(manager.ipc.removeprefix("ipc://")).unlink(missing_ok=True)

    server.app.router.lifespan_context = lifespan
    return _WorkerHeader(server.app)


def _serve(port):
    import uvicorn

    from sglang.srt.server_args import PortArgs, ServerArgs

    server = _http_server()
    args = ServerArgs(
        model_path="dummy",
        device="cpu",
        api_key="test-user",
        admin_api_key="test-admin",
        tokenizer_worker_num=4,
    )
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
    shm = server.write_data_for_multi_tokenizer(
        ports, args, {"max_req_input_len": 4096, "startup_time": {}}
    )
    try:
        uvicorn.run(
            "test_http_multi_worker_auth:create_app",
            factory=True,
            host="127.0.0.1",
            port=port,
            workers=4,
            timeout_worker_healthcheck=90,
            log_level="warning",
        )
    finally:
        shm.unlink()


def _ready_workers(process, ready, previous=frozenset()):
    deadline = time.monotonic() + 120
    while time.monotonic() < deadline:
        assert process.poll() is None, f"Supervisor exited {process.returncode}"
        workers = {int(p.stem) for p in ready.glob("*.ready")} - previous
        if len(workers) == 4:
            return workers
        time.sleep(0.1)
    pytest.fail("Four workers did not become ready")


def _check_workers(base_url, workers):
    scenarios = [
        ("/v1/models", None, 401),
        ("/v1/models", "wrong", 401),
        ("/v1/models", "test-user", 200),
        ("/v1/models", "test-admin", 401),
        ("/flush_cache", None, 401),
        ("/flush_cache", "test-user", 401),
        ("/flush_cache", "test-admin", 200),
    ]
    for path in ("/v1/chat/completions", "/v1/responses"):
        scenarios.extend(
            (path, key, status)
            for key, status in ((None, 401), ("wrong", 401), ("test-user", 200))
        )
    with httpx.Client(
        base_url=base_url, headers={"Connection": "close"}, timeout=10
    ) as client:
        for path, key, status in scenarios:
            seen = set()
            for _ in range(200):
                headers = {} if key is None else {"Authorization": f"Bearer {key}"}
                if path.startswith("/v1/") and path != "/v1/models":
                    body = {"model": "auth-fixture", "stream": True}
                    if path.endswith("completions"):
                        body["messages"] = [{"role": "user", "content": "hello"}]
                    else:
                        body["input"] = "hello"
                    response = client.post(path, json=body, headers=headers)
                else:
                    response = client.get(path, headers=headers)
                assert response.status_code == status, (path, response.text)
                pid = int(response.headers["x-test-worker"])
                assert pid in workers
                seen.add(pid)
                if status == 200 and path in ("/v1/chat/completions", "/v1/responses"):
                    assert response.text == "data: first\n\ndata: [DONE]\n\n"
                if seen == workers:
                    break
            assert seen == workers, (path, key, seen, workers)


@pytest.mark.skipif(sys.platform != "linux", reason="POSIX workers and SIGHUP")
def test_four_workers_authenticate_before_and_after_replacement(tmp_path):
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
    env = dict(os.environ, SGLANG_AUTH_TEST_READY=str(tmp_path))
    # Do not initialize devices for this CPU-only model double. Keep CUDA's
    # legacy numeric spelling for test_utils' port selection on CPU runners.
    env.update(
        CUDA_VISIBLE_DEVICES="0", HIP_VISIBLE_DEVICES="", ROCR_VISIBLE_DEVICES=""
    )
    # Preserve source-checkout imports in both the supervisor and spawned workers.
    env["PYTHONPATH"] = os.pathsep.join([str(Path(__file__).parent), *sys.path])
    with (tmp_path / "server.log").open("w") as log:
        process = subprocess.Popen(
            [sys.executable, str(Path(__file__).resolve()), "--serve", str(port)],
            env=env,
            stdout=log,
            stderr=log,
            start_new_session=True,
        )
        try:
            original = _ready_workers(process, tmp_path)
            _check_workers(f"http://127.0.0.1:{port}", original)
            process.send_signal(signal.SIGHUP)
            replacement = _ready_workers(process, tmp_path, original)
            deadline = time.monotonic() + 20
            while time.monotonic() < deadline:
                alive = set()
                for pid in original:
                    try:
                        os.kill(pid, 0)
                        alive.add(pid)
                    except ProcessLookupError:
                        pass
                if not alive:
                    break
                time.sleep(0.1)
            assert not alive
            _check_workers(f"http://127.0.0.1:{port}", replacement)
        except BaseException:
            print((tmp_path / "server.log").read_text()[-12000:])
            raise
        finally:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait(timeout=5)


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "--serve":
        _serve(int(sys.argv[2]))
    else:
        sys.exit(pytest.main([__file__, *sys.argv[1:]]))
