"""Supervisor tests that do not require CUDA or the Rust router extension."""

import importlib
import json
import sys
import types


def _import_launch_server_with_stubs(monkeypatch):
    sglang_mod = types.ModuleType("sglang")
    srt_mod = types.ModuleType("sglang.srt")
    entry_mod = types.ModuleType("sglang.srt.entrypoints")
    server_args_mod = types.ModuleType("sglang.srt.server_args")
    utils_mod = types.ModuleType("sglang.srt.utils")
    network_mod = types.ModuleType("sglang.srt.utils.network")
    launch_router_mod = types.ModuleType("sglang_router.launch_router")

    class ServerArgs:
        def __init__(self, **fields):
            for name, value in fields.items():
                setattr(self, name, value)
            self.tp_size = getattr(self, "tp_size", 1)
            self.grpc_mode = getattr(self, "grpc_mode", False)

    class RouterArgs:
        def __init__(self, **fields):
            self.host = fields.pop("host", "0.0.0.0")
            self.port = fields.pop("port", 30000)
            for name, value in fields.items():
                setattr(self, name, value)

    def launch_router(_args):
        return None

    def is_port_available(_port):
        return True

    requests_stub = types.SimpleNamespace(
        exceptions=types.SimpleNamespace(RequestException=Exception)
    )
    setproctitle_stub = types.SimpleNamespace(setproctitle=lambda *_args: None)

    server_args_mod.ServerArgs = ServerArgs
    network_mod.is_port_available = is_port_available
    utils_mod.network = network_mod
    launch_router_mod.RouterArgs = RouterArgs
    launch_router_mod.launch_router = launch_router

    monkeypatch.setitem(sys.modules, "requests", requests_stub)
    monkeypatch.setitem(sys.modules, "setproctitle", setproctitle_stub)
    monkeypatch.setitem(sys.modules, "sglang", sglang_mod)
    monkeypatch.setitem(sys.modules, "sglang.srt", srt_mod)
    monkeypatch.setitem(sys.modules, "sglang.srt.entrypoints", entry_mod)
    monkeypatch.setitem(sys.modules, "sglang.srt.server_args", server_args_mod)
    monkeypatch.setitem(sys.modules, "sglang.srt.utils", utils_mod)
    monkeypatch.setitem(sys.modules, "sglang.srt.utils.network", network_mod)
    monkeypatch.setitem(sys.modules, "sglang_router.launch_router", launch_router_mod)
    sys.modules.pop("sglang_router.launch_server", None)
    return importlib.import_module("sglang_router.launch_server")


def test_launch_multi_model_server_builds_isolated_worker_processes(
    monkeypatch, tmp_path
):
    ls = _import_launch_server_with_stubs(monkeypatch)
    config_path = tmp_path / "models.json"
    config_path.write_text(
        json.dumps(
            {
                "router": {"host": "0.0.0.0", "port": 30000},
                "models": [
                    {
                        "model_id": "qwen",
                        "model_path": "/models/qwen",
                        "gpu_groups": [[0], [1]],
                    },
                    {
                        "model_id": "glm",
                        "model_path": "/models/glm",
                        "gpu_groups": [[2, 3]],
                        "server_args": {"tp_size": 2},
                    },
                ],
            }
        )
    )

    class FakeProcess:
        next_pid = 100

        def __init__(self, target, args):
            self.target = target
            self.args = args
            self.pid = FakeProcess.next_pid
            FakeProcess.next_pid += 1

        def start(self):
            return None

        def join(self, timeout=None):
            return None

        def is_alive(self):
            return False

    launches = []
    registrations = []
    cleanups = []

    monkeypatch.setattr(ls.mp, "Process", FakeProcess)
    monkeypatch.setattr(ls, "wait_for_server_health", lambda *_args: True)
    monkeypatch.setattr(ls, "find_available_ports", lambda _base, _count: [31000, 31100, 31200])
    monkeypatch.setattr(ls.signal, "signal", lambda *_args: None)
    monkeypatch.setattr(ls, "cleanup_processes", lambda processes: cleanups.append(list(processes)))

    def fake_launch(worker_args, worker_port, dp_rank, **kwargs):
        launches.append((worker_args, worker_port, dp_rank, kwargs))
        return FakeProcess(None, ())

    monkeypatch.setattr(ls, "launch_server_process", fake_launch)
    monkeypatch.setattr(
        ls,
        "register_igw_worker",
        lambda _host, _port, url, model_id, _timeout: registrations.append(
            (url, model_id)
        )
        or f"{model_id}-worker",
    )

    ls.launch_multi_model_server(str(config_path))

    assert [(worker.port if hasattr(worker, "port") else port, rank) for worker, port, rank, _ in launches] == [
        (31000, 0),
        (31100, 0),
        (31200, 0),
    ]
    assert [worker.served_model_name for worker, *_ in launches] == [
        "qwen",
        "qwen",
        "glm",
    ]
    assert [kwargs["worker_env"] for *_, kwargs in launches] == [
        {"CUDA_VISIBLE_DEVICES": "0"},
        {"CUDA_VISIBLE_DEVICES": "1"},
        {"CUDA_VISIBLE_DEVICES": "2,3"},
    ]
    assert registrations == [
        ("http://127.0.0.1:31000", "qwen"),
        ("http://127.0.0.1:31100", "qwen"),
        ("http://127.0.0.1:31200", "glm"),
    ]
    assert len(cleanups) == 2


def test_register_igw_worker_uses_the_configured_model_id(monkeypatch):
    ls = _import_launch_server_with_stubs(monkeypatch)
    captured = {}

    class Response:
        status_code = 202
        text = "accepted"

        @staticmethod
        def json():
            return {"worker_id": "worker-123"}

    def fake_post(url, json, timeout):
        captured.update(url=url, json=json, timeout=timeout)
        return Response()

    monkeypatch.setattr(ls.requests, "post", fake_post, raising=False)
    monkeypatch.setattr(ls, "wait_for_worker_registration", lambda *_args: True)

    worker_id = ls.register_igw_worker(
        "0.0.0.0", 30000, "http://127.0.0.1:31000", "qwen", 60
    )

    assert worker_id == "worker-123"
    assert captured == {
        "url": "http://127.0.0.1:30000/workers",
        "json": {"url": "http://127.0.0.1:31000", "model_id": "qwen"},
        "timeout": 10,
    }


def test_launch_multi_model_server_starts_resource_aware_router(monkeypatch, tmp_path):
    ls = _import_launch_server_with_stubs(monkeypatch)
    config_path = tmp_path / "models.json"
    config_path.write_text(
        json.dumps(
            {
                "router": {"host": "127.0.0.1", "port": 30001},
                "model_resolver": {
                    "host": "127.0.0.1",
                    "port": 30000,
                    "profiles": [
                        {
                            "model_id": "general-chat",
                            "candidates": ["qwen", "glm"],
                        }
                    ],
                },
                "models": [
                    {
                        "model_id": "qwen",
                        "model_path": "/models/qwen",
                        "gpu_groups": [[0]],
                    },
                    {
                        "model_id": "glm",
                        "model_path": "/models/glm",
                        "gpu_groups": [[1]],
                    },
                ],
            }
        )
    )

    class FakeProcess:
        created = []
        next_pid = 200

        def __init__(self, target, args):
            self.target = target
            self.args = args
            self.pid = FakeProcess.next_pid
            FakeProcess.next_pid += 1
            FakeProcess.created.append(self)

        def start(self):
            return None

        def join(self, timeout=None):
            return None

        def is_alive(self):
            return False

    monkeypatch.setattr(ls.mp, "Process", FakeProcess)
    monkeypatch.setattr(ls, "wait_for_server_health", lambda *_args: True)
    monkeypatch.setattr(ls, "find_available_ports", lambda _base, _count: [31000, 31100])
    monkeypatch.setattr(ls.signal, "signal", lambda *_args: None)
    monkeypatch.setattr(ls, "cleanup_processes", lambda _processes: None)
    monkeypatch.setattr(
        ls,
        "launch_server_process",
        lambda *_args, **_kwargs: FakeProcess(None, ()),
    )
    monkeypatch.setattr(
        ls,
        "register_igw_worker",
        lambda _host, _port, _url, model_id, _timeout: f"{model_id}-worker",
    )

    ls.launch_multi_model_server(str(config_path))

    resolver_processes = [
        process
        for process in FakeProcess.created
        if process.target is ls.run_resource_aware_router
    ]
    assert len(resolver_processes) == 1
    resolver_config, endpoints, backend_url = resolver_processes[0].args
    assert resolver_config.port == 30000
    assert [(endpoint.model_id, endpoint.url) for endpoint in endpoints] == [
        ("qwen", "http://127.0.0.1:31000"),
        ("glm", "http://127.0.0.1:31100"),
    ]
    assert backend_url == "http://127.0.0.1:30001"
