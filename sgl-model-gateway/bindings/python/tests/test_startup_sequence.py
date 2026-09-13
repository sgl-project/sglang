"""Focused runtime tests for the Python router and server-launch helpers."""


def _install_sglang_stubs(monkeypatch):
    """Install lightweight stubs for sglang.srt to avoid heavy deps during unit tests."""
    import sys
    import types

    sglang_mod = types.ModuleType("sglang")
    srt_mod = types.ModuleType("sglang.srt")
    entry_mod = types.ModuleType("sglang.srt.entrypoints")
    http_server_mod = types.ModuleType("sglang.srt.entrypoints.http_server")
    server_args_mod = types.ModuleType("sglang.srt.server_args")
    # sglang.srt.utils was refactored from a module into a package; launch_server.py
    # imports from sglang.srt.utils.network, so the stub must model the submodule.
    utils_mod = types.ModuleType("sglang.srt.utils")
    network_mod = types.ModuleType("sglang.srt.utils.network")

    def launch_server(_args):
        return None

    class ServerArgs:
        # Minimal fields used by launch_server_process
        def __init__(self):
            self.port = 0
            self.base_gpu_id = 0
            self.dp_size = 1
            self.tp_size = 1

        @staticmethod
        def add_cli_args(_parser):
            return None

        @staticmethod
        def from_cli_args(_args):
            sa = ServerArgs()
            if hasattr(_args, "dp_size"):
                sa.dp_size = _args.dp_size
            if hasattr(_args, "tp_size"):
                sa.tp_size = _args.tp_size
            if hasattr(_args, "host"):
                sa.host = _args.host
            else:
                sa.host = "127.0.0.1"
            return sa

    def is_port_available(_port: int) -> bool:
        return True

    http_server_mod.launch_server = launch_server
    server_args_mod.ServerArgs = ServerArgs
    network_mod.is_port_available = is_port_available
    utils_mod.network = network_mod

    # Also stub external deps imported at module top-level
    def _dummy_get(*_a, **_k):
        raise NotImplementedError

    requests_stub = types.SimpleNamespace(
        exceptions=types.SimpleNamespace(RequestException=Exception), get=_dummy_get
    )
    setproctitle_stub = types.SimpleNamespace(setproctitle=lambda *_a, **_k: None)

    monkeypatch.setitem(sys.modules, "requests", requests_stub)
    monkeypatch.setitem(sys.modules, "setproctitle", setproctitle_stub)

    monkeypatch.setitem(sys.modules, "sglang", sglang_mod)
    monkeypatch.setitem(sys.modules, "sglang.srt", srt_mod)
    monkeypatch.setitem(sys.modules, "sglang.srt.entrypoints", entry_mod)
    monkeypatch.setitem(
        sys.modules, "sglang.srt.entrypoints.http_server", http_server_mod
    )
    monkeypatch.setitem(sys.modules, "sglang.srt.server_args", server_args_mod)
    monkeypatch.setitem(sys.modules, "sglang.srt.utils", utils_mod)
    monkeypatch.setitem(sys.modules, "sglang.srt.utils.network", network_mod)


def test_router_defaults_and_start(monkeypatch):
    """Router wrapper: defaults normalization and start() call.

    Mocks the Rust-backed _Router to avoid native deps.
    """
    from sglang_router import router as router_mod

    captured = {}

    class FakeRouter:
        def __init__(self, **kwargs):
            captured.update(kwargs)

        def start(self):
            captured["started"] = True

    monkeypatch.setattr(router_mod, "_Router", FakeRouter, raising=True)

    from sglang_router.router_args import RouterArgs as _RouterArgs

    Router = router_mod.Router
    args = _RouterArgs(
        worker_urls=["http://w1:8000"],
        policy="round_robin",
        selector=None,
        prefill_selector=None,
        decode_selector=None,
        cors_allowed_origins=None,
    )

    r = Router.from_args(args)

    # Defaults preserved/normalized by Router.from_args
    assert captured["selector"] is None
    assert captured["prefill_selector"] is None
    assert captured["decode_selector"] is None
    assert captured["cors_allowed_origins"] is None
    assert captured["worker_urls"] == ["http://w1:8000"]
    from sglang_router.sglang_router_rs import PolicyType

    assert captured["policy"] == PolicyType.RoundRobin

    r.start()
    assert captured.get("started") is True


def test_find_available_ports_and_wait_health(monkeypatch):
    """launch_server helpers: port finding and health waiting with transient error."""
    _install_sglang_stubs(monkeypatch)
    import importlib

    ls = importlib.import_module("sglang_router.launch_server")

    # Deterministic increments
    monkeypatch.setattr(ls.random, "randint", lambda a, b: 100)
    ports = ls.find_available_ports(30000, 3)
    assert ports == [30000, 30100, 30200]

    calls = {"n": 0}

    class Ok:
        status_code = 200

    def fake_get(_url, timeout=5):
        calls["n"] += 1
        if calls["n"] == 1:
            raise ls.requests.exceptions.RequestException("boom")
        return Ok()

    monkeypatch.setattr(ls.requests, "get", fake_get)
    monkeypatch.setattr(ls.time, "sleep", lambda _s: None)
    base = {"t": 0.0}
    monkeypatch.setattr(
        ls.time,
        "perf_counter",
        lambda: base.__setitem__("t", base["t"] + 0.1) or base["t"],
    )

    assert ls.wait_for_server_health("127.0.0.1", 12345, timeout=1)


def test_launch_server_process_and_cleanup(monkeypatch):
    """launch_server: process creation args and cleanup SIGTERM/SIGKILL logic."""
    _install_sglang_stubs(monkeypatch)
    import importlib

    ls = importlib.import_module("sglang_router.launch_server")

    created = {}

    class FakeProcess:
        def __init__(self, target, args):
            created["target"] = target
            created["args"] = args
            self.pid = 4242
            self._alive = True

        def start(self):
            created["started"] = True

        def join(self, timeout=None):
            return None

        def is_alive(self):
            return self._alive

    monkeypatch.setattr(ls.mp, "Process", FakeProcess)

    import sys as _sys

    SA = _sys.modules["sglang.srt.server_args"].ServerArgs
    sa = SA()
    sa.tp_size = 2

    ls.launch_server_process(sa, worker_port=31001, dp_id=3)
    assert created.get("started") is True
    targ, targ_args = created["target"], created["args"]
    assert targ is ls.run_server
    passed_sa = targ_args[0]
    assert passed_sa.port == 31001
    assert passed_sa.base_gpu_id == 3 * 2
    assert passed_sa.dp_size == 1

    # cleanup_processes
    p1 = FakeProcess(target=None, args=())
    p1._alive = False
    p2 = FakeProcess(target=None, args=())
    p2._alive = True

    calls = []

    def fake_killpg(pid, sig):
        calls.append((pid, sig))

    monkeypatch.setattr(ls.os, "killpg", fake_killpg)

    ls.cleanup_processes([p1, p2])

    import signal as _sig

    assert (p1.pid, _sig.SIGTERM) in calls and (p2.pid, _sig.SIGTERM) in calls
    assert (p2.pid, _sig.SIGKILL) in calls


def test_launch_server_process_declares_on_a_resolved_record(monkeypatch):
    """A record that carries its resolution takes the declaration channel.

    The stub above has no `replace_resolved`, so it exercises the older wheels'
    path. A current `ServerArgs` refuses plain assignment once resolution has
    finished; the per-worker values reach the child as a declaration on a copy,
    and the parent keeps what the operator passed.
    """
    _install_sglang_stubs(monkeypatch)
    import importlib

    ls = importlib.import_module("sglang_router.launch_server")

    created = {}

    class FakeProcess:
        def __init__(self, target, args):
            created["target"] = target
            created["args"] = args
            self.pid = 4243

        def start(self):
            created["started"] = True

    monkeypatch.setattr(ls.mp, "Process", FakeProcess)

    calls = []

    class ResolvedServerArgs:
        def __init__(self, **fields):
            self.port = fields.get("port", 30000)
            self.base_gpu_id = fields.get("base_gpu_id", 0)
            self.dp_size = fields.get("dp_size", 4)
            self.tp_size = fields.get("tp_size", 2)

        def replace_resolved(self, source, **changes):
            calls.append((source, dict(changes)))
            fields = dict(vars(self))
            fields.update(changes)
            return ResolvedServerArgs(**fields)

    parent = ResolvedServerArgs()
    proc = ls.launch_server_process(parent, worker_port=31002, dp_id=3)

    assert created.get("started") is True
    assert proc.pid == 4243
    assert len(calls) == 1
    source, changes = calls[0]
    assert source == "sglang_router.launch_server_process"
    assert changes == {"port": 31002, "base_gpu_id": 6, "dp_size": 1}

    worker = created["args"][0]
    assert (worker.port, worker.base_gpu_id, worker.dp_size) == (31002, 6, 1)
    assert (parent.port, parent.base_gpu_id, parent.dp_size) == (30000, 0, 4)
