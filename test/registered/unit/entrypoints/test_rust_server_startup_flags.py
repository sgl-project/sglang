"""Rust-server mode honors the Python HTTP server's startup flags
``--warmups`` and ``--gc-warning-threshold-secs``."""

import asyncio
import json
import math
import socket
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest

from sglang.srt import rust_extensions
from sglang.srt.disaggregation.mode import FAKE_BOOTSTRAP_HOST
from sglang.srt.entrypoints import engine
from sglang.srt.entrypoints import warmup as warmup_registry
from sglang.srt.environ import envs
from sglang.srt.managers.io_struct import GenerateReqInput
from sglang.srt.runtime_context import get_context, get_parallel
from sglang.srt.rust_server import server as rust_server
from sglang.srt.rust_server import warmups as rust_warmups
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class _FakeListener:
    """Stands in for the Rust listener: records `/generate` bodies and answers
    them with `status`, or never within a test (`hang`). With `api_key`,
    `/generate` answers 401 without `Authorization: Bearer <api_key>`, as the
    listener's auth layer does."""

    def __init__(self, api_key=None, status=200, hang=False):
        self.generated = []
        self.authorizations = []
        self.api_key = api_key
        self.status = status
        self.hang = hang
        owner = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def _reply(self, code, body=b"{}"):
                self.send_response(code)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_POST(self):
                length = int(self.headers["Content-Length"])
                body = json.loads(self.rfile.read(length))
                authorization = self.headers.get("Authorization")
                owner.authorizations.append(authorization)
                if owner.api_key and authorization != f"Bearer {owner.api_key}":
                    self._reply(401, b'{"error": "Unauthorized"}')
                    return
                owner.generated.append(body)
                if owner.hang:
                    time.sleep(5)
                self._reply(owner.status, b'{"text": ""}')

        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.port = self.httpd.server_address[1]
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()

    def close(self):
        self.httpd.shutdown()


@pytest.fixture
def listener():
    fake = _FakeListener()
    yield fake
    fake.close()


@pytest.fixture
def registered_warmup():
    names = []

    def register(name, fn, rust_server):
        warmup_registry.warmup(name, rust_server=rust_server)(fn)
        names.append(name)

    yield register
    for name in names:
        warmup_registry._warmup_registry.pop(name, None)
        warmup_registry._rust_server_warmups.discard(name)


def _launch(
    port,
    parallel=None,
    server_cls=None,
    handle=None,
    drain_state=0,
    loop_starts=True,
    on_finish=None,
    **overrides,
):
    """`RustServer.launch` (or a subclass's) with the extension stubbed; returns
    the stub server class and the `_fail_startup` mock after the warmups
    finished. `drain_state` is the listener's `DRAIN_*` bits, or a callable
    returning them. With `loop_starts`, the scheduler loop's first `drain`
    follows the launch."""
    scheduler = SimpleNamespace(
        server_args=SimpleNamespace(),
        model_config=SimpleNamespace(is_multimodal=False, is_generation=True),
    )
    with (
        get_context().override_server_args(host="127.0.0.1", port=port, **overrides),
        get_parallel().override(
            **{
                "tp_rank": 0,
                "attn_tp_rank": 0,
                "attn_cp_rank": 0,
                "attn_dp_rank": 0,
                **(parallel or {}),
            }
        ),
        patch.object(rust_extensions, "load_rust_extension") as extension,
        patch.object(rust_server, "_partition_cores", return_value=(None, None)),
        patch.object(rust_server, "_build_server_args"),
        # The stub server's startup failure is a truthy Mock: the watcher would
        # SIGQUIT pytest's parent.
        patch.object(rust_server.RustServer, "_watch_startup_failure"),
        # Would SIGQUIT pytest's parent.
        patch.object(rust_server.RustServer, "_fail_startup") as fail_startup,
        patch.object(
            rust_server.RustServer,
            "drain_state",
            side_effect=drain_state if callable(drain_state) else lambda: drain_state,
        ),
    ):
        if handle is not None:
            extension.return_value.Server.return_value = handle
        if on_finish is not None:
            extension.return_value.Server.return_value.finish_custom_warmups.side_effect = on_finish
        instance = (server_cls or rust_server.RustServer).launch(scheduler)
        instance.server.recv_requests.return_value = []
        instance.server.drain_complete.return_value = False
        if loop_starts:
            assert instance.drain(1) == []
        if instance.warmup_thread is not None:
            instance.warmup_thread.join(timeout=30)
            assert not instance.warmup_thread.is_alive()
    return extension.return_value.Server, fail_startup


@pytest.mark.parametrize("disaggregation_mode", ["null", "prefill"])
def test_warmups_run_through_the_listener_before_health_is_released(
    listener, registered_warmup, disaggregation_mode
):
    async def shapes(mode, tokenizer_manager):
        req = GenerateReqInput(
            input_ids=[1, 2, 3], sampling_params={"max_new_tokens": 1}
        )
        if mode != "null":
            req.bootstrap_room = 0
            req.bootstrap_host = FAKE_BOOTSTRAP_HOST
        await tokenizer_manager.generate_request(req, None).__anext__()

    registered_warmup("rust_test_shapes", shapes, rust_server=True)
    registered_warmup("rust_test_more", shapes, rust_server=True)
    sent_before_release = []
    server_cls, fail_startup = _launch(
        listener.port,
        warmups="rust_test_shapes,rust_test_more",
        disaggregation_mode=disaggregation_mode,
        on_finish=lambda: sent_before_release.append(len(listener.generated)),
    )

    expected = {"input_ids": [1, 2, 3], "sampling_params": {"max_new_tokens": 1}}
    if disaggregation_mode != "null":
        expected |= {"bootstrap_room": 0, "bootstrap_host": FAKE_BOOTSTRAP_HOST}
    assert listener.generated == [expected, expected]
    server_cls.return_value.finish_custom_warmups.assert_called_once_with()
    # Health is released only after every warmup request was answered.
    assert sent_before_release == [2]
    fail_startup.assert_not_called()


class _WrappingServer(rust_server.RustServer):
    """A package's server whose extension handle wraps the base server, as a
    model package's extension may; the base handle is what owns the warmup gate."""

    def __init__(self, server, http_port):
        super().__init__(server.base_server, http_port=http_port)


def test_warmups_release_health_on_the_wrapped_base_server(listener, registered_warmup):
    async def one(mode, tokenizer_manager):
        await tokenizer_manager.generate_request(
            GenerateReqInput(input_ids=[1]), None
        ).__anext__()

    registered_warmup("rust_test_wrapped", one, rust_server=True)
    base_server = Mock()
    _launch(
        listener.port,
        server_cls=_WrappingServer,
        handle=SimpleNamespace(base_server=base_server),
        warmups="rust_test_wrapped",
    )

    assert len(listener.generated) == 1
    base_server.finish_custom_warmups.assert_called_once_with()


@pytest.mark.parametrize(
    ("api_key", "admin_api_key"), [("sk-api", None), (None, "sk-admin")]
)
def test_warmups_pass_the_api_key_auth(registered_warmup, api_key, admin_api_key):
    """`/generate` needs `--api-key` (the admin key does not unlock it), so a
    warmup without the key would get 401 and hold health forever."""

    async def one(mode, tokenizer_manager):
        await tokenizer_manager.generate_request(
            GenerateReqInput(input_ids=[1]), None
        ).__anext__()

    registered_warmup("rust_test_auth", one, rust_server=True)
    listener = _FakeListener(api_key=api_key)
    try:
        server_cls, _ = _launch(
            listener.port,
            warmups="rust_test_auth",
            api_key=api_key,
            admin_api_key=admin_api_key,
        )
    finally:
        listener.close()

    assert listener.generated == [{"input_ids": [1]}]
    assert listener.authorizations == [f"Bearer {api_key}" if api_key else None]
    server_cls.return_value.finish_custom_warmups.assert_called_once_with()


def test_warmups_bypass_the_proxy_env(listener, registered_warmup, monkeypatch):
    """The listener is the process's own: a proxy from the environment, with no
    `no_proxy` entry for the bind address, must not carry the warmup."""

    async def one(mode, tokenizer_manager):
        await tokenizer_manager.generate_request(
            GenerateReqInput(input_ids=[1]), None
        ).__anext__()

    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        dead_proxy = f"http://127.0.0.1:{sock.getsockname()[1]}"
    for name in ("http_proxy", "HTTP_PROXY", "https_proxy", "HTTPS_PROXY"):
        monkeypatch.setenv(name, dead_proxy)
    for name in ("no_proxy", "NO_PROXY"):
        monkeypatch.delenv(name, raising=False)

    registered_warmup("rust_test_proxy", one, rust_server=True)
    server_cls, fail_startup = _launch(listener.port, warmups="rust_test_proxy")

    assert listener.generated == [{"input_ids": [1]}]
    server_cls.return_value.finish_custom_warmups.assert_called_once_with()
    fail_startup.assert_not_called()


@pytest.mark.parametrize(
    ("request_input", "listener_kwargs"),
    [
        (GenerateReqInput(text="x", return_logprob=True), {}),
        (GenerateReqInput(input_ids=[1]), {"status": 500}),
        (GenerateReqInput(input_ids=[1]), {"hang": True}),
        # The listener answers a batch 200 even when an item failed.
        (GenerateReqInput(input_ids=[[1], [2]]), {}),
        # Not an Exception: it would escape the runner and hold health.
        (asyncio.CancelledError(), {}),
    ],
    ids=[
        "unforwarded-field",
        "listener-error",
        "listener-hangs",
        "batch",
        "base-exception",
    ],
)
def test_failed_warmup_takes_the_server_down(
    registered_warmup, request_input, listener_kwargs
):
    """A warmup that fails, or whose request outlives SGLANG_WARMUP_TIMEOUT,
    takes the server down as a failed startup warmup does, rather than leave
    health at 503 forever; the later warmups do not run."""

    async def boom(mode, tokenizer_manager):
        if isinstance(request_input, BaseException):
            raise request_input
        await tokenizer_manager.generate_request(request_input, None).__anext__()

    async def after(mode, tokenizer_manager):
        raise AssertionError("ran after a failed warmup")

    registered_warmup("rust_test_boom", boom, rust_server=True)
    registered_warmup("rust_test_after", after, rust_server=True)
    listener = _FakeListener(**listener_kwargs)
    try:
        with envs.SGLANG_WARMUP_TIMEOUT.override(1):
            started = time.monotonic()
            server_cls, fail_startup = _launch(
                listener.port, warmups="rust_test_boom,rust_test_after"
            )
            assert time.monotonic() - started < 10
    finally:
        listener.close()

    fail_startup.assert_called_once()
    assert "rust_test_boom" in fail_startup.call_args.args[0]
    server_cls.return_value.finish_custom_warmups.assert_not_called()


@pytest.mark.parametrize(
    ("drain_begins", "drain_bit"),
    [
        ("before", rust_server.DRAIN_DRAINING),
        ("between-warmups", rust_server.DRAIN_DRAINING),
        # Another rank's shutdown closed this listener without a drain here.
        ("between-requests", rust_server.DRAIN_CLOSED),
    ],
)
def test_drain_stops_the_warmups_without_failing(
    listener, registered_warmup, drain_begins, drain_bit
):
    """A SIGTERM drain stops the warmups, between two warmups or between two
    requests of one, and closes the listener under them; nothing reports a
    startup failure, which would SIGQUIT the launcher and cut the requests
    other listeners are draining."""
    draining = drain_begins == "before"

    async def first(mode, tokenizer_manager):
        nonlocal draining
        for _ in range(3):
            await tokenizer_manager.generate_request(
                GenerateReqInput(input_ids=[1]), None
            ).__anext__()
            if drain_begins == "between-requests":
                draining = True
        draining = True
        # The drain closed the listener.
        raise ConnectionRefusedError("listener closed")

    async def second(mode, tokenizer_manager):
        raise AssertionError("ran after the drain began")

    registered_warmup("rust_test_first", first, rust_server=True)
    registered_warmup("rust_test_second", second, rust_server=True)
    server_cls, fail_startup = _launch(
        listener.port,
        warmups="rust_test_first,rust_test_second",
        drain_state=lambda: drain_bit if draining else 0,
    )

    assert (
        len(listener.generated)
        == {
            "before": 0,
            "between-warmups": 3,
            "between-requests": 1,
        }[drain_begins]
    )
    fail_startup.assert_not_called()
    server_cls.return_value.finish_custom_warmups.assert_not_called()


def test_drain_during_the_last_request_keeps_health_held(listener, registered_warmup):
    """A drain that begins while the last warmup request runs: the request
    succeeds, but health is not released, which would start the Rust startup
    warmup under the drain."""
    draining = False

    async def last(mode, tokenizer_manager):
        nonlocal draining
        await tokenizer_manager.generate_request(
            GenerateReqInput(input_ids=[1]), None
        ).__anext__()
        draining = True

    registered_warmup("rust_test_last", last, rust_server=True)
    server_cls, fail_startup = _launch(
        listener.port,
        warmups="rust_test_last",
        drain_state=lambda: rust_server.DRAIN_DRAINING if draining else 0,
    )

    assert listener.generated == [{"input_ids": [1]}]
    fail_startup.assert_not_called()
    server_cls.return_value.finish_custom_warmups.assert_not_called()


@pytest.mark.parametrize("warmups", ["rust_test_blank,", " ,rust_test_blank"])
def test_blank_warmup_names_are_skipped(listener, registered_warmup, warmups):
    """An empty name (a trailing comma) runs nothing, as in Python mode."""

    async def one(mode, tokenizer_manager):
        await tokenizer_manager.generate_request(
            GenerateReqInput(input_ids=[1]), None
        ).__anext__()

    registered_warmup("rust_test_blank", one, rust_server=True)
    server_cls, fail_startup = _launch(listener.port, warmups=warmups)

    assert listener.generated == [{"input_ids": [1]}]
    server_cls.return_value.finish_custom_warmups.assert_called_once_with()
    fail_startup.assert_not_called()


def test_warmups_wait_for_the_scheduler_loop(listener, registered_warmup):
    """The first request waits for the scheduler loop, so its timeout does not
    count the rest of scheduler initialization after the listener binds."""

    async def one(mode, tokenizer_manager):
        await tokenizer_manager.generate_request(
            GenerateReqInput(input_ids=[1]), None
        ).__anext__()

    registered_warmup("rust_test_wait", one, rust_server=True)
    with patch.object(rust_server.RustServer, "drain_state", return_value=0):
        instance = None

        def launch():
            nonlocal instance
            with (
                get_context().override_server_args(
                    host="127.0.0.1", port=listener.port, warmups="rust_test_wait"
                ),
                get_parallel().override(
                    tp_rank=0, attn_tp_rank=0, attn_cp_rank=0, attn_dp_rank=0
                ),
                patch.object(rust_extensions, "load_rust_extension"),
                patch.object(
                    rust_server, "_partition_cores", return_value=(None, None)
                ),
                patch.object(rust_server, "_build_server_args"),
                patch.object(rust_server.RustServer, "_watch_startup_failure"),
                # Would SIGQUIT pytest's parent.
                patch.object(rust_server.RustServer, "_fail_startup"),
            ):
                instance = rust_server.RustServer.launch(
                    SimpleNamespace(
                        server_args=SimpleNamespace(),
                        model_config=SimpleNamespace(
                            is_multimodal=False, is_generation=True
                        ),
                    )
                )

        launch()
        instance.warmup_thread.join(timeout=1.5)
        assert instance.warmup_thread.is_alive()
        assert listener.generated == []

        instance.server.recv_requests.return_value = []
        instance.server.drain_complete.return_value = False
        instance.drain(1)
        instance.warmup_thread.join(timeout=30)
    assert listener.generated == [{"input_ids": [1]}]
    instance.server.finish_custom_warmups.assert_called_once_with()


@pytest.mark.parametrize(
    ("host", "url"),
    [
        ("0.0.0.0", "http://127.0.0.1:30001"),
        ("::", "http://[::1]:30001"),
        ("2001:db8::1", "http://[2001:db8::1]:30001"),
        ("10.0.0.2", "http://10.0.0.2:30001"),
    ],
)
def test_warmups_reach_a_wildcard_bind_over_loopback(host, url):
    with get_context().override_server_args(host=host):
        assert rust_server.RustServer._local_url(30001) == url


@pytest.mark.parametrize(
    ("configured", "mode", "expected"),
    [
        (-1, "null", 600),
        (-1, "prefill", 1800),
        (5, "null", 5),
        (math.inf, "null", 600),
        (math.nan, "decode", 1800),
    ],
)
def test_warmup_timeout_follows_the_startup_warmup(configured, mode, expected):
    """`SGLANG_WARMUP_TIMEOUT` as the Rust startup warmup reads it: a positive,
    finite value, else 600 s (1800 s in PD mode); `requests` cannot take inf."""
    with envs.SGLANG_WARMUP_TIMEOUT.override(configured):
        assert rust_warmups._warmup_timeout(mode) == expected


class _RecordingClient(rust_warmups._ListenerClient):
    """The listener client minus HTTP: records each request's body."""

    def __init__(self):
        self.payloads = []

    async def generate_request(self, req, request=None):
        self.payloads.append(self._payload(req))
        yield {"text": ""}


@pytest.mark.parametrize("disaggregation_mode", ["null", "prefill"])
@pytest.mark.parametrize(
    ("name", "requests"), [("voice_chat", 511), ("prefill_shapes", 18)]
)
def test_registered_warmups_send_forwardable_requests(
    disaggregation_mode, name, requests
):
    """Each built-in warmup marked Rust-capable sends only plain token-in generate
    requests the client forwards (with the fake bootstrap in PD mode)."""
    assert name in warmup_registry._rust_server_warmups
    client = _RecordingClient()
    asyncio.run(warmup_registry._warmup_registry[name](disaggregation_mode, client))

    assert len(client.payloads) == requests
    keys = {"input_ids", "sampling_params"}
    if disaggregation_mode != "null":
        keys |= {"bootstrap_host", "bootstrap_room"}
    for payload in client.payloads:
        assert set(payload) == keys
        assert all(isinstance(token, int) for token in payload["input_ids"])
        if disaggregation_mode != "null":
            assert payload["bootstrap_host"] == FAKE_BOOTSTRAP_HOST


def test_launcher_refuses_warmups_rust_cannot_run(registered_warmup):
    """The launcher refuses, before any scheduler loads weights, a warmup an
    entry point registered only in the launcher process, which the spawned
    scheduler could not see."""
    registered_warmup("rust_test_launcher_only", lambda *args: None, rust_server=False)
    registered_warmup("rust_test_capable", lambda *args: None, rust_server=True)
    for warmups in ("rust_test_launcher_only", "rust_test_capable,not_registered"):
        with get_context().override_server_args(warmups=warmups):
            with envs.SGLANG_RUST_SERVER.override(False):
                engine._check_rust_server_warmups()
            with (
                envs.SGLANG_RUST_SERVER.override(True),
                pytest.raises(ValueError) as refused,
            ):
                engine._check_rust_server_warmups()
        assert warmups.split(",")[-1] in str(refused.value)
        assert "rust_test_capable" not in str(refused.value)
    with (
        get_context().override_server_args(warmups="rust_test_capable"),
        envs.SGLANG_RUST_SERVER.override(True),
    ):
        engine._check_rust_server_warmups()


def test_launch_checks_warmups_before_spawning_schedulers():
    """`Engine._launch_subprocesses` runs the warmup check before it spawns any
    scheduler, and gives the published context back when it refuses."""
    refused = ValueError("refused --warmups")
    with (
        patch.multiple(
            engine,
            configure_logger=Mock(),
            _set_envs_and_config=Mock(),
            load_plugins=Mock(),
            resolve_auto_parsers=Mock(),
            snapshot_context=Mock(return_value="before"),
            publish=Mock(),
            restore_context=Mock(),
            _check_rust_server_warmups=Mock(side_effect=refused),
        ),
        patch.object(engine.Engine, "_launch_scheduler_processes") as spawn,
        pytest.raises(ValueError, match="refused --warmups"),
    ):
        engine.Engine._launch_subprocesses(Mock(), Mock(), Mock(), Mock())
    spawn.assert_not_called()


@pytest.mark.parametrize("name", ["rust_test_tokenizer", "not_registered"])
def test_warmup_rust_cannot_run_is_refused(listener, registered_warmup, name):
    """A warmup not marked Rust-capable, or one this scheduler does not know
    (maybe registered only in the launcher; Python mode warns and skips it),
    is refused before the listener binds rather than left unrun."""

    async def needs_tokenizer(mode, tokenizer_manager):
        tokenizer_manager.tokenizer.encode("x")

    registered_warmup("rust_test_tokenizer", needs_tokenizer, rust_server=False)
    with (
        get_context().override_server_args(
            host="127.0.0.1", port=listener.port, warmups=name
        ),
        patch.object(rust_extensions, "load_rust_extension") as extension,
        pytest.raises(ValueError, match=name),
    ):
        rust_server.RustServer.launch(
            SimpleNamespace(
                model_config=SimpleNamespace(is_multimodal=False, is_generation=True)
            )
        )
    # Refused before the listener port is bound.
    extension.return_value.Server.assert_not_called()


@pytest.mark.parametrize("threshold", [0.25, 0.0, -1.0])
def test_gc_warning_watches_the_listener_scheduler(listener, threshold):
    """Installed with the configured threshold when positive; Python disables
    it otherwise."""
    with patch.object(rust_server, "configure_gc_warning") as configure:
        _launch(listener.port, gc_warning_threshold_secs=threshold)
    if threshold > 0:
        configure.assert_called_once_with(threshold)
    else:
        configure.assert_not_called()


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
