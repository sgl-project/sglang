"""uvicorn workers spawned by the multi-tokenizer HTTP server must not re-run
sglang's launcher module as ``__main__``: a slow main module would delay each
worker's health-check thread past the supervisor's timeout, and the
supervisor would kill and respawn it in a loop. A user's own launcher script
still runs in every worker.
"""

import importlib.machinery
import os
import signal
import socket
import subprocess
import sys
import tempfile
import textwrap
import time
import types
import unittest
import urllib.request
from multiprocessing import spawn
from types import SimpleNamespace
from unittest import mock

from uvicorn.config import LOGGING_CONFIG

from sglang.srt.entrypoints import http_server, uvicorn_worker_spawn
from sglang.srt.entrypoints.uvicorn_worker_spawn import (
    skip_main_reimport_in_spawned_workers,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=15, suite="base-a-test-cpu")

_MAIN_KEYS = ("init_main_from_name", "init_main_from_path")

_SLOW_MAIN_SERVER = textwrap.dedent(
    """
    import sys
    import time

    if __name__ == "__mp_main__":
        time.sleep(600)


    async def app(scope, receive, send):
        await send({"type": "http.response.start", "status": 200, "headers": []})
        await send({"type": "http.response.body", "body": b"ok"})


    if __name__ == "__main__":
        import uvicorn

        from sglang.srt.entrypoints import uvicorn_worker_spawn

        # Stand in for python -m sglang.launch_server, whose re-import is slow
        # on a loaded host.
        uvicorn_worker_spawn._launched_by_sglang_entrypoint = lambda: True
        with uvicorn_worker_spawn.skip_main_reimport_in_spawned_workers():
            uvicorn.run(
                "slow_main_server:app",
                host="127.0.0.1",
                port=int(sys.argv[1]),
                workers=2,
                # Far below the 600 s re-import, with headroom for a loaded
                # CI runner to import uvicorn in the worker.
                timeout_worker_healthcheck=20,
                lifespan="off",
                log_level="warning",
            )
    """
)


def _free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class TestLaunchedBySglangEntrypoint(CustomTestCase):
    def _launched(self, **main_attrs):
        fake = types.ModuleType("__main__")
        for name, value in main_attrs.items():
            setattr(fake, name, value)
        with mock.patch.dict(sys.modules, {"__main__": fake}):
            return uvicorn_worker_spawn._launched_by_sglang_entrypoint()

    def test_python_m_launch_server(self):
        spec = importlib.machinery.ModuleSpec("sglang.launch_server", None)
        self.assertTrue(self._launched(__spec__=spec))

    def test_sglang_console_script(self):
        self.assertTrue(self._launched(__spec__=None, __file__="/venv/bin/sglang"))

    def test_user_launcher_script(self):
        self.assertFalse(self._launched(__spec__=None, __file__="/work/my_launcher.py"))


class TestUvicornWorkerSpawn(CustomTestCase):
    def test_main_module_dropped_only_inside_context(self):
        self.assertTrue(
            any(k in spawn.get_preparation_data("outside") for k in _MAIN_KEYS)
        )
        original = spawn.get_preparation_data

        with mock.patch.object(
            uvicorn_worker_spawn, "_launched_by_sglang_entrypoint", return_value=True
        ):
            with skip_main_reimport_in_spawned_workers():
                data = spawn.get_preparation_data("inside")
                self.assertFalse(any(k in data for k in _MAIN_KEYS))
                self.assertIn("sys_path", data)

        self.assertIs(spawn.get_preparation_data, original)

    def test_user_launcher_script_still_runs_in_workers(self):
        self.assertFalse(uvicorn_worker_spawn._launched_by_sglang_entrypoint())
        with skip_main_reimport_in_spawned_workers():
            data = spawn.get_preparation_data("inside")
            self.assertTrue(any(k in data for k in _MAIN_KEYS))

    def test_http_server_starts_uvicorn_workers_inside_the_context(self):
        original = spawn.get_preparation_data
        main_keys_seen_by_workers = []

        def fake_uvicorn_run(app_path, **kwargs):
            data = spawn.get_preparation_data("worker")
            main_keys_seen_by_workers.append([k for k in _MAIN_KEYS if k in data])

        serving = SimpleNamespace(
            tokenizer_worker_num=2,
            enable_http2=False,
            enable_ssl_refresh=False,
            enable_http_worker_reuseport=False,
            host="127.0.0.1",
            port=30000,
            fastapi_root_path="",
            ssl_certfile=None,
            ssl_keyfile=None,
            ssl_ca_certs=None,
            ssl_keyfile_password=None,
        )
        with (
            mock.patch.object(
                uvicorn_worker_spawn,
                "_launched_by_sglang_entrypoint",
                return_value=True,
            ),
            mock.patch.object(http_server, "get_serving", return_value=serving),
            mock.patch.object(
                http_server,
                "get_observability",
                return_value=SimpleNamespace(log_level_http=None, log_level="error"),
            ),
            mock.patch.object(http_server, "set_global_state"),
            mock.patch.object(http_server, "_global_state", None),
            mock.patch.object(http_server, "write_data_for_multi_tokenizer"),
            mock.patch.object(http_server, "set_uvicorn_logging_configs"),
            mock.patch.dict(LOGGING_CONFIG["loggers"]),
            mock.patch.object(http_server.uvicorn, "run", side_effect=fake_uvicorn_run),
        ):
            http_server._setup_and_run_http_server(
                server_args=None,
                tokenizer_manager=mock.Mock(),
                template_manager=None,
                port_args=None,
                scheduler_infos=[{}],
                subprocess_watchdog=None,
            )

        self.assertEqual(main_keys_seen_by_workers, [[]])
        self.assertIs(spawn.get_preparation_data, original)

    def test_workers_serve_although_main_module_is_slow(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            with open(os.path.join(tmp_dir, "slow_main_server.py"), "w") as f:
                f.write(_SLOW_MAIN_SERVER)
            port = _free_port()
            proc = subprocess.Popen(
                [sys.executable, "slow_main_server.py", str(port)],
                cwd=tmp_dir,
                start_new_session=True,
            )
            try:
                deadline = time.monotonic() + 90
                body = None
                while time.monotonic() < deadline and proc.poll() is None:
                    try:
                        with urllib.request.urlopen(
                            f"http://127.0.0.1:{port}/", timeout=2
                        ) as r:
                            body = r.read()
                        break
                    except OSError:
                        time.sleep(0.5)
                self.assertEqual(body, b"ok")
            finally:
                # The workers share the supervisor's session: this also kills
                # the 600 s re-imports if the assertion failed.
                os.killpg(proc.pid, signal.SIGKILL)
                proc.wait()


if __name__ == "__main__":
    unittest.main()
