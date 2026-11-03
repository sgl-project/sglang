"""uvicorn workers spawned by the multi-tokenizer HTTP server must not re-run
sglang's launcher module as ``__main__``: a slow main module would delay each
worker's health-check thread past the supervisor's timeout, and the
supervisor would kill and respawn it in a loop. A user's own launcher script
still runs in every worker.
"""

import os
import signal
import socket
import subprocess
import sys
import tempfile
import textwrap
import time
import unittest
import urllib.request
from multiprocessing import spawn
from unittest import mock

from sglang.srt.entrypoints import uvicorn_worker_spawn
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
                timeout_worker_healthcheck=5,
                lifespan="off",
                log_level="warning",
            )
    """
)


def _free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


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
                deadline = time.monotonic() + 60
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
                os.killpg(proc.pid, signal.SIGKILL)
                proc.wait()


if __name__ == "__main__":
    unittest.main()
