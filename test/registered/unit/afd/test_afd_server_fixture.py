"""Real CPU subprocess tests for A/F fixture ownership, not model validation."""

import dataclasses
import json
import socket
import subprocess
import sys
import tempfile
import textwrap
import time
import unittest
from pathlib import Path
from unittest.mock import patch

import psutil

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.server_fixtures.afd_fixture import (
    AFDProcessSpec,
    AFDServerGroup,
    _OwnedProcess,
)
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


class TestAFDServerFixture(CustomTestCase):
    def setUp(self):
        super().setUp()
        self.tmp = tempfile.TemporaryDirectory(prefix="afd-fixture-test-")
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            self.port = sock.getsockname()[1]
        self.url = f"http://127.0.0.1:{self.port}"

    def spec(self, name, role, code, lanes=()):
        return AFDProcessSpec(
            name,
            role,
            [sys.executable, "-u", "-c", textwrap.dedent(code)],
            ffn_lanes=lanes,
        )

    def attention(self, *, wait_for_ffn=False, extra=""):
        wait = (
            f"while not Path({str(self.root / 'f-started')!r}).exists(): time.sleep(.01)"
            if wait_for_ffn
            else ""
        )
        return self.spec(
            "a",
            "attention",
            f"""
import http.server, time
from pathlib import Path
from unittest.mock import patch
Path({str(self.root / "a-started")!r}).touch()
{extra}
{wait}
class Handler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200 if self.path == '/health_generate' else 404)
        self.end_headers()
    def log_message(self, *args): pass
http.server.HTTPServer(('127.0.0.1', {self.port}), Handler).serve_forever()
""",
        )

    def ffn(self, code=None, lanes=(0,)):
        return self.spec(
            "f",
            "ffn",
            code
            or """
import time
print('AFD FFN lane 0 ready', flush=True)
time.sleep(30)
""",
            lanes,
        )

    def group(self, attention=None, ffn=None, timeout=3):
        group = AFDServerGroup(
            [attention or self.attention(), ffn or self.ffn()],
            base_url=self.url,
            log_dir=self.root / "logs",
            startup_timeout=timeout,
            shutdown_timeout=0.15,
            poll_interval=0.01,
        )
        self.addCleanup(group.close)
        return group

    def assert_stopped(self, group):
        self.assertTrue(group.processes)
        self.assertTrue(all(p.poll() is not None for p in group.processes.values()))

    def test_starts_both_roles_before_waiting_for_http_and_requires_all_lanes(self):
        ffn = self.ffn(
            f"""
import time
from pathlib import Path
from unittest.mock import patch
while not Path({str(self.root / "a-started")!r}).exists(): time.sleep(.01)
Path({str(self.root / "f-started")!r}).touch()
print('AFD FFN lane 0 ready', flush=True)
print('AFD FFN lane 1 ready', flush=True)
time.sleep(30)
""",
            lanes=(0, 1),
        )
        group = self.group(self.attention(wait_for_ffn=True), ffn)
        with group:
            self.assertEqual(group.ready_lanes, {"f": {0, 1}})
            group.check_healthy()
        self.assert_stopped(group)
        group.close()  # idempotent
        with self.assertRaisesRegex(RuntimeError, "single-use"):
            group.start()

    def test_old_ready_log_and_partial_lane_readiness_cannot_pass(self):
        group = self.group(ffn=self.ffn(lanes=(0, 1)), timeout=0.4)
        group.log_dir.mkdir()
        group.log_paths["f"].write_text("AFD FFN lane 1 ready\n")
        started = time.monotonic()
        with self.assertRaisesRegex(TimeoutError, "readiness timed out"):
            group.start()
        self.assertLess(time.monotonic() - started, 3)
        self.assertEqual(group.ready_lanes["f"], {0})
        self.assertNotIn("lane 1 ready", group.log_paths["f"].read_text())
        self.assert_stopped(group)

    def test_early_zero_and_nonzero_exit_fail_without_single_side_retry(self):
        for code in (0, 7):
            with self.subTest(code=code):
                count = self.root / f"attempt-{code}"
                ffn = self.ffn(f"""
from pathlib import Path
from unittest.mock import patch
import sys
with Path({str(count)!r}).open('a') as f: f.write('attempt\\n')
sys.exit({code})
""")
                group = self.group(ffn=ffn)
                with self.assertRaisesRegex(
                    RuntimeError, f"f exited unexpectedly: rc={code}"
                ):
                    group.start()
                self.assertEqual(count.read_text().splitlines(), ["attempt"])
                self.assert_stopped(group)

    def test_post_ready_failure_stops_the_peer_and_is_reported(self):
        trigger = self.root / "fail-f"
        ffn = self.ffn(f"""
from pathlib import Path
from unittest.mock import patch
import sys, time
print('AFD FFN lane 0 ready', flush=True)
while not Path({str(trigger)!r}).exists(): time.sleep(.01)
sys.exit(9)
""")
        group = self.group(ffn=ffn).start()
        trigger.touch()
        deadline = time.monotonic() + 3
        while time.monotonic() < deadline and any(
            p.poll() is None for p in group.processes.values()
        ):
            time.sleep(0.01)
        self.assert_stopped(group)
        with self.assertRaisesRegex(RuntimeError, "f exited unexpectedly: rc=9"):
            group.check_healthy()
        with self.assertRaisesRegex(RuntimeError, "f exited unexpectedly: rc=9"):
            group.__exit__(None, None, None)

    def test_spawn_error_cleans_already_launched_role(self):
        group = self.group(
            ffn=AFDProcessSpec(
                "f", "ffn", ["/no-such-afd-test-program"], ffn_lanes=(0,)
            )
        )
        with self.assertRaises(FileNotFoundError):
            group.start()
        self.assert_stopped(group)

    def test_close_kills_term_ignoring_descendant_after_root_exit(self):
        pidfile = self.root / "child-pid"
        child_code = "import signal,time; signal.signal(signal.SIGTERM, signal.SIG_IGN); time.sleep(30)"
        extra = f"""import subprocess, sys
child = subprocess.Popen([sys.executable, '-c', {child_code!r}])
Path({str(pidfile)!r}).write_text(str(child.pid))"""
        group = self.group(attention=self.attention(extra=extra)).start()
        child = psutil.Process(int(pidfile.read_text()))
        group.close()
        self.assertTrue(group.forced_kill_pids)
        deadline = time.monotonic() + 2
        while (
            time.monotonic() < deadline
            and child.is_running()
            and child.status() != psutil.STATUS_ZOMBIE
        ):
            time.sleep(0.01)
        self.assertTrue(
            not child.is_running() or child.status() == psutil.STATUS_ZOMBIE
        )
        self.assert_stopped(group)

    def test_close_allows_attention_notification_and_ffn_ack_before_kill(self):
        close_marker = self.root / "close"
        ack_marker = self.root / "ack"
        extra = f"""import signal
def close_attention(*args):
    Path({str(close_marker)!r}).touch()
    raise SystemExit(0)
signal.signal(signal.SIGTERM, close_attention)"""
        ffn = self.ffn(f"""
from pathlib import Path
from unittest.mock import patch
import time
print('AFD FFN lane 0 ready', flush=True)
while not Path({str(close_marker)!r}).exists(): time.sleep(.01)
Path({str(ack_marker)!r}).touch()
""")
        group = self.group(attention=self.attention(extra=extra), ffn=ffn).start()
        group.close()
        self.assertTrue(ack_marker.exists())
        self.assertTrue(all(p.returncode == 0 for p in group.processes.values()))
        self.assertFalse(group.forced_kill_pids)

    def test_slow_body_is_closed_without_delaying_http_readiness(self):
        attention = self.spec(
            "a",
            "attention",
            f"""
import http.server, time
class Handler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.send_header('Content-Length', '1000')
        self.end_headers()
        for _ in range(1000):
            self.wfile.write(b'x')
            self.wfile.flush()
            time.sleep(.02)
    def log_message(self, *args): pass
http.server.HTTPServer(('127.0.0.1', {self.port}), Handler).serve_forever()
""",
        )
        group = self.group(attention=attention, timeout=1)
        before = time.monotonic()
        group.start()
        self.assertLess(time.monotonic() - before, 1)
        group.close()
        self.assert_stopped(group)

    def test_dripping_headers_obey_absolute_startup_deadline(self):
        probe_marker = self.root / "header-probe-started"
        attention = self.spec(
            "a",
            "attention",
            f"""
import http.server, time
from pathlib import Path
class Handler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        Path({str(probe_marker)!r}).touch()
        self.wfile.write(b'HTTP/1.1 200 OK\\r\\nX-Slow: ')
        self.wfile.flush()
        for _ in range(1000):
            self.wfile.write(b'x')
            self.wfile.flush()
            time.sleep(.02)
    def log_message(self, *args): pass
http.server.HTTPServer(('127.0.0.1', {self.port}), Handler).serve_forever()
""",
        )
        group = self.group(attention=attention, timeout=0.4)
        before = time.monotonic()
        with self.assertRaisesRegex(TimeoutError, "readiness timed out"):
            group.start()
        self.assertLess(time.monotonic() - before, 1.5)
        self.assertTrue(probe_marker.exists())
        self.assert_stopped(group)

    def test_reused_pid_identity_cannot_authorize_process_or_group_kill(self):
        outsider = subprocess.Popen(
            [sys.executable, "-c", "import time; time.sleep(30)"],
            start_new_session=True,
        )
        self.addCleanup(lambda: outsider.wait(timeout=2))
        self.addCleanup(outsider.kill)
        original = _OwnedProcess.capture(outsider.pid)
        stale = dataclasses.replace(original, created=original.created - 10)
        self.assertFalse(stale.is_live())
        group = self.group()
        # Simulate an old captured identity whose PID now belongs to outsider.
        group._roots["a"] = stale
        group._owned[(stale.process.pid, stale.created)] = stale
        with patch("sglang.test.server_fixtures.afd_fixture.os.killpg") as group_kill:
            group.close()
        group_kill.assert_not_called()
        self.assertIsNone(outsider.poll())

    def test_pid_reuse_during_tree_discovery_cannot_adopt_an_outsider(self):
        outsider = subprocess.Popen(
            [sys.executable, "-c", "import time; time.sleep(30)"],
            start_new_session=True,
        )
        self.addCleanup(lambda: outsider.wait(timeout=2))
        self.addCleanup(outsider.kill)
        group = self.group().start()
        # The tree scan saw an old child, but by capture time that PID denotes
        # an unrelated live process. Root liveness alone cannot authorize it.
        with patch(
            "sglang.test.server_fixtures.afd_fixture.collect_process_tree_pids",
            return_value=[outsider.pid],
        ):
            group._remember_children()
        self.assertNotIn(outsider.pid, {pid for pid, _ in group._owned})
        group.close()
        self.assertIsNone(outsider.poll())
        self.assert_stopped(group)

    def test_cleanup_error_does_not_abandon_other_owned_processes(self):
        group = self.group().start()
        with patch.object(
            group._roots["a"].process,
            "terminate",
            side_effect=PermissionError("injected TERM error"),
        ):
            with self.assertRaisesRegex(RuntimeError, "injected TERM error"):
                group.close()
        self.assert_stopped(group)
        self.assertTrue(all(log.closed for log in group._logs))

    def test_local_server_factory_refuses_multihost_overrides(self):
        for args in (["--nnodes", "2"], ["--nnodes=2"]):
            with (
                self.subTest(args=args),
                self.assertRaisesRegex(ValueError, "local role"),
            ):
                AFDProcessSpec.server(
                    role="ffn", model="/model", afd_config={"lanes": 2}, other_args=args
                )

    def test_real_server_command_uses_shared_config_and_explicit_roles(self):
        config = {"lanes": 2, "attention_lanes": 4}
        for role in ("attention", "ffn"):
            with self.subTest(role=role):
                spec = AFDProcessSpec.server(
                    role=role,
                    model="/public/model/path",
                    afd_config=config,
                    base_url=self.url,
                    other_args=["--tp-size", "4" if role == "attention" else "2"],
                )
                self.assertEqual(
                    spec.command[:3], [sys.executable, "-m", "sglang.launch_server"]
                )
                self.assertEqual(
                    spec.command[spec.command.index("--afd-execution-mode") + 1], role
                )
                self.assertEqual(
                    json.loads(spec.command[spec.command.index("--afd-config") + 1]),
                    config,
                )
                self.assertEqual(spec.ffn_lanes, (0, 1) if role == "ffn" else ())


if __name__ == "__main__":
    unittest.main()
