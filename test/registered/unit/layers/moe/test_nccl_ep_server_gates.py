"""Failure handling in the paid-hardware validation driver; no GPU required."""

import json
import os
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from nccl_ep_test.followup_server import check_port_available, command, prerequisite

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


def test_serving_probe_accepts_time_wait_but_rejects_live_listener():
    with socket.socket() as listener:
        listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
        listener.listen()
        with pytest.raises(OSError):
            check_port_available(port)
        with socket.create_connection(("127.0.0.1", port)) as client:
            connection, _ = listener.accept()
            connection.close()  # Active close leaves the server port in TIME_WAIT.
            assert client.recv(1) == b""
    check_port_available(port)


@pytest.mark.parametrize("passed,head", [(False, "current"), (True, "old")])
def test_failed_or_stale_gate_cannot_launch_next_phase(
    tmp_path, monkeypatch, passed, head
):
    monkeypatch.setattr("nccl_ep_test.followup_server.source_head", lambda: "current")
    (tmp_path / "single.json").write_text(
        json.dumps({"passed": passed, "source_head": head})
    )
    with pytest.raises(RuntimeError, match="Required successful gate"):
        prerequisite(tmp_path, "single.json")


def test_timeout_kills_child_worker_group(tmp_path):
    worker_pid = tmp_path / "worker.pid"
    script = (
        "import os, pathlib, signal, time\n"
        "child = os.fork()\n"
        "if child == 0:\n"
        "    signal.signal(signal.SIGTERM, signal.SIG_IGN)\n"
        f"    pathlib.Path({str(worker_pid)!r}).write_text(str(os.getpid()))\n"
        "    time.sleep(120)\n"
        "else:\n"
        "    time.sleep(120)\n"
    )
    try:
        with pytest.raises(subprocess.TimeoutExpired):
            command([sys.executable, "-c", script], tmp_path / "worker.log", timeout=1)
        pid = int(worker_pid.read_text())
        # A killed orphan may briefly remain a zombie until PID 1 reaps it.
        status = Path(f"/proc/{pid}/status")
        deadline = time.monotonic() + 2
        while True:
            try:
                state = next(
                    line
                    for line in status.read_text().splitlines()
                    if line.startswith("State:")
                )
            except FileNotFoundError:
                break
            if "Z" in state or "X" in state:
                break
            assert time.monotonic() < deadline, "Timed-out worker is still running"
            time.sleep(0.01)
    finally:
        if worker_pid.exists():
            try:
                os.kill(int(worker_pid.read_text()), signal.SIGKILL)
            except ProcessLookupError:
                pass


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
