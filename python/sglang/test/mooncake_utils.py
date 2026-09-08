import os
import signal
import socket
import subprocess
import time

import requests

from sglang.test.server_fixtures.disaggregation_fixture import (
    get_rdma_devices_args,
)
from sglang.test.test_utils import find_available_port


class MooncakeTestServices:
    """Lifecycle for a local Mooncake master and external storage client."""

    def __init__(
        self,
        *,
        protocol: str = "rdma",
        store_segment_size: int = 4 * 1024**3,
        device: str | None = None,
        local_hostname: str | None = None,
    ):
        self.protocol = protocol
        self.store_segment_size = store_segment_size
        self.device = device if device is not None else self._default_device(protocol)
        self.local_hostname = local_hostname or socket.gethostbyname(
            socket.gethostname()
        )

        self.master_port = find_available_port(50051)
        self.master_metrics_port = find_available_port(9003)
        self.metadata_port = find_available_port(8080)
        self.store_port = find_available_port(50052)
        self.store_http_port = find_available_port(8081)

        self.metadata_process = None
        self.master_process = None
        self.store_process = None

    @staticmethod
    def _default_device(protocol: str) -> str:
        configured = os.environ.get("SGLANG_TEST_MOONCAKE_DEVICE")
        if configured is not None:
            return configured
        if protocol == "rdma":
            return get_rdma_devices_args().split(",")[0]
        return ""

    def start(self):
        self.metadata_process = self._launch(
            [
                "python3",
                "-m",
                "mooncake.http_metadata_server",
                "--port",
                str(self.metadata_port),
            ]
        )
        try:
            self.master_process = self._launch(
                [
                    "mooncake_master",
                    "--port",
                    str(self.master_port),
                    "--metrics_port",
                    str(self.master_metrics_port),
                ]
            )
            self._wait_for_core_services()
            self.store_process = self._launch(
                [
                    "mooncake_client",
                    f"--host={self.local_hostname}",
                    f"--port={self.store_port}",
                    f"--master_server_address=127.0.0.1:{self.master_port}",
                    f"--metadata_server=http://127.0.0.1:{self.metadata_port}/metadata",
                    f"--protocol={self.protocol}",
                    f"--device_names={self.device}",
                    f"--global_segment_size={self.store_segment_size}",
                    "--enable_http_server=true",
                    f"--http_port={self.store_http_port}",
                ],
                env={**os.environ, "MC_MS_AUTO_DISC": "0"},
            )
            self._wait_for_store()
        except Exception:
            self.stop()
            raise

    def stop(self):
        for name in ("store_process", "master_process", "metadata_process"):
            process = getattr(self, name)
            if process is None:
                continue
            self._stop_process_group(process)
            setattr(self, name, None)

    def server_env(self) -> dict[str, str]:
        return {
            "MOONCAKE_MASTER": f"127.0.0.1:{self.master_port}",
            "MOONCAKE_PROTOCOL": self.protocol,
            "MC_MS_AUTO_DISC": "0",
            "MOONCAKE_DEVICE": self.device,
            "MOONCAKE_LOCAL_HOSTNAME": self.local_hostname,
            "MOONCAKE_TE_META_DATA_SERVER": (
                f"http://127.0.0.1:{self.metadata_port}/metadata"
            ),
            "MOONCAKE_GLOBAL_SEGMENT_SIZE": "0",
        }

    def master_metric(self, name: str) -> float:
        response = requests.get(
            f"http://127.0.0.1:{self.master_metrics_port}/metrics",
            timeout=5,
        )
        response.raise_for_status()
        prefix = f"{name} "
        for line in response.text.splitlines():
            if line.startswith(prefix):
                return float(line.split()[1])
        raise AssertionError(f"{name} is missing from Mooncake master metrics")

    @staticmethod
    def _launch(command, env=None):
        return subprocess.Popen(
            command,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.STDOUT,
            preexec_fn=os.setsid,
            env=env,
        )

    def _wait_for_core_services(self, timeout: int = 30):
        deadline = time.monotonic() + timeout
        master_ready_at = time.monotonic() + 3
        while time.monotonic() < deadline:
            self._raise_if_exited(self.metadata_process, "metadata service")
            self._raise_if_exited(self.master_process, "master service")
            try:
                requests.get(
                    f"http://127.0.0.1:{self.metadata_port}/metadata",
                    timeout=2,
                )
                if time.monotonic() >= master_ready_at:
                    return
            except requests.RequestException:
                pass
            time.sleep(1)
        raise TimeoutError("Timed out waiting for Mooncake metadata and master")

    def _wait_for_store(self, timeout: int = 90):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            self._raise_if_exited(self.store_process, "store service")
            try:
                with socket.create_connection(
                    (self.local_hostname, self.store_port), timeout=2
                ):
                    return
            except OSError:
                time.sleep(1)
        raise TimeoutError("Timed out waiting for Mooncake store")

    @staticmethod
    def _raise_if_exited(process, name: str):
        returncode = process.poll()
        if returncode is not None:
            raise RuntimeError(f"Mooncake {name} exited with code {returncode}")

    @staticmethod
    def _stop_process_group(process):
        try:
            os.killpg(os.getpgid(process.pid), signal.SIGTERM)
            process.wait(timeout=10)
        except ProcessLookupError:
            return
        except (subprocess.TimeoutExpired, OSError):
            try:
                os.killpg(os.getpgid(process.pid), signal.SIGKILL)
                process.wait(timeout=5)
            except (ProcessLookupError, subprocess.TimeoutExpired, OSError):
                pass
