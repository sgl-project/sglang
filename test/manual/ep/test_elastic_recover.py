"""Manual single-host Elastic EP recovery test.

Run:

    CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m pytest \
        test/manual/ep/test_elastic_recover.py -v -s
"""

import os
import shlex
import subprocess
import time
import unittest
from pathlib import Path

import requests

from sglang.srt.utils import kill_process_tree
from sglang.test.server_fixtures.disaggregation_fixture import get_rdma_devices_args
from sglang.test.test_utils import (
    DEFAULT_MODEL_NAME_FOR_TEST_MLA,
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    CustomTestCase,
    try_cached_model,
)
from sglang.utils import wait_for_http_ready


TEST_MODEL = os.environ.get(
    "SGLANG_ELASTIC_RECOVER_TEST_MODEL",
    try_cached_model(DEFAULT_MODEL_NAME_FOR_TEST_MLA),
)
EP_SIZE = 8
LOCAL_EP_SIZE = 4
DIST_INIT_ADDR = os.environ.get("SGLANG_ELASTIC_RECOVER_DIST_INIT", "127.0.0.1:25555")
PRIMARY_PORT = int(os.environ.get("SGLANG_ELASTIC_RECOVER_PRIMARY_PORT", "21000"))
JOINER_PORT = int(os.environ.get("SGLANG_ELASTIC_RECOVER_JOINER_PORT", "22000"))
RECOVER_WAIT_SECONDS = float(
    os.environ.get("SGLANG_ELASTIC_RECOVER_WAIT_SECONDS", "5")
)
RECOVER_TIMEOUT_SECONDS = float(
    os.environ.get("SGLANG_ELASTIC_RECOVER_TIMEOUT_SECONDS", "300")
)
RANDOM_SEED = int(os.environ.get("SGLANG_ELASTIC_RECOVER_RANDOM_SEED", "42"))
ib_devices = get_rdma_devices_args()


def _visible_device_ids() -> list[str]:
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible:
        return [device.strip() for device in visible.split(",") if device.strip()]
    try:
        import torch

        return [str(index) for index in range(torch.cuda.device_count())]
    except Exception:
        return []


def _server_args(node_rank: int, port: int, recover: bool = False) -> list[str]:
    args = [
        "sglang",
        "serve",
        "--model-path",
        TEST_MODEL,
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--device",
        "cuda",
        "--trust-remote-code",
        "--tp",
        str(EP_SIZE),
        "--dp",
        str(EP_SIZE),
        "--nnodes",
        "2",
        "--node-rank",
        str(node_rank),
        "--dist-init-addr",
        DIST_INIT_ADDR,
        "--random-seed",
        str(RANDOM_SEED),
        "--enable-dp-attention",
        "--enable-dp-lm-head",
        "--elastic-ep-backend",
        "mooncake",
        "--mooncake-ib-device",
        ib_devices,
        "--moe-a2a-backend",
        "mooncake",
        "--deepep-mode",
        "low_latency",
        "--moe-dense-tp-size",
        "1",
        "--disable-custom-all-reduce",
        "--enable-eplb",
        "--ep-num-redundant-experts",
        "72",
        "--chunked-prefill-size",
        "512",
        "--cuda-graph-max-bs-decode",
        "16",
        "--mem-fraction-static",
        "0.5",
    ]
    if recover:
        args.extend(["--elastic-ep-join-mode", "recover"])
    extra_args = os.environ.get("SGLANG_ELASTIC_RECOVER_EXTRA_SERVER_ARGS", "")
    return args + shlex.split(extra_args)


@unittest.skipUnless(
    len(_visible_device_ids()) >= EP_SIZE,
    "Elastic EP recovery E2E needs 8 visible GPUs.",
)
class TestElasticRecover4To4(CustomTestCase):
    """Kill one four-rank node and recover it with a fresh process group."""

    @classmethod
    def setUpClass(cls):
        cls.base_url = f"http://127.0.0.1:{PRIMARY_PORT}"
        cls.processes: list[subprocess.Popen] = []
        cls.log_files = []
        cls.log_paths: dict[str, Path] = {}
        visible_devices = _visible_device_ids()

        cls.primary = cls._launch(
            node_rank=0,
            port=PRIMARY_PORT,
            visible_devices=visible_devices[:LOCAL_EP_SIZE],
            name="primary",
        )
        cls.initial_joiner = cls._launch(
            node_rank=1,
            port=JOINER_PORT,
            visible_devices=visible_devices[LOCAL_EP_SIZE:EP_SIZE],
            name="initial_joiner",
        )
        wait_for_http_ready(
            f"{cls.base_url}/health_generate",
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            process=cls.primary,
        )

    @classmethod
    def _launch(
        cls,
        *,
        node_rank: int,
        port: int,
        visible_devices: list[str],
        name: str,
        recover: bool = False,
    ) -> subprocess.Popen:
        log_path = Path(f"/tmp/elastic_ep_recover_{name}_{int(time.time())}.log")
        log_file = open(log_path, "w")
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = ",".join(visible_devices)
        process = subprocess.Popen(
            _server_args(node_rank, port, recover),
            env=env,
            stdout=log_file,
            stderr=subprocess.STDOUT,
        )
        cls.processes.append(process)
        cls.log_files.append(log_file)
        cls.log_paths[name] = log_path
        print(f"Started {name}; log: {log_path}")
        return process

    @classmethod
    def tearDownClass(cls):
        for process in reversed(getattr(cls, "processes", [])):
            if process.poll() is None:
                kill_process_tree(process.pid, wait_timeout=60)
        for log_file in getattr(cls, "log_files", []):
            log_file.close()

    def _generate(self, routed_dp_rank: int | None = None) -> requests.Response:
        payload = {
            "text": "The capital of France is",
            "sampling_params": {"max_new_tokens": 4, "temperature": 0.0},
        }
        if routed_dp_rank is not None:
            payload["routed_dp_rank"] = routed_dp_rank
        return requests.post(f"{self.base_url}/generate", json=payload, timeout=90)

    def _generate_ok(
        self, description: str, routed_dp_rank: int | None = None
    ) -> None:
        response = self._generate(routed_dp_rank)
        self.assertEqual(response.status_code, 200, f"{description}: {response.text}")

    def _wait_for_recover_capture(self) -> None:
        deadline = time.monotonic() + RECOVER_TIMEOUT_SECONDS
        log_path = self.log_paths["recover_joiner"]
        marker = "Capture target decode CUDA graph end"
        while time.monotonic() < deadline:
            self.assertIsNone(
                self.recover_joiner.poll(),
                "Recover joiner exited during CUDA graph capture",
            )
            if log_path.exists() and log_path.read_text(errors="replace").count(marker) >= (
                EP_SIZE - LOCAL_EP_SIZE
            ):
                return
            time.sleep(2)
        self.fail(f"Timed out waiting for recover CUDA graph capture: {log_path}")

    def _wait_for_recovered_ranks(self) -> None:
        self._wait_for_recover_capture()
        self._generate_ok("recovery trigger")
        deadline = time.monotonic() + RECOVER_TIMEOUT_SECONDS
        marker = f"recover ranks {list(range(LOCAL_EP_SIZE, EP_SIZE))} done"
        primary_log = self.log_paths["primary"]
        while time.monotonic() < deadline:
            self.assertIsNone(
                self.recover_joiner.poll(), "Recover joiner exited before rejoining"
            )
            if primary_log.exists() and primary_log.read_text(errors="replace").count(
                marker
            ) >= LOCAL_EP_SIZE:
                for request_index in range(3):
                    self._generate_ok(f"post-recovery request {request_index + 1}")
                return
            time.sleep(2)
        self.fail(f"Timed out waiting for recovery collective: {primary_log}")

    def test_recover_four_ranks(self):
        self._generate_ok("initial service")

        kill_process_tree(self.initial_joiner.pid, wait_timeout=60)
        # Give the terminated schedulers time to disappear before fault handling.
        time.sleep(RECOVER_WAIT_SECONDS)
        self._generate_ok("degraded service after node1 failure")

        visible_devices = _visible_device_ids()
        self.recover_joiner = self._launch(
            node_rank=1,
            port=JOINER_PORT,
            visible_devices=visible_devices[LOCAL_EP_SIZE:EP_SIZE],
            name="recover_joiner",
            recover=True,
        )
        self._wait_for_recovered_ranks()


if __name__ == "__main__":
    unittest.main()
