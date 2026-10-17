"""Manual TP1-per-allocation NIXL fixed-slot recovery test.

Run with two visible GPUs:

    CUDA_VISIBLE_DEVICES=0,1 python -m pytest \
        test/manual/ep/test_nixl_fixed_slot_recovery.py -v -s
"""

import os
import subprocess
import time
import unittest
from pathlib import Path
from urllib.parse import quote

import requests
import torch.distributed as dist

from sglang.test.test_utils import (
    DEFAULT_SMALL_MOE_MODEL_NAME_FOR_TEST_CHAT,
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    CustomTestCase,
    terminate_and_kill_process_tree,
)
from sglang.utils import wait_for_http_ready

MODEL = os.environ.get(
    "SGLANG_NIXL_RECOVERY_MODEL", DEFAULT_SMALL_MOE_MODEL_NAME_FOR_TEST_CHAT
)
DIST_INIT_ADDR = os.environ.get("SGLANG_NIXL_RECOVERY_DIST_INIT", "127.0.0.1:26555")
PRIMARY_PORT = int(os.environ.get("SGLANG_NIXL_RECOVERY_PRIMARY_PORT", "21000"))
REPLACEMENT_PORT = int(os.environ.get("SGLANG_NIXL_RECOVERY_REPLACEMENT_PORT", "22000"))
RECOVERY_TIMEOUT = float(os.environ.get("SGLANG_NIXL_RECOVERY_TIMEOUT", "600"))
INITIAL_ALLOCATION_IDS = ("initial-rank-0", "initial-rank-1")
REPLACEMENT_ALLOCATION_ID = "replacement-rank-1"


def _visible_devices() -> list[str]:
    configured = os.environ.get("CUDA_VISIBLE_DEVICES")
    if configured:
        return [device.strip() for device in configured.split(",") if device.strip()]
    try:
        import torch

        return [str(index) for index in range(torch.cuda.device_count())]
    except Exception:
        return []


def _server_args(
    allocation_index: int,
    allocation_id: str,
    port: int,
    *,
    recover: bool,
) -> list[str]:
    args = [
        "sglang",
        "serve",
        "--model-path",
        MODEL,
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--device",
        "cuda",
        "--trust-remote-code",
        "--dist-init-addr",
        DIST_INIT_ADDR,
        "--enable-dp-lm-head",
        "--elastic-ep-backend",
        "mooncake",
        "--moe-a2a-backend",
        "nixl",
        "--deepep-mode",
        "low_latency",
        "--moe-dense-tp-size",
        "1",
        "--disable-custom-all-reduce",
        "--enable-eplb",
        "--ep-num-redundant-experts",
        "2",
        "--elastic-ep-initial-size",
        "2",
        "--max-ep-size",
        "3",
        "--elastic-ep-allocation-id",
        allocation_id,
        "--cuda-graph-backend-prefill",
        "disabled",
        "--cuda-graph-backend-decode",
        "full",
        "--cuda-graph-max-bs-decode",
        "16",
        "--mem-fraction-static",
        "0.7",
    ]
    if recover:
        args.extend(
            [
                "--tp",
                "2",
                "--nnodes",
                "2",
                "--node-rank",
                str(allocation_index),
                "--attn-dp-size",
                "2",
                "--elastic-ep-join-mode",
                "recover",
            ]
        )
    else:
        args.extend(
            [
                "--elastic-ep-join-mode",
                "auto",
                "--elastic-ep-allocation-index",
                str(allocation_index),
            ]
        )
    return args


@unittest.skipUnless(
    len(_visible_devices()) >= 2,
    "NIXL fixed-slot recovery needs two visible GPUs.",
)
class TestNixlFixedSlotRecovery(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        cls.base_url = f"http://127.0.0.1:{PRIMARY_PORT}"
        cls.processes: list[subprocess.Popen] = []
        cls.logs = []
        devices = _visible_devices()
        cls.primary = cls._launch(
            allocation_index=0,
            allocation_id=INITIAL_ALLOCATION_IDS[0],
            port=PRIMARY_PORT,
            device=devices[0],
            name="primary",
        )
        cls.failed_rank = cls._launch(
            allocation_index=1,
            allocation_id=INITIAL_ALLOCATION_IDS[1],
            port=REPLACEMENT_PORT,
            device=devices[1],
            name="initial-rank-1",
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
        allocation_index: int,
        allocation_id: str,
        port: int,
        device: str,
        name: str,
        recover: bool = False,
    ) -> subprocess.Popen:
        log_path = Path(f"/tmp/nixl_fixed_slot_{name}_{int(time.time())}.log")
        log_file = open(log_path, "w")
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = device
        env["SGLANG_NIXL_EP_BF16_DISPATCH"] = "1"
        process = subprocess.Popen(
            _server_args(allocation_index, allocation_id, port, recover=recover),
            env=env,
            stdout=log_file,
            stderr=subprocess.STDOUT,
        )
        cls.processes.append(process)
        cls.logs.append(log_file)
        print(f"Started {name}; log: {log_path}")
        return process

    @classmethod
    def tearDownClass(cls):
        for process in reversed(getattr(cls, "processes", [])):
            if process.poll() is None:
                terminate_and_kill_process_tree(process)
        for log_file in getattr(cls, "logs", []):
            log_file.close()

    def _generate(self, routed_dp_rank: int = 0) -> requests.Response:
        return requests.post(
            f"{self.base_url}/generate",
            json={
                "text": "The capital of France is",
                "sampling_params": {"max_new_tokens": 4, "temperature": 0.0},
                "routed_dp_rank": routed_dp_rank,
            },
            timeout=120,
        )

    def _state(self) -> dict:
        response = requests.get(f"{self.base_url}/is_scaling_elastic_ep", timeout=30)
        self.assertEqual(response.status_code, 200, response.text)
        return response.json()

    def _publish_stale_endpoint(self, runtime_instance_id: str) -> None:
        host, port = DIST_INIT_ADDR.rsplit(":", 1)
        store = dist.TCPStore(
            host,
            int(port),
            world_size=None,
            is_master=False,
            wait_for_workers=False,
        )
        prefix = (
            "elastic_ep/nixl_recovery/"
            f"{quote(runtime_instance_id, safe='')}/0/"
            f"{quote('stale-operation', safe='')}/"
            f"{quote(INITIAL_ALLOCATION_IDS[1], safe='')}/"
        )
        dist.PrefixStore(prefix, store).set("NIXL_EP/1", b"stale-endpoint")

    def test_same_rank_replacement(self):
        response = self._generate()
        self.assertEqual(response.status_code, 200, response.text)

        terminate_and_kill_process_tree(self.failed_rank)
        response = self._generate()
        self.assertEqual(response.status_code, 200, response.text)

        operation_id = f"recover-rank-1-{int(time.time())}"
        response = requests.post(
            f"{self.base_url}/recover_elastic_ep",
            json={
                "operation_id": operation_id,
                "topology_generation": 0,
                "allocation_id": REPLACEMENT_ALLOCATION_ID,
                "rank_offset": 1,
            },
            timeout=30,
        )
        self.assertEqual(response.status_code, 200, response.text)

        self._publish_stale_endpoint(self._state()["instance_id"])

        self.replacement = self._launch(
            allocation_index=1,
            allocation_id=REPLACEMENT_ALLOCATION_ID,
            port=REPLACEMENT_PORT,
            device=_visible_devices()[1],
            name="replacement",
            recover=True,
        )
        deadline = time.monotonic() + RECOVERY_TIMEOUT
        while time.monotonic() < deadline:
            self.assertIsNone(
                self.replacement.poll(), "Replacement exited before readiness."
            )
            state = self._state()
            if state["recovery_phase"] == "ready":
                self.assertTrue(state["recovery_succeeded"])
                response = self._generate(routed_dp_rank=1)
                self.assertEqual(response.status_code, 200, response.text)
                return
            self.assertNotEqual(state["recovery_phase"], "failed", state)
            time.sleep(2)
        self.fail(f"Timed out waiting for recovery readiness: {self._state()}")


if __name__ == "__main__":
    unittest.main()
