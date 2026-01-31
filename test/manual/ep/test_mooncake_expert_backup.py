import time
import unittest
from types import SimpleNamespace

import requests

from sglang.srt.utils import kill_process_tree
from sglang.test.few_shot_gsm8k import run_eval as run_eval_few_shot_gsm8k
from sglang.test.server_fixtures.disaggregation_fixture import get_rdma_devices_args
from sglang.test.test_utils import (
    DEFAULT_MODEL_NAME_FOR_TEST_MLA,
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    CustomTestCase,
    popen_launch_pd_server,
)

ib_devices = get_rdma_devices_args()


class TestBackup(CustomTestCase):
    extra_args = []

    @classmethod
    def setUpClass(cls):
        cls.model = DEFAULT_MODEL_NAME_FOR_TEST_MLA
        cls.base_port = 20000
        cls.num_processes = 2
        # TODO (stage 100): in the future, implement a specified multiprocess launcher
        cls.processes = [
            popen_launch_pd_server(
                cls.model,
                f"http://127.0.0.1:{cls.base_port + i}",
                timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
                other_args=[
                    "--trust-remote-code",
                    "--tp",
                    "4",
                    "--enable-dp-attention",
                    "--dp",
                    "4",
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
                    "--enable-dp-lm-head",
                    "--enable-two-batch-overlap",
                    "--disable-custom-all-reduce",
                    "--enable-elastic-expert-backup",
                    "--enable-eplb",
                    "--eplb-rebalance-num-iterations",
                    "50",
                    "--chunked-prefill-size",
                    "512",
                    "--cuda-graph-max-bs",
                    "128",
                    "--max-running-requests",
                    "512",
                    "--mem-fraction-static",
                    "0.5",
                    "--dist-init-addr",
                    "127.0.0.1:5000",
                    "--nnodes",
                    f"{cls.num_processes}",
                    "--node-rank",
                    f"{i}",
                    "--base-gpu-id",
                    f"{i * 2}",
                ],
            )
            for i in range(cls.num_processes)
        ]

        server_ready = [False] * cls.num_processes
        start_time = time.perf_counter()
        with requests.Session() as session:
            while (
                time.perf_counter() - start_time < DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH
                and not all(server_ready)
            ):
                for i, process in enumerate(cls.processes):
                    return_code = process.poll()
                    if return_code is not None:
                        # Server failed to start (non-zero exit code) or crashed
                        raise Exception(
                            f"Server process exited with code {return_code}. "
                            "Check server logs for errors."
                        )

                    try:
                        headers = {
                            "Content-Type": "application/json; charset=utf-8",
                        }
                        response = session.get(
                            f"http://127.0.0.1:{cls.base_port + i}/health_generate",
                            headers=headers,
                        )
                        if response.status_code == 200:
                            server_ready[i] = True
                    except requests.RequestException:
                        pass

                    return_code = process.poll()
                    if return_code is not None:
                        raise Exception(
                            f"Server unexpectedly exits ({return_code=}). Usually there will be error logs describing the cause far above this line."
                        )

                    time.sleep(10)
        if not all(server_ready):
            for process in cls.processes:
                kill_process_tree(process.pid)
            raise TimeoutError("Server failed to start within the timeout period.")

    @classmethod
    def tearDownClass(cls):
        for process in cls.processes:
            kill_process_tree(process.pid)

    def test_gsm8k(self):
        args = SimpleNamespace(
            num_shots=5,
            data_path=None,
            num_questions=200,
            max_new_tokens=512,
            parallel=128,
            host="http://127.0.0.1",
            port=self.base_port,
        )
        metrics = run_eval_few_shot_gsm8k(args)
        print(metrics)

        self.assertGreater(metrics["accuracy"], 0.60)


if __name__ == "__main__":
    unittest.main()
