import os
import time
import unittest
from concurrent.futures import ThreadPoolExecutor
from urllib.parse import urlparse

import requests

from sglang.srt.utils import kill_process_tree
from sglang.test.ascend.test_ascend_utils import (
    QWEN3_0_6B_WEIGHTS_PATH,
    logger,
    popen_with_error_check,
)
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

register_npu_ci(
    est_time=400,
    suite="full-16-npu-a3",
    nightly=True,
)

MODEL_PATH = QWEN3_0_6B_WEIGHTS_PATH

COMMON_ARGS = [
    "--trust-remote-code",
    "--mem-fraction-static",
    "0.8",
    "--attention-backend",
    "ascend",
    "--disable-cuda-graph",
    "--disable-radix-cache",
    "--skip-server-warmup",
]

PROMPT_TEXT = "The capital of France is"
EXPECTED_ANSWER = "Paris"


def send_request(url):
    return requests.post(
        f"{url}/generate",
        json={
            "text": PROMPT_TEXT,
            "sampling_params": {
                "temperature": 0,
                "max_new_tokens": 100,
            },
        },
    )


class PDDisaggregationVariableLengthScheduleFallbackTest(CustomTestCase):
    """Verify PD disaggregation: bucket scheduling under balance, fallback to load-aware routing on prefill imbalance.

    [Test Category] Functional
    [Test Target] --prefill-policy bucket, --balance-rel-threshold, --balance-abs-threshold, --bucket-adjust-interval-secs
    """

    @classmethod
    def setUpClass(cls):
        cls._init_ports()
        cls.log_files = {}
        cls.processes = {}

        cls._start_prefill("prefill_0", base_gpu_id=0)
        cls._start_prefill("prefill_1", base_gpu_id=1)
        cls._start_prefill("prefill_2", base_gpu_id=2)
        cls._start_decode()
        cls._start_router()

    @classmethod
    def _init_ports(cls):
        parsed = urlparse(DEFAULT_URL_FOR_TEST)
        cls.base_host = parsed.hostname

        base_port = int(parsed.port)
        cls.router_port = base_port

        cls.ports = {
            "prefill_0": base_port + 100,
            "prefill_1": base_port + 200,
            "prefill_2": base_port + 300,
            "decode_1": base_port + 400,
        }

        cls.bootstrap_ports = {
            "prefill_0": base_port + 500,
            "prefill_1": base_port + 600,
            "prefill_2": base_port + 700,
        }

        cls.ascend_mf_store_url = f"tcp://{cls.base_host}:{base_port + 800}"

        cls.urls = {
            name: f"http://{cls.base_host}:{port}" for name, port in cls.ports.items()
        }

        cls.router_url = f"http://{cls.base_host}:{cls.router_port}"
        cls.base_url = cls.router_url

    @classmethod
    def _open_logs(cls, name):
        out = open(f"./{name}_out_log.txt", "w+", encoding="utf-8")
        err = open(f"./{name}_err_log.txt", "w+", encoding="utf-8")
        cls.log_files[name] = (out, err)
        return out, err

    @classmethod
    def _mf_env(cls):
        return {
            **os.environ,
            "ASCEND_MF_STORE_URL": cls.ascend_mf_store_url,
        }

    @classmethod
    def _start_prefill(cls, name, base_gpu_id):
        out, err = cls._open_logs(name)

        args = COMMON_ARGS + [
            "--base-gpu-id",
            str(base_gpu_id),
            "--disaggregation-transfer-backend",
            "ascend",
            "--disaggregation-mode",
            "prefill",
            "--disaggregation-bootstrap-port",
            str(cls.bootstrap_ports[name]),
        ]

        cls.processes[name] = popen_launch_server(
            MODEL_PATH,
            cls.urls[name],
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=args,
            env=cls._mf_env(),
            return_stdout_stderr=(out, err),
        )

    @classmethod
    def _start_decode(cls):
        out, err = cls._open_logs("decode_1")

        args = COMMON_ARGS + [
            "--base-gpu-id",
            "3",
            "--disaggregation-transfer-backend",
            "ascend",
            "--disaggregation-mode",
            "decode",
            "--load-balance-method",
            "round_robin",
        ]

        cls.processes["decode_1"] = popen_launch_server(
            MODEL_PATH,
            cls.urls["decode_1"],
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=args,
            env=cls._mf_env(),
            return_stdout_stderr=(out, err),
        )

    @classmethod
    def _start_router(cls):
        out, err = cls._open_logs("router")

        cmd = [
            "python3",
            "-m",
            "sglang_router.launch_router",
            "--pd-disaggregation",
            "--decode",
            cls.urls["decode_1"],
            "--prefill",
            cls.urls["prefill_0"],
            str(cls.bootstrap_ports["prefill_0"]),
            "--prefill",
            cls.urls["prefill_1"],
            str(cls.bootstrap_ports["prefill_1"]),
            "--prefill",
            cls.urls["prefill_2"],
            str(cls.bootstrap_ports["prefill_2"]),
            "--host",
            cls.base_host,
            "--port",
            str(cls.router_port),
            "--prefill-policy",
            "bucket",
            "--balance-rel-threshold",
            "1.0001",
            "--balance-abs-threshold",
            "32",
            "--bucket-adjust-interval-secs",
            "5",
        ]

        cls.processes["router"] = popen_with_error_check(
            cmd, return_stdout_stderr=(out, err)
        )

        cls._wait_ready(f"{cls.router_url}/health")
        logger.info("Waiting 60 seconds for the server to fully initialize...")
        time.sleep(60)

    @classmethod
    def _wait_ready(cls, url, timeout=120):
        start = time.perf_counter()
        while True:
            try:
                if requests.get(url).status_code == 200:
                    logger.info(f"Server {url} is ready")
                    return
            except Exception:
                pass

            if time.perf_counter() - start > timeout:
                raise RuntimeError(f"Server {url} failed to start in {timeout}s")

            time.sleep(1)

    @classmethod
    def tearDownClass(cls):
        for name, proc in cls.processes.items():
            if proc:
                try:
                    kill_process_tree(proc.pid)
                    logger.info(f"Killed {name} pid={proc.pid}")
                except Exception as e:
                    logger.warning(f"Failed to kill {name}: {e}")

        time.sleep(5)

        for out, err in cls.log_files.values():
            for f in (out, err):
                try:
                    f.close()
                except Exception:
                    pass

        for name in list(cls.log_files.keys()):
            for suffix in ("out_log.txt", "err_log.txt"):
                path = f"./{name}_{suffix}"
                try:
                    if os.path.exists(path):
                        os.remove(path)
                except Exception as e:
                    logger.warning(f"Failed to remove {path}: {e}")

    def count_generate_requests(self, name):
        out, _ = self.log_files[name]
        out.seek(0)
        return out.read().count("POST /generate HTTP/1.1")

    def test_variable_length_schedule_fallback_to_load_balance(self):
        # 60 requests → prefill load balanced via bucket fallback (≥18 each)
        with ThreadPoolExecutor(max_workers=60) as ex:
            futures = [ex.submit(send_request, self.base_url) for _ in range(60)]
            for f in futures:
                resp = f.result()
                self.assertEqual(resp.status_code, 200)
                self.assertIn(EXPECTED_ANSWER, resp.text)

        logger.info(
            f'prefill_0 processed requests: {self.count_generate_requests("prefill_0")}'
        )
        logger.info(
            f'prefill_1 processed requests: {self.count_generate_requests("prefill_1")}'
        )
        logger.info(
            f'prefill_2 processed requests: {self.count_generate_requests("prefill_2")}'
        )
        self.assertGreaterEqual(self.count_generate_requests("prefill_0"), 18)
        self.assertGreaterEqual(self.count_generate_requests("prefill_1"), 18)
        self.assertGreaterEqual(self.count_generate_requests("prefill_2"), 18)


if __name__ == "__main__":
    unittest.main()
