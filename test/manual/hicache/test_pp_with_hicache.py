"""
Usage:
python3 -m unittest test_pp_with_hicache.TestPPWithHiCache.test_eval_accuracy
"""

import os
import subprocess
import time
import unittest
from types import SimpleNamespace
from urllib.parse import urlparse

import requests

from sglang.srt.utils import kill_process_tree
from sglang.test.few_shot_gsm8k import run_eval as run_eval_few_shot_gsm8k
from sglang.test.test_utils import (
    DEFAULT_MODEL_NAME_FOR_TEST,
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    find_available_port,
    popen_launch_server,
)


class TestPPWithHiCache(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.base_url = f"http://127.0.0.1:{find_available_port(23337)}"
        parsed_url = urlparse(cls.base_url)
        cls.base_host = parsed_url.hostname
        cls.base_port = str(parsed_url.port)
        cls.model = DEFAULT_MODEL_NAME_FOR_TEST

        cls._start_mooncake_services()

        server_args_dict = {
            "--enable-hierarchical-cache": True,
            "--mem-fraction-static": 0.6,
            "--hicache-ratio": 1.2,
            "--page-size": 64,
            "--enable-cache-report": True,
            "--hicache-storage-prefetch-policy": "wait_complete",
            "--hicache-storage-backend": "mooncake",
            "--tp-size": 2,
            "--pp-size": 2,
            "--chunked-prefill-size": 256,
            "--hicache-mem-layout": "page_first",
        }

        final_server_args = []
        for key, value in server_args_dict.items():
            final_server_args.append(str(key))
            if value is not True:
                final_server_args.append(str(value))

        env_vars = {**os.environ, **cls._mooncake_env()}

        try:
            cls.process = popen_launch_server(
                cls.model,
                cls.base_url,
                timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
                other_args=final_server_args,
                env=env_vars,
            )
        except Exception:
            cls._stop_mooncake_services()
            raise

    @classmethod
    def tearDownClass(cls):
        if hasattr(cls, "process"):
            kill_process_tree(cls.process.pid)
        cls._stop_mooncake_services()

    @classmethod
    def _start_mooncake_services(cls):
        try:
            import mooncake.http_metadata_server  # type: ignore  # noqa: F401
        except Exception as exc:  # pragma: no cover - environment dependent
            raise unittest.SkipTest(
                f"Mooncake metadata server module unavailable: {exc}"
            ) from exc

        cls._mooncake_master_port = find_available_port(50051)
        cls._mooncake_metadata_port = find_available_port(8080)

        try:
            cls._mooncake_metadata_process = subprocess.Popen(
                [
                    "python3",
                    "-m",
                    "mooncake.http_metadata_server",
                    "--port",
                    str(cls._mooncake_metadata_port),
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                preexec_fn=os.setsid,
            )
        except (FileNotFoundError, subprocess.SubprocessError) as exc:
            cls._stop_mooncake_services()
            raise unittest.SkipTest(
                f"Could not start Mooncake metadata service: {exc}"
            ) from exc

        try:
            cls._mooncake_master_process = subprocess.Popen(
                ["mooncake_master", "--port", str(cls._mooncake_master_port)],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                preexec_fn=os.setsid,
            )
        except (FileNotFoundError, subprocess.SubprocessError) as exc:
            cls._stop_mooncake_services()
            raise unittest.SkipTest(f"Could not start mooncake_master: {exc}") from exc

        if not cls._wait_for_mooncake_ready():
            cls._stop_mooncake_services()
            raise unittest.SkipTest("Mooncake services did not become ready in time")

    @classmethod
    def _stop_mooncake_services(cls):
        for attr in ("_mooncake_metadata_process", "_mooncake_master_process"):
            proc = getattr(cls, attr, None)
            if proc:
                try:
                    os.killpg(os.getpgid(proc.pid), 9)
                    proc.wait(timeout=5)
                except Exception:
                    pass
        cls._mooncake_metadata_process = None
        cls._mooncake_master_process = None

    @classmethod
    def _mooncake_env(cls):
        return {
            "MOONCAKE_MASTER": f"127.0.0.1:{cls._mooncake_master_port}",
            "MOONCAKE_PROTOCOL": "tcp",
            "MC_MS_AUTO_DISC": "0",
            "MOONCAKE_DEVICE": "",
            "MOONCAKE_TE_META_DATA_SERVER": f"http://127.0.0.1:{cls._mooncake_metadata_port}/metadata",
            "MOONCAKE_GLOBAL_SEGMENT_SIZE": "4294967296",
            "SGLANG_ENABLE_DETERMINISTIC_INFERENCE": "1",
        }

    @classmethod
    def _wait_for_mooncake_ready(cls, timeout: int = 30) -> bool:
        start_time = time.time()
        while time.time() - start_time < timeout:
            metadata_ready = False
            master_ready = False

            if (
                getattr(cls, "_mooncake_metadata_process", None)
                and cls._mooncake_metadata_process.poll() is None
            ):
                try:
                    resp = requests.get(
                        f"http://127.0.0.1:{cls._mooncake_metadata_port}/metadata",
                        timeout=2,
                    )
                    print(resp)
                    metadata_ready = True
                except requests.RequestException:
                    metadata_ready = False

            if (
                getattr(cls, "_mooncake_master_process", None)
                and cls._mooncake_master_process.poll() is None
            ):
                if time.time() - start_time > 3:
                    master_ready = True

            if metadata_ready and master_ready:
                return True

            time.sleep(1.5)

        return False

    def flush_cache(self) -> bool:
        try:
            response = requests.post(f"{self.base_url}/flush_cache", timeout=10)
            return response.status_code == 200
        except requests.RequestException:
            return False

    def test_eval_accuracy(self):
        args = SimpleNamespace(
            num_shots=5,
            data_path=None,
            num_questions=40,
            max_new_tokens=256,
            parallel=24,
            host=f"http://{self.base_host}",
            port=int(self.base_port),
        )

        metrics_initial = run_eval_few_shot_gsm8k(args)
        self.assertGreater(metrics_initial["accuracy"], 0.6)

        self.assertTrue(self.flush_cache())
        time.sleep(2)

        metrics_cached = run_eval_few_shot_gsm8k(args)
        self.assertGreater(metrics_cached["accuracy"], 0.6)

        accuracy_diff = abs(metrics_initial["accuracy"] - metrics_cached["accuracy"])
        self.assertLess(accuracy_diff, 0.05)


if __name__ == "__main__":
    unittest.main()
