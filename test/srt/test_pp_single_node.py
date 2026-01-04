"""
Usage:
python3 -m unittest test_pp_single_node.TestPPAccuracy.test_gsm8k
python3 -m unittest test_pp_single_node.TestQwenPPAccuracy.test_pp_consistency
python3 -m unittest test_pp_single_node.TestFixedBugs.test_chunked_prefill_with_small_bs
python3 -m unittest test_pp_single_node.TestQwenVLPPAccuracy.test_mmmu
python3 -m unittest test_pp_single_node.TestPPWithHiCache.test_eval_accuracy
"""

import os
import subprocess
import time
import unittest
from types import SimpleNamespace
from urllib.parse import urlparse

import requests

from sglang.bench_one_batch_server import BenchArgs as OneBatchBenchArgs
from sglang.srt.server_args import ServerArgs
from sglang.srt.utils import kill_process_tree
from sglang.test.few_shot_gsm8k import run_eval as run_eval_few_shot_gsm8k
from sglang.test.run_eval import run_eval
from sglang.test.test_utils import (
    DEFAULT_MLA_MODEL_NAME_FOR_TEST,
    DEFAULT_MODEL_NAME_FOR_TEST,
    DEFAULT_MODEL_NAME_FOR_TEST_GLM_41V_PP,
    DEFAULT_MODEL_NAME_FOR_TEST_VL_PP,
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    find_available_port,
    is_in_ci,
    popen_launch_server,
    run_bench_one_batch_server,
)


class TestPPAccuracy(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.base_url = "http://127.0.0.1:23333"
        cls.process = popen_launch_server(
            DEFAULT_MODEL_NAME_FOR_TEST,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=[
                "--tp-size",
                2,
                "--pp-size",
                2,
                "--chunked-prefill-size",
                256,
            ],
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def test_gsm8k(self):
        args = SimpleNamespace(
            num_shots=5,
            data_path=None,
            num_questions=200,
            max_new_tokens=512,
            parallel=128,
            host="http://127.0.0.1",
            port=int(self.base_url.split(":")[-1]),
        )
        metrics = run_eval_few_shot_gsm8k(args)
        print(f"{metrics=}")

        self.assertGreater(metrics["accuracy"], 0.74)
        # Wait a little bit so that the memory check happens.
        time.sleep(4)

    def test_logprob(self):
        response = requests.post(
            f"{self.base_url}/generate",
            json={
                "text": "The capital of France is",
                "sampling_params": {
                    "temperature": 0,
                    "max_new_tokens": 16,
                },
                "return_logprob": True,
                "top_logprobs_num": 5,
                "logprob_start_len": 0,
            },
        )
        response_json = response.json()
        input_token_logprobs = response_json["meta_info"]["input_token_logprobs"]
        output_token_logprobs = response_json["meta_info"]["output_token_logprobs"]
        output_top_logprobs = response_json["meta_info"]["output_top_logprobs"]

        assert len(input_token_logprobs) == 6
        assert len(output_token_logprobs) == 16
        assert len(output_top_logprobs) == 16


class TestDPAttentionDP2PP2(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = DEFAULT_MLA_MODEL_NAME_FOR_TEST
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=[
                "--trust-remote-code",
                "--tp",
                "2",
                "--pp-size",
                "2",
                "--enable-dp-attention",
                "--dp",
                "2",
            ],
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def test_mgsm_en(self):
        args = SimpleNamespace(
            base_url=self.base_url,
            model=self.model,
            eval_name="mgsm_en",
            num_examples=None,
            num_threads=1024,
        )

        metrics = run_eval(args)
        print(f"{metrics=}")
        self.assertGreater(metrics["score"], 0.8)


class TestQwenVLPPAccuracy(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = DEFAULT_MODEL_NAME_FOR_TEST_VL_PP
        cls.base_url = "http://127.0.0.1:23333"
        cls.process = popen_launch_server(
            DEFAULT_MODEL_NAME_FOR_TEST_VL_PP,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=[
                "--tp-size",
                1,
                "--pp-size",
                4,
                "--chunked-prefill-size",
                8192,
                "--enable-multimodal",
            ],
        )

    def test_gsm8k(self):
        args = SimpleNamespace(
            num_shots=5,
            data_path=None,
            num_questions=200,
            max_new_tokens=512,
            parallel=128,
            host="http://127.0.0.1",
            port=int(self.base_url.split(":")[-1]),
        )
        metrics = run_eval_few_shot_gsm8k(args)
        print(f"{metrics=}")

        self.assertGreater(metrics["accuracy"], 0.65)
        # Wait a little bit so that the memory check happens.
        time.sleep(4)

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    @unittest.skipIf(is_in_ci(), "To reduce the CI execution time.")
    def test_mmmu(self):
        args = SimpleNamespace(
            base_url=self.base_url,
            model=self.model,
            eval_name="mmmu",
            num_examples=None,
            num_threads=32,
        )
        metrics = run_eval(args)
        print(f"{metrics=}")
        self.assertGreater(metrics["score"], 0.26)


class TestQwenPPAccuracy(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.base_url = "http://127.0.0.1:23334"  # different ports to avoid conflicts
        cls.model_name = "Qwen/Qwen3-8B"  # replace with your Qwen Model if needed

    def run_gsm8k_test(self, pp_size):
        process = popen_launch_server(
            self.model_name,
            self.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=[
                "--pp-size",
                pp_size,
                "--chunked-prefill-size",
                256,
            ],
        )

        try:
            args = SimpleNamespace(
                num_shots=5,
                data_path=None,
                num_questions=200,
                max_new_tokens=512,
                parallel=128,
                host="http://127.0.0.1",
                port=int(self.base_url.split(":")[-1]),
            )
            metrics = run_eval_few_shot_gsm8k(args)
            time.sleep(5)
            return metrics
        finally:
            kill_process_tree(process.pid)

    @unittest.skipIf(is_in_ci(), "To reduce the CI execution time.")
    def test_pp_consistency(self):
        baseline = self.run_gsm8k_test(pp_size=1)
        pp_metrics = self.run_gsm8k_test(pp_size=2)

        print(f"[Qwen PP Comparison] Baseline: {baseline} | PP: {pp_metrics}")

        self.assertGreaterEqual(baseline["accuracy"], 0.74)
        self.assertGreaterEqual(
            pp_metrics["accuracy"],
            baseline["accuracy"] - 0.02,
            msg=(
                f"PP accuracy dropped more than 1% compared to baseline. "
                f"Baseline: {baseline['accuracy']:.2%}, PP: {pp_metrics['accuracy']:.2%}"
            ),
        )


class TestQwenPPTieWeightsAccuracy(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.base_url = "http://127.0.0.1:23335"  # different ports to avoid conflicts
        cls.model_name = (
            "Qwen/Qwen3-0.6B"  # qwen3 < 8B all have tie_word_embeddings = True
        )

    def run_gsm8k_test(self, pp_size):
        process = popen_launch_server(
            self.model_name,
            self.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=[
                "--pp-size",
                pp_size,
                "--chunked-prefill-size",
                256,
            ],
        )

        try:
            args = SimpleNamespace(
                num_shots=5,
                data_path=None,
                num_questions=200,
                max_new_tokens=512,
                parallel=128,
                host="http://127.0.0.1",
                port=int(self.base_url.split(":")[-1]),
            )
            metrics = run_eval_few_shot_gsm8k(args)
            time.sleep(5)
            return metrics
        finally:
            kill_process_tree(process.pid)

    def test_pp_consistency(self):
        baseline = self.run_gsm8k_test(pp_size=1)
        pp_metrics = self.run_gsm8k_test(pp_size=2)

        print(f"[Qwen PP Comparison] Baseline: {baseline} | PP: {pp_metrics}")

        self.assertGreaterEqual(baseline["accuracy"], 0.38)
        self.assertGreaterEqual(
            pp_metrics["accuracy"],
            baseline["accuracy"] - 0.02,
            msg=(
                f"PP accuracy dropped more than 1% compared to baseline. "
                f"Baseline: {baseline['accuracy']:.2%}, PP: {pp_metrics['accuracy']:.2%}"
            ),
        )


class TestQwenMoePPAccuracy(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.base_url = "http://127.0.0.1:23336"  # different ports to avoid conflicts
        cls.model_name = "Qwen/Qwen3-30B-A3B"  # replace with your Qwen Model if needed

    def run_gsm8k_test(self, pp_size):
        process = popen_launch_server(
            self.model_name,
            self.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=[
                "--pp-size",
                pp_size,
                "--chunked-prefill-size",
                256,
            ],
        )

        try:
            args = SimpleNamespace(
                num_shots=5,
                data_path=None,
                num_questions=200,
                max_new_tokens=512,
                parallel=128,
                host="http://127.0.0.1",
                port=int(self.base_url.split(":")[-1]),
            )
            metrics = run_eval_few_shot_gsm8k(args)
            time.sleep(5)
            return metrics
        finally:
            kill_process_tree(process.pid)

    def test_pp_consistency(self):
        baseline = self.run_gsm8k_test(pp_size=1)
        pp_metrics = self.run_gsm8k_test(pp_size=2)

        print(f"[Qwen PP Comparison] Baseline: {baseline} | PP: {pp_metrics}")

        self.assertGreaterEqual(baseline["accuracy"], 0.74)
        self.assertGreaterEqual(
            pp_metrics["accuracy"],
            baseline["accuracy"] - 0.02,
            msg=(
                f"PP accuracy dropped more than 1% compared to baseline. "
                f"Baseline: {baseline['accuracy']:.2%}, PP: {pp_metrics['accuracy']:.2%}"
            ),
        )


class TestFixedBugs(unittest.TestCase):
    def test_chunked_prefill_with_small_bs(self):
        model = DEFAULT_MODEL_NAME_FOR_TEST
        server_args = ServerArgs(model_path=model)
        bench_args = OneBatchBenchArgs(
            batch_size=(1,),
            input_len=(1,),
            output_len=(1,),
            base_url=DEFAULT_URL_FOR_TEST,
        )
        other_server_args = [
            "--tp-size",
            2,
            "--pp-size",
            2,
            "--chunked-prefill",
            256,
            "--max-running-requests",
            2,
        ]
        run_bench_one_batch_server(
            model,
            DEFAULT_URL_FOR_TEST,
            server_args,
            bench_args,
            other_server_args,
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


@unittest.skipIf(
    is_in_ci(), "Skipping GLM41V PP accuracy test before it gets more stable"
)
class TestGLM41VPPAccuracy(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = DEFAULT_MODEL_NAME_FOR_TEST_GLM_41V_PP
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.process = popen_launch_server(
            DEFAULT_MODEL_NAME_FOR_TEST_GLM_41V_PP,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=[
                "--tp-size",
                1,
                "--pp-size",
                2,
                "--chunked-prefill-size",
                8192,
                "--enable-multimodal",
                "--reasoning-parser",
                "glm45",
            ],
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def test_mmmu(self):
        args = SimpleNamespace(
            base_url=self.base_url,
            model=self.model,
            eval_name="mmmu",
            num_examples=None,
            num_threads=32,
            response_answer_regex=r"<\|begin_of_box\|>(.*)<\|end_of_box\|>",
        )

        metrics = run_eval(args)
        print(f"{metrics=}")
        self.assertGreater(metrics["score"], 0.45)


if __name__ == "__main__":
    unittest.main()
