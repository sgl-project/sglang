import io
import re
import unittest
from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci

register_cuda_ci(est_time=103, suite="stage-b-test-small-1-gpu")
register_amd_ci(est_time=106, suite="stage-b-test-small-1-gpu-amd")
from types import SimpleNamespace

from sglang.srt.utils import kill_process_tree
from sglang.srt.utils.common import is_cuda_alike
from sglang.test.few_shot_gsm8k import run_eval
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
    is_in_ci,
)


class TestOnlineQuantizationMemoryLoad(CustomTestCase):
    runner_args = []

    @classmethod
    def setUpClass(cls):
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.stdout = io.StringIO()
        cls.stderr = io.StringIO()
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=[
                "--quantization",
                "quark_mxfp4",
                # `context-length` limitation required for Qwen MOE model
                # TODO: Remove once https://github.com/sgl-project/sglang/pull/18255 and https://github.com/sgl-project/sglang/pull/18263 are merged.
                "--context-length",
                "3000",
                "--tensor-parallel-size",
                cls.tp if hasattr(cls, "tp") else "1",
                "--log-level",
                "debug",
                *cls.runner_args,
            ],
            return_stdout_stderr=(cls.stdout, cls.stderr),
        )
        # cls.wait_server_ready(
        #     cls.base_url + "/health", timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH
        # )
        import time

        import requests

        url = cls.base_url + "/health"
        timeout = DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH
        start_time = time.perf_counter()
        while True:
            try:
                response = requests.get(url)
                if response.status_code == 200:
                    print(f"Server {url} is ready")
                    break
            except Exception:
                pass

            if time.perf_counter() - start_time > timeout:
                raise RuntimeError(f"Server {url} failed to start in {timeout}s")
            time.sleep(1)

        # # Extract and display peak GPU memory from logs
        combined_output = cls.stdout.getvalue() + cls.stderr.getvalue()

        peak_memory_before_load = cls._extract_peak_memory_before_load(combined_output)
        if is_cuda_alike() and not peak_memory_before_load:
            raise ValueError("Should have found peak memory")
        cls.peak_memory_before_load = float(peak_memory_before_load)

        memory_increase_load_weights = cls._extract_memory_increase_load_weights(
            combined_output
        )
        if is_cuda_alike() and not memory_increase_load_weights:
            raise ValueError("Should have found memory increase in load_weights")
        cls.memory_increase_load_weights = float(memory_increase_load_weights)

    @classmethod
    def _extract_peak_memory_before_load(cls, log_output):
        """Extract peak GPU memory value from log output."""
        # Search for the log message pattern
        pattern = r"Peak GPU memory before loading weights:\s+([\d.]+)\s+GiB"
        match = re.search(pattern, log_output)
        if match:
            return match.group(1)
        return None

    @classmethod
    def _extract_memory_increase_load_weights(cls, log_output):
        """Extract memory increase during load_weights call."""
        # Search for the log message pattern
        pattern = r"Memory increase during load_weights:\s+([\d.]+)\s+GiB"
        match = re.search(pattern, log_output)
        if match:
            return match.group(1)
        return None

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)
        cls.stdout.close()
        cls.stderr.close()

    def _test_peak_memory(
        self, threshold, test_start: bool, add_peak_memory_before_load: bool
    ):
        """Helper method to test peak memory against a threshold."""
        if not is_cuda_alike():
            self.skipTest("not is_cuda_alike")

        # NOTE: We can not simply rely on peak memory after `load_weights` as functions used
        # in-between (e.g. FP8->MXFP4 requantization) during weight loading may have a higher peak memory footprint
        # than simply the allocated weights.
        if add_peak_memory_before_load:
            reference_gib = (
                self.memory_increase_load_weights + self.peak_memory_before_load
            )
        else:
            reference_gib = self.memory_increase_load_weights

        assert reference_gib < threshold

        if test_start:
            # Weights initialized on meta device (not for dense BF16->MXFP4)
            assert self.peak_memory_before_load < 5

    def _test_gsm8k(self, accuracy_threshold):
        """Helper method to test GSM8K accuracy against a threshold."""
        args = SimpleNamespace(
            num_shots=8,
            data_path=None,
            num_questions=500,
            max_new_tokens=512,
            parallel=128,
            host="http://127.0.0.1",
            port=int(self.base_url.split(":")[-1]),
        )
        metrics = run_eval(args)
        print(f"{metrics=}")
        self.assertGreater(metrics["accuracy"], accuracy_threshold)


class TestOnlineQuantizationMemoryLoadDense(TestOnlineQuantizationMemoryLoad):
    model = "Qwen/Qwen3-8B"

    def test_peak_memory(self):
        # Original Qwen/Qwen3-8B BF16 model: 15.268 GiB
        self._test_peak_memory(
            threshold=6, test_start=False, add_peak_memory_before_load=True
        )

    def test_gsm8k(self):
        # Original Qwen/Qwen3-8B reference accuracy: ~0.92
        self._test_gsm8k(accuracy_threshold=0.85)


class TestOnlineQuantizationMemoryLoadMOE(TestOnlineQuantizationMemoryLoad):
    # Unfortunately, smaller models as Qwen/Qwen1.5-MoE-A2.7B or ibm-granite/granite-3.0-3b-a800m-base currently crash in AITER:
    # - Qwen/Qwen1.5-MoE-A2.7B => K // 2 = 704 as intermediate size, not multiple of 128.
    # - ibm-granite/granite-3.0-3b-a800m-base: dtype issue with fp16 in AITER MOE MLP activation
    # so using a large model here.
    model = "Qwen/Qwen3-30B-A3B-Instruct-2507"
    # TODO: test TP>=2 with an other model (Qwen/Qwen3-30B-A3B-Instruct-2507 crashes in this case as 768/2 = 384, and 384/32 = 12 not divisible by BLOCK_SIZE_N=8. in fused_dynamic_mxfp4_quant_moe_sort.

    def test_peak_memory(self):
        # Original Qwen/Qwen3-30B-A3B-Instruct-2507 BF16 model: 56.940 GiB  # TODO update
        self._test_peak_memory(
            threshold=17, test_start=False, add_peak_memory_before_load=True
        )

    def test_gsm8k(self):
        # Original Qwen/Qwen3-30B-A3B-Instruct-2507 reference accuracy: 0.94
        self._test_gsm8k(accuracy_threshold=0.89)


class TestFP8ToMXFP4Dense(TestOnlineQuantizationMemoryLoad):
    model = "Qwen/Qwen3-8B-FP8"

    def test_peak_memory(self):
        # Original Qwen/Qwen3-8B-FP8 model: TODO update GiB (TP=1, peak_memory_before_load)
        self._test_peak_memory(threshold=7)  # TP=1 TODO update

    def test_gsm8k(self):
        # Original Qwen/Qwen3-8B-FP8 reference accuracy: ~0.92
        self._test_gsm8k(accuracy_threshold=0.88)


class TestFP8ToMXFP4DenseTP1(TestFP8ToMXFP4Dense):
    tp = 1


class TestFP8ToMXFP4DenseTP2(TestFP8ToMXFP4Dense):
    tp = 2


class TestFP8ToMXFP4MOETP1(TestOnlineQuantizationMemoryLoad):
    model = "Qwen/Qwen3-30B-A3B-Instruct-2507-FP8"  # FP8 model
    tp = 1

    def test_peak_memory(self):
        # Original Qwen/Qwen3-30B-A3B-Instruct-2507-FP8 model: TODO update GiB (TP=1, peak_memory_before_load)
        self._test_peak_memory(threshold=21)  # TP=1 TODO update

    def test_gsm8k(self):
        # Original Qwen/Qwen3-30B-A3B-Instruct-2507-FP8 reference accuracy: ~0.948
        self._test_gsm8k(accuracy_threshold=0.92)

@unittest.skipIf(is_in_ci(), "local test only")
class TestDeepSeekFP8ToMXFP4(TestOnlineQuantizationMemoryLoad):
    model = "deepseek-ai/DeepSeek-V3.2"  # FP8 model
    tp = 8

    def test_gsm8k(self):
        # Original deepseek-ai/DeepSeek-V3.2 reference accuracy: TODO update
        self._test_gsm8k(accuracy_threshold=0.92) # TODO

@unittest.skipIf(is_in_ci(), "local test only")
class TestKimiK2FP8ToMXFP4(TestOnlineQuantizationMemoryLoad):
    model = "moonshotai/Kimi-K2-Instruct-0905"  # FP8 model
    tp = 8

    def test_gsm8k(self):
        # Original moonshotai/Kimi-K2-Instruct-0905 reference accuracy: TODO
        self._test_gsm8k(accuracy_threshold=0.92) # TODO

@unittest.skipIf(is_in_ci(), "local test only")
class TestMiniMaxFP8ToMXFP4(TestOnlineQuantizationMemoryLoad):
    model = "MiniMaxAI/MiniMax-M2.1"  # FP8 model

    tp = 2
    # NOTE: this test is failing in FP16 (default dtype of the original MiniMax-M2.1 model).
    # Hence the usage of `--dtype bfloat16`
    # NOTE: this test requires the following fix for TP>1: https://github.com/sgl-project/sglang/pull/18310
    runner_args = ["--trust-remote-code", "--dtype", "bfloat16"]

    def test_peak_memory(self):
        # Original MiniMaxAI/MiniMax-M2.1 model: 107.375 GiB (TP=2, peak_memory_before_load)
        self._test_peak_memory(threshold=65)  # TP=2

    def test_gsm8k(self):
        # Original MiniMaxAI/MiniMax-M2.1 reference accuracy: 0.954
        self._test_gsm8k(accuracy_threshold=0.92)
