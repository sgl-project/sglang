import io
import re

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
)


class TestOnlineQuantizationMemoryLoad(CustomTestCase):
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
            ],
            return_stdout_stderr=(cls.stdout, cls.stderr),
        )

        # # Extract and display peak GPU memory from logs
        combined_output = cls.stdout.getvalue() + cls.stderr.getvalue()
        peak_memory = cls._extract_peak_memory(combined_output)

        if is_cuda_alike() and not peak_memory:
            raise ValueError("Should have found peak memory")

        cls.peak_memory = float(peak_memory)

    @classmethod
    def _extract_peak_memory(cls, log_output):
        """Extract peak GPU memory value from log output."""
        # Search for the log message pattern
        pattern = r"Peak GPU memory after loading weights:\s+([\d.]+)\s+GiB"
        match = re.search(pattern, log_output)
        if match:
            return match.group(1)
        return None

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)
        cls.stdout.close()
        cls.stderr.close()

    def _test_peak_memory(self, threshold):
        """Helper method to test peak memory against a threshold."""
        if not is_cuda_alike():
            self.skipTest("not is_cuda_alike")
        assert self.peak_memory < threshold

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
        self._test_peak_memory(threshold=6)  # TP=1

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
        # Original Qwen/Qwen3-30B-A3B-Instruct-2507 BF16 model: 56.940 GiB
        self._test_peak_memory(threshold=21.5)  # TP=1

    def test_gsm8k(self):
        # Original Qwen/Qwen3-30B-A3B-Instruct-2507 reference accuracy: 0.94
        self._test_gsm8k(accuracy_threshold=0.9)


class TestFP8ToMXFP4Dense(TestOnlineQuantizationMemoryLoad):
    model = "Qwen/Qwen3-8B-FP8"

    def test_peak_memory(self):
        # Original Qwen/Qwen3-8B-FP8 model: 8.801 GiB (TP=1)
        self._test_peak_memory(threshold=7)  # TP=1

    def test_gsm8k(self):
        # Original Qwen/Qwen3-8B-FP8 reference accuracy: ~0.92
        self._test_gsm8k(accuracy_threshold=0.88)


class TestFP8ToMXFP4DenseTP1(TestFP8ToMXFP4Dense):
    tp = 1


class TestFP8ToMXFP4DenseTP2(TestFP8ToMXFP4Dense):
    tp = 2


class TestFP8ToMXFP4MOETP1(TestOnlineQuantizationMemoryLoad):
    model = "Qwen/Qwen3-30B-A3B-Instruct-2507-FP8"
    tp = 1

    def test_peak_memory(self):
        # Original Qwen/Qwen3-30B-A3B-Instruct-2507-FP8 model: 29.103 GiB (TP=1)
        self._test_peak_memory(threshold=21)  # TP=1

    def test_gsm8k(self):
        # Original Qwen/Qwen3-30B-A3B-Instruct-2507-FP8 reference accuracy: ~0.948
        self._test_gsm8k(accuracy_threshold=0.92)
