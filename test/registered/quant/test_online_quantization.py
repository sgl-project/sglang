import io
import os
import re

from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci

register_cuda_ci(est_time=103, suite="stage-b-test-small-1-gpu")
register_amd_ci(est_time=106, suite="stage-b-test-small-1-gpu-amd")
from types import SimpleNamespace

from sglang.srt.utils import kill_process_tree
from sglang.srt.utils.common import is_cuda_alike
from sglang.test.few_shot_gsm8k import run_eval
from sglang.test.test_utils import (
    DEFAULT_SMALL_MODEL_NAME_FOR_TEST_QWEN,
    DEFAULT_SMALL_MOE_MODEL_NAME_FOR_TEST_BASE,
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)


class TestOnlineQuantizationMemoryLoad(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        cls.SGLANG_USE_AITER = os.environ.get("SGLANG_USE_AITER", None)

        # DEFAULT_SMALL_MOE_MODEL_NAME_FOR_TEST_BASE has a shape not compatible with aiter.
        os.environ["SGLANG_USE_AITER"] = "0"

        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.stdout = io.StringIO()
        cls.stderr = io.StringIO()
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=[
                "--quantization",
                "fp8",
                "--tensor-parallel-size",
                "1",
                "--log-level",
                "debug",
            ],
            return_stdout_stderr=(cls.stdout, cls.stderr),
        )

        # Extract and display peak GPU memory from logs
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

        if cls.SGLANG_USE_AITER:
            os.environ["SGLANG_USE_AITER"] = cls.SGLANG_USE_AITER


class TestOnlineQuantizationMemoryLoadDense(TestOnlineQuantizationMemoryLoad):
    model = DEFAULT_SMALL_MODEL_NAME_FOR_TEST_QWEN

    def test_peak_memory(self):
        if not is_cuda_alike():
            self.skipTest("not is_cuda_alike")

        # Original BF16 model: 2.887 GiB
        assert self.peak_memory < 2

    def test_gsm8k(self):
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
        self.assertGreater(metrics["accuracy"], 0.01)


class TestOnlineQuantizationMemoryLoadMOE(TestOnlineQuantizationMemoryLoad):
    model = DEFAULT_SMALL_MOE_MODEL_NAME_FOR_TEST_BASE

    def test_peak_memory(self):
        if not is_cuda_alike():
            self.skipTest("not is_cuda_alike")

        # Original BF16 model: 26.695 GiB
        assert self.peak_memory < 15.5

    def test_gsm8k(self):
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
        self.assertGreater(metrics["accuracy"], 0.03)
