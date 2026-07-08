import io
import re
import unittest

from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci

register_cuda_ci(est_time=103, stage="base-b", runner_config="1-gpu-small")
register_amd_ci(est_time=106, suite="stage-b-test-1-gpu-small-amd-mi35x")
import os
import time
from types import SimpleNamespace

import requests

from sglang.srt.utils import kill_process_tree
from sglang.srt.utils.common import is_cuda_alike, mxfp_supported
from sglang.test.few_shot_gsm8k import run_eval
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    is_in_ci,
    popen_launch_server,
)


class TestOnlineQuantizationMemoryLoad(CustomTestCase):
    runner_args = []
    environment = {}

    @classmethod
    def setUpClass(cls):
        if not mxfp_supported():
            raise unittest.SkipTest(
                "online MXFP4 quantization requires an AMD ROCm device with "
                "FP4 hardware support (gfx95x, e.g. MI355x)"
            )
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

        cls.original_envs = {}
        for env_name, env_value in cls.environment.items():
            original_env = os.environ.get(env_name, None)
            if original_env is not None:
                cls.original_envs[env_name] = os.environ.get(env_name, None)

            os.environ[env_name] = env_value

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

        # Keep the raw server for memory numbers, which are parsed lazily by
        # _test_peak_memory so subclasses that don't test memory (e.g. the
        # NVFP4->MXFP4 accuracy-only class) don't require these log lines.
        cls.combined_output = cls.stdout.getvalue() + cls.stderr.getvalue()

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
        # Signed: the value is (free_before - free_after) around load_weights.
        # When the on-device source representation is larger than the loaded
        # result (e.g. requantizing to a more compact format), loading frees
        # net memory and the reported increase is negative.
        pattern = r"Memory increase during load_weights:\s+(-?[\d.]+)\s+GiB"
        match = re.search(pattern, log_output)
        if match:
            return match.group(1)
        return None

    @classmethod
    def tearDownClass(cls):
        for env_name, env_value in cls.original_envs.items():
            os.environ[env_name] = env_value

        kill_process_tree(cls.process.pid)
        cls.stdout.close()
        cls.stderr.close()

    def _test_peak_memory(
        self, threshold, test_start: bool, add_peak_memory_before_load: bool
    ):
        """Helper method to test peak memory against a threshold."""
        if not is_cuda_alike():
            self.skipTest("not is_cuda_alike")

        peak_memory_before_load = self._extract_peak_memory_before_load(
            self.combined_output
        )
        if not peak_memory_before_load:
            raise ValueError("Should have found peak memory")
        peak_memory_before_load = float(peak_memory_before_load)

        memory_increase_load_weights = self._extract_memory_increase_load_weights(
            self.combined_output
        )
        if not memory_increase_load_weights:
            raise ValueError("Should have found memory increase in load_weights")
        memory_increase_load_weights = float(memory_increase_load_weights)

        # NOTE: We can not simply rely on peak memory after `load_weights` as functions used
        # in-between (e.g. NVFP4->MXFP4 requantization) during weight loading may have a higher peak memory footprint
        # than simply the allocated weights.
        if add_peak_memory_before_load:
            reference_gib = memory_increase_load_weights + peak_memory_before_load
        else:
            reference_gib = memory_increase_load_weights

        assert reference_gib < threshold

        if test_start:
            # Weights initialized on meta device (not for dense BF16->MXFP4)
            assert peak_memory_before_load < 5

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
        # Original Qwen/Qwen3-30B-A3B-Instruct-2507 BF16 model: 56.940 GiB
        self._test_peak_memory(
            threshold=17, test_start=False, add_peak_memory_before_load=True
        )

    def test_gsm8k(self):
        # Original Qwen/Qwen3-30B-A3B-Instruct-2507 reference accuracy: 0.94
        self._test_gsm8k(accuracy_threshold=0.89)


class TestNVFP4ToMXFP4MOETP1(TestOnlineQuantizationMemoryLoad):
    # ModelOpt NVFP4 export (quant_method="modelopt", quant_algo="NVFP4") =>
    # Nvfp4SourceConfig(). Exercises the NVFP4 -> MXFP4 MoE requantization path:
    # the per-expert dequantize_nvfp4 + dynamic_mxfp4_quant requant, and the w13
    # gate/up weight_scale_2 split in _requantize_nvfp4_to_mxfp4.
    model = "nvidia/Qwen3-30B-A3B-NVFP4"  # NVFP4 model
    tp = 1

    def test_gsm8k(self):
        # Requantized NVFP4 -> MXFP4 observed accuracy: ~0.88
        # (BF16 Qwen/Qwen3-30B-A3B reference: ~0.94).
        self._test_gsm8k(accuracy_threshold=0.85)


@unittest.skipIf(is_in_ci(), "local test only")
class TestDeepSeekR10528NVFP4ToMXFP4(TestOnlineQuantizationMemoryLoad):
    # NVFP4 to MXFP4 online requantization for DeepSeek-R1-0528-NVFP4 on TP=8.
    # Exercises the MLA attention path (attention_backend=aiter), multi-threaded
    # weight loading, and the per-expert NVFP4 MoE requantization path.
    model = "nvidia/DeepSeek-R1-0528-NVFP4"  # NVFP4 model
    tp = 8
    runner_args = [
        "--attention-backend",
        "aiter",
        "--model-loader-extra-config",
        '{"enable_multithread_load": true}',
    ]

    def test_gsm8k(self):
        # Requantized NVFP4 -> MXFP4 observed accuracy: ~0.95.
        self._test_gsm8k(accuracy_threshold=0.90)


if __name__ == "__main__":
    unittest.main()
