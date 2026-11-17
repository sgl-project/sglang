"""Multi-GPU online MXFP4 (quark_mxfp4) requantization checks for gfx95x (MI355X).

Split out of test/registered/e2e/quantization/test_quark_mxfp4.py, whose CI
runner has a single GPU; the base class is imported from there.

Run: python3 test/manual/quantization/test_quark_mxfp4_multi_gpu.py
"""

import sys
import unittest
from pathlib import Path

sys.path.insert(
    0, str(Path(__file__).resolve().parents[2] / "registered" / "e2e" / "quantization")
)
from test_quark_mxfp4 import TestOnlineQuantizationMemoryLoad  # noqa: E402


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


class TestFP8ToMXFP4DenseTP2(TestOnlineQuantizationMemoryLoad):
    tp = 2
    model = "Qwen/Qwen3-8B-FP8"

    def test_peak_memory(self):
        # Original Qwen/Qwen3-8B-FP8 model: 4.663 GiB (TP=2, peak_memory_before_load)
        self._test_peak_memory(
            threshold=4.2, test_start=False, add_peak_memory_before_load=True
        )

    def test_gsm8k(self):
        # Original Qwen/Qwen3-8B-FP8 reference accuracy: ~0.92
        self._test_gsm8k(accuracy_threshold=0.868)


class TestDeepSeekFP8ToMXFP4(TestOnlineQuantizationMemoryLoad):
    # Loading should take ~51.65 seconds on TP=8 on MI355X.
    # model = "deepseek-ai/DeepSeek-V3.2"  # FP8 model
    model = "deepseek-ai/DeepSeek-V3.2"
    tp = 8

    def test_peak_memory(self):
        # Original deepseek-ai/DeepSeek-V3.2 model: 80.366 GiB (TP=8, peak_memory_before_load)
        self._test_peak_memory(
            threshold=70, test_start=True, add_peak_memory_before_load=False
        )  # TP=8

    def test_gsm8k(self):
        # Original deepseek-ai/DeepSeek-V3.2 reference accuracy: ~0.948
        self._test_gsm8k(accuracy_threshold=0.94)


class TestKimiK2FP8ToMXFP4(TestOnlineQuantizationMemoryLoad):
    model = "moonshotai/Kimi-K2-Instruct-0905"  # FP8 model
    tp = 8

    # Same as in test/registered/amd/test_kimi_k2_instruct.py
    runner_args = [
        "--decode-attention-backend",
        "triton",
        "--prefill-attention-backend",
        "aiter",
        "--trust-remote-code",
    ]

    # Same as in test/registered/amd/test_kimi_k2_instruct.py, getting an error otherwise.
    environment = {"SGLANG_ROCM_FUSED_DECODE_MLA": "0"}

    def test_peak_memory(self):
        # Original moonshotai/Kimi-K2-Instruct-0905 model: 121.020 GiB (TP=8, peak_memory_before_load)
        self._test_peak_memory(
            threshold=82, test_start=True, add_peak_memory_before_load=False
        )  # TP=8

    def test_gsm8k(self):
        # Original moonshotai/Kimi-K2-Instruct-0905 reference accuracy: ~0.962
        self._test_gsm8k(accuracy_threshold=0.96)


class TestMiniMaxFP8ToMXFP4(TestOnlineQuantizationMemoryLoad):
    model = "MiniMaxAI/MiniMax-M2.1"  # FP8 model

    tp = 2
    # NOTE: this test is failing in FP16 (default dtype of the original MiniMax-M2.1 model).
    # Hence the usage of `--dtype bfloat16`
    # NOTE: this test requires the following fix for TP>1: https://github.com/sgl-project/sglang/pull/18310
    runner_args = ["--trust-remote-code", "--dtype", "bfloat16"]

    def test_peak_memory(self):
        # Original MiniMaxAI/MiniMax-M2.1 model: 107.375 GiB (TP=2, peak_memory_before_load)
        self._test_peak_memory(
            threshold=72, test_start=True, add_peak_memory_before_load=False
        )  # TP=2

    def test_gsm8k(self):
        # Original MiniMaxAI/MiniMax-M2.1 reference accuracy: 0.954
        self._test_gsm8k(accuracy_threshold=0.92)


if __name__ == "__main__":
    unittest.main()
