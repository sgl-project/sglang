import unittest

from sglang.srt.environ import envs
from sglang.test.accuracy_test_runner import AccuracyTestParams
from sglang.test.cache_consistency_jitter import get_jitter_engine, run_jitter_test
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.performance_test_runner import PerformanceTestParams
from sglang.test.run_combined_tests import run_combined_tests
from sglang.test.test_utils import ModelLaunchSettings, is_blackwell_system

# NVFP4 needs Blackwell FP4 kernels, so this runs on the Blackwell leg of the
# common 8-GPU suite (Hopper is skipped below).
register_cuda_ci(est_time=3600, stage="nightly", runner_config="8-gpu-h200")
register_cuda_ci(est_time=3600, stage="nightly", runner_config="8-gpu-b200")

INKLING_NVFP4_MODEL = "thinkingmachines/Inkling-NVFP4"
INKLING_SMALL_NVFP4_MODEL = "thinkingmachines/Inkling-Small-NVFP4"

# Verified Blackwell NVFP4 recipe; see the SGLang cookbook Inkling page.
NVFP4_ARGS = [
    "--tp=8",
    "--trust-remote-code",
    "--quantization=modelopt_fp4",
    "--attention-backend=fa4",
    "--page-size=128",
    "--fp4-gemm-backend=flashinfer_trtllm",
    "--moe-runner-backend=flashinfer_trtllm_routed",
    "--enable-torch-symm-mem",
    "--mamba-radix-cache-strategy=extra_buffer",
    "--mem-fraction-static=0.85",
    "--swa-full-tokens-ratio=0.1",
    "--mamba-full-memory-ratio=0.1",
    "--reasoning-parser=inkling",
]

# gsm8k measured ~0.98 (temp 0, 300 examples); floored with margin for the
# temperature=1.0 nightly config and FP4-kernel variance.
GSM8K_BASELINE = 0.95


class TestInklingNVFP4Nightly(unittest.TestCase):
    """Nightly test for Inkling-NVFP4 (975B MoE), TP=8, Blackwell only.

    Runs BOTH:
    - Performance test (using NightlyBenchmarkRunner)
    - Accuracy test (using run_eval with gsm8k)
    """

    @unittest.skipIf(not is_blackwell_system(), "NVFP4 requires Blackwell")
    def test_inkling_nvfp4(self):
        """Run performance and accuracy for Inkling-NVFP4 (TP8)."""
        variants = [
            ModelLaunchSettings(
                INKLING_NVFP4_MODEL,
                tp_size=8,
                extra_args=NVFP4_ARGS,
                variant="TP8",
            ),
        ]

        with envs.SGLANG_ENABLE_UNIFIED_RADIX_TREE.override(1):
            run_combined_tests(
                models=variants,
                test_name="Inkling-NVFP4",
                accuracy_params=AccuracyTestParams(
                    dataset="gsm8k",
                    baseline_accuracy=GSM8K_BASELINE,
                    num_examples=1314,
                    num_threads=512,
                    max_tokens=16000,
                    temperature=1.0,
                    top_p=0.95,
                    repeat=1,
                ),
                performance_params=PerformanceTestParams(
                    result_dir="performance_results_inkling_nvfp4",
                ),
            )


class TestInklingSmallCacheConsistencyNightly(unittest.TestCase):
    """Bitwise version of the per-commit check in
    ``test/registered/models_e2e/test_inkling.py``, on the real checkpoint and
    at a batch shape the tiny checkpoint never reaches."""

    @unittest.skipIf(not is_blackwell_system(), "NVFP4 requires Blackwell")
    def test_scored_contexts_are_bitwise_identical(self):
        with envs.SGLANG_ENABLE_UNIFIED_RADIX_TREE.override(1):
            with get_jitter_engine(
                model_path=INKLING_SMALL_NVFP4_MODEL,
                tp_size=8,
                trust_remote_code=True,
                quantization="modelopt_fp4",
                attention_backend="fa4",
                page_size=128,
                fp4_gemm_runner_backend="flashinfer_trtllm",
                moe_runner_backend="flashinfer_trtllm_routed",
                enable_torch_symm_mem=True,
                mamba_radix_cache_strategy="extra_buffer",
                swa_full_tokens_ratio=0.1,
                mamba_full_memory_ratio=0.1,
                mem_fraction_static=0.6,
                enable_deterministic_inference=True,
                # The harness pins a tight pool; this checkpoint needs room for
                # the 2048-4096 token prefixes the default workload draws.
                max_total_tokens=131_072,
            ) as engine:
                run_jitter_test(engine)


if __name__ == "__main__":
    unittest.main()
