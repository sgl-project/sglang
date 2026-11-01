"""Check MoE LoRA text and logprobs against the recorded vLLM reference."""

import math
import multiprocessing as mp
import unittest

import torch

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.lora_utils import (
    MOE_BASE_MODEL_PATH,
    MOE_LORA_PATH,
    MOE_LORA_TEST_PROMPTS,
    MOE_LORA_VLLM_REFERENCE,
    MOE_LORA_VLLM_REFERENCE_STATS,
)
from sglang.test.runners import SRTRunner

register_cuda_ci(
    est_time=100,
    stage="base-b",
    runner_config="1-gpu-large",
)


class TestMoELoraRegression(unittest.TestCase):
    def test_sglang_moe_parity_flashinfer(self):
        # flashinfer is CUDA-only.
        from sglang.srt.utils import is_cuda

        if not is_cuda():
            self.skipTest("flashinfer backend requires CUDA")
        self._run_parity_strict(attention_backend="flashinfer")

    def test_sglang_moe_parity_flashinfer_virtual_experts(self):
        """Fold expert LoRA into the base GEMM as virtual experts. Different
        accumulation order needs its own bounds: on GB300 all vLLM strings
        matched, with max/mean logprob errors <= 0.12/0.017. Bounds 0.5/0.05
        still reject missing/misapplied adapters (7-17 shifts and wrong text)."""
        from sglang.srt.utils import is_cuda

        if not is_cuda():
            self.skipTest("flashinfer backend requires CUDA")
        self._run_parity_strict(
            attention_backend="flashinfer",
            logprob_bounds=(0.5, 0.05),
            lora_use_virtual_experts=True,
        )

    def test_sglang_moe_parity_intel_xpu(self):
        from sglang.srt.utils import is_xpu

        if not is_xpu():
            self.skipTest("intel_xpu backend requires XPU")
        self._run_parity_strict(attention_backend="intel_xpu")

    def test_sglang_moe_parity_triton(self):
        from sglang.srt.utils import is_cuda, is_xpu

        if is_cuda() or is_xpu():
            self.skipTest("triton backend is the fallback for non-CUDA/non-XPU")
        self._run_parity_strict(attention_backend="triton")

    def _run_parity_strict(
        self, *, attention_backend, logprob_bounds=None, **runner_kwargs
    ):
        """Match vLLM text and recorded stats + eps, or absolute (max, mean) bounds."""

        with SRTRunner(
            model_path=MOE_BASE_MODEL_PATH,
            torch_dtype=torch.bfloat16,
            model_type="generation",
            lora_paths=[MOE_LORA_PATH],
            max_loras_per_batch=1,
            tp_size=1,
            trust_remote_code=True,
            disable_radix_cache=True,
            attention_backend=attention_backend,
            mem_fraction_static=0.80,
            **runner_kwargs,
        ) as srt_runner:
            srt_outputs = srt_runner.forward(
                MOE_LORA_TEST_PROMPTS,
                max_new_tokens=10,
                lora_paths=[MOE_LORA_PATH] * len(MOE_LORA_TEST_PROMPTS),
            )

        print("\n" + "=" * 140)
        print(
            f"{'ID':<4} | {'Max Diff':<12} | {'Mean Diff':<12} | {'Status':<8} | {'Prompt'}"
        )
        print("-" * 140)

        for i, prompt in enumerate(MOE_LORA_TEST_PROMPTS):
            v_data = MOE_LORA_VLLM_REFERENCE[i]
            v_lps = v_data["lps"]
            v_text = v_data["text"].strip()

            s_lps_raw = srt_outputs.top_output_logprobs[i]
            s_lps = [
                float(token[0]) if isinstance(token, list) else float(token)
                for token in s_lps_raw
            ]
            s_text = srt_outputs.output_strs[i].strip()

            self.assertEqual(v_text, s_text, f"String mismatch on prompt {i}")
            # Both sides carry one logprob per generated token, including the
            # EOS token that ends a short answer, so the comparison must cover
            # the whole reference: a missing or truncated result fails instead
            # of shrinking the comparison.
            self.assertTrue(s_lps, f"No output logprobs on prompt {i}")
            self.assertEqual(
                len(s_lps), len(v_lps), f"Logprob count mismatch on prompt {i}"
            )
            self.assertTrue(
                all(map(math.isfinite, s_lps)), f"Non-finite logprob on prompt {i}"
            )
            diffs = [abs(v - s) for v, s in zip(v_lps, s_lps)]
            actual_max = max(diffs)
            actual_mean = sum(diffs) / len(diffs)

            if logprob_bounds is None:
                ref = MOE_LORA_VLLM_REFERENCE_STATS[i]
                # Epsilon to allow room for different, but correct, implementations
                eps = 2e-4
                max_bound, mean_bound = ref["max"] + eps, ref["mean"] + eps
            else:
                max_bound, mean_bound = logprob_bounds

            self.assertLessEqual(
                actual_max, max_bound, f"Max LogProb Diff exceeded on prompt {i}"
            )
            self.assertLessEqual(
                actual_mean, mean_bound, f"Mean LogProb Diff exceeded on prompt {i}"
            )

            print(
                f"{i:<4} | {actual_max:<12.6f} | {actual_mean:<12.6f} | {'✅ PASS':<8} | {prompt}"
            )

        print("=" * 140)


if __name__ == "__main__":
    try:
        mp.set_start_method("spawn")
    except RuntimeError:
        pass

    try:
        unittest.main(warnings="ignore", verbosity=2)
    finally:
        # Final cleanup
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
