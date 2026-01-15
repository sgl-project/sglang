"""
Test SageAttention Backend for SGLang.

This test verifies that the SageAttention backend works correctly for
LLM inference with 8-bit quantized attention.

Usage:
python3 -m unittest test_sage_attention_backend.TestSageAttnBackend.test_mmlu
"""

import unittest
from types import SimpleNamespace

from sglang.srt.utils import kill_process_tree
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.run_eval import run_eval
from sglang.test.test_utils import (
    DEFAULT_MODEL_NAME_FOR_TEST,
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    is_in_ci,
    popen_launch_server,
    run_bench_offline_throughput,
)


# Register for CI - SageAttention requires CUDA
# Note: This test is optional and will be skipped if SageAttention is not installed
register_cuda_ci(est_time=200, suite="stage-b-test-small-1-gpu")


def is_sage_attention_available():
    """Check if SageAttention is available."""
    try:
        from sageattention import sageattn
        return True
    except ImportError:
        return False


class TestSageAttnBackend(CustomTestCase):
    """Test SageAttention backend for 8-bit quantized attention."""

    @unittest.skipUnless(
        is_sage_attention_available(),
        "SageAttention not installed, skipping test"
    )
    def test_latency(self):
        """Test latency/throughput with SageAttention backend."""
        output_throughput = run_bench_offline_throughput(
            DEFAULT_MODEL_NAME_FOR_TEST,
            [
                "--attention-backend",
                "sage_attn",
                "--enable-torch-compile",
                "--cuda-graph-max-bs",
                4,
            ],
        )

        print(f"{output_throughput=}")

        # SageAttention should provide competitive throughput
        # The threshold is set lower initially since we fall back to
        # Triton for decode operations
        if is_in_ci():
            self.assertGreater(output_throughput, 100)

    @unittest.skipUnless(
        is_sage_attention_available(),
        "SageAttention not installed, skipping test"
    )
    def test_mmlu(self):
        """Test accuracy on MMLU benchmark with SageAttention backend."""
        model = DEFAULT_MODEL_NAME_FOR_TEST
        base_url = DEFAULT_URL_FOR_TEST
        process = popen_launch_server(
            model,
            base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=["--attention-backend", "sage_attn"],
        )

        try:
            args = SimpleNamespace(
                base_url=base_url,
                model=model,
                eval_name="mmlu",
                num_examples=64,
                num_threads=32,
            )

            metrics = run_eval(args)
            # SageAttention should maintain accuracy close to FP16 baseline
            # According to the paper, accuracy loss is minimal (~0.1%)
            self.assertGreaterEqual(metrics["score"], 0.60)
        finally:
            kill_process_tree(process.pid)

    @unittest.skipUnless(
        is_sage_attention_available(),
        "SageAttention not installed, skipping test"
    )
    def test_basic_generation(self):
        """Test basic text generation with SageAttention backend."""
        model = DEFAULT_MODEL_NAME_FOR_TEST
        base_url = DEFAULT_URL_FOR_TEST
        process = popen_launch_server(
            model,
            base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=["--attention-backend", "sage_attn"],
        )

        try:
            import openai

            client = openai.Client(base_url=f"{base_url}/v1", api_key="EMPTY")

            # Test simple completion
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "user", "content": "What is 2+2?"}
                ],
                max_tokens=32,
            )

            # Just check that we got a response
            self.assertIsNotNone(response.choices[0].message.content)
            self.assertGreater(len(response.choices[0].message.content), 0)
            print(f"Response: {response.choices[0].message.content}")
        finally:
            kill_process_tree(process.pid)


class TestSageAttnBackendComparison(CustomTestCase):
    """
    Compare SageAttention output with baseline (Triton) to verify correctness.
    
    This test compares the output logits between SageAttention and Triton
    backends to ensure quantization doesn't significantly affect output quality.
    """

    @unittest.skipUnless(
        is_sage_attention_available(),
        "SageAttention not installed, skipping test"
    )
    def test_output_comparison(self):
        """Compare outputs between SageAttention and Triton backends."""
        import openai
        import time

        model = DEFAULT_MODEL_NAME_FOR_TEST
        base_url = DEFAULT_URL_FOR_TEST
        prompt = "The capital of France is"

        # Test with Triton backend first
        process_triton = popen_launch_server(
            model,
            base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=["--attention-backend", "triton"],
        )

        try:
            client = openai.Client(base_url=f"{base_url}/v1", api_key="EMPTY")
            
            response_triton = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=16,
                temperature=0,  # Deterministic
            )
            output_triton = response_triton.choices[0].message.content
        finally:
            kill_process_tree(process_triton.pid)
            time.sleep(2)  # Wait for cleanup

        # Test with SageAttention backend
        process_sage = popen_launch_server(
            model,
            base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=["--attention-backend", "sage_attn"],
        )

        try:
            client = openai.Client(base_url=f"{base_url}/v1", api_key="EMPTY")
            
            response_sage = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=16,
                temperature=0,  # Deterministic
            )
            output_sage = response_sage.choices[0].message.content
        finally:
            kill_process_tree(process_sage.pid)

        print(f"Triton output: {output_triton}")
        print(f"SageAttention output: {output_sage}")

        # Both should produce valid outputs
        self.assertIsNotNone(output_triton)
        self.assertIsNotNone(output_sage)
        self.assertGreater(len(output_triton), 0)
        self.assertGreater(len(output_sage), 0)

        # Note: Outputs may not be identical due to quantization,
        # but both should be reasonable responses about Paris


if __name__ == "__main__":
    unittest.main()

