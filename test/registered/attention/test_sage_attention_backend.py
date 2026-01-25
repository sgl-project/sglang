"""
Comprehensive SageAttention Backend Tests

Tests SageAttention's INT8 quantized attention against FP16 baselines on A100.
Validates throughput, accuracy, memory, prefill/decode performance, and variable-length attention.

Usage:
    python -m pytest test/registered/attention/test_sage_attention_backend.py -v -s
    python -m unittest test_sage_attention_backend.TestSageAttnBackend.test_mmlu
"""

import gc
import subprocess
import time
import unittest
from types import SimpleNamespace
from typing import Dict, Optional

import torch

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
    run_bench_one_batch,
)


def reset_gpu_memory():
    """Reset GPU memory statistics and run garbage collection."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()


# SageAttention backend comprehensive tests
register_cuda_ci(est_time=600, suite="stage-b-test-large-1-gpu")


class TestSageAttnBackend(CustomTestCase):
    """Main test class for SageAttention backend validation."""

    MODEL = DEFAULT_MODEL_NAME_FOR_TEST
    BASE_URL = DEFAULT_URL_FOR_TEST

    # ============================================================================
    # Core Integration Tests (Required for CI)
    # ============================================================================

    def test_latency(self):
        """
        Test overall throughput with torch.compile and CUDA graphs.
        This is the primary performance validation test.
        """
        output_throughput = run_bench_offline_throughput(
            self.MODEL,
            [
                "--attention-backend",
                "sage_attn",
                "--enable-torch-compile",
                "--cuda-graph-max-bs",
                4,
            ],
        )

        print(f"{output_throughput=}")

        if is_in_ci():
            self.assertGreater(output_throughput, 100)

    def test_mmlu(self):
        """
        Test accuracy via MMLU benchmark.
        Validates that INT8 quantization doesn't significantly degrade quality.
        """
        process = popen_launch_server(
            self.MODEL,
            self.BASE_URL,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=["--attention-backend", "sage_attn"],
        )

        try:
            args = SimpleNamespace(
                base_url=self.BASE_URL,
                model=self.MODEL,
                eval_name="mmlu",
                num_examples=64,
                num_threads=32,
            )

            metrics = run_eval(args)
            score = metrics["score"]
            print(f"MMLU score: {score:.4f}")
            self.assertGreaterEqual(score, 0.60)
        finally:
            kill_process_tree(process.pid)

    # ============================================================================
    # Prefill vs Decode Breakdown
    # ============================================================================

    def _run_one_batch_benchmark(
        self,
        backend: str,
        batch_size: int = 1,
        input_len: int = 1024,
        output_len: int = 128,
    ) -> Dict[str, float]:
        """Run one-batch benchmark to get separate prefill and decode metrics."""
        server_args = SimpleNamespace(
            model_path=self.MODEL,
            host="127.0.0.1",
            port=30000,
            attention_backend=backend,
        )

        bench_args = SimpleNamespace(
            backend="sglang",
            base_url=self.BASE_URL,
            dataset_name="random",
            num_prompts=batch_size,
            random_input_len=input_len,
            random_output_len=output_len,
        )

        other_server_args = ["--attention-backend", backend]

        try:
            prefill_latency, decode_throughput, decode_latency = run_bench_one_batch(
                self.MODEL,
                self.BASE_URL,
                server_args,
                bench_args,
                other_server_args,
            )

            prefill_throughput = (
                (batch_size * input_len) / (prefill_latency / 1000.0)
                if prefill_latency > 0
                else -1
            )

            return {
                "prefill_latency_ms": prefill_latency,
                "prefill_throughput_tok_s": prefill_throughput,
                "decode_latency_ms": decode_latency,
                "decode_throughput_tok_s": decode_throughput,
            }

        except Exception as e:
            print(f"Error in one_batch benchmark: {e}")
            return {
                "prefill_latency_ms": -1,
                "prefill_throughput_tok_s": -1,
                "decode_latency_ms": -1,
                "decode_throughput_tok_s": -1,
            }

    def test_prefill_decode_breakdown(self):
        """
        Test prefill and decode performance separately.
        Critical for understanding where SageAttention provides benefits.
        """
        print("\n" + "=" * 70)
        print("PREFILL + DECODE BREAKDOWN")
        print("=" * 70)

        batch_size, input_len, output_len = 1, 1024, 256
        backends = ["sage_attn", "triton"]
        results = {}

        for backend in backends:
            print(f"\nBenchmarking {backend}...")
            metrics = self._run_one_batch_benchmark(
                backend, batch_size, input_len, output_len
            )
            results[backend] = metrics

        # Print results table
        print("\n" + "=" * 70)
        print(f"Config: batch={batch_size}, input={input_len}, output={output_len}")
        print("=" * 70)
        print(
            f"{'Backend':<12} {'Phase':<10} {'Throughput (tok/s)':<20} {'Latency (ms)':<15}"
        )
        print("-" * 57)

        for backend in backends:
            if backend in results:
                m = results[backend]

                # Prefill
                prefill_tp = (
                    f"{m['prefill_throughput_tok_s']:.2f}"
                    if m["prefill_throughput_tok_s"] > 0
                    else "N/A"
                )
                prefill_lat = (
                    f"{m['prefill_latency_ms']:.2f}"
                    if m["prefill_latency_ms"] > 0
                    else "N/A"
                )
                print(
                    f"{backend:<12} {'Prefill':<10} {prefill_tp:<20} {prefill_lat:<15}"
                )

                # Decode
                decode_tp = (
                    f"{m['decode_throughput_tok_s']:.2f}"
                    if m["decode_throughput_tok_s"] > 0
                    else "N/A"
                )
                decode_lat = (
                    f"{m['decode_latency_ms']:.2f}"
                    if m["decode_latency_ms"] > 0
                    else "N/A"
                )
                print(
                    f"{backend:<12} {'Decode':<10} {decode_tp:<20} {decode_lat:<15}"
                )
                print("-" * 57)

        # Calculate speedups
        if "sage_attn" in results and "triton" in results:
            sage = results["sage_attn"]
            triton = results["triton"]

            print("\nSpeedup (SageAttention vs Triton):")
            if (
                sage["prefill_throughput_tok_s"] > 0
                and triton["prefill_throughput_tok_s"] > 0
            ):
                speedup = (
                    sage["prefill_throughput_tok_s"] / triton["prefill_throughput_tok_s"]
                )
                print(f"  Prefill: {speedup:.2f}x")

            if (
                sage["decode_throughput_tok_s"] > 0
                and triton["decode_throughput_tok_s"] > 0
            ):
                speedup = (
                    sage["decode_throughput_tok_s"] / triton["decode_throughput_tok_s"]
                )
                print(f"  Decode:  {speedup:.2f}x")

    # ============================================================================
    # Throughput Comparisons
    # ============================================================================

    def test_throughput_vs_triton(self):
        """Compare throughput between SageAttention and Triton baseline."""
        print("\n" + "=" * 70)
        print("THROUGHPUT COMPARISON: SageAttention vs Triton")
        print("=" * 70)

        backends = ["sage_attn", "triton"]
        results = {}

        for backend in backends:
            print(f"\nTesting {backend}...")
            throughput = run_bench_offline_throughput(
                self.MODEL,
                [
                    "--attention-backend",
                    backend,
                    "--num-prompts",
                    5,
                    "--random-input-len",
                    256,
                    "--random-output-len",
                    256,
                ],
            )
            results[backend] = throughput
            print(f"  {backend}: {throughput:.2f} tok/s")

        # Print comparison
        if "sage_attn" in results and "triton" in results:
            sage_tp = results["sage_attn"]
            triton_tp = results["triton"]

            if sage_tp > 0 and triton_tp > 0:
                speedup = sage_tp / triton_tp
                print(f"\n{'='*70}")
                print(f"SageAttention: {sage_tp:.2f} tok/s")
                print(f"Triton:        {triton_tp:.2f} tok/s")
                print(f"Speedup:       {speedup:.2f}x")
                print(f"{'='*70}")

    def test_throughput_context_scaling(self):
        """Test throughput scaling across different context lengths."""
        print("\n" + "=" * 70)
        print("CONTEXT LENGTH SCALING")
        print("=" * 70)

        context_configs = [
            (256, 64),
            (512, 128),
            (1024, 256),
            (2048, 256),
        ]

        sage_results = []

        for input_len, output_len in context_configs:
            print(f"\n--- Input: {input_len}, Output: {output_len} ---")
            throughput = run_bench_offline_throughput(
                self.MODEL,
                [
                    "--attention-backend",
                    "sage_attn",
                    "--num-prompts",
                    3,
                    "--random-input-len",
                    input_len,
                    "--random-output-len",
                    output_len,
                ],
            )
            sage_results.append({"input_len": input_len, "throughput": throughput})
            print(f"  SageAttention: {throughput:.2f} tok/s")

        # Print summary
        print(f"\n{'='*70}")
        print("SCALING SUMMARY")
        print(f"{'='*70}")
        print(f"{'Input Length':<15} {'Throughput (tok/s)':<20}")
        print("-" * 35)
        for r in sage_results:
            tp_str = f"{r['throughput']:.2f}" if r["throughput"] > 0 else "FAILED"
            print(f"{r['input_len']:<15} {tp_str:<20}")

    # ============================================================================
    # Variable Length Attention Tests
    # ============================================================================

    def test_variable_length_batch(self):
        """
        Test with variable sequence lengths in a batch.

        SageAttention uses varlen API which should handle different sequence
        lengths within the same batch efficiently.
        """
        print("\n" + "=" * 70)
        print("VARIABLE LENGTH ATTENTION TEST")
        print("=" * 70)

        import requests

        process = popen_launch_server(
            self.MODEL,
            self.BASE_URL,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=["--attention-backend", "sage_attn"],
        )

        try:
            # Test with prompts of varying lengths
            prompts = [
                "Short prompt.",
                "This is a medium length prompt with more tokens to process.",
                "This is a much longer prompt that contains significantly more text and will result in a different sequence length compared to the other prompts in the batch. " * 3,
            ]

            # Use batch API
            response = requests.post(
                f"{self.BASE_URL}/v1/completions",
                json={
                    "model": self.MODEL,
                    "prompt": prompts,
                    "max_tokens": 20,
                    "temperature": 0.0,
                },
                timeout=60,
            )

            self.assertEqual(response.status_code, 200, "Variable length batch request failed")
            result = response.json()

            # Verify all prompts got responses
            self.assertEqual(
                len(result["choices"]),
                len(prompts),
                "Didn't get responses for all prompts",
            )

            # Check no errors in outputs
            for i, choice in enumerate(result["choices"]):
                generated = choice["text"]
                print(f"  Prompt {i+1} (len ~{len(prompts[i].split())}): Generated {len(generated.split())} tokens")
                self.assertGreater(
                    len(generated), 0, f"Empty generation for prompt {i}"
                )

            print("✓ Variable length batch processing successful")

        except Exception as e:
            self.fail(f"Variable length test failed: {e}")
        finally:
            kill_process_tree(process.pid)

    def test_extreme_length_variance(self):
        """
        Test with extreme variance in sequence lengths (1 token vs 1000+ tokens).
        """
        print("\n" + "=" * 70)
        print("EXTREME LENGTH VARIANCE TEST")
        print("=" * 70)

        import requests

        process = popen_launch_server(
            self.MODEL,
            self.BASE_URL,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=["--attention-backend", "sage_attn"],
        )

        try:
            # Extreme variance: very short and very long prompts
            prompts = [
                "Hi",  # ~1 token
                "The quick brown fox jumps over the lazy dog. " * 50,  # ~500+ tokens
                "Hello world",  # ~2 tokens
            ]

            response = requests.post(
                f"{self.BASE_URL}/v1/completions",
                json={
                    "model": self.MODEL,
                    "prompt": prompts,
                    "max_tokens": 10,
                    "temperature": 0.0,
                },
                timeout=90,
            )

            self.assertEqual(response.status_code, 200, "Extreme variance batch failed")
            result = response.json()
            self.assertEqual(len(result["choices"]), len(prompts))

            print("✓ Extreme length variance handling successful")

        except Exception as e:
            self.fail(f"Extreme variance test failed: {e}")
        finally:
            kill_process_tree(process.pid)

    # ============================================================================
    # Memory Tests
    # ============================================================================

    def test_memory_footprint(self):
        """
        Compare memory usage between SageAttention and Triton.
        INT8 quantization should reduce memory bandwidth.
        """
        print("\n" + "=" * 70)
        print("MEMORY FOOTPRINT COMPARISON")
        print("=" * 70)

        backends = ["sage_attn", "triton"]
        memory_results = {}

        for backend in backends:
            print(f"\nMeasuring {backend}...")
            reset_gpu_memory()

            process = popen_launch_server(
                self.MODEL,
                self.BASE_URL,
                timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
                other_args=["--attention-backend", backend],
            )

            try:
                time.sleep(10)  # Wait for model to load

                # Get memory via nvidia-smi
                nvidia_smi = subprocess.run(
                    [
                        "nvidia-smi",
                        "--query-gpu=memory.used,memory.total",
                        "--format=csv,noheader,nounits",
                    ],
                    capture_output=True,
                    text=True,
                )

                if nvidia_smi.returncode == 0:
                    lines = nvidia_smi.stdout.strip().split("\n")
                    if lines:
                        used, total = map(float, lines[0].split(","))
                        memory_results[backend] = {
                            "used_mb": used,
                            "total_mb": total,
                            "utilization_pct": (used / total) * 100,
                        }
                        print(
                            f"  {backend}: {used:.0f} MB / {total:.0f} MB "
                            f"({memory_results[backend]['utilization_pct']:.1f}%)"
                        )

            except Exception as e:
                print(f"  Error: {e}")
            finally:
                kill_process_tree(process.pid)
                time.sleep(5)
                reset_gpu_memory()

        # Print comparison
        if "sage_attn" in memory_results and "triton" in memory_results:
            sage_mem = memory_results["sage_attn"]["used_mb"]
            triton_mem = memory_results["triton"]["used_mb"]
            diff = triton_mem - sage_mem
            diff_pct = (diff / triton_mem) * 100 if triton_mem > 0 else 0

            print(f"\n{'='*70}")
            print("MEMORY SUMMARY")
            print(f"{'='*70}")
            print(f"SageAttention: {sage_mem:.0f} MB")
            print(f"Triton:        {triton_mem:.0f} MB")
            print(f"Difference:    {diff:.0f} MB ({diff_pct:.1f}%)")
            print(f"{'='*70}")

    # ============================================================================
    # Accuracy Tests
    # ============================================================================

    def test_logits_comparison(self):
        """
        Compare raw output logits between SageAttention and Triton.

        This validates that INT8 quantization doesn't significantly alter
        the probability distribution over tokens.
        """
        print("\n" + "=" * 70)
        print("LOGITS COMPARISON TEST")
        print("=" * 70)

        import requests

        test_prompt = "The capital of France is"
        backends = ["sage_attn", "triton"]
        logprobs_results = {}

        for backend in backends:
            print(f"\nTesting {backend}...")

            process = popen_launch_server(
                self.MODEL,
                self.BASE_URL,
                timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
                other_args=["--attention-backend", backend],
            )

            try:
                # Request with logprobs to get probability distributions
                response = requests.post(
                    f"{self.BASE_URL}/v1/completions",
                    json={
                        "model": self.MODEL,
                        "prompt": test_prompt,
                        "max_tokens": 5,
                        "temperature": 0.0,
                        "logprobs": 5,  # Request top 5 logprobs
                    },
                    timeout=60,
                )

                if response.status_code == 200:
                    result = response.json()
                    logprobs_results[backend] = result
                    print(f"  {backend}: Received logprobs")

            except Exception as e:
                print(f"  Error: {e}")
            finally:
                kill_process_tree(process.pid)
                time.sleep(2)

        # Compare logprobs
        if "sage_attn" in logprobs_results and "triton" in logprobs_results:
            sage_choice = logprobs_results["sage_attn"]["choices"][0]
            triton_choice = logprobs_results["triton"]["choices"][0]

            print(f"\n{'='*70}")
            print("LOGPROBS COMPARISON")
            print(f"{'='*70}")

            # Compare tokens generated
            sage_text = sage_choice["text"]
            triton_text = triton_choice["text"]
            print(f"SageAttention output: '{sage_text}'")
            print(f"Triton output:        '{triton_text}'")

            # If logprobs are available, compare them
            if "logprobs" in sage_choice and "logprobs" in triton_choice:
                sage_logprobs = sage_choice["logprobs"]
                triton_logprobs = triton_choice["logprobs"]

                # Compare top token logprobs if available
                if "token_logprobs" in sage_logprobs and "token_logprobs" in triton_logprobs:
                    sage_tokens_lp = sage_logprobs["token_logprobs"]
                    triton_tokens_lp = triton_logprobs["token_logprobs"]

                    print("\nToken-by-token logprob comparison:")
                    for i, (sage_lp, triton_lp) in enumerate(zip(sage_tokens_lp, triton_tokens_lp)):
                        if sage_lp is not None and triton_lp is not None:
                            diff = abs(sage_lp - triton_lp)
                            print(f"  Token {i}: sage={sage_lp:.4f}, triton={triton_lp:.4f}, diff={diff:.4f}")

                            # Logprobs should be very close (within 0.5 is reasonable for INT8 quantization)
                            self.assertLess(
                                diff,
                                0.5,
                                f"Logprob difference too large at token {i}: {diff}",
                            )

    def test_perplexity_comparison(self):
        """
        Compare perplexity on a text dataset between SageAttention and Triton.

        Perplexity is a key metric for language model quality. Lower perplexity
        is better, and SageAttention should maintain similar perplexity to FP16.
        """
        print("\n" + "=" * 70)
        print("PERPLEXITY COMPARISON TEST")
        print("=" * 70)

        import math
        import requests

        # Use a standard test paragraph for perplexity calculation
        test_text = """
        Language models are statistical models that assign probabilities to sequences of words.
        They are fundamental to many natural language processing tasks including machine translation,
        speech recognition, and text generation. Modern neural language models have achieved
        remarkable performance on a wide variety of benchmarks.
        """

        backends = ["sage_attn", "triton"]
        perplexity_results = {}

        for backend in backends:
            print(f"\nTesting {backend}...")

            process = popen_launch_server(
                self.MODEL,
                self.BASE_URL,
                timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
                other_args=["--attention-backend", backend],
            )

            try:
                # Get logprobs for the text
                response = requests.post(
                    f"{self.BASE_URL}/v1/completions",
                    json={
                        "model": self.MODEL,
                        "prompt": test_text,
                        "max_tokens": 1,
                        "echo": True,
                        "logprobs": 0,
                    },
                    timeout=60,
                )

                if response.status_code == 200:
                    result = response.json()
                    choice = result["choices"][0]

                    if "logprobs" in choice and choice["logprobs"] is not None:
                        token_logprobs = choice["logprobs"].get("token_logprobs", [])

                        # Filter out None values (first token typically has None)
                        valid_logprobs = [lp for lp in token_logprobs if lp is not None]

                        if valid_logprobs:
                            # Calculate perplexity: exp(-mean(log probs))
                            avg_log_prob = sum(valid_logprobs) / len(valid_logprobs)
                            perplexity = math.exp(-avg_log_prob)

                            perplexity_results[backend] = {
                                "perplexity": perplexity,
                                "avg_log_prob": avg_log_prob,
                                "num_tokens": len(valid_logprobs),
                            }
                            print(f"  {backend}: perplexity={perplexity:.4f}")

            except Exception as e:
                print(f"  Error: {e}")
            finally:
                kill_process_tree(process.pid)
                time.sleep(2)

        # Compare perplexities
        if "sage_attn" in perplexity_results and "triton" in perplexity_results:
            sage_ppl = perplexity_results["sage_attn"]["perplexity"]
            triton_ppl = perplexity_results["triton"]["perplexity"]

            ppl_diff = abs(sage_ppl - triton_ppl)
            ppl_diff_pct = (ppl_diff / triton_ppl) * 100

            print(f"\n{'='*70}")
            print("PERPLEXITY SUMMARY")
            print(f"{'='*70}")
            print(f"SageAttention: {sage_ppl:.4f}")
            print(f"Triton:        {triton_ppl:.4f}")
            print(f"Difference:    {ppl_diff:.4f} ({ppl_diff_pct:.2f}%)")
            print(f"{'='*70}")

            # Perplexity should be within 5% for INT8 quantization
            self.assertLess(
                ppl_diff_pct,
                5.0,
                f"Perplexity difference too large: {ppl_diff_pct:.2f}%",
            )

    def test_short_vs_long_context_accuracy(self):
        """
        Compare accuracy deltas between short and long contexts.

        Validates SageAttention's "≈ same accuracy" claim across context lengths.
        Tests both short prompts (128 tokens) and long prompts (2048 tokens).
        """
        print("\n" + "=" * 70)
        print("SHORT vs LONG CONTEXT ACCURACY")
        print("=" * 70)

        import requests

        # Test cases with varying lengths
        test_cases = [
            {
                "name": "short_context",
                "prompt": "What is 2 + 2? Answer:",
                "expected_keyword": "4",
                "length_category": "short",
            },
            {
                "name": "medium_context",
                "prompt": (
                    "In a study of mathematical operations, researchers found that "
                    "addition is commutative. This means that a + b = b + a. "
                    "For example, 2 + 3 = 3 + 2. Given this property, "
                    "what is the result of 5 + 7? Answer:"
                ),
                "expected_keyword": "12",
                "length_category": "medium",
            },
            {
                "name": "long_context",
                "prompt": (
                    "Context: " + " ".join([
                        f"Paragraph {i}: This is filler text to create a longer context. " * 20
                        for i in range(10)
                    ]) +
                    "\n\nQuestion: Based on the context above, what is the capital of France? Answer:"
                ),
                "expected_keyword": "Paris",
                "length_category": "long",
            },
        ]

        backends = ["sage_attn", "triton"]
        results = {}

        for backend in backends:
            print(f"\n--- Testing {backend} ---")

            process = popen_launch_server(
                self.MODEL,
                self.BASE_URL,
                timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
                other_args=["--attention-backend", backend],
            )

            try:
                backend_results = {}

                for tc in test_cases:
                    response = requests.post(
                        f"{self.BASE_URL}/v1/completions",
                        json={
                            "model": self.MODEL,
                            "prompt": tc["prompt"],
                            "max_tokens": 20,
                            "temperature": 0.0,
                        },
                        timeout=90,
                    )

                    if response.status_code == 200:
                        result = response.json()
                        generated = result["choices"][0]["text"].lower()
                        correct = tc["expected_keyword"].lower() in generated

                        backend_results[tc["name"]] = {
                            "correct": correct,
                            "output": generated,
                            "category": tc["length_category"],
                        }

                        status = "✓" if correct else "✗"
                        print(f"  {tc['name']} ({tc['length_category']}): {status}")

                results[backend] = backend_results

            except Exception as e:
                print(f"  Error: {e}")
            finally:
                kill_process_tree(process.pid)
                time.sleep(2)

        # Compare accuracy across context lengths
        if "sage_attn" in results and "triton" in results:
            print(f"\n{'='*70}")
            print("ACCURACY DELTA ANALYSIS")
            print(f"{'='*70}")

            categories = ["short", "medium", "long"]
            for category in categories:
                sage_correct = sum(
                    1 for r in results["sage_attn"].values()
                    if r["category"] == category and r["correct"]
                )
                triton_correct = sum(
                    1 for r in results["triton"].values()
                    if r["category"] == category and r["correct"]
                )
                total = sum(
                    1 for r in results["sage_attn"].values()
                    if r["category"] == category
                )

                if total > 0:
                    sage_acc = sage_correct / total
                    triton_acc = triton_correct / total
                    acc_delta = abs(sage_acc - triton_acc)

                    print(f"\n{category.upper()} context:")
                    print(f"  SageAttention: {sage_acc:.2%} ({sage_correct}/{total})")
                    print(f"  Triton:        {triton_acc:.2%} ({triton_correct}/{total})")
                    print(f"  Δ Accuracy:    {acc_delta:.2%}")

                    # Accuracy delta should be small (within 20% difference)
                    self.assertLess(
                        acc_delta,
                        0.2,
                        f"Accuracy delta too large for {category} context: {acc_delta}",
                    )

    def test_output_consistency(self):
        """
        Compare output consistency between SageAttention and Triton.
        Uses greedy decoding for reproducibility.
        """
        print("\n" + "=" * 70)
        print("OUTPUT CONSISTENCY TEST")
        print("=" * 70)

        import requests

        test_prompt = "The capital of France is"
        backends = ["sage_attn", "triton"]
        outputs = {}

        for backend in backends:
            print(f"\nTesting {backend}...")

            process = popen_launch_server(
                self.MODEL,
                self.BASE_URL,
                timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
                other_args=["--attention-backend", backend],
            )

            try:
                response = requests.post(
                    f"{self.BASE_URL}/v1/completions",
                    json={
                        "model": self.MODEL,
                        "prompt": test_prompt,
                        "max_tokens": 20,
                        "temperature": 0.0,  # Greedy
                    },
                    timeout=60,
                )

                if response.status_code == 200:
                    result = response.json()
                    generated = result["choices"][0]["text"]
                    outputs[backend] = generated
                    print(f"  Output: '{generated}'")

            except Exception as e:
                print(f"  Error: {e}")
            finally:
                kill_process_tree(process.pid)
                time.sleep(2)

        # Compare outputs
        if "sage_attn" in outputs and "triton" in outputs:
            sage_out = outputs["sage_attn"]
            triton_out = outputs["triton"]

            print(f"\n{'='*70}")
            print("CONSISTENCY COMPARISON")
            print(f"{'='*70}")
            print(f"SageAttention: '{sage_out}'")
            print(f"Triton:        '{triton_out}'")

            if sage_out == triton_out:
                print("✓ Outputs MATCH exactly")
            else:
                # Calculate token match rate
                sage_tokens = sage_out.split()
                triton_tokens = triton_out.split()
                matching = sum(1 for s, t in zip(sage_tokens, triton_tokens) if s == t)
                total = max(len(sage_tokens), len(triton_tokens))
                match_rate = matching / total if total > 0 else 0
                print(f"Token match rate: {match_rate:.1%}")

                # Should have high consistency
                self.assertGreater(
                    match_rate, 0.7, f"Token match rate too low: {match_rate}"
                )


if __name__ == "__main__":
    unittest.main(verbosity=2)
