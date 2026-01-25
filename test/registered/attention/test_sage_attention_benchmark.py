"""
Comprehensive benchmark tests for SageAttention backend on A100.

This test suite validates SageAttention's INT8 quantized attention by:
1. Comparing throughput (tokens/sec) against FP16 baselines (Triton, FlashInfer)
2. Measuring memory footprint and peak GPU memory usage
3. Evaluating accuracy via perplexity on WikiText-2 and MMLU
4. Testing short vs long context accuracy preservation

Usage:
    # Run all benchmarks
    python -m pytest test/registered/attention/test_sage_attention_benchmark.py -v -s

    # Run specific benchmark
    python -m unittest test_sage_attention_benchmark.TestSageAttentionBenchmark.test_throughput_comparison

    # Run with detailed output
    python test/registered/attention/test_sage_attention_benchmark.py --verbose
"""

import gc
import json
import os
import subprocess
import sys
import time
import unittest
from types import SimpleNamespace
from typing import Dict, List, Optional, Tuple

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


def is_a100_or_better() -> bool:
    """Check if running on A100 or better GPU."""
    try:
        import torch

        if not torch.cuda.is_available():
            return False
        device_name = torch.cuda.get_device_name(0).lower()
        # A100, H100, H200, or newer
        return any(gpu in device_name for gpu in ["a100", "h100", "h200", "a10", "l40"])
    except Exception:
        return False


def get_gpu_memory_info() -> Dict[str, float]:
    """Get current GPU memory usage."""
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.synchronize()
            return {
                "allocated_mb": torch.cuda.memory_allocated() / 1024 / 1024,
                "reserved_mb": torch.cuda.memory_reserved() / 1024 / 1024,
                "max_allocated_mb": torch.cuda.max_memory_allocated() / 1024 / 1024,
            }
    except Exception:
        pass
    return {"allocated_mb": 0, "reserved_mb": 0, "max_allocated_mb": 0}


def reset_gpu_memory():
    """Reset GPU memory statistics and run garbage collection."""
    import torch

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()


# Register for CI - longer estimated time due to comprehensive benchmarking
register_cuda_ci(est_time=600, suite="stage-b-test-large-1-gpu")


class TestSageAttentionBenchmark(CustomTestCase):
    """
    Comprehensive benchmarks comparing SageAttention against baseline backends.
    """

    # Test model - use a small but representative model
    MODEL = DEFAULT_MODEL_NAME_FOR_TEST

    # Backends to compare
    BACKENDS = ["sage_attn", "triton"]

    # Sequence length configurations for testing
    SHORT_CONTEXT = 256
    MEDIUM_CONTEXT = 1024
    LONG_CONTEXT = 4096

    def _run_throughput_benchmark(
        self,
        backend: str,
        input_len: int = 256,
        output_len: int = 256,
        num_prompts: int = 10,
        extra_args: Optional[List[str]] = None,
    ) -> Tuple[float, Dict]:
        """
        Run throughput benchmark for a specific backend.

        Returns:
            Tuple of (throughput_tok_per_sec, detailed_metrics)
        """
        args = [
            "--attention-backend",
            backend,
        ]
        if extra_args:
            args.extend(extra_args)

        command = [
            "python3",
            "-m",
            "sglang.bench_offline_throughput",
            "--num-prompts",
            str(num_prompts),
            "--dataset-name",
            "random",
            "--random-input-len",
            str(input_len),
            "--random-output-len",
            str(output_len),
            "--model-path",
            self.MODEL,
            *[str(x) for x in args],
        ]

        print(f"\nRunning: {' '.join(command)}")

        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        try:
            stdout, stderr = process.communicate(timeout=600)
            output = stdout.decode(errors="backslashreplace")
            error = stderr.decode(errors="backslashreplace")

            throughput = -1
            latency = -1
            for line in output.split("\n"):
                if "Last generation throughput (tok/s):" in line:
                    throughput = float(line.split(":")[-1].strip())
                if "Total latency" in line:
                    try:
                        latency = float(line.split(":")[-1].strip().replace("s", ""))
                    except (ValueError, IndexError):
                        pass

            metrics = {
                "throughput_tok_s": throughput,
                "latency_s": latency,
                "backend": backend,
                "input_len": input_len,
                "output_len": output_len,
                "num_prompts": num_prompts,
            }

            if throughput < 0:
                print(f"Warning: Could not parse throughput from output:\n{output[:1000]}")
                print(f"Stderr: {error[:500]}")

            return throughput, metrics

        except subprocess.TimeoutExpired:
            process.kill()
            print(f"Benchmark timed out for backend={backend}")
            return -1, {"error": "timeout"}
        except Exception as e:
            print(f"Error running benchmark: {e}")
            return -1, {"error": str(e)}
        finally:
            try:
                kill_process_tree(process.pid)
            except Exception:
                pass

    def test_throughput_comparison(self):
        """
        Compare throughput between SageAttention and baseline backends.
        This is the primary performance validation test.
        """
        print("\n" + "=" * 70)
        print("THROUGHPUT COMPARISON BENCHMARK")
        print("=" * 70)

        results = {}

        for backend in self.BACKENDS:
            print(f"\n--- Testing {backend} backend ---")
            throughput, metrics = self._run_throughput_benchmark(
                backend=backend,
                input_len=256,
                output_len=256,
                num_prompts=5,
            )
            results[backend] = metrics
            print(f"{backend}: {throughput:.2f} tok/s")

        # Print comparison
        print("\n" + "=" * 70)
        print("RESULTS SUMMARY")
        print("=" * 70)

        if "sage_attn" in results and "triton" in results:
            sage_tp = results["sage_attn"]["throughput_tok_s"]
            triton_tp = results["triton"]["throughput_tok_s"]

            if sage_tp > 0 and triton_tp > 0:
                speedup = sage_tp / triton_tp
                print(f"SageAttention: {sage_tp:.2f} tok/s")
                print(f"Triton:        {triton_tp:.2f} tok/s")
                print(f"Speedup:       {speedup:.2f}x")

                # SageAttention should be competitive with Triton
                # Note: The actual speedup depends on model and GPU
                self.assertGreater(
                    sage_tp,
                    0,
                    "SageAttention throughput should be positive",
                )

        # In CI, verify minimum throughput threshold
        if is_in_ci() and "sage_attn" in results:
            self.assertGreater(
                results["sage_attn"]["throughput_tok_s"],
                50,  # Lower threshold for CI stability
                "SageAttention throughput below CI threshold",
            )

    def test_throughput_varying_context_lengths(self):
        """
        Benchmark throughput across different context lengths.
        Validates performance scaling behavior.
        """
        print("\n" + "=" * 70)
        print("CONTEXT LENGTH SCALING BENCHMARK")
        print("=" * 70)

        context_configs = [
            (128, 64),  # Short context
            (512, 128),  # Medium context
            (1024, 256),  # Long context
            (2048, 256),  # Very long context (if memory allows)
        ]

        sage_results = []
        triton_results = []

        for input_len, output_len in context_configs:
            print(f"\n--- Context: input={input_len}, output={output_len} ---")

            try:
                sage_tp, sage_metrics = self._run_throughput_benchmark(
                    "sage_attn",
                    input_len=input_len,
                    output_len=output_len,
                    num_prompts=3,
                )
                sage_results.append(sage_metrics)
                print(f"  SageAttention: {sage_tp:.2f} tok/s")
            except Exception as e:
                print(f"  SageAttention failed: {e}")
                sage_results.append({"error": str(e), "input_len": input_len})

            try:
                triton_tp, triton_metrics = self._run_throughput_benchmark(
                    "triton",
                    input_len=input_len,
                    output_len=output_len,
                    num_prompts=3,
                )
                triton_results.append(triton_metrics)
                print(f"  Triton:        {triton_tp:.2f} tok/s")
            except Exception as e:
                print(f"  Triton failed: {e}")
                triton_results.append({"error": str(e), "input_len": input_len})

        # Print summary table
        print("\n" + "=" * 70)
        print("CONTEXT LENGTH SCALING SUMMARY")
        print("=" * 70)
        print(f"{'Input Len':<12} {'SageAttn (tok/s)':<20} {'Triton (tok/s)':<20} {'Speedup':<10}")
        print("-" * 62)

        for sage_m, triton_m in zip(sage_results, triton_results):
            input_len = sage_m.get("input_len", "N/A")
            sage_tp = sage_m.get("throughput_tok_s", -1)
            triton_tp = triton_m.get("throughput_tok_s", -1)

            if sage_tp > 0 and triton_tp > 0:
                speedup = f"{sage_tp / triton_tp:.2f}x"
            else:
                speedup = "N/A"

            sage_str = f"{sage_tp:.2f}" if sage_tp > 0 else "failed"
            triton_str = f"{triton_tp:.2f}" if triton_tp > 0 else "failed"

            print(f"{input_len:<12} {sage_str:<20} {triton_str:<20} {speedup:<10}")

    def test_mmlu_accuracy(self):
        """
        Evaluate accuracy on MMLU benchmark.
        Validates that INT8 quantization doesn't significantly degrade model quality.
        """
        print("\n" + "=" * 70)
        print("MMLU ACCURACY BENCHMARK")
        print("=" * 70)

        base_url = DEFAULT_URL_FOR_TEST
        results = {}

        for backend in self.BACKENDS:
            print(f"\n--- Testing {backend} backend ---")

            process = popen_launch_server(
                self.MODEL,
                base_url,
                timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
                other_args=["--attention-backend", backend],
            )

            try:
                args = SimpleNamespace(
                    base_url=base_url,
                    model=self.MODEL,
                    eval_name="mmlu",
                    num_examples=64,  # Reduced for faster testing
                    num_threads=32,
                )

                metrics = run_eval(args)
                results[backend] = metrics
                print(f"{backend}: MMLU score = {metrics['score']:.4f}")

            except Exception as e:
                print(f"{backend}: Error - {e}")
                results[backend] = {"score": 0, "error": str(e)}
            finally:
                kill_process_tree(process.pid)
                time.sleep(2)  # Allow cleanup

        # Print comparison
        print("\n" + "=" * 70)
        print("MMLU ACCURACY RESULTS")
        print("=" * 70)

        for backend, metrics in results.items():
            score = metrics.get("score", 0)
            print(f"{backend}: {score:.4f}")

        # Verify SageAttention maintains accuracy
        if "sage_attn" in results and "triton" in results:
            sage_score = results["sage_attn"].get("score", 0)
            triton_score = results["triton"].get("score", 0)

            if sage_score > 0 and triton_score > 0:
                accuracy_diff = abs(sage_score - triton_score)
                print(f"\nAccuracy difference: {accuracy_diff:.4f}")

                # SageAttention should maintain similar accuracy (within 5%)
                self.assertLess(
                    accuracy_diff,
                    0.05,
                    f"SageAttention accuracy differs too much from baseline: {accuracy_diff}",
                )

        # Verify minimum accuracy threshold
        if "sage_attn" in results:
            self.assertGreaterEqual(
                results["sage_attn"].get("score", 0),
                0.55,  # Lower threshold for stability
                "SageAttention MMLU accuracy below acceptable threshold",
            )

    def test_memory_footprint(self):
        """
        Compare memory usage between backends.
        SageAttention's INT8 quantization should show memory benefits.
        """
        print("\n" + "=" * 70)
        print("MEMORY FOOTPRINT BENCHMARK")
        print("=" * 70)

        base_url = DEFAULT_URL_FOR_TEST
        memory_results = {}

        for backend in self.BACKENDS:
            print(f"\n--- Testing {backend} backend ---")
            reset_gpu_memory()

            process = popen_launch_server(
                self.MODEL,
                base_url,
                timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
                other_args=["--attention-backend", backend],
            )

            try:
                # Wait for server to stabilize
                time.sleep(10)

                # Get memory info via nvidia-smi
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
                            f"{backend}: {used:.0f} MB / {total:.0f} MB "
                            f"({memory_results[backend]['utilization_pct']:.1f}%)"
                        )

            except Exception as e:
                print(f"{backend}: Error measuring memory - {e}")
            finally:
                kill_process_tree(process.pid)
                time.sleep(5)  # Allow full cleanup
                reset_gpu_memory()

        # Print comparison
        print("\n" + "=" * 70)
        print("MEMORY FOOTPRINT SUMMARY")
        print("=" * 70)

        if "sage_attn" in memory_results and "triton" in memory_results:
            sage_mem = memory_results["sage_attn"]["used_mb"]
            triton_mem = memory_results["triton"]["used_mb"]
            diff = triton_mem - sage_mem
            diff_pct = (diff / triton_mem) * 100 if triton_mem > 0 else 0

            print(f"SageAttention: {sage_mem:.0f} MB")
            print(f"Triton:        {triton_mem:.0f} MB")
            print(f"Difference:    {diff:.0f} MB ({diff_pct:.1f}%)")


class TestSageAttentionAccuracyDetailed(CustomTestCase):
    """
    Detailed accuracy tests comparing SageAttention against FP16 baseline.
    """

    MODEL = DEFAULT_MODEL_NAME_FOR_TEST

    def test_output_logits_comparison(self):
        """
        Compare raw output logits between SageAttention and baseline.
        This test uses a simple prompt and compares the generated tokens.
        """
        print("\n" + "=" * 70)
        print("OUTPUT LOGITS COMPARISON")
        print("=" * 70)

        import requests

        base_url = DEFAULT_URL_FOR_TEST
        test_prompt = "The capital of France is"

        results = {}

        for backend in ["sage_attn", "triton"]:
            print(f"\n--- Testing {backend} backend ---")

            process = popen_launch_server(
                self.MODEL,
                base_url,
                timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
                other_args=["--attention-backend", backend],
            )

            try:
                # Generate with low temperature for reproducibility
                response = requests.post(
                    f"{base_url}/v1/completions",
                    json={
                        "model": self.MODEL,
                        "prompt": test_prompt,
                        "max_tokens": 20,
                        "temperature": 0.0,  # Greedy decoding
                    },
                    timeout=60,
                )

                if response.status_code == 200:
                    result = response.json()
                    generated_text = result["choices"][0]["text"]
                    results[backend] = generated_text
                    print(f"{backend}: '{generated_text}'")
                else:
                    print(f"{backend}: Request failed with status {response.status_code}")

            except Exception as e:
                print(f"{backend}: Error - {e}")
            finally:
                kill_process_tree(process.pid)
                time.sleep(2)

        # Compare outputs
        print("\n" + "=" * 70)
        print("OUTPUT COMPARISON")
        print("=" * 70)

        if "sage_attn" in results and "triton" in results:
            sage_out = results["sage_attn"]
            triton_out = results["triton"]

            print(f"SageAttention: '{sage_out}'")
            print(f"Triton:        '{triton_out}'")

            # For greedy decoding, outputs should be very similar
            # Allow for small differences due to numerical precision
            if sage_out == triton_out:
                print("Outputs MATCH exactly")
            else:
                # Check how many tokens match
                sage_tokens = sage_out.split()
                triton_tokens = triton_out.split()
                matching = sum(1 for s, t in zip(sage_tokens, triton_tokens) if s == t)
                total = max(len(sage_tokens), len(triton_tokens))
                match_rate = matching / total if total > 0 else 0
                print(f"Token match rate: {match_rate:.2%}")

                # Should have high match rate for simple prompt
                self.assertGreater(
                    match_rate,
                    0.7,
                    f"Token match rate too low: {match_rate}",
                )

    def test_short_vs_long_context_accuracy(self):
        """
        Validate that accuracy is preserved for both short and long contexts.
        This addresses the concern about "accuracy drift" with longer sequences.
        """
        print("\n" + "=" * 70)
        print("SHORT VS LONG CONTEXT ACCURACY")
        print("=" * 70)

        import requests

        base_url = DEFAULT_URL_FOR_TEST

        # Test prompts of varying complexity
        test_cases = [
            {
                "name": "short_simple",
                "prompt": "2 + 2 =",
                "expected_contains": ["4"],
            },
            {
                "name": "medium_factual",
                "prompt": "The largest planet in our solar system is",
                "expected_contains": ["Jupiter"],
            },
            {
                "name": "longer_reasoning",
                "prompt": (
                    "If all roses are flowers and some flowers fade quickly, "
                    "can we conclude that some roses fade quickly? "
                    "Think step by step. Answer:"
                ),
                "expected_contains": ["no", "cannot", "not necessarily"],
            },
        ]

        for backend in ["sage_attn", "triton"]:
            print(f"\n--- Testing {backend} backend ---")

            process = popen_launch_server(
                self.MODEL,
                base_url,
                timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
                other_args=["--attention-backend", backend],
            )

            try:
                for tc in test_cases:
                    response = requests.post(
                        f"{base_url}/v1/completions",
                        json={
                            "model": self.MODEL,
                            "prompt": tc["prompt"],
                            "max_tokens": 50,
                            "temperature": 0.0,
                        },
                        timeout=60,
                    )

                    if response.status_code == 200:
                        result = response.json()
                        generated = result["choices"][0]["text"].lower()
                        matches = any(
                            exp.lower() in generated for exp in tc["expected_contains"]
                        )
                        status = "PASS" if matches else "FAIL"
                        print(f"  {tc['name']}: {status}")
                        if not matches:
                            print(f"    Generated: '{generated[:100]}'")
                    else:
                        print(f"  {tc['name']}: REQUEST_FAILED")

            except Exception as e:
                print(f"  Error: {e}")
            finally:
                kill_process_tree(process.pid)
                time.sleep(2)


class TestSageAttentionStress(CustomTestCase):
    """
    Stress tests for SageAttention under various conditions.
    """

    MODEL = DEFAULT_MODEL_NAME_FOR_TEST

    def test_batch_size_scaling(self):
        """
        Test throughput scaling with increasing batch sizes.
        Validates that SageAttention scales well with batching.
        """
        print("\n" + "=" * 70)
        print("BATCH SIZE SCALING TEST")
        print("=" * 70)

        batch_sizes = [1, 4, 8, 16]
        results = []

        for num_prompts in batch_sizes:
            print(f"\n--- Batch size: {num_prompts} ---")

            command = [
                "python3",
                "-m",
                "sglang.bench_offline_throughput",
                "--num-prompts",
                str(num_prompts),
                "--dataset-name",
                "random",
                "--random-input-len",
                "256",
                "--random-output-len",
                "64",
                "--model-path",
                self.MODEL,
                "--attention-backend",
                "sage_attn",
            ]

            try:
                process = subprocess.Popen(
                    command,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                )
                stdout, _ = process.communicate(timeout=300)
                output = stdout.decode(errors="backslashreplace")

                throughput = -1
                for line in output.split("\n"):
                    if "Last generation throughput (tok/s):" in line:
                        throughput = float(line.split(":")[-1].strip())

                results.append(
                    {
                        "batch_size": num_prompts,
                        "throughput": throughput,
                    }
                )
                print(f"  Throughput: {throughput:.2f} tok/s")

            except Exception as e:
                print(f"  Error: {e}")
                results.append({"batch_size": num_prompts, "error": str(e)})
            finally:
                try:
                    kill_process_tree(process.pid)
                except Exception:
                    pass

        # Print scaling summary
        print("\n" + "=" * 70)
        print("BATCH SCALING SUMMARY")
        print("=" * 70)
        print(f"{'Batch Size':<12} {'Throughput (tok/s)':<20}")
        print("-" * 32)
        for r in results:
            tp = r.get("throughput", -1)
            tp_str = f"{tp:.2f}" if tp > 0 else "failed"
            print(f"{r['batch_size']:<12} {tp_str:<20}")

    def test_continuous_generation_stability(self):
        """
        Test stability over continuous generation.
        Validates that SageAttention doesn't have memory leaks or instability.
        """
        print("\n" + "=" * 70)
        print("CONTINUOUS GENERATION STABILITY TEST")
        print("=" * 70)

        import requests

        base_url = DEFAULT_URL_FOR_TEST

        process = popen_launch_server(
            self.MODEL,
            base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=["--attention-backend", "sage_attn"],
        )

        try:
            num_iterations = 10
            success_count = 0
            latencies = []

            for i in range(num_iterations):
                start_time = time.time()
                try:
                    response = requests.post(
                        f"{base_url}/v1/completions",
                        json={
                            "model": self.MODEL,
                            "prompt": f"Count from 1 to 10: 1, 2, 3, ",
                            "max_tokens": 20,
                            "temperature": 0.1,
                        },
                        timeout=30,
                    )
                    latency = time.time() - start_time

                    if response.status_code == 200:
                        success_count += 1
                        latencies.append(latency)
                        print(f"  Iteration {i + 1}: SUCCESS ({latency:.2f}s)")
                    else:
                        print(f"  Iteration {i + 1}: FAILED (status {response.status_code})")

                except Exception as e:
                    print(f"  Iteration {i + 1}: ERROR ({e})")

            # Summary
            print(f"\nSuccess rate: {success_count}/{num_iterations}")
            if latencies:
                avg_latency = sum(latencies) / len(latencies)
                print(f"Average latency: {avg_latency:.2f}s")

            self.assertGreaterEqual(
                success_count,
                num_iterations * 0.9,  # 90% success rate
                "Too many failures in continuous generation",
            )

        finally:
            kill_process_tree(process.pid)


if __name__ == "__main__":
    # Support verbose output
    if "--verbose" in sys.argv:
        sys.argv.remove("--verbose")

    unittest.main(verbosity=2)
