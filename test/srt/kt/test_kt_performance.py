"""
Performance tests for KT-kernel integration

Tests throughput performance with different GPU configurations.
Measures output/input throughput, latency metrics using bench_serving.
"""

import json
import os
import unittest
from datetime import datetime

from sglang.bench_serving import run_benchmark
from sglang.srt.utils import kill_process_tree
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    CustomTestCase,
    get_benchmark_args,
    popen_launch_server,
    write_github_step_summary,
)

from .utils import get_kt_env, get_kt_model_paths, get_kt_server_args


def save_performance_results(
    test_name: str,
    gpu_config: str,
    metrics: dict,
    config: dict,
    filename: str = None,
):
    """
    Save performance results to JSON file

    Args:
        test_name: Name of the test (e.g., "test_throughput")
        gpu_config: GPU configuration (e.g., "1GPU", "4GPU", "8GPU")
        metrics: Performance metrics dict from run_bench_serving
        config: Test configuration dict
        filename: Output filename (default: kt_performance_{gpu_config}_results.json)
    """
    if filename is None:
        filename = f"kt_performance_{gpu_config.lower()}_results.json"

    result = {
        "timestamp": datetime.now().isoformat(),
        "test_name": test_name,
        "gpu_config": gpu_config,
        "metrics": {
            "output_throughput_tokens_per_sec": metrics.get("output_throughput", 0),
            "input_throughput_tokens_per_sec": metrics.get("input_throughput", 0),
            "median_e2e_latency_ms": metrics.get("median_e2e_latency_ms", 0),
            "median_ttft_ms": metrics.get("median_ttft_ms", 0),
            "median_itl_ms": metrics.get("median_itl_ms", 0),
            "completed_requests": metrics.get("completed", 0),
        },
        "config": config,
    }

    # Load existing results if file exists
    existing_results = []
    if os.path.exists(filename):
        try:
            with open(filename, "r") as f:
                existing_results = json.load(f)
        except json.JSONDecodeError:
            existing_results = []

    # Ensure it's a list
    if not isinstance(existing_results, list):
        existing_results = []

    # Append new result
    existing_results.append(result)

    # Save to file
    with open(filename, "w") as f:
        json.dump(existing_results, f, indent=2)

    print(f"\n✓ Performance results saved to: {filename}")


def write_performance_summary(
    test_name: str,
    gpu_config: str,
    metrics: dict,
    config: dict,
):
    """
    Write performance summary to GitHub Step Summary (if in CI environment)

    Args:
        test_name: Name of the test
        gpu_config: GPU configuration
        metrics: Performance metrics dict from run_bench_serving
        config: Test configuration dict
    """
    # Generate markdown summary
    summary = f"""
## KT Performance Test - {gpu_config}

**Test**: {test_name}
**Timestamp**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

### Configuration
| Parameter | Value |
|-----------|-------|
| GPU Config | {gpu_config} |
| Num Prompts | {config.get('num_prompts', 'N/A')} |
| Input Length | {config.get('input_length', 'N/A')} tokens |
| Output Length | {config.get('output_length', 'N/A')} tokens |
| KT Num GPU Experts | {config.get('kt_num_gpu_experts', 'N/A')} |
| KT CPU Infer | {config.get('kt_cpuinfer', 'N/A')} |
| Tensor Parallel Size | {config.get('tensor_parallel_size', 'N/A')} |

### Performance Metrics
| Metric | Value |
|--------|-------|
| **Output Throughput** | {metrics.get('output_throughput', 0):.2f} tokens/sec |
| **Input Throughput** | {metrics.get('input_throughput', 0):.2f} tokens/sec |
| **Median E2E Latency** | {metrics.get('median_e2e_latency_ms', 0):.2f} ms |
| **Median TTFT** | {metrics.get('median_ttft_ms', 0):.2f} ms |
| **Median ITL** | {metrics.get('median_itl_ms', 0):.2f} ms |
| **Completed Requests** | {metrics.get('completed', 0)} / {config.get('num_prompts', 0)} |

---

"""

    # Write to GitHub Step Summary if in CI environment
    write_github_step_summary(summary)


class TestKTPerformance1GPU(CustomTestCase):
    """
    Test throughput performance with 1 GPU configuration

    Configuration:
    - tensor_parallel_size: 1
    - kt_num_gpu_experts: 1
    - kt_cpuinfer: 60
    - Test: 50 prompts, 512 input tokens, 256 output tokens
    """

    @classmethod
    def setUpClass(cls):
        model_paths = get_kt_model_paths()
        cls.model = model_paths["gpu_model_path"]
        cls.base_url = "http://127.0.0.1:30020"

        # Build KT-specific server arguments
        other_args = get_kt_server_args(
            kt_weight_path=model_paths["cpu_model_path"],
            kt_num_gpu_experts=1,
            kt_cpuinfer=60,
            kt_threadpool_count=2,
            kt_method="AMXINT4",
            tensor_parallel_size=1,
            served_model_name=model_paths["served_model_name"],
            max_running_requests=40,
            max_total_tokens=40000,
            additional_args=["--log-level", "error"],
        )

        # Launch server
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=other_args,
            env=get_kt_env(),
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def test_throughput(self):
        """Test throughput with 50 prompts (512 input, 256 output tokens)"""
        print("\n" + "=" * 70)
        print("Testing Throughput Performance (1 GPU)")
        print("=" * 70)

        num_prompts = 50
        input_length = 512
        output_length = 256

        print(f"\nRunning benchmark:")
        print(f"  Num Prompts: {num_prompts}")
        print(f"  Input Length: {input_length} tokens")
        print(f"  Output Length: {output_length} tokens")
        print(f"  Mode: Offline (parallel requests)")

        # Run benchmark
        try:
            # Prepare benchmark arguments
            args = get_benchmark_args(
                base_url=self.base_url,
                dataset_name="random",
                tokenizer=self.model,  # Use model path as tokenizer
                num_prompts=num_prompts,
                random_input_len=input_length,
                random_output_len=output_length,
                request_rate=float("inf"),  # Offline mode
                disable_stream=False,
                seed=42,
            )

            # Warmup run
            warmup_args = get_benchmark_args(
                base_url=self.base_url,
                dataset_name="random",
                tokenizer=self.model,  # Use model path as tokenizer
                num_prompts=16,
                random_input_len=input_length,
                random_output_len=output_length,
                request_rate=float("inf"),
                disable_stream=False,
                seed=42,
            )
            print("\nWarming up...")
            run_benchmark(warmup_args)

            # Actual benchmark run
            print("Running actual benchmark...")
            res = run_benchmark(args)

            # Print performance metrics
            print("\n" + "-" * 70)
            print("Performance Metrics:")
            print("-" * 70)
            print(
                f"Output Throughput:       {res.get('output_throughput', 0):.2f} tokens/sec"
            )
            print(
                f"Input Throughput:        {res.get('input_throughput', 0):.2f} tokens/sec"
            )
            print(
                f"Median E2E Latency:      {res.get('median_e2e_latency_ms', 0):.2f} ms"
            )
            print(f"Median TTFT:             {res.get('median_ttft_ms', 0):.2f} ms")
            print(f"Median ITL:              {res.get('median_itl_ms', 0):.2f} ms")
            print(f"Completed Requests:      {res.get('completed', 0)} / {num_prompts}")
            print("-" * 70)

            # Validate results
            self.assertEqual(
                res.get("completed", 0),
                num_prompts,
                f"Not all requests completed: {res.get('completed', 0)}/{num_prompts}",
            )
            self.assertGreater(
                res.get("output_throughput", 0),
                0,
                "Output throughput should be > 0",
            )

            # Save performance results to JSON
            test_config = {
                "num_prompts": num_prompts,
                "input_length": input_length,
                "output_length": output_length,
                "kt_num_gpu_experts": 1,
                "kt_cpuinfer": 60,
                "tensor_parallel_size": 1,
            }
            save_performance_results(
                test_name="test_throughput",
                gpu_config="1GPU",
                metrics=res,
                config=test_config,
            )

            # Write to GitHub Step Summary (if in CI)
            write_performance_summary(
                test_name="test_throughput",
                gpu_config="1GPU",
                metrics=res,
                config=test_config,
            )

            print(f"\n✓ Performance test passed")
            print(f"✓ All {num_prompts} requests completed successfully")

        except Exception as e:
            self.fail(f"Performance test failed: {e}")


class TestKTPerformance4GPU(CustomTestCase):
    """
    Test throughput performance with 4 GPU configuration

    Configuration:
    - tensor_parallel_size: 4
    - kt_num_gpu_experts: 80
    - kt_cpuinfer: 60
    - Test: 50 prompts, 512 input tokens, 256 output tokens
    """

    @classmethod
    def setUpClass(cls):
        model_paths = get_kt_model_paths()
        cls.model = model_paths["gpu_model_path"]
        cls.base_url = "http://127.0.0.1:30021"

        # Build KT-specific server arguments
        other_args = get_kt_server_args(
            kt_weight_path=model_paths["cpu_model_path"],
            kt_num_gpu_experts=80,
            kt_cpuinfer=60,
            kt_threadpool_count=2,
            kt_method="AMXINT4",
            tensor_parallel_size=4,
            served_model_name=model_paths["served_model_name"],
            max_running_requests=40,
            max_total_tokens=40000,
            additional_args=["--log-level", "error"],
        )

        # Launch server
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=other_args,
            env=get_kt_env(),
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def test_throughput(self):
        """Test throughput with 50 prompts (512 input, 256 output tokens)"""
        print("\n" + "=" * 70)
        print("Testing Throughput Performance (4 GPU)")
        print("=" * 70)

        num_prompts = 50
        input_length = 512
        output_length = 256

        print(f"\nRunning benchmark:")
        print(f"  Num Prompts: {num_prompts}")
        print(f"  Input Length: {input_length} tokens")
        print(f"  Output Length: {output_length} tokens")
        print(f"  Mode: Offline (parallel requests)")

        # Run benchmark
        try:
            # Prepare benchmark arguments
            args = get_benchmark_args(
                base_url=self.base_url,
                dataset_name="random",
                tokenizer=self.model,  # Use model path as tokenizer
                num_prompts=num_prompts,
                random_input_len=input_length,
                random_output_len=output_length,
                request_rate=float("inf"),  # Offline mode
                disable_stream=False,
                seed=42,
            )

            # Warmup run
            warmup_args = get_benchmark_args(
                base_url=self.base_url,
                dataset_name="random",
                tokenizer=self.model,  # Use model path as tokenizer
                num_prompts=16,
                random_input_len=input_length,
                random_output_len=output_length,
                request_rate=float("inf"),
                disable_stream=False,
                seed=42,
            )
            print("\nWarming up...")
            run_benchmark(warmup_args)

            # Actual benchmark run
            print("Running actual benchmark...")
            res = run_benchmark(args)

            # Print performance metrics
            print("\n" + "-" * 70)
            print("Performance Metrics:")
            print("-" * 70)
            print(
                f"Output Throughput:       {res.get('output_throughput', 0):.2f} tokens/sec"
            )
            print(
                f"Input Throughput:        {res.get('input_throughput', 0):.2f} tokens/sec"
            )
            print(
                f"Median E2E Latency:      {res.get('median_e2e_latency_ms', 0):.2f} ms"
            )
            print(f"Median TTFT:             {res.get('median_ttft_ms', 0):.2f} ms")
            print(f"Median ITL:              {res.get('median_itl_ms', 0):.2f} ms")
            print(f"Completed Requests:      {res.get('completed', 0)} / {num_prompts}")
            print("-" * 70)

            # Validate results
            self.assertEqual(
                res.get("completed", 0),
                num_prompts,
                f"Not all requests completed: {res.get('completed', 0)}/{num_prompts}",
            )
            self.assertGreater(
                res.get("output_throughput", 0),
                0,
                "Output throughput should be > 0",
            )

            # Save performance results to JSON
            test_config = {
                "num_prompts": num_prompts,
                "input_length": input_length,
                "output_length": output_length,
                "kt_num_gpu_experts": 80,
                "kt_cpuinfer": 60,
                "tensor_parallel_size": 4,
            }
            save_performance_results(
                test_name="test_throughput",
                gpu_config="4GPU",
                metrics=res,
                config=test_config,
            )

            # Write to GitHub Step Summary (if in CI)
            write_performance_summary(
                test_name="test_throughput",
                gpu_config="4GPU",
                metrics=res,
                config=test_config,
            )

            print(f"\n✓ Performance test passed")
            print(f"✓ All {num_prompts} requests completed successfully")

        except Exception as e:
            self.fail(f"Performance test failed: {e}")


class TestKTPerformance8GPU(CustomTestCase):
    """
    Test throughput performance with 8 GPU configuration

    Configuration:
    - tensor_parallel_size: 8
    - kt_num_gpu_experts: 200
    - kt_cpuinfer: 60
    - Test: 50 prompts, 512 input tokens, 256 output tokens
    - Success Criteria: Throughput > 150 tok/s
    """

    @classmethod
    def setUpClass(cls):
        model_paths = get_kt_model_paths()
        cls.model = model_paths["gpu_model_path"]
        cls.base_url = "http://127.0.0.1:30022"

        # Build KT-specific server arguments
        other_args = get_kt_server_args(
            kt_weight_path=model_paths["cpu_model_path"],
            kt_num_gpu_experts=200,
            kt_cpuinfer=60,
            kt_threadpool_count=2,
            kt_method="AMXINT4",
            tensor_parallel_size=8,
            served_model_name=model_paths["served_model_name"],
            max_running_requests=40,
            max_total_tokens=40000,
            additional_args=["--log-level", "error"],
        )

        # Launch server
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=other_args,
            env=get_kt_env(),
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def test_throughput(self):
        """Test throughput with 50 prompts (512 input, 256 output tokens)"""
        print("\n" + "=" * 70)
        print("Testing Throughput Performance (8 GPU)")
        print("=" * 70)

        num_prompts = 50
        input_length = 512
        output_length = 256
        min_throughput = 150  # Success criteria

        print(f"\nRunning benchmark:")
        print(f"  Num Prompts: {num_prompts}")
        print(f"  Input Length: {input_length} tokens")
        print(f"  Output Length: {output_length} tokens")
        print(f"  Mode: Offline (parallel requests)")
        print(f"  Success Criteria: Throughput > {min_throughput} tok/s")

        # Run benchmark
        try:
            # Prepare benchmark arguments
            args = get_benchmark_args(
                base_url=self.base_url,
                dataset_name="random",
                tokenizer=self.model,  # Use model path as tokenizer
                num_prompts=num_prompts,
                random_input_len=input_length,
                random_output_len=output_length,
                request_rate=float("inf"),  # Offline mode
                disable_stream=False,
                seed=42,
            )

            # Warmup run
            warmup_args = get_benchmark_args(
                base_url=self.base_url,
                dataset_name="random",
                tokenizer=self.model,  # Use model path as tokenizer
                num_prompts=16,
                random_input_len=input_length,
                random_output_len=output_length,
                request_rate=float("inf"),
                disable_stream=False,
                seed=42,
            )
            print("\nWarming up...")
            run_benchmark(warmup_args)

            # Actual benchmark run
            print("Running actual benchmark...")
            res = run_benchmark(args)

            # Print performance metrics
            print("\n" + "-" * 70)
            print("Performance Metrics:")
            print("-" * 70)
            print(
                f"Output Throughput:       {res.get('output_throughput', 0):.2f} tokens/sec"
            )
            print(
                f"Input Throughput:        {res.get('input_throughput', 0):.2f} tokens/sec"
            )
            print(
                f"Median E2E Latency:      {res.get('median_e2e_latency_ms', 0):.2f} ms"
            )
            print(f"Median TTFT:             {res.get('median_ttft_ms', 0):.2f} ms")
            print(f"Median ITL:              {res.get('median_itl_ms', 0):.2f} ms")
            print(f"Completed Requests:      {res.get('completed', 0)} / {num_prompts}")
            print("-" * 70)

            # Validate results
            self.assertEqual(
                res.get("completed", 0),
                num_prompts,
                f"Not all requests completed: {res.get('completed', 0)}/{num_prompts}",
            )

            # Validate throughput meets success criteria
            output_throughput = res.get("output_throughput", 0)
            self.assertGreater(
                output_throughput,
                min_throughput,
                f"Output throughput {output_throughput:.2f} tok/s below threshold {min_throughput} tok/s",
            )

            # Save performance results to JSON
            test_config = {
                "num_prompts": num_prompts,
                "input_length": input_length,
                "output_length": output_length,
                "kt_num_gpu_experts": 200,
                "kt_cpuinfer": 60,
                "tensor_parallel_size": 8,
                "min_throughput": min_throughput,
            }
            save_performance_results(
                test_name="test_throughput",
                gpu_config="8GPU",
                metrics=res,
                config=test_config,
            )

            # Write to GitHub Step Summary (if in CI)
            write_performance_summary(
                test_name="test_throughput",
                gpu_config="8GPU",
                metrics=res,
                config=test_config,
            )

            print(f"\n✓ Performance test passed")
            print(f"✓ All {num_prompts} requests completed successfully")
            print(
                f"✓ Throughput {output_throughput:.2f} tok/s exceeds threshold {min_throughput} tok/s"
            )

        except Exception as e:
            self.fail(f"Performance test failed: {e}")


if __name__ == "__main__":
    unittest.main()
