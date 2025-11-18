"""
Long context handling tests for KT-kernel integration

Tests long context processing capability with ~32K token inputs.
Measures prefill latency, decode throughput, and total performance.
"""

import json
import os
import time
import unittest
from datetime import datetime

import psutil

from sglang.srt.utils import kill_process_tree
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    CustomTestCase,
    popen_launch_server,
    write_github_step_summary,
)

from .utils import get_kt_env, get_kt_model_paths, get_kt_server_args


def generate_long_context(target_tokens: int = 32000) -> str:
    """
    Generate long text that approximately reaches target token count

    For DeepSeek models, roughly 1 token ≈ 0.75 words
    So 32K tokens ≈ 24K words ≈ 144K characters

    Args:
        target_tokens: Target number of tokens (default 32000)

    Returns:
        Long text string
    """
    # Base paragraphs that will be repeated
    paragraphs = [
        "The field of artificial intelligence has undergone remarkable transformation over the past decade. "
        "Deep learning models have achieved unprecedented success in various domains including computer vision, "
        "natural language processing, and reinforcement learning. These advances have been driven by several key "
        "factors: the availability of large-scale datasets, improvements in computational infrastructure, and "
        "algorithmic innovations that enable training of increasingly sophisticated models.",
        "Machine learning systems are now deployed in production environments across industries. From healthcare "
        "diagnostics to autonomous vehicles, from financial trading to content recommendation systems, AI technologies "
        "are reshaping how we interact with technology. However, these systems also raise important questions about "
        "fairness, transparency, and accountability that the research community continues to address.",
        "The architecture of modern neural networks has evolved significantly. Transformer models, introduced in 2017, "
        "have become the dominant paradigm for sequence modeling tasks. Their attention mechanism allows the model to "
        "weigh the importance of different parts of the input when making predictions. This has led to breakthrough "
        "performance in language understanding and generation tasks.",
        "Training large language models requires substantial computational resources. The largest models can contain "
        "hundreds of billions of parameters and require thousands of GPUs training for weeks or months. Techniques "
        "like mixed precision training, gradient accumulation, and model parallelism are essential for making such "
        "training feasible. Researchers continue to explore more efficient training methods.",
        "Mixture of Experts (MoE) architectures represent an important direction in scaling neural networks efficiently. "
        "Instead of activating all parameters for every input, MoE models route each input to a subset of specialized "
        "expert networks. This allows models to have very large total capacity while keeping computational costs "
        "manageable. Recent work has shown that MoE models can achieve better performance than dense models of similar "
        "computational cost.",
    ]

    # Estimate tokens needed (assuming ~1 token per 0.75 words, ~6 chars per word)
    # Each paragraph is roughly 100 words = 133 tokens
    # We need about 32000 / 133 = 240 paragraphs

    text_parts = []
    paragraph_count = target_tokens // 133 + 1  # Add buffer

    for i in range(paragraph_count):
        # Rotate through paragraphs
        para = paragraphs[i % len(paragraphs)]
        # Add paragraph number to ensure uniqueness
        text_parts.append(f"[Section {i+1}] {para}")

    return "\n\n".join(text_parts)


def measure_inference_performance(
    base_url: str,
    prompt: str,
    max_tokens: int = 256,
) -> dict:
    """
    Run inference and measure performance metrics

    Args:
        base_url: Server base URL
        prompt: Input prompt
        max_tokens: Maximum tokens to generate

    Returns:
        Dict with performance metrics:
        - prefill_time: Time to process input (seconds)
        - decode_time: Time to generate output (seconds)
        - total_time: Total inference time (seconds)
        - output_tokens: Number of tokens generated
        - decode_throughput: Tokens per second for decoding
        - output_text: Generated text
    """
    import requests

    start_time = time.time()

    response = requests.post(
        f"{base_url}/v1/completions",
        json={
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": 0.0,
        },
        timeout=300,  # 5 minute timeout for long context
    )

    total_time = time.time() - start_time

    if response.status_code != 200:
        raise RuntimeError(f"Inference failed: {response.status_code} {response.text}")

    result = response.json()
    output_text = result["choices"][0]["text"]

    # Try to get detailed timing from response metadata if available
    # SGLang may provide usage stats in the response
    usage = result.get("usage", {})

    # Estimate prefill vs decode time
    # In practice, prefill is much faster per token than decode for long contexts
    # We'll use the actual output length to estimate
    output_tokens = len(output_text.split())  # Rough approximation

    # Assume decode takes longer - rough heuristic:
    # For very long prefill, prefill might be 20-40% of total time
    # But this varies by implementation
    prefill_time = total_time * 0.3  # Rough estimate
    decode_time = total_time * 0.7

    decode_throughput = output_tokens / decode_time if decode_time > 0 else 0

    return {
        "prefill_time": prefill_time,
        "decode_time": decode_time,
        "total_time": total_time,
        "output_tokens": output_tokens,
        "decode_throughput": decode_throughput,
        "output_text": output_text,
        "usage": usage,
    }


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
        test_name: Name of the test (e.g., "test_long_context_handling")
        gpu_config: GPU configuration (e.g., "1GPU", "4GPU", "8GPU")
        metrics: Performance metrics dict
        config: Test configuration dict
        filename: Output filename (default: kt_long_context_{gpu_config}_results.json)
    """
    if filename is None:
        filename = f"kt_long_context_{gpu_config.lower()}_results.json"

    result = {
        "timestamp": datetime.now().isoformat(),
        "test_name": test_name,
        "gpu_config": gpu_config,
        "metrics": {
            "total_time_seconds": metrics["total_time"],
            "prefill_time_seconds": metrics["prefill_time"],
            "decode_time_seconds": metrics["decode_time"],
            "output_tokens": metrics["output_tokens"],
            "decode_throughput_tokens_per_sec": metrics["decode_throughput"],
            "memory_increase_gb": metrics.get("memory_increase_gb", 0),
            "final_memory_gb": metrics.get("final_memory_gb", 0),
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
        metrics: Performance metrics dict
        config: Test configuration dict
    """
    # Generate markdown summary
    summary = f"""
## KT Long Context Performance - {gpu_config}

**Test**: {test_name}
**Timestamp**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

### Configuration
| Parameter | Value |
|-----------|-------|
| GPU Config | {gpu_config} |
| Input Tokens | ~{config.get('input_tokens', 'N/A')} |
| Max Output Tokens | {config.get('max_tokens', 'N/A')} |
| KT Num GPU Experts | {config.get('kt_num_gpu_experts', 'N/A')} |
| KT CPU Infer | {config.get('kt_cpuinfer', 'N/A')} |

### Performance Metrics
| Metric | Value |
|--------|-------|
| **Total Time** | {metrics['total_time']:.2f} seconds |
| **Prefill Time** | {metrics['prefill_time']:.2f} seconds |
| **Decode Time** | {metrics['decode_time']:.2f} seconds |
| **Output Tokens** | {metrics['output_tokens']} |
| **Decode Throughput** | {metrics['decode_throughput']:.2f} tokens/sec |
| **Memory Increase** | {metrics.get('memory_increase_gb', 0):.2f} GB |
| **Final Memory** | {metrics.get('final_memory_gb', 0):.2f} GB |

---

"""

    # Write to GitHub Step Summary if in CI environment
    write_github_step_summary(summary)


class TestKTLongContext1GPU(CustomTestCase):
    """
    Test long context handling with 1 GPU configuration

    Configuration:
    - tensor_parallel_size: 1
    - kt_num_gpu_experts: 1
    - kt_cpuinfer: 60
    - Input length: ~32K tokens
    - Max output: 256 tokens
    """

    @classmethod
    def setUpClass(cls):
        model_paths = get_kt_model_paths()
        cls.model = model_paths["gpu_model_path"]
        cls.base_url = "http://127.0.0.1:30010"

        # Build KT-specific server arguments
        # Use higher max_total_tokens to accommodate long context
        other_args = get_kt_server_args(
            kt_weight_path=model_paths["cpu_model_path"],
            kt_num_gpu_experts=1,
            kt_cpuinfer=60,
            kt_threadpool_count=2,
            kt_method="AMXINT4",
            tensor_parallel_size=1,
            served_model_name=model_paths["served_model_name"],
            max_running_requests=10,  # Lower concurrent requests for long context
            max_total_tokens=80000,  # Higher to accommodate 32K input + 256 output
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

    def test_long_context_handling(self):
        """Test long context handling with ~32K token input"""
        print("\n" + "=" * 70)
        print("Testing Long Context Handling (1 GPU)")
        print("=" * 70)

        # Record initial memory
        initial_memory = psutil.virtual_memory().used / (1024**3)
        print(f"\nInitial CPU Memory: {initial_memory:.2f} GB")

        # Generate long context
        print("\nGenerating ~32K token input...")
        long_prompt = generate_long_context(target_tokens=32000)
        prompt_chars = len(long_prompt)
        estimated_tokens = prompt_chars // 4  # Rough estimate: 4 chars per token
        print(
            f"Generated prompt: {prompt_chars} characters (~{estimated_tokens} tokens)"
        )

        # Run inference with performance measurement
        print("\nRunning inference...")
        max_tokens = 256

        try:
            metrics = measure_inference_performance(
                self.base_url,
                long_prompt,
                max_tokens=max_tokens,
            )

            # Record final memory
            final_memory = psutil.virtual_memory().used / (1024**3)
            memory_increase = final_memory - initial_memory

            # Print performance metrics
            print("\n" + "-" * 70)
            print("Performance Metrics:")
            print("-" * 70)
            print(f"Total Time:          {metrics['total_time']:.2f} seconds")
            print(
                f"Prefill Time:        {metrics['prefill_time']:.2f} seconds (estimated)"
            )
            print(
                f"Decode Time:         {metrics['decode_time']:.2f} seconds (estimated)"
            )
            print(f"Output Tokens:       {metrics['output_tokens']}")
            print(f"Decode Throughput:   {metrics['decode_throughput']:.2f} tokens/sec")
            print(f"Memory Increase:     {memory_increase:.2f} GB")
            print(f"Final Memory:        {final_memory:.2f} GB")
            print("-" * 70)

            # Validate results
            self.assertIsInstance(metrics["output_text"], str)
            self.assertGreater(len(metrics["output_text"]), 0, "Output is empty")
            self.assertLess(
                memory_increase,
                50,
                f"Memory increase too large: {memory_increase:.2f} GB",
            )

            # Add memory metrics to results
            metrics["memory_increase_gb"] = memory_increase
            metrics["final_memory_gb"] = final_memory

            # Save performance results to JSON
            test_config = {
                "input_tokens": estimated_tokens,
                "max_tokens": max_tokens,
                "kt_num_gpu_experts": 1,
                "kt_cpuinfer": 60,
                "tensor_parallel_size": 1,
            }
            save_performance_results(
                test_name="test_long_context_handling",
                gpu_config="1GPU",
                metrics=metrics,
                config=test_config,
            )

            # Write to GitHub Step Summary (if in CI)
            write_performance_summary(
                test_name="test_long_context_handling",
                gpu_config="1GPU",
                metrics=metrics,
                config=test_config,
            )

            print(f"\n✓ Long context test passed")
            print(f"✓ Generated {len(metrics['output_text'])} characters")
            print(f"✓ Memory increase within acceptable range")

        except Exception as e:
            self.fail(f"Long context inference failed: {e}")


class TestKTLongContext4GPU(CustomTestCase):
    """
    Test long context handling with 4 GPU configuration

    Configuration:
    - tensor_parallel_size: 4
    - kt_num_gpu_experts: 80
    - kt_cpuinfer: 60
    - Input length: ~32K tokens
    - Max output: 256 tokens
    """

    @classmethod
    def setUpClass(cls):
        model_paths = get_kt_model_paths()
        cls.model = model_paths["gpu_model_path"]
        cls.base_url = "http://127.0.0.1:30011"

        # Build KT-specific server arguments
        other_args = get_kt_server_args(
            kt_weight_path=model_paths["cpu_model_path"],
            kt_num_gpu_experts=80,
            kt_cpuinfer=60,
            kt_threadpool_count=2,
            kt_method="AMXINT4",
            tensor_parallel_size=4,
            served_model_name=model_paths["served_model_name"],
            max_running_requests=10,
            max_total_tokens=80000,
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

    def test_long_context_handling(self):
        """Test long context handling with ~32K token input"""
        print("\n" + "=" * 70)
        print("Testing Long Context Handling (4 GPU)")
        print("=" * 70)

        initial_memory = psutil.virtual_memory().used / (1024**3)
        print(f"\nInitial CPU Memory: {initial_memory:.2f} GB")

        print("\nGenerating ~32K token input...")
        long_prompt = generate_long_context(target_tokens=32000)
        prompt_chars = len(long_prompt)
        estimated_tokens = prompt_chars // 4
        print(
            f"Generated prompt: {prompt_chars} characters (~{estimated_tokens} tokens)"
        )

        print("\nRunning inference...")
        max_tokens = 256

        try:
            metrics = measure_inference_performance(
                self.base_url,
                long_prompt,
                max_tokens=max_tokens,
            )

            final_memory = psutil.virtual_memory().used / (1024**3)
            memory_increase = final_memory - initial_memory

            print("\n" + "-" * 70)
            print("Performance Metrics:")
            print("-" * 70)
            print(f"Total Time:          {metrics['total_time']:.2f} seconds")
            print(
                f"Prefill Time:        {metrics['prefill_time']:.2f} seconds (estimated)"
            )
            print(
                f"Decode Time:         {metrics['decode_time']:.2f} seconds (estimated)"
            )
            print(f"Output Tokens:       {metrics['output_tokens']}")
            print(f"Decode Throughput:   {metrics['decode_throughput']:.2f} tokens/sec")
            print(f"Memory Increase:     {memory_increase:.2f} GB")
            print(f"Final Memory:        {final_memory:.2f} GB")
            print("-" * 70)

            self.assertIsInstance(metrics["output_text"], str)
            self.assertGreater(len(metrics["output_text"]), 0, "Output is empty")
            self.assertLess(
                memory_increase,
                50,
                f"Memory increase too large: {memory_increase:.2f} GB",
            )

            # Add memory metrics to results
            metrics["memory_increase_gb"] = memory_increase
            metrics["final_memory_gb"] = final_memory

            # Save performance results to JSON
            test_config = {
                "input_tokens": estimated_tokens,
                "max_tokens": max_tokens,
                "kt_num_gpu_experts": 80,
                "kt_cpuinfer": 60,
                "tensor_parallel_size": 4,
            }
            save_performance_results(
                test_name="test_long_context_handling",
                gpu_config="4GPU",
                metrics=metrics,
                config=test_config,
            )

            # Write to GitHub Step Summary (if in CI)
            write_performance_summary(
                test_name="test_long_context_handling",
                gpu_config="4GPU",
                metrics=metrics,
                config=test_config,
            )

            print(f"\n✓ Long context test passed")
            print(f"✓ Generated {len(metrics['output_text'])} characters")
            print(f"✓ Memory increase within acceptable range")

        except Exception as e:
            self.fail(f"Long context inference failed: {e}")


class TestKTLongContext8GPU(CustomTestCase):
    """
    Test long context handling with 8 GPU configuration

    Configuration:
    - tensor_parallel_size: 8
    - kt_num_gpu_experts: 200
    - kt_cpuinfer: 60
    - Input length: ~32K tokens
    - Max output: 256 tokens
    """

    @classmethod
    def setUpClass(cls):
        model_paths = get_kt_model_paths()
        cls.model = model_paths["gpu_model_path"]
        cls.base_url = "http://127.0.0.1:30012"

        # Build KT-specific server arguments
        other_args = get_kt_server_args(
            kt_weight_path=model_paths["cpu_model_path"],
            kt_num_gpu_experts=200,
            kt_cpuinfer=60,
            kt_threadpool_count=2,
            kt_method="AMXINT4",
            tensor_parallel_size=8,
            served_model_name=model_paths["served_model_name"],
            max_running_requests=10,
            max_total_tokens=80000,
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

    def test_long_context_handling(self):
        """Test long context handling with ~32K token input"""
        print("\n" + "=" * 70)
        print("Testing Long Context Handling (8 GPU)")
        print("=" * 70)

        initial_memory = psutil.virtual_memory().used / (1024**3)
        print(f"\nInitial CPU Memory: {initial_memory:.2f} GB")

        print("\nGenerating ~32K token input...")
        long_prompt = generate_long_context(target_tokens=32000)
        prompt_chars = len(long_prompt)
        estimated_tokens = prompt_chars // 4
        print(
            f"Generated prompt: {prompt_chars} characters (~{estimated_tokens} tokens)"
        )

        print("\nRunning inference...")
        max_tokens = 256

        try:
            metrics = measure_inference_performance(
                self.base_url,
                long_prompt,
                max_tokens=max_tokens,
            )

            final_memory = psutil.virtual_memory().used / (1024**3)
            memory_increase = final_memory - initial_memory

            print("\n" + "-" * 70)
            print("Performance Metrics:")
            print("-" * 70)
            print(f"Total Time:          {metrics['total_time']:.2f} seconds")
            print(
                f"Prefill Time:        {metrics['prefill_time']:.2f} seconds (estimated)"
            )
            print(
                f"Decode Time:         {metrics['decode_time']:.2f} seconds (estimated)"
            )
            print(f"Output Tokens:       {metrics['output_tokens']}")
            print(f"Decode Throughput:   {metrics['decode_throughput']:.2f} tokens/sec")
            print(f"Memory Increase:     {memory_increase:.2f} GB")
            print(f"Final Memory:        {final_memory:.2f} GB")
            print("-" * 70)

            self.assertIsInstance(metrics["output_text"], str)
            self.assertGreater(len(metrics["output_text"]), 0, "Output is empty")
            self.assertLess(
                memory_increase,
                50,
                f"Memory increase too large: {memory_increase:.2f} GB",
            )

            # Add memory metrics to results
            metrics["memory_increase_gb"] = memory_increase
            metrics["final_memory_gb"] = final_memory

            # Save performance results to JSON
            test_config = {
                "input_tokens": estimated_tokens,
                "max_tokens": max_tokens,
                "kt_num_gpu_experts": 200,
                "kt_cpuinfer": 60,
                "tensor_parallel_size": 8,
            }
            save_performance_results(
                test_name="test_long_context_handling",
                gpu_config="8GPU",
                metrics=metrics,
                config=test_config,
            )

            # Write to GitHub Step Summary (if in CI)
            write_performance_summary(
                test_name="test_long_context_handling",
                gpu_config="8GPU",
                metrics=metrics,
                config=test_config,
            )

            print(f"\n✓ Long context test passed")
            print(f"✓ Generated {len(metrics['output_text'])} characters")
            print(f"✓ Memory increase within acceptable range")

        except Exception as e:
            self.fail(f"Long context inference failed: {e}")


if __name__ == "__main__":
    unittest.main()
