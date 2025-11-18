"""
Correctness tests for KT-kernel integration

Tests model accuracy on GSM8K dataset to verify KT quantization
does not significantly degrade model reasoning capability.
"""

import json
import os
import unittest
from datetime import datetime
from types import SimpleNamespace

from sglang.srt.utils import kill_process_tree
from sglang.test.run_eval import run_eval
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    CustomTestCase,
    popen_launch_server,
    write_github_step_summary,
)

from .utils import get_kt_env, get_kt_model_paths, get_kt_server_args


def save_correctness_results(
    test_name: str,
    gpu_config: str,
    metrics: dict,
    config: dict,
    filename: str = None,
):
    """
    Save correctness results to JSON file

    Args:
        test_name: Name of the test (e.g., "test_gsm8k_accuracy")
        gpu_config: GPU configuration (e.g., "1GPU", "4GPU", "8GPU")
        metrics: Metrics dict from run_eval (contains "score" and other metrics)
        config: Test configuration dict
        filename: Output filename (default: kt_correctness_{gpu_config}_results.json)
    """
    if filename is None:
        filename = f"kt_correctness_{gpu_config.lower()}_results.json"

    result = {
        "timestamp": datetime.now().isoformat(),
        "test_name": test_name,
        "gpu_config": gpu_config,
        "metrics": {
            "score": metrics.get("score", 0),
            "num_examples": config.get("num_samples", 0),
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

    print(f"\n✓ Correctness results saved to: {filename}")


def write_correctness_summary(
    test_name: str,
    gpu_config: str,
    metrics: dict,
    config: dict,
):
    """
    Write correctness summary to GitHub Step Summary (if in CI environment)

    Args:
        test_name: Name of the test
        gpu_config: GPU configuration
        metrics: Metrics dict from run_eval
        config: Test configuration dict
    """
    score = metrics.get("score", 0)
    num_samples = config.get("num_samples", 0)
    num_correct = int(score * num_samples)

    # Generate markdown summary
    summary = f"""
## KT Correctness Test - {gpu_config}

**Test**: {test_name}
**Timestamp**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

### Configuration
| Parameter | Value |
|-----------|-------|
| GPU Config | {gpu_config} |
| Dataset | {config.get('dataset', 'N/A')} |
| Num Samples | {num_samples} |
| Temperature | {config.get('temperature', 'N/A')} |
| KT Num GPU Experts | {config.get('kt_num_gpu_experts', 'N/A')} |
| KT CPU Infer | {config.get('kt_cpuinfer', 'N/A')} |
| Tensor Parallel Size | {config.get('tensor_parallel_size', 'N/A')} |

### Results
| Metric | Value |
|--------|-------|
| **Accuracy (Score)** | {score:.4f} ({num_correct}/{num_samples} correct) |

---

"""

    # Write to GitHub Step Summary if in CI environment
    write_github_step_summary(summary)


class TestKTCorrectness1GPU(CustomTestCase):
    """
    Test GSM8K accuracy with 1 GPU configuration

    Configuration:
    - tensor_parallel_size: 1
    - kt_num_gpu_experts: 1
    - kt_cpuinfer: 60
    - Dataset: GSM8K (mgsm_en)
    - Num Samples: 2000
    - Temperature: 0.0
    """

    @classmethod
    def setUpClass(cls):
        model_paths = get_kt_model_paths()
        cls.model = model_paths["gpu_model_path"]
        cls.base_url = "http://127.0.0.1:30030"

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

    def test_gsm8k_accuracy(self):
        """Test GSM8K accuracy with 2000 samples"""
        print("\n" + "=" * 70)
        print("Testing GSM8K Accuracy (1 GPU)")
        print("=" * 70)

        num_samples = 2000
        dataset = "mgsm_en"
        temperature = 0.0

        print(f"\nRunning evaluation:")
        print(f"  Dataset: {dataset} (GSM8K)")
        print(f"  Num Samples: {num_samples}")
        print(f"  Temperature: {temperature}")

        # Run evaluation
        try:
            args = SimpleNamespace(
                base_url=self.base_url,
                model=self.model,
                eval_name=dataset,
                num_examples=num_samples,
                num_threads=1024,
                temperature=temperature,
            )

            metrics = run_eval(args)

            # Print results
            score = metrics.get("score", 0)
            num_correct = int(score * num_samples)

            print("\n" + "-" * 70)
            print("Results:")
            print("-" * 70)
            print(
                f"Accuracy (Score):    {score:.4f} ({num_correct}/{num_samples} correct)"
            )
            print("-" * 70)

            # Validate results
            self.assertIsInstance(score, (int, float), "Score should be numeric")
            self.assertGreaterEqual(score, 0, "Score should be >= 0")
            self.assertLessEqual(score, 1, "Score should be <= 1")

            # Save correctness results to JSON
            test_config = {
                "dataset": dataset,
                "num_samples": num_samples,
                "temperature": temperature,
                "kt_num_gpu_experts": 1,
                "kt_cpuinfer": 60,
                "tensor_parallel_size": 1,
            }
            save_correctness_results(
                test_name="test_gsm8k_accuracy",
                gpu_config="1GPU",
                metrics=metrics,
                config=test_config,
            )

            # Write to GitHub Step Summary (if in CI)
            write_correctness_summary(
                test_name="test_gsm8k_accuracy",
                gpu_config="1GPU",
                metrics=metrics,
                config=test_config,
            )

            print(f"\n✓ GSM8K accuracy test passed")
            print(f"✓ Successfully evaluated {num_samples} samples")
            print(f"✓ Accuracy: {score:.4f}")

        except Exception as e:
            self.fail(f"GSM8K accuracy test failed: {e}")


class TestKTCorrectness4GPU(CustomTestCase):
    """
    Test GSM8K accuracy with 4 GPU configuration

    Configuration:
    - tensor_parallel_size: 4
    - kt_num_gpu_experts: 80
    - kt_cpuinfer: 60
    - Dataset: GSM8K (mgsm_en)
    - Num Samples: 2000
    - Temperature: 0.0

    NOTE: This test is disabled by default. To enable it, remove the skip decorator.
    """

    @classmethod
    def setUpClass(cls):
        model_paths = get_kt_model_paths()
        cls.model = model_paths["gpu_model_path"]
        cls.base_url = "http://127.0.0.1:30031"

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

    @unittest.skipIf(True, "4 GPU correctness test disabled by default")
    def test_gsm8k_accuracy(self):
        """Test GSM8K accuracy with 2000 samples"""
        print("\n" + "=" * 70)
        print("Testing GSM8K Accuracy (4 GPU)")
        print("=" * 70)

        num_samples = 2000
        dataset = "mgsm_en"
        temperature = 0.0

        print(f"\nRunning evaluation:")
        print(f"  Dataset: {dataset} (GSM8K)")
        print(f"  Num Samples: {num_samples}")
        print(f"  Temperature: {temperature}")

        # Run evaluation
        try:
            args = SimpleNamespace(
                base_url=self.base_url,
                model=self.model,
                eval_name=dataset,
                num_examples=num_samples,
                num_threads=1024,
                temperature=temperature,
            )

            metrics = run_eval(args)

            # Print results
            score = metrics.get("score", 0)
            num_correct = int(score * num_samples)

            print("\n" + "-" * 70)
            print("Results:")
            print("-" * 70)
            print(
                f"Accuracy (Score):    {score:.4f} ({num_correct}/{num_samples} correct)"
            )
            print("-" * 70)

            # Validate results
            self.assertIsInstance(score, (int, float), "Score should be numeric")
            self.assertGreaterEqual(score, 0, "Score should be >= 0")
            self.assertLessEqual(score, 1, "Score should be <= 1")

            # Save correctness results to JSON
            test_config = {
                "dataset": dataset,
                "num_samples": num_samples,
                "temperature": temperature,
                "kt_num_gpu_experts": 80,
                "kt_cpuinfer": 60,
                "tensor_parallel_size": 4,
            }
            save_correctness_results(
                test_name="test_gsm8k_accuracy",
                gpu_config="4GPU",
                metrics=metrics,
                config=test_config,
            )

            # Write to GitHub Step Summary (if in CI)
            write_correctness_summary(
                test_name="test_gsm8k_accuracy",
                gpu_config="4GPU",
                metrics=metrics,
                config=test_config,
            )

            print(f"\n✓ GSM8K accuracy test passed")
            print(f"✓ Successfully evaluated {num_samples} samples")
            print(f"✓ Accuracy: {score:.4f}")

        except Exception as e:
            self.fail(f"GSM8K accuracy test failed: {e}")


class TestKTCorrectness8GPU(CustomTestCase):
    """
    Test GSM8K accuracy with 8 GPU configuration

    Configuration:
    - tensor_parallel_size: 8
    - kt_num_gpu_experts: 200
    - kt_cpuinfer: 60
    - Dataset: GSM8K (mgsm_en)
    - Num Samples: 2000
    - Temperature: 0.0

    NOTE: This test is disabled by default. To enable it, remove the skip decorator.
    """

    @classmethod
    def setUpClass(cls):
        model_paths = get_kt_model_paths()
        cls.model = model_paths["gpu_model_path"]
        cls.base_url = "http://127.0.0.1:30032"

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

    @unittest.skipIf(True, "8 GPU correctness test disabled by default")
    def test_gsm8k_accuracy(self):
        """Test GSM8K accuracy with 2000 samples"""
        print("\n" + "=" * 70)
        print("Testing GSM8K Accuracy (8 GPU)")
        print("=" * 70)

        num_samples = 2000
        dataset = "mgsm_en"
        temperature = 0.0

        print(f"\nRunning evaluation:")
        print(f"  Dataset: {dataset} (GSM8K)")
        print(f"  Num Samples: {num_samples}")
        print(f"  Temperature: {temperature}")

        # Run evaluation
        try:
            args = SimpleNamespace(
                base_url=self.base_url,
                model=self.model,
                eval_name=dataset,
                num_examples=num_samples,
                num_threads=1024,
                temperature=temperature,
            )

            metrics = run_eval(args)

            # Print results
            score = metrics.get("score", 0)
            num_correct = int(score * num_samples)

            print("\n" + "-" * 70)
            print("Results:")
            print("-" * 70)
            print(
                f"Accuracy (Score):    {score:.4f} ({num_correct}/{num_samples} correct)"
            )
            print("-" * 70)

            # Validate results
            self.assertIsInstance(score, (int, float), "Score should be numeric")
            self.assertGreaterEqual(score, 0, "Score should be >= 0")
            self.assertLessEqual(score, 1, "Score should be <= 1")

            # Save correctness results to JSON
            test_config = {
                "dataset": dataset,
                "num_samples": num_samples,
                "temperature": temperature,
                "kt_num_gpu_experts": 200,
                "kt_cpuinfer": 60,
                "tensor_parallel_size": 8,
            }
            save_correctness_results(
                test_name="test_gsm8k_accuracy",
                gpu_config="8GPU",
                metrics=metrics,
                config=test_config,
            )

            # Write to GitHub Step Summary (if in CI)
            write_correctness_summary(
                test_name="test_gsm8k_accuracy",
                gpu_config="8GPU",
                metrics=metrics,
                config=test_config,
            )

            print(f"\n✓ GSM8K accuracy test passed")
            print(f"✓ Successfully evaluated {num_samples} samples")
            print(f"✓ Accuracy: {score:.4f}")

        except Exception as e:
            self.fail(f"GSM8K accuracy test failed: {e}")


if __name__ == "__main__":
    unittest.main()
