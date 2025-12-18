"""Generic metrics runner for running performance + accuracy on models.

This provides a simple way to run both performance and accuracy tests on a list of models,
using the same abstractions as the existing nightly tests:
- NightlyBenchmarkRunner for performance
- run_eval (SimpleNamespace) for accuracy

Hardware is determined by GitHub Actions jobs, not by this layer.

Usage:
    from nightly_metrics import run_metrics
    from sglang.test.test_utils import ModelLaunchSettings

    models = [
        ModelLaunchSettings("meta-llama/Llama-3.1-8B-Instruct", tp_size=1),
        ModelLaunchSettings("Qwen/Qwen2-57B-A14B-Instruct", tp_size=2),
    ]

    # Run both perf + accuracy for all models
    result = run_metrics(
        models=models,
        run_perf=True,
        run_accuracy=True,
        is_vlm=False,
    )
"""

import gc
import time
from types import SimpleNamespace
from typing import List, Optional, Tuple

from nightly_utils import NightlyBenchmarkRunner

from sglang.srt.utils import kill_process_tree
from sglang.test.run_eval import run_eval

try:
    import torch
except ImportError:
    torch = None
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    ModelLaunchSettings,
    _parse_int_list_env,
    popen_launch_server,
)


def cleanup_between_tests(delay_seconds: int = 10) -> None:
    """Clean up resources between performance and accuracy tests.

    This helps ensure GPU memory is freed and ports are released before
    starting the next test. Addresses transient failures where the accuracy
    test server fails to start after a performance test.

    Args:
        delay_seconds: Time to wait for resource cleanup (default: 10s)
    """
    print(f"\n{'='*60}")
    print("Cleaning up resources between tests...")
    print(f"{'='*60}")

    # Force Python garbage collection
    gc.collect()

    # Clear CUDA cache if torch is available
    if torch is not None and torch.cuda.is_available():
        try:
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            print("  - Cleared CUDA cache")
        except Exception as e:
            print(f"  - Warning: Could not clear CUDA cache: {e}")

    # Wait for resources to be released (port unbinding, GPU memory deallocation)
    print(f"  - Waiting {delay_seconds}s for resources to release...")
    time.sleep(delay_seconds)
    print("  - Cleanup complete\n")


def run_performance_for_model(
    model: ModelLaunchSettings,
    perf_runner: NightlyBenchmarkRunner,
    batch_sizes: List[int],
    input_lens: Tuple[int, ...],
    output_lens: Tuple[int, ...],
    is_vlm: bool = False,
    dataset_name: str = "mmmu",
) -> Tuple[bool, Optional[str]]:
    """Run performance test for a single model using NightlyBenchmarkRunner.

    Args:
        model: ModelLaunchSettings with model config
        perf_runner: NightlyBenchmarkRunner instance
        batch_sizes: Batch sizes for perf test
        input_lens: Input lengths
        output_lens: Output lengths
        is_vlm: Whether this is a VLM model
        dataset_name: Dataset name for VLM benchmarks

    Returns:
        Tuple of (success, error_message)
    """
    print(f"\n{'='*60}")
    print(f"Running PERFORMANCE test for {model.model_path}")
    print(f"{'='*60}\n")

    # Build extra args for benchmarks
    # Always include --trust-remote-code for models that need custom code
    extra_bench_args = ["--trust-remote-code"]
    if is_vlm:
        extra_bench_args.append(f"--dataset-name={dataset_name}")

    try:
        results, success = perf_runner.run_benchmark_for_model(
            model_path=model.model_path,
            batch_sizes=batch_sizes,
            input_lens=input_lens,
            output_lens=output_lens,
            other_args=model.extra_args,
            extra_bench_args=extra_bench_args,
        )

        if success:
            perf_runner.add_report(results)
            print(f"✓ Performance test succeeded for {model.model_path}")
            return True, None
        else:
            error_msg = f"Performance test failed for {model.model_path}"
            print(f"✗ {error_msg}")
            return False, error_msg

    except Exception as e:
        error_msg = f"Performance test exception for {model.model_path}: {str(e)}"
        print(f"✗ {error_msg}")
        return False, error_msg


def run_accuracy_for_model(
    model: ModelLaunchSettings,
    base_url: str,
    eval_name: str,
    num_examples: Optional[int],
    num_threads: int,
    max_tokens: Optional[int] = None,
    return_latency: bool = False,
    **eval_kwargs,
) -> Tuple[bool, Optional[str], Optional[dict]]:
    """Run accuracy test for a single model using run_eval (SimpleNamespace).

    Args:
        model: ModelLaunchSettings with model config
        base_url: Server base URL
        eval_name: Evaluation name (e.g., "mgsm_en", "mmmu", "gpqa")
        num_examples: Number of examples (None means all)
        num_threads: Number of threads for evaluation
        max_tokens: Max tokens for generation
        return_latency: Whether to return latency metrics
        **eval_kwargs: Additional kwargs to pass to run_eval

    Returns:
        Tuple of (success, error_message, metrics_dict)
    """
    print(f"\n{'='*60}")
    print(f"Running ACCURACY test for {model.model_path}")
    print(f"  Eval: {eval_name}, Examples: {num_examples}")
    print(f"{'='*60}\n")

    process = None
    try:
        # Launch server
        process = popen_launch_server(
            model=model.model_path,
            base_url=base_url,
            other_args=model.extra_args,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
        )

        # Build eval args
        args = SimpleNamespace(
            base_url=base_url,
            model=model.model_path,
            eval_name=eval_name,
            num_examples=num_examples,
            num_threads=num_threads,
        )

        # Add optional parameters
        if max_tokens is not None:
            args.max_tokens = max_tokens

        if return_latency:
            args.return_latency = True

        # Add any extra kwargs
        for key, value in eval_kwargs.items():
            setattr(args, key, value)

        # Run evaluation
        result = run_eval(args)

        # Handle result format (run_eval can return metrics or (metrics, latency))
        if return_latency and isinstance(result, tuple):
            metrics, latency = result
            metrics["latency"] = round(latency, 4)
        else:
            metrics = result

        print(f"✓ Accuracy test succeeded for {model.model_path}: {metrics}")
        return True, None, metrics

    except Exception as e:
        error_msg = f"Accuracy test exception for {model.model_path}: {str(e)}"
        print(f"✗ {error_msg}")
        return False, error_msg, None

    finally:
        if process:
            kill_process_tree(process.pid)


def run_metrics(
    models: List[ModelLaunchSettings],
    run_perf: bool = True,
    run_accuracy: bool = True,
    is_vlm: bool = False,
    base_url: Optional[str] = None,
    profile_dir: Optional[str] = None,
    test_name: str = "NightlyMetrics",
    # Performance test parameters
    batch_sizes: Optional[List[int]] = None,
    input_lens: Optional[Tuple[int, ...]] = None,
    output_lens: Optional[Tuple[int, ...]] = None,
    # Accuracy test parameters
    eval_name: Optional[str] = None,
    num_examples: Optional[int] = None,
    num_threads: int = 1024,
    max_tokens: Optional[int] = None,
    # VLM-specific parameters
    dataset_name: Optional[str] = None,
    return_latency: bool = False,
    # Additional eval kwargs
    **eval_kwargs,
) -> dict:
    """Run performance and/or accuracy tests for a list of models.

    This function uses the same abstractions as existing nightly tests:
    - NightlyBenchmarkRunner for performance tests
    - run_eval (SimpleNamespace) for accuracy tests

    Args:
        models: List of ModelLaunchSettings to test
        run_perf: Whether to run performance tests
        run_accuracy: Whether to run accuracy tests
        is_vlm: Whether these are VLM models (affects defaults)
        base_url: Server base URL (default: DEFAULT_URL_FOR_TEST)
        profile_dir: Directory for performance profiles
        test_name: Name for the test (used in reports)

        # Performance test parameters:
        batch_sizes: Batch sizes for perf test (default: [1, 1, 8, 16, 64])
        input_lens: Input lengths (default: from NIGHTLY_INPUT_LENS env or "4096")
        output_lens: Output lengths (default: from NIGHTLY_OUTPUT_LENS env or "512")

        # Accuracy test parameters:
        eval_name: Evaluation name (default: "mgsm_en" for text, "mmmu" for VLM)
        num_examples: Number of examples (default: None for text, 100 for VLM)
        num_threads: Number of threads for evaluation (default: 1024)
        max_tokens: Max tokens for generation

        # VLM-specific:
        dataset_name: Dataset name for VLM benchmarks (default: "mmmu")
        return_latency: Whether to return latency metrics for VLM eval

        # Additional eval parameters:
        **eval_kwargs: Any additional parameters to pass to run_eval
                       (e.g., thinking_mode, temperature, repeat)

    Returns:
        dict with test results:
        {
            "all_passed": bool,
            "results": [
                {
                    "model": str,
                    "perf_passed": bool/None,
                    "accuracy_passed": bool/None,
                    "accuracy_metrics": dict/None,
                    "errors": list,
                },
                ...
            ]
        }

    Example:
        # Text models - run both perf + accuracy
        models = [
            ModelLaunchSettings("meta-llama/Llama-3.1-8B-Instruct", tp_size=1),
        ]
        result = run_metrics(models=models, run_perf=True, run_accuracy=True, is_vlm=False)

        # VLM models - run both perf + accuracy
        models = [
            ModelLaunchSettings("Qwen/Qwen2.5-VL-7B-Instruct"),
        ]
        result = run_metrics(
            models=models,
            run_perf=True,
            run_accuracy=True,
            is_vlm=True,
            return_latency=True,
        )

        # Custom eval - GPQA with thinking mode
        models = [ModelLaunchSettings("deepseek-ai/DeepSeek-V3.2", tp_size=8)]
        result = run_metrics(
            models=models,
            run_perf=False,
            run_accuracy=True,
            is_vlm=False,
            eval_name="gpqa",
            num_examples=198,
            max_tokens=120000,
            thinking_mode="deepseek-v3",
            temperature=0.1,
            repeat=4,
        )
    """
    base_url = base_url or DEFAULT_URL_FOR_TEST

    # Set defaults based on is_vlm
    batch_sizes = batch_sizes or [1, 1, 8, 16, 64]
    input_lens = input_lens or tuple(_parse_int_list_env("NIGHTLY_INPUT_LENS", "4096"))
    output_lens = output_lens or tuple(
        _parse_int_list_env("NIGHTLY_OUTPUT_LENS", "512")
    )

    if is_vlm:
        eval_name = eval_name or "mmmu"
        num_examples = num_examples if num_examples is not None else 100
        max_tokens = max_tokens if max_tokens is not None else 30
        dataset_name = dataset_name or "mmmu"
    else:
        eval_name = eval_name or "mgsm_en"
        # num_examples, max_tokens stay as provided or None
        dataset_name = dataset_name or "mmmu"

    # Setup performance runner if needed
    perf_runner = None
    if run_perf:
        profile_dir = profile_dir or (
            "performance_profiles_vlms"
            if is_vlm
            else "performance_profiles_text_models"
        )
        perf_runner = NightlyBenchmarkRunner(
            profile_dir=profile_dir,
            test_name=test_name,
            base_url=base_url,
        )
        perf_runner.setup_profile_directory()

    # Run tests for each model
    all_results = []
    all_passed = True

    for model in models:
        # Print configuration being tested
        print("\n" + "=" * 80)
        print(f"TESTING MODEL CONFIG: {model.model_path}")
        print(f"  TP Size: {model.tp_size}")
        print(f"  Extra Args: {model.extra_args}")
        print("=" * 80)

        model_result = {
            "model": model.model_path,
            "perf_passed": None,
            "accuracy_passed": None,
            "accuracy_metrics": None,
            "errors": [],
        }

        # Run performance test
        if run_perf:
            perf_success, perf_error = run_performance_for_model(
                model=model,
                perf_runner=perf_runner,
                batch_sizes=batch_sizes,
                input_lens=input_lens,
                output_lens=output_lens,
                is_vlm=is_vlm,
                dataset_name=dataset_name,
            )
            model_result["perf_passed"] = perf_success
            if not perf_success:
                all_passed = False
                model_result["errors"].append(perf_error)

        # Clean up between performance and accuracy tests to ensure
        # GPU memory is freed and port is released
        if run_perf and run_accuracy:
            cleanup_between_tests(delay_seconds=10)

        # Run accuracy test
        if run_accuracy:
            acc_success, acc_error, metrics = run_accuracy_for_model(
                model=model,
                base_url=base_url,
                eval_name=eval_name,
                num_examples=num_examples,
                num_threads=num_threads,
                max_tokens=max_tokens,
                return_latency=return_latency,
                **eval_kwargs,
            )
            model_result["accuracy_passed"] = acc_success
            model_result["accuracy_metrics"] = metrics
            if not acc_success:
                all_passed = False
                model_result["errors"].append(acc_error)

        all_results.append(model_result)

        # Clean up between models to ensure resources are released
        cleanup_between_tests(delay_seconds=10)

    # Write performance report if we ran perf tests
    if run_perf and perf_runner:
        perf_runner.write_final_report()

    # Print summary
    print("\n" + "=" * 60)
    print(f"{test_name} Results Summary")
    print("=" * 60)
    for i, model_result in enumerate(all_results):
        print(f"\nModel {i + 1}: {model_result['model']}")
        if run_perf:
            print(f"  Performance: {'PASS' if model_result['perf_passed'] else 'FAIL'}")
        if run_accuracy:
            print(
                f"  Accuracy: {'PASS' if model_result['accuracy_passed'] else 'FAIL'}"
            )
            if model_result["accuracy_metrics"]:
                print(f"  Metrics: {model_result['accuracy_metrics']}")
        if model_result["errors"]:
            print(f"  Errors: {model_result['errors']}")

    print("\n" + "=" * 60)
    print(f"OVERALL: {'ALL TESTS PASSED' if all_passed else 'SOME TESTS FAILED'}")
    print("=" * 60 + "\n")

    # Raise assertion error if any test failed (so unittest marks it as failure)
    if not all_passed:
        failed_models = [r["model"] for r in all_results if r["errors"]]
        raise AssertionError(
            f"Tests failed for models: {failed_models}. See results above for details."
        )

    return {
        "all_passed": all_passed,
        "results": all_results,
    }
