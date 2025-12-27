from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from sglang.test.nightly_bench_utils import BenchmarkResult
from sglang.test.nightly_utils import NightlyBenchmarkRunner
from sglang.test.test_utils import DEFAULT_URL_FOR_TEST, ModelLaunchSettings


@dataclass
class PerformanceTestParams:
    """Parameters for performance testing."""

    batch_sizes: List[int] = field(default_factory=lambda: [1, 8, 16, 64])
    input_lens: Tuple[int, ...] = (4096,)
    output_lens: Tuple[int, ...] = (512,)
    profile_dir: Optional[str] = None  # None = auto-generate based on is_vlm
    dataset_name: str = "mmmu"  # For VLM perf test
    # MTP/EAGLE speculative decoding: minimum accept length threshold (None = no validation)
    spec_accept_length_threshold: Optional[float] = None


@dataclass
class PerformanceTestResult:
    """Result of a performance test.

    Aggregates metrics across all batch sizes tested for a single model.
    """

    model: str
    passed: bool
    error: Optional[str]
    # Aggregate metrics (from the largest batch size result, or None if failed)
    latency: Optional[float] = None
    input_throughput: Optional[float] = None
    output_throughput: Optional[float] = None
    overall_throughput: Optional[float] = None
    # All individual benchmark results
    benchmark_results: Optional[List[BenchmarkResult]] = None
    # MTP/EAGLE speculative decoding metric
    avg_spec_accept_length: Optional[float] = None


def run_performance_test(
    model: ModelLaunchSettings,
    perf_runner: NightlyBenchmarkRunner,
    batch_sizes: List[int] = None,
    input_lens: Tuple[int, ...] = (4096,),
    output_lens: Tuple[int, ...] = (512,),
    is_vlm: bool = False,
    dataset_name: str = "mmmu",
    spec_accept_length_threshold: Optional[float] = None,
) -> PerformanceTestResult:

    # Set default for mutable argument
    if batch_sizes is None:
        batch_sizes = [1, 8, 16, 64]

    print(f"\n{'='*60}")
    print(f"Running PERFORMANCE test for {model.model_path}")
    print(f"  Batch sizes: {batch_sizes}")
    print(f"  Input lens: {input_lens}")
    print(f"  Output lens: {output_lens}")
    if spec_accept_length_threshold is not None:
        print(f"  Spec accept length threshold: {spec_accept_length_threshold}")
    print(f"{'='*60}\n")

    # Build extra args for benchmarks
    extra_bench_args = ["--trust-remote-code"]
    if is_vlm:
        extra_bench_args.append(f"--dataset-name={dataset_name}")

    try:
        results, success, avg_spec_accept_length = perf_runner.run_benchmark_for_model(
            model_path=model.model_path,
            batch_sizes=batch_sizes,
            input_lens=input_lens,
            output_lens=output_lens,
            other_args=model.extra_args,
            extra_bench_args=extra_bench_args,
        )

        if success and results:
            perf_runner.add_report(results)
            print(f"✓ Performance test succeeded for {model.model_path}")

            # Validate speculative decoding accept length if threshold is set
            error_msg = None
            passed = True
            if spec_accept_length_threshold is not None:
                if avg_spec_accept_length is None:
                    error_msg = f"Spec accept length threshold set but no accept length reported"
                    passed = False
                    print(f"✗ {error_msg}")
                elif avg_spec_accept_length < spec_accept_length_threshold:
                    error_msg = (
                        f"Spec accept length {avg_spec_accept_length:.2f} < "
                        f"threshold {spec_accept_length_threshold}"
                    )
                    passed = False
                    print(f"✗ {error_msg}")
                else:
                    print(
                        f"✓ Spec accept length {avg_spec_accept_length:.2f} >= "
                        f"threshold {spec_accept_length_threshold}"
                    )

            # Extract aggregate metrics from the largest batch size result
            largest_batch_result = max(results, key=lambda r: r.batch_size)
            return PerformanceTestResult(
                model=model.model_path,
                passed=passed,
                error=error_msg,
                latency=largest_batch_result.latency,
                input_throughput=largest_batch_result.input_throughput,
                output_throughput=largest_batch_result.output_throughput,
                overall_throughput=largest_batch_result.overall_throughput,
                benchmark_results=results,
                avg_spec_accept_length=avg_spec_accept_length,
            )
        else:
            error_msg = f"Performance test failed for {model.model_path}"
            print(f"✗ {error_msg}")
            return PerformanceTestResult(
                model=model.model_path,
                passed=False,
                error=error_msg,
            )

    except Exception as e:
        error_msg = f"Performance test exception for {model.model_path}: {str(e)}"
        print(f"✗ {error_msg}")
        return PerformanceTestResult(
            model=model.model_path,
            passed=False,
            error=error_msg,
        )


def run_performance_for_models(
    models: List[ModelLaunchSettings],
    profile_dir: str,
    test_name: str,
    base_url: Optional[str] = None,
    batch_sizes: List[int] = None,
    input_lens: Tuple[int, ...] = (4096,),
    output_lens: Tuple[int, ...] = (512,),
    is_vlm: bool = False,
    dataset_name: str = "mmmu",
) -> dict:
    """Run performance tests for multiple models.

    Args:
        models: List of ModelLaunchSettings to test
        profile_dir: Directory for performance profiles
        test_name: Name for the test (used in reports)
        base_url: Server base URL (default: DEFAULT_URL_FOR_TEST)
        batch_sizes: Batch sizes for perf test
        input_lens: Input lengths
        output_lens: Output lengths
        is_vlm: Whether these are VLM models
        dataset_name: Dataset name for VLM benchmarks

    Returns:
        dict with results:
        {
            "all_passed": bool,
            "results": [PerformanceTestResult, ...]
        }
    """
    base_url = base_url or DEFAULT_URL_FOR_TEST

    # Setup performance runner
    perf_runner = NightlyBenchmarkRunner(
        profile_dir=profile_dir,
        test_name=test_name,
        base_url=base_url,
    )
    perf_runner.setup_profile_directory()

    all_results = []
    all_passed = True

    for model in models:
        print("\n" + "=" * 80)
        print(f"PERFORMANCE TEST: {model.model_path}")
        print(f"  TP Size: {model.tp_size}")
        print(f"  Extra Args: {model.extra_args}")
        print("=" * 80)

        result = run_performance_test(
            model=model,
            perf_runner=perf_runner,
            batch_sizes=batch_sizes,
            input_lens=input_lens,
            output_lens=output_lens,
            is_vlm=is_vlm,
            dataset_name=dataset_name,
        )

        all_results.append(result)

        if not result.passed:
            all_passed = False

    # Write performance report
    perf_runner.write_final_report()

    # Print summary
    print("\n" + "=" * 60)
    print(f"Performance Test Summary: {test_name}")
    print("=" * 60)
    for result in all_results:
        status = "PASS" if result.passed else "FAIL"
        throughput_str = (
            f", output: {result.output_throughput:.1f} tok/s"
            if result.output_throughput
            else ""
        )
        print(f"  {result.model}: {status}{throughput_str}")
        if result.error:
            print(f"    Error: {result.error}")

    print("\n" + "=" * 60)
    print(f"OVERALL: {'ALL PASSED' if all_passed else 'SOME FAILED'}")
    print("=" * 60 + "\n")

    return {
        "all_passed": all_passed,
        "results": all_results,
    }
