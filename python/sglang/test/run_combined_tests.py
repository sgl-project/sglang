import time
from typing import List, Optional

from sglang.test.accuracy_test_runner import (
    AccuracyTestParams,
    AccuracyTestResult,
    run_accuracy_test,
    write_accuracy_github_summary,
)
from sglang.test.nightly_utils import NightlyBenchmarkRunner
from sglang.test.performance_test_runner import (
    PerformanceTestParams,
    PerformanceTestResult,
    run_performance_test,
)
from sglang.test.test_utils import DEFAULT_URL_FOR_TEST, ModelLaunchSettings, is_in_ci


def run_combined_tests(
    models: List[ModelLaunchSettings],
    test_name: str = "NightlyTest",
    base_url: Optional[str] = None,
    is_vlm: bool = False,
    accuracy_params: Optional[AccuracyTestParams] = None,
    performance_params: Optional[PerformanceTestParams] = None,
) -> dict:
    """Run performance and/or accuracy tests for a list of models.

    Args:
        models: List of ModelLaunchSettings to test
        test_name: Name for the test (used in reports)
        base_url: Server base URL (default: DEFAULT_URL_FOR_TEST)
        is_vlm: Whether these are VLM models (affects defaults)
        accuracy_params: Parameters for accuracy tests (None to skip accuracy)
        performance_params: Parameters for performance tests (None to skip perf)

    Returns:
        dict with test results:
        {
            "all_passed": bool,
            "results": [
                {
                    "model": str,
                    "perf_result": PerformanceTestResult/None,
                    "accuracy_result": AccuracyTestResult/None,
                    "errors": list,
                },
                ...
            ]
        }
    """
    base_url = base_url or DEFAULT_URL_FOR_TEST
    run_perf = performance_params is not None
    run_accuracy = accuracy_params is not None

    # Print test header
    print("\n" + "=" * 80)
    print(f"RUNNING: {test_name}")
    print(f"  Models: {len(models)}")
    if run_accuracy:
        print(f"  Accuracy dataset: {accuracy_params.dataset}")
    if run_perf:
        print(f"  Performance batches: {performance_params.batch_sizes}")
    print("=" * 80)

    # Set up performance parameters
    if run_perf:
        perf = performance_params
        profile_dir = perf.profile_dir or (
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
    else:
        perf_runner = None

    # Run tests for each model
    all_results = []
    all_passed = True

    for model in models:
        print("\n" + "=" * 80)
        print(f"TESTING MODEL CONFIG: {model.model_path}")
        print(f"  TP Size: {model.tp_size}")
        print(f"  Extra Args: {model.extra_args}")
        print("=" * 80)

        model_result = {
            "model": model.model_path,
            "perf_result": None,
            "accuracy_result": None,
            "errors": [],
        }

        # Run performance test
        if run_perf:
            perf_result: PerformanceTestResult = run_performance_test(
                model=model,
                perf_runner=perf_runner,
                batch_sizes=performance_params.batch_sizes,
                input_lens=performance_params.input_lens,
                output_lens=performance_params.output_lens,
                is_vlm=is_vlm,
                dataset_name=performance_params.dataset_name,
                spec_accept_length_threshold=performance_params.spec_accept_length_threshold,
            )
            model_result["perf_result"] = perf_result
            if not perf_result.passed:
                all_passed = False
                model_result["errors"].append(perf_result.error)

            # Wait for GPU memory and port cleanup
            print("\nWaiting 20 seconds for resource cleanup...")
            time.sleep(20)

        # Run accuracy test
        if run_accuracy:
            acc_result: AccuracyTestResult = run_accuracy_test(
                model=model,
                params=accuracy_params,
                base_url=base_url,
            )
            model_result["accuracy_result"] = acc_result
            if not acc_result.passed:
                all_passed = False
                model_result["errors"].append(acc_result.error)

            # Wait for GPU memory and port cleanup
            print("\nWaiting 20 seconds for resource cleanup...")
            time.sleep(20)

        all_results.append(model_result)

    # Write performance report if we ran perf tests
    if run_perf and perf_runner:
        perf_runner.write_final_report()

    # Write accuracy results to GitHub summary if in CI
    if run_accuracy and is_in_ci():
        accuracy_results = [
            r["accuracy_result"] for r in all_results if r["accuracy_result"]
        ]
        write_accuracy_github_summary(
            test_name, accuracy_params.dataset, accuracy_results
        )

    # Print summary
    print("\n" + "=" * 60)
    print(f"{test_name} Results Summary")
    if run_accuracy:
        print(f"Dataset: {accuracy_params.dataset}")
        print(f"Baseline: {accuracy_params.baseline_accuracy}")
    print("=" * 60)
    for i, model_result in enumerate(all_results):
        print(f"\nModel {i + 1}: {model_result['model']}")
        if run_perf and model_result["perf_result"]:
            perf = model_result["perf_result"]
            throughput_str = (
                f", output: {perf.output_throughput:.1f} tok/s"
                if perf.output_throughput
                else ""
            )
            accept_str = (
                f", accept_len: {perf.avg_spec_accept_length:.2f}"
                if perf.avg_spec_accept_length
                else ""
            )
            print(
                f"  Performance: {'PASS' if perf.passed else 'FAIL'}{throughput_str}{accept_str}"
            )
        if run_accuracy and model_result["accuracy_result"]:
            acc = model_result["accuracy_result"]
            print(f"  Accuracy: {'PASS' if acc.passed else 'FAIL'}")
            if acc.score is not None:
                print(f"  Score: {acc.score:.3f}")
        if model_result["errors"]:
            print(f"  Errors: {model_result['errors']}")

    print("\n" + "=" * 60)
    print(f"OVERALL: {'ALL TESTS PASSED' if all_passed else 'SOME TESTS FAILED'}")
    print("=" * 60 + "\n")

    # Raise assertion error if any test failed
    if not all_passed:
        failed_models = [r["model"] for r in all_results if r["errors"]]
        raise AssertionError(
            f"Tests failed for models: {failed_models}. See results above for details."
        )

    return {
        "all_passed": all_passed,
        "results": all_results,
    }
