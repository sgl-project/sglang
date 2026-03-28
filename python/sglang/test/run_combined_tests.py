import os
import time
from typing import List, Optional, Union

from sglang.test.accuracy_test_runner import (
    AccuracyTestParams,
    AccuracyTestResult,
    run_accuracy_test,
    write_accuracy_github_summary,
    write_accuracy_results_json,
)
from sglang.test.nightly_utils import NightlyBenchmarkRunner
from sglang.test.performance_test_runner import (
    PerformanceTestParams,
    PerformanceTestResult,
    run_performance_test,
)
from sglang.test.test_utils import DEFAULT_URL_FOR_TEST, ModelLaunchSettings, is_in_ci
from sglang.test.tool_call_test_runner import (
    ToolCallTestParams,
    ToolCallTestResult,
    run_tool_call_test,
)


def _auto_profile_dir(is_vlm: bool, params: PerformanceTestParams) -> str:
    """Generate a profile directory name from workload parameters."""
    base = "performance_profiles_vlms" if is_vlm else "performance_profiles_text_models"
    if params.input_lens != (4096,) or params.output_lens != (512,):
        in_str = "_".join(str(x) for x in params.input_lens)
        out_str = "_".join(str(x) for x in params.output_lens)
        return f"{base}_in{in_str}_out{out_str}"
    return base


def run_combined_tests(
    models: List[ModelLaunchSettings],
    test_name: str = "NightlyTest",
    base_url: Optional[str] = None,
    is_vlm: bool = False,
    accuracy_params: Optional[
        Union[AccuracyTestParams, List[AccuracyTestParams]]
    ] = None,
    performance_params: Optional[
        Union[PerformanceTestParams, List[PerformanceTestParams]]
    ] = None,
    tool_call_params: Optional[ToolCallTestParams] = None,
) -> dict:
    """Run performance, accuracy, and/or tool call tests for a list of models.

    Args:
        models: List of ModelLaunchSettings to test
        test_name: Name for the test (used in reports)
        base_url: Server base URL (default: DEFAULT_URL_FOR_TEST)
        is_vlm: Whether these are VLM models (affects defaults)
        accuracy_params: AccuracyTestParams or list of them (None to skip accuracy).
            Pass a list to run multiple datasets (e.g., GSM8K + GPQA + MMMU).
        performance_params: PerformanceTestParams or list of them (None to skip perf).
            Pass a list to run multiple workload patterns (e.g., prefill-heavy +
            decode-heavy). Each entry uses its own profile directory.
        tool_call_params: Parameters for tool call tests (None to skip tool call)

    Returns:
        dict with test results:
        {
            "all_passed": bool,
            "results": [
                {
                    "model": str,
                    "perf_results": List[PerformanceTestResult],
                    "accuracy_results": List[AccuracyTestResult],
                    "tool_call_result": ToolCallTestResult/None,
                    "errors": list,
                },
                ...
            ]
        }
    """
    base_url = base_url or DEFAULT_URL_FOR_TEST

    # Normalize params to lists
    if isinstance(accuracy_params, AccuracyTestParams):
        accuracy_params_list = [accuracy_params]
    else:
        accuracy_params_list = accuracy_params or []

    if isinstance(performance_params, PerformanceTestParams):
        performance_params_list = [performance_params]
    else:
        performance_params_list = performance_params or []

    run_perf = len(performance_params_list) > 0
    run_accuracy = len(accuracy_params_list) > 0
    run_tool_call = tool_call_params is not None

    # Print test header
    print("\n" + "=" * 80)
    print(f"RUNNING: {test_name}")
    print(f"  Models: {len(models)}")
    if run_accuracy:
        datasets = [p.dataset for p in accuracy_params_list]
        print(f"  Accuracy datasets: {datasets}")
    if run_perf:
        for i, p in enumerate(performance_params_list):
            print(
                f"  Performance[{i}] batches={p.batch_sizes} "
                f"input_lens={p.input_lens} output_lens={p.output_lens}"
            )
    if run_tool_call:
        print("  Tool call tests: enabled")
    print("=" * 80)

    # Set up one perf runner per params set (each may have different workload/profile_dir)
    perf_runners = []
    for perf in performance_params_list:
        profile_dir = perf.profile_dir or _auto_profile_dir(is_vlm, perf)
        runner = NightlyBenchmarkRunner(
            profile_dir=profile_dir,
            test_name=test_name,
            base_url=base_url,
        )
        runner.setup_profile_directory()
        perf_runners.append((perf, runner))

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
            "perf_results": [],
            "accuracy_results": [],
            "tool_call_result": None,
            "errors": [],
        }

        # Run each performance workload
        for perf_params, perf_runner in perf_runners:
            perf_result: PerformanceTestResult = run_performance_test(
                model=model,
                perf_runner=perf_runner,
                batch_sizes=perf_params.batch_sizes,
                input_lens=perf_params.input_lens,
                output_lens=perf_params.output_lens,
                is_vlm=is_vlm,
                dataset_name=perf_params.dataset_name,
                spec_accept_length_threshold=perf_params.spec_accept_length_threshold,
            )
            model_result["perf_results"].append(perf_result)
            if not perf_result.passed:
                all_passed = False
                if perf_result.error:
                    model_result["errors"].append(perf_result.error)

            print("\nWaiting 20 seconds for resource cleanup...")
            time.sleep(20)

        # Run each accuracy dataset
        for acc_params in accuracy_params_list:
            acc_result: AccuracyTestResult = run_accuracy_test(
                model=model,
                params=acc_params,
                base_url=base_url,
            )
            model_result["accuracy_results"].append(acc_result)
            if not acc_result.passed:
                all_passed = False
                if acc_result.error:
                    model_result["errors"].append(acc_result.error)

            print("\nWaiting 20 seconds for resource cleanup...")
            time.sleep(20)

        # Run tool call test
        if run_tool_call:
            tc_result: ToolCallTestResult = run_tool_call_test(
                model=model,
                params=tool_call_params,
                base_url=base_url,
            )
            model_result["tool_call_result"] = tc_result
            if not tc_result.passed:
                all_passed = False
                model_result["errors"].extend(tc_result.failures)

            print("\nWaiting 20 seconds for resource cleanup...")
            time.sleep(20)

        all_results.append(model_result)

    # Write performance reports
    for _, perf_runner in perf_runners:
        perf_runner.write_final_report()

    # Write accuracy results to GitHub summary and JSON artifact
    if run_accuracy and is_in_ci():
        all_accuracy_results = [
            r for model_r in all_results for r in model_r["accuracy_results"]
        ]
        # One summary section per dataset
        for acc_params in accuracy_params_list:
            dataset_results = [
                r for r in all_accuracy_results if r.dataset == acc_params.dataset
            ]
            write_accuracy_github_summary(
                test_name, acc_params.dataset, dataset_results
            )

        # Emit JSON artifact for dashboard ingestion
        run_id = os.environ.get("GITHUB_RUN_ID", "")
        output_file = f"test/accuracy_results_{run_id or 'local'}.json"
        write_accuracy_results_json(all_accuracy_results, output_file)

    # Print summary
    print("\n" + "=" * 60)
    print(f"{test_name} Results Summary")
    if run_accuracy:
        print(f"Datasets: {[p.dataset for p in accuracy_params_list]}")
    print("=" * 60)
    for i, model_result in enumerate(all_results):
        print(f"\nModel {i + 1}: {model_result['model']}")
        for perf in model_result["perf_results"]:
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
        for acc in model_result["accuracy_results"]:
            score_str = f"{acc.score:.3f}" if acc.score is not None else "N/A"
            print(
                f"  Accuracy ({acc.dataset}): {'PASS' if acc.passed else 'FAIL'} score={score_str}"
            )
        if run_tool_call and model_result["tool_call_result"]:
            tc = model_result["tool_call_result"]
            print(
                f"  Tool Call: {'PASS' if tc.passed else 'FAIL'} ({tc.num_passed}/{tc.num_total})"
            )
        if model_result["errors"]:
            print(f"  Errors: {model_result['errors']}")

    print("\n" + "=" * 60)
    print(f"OVERALL: {'ALL TESTS PASSED' if all_passed else 'SOME TESTS FAILED'}")
    print("=" * 60 + "\n")

    # Raise assertion error if any test failed
    if not all_passed:
        failure_lines = []
        for i, r in enumerate(all_results):
            has_failed_test = (
                any(not p.passed for p in r.get("perf_results", []))
                or any(not a.passed for a in r.get("accuracy_results", []))
                or (r.get("tool_call_result") and not r["tool_call_result"].passed)
            )
            if r["errors"] or has_failed_test:
                failed_tests = []
                if any(not p.passed for p in r.get("perf_results", [])):
                    failed_tests.append("performance")
                failed_acc = [
                    a.dataset for a in r.get("accuracy_results", []) if not a.passed
                ]
                if failed_acc:
                    failed_tests.append(f"accuracy({', '.join(failed_acc)})")
                if r.get("tool_call_result") and not r["tool_call_result"].passed:
                    tc = r["tool_call_result"]
                    failed_tests.append(f"tool_call ({tc.num_passed}/{tc.num_total})")

                failed_test_str = ", ".join(failed_tests) if failed_tests else "unknown"
                error_str = "; ".join(str(e) for e in r["errors"])
                failure_lines.append(
                    f"  Model {i + 1} ({r['model']}): {failed_test_str} - {error_str}"
                )

        failure_summary = "\n".join(failure_lines)
        raise AssertionError(f"Tests failed:\n{failure_summary}")

    return {
        "all_passed": all_passed,
        "results": all_results,
    }
