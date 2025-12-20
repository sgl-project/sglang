from dataclasses import dataclass
from types import SimpleNamespace
from typing import List, Optional, Tuple

from sglang.srt.utils import kill_process_tree
from sglang.test.run_eval import run_eval
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    ModelLaunchSettings,
    popen_launch_server,
)


@dataclass
class AccuracyTestParams:
    """Parameters for accuracy testing."""

    dataset: str  # e.g., "mgsm_en", "gsm8k", "mmmu", "gpqa"
    baseline_accuracy: float  # Required: minimum accuracy threshold
    num_examples: Optional[int] = None
    num_threads: Optional[int] = None
    max_tokens: Optional[int] = None
    return_latency: bool = False
    # Extended parameters for special evaluations (e.g., GPQA with thinking mode)
    thinking_mode: Optional[str] = None  # e.g., "deepseek-v3"
    temperature: Optional[float] = None
    repeat: Optional[int] = None


@dataclass
class AccuracyTestResult:
    """Result of an accuracy test."""

    model: str
    dataset: str
    passed: bool
    score: Optional[float]
    baseline_accuracy: float
    error: Optional[str]
    latency: Optional[float] = None


def _run_simple_eval(
    model: ModelLaunchSettings,
    base_url: str,
    dataset: str,
    num_examples: Optional[int] = None,
    num_threads: Optional[int] = None,
    max_tokens: Optional[int] = None,
    return_latency: bool = False,
    thinking_mode: Optional[str] = None,
    temperature: Optional[float] = None,
    repeat: Optional[int] = None,
) -> Tuple[bool, Optional[str], Optional[dict]]:
    """Run evaluation using simple_eval backend (run_eval.py).

    Returns:
        Tuple of (success, error_message, metrics_dict)
    """
    process = None
    try:
        process = popen_launch_server(
            model.model_path,
            base_url,
            other_args=model.extra_args,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
        )

        args = SimpleNamespace(
            base_url=base_url,
            model=model.model_path,
            eval_name=dataset,
            num_examples=num_examples,
            num_threads=num_threads or 1024,
        )

        if max_tokens is not None:
            args.max_tokens = max_tokens

        if return_latency:
            args.return_latency = True

        if thinking_mode is not None:
            args.thinking_mode = thinking_mode

        if temperature is not None:
            args.temperature = temperature

        if repeat is not None:
            args.repeat = repeat

        result = run_eval(args)

        # Handle result format (run_eval can return metrics or (metrics, latency))
        if return_latency and isinstance(result, tuple):
            metrics, latency = result
            metrics["latency"] = round(latency, 4)
        else:
            metrics = result

        return True, None, metrics

    except Exception as e:
        return False, f"Accuracy test exception: {str(e)}", None

    finally:
        if process:
            kill_process_tree(process.pid)


def _run_few_shot_eval(
    model: ModelLaunchSettings,
    base_url: str,
    num_questions: Optional[int] = None,
    num_shots: int = 8,
    max_tokens: int = 512,
) -> Tuple[bool, Optional[str], Optional[dict]]:
    """Run evaluation using few_shot backend (few_shot_gsm8k.py).

    Returns:
        Tuple of (success, error_message, metrics_dict)
    """
    from sglang.test.few_shot_gsm8k import run_eval as run_few_shot_eval

    process = None
    try:
        process = popen_launch_server(
            model.model_path,
            base_url,
            other_args=model.extra_args,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
        )

        args = SimpleNamespace(
            num_shots=num_shots,
            data_path=None,
            num_questions=num_questions or 200,
            max_new_tokens=max_tokens,
            parallel=128,
            host="http://127.0.0.1",
            port=int(base_url.split(":")[-1]),
        )

        metrics = run_few_shot_eval(args)

        # Normalize metrics format (few_shot returns "accuracy", simple_eval returns "score")
        if "accuracy" in metrics and "score" not in metrics:
            metrics["score"] = metrics["accuracy"]

        return True, None, metrics

    except Exception as e:
        return False, f"Few-shot evaluation exception: {str(e)}", None

    finally:
        if process:
            kill_process_tree(process.pid)


def run_accuracy_test(
    model: ModelLaunchSettings,
    params: AccuracyTestParams,
    base_url: Optional[str] = None,
) -> AccuracyTestResult:
    """Run accuracy test for a single model.

    Args:
        model: ModelLaunchSettings with model config
        params: AccuracyTestParams with dataset, baseline, and optional settings
        base_url: Server base URL (default: DEFAULT_URL_FOR_TEST)

    Returns:
        AccuracyTestResult with test outcome
    """
    base_url = base_url or DEFAULT_URL_FOR_TEST

    print(f"\n{'='*60}")
    print(f"Running ACCURACY test for {model.model_path}")
    print(f"  Dataset: {params.dataset}")
    print(f"  Baseline: {params.baseline_accuracy}")
    print(f"{'='*60}\n")

    # Run evaluation based on dataset type
    if params.dataset == "gsm8k":
        success, error, metrics = _run_few_shot_eval(
            model=model,
            base_url=base_url,
            num_questions=params.num_examples,
            max_tokens=params.max_tokens or 512,
        )
    else:
        success, error, metrics = _run_simple_eval(
            model=model,
            base_url=base_url,
            dataset=params.dataset,
            num_examples=params.num_examples,
            num_threads=params.num_threads,
            max_tokens=params.max_tokens,
            return_latency=params.return_latency,
            thinking_mode=params.thinking_mode,
            temperature=params.temperature,
            repeat=params.repeat,
        )

    if not success:
        print(f"✗ Accuracy test failed for {model.model_path}: {error}")
        return AccuracyTestResult(
            model=model.model_path,
            dataset=params.dataset,
            passed=False,
            score=None,
            baseline_accuracy=params.baseline_accuracy,
            error=error,
        )

    # Validate against baseline
    score = metrics.get("score", 0.0)
    passed = score >= params.baseline_accuracy
    latency = metrics.get("latency")

    if passed:
        print(f"✓ Accuracy {score:.3f} >= baseline {params.baseline_accuracy:.3f}")
    else:
        error = f"Accuracy {score:.3f} below baseline {params.baseline_accuracy:.3f}"
        print(f"✗ {error}")

    return AccuracyTestResult(
        model=model.model_path,
        dataset=params.dataset,
        passed=passed,
        score=score,
        baseline_accuracy=params.baseline_accuracy,
        error=error if not passed else None,
        latency=latency,
    )


def run_accuracy_for_models(
    models: List[ModelLaunchSettings],
    params: AccuracyTestParams,
    test_name: str = "AccuracyTest",
    base_url: Optional[str] = None,
) -> dict:
    """Run accuracy tests for multiple models.

    Args:
        models: List of ModelLaunchSettings to test
        params: AccuracyTestParams (shared across all models)
        test_name: Name for the test (used in summary)
        base_url: Server base URL

    Returns:
        dict with results:
        {
            "all_passed": bool,
            "dataset": str,
            "results": [AccuracyTestResult, ...]
        }
    """
    base_url = base_url or DEFAULT_URL_FOR_TEST

    print("\n" + "=" * 80)
    print(f"ACCURACY TESTS: {test_name}")
    print(f"  Dataset: {params.dataset}")
    print(f"  Baseline: {params.baseline_accuracy}")
    print(f"  Models: {len(models)}")
    print("=" * 80)

    all_results = []
    all_passed = True

    for model in models:
        print("\n" + "-" * 60)
        print(f"Model: {model.model_path}")
        print(f"  TP Size: {model.tp_size}")
        print(f"  Extra Args: {model.extra_args}")
        print("-" * 60)

        result = run_accuracy_test(
            model=model,
            params=params,
            base_url=base_url,
        )

        all_results.append(result)

        if not result.passed:
            all_passed = False

    # Print summary
    print("\n" + "=" * 60)
    print(f"Accuracy Test Summary: {test_name}")
    print(f"Dataset: {params.dataset}")
    print(f"Baseline: {params.baseline_accuracy}")
    print("=" * 60)
    for result in all_results:
        status = "PASS" if result.passed else "FAIL"
        score_str = f"{result.score:.3f}" if result.score is not None else "N/A"
        print(f"  {result.model}: {status} (score: {score_str})")
        if result.error:
            print(f"    Error: {result.error}")

    print("\n" + "=" * 60)
    print(f"OVERALL: {'ALL PASSED' if all_passed else 'SOME FAILED'}")
    print("=" * 60 + "\n")

    return {
        "all_passed": all_passed,
        "dataset": params.dataset,
        "results": all_results,
    }
