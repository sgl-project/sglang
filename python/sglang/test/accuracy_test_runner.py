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
    write_github_step_summary,
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
    top_p: Optional[float] = None
    top_k: Optional[int] = None
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
    variant: Optional[str] = None


def write_accuracy_github_summary(
    test_name: str,
    dataset: str,
    results: List[AccuracyTestResult],
) -> None:
    """Write accuracy test results to GitHub step summary.

    Args:
        test_name: Name of the test
        dataset: Dataset name used for evaluation
        results: List of AccuracyTestResult objects
    """
    summary = f"#### {test_name} - Accuracy ({dataset})\n"
    summary += "| config | status | score | baseline | error |\n"
    summary += "| ------ | ------ | ----- | -------- | ----- |\n"

    for result in results:
        status_emoji = "✅" if result.passed else "❌"
        score_str = f"{result.score:.4f}" if result.score is not None else "N/A"
        baseline_str = f"{result.baseline_accuracy:.4f}"
        error_str = result.error if result.error else "-"
        # Use variant name if available, otherwise use model path
        config_name = result.variant if result.variant else result.model
        summary += f"| {config_name} | {status_emoji} | {score_str} | {baseline_str} | {error_str} |\n"

    write_github_step_summary(summary)


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
    top_p: Optional[float] = None,
    top_k: Optional[int] = None,
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
            env=model.env,
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

        if top_p is not None:
            args.top_p = top_p

        if top_k is not None:
            args.top_k = top_k

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


# Cached uv venv for NeMo Skills (persists across variants within a process).
_nemo_venv_dir: Optional[str] = None
_nemo_data_prepared: set = set()


def _get_nemo_venv() -> Tuple[str, dict]:
    """Get or create a uv venv with nemo_skills installed.

    Returns (venv_python_path, env_dict) reusable across calls.
    """
    import os
    import subprocess
    import tempfile

    global _nemo_venv_dir

    if _nemo_venv_dir is not None:
        venv_python = f"{_nemo_venv_dir}/venv/bin/python"
        env = {
            **dict(os.environ),
            "NEMO_SKILLS_DISABLE_UNCOMMITTED_CHANGES_CHECK": "1",
            "OPENAI_API_KEY": "dummy",
            "VIRTUAL_ENV": f"{_nemo_venv_dir}/venv",
            "PATH": f"{_nemo_venv_dir}/venv/bin:" + os.environ.get("PATH", ""),
        }
        return venv_python, env

    _nemo_venv_dir = tempfile.mkdtemp(prefix="nemo_skills_")
    print(f"Creating NeMo Skills venv in {_nemo_venv_dir}...")

    # Create venv
    result = subprocess.run(
        ["uv", "venv", f"{_nemo_venv_dir}/venv", "--python", "3.12"],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        subprocess.run(
            ["uv", "venv", f"{_nemo_venv_dir}/venv"],
            capture_output=True,
            text=True,
        )

    # Install nemo_skills
    print("Installing nemo_skills...")
    pip_result = subprocess.run(
        [
            "uv",
            "pip",
            "install",
            "--python",
            f"{_nemo_venv_dir}/venv/bin/python",
            "git+https://github.com/NVIDIA/NeMo-Skills.git",
        ],
        capture_output=True,
        text=True,
        timeout=300,
    )
    if pip_result.returncode != 0:
        raise RuntimeError(f"Failed to install nemo_skills: {pip_result.stderr[-500:]}")

    print("NeMo Skills installed successfully")
    return _get_nemo_venv()


def _ensure_nemo_data_prepared(
    venv_python: str, env: dict, dataset: str
) -> Tuple[bool, Optional[str]]:
    """Prepare NeMo Skills dataset data if not already done.

    Uses the venv python so data lands inside the venv's nemo_skills package.
    """
    import subprocess

    if dataset in _nemo_data_prepared:
        return True, None

    print(f"Preparing {dataset} data (this may take a few minutes for VLM datasets)...")
    result = subprocess.run(
        [venv_python, "-m", "nemo_skills.dataset.prepare", dataset],
        text=True,
        timeout=600,
        env=env,
    )
    if result.returncode != 0:
        return False, f"Failed to prepare {dataset} data (exit {result.returncode})"

    _nemo_data_prepared.add(dataset)
    return True, None


def _run_nemo_skills_eval(
    model: ModelLaunchSettings,
    base_url: str,
    dataset: str,
    max_tokens: Optional[int] = None,
    repeat: Optional[int] = None,
    temperature: Optional[float] = None,
    top_p: Optional[float] = None,
) -> Tuple[bool, Optional[str], Optional[dict]]:
    """Run evaluation using NeMo Skills (ns eval) for benchmarks like mmmu-pro.

    Uses an isolated uv venv (shared across variants) so nemo_skills dependencies
    don't interfere with the system python / sglang server.

    Returns:
        Tuple of (success, error_message, metrics_dict)
    """
    import subprocess
    import tempfile

    process = None
    try:
        # Get or create the shared venv (once per process)
        venv_python, env = _get_nemo_venv()

        # Prepare dataset (once per process, cached)
        ok, err = _ensure_nemo_data_prepared(venv_python, env, dataset)
        if not ok:
            return False, err, None

        process = popen_launch_server(
            model.model_path,
            base_url,
            other_args=model.extra_args,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            env=model.env,
        )

        port = int(base_url.split(":")[-1])
        server_address = f"http://127.0.0.1:{port}/v1"
        repeat_val = repeat or 1
        max_tokens_val = max_tokens or 32768
        benchmark_spec = f"{dataset}:{repeat_val}"

        # Build ns eval command using venv python
        # Note: nemo_skills.pipeline.eval requires the "eval" subcommand
        output_dir = tempfile.mkdtemp(prefix="ns_eval_output_")
        cmd = [
            venv_python,
            "-m",
            "nemo_skills.pipeline.eval",
            "eval",
            f"--benchmarks={benchmark_spec}",
            "--server_type=sglang",
            f"--model={model.model_path}",
            f"--server_address={server_address}",
            f"--output_dir={output_dir}",
            f"++inference.tokens_to_generate={max_tokens_val}",
        ]

        if temperature is not None:
            cmd.append(f"++inference.temperature={temperature}")
        if top_p is not None:
            cmd.append(f"++inference.top_p={top_p}")

        # Add VLM-specific config
        if dataset in ("mmmu-pro", "mmmu_pro"):
            cmd.append("++prompt_config=vlm/mmmu-pro")
            cmd.append("++max_concurrent_requests=512")
            cmd.append("++max_samples=500")

        print(f"Running: {' '.join(cmd)}")
        eval_result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=7200,
            env=env,
        )

        print(eval_result.stdout[-2000:] if eval_result.stdout else "(no stdout)")
        if eval_result.stderr:
            print(eval_result.stderr[-1000:])

        if eval_result.returncode != 0:
            return (
                False,
                f"ns eval failed (exit {eval_result.returncode}): {eval_result.stderr[-500:]}",
                None,
            )

        # Parse results
        summarize_result = subprocess.run(
            [
                venv_python,
                "-m",
                "nemo_skills.pipeline.summarize_results",
                f"{output_dir}/eval-results",
            ],
            capture_output=True,
            text=True,
            timeout=60,
            env=env,
        )

        output = summarize_result.stdout + "\n" + eval_result.stdout
        print(f"Summary: {summarize_result.stdout[:1000]}")

        # Parse accuracy from output (format varies, look for common patterns)
        import re

        score = None
        for line in output.split("\n"):
            match = re.search(r"(?:accuracy|score)[:\s]+([0-9.]+)", line, re.IGNORECASE)
            if match:
                score = float(match.group(1))

        if score is None:
            # Try to find it in eval-results directory
            import glob
            import json

            for result_file in glob.glob(
                f"{output_dir}/eval-results/**/*.json", recursive=True
            ):
                try:
                    with open(result_file) as f:
                        data = json.load(f)
                    if isinstance(data, dict):
                        score = (
                            data.get("accuracy")
                            or data.get("score")
                            or data.get("mean_score")
                        )
                        if score is not None:
                            break
                except (json.JSONDecodeError, KeyError):
                    continue

        if score is None:
            # Last resort: compute accuracy directly from JSONL output
            import glob
            import json

            for jsonl_file in sorted(
                glob.glob(f"{output_dir}/eval-results/**/*.jsonl*", recursive=True)
            ):
                correct = 0
                total = 0
                try:
                    with open(jsonl_file) as f:
                        for line in f:
                            line = line.strip()
                            if not line:
                                continue
                            entry = json.loads(line)
                            expected = entry.get("expected_answer", "")
                            generation = entry.get("generation", "")
                            # Extract "Answer: X" from the end of generation
                            answer_match = re.search(
                                r"Answer:\s*([A-J])", generation, re.IGNORECASE
                            )
                            if answer_match:
                                predicted = answer_match.group(1).upper()
                                if predicted == expected.upper():
                                    correct += 1
                            total += 1
                except (json.JSONDecodeError, KeyError, OSError):
                    continue
                if total > 0:
                    score = correct / total
                    print(
                        f"Computed accuracy from {jsonl_file}: "
                        f"{correct}/{total} = {score:.4f}"
                    )
                    break

        if score is None:
            return False, "Could not parse accuracy from ns eval output", None

        return True, None, {"score": score}

    except subprocess.TimeoutExpired:
        return False, "NeMo Skills eval timed out", None
    except Exception as e:
        return False, f"NeMo Skills eval exception: {str(e)}", None
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
    # - NeMo Skills: mmmu-pro (and other VLM evals needing ns eval)
    # - simple_eval: everything else (gsm8k, gpqa, mmlu, mmmu, etc.)
    if params.dataset in ("mmmu-pro", "mmmu_pro"):
        success, error, metrics = _run_nemo_skills_eval(
            model=model,
            base_url=base_url,
            dataset="mmmu-pro",
            max_tokens=params.max_tokens,
            repeat=params.repeat or 1,
            temperature=params.temperature,
            top_p=params.top_p,
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
            top_p=params.top_p,
            top_k=params.top_k,
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
            variant=model.variant,
        )

    # Validate against baseline
    # Handle different metric key names: "score", "mean_score" (for GPQA with repeat), "accuracy"
    score = (
        metrics.get("score")
        or metrics.get("mean_score")
        or metrics.get("accuracy", 0.0)
    )
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
        variant=model.variant,
    )
