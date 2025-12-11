import os
import re
import subprocess
import threading
import time
from dataclasses import dataclass
from typing import Callable, List, Optional

from sglang.srt.utils.common import kill_process_tree


@dataclass
class TestFile:
    name: str
    estimated_time: float = 60


# Patterns that indicate retriable accuracy/performance failures
RETRIABLE_PATTERNS = [
    r"AssertionError:.*not greater than",
    r"AssertionError:.*not less than",
    r"AssertionError:.*not equal to",
    r"AssertionError:.*!=.*expected",
    r"accuracy",
    r"score",
    r"latency",
    r"throughput",
]

# Patterns that indicate non-retriable failures (real code errors)
NON_RETRIABLE_PATTERNS = [
    r"SyntaxError",
    r"ImportError",
    r"ModuleNotFoundError",
    r"NameError",
    r"TypeError",
    r"AttributeError",
    r"RuntimeError",
    r"CUDA out of memory",
    r"OOM",
    r"Segmentation fault",
    r"core dumped",
    r"ConnectionRefusedError",
    r"FileNotFoundError",
]


def is_retriable_failure(output: str) -> tuple[bool, str]:
    """
    Determine if a test failure is retriable based on output patterns.

    Returns:
        tuple: (is_retriable, reason)
    """
    # Check for non-retriable patterns first
    for pattern in NON_RETRIABLE_PATTERNS:
        if re.search(pattern, output, re.IGNORECASE):
            return False, f"non-retriable error: {pattern}"

    # Check for retriable patterns
    for pattern in RETRIABLE_PATTERNS:
        if re.search(pattern, output, re.IGNORECASE):
            return True, f"retriable pattern: {pattern}"

    # If we have an AssertionError but didn't match non-retriable, assume retriable
    if re.search(r"AssertionError", output):
        return True, "AssertionError (assuming retriable)"

    # Default: not retriable
    return False, "unknown failure type"


def run_with_timeout(
    func: Callable,
    args: tuple = (),
    kwargs: Optional[dict] = None,
    timeout: float = None,
):
    """Run a function with timeout."""
    ret_value = []

    def _target_func():
        ret_value.append(func(*args, **(kwargs or {})))

    t = threading.Thread(target=_target_func)
    t.start()
    t.join(timeout=timeout)
    if t.is_alive():
        raise TimeoutError()

    if not ret_value:
        raise RuntimeError()

    return ret_value[0]


def write_github_step_summary(content: str):
    """Write content to GitHub Step Summary if available."""
    summary_file = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary_file:
        with open(summary_file, "a") as f:
            f.write(content)


def run_unittest_files(
    files: List[TestFile],
    timeout_per_file: float,
    continue_on_error: bool = False,
    enable_retry: bool = False,
    max_retry_attempts: int = 3,
    retry_wait_seconds: int = 60,
):
    """
    Run a list of test files.

    Args:
        files: List of TestFile objects to run
        timeout_per_file: Timeout in seconds for each test file
        continue_on_error: If True, continue running remaining tests even if one fails.
                          If False, stop at first failure (default behavior for PR tests).
        enable_retry: If True, retry failed tests that appear to be accuracy/performance
                     assertion failures (not code errors).
        max_retry_attempts: Maximum number of retry attempts per file (default: 3).
        retry_wait_seconds: Seconds to wait between retries (default: 60).
    """
    tic = time.perf_counter()
    success = True
    passed_tests = []
    failed_tests = []
    retried_tests = []  # Track which tests were retried

    for i, file in enumerate(files):
        filename, estimated_time = file.name, file.estimated_time
        process = None
        output_lines = []

        def run_one_file(filename, capture_output=False):
            nonlocal process, output_lines

            full_path = os.path.join(os.getcwd(), filename)
            print(
                f".\n.\nBegin ({i}/{len(files) - 1}):\npython3 {full_path}\n.\n.\n",
                flush=True,
            )
            file_tic = time.perf_counter()

            if capture_output:
                # Capture output for retry decision
                process = subprocess.Popen(
                    ["python3", full_path],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    env=os.environ,
                    text=True,
                )
                output_lines = []
                for line in process.stdout:
                    print(line, end="", flush=True)
                    output_lines.append(line)
                process.wait()
            else:
                process = subprocess.Popen(
                    ["python3", full_path], stdout=None, stderr=None, env=os.environ
                )
                process.wait()

            elapsed = time.perf_counter() - file_tic

            print(
                f".\n.\nEnd ({i}/{len(files) - 1}):\n{filename=}, {elapsed=:.0f}, {estimated_time=}\n.\n.\n",
                flush=True,
            )
            return process.returncode

        # Retry loop for each file
        attempt = 1
        file_passed = False
        was_retried = False

        while attempt <= (max_retry_attempts if enable_retry else 1):
            if attempt > 1:
                print(
                    f"\n[CI Retry] Attempt {attempt}/{max_retry_attempts} for {filename}\n",
                    flush=True,
                )
                was_retried = True

            try:
                ret_code = run_with_timeout(
                    run_one_file,
                    args=(filename,),
                    kwargs={"capture_output": enable_retry},
                    timeout=timeout_per_file,
                )

                if ret_code == 0:
                    file_passed = True
                    if was_retried:
                        print(
                            f"\n✓ PASSED on retry (attempt {attempt}): {filename}\n",
                            flush=True,
                        )
                        retried_tests.append((filename, attempt, "passed"))
                    passed_tests.append(filename)
                    break
                else:
                    # Check if we should retry
                    if enable_retry and attempt < max_retry_attempts:
                        output = "".join(output_lines)
                        is_retriable, reason = is_retriable_failure(output)

                        if is_retriable:
                            print(
                                f"\n[CI Retry] {filename} failed with {reason}",
                                flush=True,
                            )
                            print(
                                f"[CI Retry] Waiting {retry_wait_seconds}s before retry...\n",
                                flush=True,
                            )
                            time.sleep(retry_wait_seconds)
                            attempt += 1
                            continue
                        else:
                            print(
                                f"\n[CI Retry] {filename} failed with {reason} - not retrying\n",
                                flush=True,
                            )

                    # No retry or not retriable
                    print(
                        f"\n✗ FAILED: {filename} returned exit code {ret_code}\n",
                        flush=True,
                    )
                    if was_retried:
                        retried_tests.append((filename, attempt, "failed"))
                    failed_tests.append((filename, f"exit code {ret_code}"))
                    break

            except TimeoutError:
                kill_process_tree(process.pid)
                time.sleep(5)
                print(
                    f"\n✗ TIMEOUT: {filename} after {timeout_per_file} seconds\n",
                    flush=True,
                )
                if was_retried:
                    retried_tests.append((filename, attempt, "timeout"))
                failed_tests.append((filename, f"timeout after {timeout_per_file}s"))
                break

        if not file_passed:
            success = False
            if not continue_on_error:
                break

    elapsed_total = time.perf_counter() - tic

    if success:
        print(f"Success. Time elapsed: {elapsed_total:.2f}s", flush=True)
    else:
        print(f"Fail. Time elapsed: {elapsed_total:.2f}s", flush=True)

    # Print summary
    print(f"\n{'='*60}", flush=True)
    print(f"Test Summary: {len(passed_tests)}/{len(files)} passed", flush=True)
    if enable_retry and retried_tests:
        print(f"Retries: {len(retried_tests)} test(s) were retried", flush=True)
    print(f"{'='*60}", flush=True)
    if passed_tests:
        print("✓ PASSED:", flush=True)
        for test in passed_tests:
            print(f"  {test}", flush=True)
    if failed_tests:
        print("\n✗ FAILED:", flush=True)
        for test, reason in failed_tests:
            print(f"  {test} ({reason})", flush=True)
    if retried_tests:
        print("\n↻ RETRIED:", flush=True)
        for test, attempts, result in retried_tests:
            print(f"  {test} ({attempts} attempts, {result})", flush=True)
    print(f"{'='*60}\n", flush=True)

    # Write GitHub Step Summary
    if retried_tests:
        summary = "\n### CI Retry Summary\n\n"
        summary += "| Test File | Attempts | Result |\n"
        summary += "|-----------|----------|--------|\n"
        for test, attempts, result in retried_tests:
            summary += f"| `{test}` | {attempts} | {result} |\n"
        summary += "\n"
        write_github_step_summary(summary)

    return 0 if success else -1
