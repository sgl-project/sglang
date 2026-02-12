import logging
import os
import re
import subprocess
import threading
import time
from dataclasses import dataclass
from typing import Callable, List, Optional, Union

from sglang.srt.utils.common import kill_process_tree
from sglang.test.ci.ci_register import CIRegistry

# Configure logger to output to stdout
logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


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
    r"timeout",
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


def write_github_step_summary(content: str, append: bool = False):
    """Write content to GitHub Step Summary if available."""
    summary_file = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary_file:
        mode = "a" if append else "w"
        with open(summary_file, mode) as f:
            f.write(content)


def generate_test_summary_md(
    passed_tests, failed_tests, retried_tests, total_files, remaining_files=None
):
    """Generate markdown summary of test results."""
    completed = len(passed_tests) + len(failed_tests)
    total_elapsed = sum(e for _, e, _ in passed_tests) + sum(
        e for _, _, e, _ in failed_tests
    )
    total_estimated = sum(est for _, _, est in passed_tests) + sum(
        est for _, _, _, est in failed_tests
    )

    # Header with progress
    if remaining_files:
        summary_md = f"## Test Progress: {len(passed_tests)}/{completed} passed ({completed}/{total_files} completed)\n"
        summary_md += (
            f"**Time so far:** {total_elapsed:.0f}s (est: {total_estimated:.0f}s)\n\n"
        )
    else:
        summary_md = f"## Test Summary: {len(passed_tests)}/{total_files} passed\n"
        summary_md += (
            f"**Total time:** {total_elapsed:.0f}s (est: {total_estimated:.0f}s)\n\n"
        )

    # Show remaining tests if job is still running
    if remaining_files:
        summary_md += "### ⏳ Remaining\n"
        for test in remaining_files[:5]:  # Show first 5
            summary_md += f"- `{test}`\n"
        if len(remaining_files) > 5:
            summary_md += f"- ... and {len(remaining_files) - 5} more\n"
        summary_md += "\n"

    # Failed tests (always show prominently)
    if failed_tests:
        summary_md += "### ✗ Failed\n"
        summary_md += "| Test | Reason | Time | Est |\n"
        summary_md += "|------|--------|------|-----|\n"
        for test, reason, elapsed, est in failed_tests:
            summary_md += f"| `{test}` | {reason} | {elapsed:.0f}s | {est:.0f}s |\n"
        summary_md += "\n"

    # Retried tests
    if retried_tests:
        passed_on_retry = [(t, a) for t, a, r in retried_tests if r == "passed"]
        failed_after_retry = [(t, a, r) for t, a, r in retried_tests if r != "passed"]
        summary_md += "### ↻ Retried\n"
        if passed_on_retry:
            for test, attempts in passed_on_retry:
                summary_md += f"- ✓ `{test}` (passed on attempt {attempts})\n"
        if failed_after_retry:
            for test, attempts, result in failed_after_retry:
                summary_md += f"- ✗ `{test}` ({attempts} attempts, {result})\n"
        summary_md += "\n"

    # Passed tests (collapsible)
    if passed_tests:
        summary_md += "<details>\n<summary>✓ Passed ({} tests)</summary>\n\n".format(
            len(passed_tests)
        )
        summary_md += "| Test | Time | Est |\n"
        summary_md += "|------|------|-----|\n"
        for test, elapsed, est in passed_tests:
            summary_md += f"| `{test}` | {elapsed:.0f}s | {est:.0f}s |\n"
        summary_md += "\n</details>\n"

    return summary_md


def run_unittest_files(
    files: Union[List[TestFile], List[CIRegistry]],
    timeout_per_file: float,
    continue_on_error: bool = False,
    enable_retry: bool = False,
    max_attempts: int = 2,
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
        max_attempts: Maximum number of attempts per file including initial run (default: 2).
        retry_wait_seconds: Seconds to wait between retries (default: 60).
    """
    tic = time.perf_counter()
    success = True
    passed_tests = []  # List of (filename, elapsed, estimated)
    failed_tests = []  # List of (filename, reason, elapsed, estimated)
    retried_tests = []  # Track which tests were retried

    # Extract all filenames for tracking remaining tests
    all_filenames = []
    for file in files:
        if isinstance(file, CIRegistry):
            all_filenames.append(file.filename)
        else:
            all_filenames.append(file.name)

    for i, file in enumerate(files):
        if isinstance(file, CIRegistry):
            filename, estimated_time = file.filename, file.est_time
        else:
            # FIXME: remove this branch after migrating all tests to use CIRegistry
            filename, estimated_time = file.name, file.estimated_time

        process = None
        output_lines = []
        file_elapsed = [0.0]  # Use list to allow modification in nested function

        def run_one_file(filename, capture_output=False):
            nonlocal process, output_lines

            full_path = os.path.join(os.getcwd(), filename)
            logger.info(
                f".\n.\nBegin ({i}/{len(files) - 1}):\npython3 {full_path}\n.\n.\n"
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
                    errors="ignore",  # Ignore non-UTF-8 bytes to prevent UnicodeDecodeError
                )
                output_lines = []
                for line in process.stdout:
                    logger.info(line.rstrip())
                    output_lines.append(line)
                process.wait()
            else:
                process = subprocess.Popen(
                    ["python3", full_path], stdout=None, stderr=None, env=os.environ
                )
                process.wait()

            elapsed = time.perf_counter() - file_tic
            file_elapsed[0] = elapsed

            logger.info(
                f".\n.\nEnd ({i}/{len(files) - 1}):\n{filename=}, {elapsed=:.0f}, {estimated_time=}\n.\n.\n"
            )
            return process.returncode

        # Retry loop for each file
        attempt = 1
        file_passed = False
        was_retried = False

        while attempt <= (max_attempts if enable_retry else 1):
            if attempt > 1:
                logger.info(
                    f"\n[CI Retry] Attempt {attempt}/{max_attempts} for {filename}\n"
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
                        logger.info(
                            f"\n✓ PASSED on retry (attempt {attempt}): {filename}\n"
                        )
                        retried_tests.append((filename, attempt, "passed"))
                    passed_tests.append((filename, file_elapsed[0], estimated_time))
                    break
                else:
                    # Check if we should retry
                    if enable_retry and attempt < max_attempts:
                        output = "".join(output_lines)
                        is_retriable, reason = is_retriable_failure(output)

                        if is_retriable:
                            logger.info(f"\n[CI Retry] {filename} failed with {reason}")
                            logger.info(
                                f"[CI Retry] Waiting {retry_wait_seconds}s before retry...\n"
                            )
                            time.sleep(retry_wait_seconds)
                            attempt += 1
                            continue
                        else:
                            logger.info(
                                f"\n[CI Retry] {filename} failed with {reason} - not retrying\n"
                            )

                    # No retry or not retriable
                    logger.info(
                        f"\n✗ FAILED: {filename} returned exit code {ret_code}\n"
                    )
                    if was_retried:
                        retried_tests.append((filename, attempt, "failed"))
                    failed_tests.append(
                        (
                            filename,
                            f"exit code {ret_code}",
                            file_elapsed[0],
                            estimated_time,
                        )
                    )
                    break

            except TimeoutError:
                kill_process_tree(process.pid)
                time.sleep(5)
                logger.info(
                    f"\n✗ TIMEOUT: {filename} after {timeout_per_file} seconds\n"
                )
                if was_retried:
                    retried_tests.append((filename, attempt, "timeout"))
                failed_tests.append(
                    (
                        filename,
                        f"timeout after {timeout_per_file}s",
                        timeout_per_file,
                        estimated_time,
                    )
                )
                break

        # Update GitHub Step Summary after each test (overwrites previous)
        # This ensures summary is available even if job is killed
        completed_filenames = [t[0] for t in passed_tests] + [
            t[0] for t in failed_tests
        ]
        remaining = [f for f in all_filenames if f not in completed_filenames]
        summary_md = generate_test_summary_md(
            passed_tests, failed_tests, retried_tests, len(files), remaining
        )
        write_github_step_summary(summary_md)

        if not file_passed:
            success = False
            if not continue_on_error:
                break

    elapsed_total = time.perf_counter() - tic

    if success:
        logger.info(f"Success. Time elapsed: {elapsed_total:.2f}s")
    else:
        logger.info(f"Fail. Time elapsed: {elapsed_total:.2f}s")

    # Print summary to logs
    logger.info(f"\n{'='*60}")
    logger.info(f"Test Summary: {len(passed_tests)}/{len(files)} passed")
    if enable_retry and retried_tests:
        logger.info(f"Retries: {len(retried_tests)} test(s) were retried")
    logger.info(f"{'='*60}")
    if passed_tests:
        logger.info("✓ PASSED:")
        for test, elapsed, est in passed_tests:
            logger.info(f"  {test} ({elapsed:.0f}s / {est:.0f}s est)")
    if failed_tests:
        logger.info("\n✗ FAILED:")
        for test, reason, elapsed, est in failed_tests:
            logger.info(f"  {test} ({reason}, {elapsed:.0f}s / {est:.0f}s est)")
    if retried_tests:
        logger.info("\n↻ RETRIED:")
        for test, attempts, result in retried_tests:
            logger.info(f"  {test} ({attempts} attempts, {result})")
    logger.info(f"{'='*60}\n")

    # Write final GitHub Step Summary (no remaining tests)
    summary_md = generate_test_summary_md(
        passed_tests, failed_tests, retried_tests, len(files), remaining_files=None
    )
    write_github_step_summary(summary_md)

    return 0 if success else -1
