import json
import logging
import os
import re
import subprocess
import threading
import time
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Union

from sglang.srt.debug_utils import cuda_coredump
from sglang.srt.utils.common import kill_process_tree
from sglang.test.ci.ci_register import CIRegistry

# Configure logger to output to stdout
logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


@dataclass
class TestFile:
    name: str
    estimated_time: float = 60


class _ForkTestWorker:
    """Preloaded interpreter that forks an isolated child for each test file."""

    def __init__(self):
        result_read_fd, result_write_fd = os.pipe()
        worker_path = os.path.join(os.path.dirname(__file__), "fork_test_worker.py")
        self.process = subprocess.Popen(
            ["python3", worker_path, "--result-fd", str(result_write_fd)],
            stdin=subprocess.PIPE,
            stdout=None,
            stderr=None,
            text=True,
            pass_fds=(result_write_fd,),
        )
        os.close(result_write_fd)
        self.result_stream = os.fdopen(result_read_fd)
        self.files_run = 0

    def run(self, filename: str) -> tuple[int, float]:
        tic = time.perf_counter()
        if self.process.poll() is not None or self.process.stdin is None:
            return 1, 0.0
        try:
            self.process.stdin.write(json.dumps({"filename": filename}) + "\n")
            self.process.stdin.flush()
            result_line = self.result_stream.readline()
        except (BrokenPipeError, OSError):
            return 1, time.perf_counter() - tic
        if not result_line:
            return 1, time.perf_counter() - tic
        try:
            result = json.loads(result_line)
        except json.JSONDecodeError:
            return 1, time.perf_counter() - tic
        self.files_run += 1
        return int(result["returncode"]), float(result["elapsed"])

    def close(self, terminate: bool = False):
        if self.process.poll() is None:
            if terminate:
                kill_process_tree(self.process.pid)
            elif self.process.stdin is not None:
                try:
                    self.process.stdin.write(json.dumps({"command": "stop"}) + "\n")
                    self.process.stdin.flush()
                    self.process.wait(timeout=10)
                except (BrokenPipeError, subprocess.TimeoutExpired):
                    kill_process_tree(self.process.pid)
        if self.process.poll() is None:
            self.process.kill()
        try:
            self.process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            self.process.kill()
            self.process.wait()
        if self.process.stdin is not None:
            self.process.stdin.close()
        self.result_stream.close()


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

# XPU/B580 device-resource flakes. Matched BEFORE the non-retriable list so
# transient GPU OOMs / slow cold-cache server starts get one clean re-run.
INFRA_RETRIABLE_PATTERNS = [
    r"UR_RESULT_ERROR_OUT_OF_RESOURCES",
    r"XPU out of memory",
    r"Server failed to start within the timeout",
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
    # XPU infra flakes take precedence over the non-retriable list.
    for pattern in INFRA_RETRIABLE_PATTERNS:
        if re.search(pattern, output, re.IGNORECASE):
            return True, f"retriable XPU infra flake: {pattern}"

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


def _repo_relative_path(p: str) -> str:
    """Return path stripped to repo-relative form (e.g. 'test/srt/foo.py').

    Used in the machine-readable TIMINGS block so downstream scrapers
    get a stable key regardless of CI runner checkout layout.
    """
    if not os.path.isabs(p):
        p = os.path.join(os.getcwd(), p)
    marker = "/sglang/"
    idx = p.rfind(marker)
    return p[idx + len(marker) :] if idx >= 0 else p


# Slow-run variance is largely additive (cold HF cache, slow server launch), so
# the multiplier alone under-provisions at both ends: test_encoder_dp runs
# 200-426s but once took over 1185s against a 1.5x budget of 765s, and
# test_lora_deepseek_v3_base_logprob_diff (est 1800) landed on exactly 1.5 * est.
# Every file gets the same absolute slack on top of the proportional one.
DERIVED_TIMEOUT_SLACK = 1800.0
DERIVED_TIMEOUT_FACTOR = 1.5


def derive_timeout_per_file(est_time: float) -> float:
    est = float(est_time)
    return max(est * DERIVED_TIMEOUT_FACTOR, est + DERIVED_TIMEOUT_SLACK)


def run_unittest_files(
    files: Union[List[TestFile], List[CIRegistry]],
    timeout_per_file: Optional[float] = None,
    continue_on_error: bool = False,
    enable_retry: bool = False,
    max_attempts: int = 2,
    retry_wait_seconds: int = 60,
    fork_worker_batch_size: int = 1,
):
    """
    Run a list of test files.

    Args:
        files: List of TestFile objects to run
        timeout_per_file: Fixed timeout in seconds for every test file, or None
                          to derive each file's budget from its own est_time.
        continue_on_error: If True, continue running remaining tests even if one fails.
                          If False, stop at first failure (default behavior for PR tests).
        enable_retry: If True, retry failed tests that appear to be accuracy/performance
                     assertion failures (not code errors).
        max_attempts: Maximum number of attempts per file including initial run (default: 2).
        retry_wait_seconds: Seconds to wait between retries (default: 60).
        fork_worker_batch_size: Number of files served by one preloaded fork
                                worker. Each file still runs in a fresh child
                                process. One keeps the existing exec behavior.
    """
    coredump_enabled = cuda_coredump.is_enabled()
    if coredump_enabled:
        cuda_coredump.cleanup_dump_dir()

    tic = time.perf_counter()
    success = True
    passed_tests = []
    failed_tests = []
    retried_tests = []  # Track which tests were retried
    # Per-file elapsed seconds, latest attempt wins. Consumed by the
    # TIMINGS block emitted at the end of this function.
    file_elapsed: Dict[str, float] = {}
    fork_worker = None
    use_fork_worker = fork_worker_batch_size > 1 and not enable_retry

    for i, file in enumerate(files):
        if isinstance(file, CIRegistry):
            filename, estimated_time = file.filename, file.est_time
        else:
            # FIXME: remove this branch after migrating all tests to use CIRegistry
            filename, estimated_time = file.name, file.estimated_time

        file_timeout = (
            timeout_per_file
            if timeout_per_file is not None
            else derive_timeout_per_file(estimated_time)
        )

        process = None
        output_lines = []

        def run_one_file(filename, capture_output=False):
            nonlocal process, output_lines, fork_worker

            full_path = os.path.join(os.getcwd(), filename)
            logger.info(
                f".\n.\nBegin ({i}/{len(files) - 1}):\npython3 {full_path}\n.\n.\n"
            )
            file_tic = time.perf_counter()

            if use_fork_worker:
                if (
                    fork_worker is None
                    or fork_worker.files_run >= fork_worker_batch_size
                ):
                    if fork_worker is not None:
                        fork_worker.close()
                    fork_worker = _ForkTestWorker()
                process = fork_worker.process
                ret_code, _ = fork_worker.run(full_path)
                if ret_code != 0 or fork_worker.files_run >= fork_worker_batch_size:
                    fork_worker.close()
                    fork_worker = None
                    process = None
                elapsed = time.perf_counter() - file_tic
            elif capture_output:
                # Capture output for retry decision
                cmd = ["python3", full_path, "-f"]
                process = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    errors="ignore",  # Ignore non-UTF-8 bytes to prevent UnicodeDecodeError
                )
                output_lines = []

                def read_output():
                    for line in process.stdout:
                        logger.info(line.rstrip())
                        output_lines.append(line)

                # Read stdout on a background thread so the main thread won't block on EOF.
                reader_thread = threading.Thread(target=read_output, daemon=True)
                reader_thread.start()
                process.wait()
                # Bounded wait for the reader to finish.
                reader_thread.join(timeout=60)
            else:
                cmd = ["python3", full_path, "-f"]
                process = subprocess.Popen(cmd, stdout=None, stderr=None)
                process.wait()

            if not use_fork_worker:
                elapsed = time.perf_counter() - file_tic
                ret_code = process.returncode
            file_elapsed[filename] = elapsed

            logger.info(
                f".\n.\nEnd ({i}/{len(files) - 1}):\n{filename=}, {elapsed=:.0f}, {estimated_time=}\n.\n.\n"
            )
            return ret_code

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
                    timeout=file_timeout,
                )

                if ret_code == 0:
                    file_passed = True
                    if was_retried:
                        logger.info(
                            f"\n✓ PASSED on retry (attempt {attempt}): {filename}\n"
                        )
                        retried_tests.append((filename, attempt, "passed"))
                    passed_tests.append(filename)
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
                    failed_tests.append((filename, f"exit code {ret_code}"))
                    break

            except TimeoutError:
                if fork_worker is not None:
                    fork_worker.close(terminate=True)
                    fork_worker = None
                elif process is not None:
                    kill_process_tree(process.pid)
                time.sleep(5)
                # TimeoutError aborts run_one_file before its elapsed write;
                # record the timeout cap as an upper bound so the file still
                # appears in the TIMINGS block below.
                file_elapsed[filename] = float(file_timeout)
                # Retry once on timeout: usually a stuck server / hung device.
                # A real hang times out again and is reported.
                if enable_retry and attempt < max_attempts:
                    logger.info(
                        f"\n[CI Retry] {filename} timed out after "
                        f"{file_timeout}s; waiting {retry_wait_seconds}s "
                        f"before retry (attempt {attempt + 1}/{max_attempts})\n"
                    )
                    time.sleep(retry_wait_seconds)
                    attempt += 1
                    continue
                logger.info(f"\n✗ TIMEOUT: {filename} after {file_timeout} seconds\n")
                if was_retried:
                    retried_tests.append((filename, attempt, "timeout"))
                failed_tests.append((filename, f"timeout after {file_timeout}s"))
                break

        if not file_passed:
            success = False
            if not continue_on_error:
                break

    if fork_worker is not None:
        fork_worker.close()

    elapsed_total = time.perf_counter() - tic

    if coredump_enabled and not success:
        cuda_coredump.report()

    if success:
        logger.info(f"Success. Time elapsed: {elapsed_total:.2f}s")
    else:
        logger.info(f"Fail. Time elapsed: {elapsed_total:.2f}s")

    # Print summary
    logger.info(f"\n{'='*60}")
    logger.info(f"Test Summary: {len(passed_tests)}/{len(files)} passed")
    if enable_retry and retried_tests:
        logger.info(f"Retries: {len(retried_tests)} test(s) were retried")
    logger.info(f"{'='*60}")
    if passed_tests:
        logger.info("✓ PASSED:")
        for test in passed_tests:
            logger.info(f"  {test}")
    if failed_tests:
        logger.info("\n✗ FAILED:")
        for test, reason in failed_tests:
            logger.info(f"  {test} ({reason})")
    if retried_tests:
        logger.info("\n↻ RETRIED:")
        for test, attempts, result in retried_tests:
            logger.info(f"  {test} ({attempts} attempts, {result})")
    logger.info(f"{'='*60}\n")

    # Machine-readable timings block for downstream scrapers/dashboards.
    # One JSON object per executed file (post-retry: only the latest
    # attempt's elapsed is recorded). Files skipped via fail-fast
    # (continue_on_error=False) are omitted. Job wall-clock is read
    # separately from the GitHub Actions API by consumers, so we don't
    # emit any aggregate fields here.
    passed_set = set(passed_tests)
    logger.info("========== TIMINGS BEGIN ==========")
    for fname, elapsed in file_elapsed.items():
        logger.info(
            json.dumps(
                {
                    "file": _repo_relative_path(fname),
                    "passed": fname in passed_set,
                    "elapsed": round(elapsed),
                }
            )
        )
    logger.info("========== TIMINGS END ==========")

    # Write GitHub Step Summary only if retries occurred
    if retried_tests:
        passed_on_retry = [t for t, _, r in retried_tests if r == "passed"]
        failed_after_retry = [t for t, _, r in retried_tests if r != "passed"]
        summary = f"**↻ Retried {len(retried_tests)} test(s):**\n"
        if passed_on_retry:
            summary += f"- ✓ Passed on retry: {', '.join(passed_on_retry)}\n"
        if failed_after_retry:
            summary += f"- ✗ Still failed: {', '.join(failed_after_retry)}\n"
        write_github_step_summary(summary)

    # Fully guarded auto-record for SGLANG_TEST_METRICS_FILE: unset (the default)
    # means zero delta for every non-XPU-nightly suite. OSError is swallowed so
    # a bad filesystem cannot turn a passing run red. Any new test file added
    # to run_suite.py is picked up here without per-test wiring.
    metrics_path = os.environ.get("SGLANG_TEST_METRICS_FILE")
    if metrics_path:
        passed_set = set(passed_tests)
        failed_reasons = dict(failed_tests)
        try:
            with open(metrics_path, "a") as f:
                for fname, elapsed in file_elapsed.items():
                    record = {
                        "kind": "file",
                        "test_file": os.path.basename(fname),
                        "status": "pass" if fname in passed_set else "fail",
                        "duration": round(elapsed, 2),
                    }
                    if fname in failed_reasons:
                        record["error"] = failed_reasons[fname]
                    f.write(json.dumps(record) + "\n")
        except OSError:
            pass

    return 0 if success else -1
