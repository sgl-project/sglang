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
from sglang.test.ci.ci_register import CIBundle, CIRegistry

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


def write_github_step_summary(content: str):
    """Write content to GitHub Step Summary if available."""
    summary_file = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary_file:
        with open(summary_file, "a") as f:
            f.write(content)


def _filename_to_module(filename: str) -> str:
    """Convert a test path to a dotted module name importable by
    ``python3 -m unittest``.

    ``run_unittest_files`` is invoked from cwd=``test/`` (see
    ``.github/workflows/_pr-test-stage.yml:157`` and
    ``test/run_suite.py``). ``CIRegistry.filename`` is sometimes
    test-relative (legacy ``TestFile`` shape) and sometimes absolute
    (``run_suite.py`` builds the glob from ``os.path.join(script_dir,
    'registered', '**', '*.py')`` so every path is absolute under the
    repo). We normalize against cwd before stripping ``.py`` and
    converting separators, so ``/.../test/registered/foo/test_x.py``
    → ``registered.foo.test_x``.
    """
    if os.path.isabs(filename):
        try:
            filename = os.path.relpath(filename, os.getcwd())
        except ValueError:
            # Different drive on Windows or similar; leave as-is and
            # let unittest report the import failure.
            pass
    if filename.endswith(".py"):
        filename = filename[:-3]
    return filename.replace(os.sep, ".").replace("/", ".")


def _run_one_bundle(
    *,
    bundle: "CIBundle",
    idx: int,
    total: int,
    timeout_per_file: float,
    enable_retry: bool,
    max_attempts: int,
    retry_wait_seconds: int,
    passed_tests: List[str],
    failed_tests: List[tuple],
    retried_tests: List[tuple],
    file_elapsed: Dict[str, float],
) -> bool:
    """Run an in-process bundle (one ``python3 -m unittest`` invocation).

    Returns True iff the bundle succeeded. On failure, marks every member
    as failed (with the bundle's reason) so downstream summary lists the
    individual files. v1 does not yet do per-member fallback re-runs;
    that's a follow-up.

    Timing: records the bundle wall under ``file_elapsed[bundle.filename]``
    (key ``"group:<group_key>"``) so the partition model can read it for
    future bin-packing. Member files don't get individual elapsed entries
    in v1 — unittest's default runner doesn't surface per-module wall time.
    """
    dotted_modules = [_filename_to_module(m.filename) for m in bundle.members]
    cmd = ["python3", "-m", "unittest", "-f", *dotted_modules]
    # Allow generous slack: the bundle's est_time already accounts for
    # amortized import; CI's `timeout_per_file` may be sized for a single
    # heavy file, so use whichever is larger.
    bundle_timeout = max(float(timeout_per_file), bundle.est_time * 2 + 60)
    bundle_label = bundle.filename  # e.g. "group:attention_unittest"

    logger.info(
        f".\n.\nBegin ({idx}/{total - 1}) BUNDLE {bundle_label} "
        f"({len(dotted_modules)} files, est={bundle.est_time:.0f}s): "
        f"{' '.join(cmd[:4])} <{len(dotted_modules)} modules>\n.\n."
    )

    process_holder: Dict[str, subprocess.Popen] = {}
    output_holder: Dict[str, list] = {"lines": []}

    def run_bundle_once(capture_output: bool):
        full_cmd = list(cmd)
        bundle_tic = time.perf_counter()
        if capture_output:
            p = subprocess.Popen(
                full_cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                errors="ignore",
            )
            process_holder["p"] = p
            output_holder["lines"] = []
            for line in p.stdout:
                logger.info(line.rstrip())
                output_holder["lines"].append(line)
            p.wait()
        else:
            p = subprocess.Popen(full_cmd, stdout=None, stderr=None)
            process_holder["p"] = p
            p.wait()
        elapsed = time.perf_counter() - bundle_tic
        file_elapsed[bundle_label] = elapsed
        logger.info(
            f".\n.\nEnd ({idx}/{total - 1}) BUNDLE {bundle_label}: "
            f"elapsed={elapsed:.0f}s\n.\n."
        )
        return p.returncode

    attempt = 1
    was_retried = False
    while attempt <= (max_attempts if enable_retry else 1):
        if attempt > 1:
            logger.info(
                f"\n[CI Retry] Attempt {attempt}/{max_attempts} for {bundle_label}\n"
            )
            was_retried = True
        try:
            ret_code = run_with_timeout(
                run_bundle_once,
                args=(enable_retry,),
                timeout=bundle_timeout,
            )
            if ret_code == 0:
                if was_retried:
                    logger.info(
                        f"\n✓ PASSED on retry (attempt {attempt}): {bundle_label}\n"
                    )
                    retried_tests.append((bundle_label, attempt, "passed"))
                # Credit the bundle key itself (so TIMINGS marks it passed)
                # plus every member (so the PASSED summary lists them).
                passed_tests.append(bundle_label)
                for m in bundle.members:
                    passed_tests.append(m.filename)
                return True

            if enable_retry and attempt < max_attempts:
                output = "".join(output_holder["lines"])
                is_retriable, reason = is_retriable_failure(output)
                if is_retriable:
                    logger.info(f"\n[CI Retry] {bundle_label} failed with {reason}")
                    logger.info(
                        f"[CI Retry] Waiting {retry_wait_seconds}s before retry...\n"
                    )
                    time.sleep(retry_wait_seconds)
                    attempt += 1
                    continue
                else:
                    logger.info(
                        f"\n[CI Retry] {bundle_label} failed with {reason} "
                        f"- not retrying\n"
                    )

            logger.info(
                f"\n✗ FAILED: {bundle_label} returned exit code {ret_code} "
                f"({len(dotted_modules)} files marked failed)\n"
            )
            if was_retried:
                retried_tests.append((bundle_label, attempt, "failed"))
            reason = f"bundle exit code {ret_code}"
            for m in bundle.members:
                failed_tests.append((m.filename, reason))
            return False

        except TimeoutError:
            p = process_holder.get("p")
            if p is not None:
                kill_process_tree(p.pid)
            time.sleep(5)
            file_elapsed[bundle_label] = float(bundle_timeout)
            logger.info(
                f"\n✗ TIMEOUT: {bundle_label} after {bundle_timeout:.0f} seconds\n"
            )
            if was_retried:
                retried_tests.append((bundle_label, attempt, "timeout"))
            reason = f"bundle timeout after {bundle_timeout:.0f}s"
            for m in bundle.members:
                failed_tests.append((m.filename, reason))
            return False

    return False


def _repo_relative_path(p: str) -> str:
    """Return path stripped to repo-relative form (e.g. 'test/srt/foo.py').

    Used in the machine-readable TIMINGS block so downstream scrapers
    get a stable key regardless of CI runner checkout layout. Group
    keys (``group:<group_key>``) emitted for in_process bundles are
    not filesystem paths and are returned verbatim.
    """
    if p.startswith("group:"):
        return p
    if not os.path.isabs(p):
        p = os.path.join(os.getcwd(), p)
    marker = "/sglang/"
    idx = p.rfind(marker)
    return p[idx + len(marker) :] if idx >= 0 else p


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
    # Unit-level counters for the summary line. A bundle is one unit
    # (regardless of member count); a CIRegistry/TestFile is one unit.
    units_passed_count = 0

    for i, file in enumerate(files):
        if isinstance(file, CIBundle):
            # In-process bundle: one `python3 -m unittest <mod1> <mod2> ...`
            # invocation that shares `import sglang` across all members.
            # Per-member retry/timing is intentionally simplified to per-bundle
            # for v1; on failure we fall back to running each member as its
            # own `python3 file.py -f` so blame attribution is preserved.
            bundle_passed = _run_one_bundle(
                bundle=file,
                idx=i,
                total=len(files),
                timeout_per_file=timeout_per_file,
                enable_retry=enable_retry,
                max_attempts=max_attempts,
                retry_wait_seconds=retry_wait_seconds,
                passed_tests=passed_tests,
                failed_tests=failed_tests,
                retried_tests=retried_tests,
                file_elapsed=file_elapsed,
            )
            if bundle_passed:
                units_passed_count += 1
            else:
                success = False
                if not continue_on_error:
                    break
            continue

        if isinstance(file, CIRegistry):
            filename, estimated_time = file.filename, file.est_time
        else:
            # FIXME: remove this branch after migrating all tests to use CIRegistry
            filename, estimated_time = file.name, file.estimated_time

        process = None
        output_lines = []

        def run_one_file(filename, capture_output=False):
            nonlocal process, output_lines

            full_path = os.path.join(os.getcwd(), filename)
            logger.info(
                f".\n.\nBegin ({i}/{len(files) - 1}):\npython3 {full_path}\n.\n.\n"
            )
            file_tic = time.perf_counter()

            cmd = ["python3", full_path, "-f"]

            if capture_output:
                # Capture output for retry decision
                process = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    errors="ignore",  # Ignore non-UTF-8 bytes to prevent UnicodeDecodeError
                )
                output_lines = []
                for line in process.stdout:
                    logger.info(line.rstrip())
                    output_lines.append(line)
                process.wait()
            else:
                process = subprocess.Popen(cmd, stdout=None, stderr=None)
                process.wait()

            elapsed = time.perf_counter() - file_tic
            file_elapsed[filename] = elapsed

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
                    passed_tests.append(filename)
                    units_passed_count += 1
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
                kill_process_tree(process.pid)
                time.sleep(5)
                # TimeoutError aborts run_one_file before its elapsed write;
                # record the timeout cap as an upper bound so the file still
                # appears in the TIMINGS block below.
                file_elapsed[filename] = float(timeout_per_file)
                logger.info(
                    f"\n✗ TIMEOUT: {filename} after {timeout_per_file} seconds\n"
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

    if coredump_enabled and not success:
        cuda_coredump.report()

    if success:
        logger.info(f"Success. Time elapsed: {elapsed_total:.2f}s")
    else:
        logger.info(f"Fail. Time elapsed: {elapsed_total:.2f}s")

    # Print summary
    logger.info(f"\n{'='*60}")
    logger.info(f"Test Summary: {units_passed_count}/{len(files)} unit(s) passed")
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

    return 0 if success else -1
