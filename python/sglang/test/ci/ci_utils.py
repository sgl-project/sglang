import os
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


def run_unittest_files(
    files: List[TestFile], timeout_per_file: float, continue_on_error: bool = False
):
    """
    Run a list of test files.

    Args:
        files: List of TestFile objects to run
        timeout_per_file: Timeout in seconds for each test file
        continue_on_error: If True, continue running remaining tests even if one fails.
                          If False, stop at first failure (default behavior for PR tests).
    """
    tic = time.perf_counter()
    success = True
    passed_tests = []
    failed_tests = []

    for i, file in enumerate(files):
        filename, estimated_time = file.name, file.estimated_time
        process = None

        def run_one_file(filename):
            nonlocal process

            filename = os.path.join(os.getcwd(), filename)
            print(
                f".\n.\nBegin ({i}/{len(files) - 1}):\npython3 {filename}\n.\n.\n",
                flush=True,
            )
            tic = time.perf_counter()

            process = subprocess.Popen(
                ["python3", filename], stdout=None, stderr=None, env=os.environ
            )
            process.wait()
            elapsed = time.perf_counter() - tic

            print(
                f".\n.\nEnd ({i}/{len(files) - 1}):\n{filename=}, {elapsed=:.0f}, {estimated_time=}\n.\n.\n",
                flush=True,
            )
            return process.returncode

        try:
            ret_code = run_with_timeout(
                run_one_file, args=(filename,), timeout=timeout_per_file
            )
            if ret_code != 0:
                print(
                    f"\n✗ FAILED: {filename} returned exit code {ret_code}\n",
                    flush=True,
                )
                success = False
                failed_tests.append((filename, f"exit code {ret_code}"))
                if not continue_on_error:
                    # Stop at first failure for PR tests
                    break
                # Otherwise continue to next test for nightly tests
            else:
                passed_tests.append(filename)
        except TimeoutError:
            kill_process_tree(process.pid)
            time.sleep(5)
            print(
                f"\n✗ TIMEOUT: {filename} after {timeout_per_file} seconds\n",
                flush=True,
            )
            success = False
            failed_tests.append((filename, f"timeout after {timeout_per_file}s"))
            if not continue_on_error:
                # Stop at first timeout for PR tests
                break
            # Otherwise continue to next test for nightly tests

    if success:
        print(f"Success. Time elapsed: {time.perf_counter() - tic:.2f}s", flush=True)
    else:
        print(f"Fail. Time elapsed: {time.perf_counter() - tic:.2f}s", flush=True)

    # Print summary
    print(f"\n{'='*60}", flush=True)
    print(f"Test Summary: {len(passed_tests)}/{len(files)} passed", flush=True)
    print(f"{'='*60}", flush=True)
    if passed_tests:
        print("✓ PASSED:", flush=True)
        for test in passed_tests:
            print(f"  {test}", flush=True)
    if failed_tests:
        print("\n✗ FAILED:", flush=True)
        for test, reason in failed_tests:
            print(f"  {test} ({reason})", flush=True)
    print(f"{'='*60}\n", flush=True)

    return 0 if success else -1
