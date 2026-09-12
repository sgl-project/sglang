"""
Simple wrapper to run a test file with retry logic.

Usage:
    python3 -m sglang.test.ci.run_with_retry test_file.py [--max-attempts 2] [--retry-wait 60]
"""

import argparse
import sys

from sglang.test.ci.ci_utils import TestFile, run_unittest_files


def main():
    parser = argparse.ArgumentParser(description="Run a test file with retry logic")
    parser.add_argument("test_file", help="The test file to run")
    parser.add_argument(
        "--max-attempts",
        type=int,
        default=2,
        help="Maximum number of attempts (default: 2)",
    )
    parser.add_argument(
        "--retry-wait",
        type=int,
        default=60,
        help="Seconds to wait between retries (default: 60)",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=1200,
        help="Timeout per attempt in seconds (default: 1200)",
    )
    args = parser.parse_args()

    # Create a TestFile with a reasonable estimated time
    test_file = TestFile(name=args.test_file, estimated_time=args.timeout)

    exit_code = run_unittest_files(
        files=[test_file],
        timeout_per_file=args.timeout,
        continue_on_error=False,
        enable_retry=True,
        max_attempts=args.max_attempts,
        retry_wait_seconds=args.retry_wait,
    )

    sys.exit(exit_code)


if __name__ == "__main__":
    main()
