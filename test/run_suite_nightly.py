import argparse
import sys
from pathlib import Path

# Add test/srt to path to import from run_suite.py
sys.path.insert(0, str(Path(__file__).parent / "srt"))

from sglang.test.ci.ci_utils import run_unittest_files

# Import suites from test/srt/run_suite.py
from run_suite import suites


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--suite",
        type=str,
        required=True,
        help="Test suite to run (e.g., nightly-1-gpu, nightly-4-gpu, etc.).",
    )
    parser.add_argument(
        "--timeout-per-file",
        type=int,
        default=1200,
        help="The time limit for running one file in seconds (default: 1200).",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        default=False,
        help="Continue running remaining tests even if one fails (default: False, useful for nightly tests).",
    )
    args = parser.parse_args()

    if args.suite not in suites:
        print(f"Error: Suite '{args.suite}' not found in available suites")
        print(f"Available suites: {list(suites.keys())}")
        sys.exit(1)

    files = suites[args.suite]

    # Change directory to test/srt where the test files are located
    srt_dir = Path(__file__).parent / "srt"
    import os
    os.chdir(srt_dir)

    run_unittest_files(
        files,
        timeout_per_file=args.timeout_per_file,
        continue_on_error=args.continue_on_error,
    )


if __name__ == "__main__":
    main()
