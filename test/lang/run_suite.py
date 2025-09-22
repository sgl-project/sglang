import argparse
import glob

from sglang.test.test_utils import TestFile, run_unittest_files

suites = {
    "per-commit": [
        TestFile("test_srt_backend.py"),
        # Skip this due to some OPENAI_API_KEY issues
        # "test_openai_backend.py",
    ],
}


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument(
        "--timeout-per-file",
        type=int,
        default=1000,
        help="The time limit for running one file in seconds.",
    )
    arg_parser.add_argument(
        "--suite",
        type=str,
        default=list(suites.keys())[0],
        choices=list(suites.keys()) + ["all"],
        help="The suite to run",
    )
    args = arg_parser.parse_args()

    if args.suite == "all":
        files = glob.glob("**/test_*.py", recursive=True)
    else:
        files = suites[args.suite]

    exit_code = run_unittest_files(files, args.timeout_per_file)
    exit(exit_code)
