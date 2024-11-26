import argparse
import glob

from sglang.test.test_utils import run_unittest_files

if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument(
        "--timeout-per-file",
        type=int,
        default=1000,
        help="The time limit for running one file in seconds.",
    )
    args = arg_parser.parse_args()

    files = glob.glob("**/test_*.py", recursive=True)

    exit_code = run_unittest_files(files, args.timeout_per_file)
    exit(exit_code)
