import argparse
import glob

from sglang.test.test_utils import TestFile, run_unittest_files

if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument(
        "--timeout-per-file",
        type=int,
        default=2000,
        help="The time limit for running one file in seconds.",
    )
    args = arg_parser.parse_args()

    files = glob.glob("**/test_*.py", recursive=True)
    # Exclude integration tests from the e2e suite; those are run separately via pytest -m integration
    files = [
        f
        for f in files
        if "/integration/" not in f and not f.startswith("integration/")
    ]
    files.sort()

    test_files = [TestFile(name=file) for file in files]
    exit_code = run_unittest_files(test_files, args.timeout_per_file)
    exit(exit_code)
