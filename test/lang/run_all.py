import argparse
import glob
import multiprocessing
import os
import time
import unittest

from sglang.utils import run_with_timeout


def run_unittest_files(files, args):
    for filename in files:

        def func():
            print(filename)
            ret = unittest.main(module=None, argv=["", "-vb"] + [filename])

        p = multiprocessing.Process(target=func)

        def run_one_file():
            p.start()
            p.join()

        try:
            run_with_timeout(run_one_file, timeout=args.time_limit_per_file)
            if p.exitcode != 0:
                return False
        except TimeoutError:
            p.terminate()
            time.sleep(5)
            print(
                f"\nTimeout after {args.time_limit_per_file} seconds "
                f"when running {filename}"
            )
            return False

    return True


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument(
        "--time-limit-per-file",
        type=int,
        default=1000,
        help="The time limit for running one file in seconds.",
    )
    args = arg_parser.parse_args()

    files = glob.glob("**/test_*.py", recursive=True)

    tic = time.time()
    success = run_unittest_files(files, args)

    if success:
        print(f"Success. Time elapsed: {time.time() - tic:.2f}s")
    else:
        print(f"Fail. Time elapsed: {time.time() - tic:.2f}s")

    exit(0 if success else -1)
