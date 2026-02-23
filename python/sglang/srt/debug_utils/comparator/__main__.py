import argparse

from sglang.srt.debug_utils.comparator.entrypoint import main

# python -m sglang.srt.debug_utils.comparator --baseline-path ... --target-path ...
parser = argparse.ArgumentParser()
parser.add_argument("--baseline-path", type=str)
parser.add_argument("--target-path", type=str)
parser.add_argument("--start-id", type=int, default=0)
parser.add_argument("--end-id", type=int, default=1000000)
parser.add_argument("--baseline-start-id", type=int, default=0)
parser.add_argument("--diff-threshold", type=float, default=1e-3)
parser.add_argument(
    "--filter", type=str, default=None, help="Regex to filter filenames"
)
args = parser.parse_args()
main(args)
