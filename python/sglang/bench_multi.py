from argparse import ArgumentParser

from sglang.srt.misc.bench_multi.runner import run_bench_multi


def _run():
    run_bench_multi(_parse_args())


def _parse_args():
    parser = ArgumentParser(description="Benchmark multiple configurations.")
    parser.add_argument("--preset-name", type=str, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    _run()
