from argparse import ArgumentParser

from sglang.srt.misc.bench_multi.runner import run_bench_multi


def _run():
    run_bench_multi(_parse_args())


def _parse_args():
    parser = ArgumentParser(description="Benchmark multiple configurations.")
    parser.add_argument("--preset-name", type=str, required=True)
    parser.add_argument("--dir-output", type=str, default="/tmp")
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--end-index", type=int, default=1000000)
    parser.add_argument("--nnodes", type=int, default=1)
    parser.add_argument("--node-rank", type=int, default=0)
    parser.add_argument("--ctrl-dist-init-addr", type=str, default="")
    return parser.parse_args()


if __name__ == "__main__":
    _run()
