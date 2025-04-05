from contextlib import contextmanager
from typing import List

from sglang import bench_serving
from sglang.srt.misc.bench_multi.configs import Config
from sglang.srt.server_args import ServerArgs
from sglang.srt.utils import kill_process_tree
from sglang.test.test_utils import launch_server_process


def run_bench_multi():
    configs = _get_configs()
    for config in configs:
        _run_one_config(config)


def _get_configs() -> List[Config]:
    return TODO


def _run_one_config(config: Config):
    with _with_server(server_args):
        for _ in range(TODO):
            bench_serving_output = bench_serving.run_benchmark(TODO)
            _write_output()


@contextmanager
def _with_server(server_args: ServerArgs):
    proc, base_url = launch_server_process(server_args)
    try:
        yield
    finally:
        kill_process_tree(proc.pid)


def _write_output():
    TODO
