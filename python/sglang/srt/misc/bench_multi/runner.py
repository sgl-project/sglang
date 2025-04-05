from contextlib import contextmanager
from typing import List

from sglang.srt.misc.bench_multi.configs import Config


def run_bench_multi():
    configs = _get_configs()
    for config in configs:
        _run_one_config(config)


def _get_configs() -> List[Config]:
    return TODO


def _run_one_config(config: Config):
    with _with_launch_server():
        for _ in range(TODO):
            _run_bench_serving()


@contextmanager
def _with_launch_server():
    TODO
    try:
        yield
    finally:
        TODO


def _run_bench_serving():
    TODO
