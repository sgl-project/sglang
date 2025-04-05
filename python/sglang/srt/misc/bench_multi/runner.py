import dataclasses
import json
import random
import time
from contextlib import contextmanager
from pathlib import Path
from typing import List, Any, Dict

import torch.cuda
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
            bench_serving_args = TODO
            bench_serving_output = bench_serving.run_benchmark(bench_serving_args)
            _write_output()


@contextmanager
def _with_server(server_args: ServerArgs):
    proc, base_url = launch_server_process(server_args)
    try:
        yield
    finally:
        kill_process_tree(proc.pid)


def _write_output(
    dir_output: Path,
    script_args,
    server_args: ServerArgs,
    bench_serving_args,
    bench_serving_output: Dict[str, Any],
):
    content = dict(
        script_args=vars(script_args),
        server_args=dataclasses.asdict(server_args),
        bench_serving_args=vars(bench_serving_args),
        bench_serving_output={k: v for k, v in bench_serving_output if k not in _BENCH_SERVING_OUTPUT_BLACKLIST_KEYS},
        metadata=dict(
            timestamp=time.time(),
            device_names=[torch.cuda.get_device_name(device) for device in range(torch.cuda.device_count())],
        )
    )

    path = dir_output / f'bench_multi_{time.time_ns() // 1_000_000}_{random.randrange(0, 1000000):06d}.json'
    path.write_text(json.dumps(content))


_BENCH_SERVING_OUTPUT_BLACKLIST_KEYS = ["generated_texts", "errors"]
