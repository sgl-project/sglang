import argparse
import dataclasses
import json
import random
import time
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import List, Any, Dict

import torch.cuda
from sglang import bench_serving
from sglang.srt.misc.bench_multi import presets
from sglang.srt.misc.bench_multi.configs import Config
from sglang.srt.server_args import ServerArgs
from sglang.srt.utils import kill_process_tree
from sglang.test.test_utils import launch_server_process, get_benchmark_args


def run_bench_multi(args: argparse.Namespace):
    configs = _get_configs(preset_name=args.preset_name, start_index=args.start_index, end_index=args.end_index)
    for config in configs:
        _run_one_config(config, args)


def _get_configs(preset_name: str, start_index: int, end_index: int) -> List[Config]:
    # If users want to provide their own presets, we can add a flag to support custom preset file path
    return getattr(presets, f"get_configs_{preset_name}")()[start_index:end_index]


def _run_one_config(config: Config, args: argparse.Namespace):
    server_args = ServerArgs(**config.server_args)
    with _with_server(server_args) as launch_server_id:
        for bench_serving_args in config.bench_serving_args_list:
            bench_serving_args = get_benchmark_args(*bench_serving_args)
            bench_serving_output = bench_serving.run_benchmark(bench_serving_args)
            _write_output(
                dir_output=Path(args.dir_output),
                script_args=args,
                server_args=server_args,
                bench_serving_args=bench_serving_args,
                bench_serving_output=bench_serving_output,
                launch_server_id=launch_server_id,
            )


@contextmanager
def _with_server(server_args: ServerArgs):
    launch_server_id = uuid.uuid4().hex
    proc, base_url = launch_server_process(server_args)
    try:
        yield launch_server_id
    finally:
        kill_process_tree(proc.pid)


def _write_output(
        dir_output: Path,
        script_args,
        server_args: ServerArgs,
        bench_serving_args,
        bench_serving_output: Dict[str, Any],
        launch_server_id: str,
):
    content = dict(
        script_args=vars(script_args),
        server_args=dataclasses.asdict(server_args),
        bench_serving_args=vars(bench_serving_args),
        bench_serving_output={k: v for k, v in bench_serving_output if k not in _BENCH_SERVING_OUTPUT_BLACKLIST_KEYS},
        metadata=dict(
            launch_server_id=launch_server_id,
            timestamp=time.time(),
            device_names=[torch.cuda.get_device_name(device) for device in range(torch.cuda.device_count())],
        )
    )

    path = dir_output / f'bench_multi_{time.time_ns() // 1_000_000}_{random.randrange(0, 1000000):06d}.json'
    path.write_text(json.dumps(content))


_BENCH_SERVING_OUTPUT_BLACKLIST_KEYS = ["generated_texts", "errors"]
