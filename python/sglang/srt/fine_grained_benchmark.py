import json
import os
import shutil
import time
from contextlib import contextmanager, nullcontext
from pathlib import Path
from typing import TYPE_CHECKING, Any, List, Dict

import torch

if TYPE_CHECKING:
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch

_dir_output = os.environ.get('SGLANG_FINE_GRAINED_BENCHMARK_DIR')


def is_enabled():
    return _dir_output is not None


def maybe_benchmark(forward_batch: 'ForwardBatch', tp_rank: int):
    return benchmark(forward_batch, tp_rank) if is_enabled() else nullcontext()


@contextmanager
def benchmark(forward_batch: 'ForwardBatch', tp_rank: int):
    torch.cuda.synchronize()
    start_time = time.time()
    try:
        yield
    finally:
        torch.cuda.synchronize()
        latency = time.time() - start_time
        _write_output(forward_batch, latency, start_time, tp_rank)


def _write_output(forward_batch, latency, start_time, tp_rank):
    num_tokens = forward_batch.input_ids.shape[0]
    data = json.dumps(dict(
        start_time=start_time,
        latency=latency,
        throughput=num_tokens / latency,
        forward_mode=forward_batch.forward_mode.name,
        batch_size=forward_batch.batch_size,
        num_tokens=num_tokens,
        tp_rank=tp_rank,
    ))
    path = Path(_dir_output) / f'TP{tp_rank}.jsonl'
    with path.open('a') as fp:
        fp.write(f'{data}\n')


def clear_output():
    shutil.rmtree(_dir_output, ignore_errors=True)
    Path(_dir_output).mkdir(parents=True, exist_ok=True)


def read_output() -> List[Dict[str, Any]]:
    return [
        json.loads(row)
        for path in sorted(list(Path(_dir_output).glob('*.jsonl')))
        for row in path.read_text().split('\n')
        if row
    ]
