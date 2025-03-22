import json
import os
import time
from contextlib import contextmanager, nullcontext
from pathlib import Path
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch

_dir_output = os.environ.get('SGLANG_FINE_GRAINED_BENCHMARK_DIR')


def maybe_benchmark(forward_batch: 'ForwardBatch', tp_rank: int):
    return benchmark(forward_batch) if _dir_output else nullcontext()


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
    data = json.dumps(dict(
        start_time=start_time,
        latency=latency,
        forward_mode=forward_batch.forward_mode.name,
        batch_size=forward_batch.batch_size,
        num_tokens=forward_batch.input_ids.shape[0],
        tp_rank=tp_rank,
    ))
    path = Path(_dir_output) / f'TPtp_rank}.jsonl'
    with path.open('a') as fp:
        fp.write(f'{data}\n')
