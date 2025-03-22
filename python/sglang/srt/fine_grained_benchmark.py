import json
import os
import time
from contextlib import contextmanager, nullcontext
from pathlib import Path

import torch
from sglang.srt.model_executor.forward_batch_info import ForwardBatch

_dir_output = os.environ.get('SGLANG_FINE_GRAINED_BENCHMARK_DIR')


def maybe_benchmark(forward_batch: ForwardBatch):
    return benchmark(forward_batch) if _dir_output else nullcontext()


@contextmanager
def benchmark(forward_batch: ForwardBatch):
    torch.cuda.synchronize()
    tic = time.time()
    try:
        yield
    finally:
        torch.cuda.synchronize()
        latency = time.time() - tic

        data = json.dumps(dict(
            start_time=tic,
            latency=latency,
            forward_mode=forward_batch.forward_mode.name,
            batch_size=forward_batch.batch_size,
            num_tokens=forward_batch.input_ids.shape[0],
            tp_rank=self.tp_rank,
        ))
        path = Path(self.fine_grained_benchmark_dir) / f'TP{self.tp_rank}.jsonl'
        with path.open('a') as fp:
            fp.write(f'{data}\n')
