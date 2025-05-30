import time
from pathlib import Path

import torch


class _Dumper:
    def __init__(self):
        self._partial_name = str(time.time())
        self.forward_pass_id = None

    def dump(self, name, value):
        from sglang.srt.distributed import get_tensor_model_parallel_rank

        rank = get_tensor_model_parallel_rank()
        full_filename = f'F{self.forward_pass_id}__{name}.pt'
        path = Path('/tmp') / f'sglang_dump_{self._partial_name}_{rank}' / full_filename

        path.mkdir(parents=True, exist_ok=True)
        torch.save(value, str(name))


dumper = _Dumper()
