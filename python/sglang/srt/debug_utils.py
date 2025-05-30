import time
from pathlib import Path


class _Dumper:
    def __init__(self):
        self._partial_name = str(time.time())

    def dump(self, name, value, **kwargs):
        from sglang.srt.distributed import get_tensor_model_parallel_rank

        rank = get_tensor_model_parallel_rank()
        full_filename = TODO
        path = Path('/tmp') / f'sglang_dump_{self._partial_name}_{rank}' / f'{full_filename}.pt'

        TODO


dumper = _Dumper()
