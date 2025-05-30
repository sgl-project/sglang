from pathlib import Path


class _Dumper:
    def __init__(self):
        self._dir_base = Path('/tmp') / TODO

    def dump(self, name, value):
        from sglang.srt.distributed import get_tensor_model_parallel_rank

        TODO


dumper = _Dumper()
