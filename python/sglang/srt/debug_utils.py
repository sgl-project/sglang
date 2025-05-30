import os
import time
from pathlib import Path

import torch


class _Dumper:
    def __init__(self):
        self._base_dir = Path(os.environ.get("SGLANG_DUMPER_DIR", "/tmp"))
        self._partial_name = str(time.time())
        self.forward_pass_id = None

    def dump(self, name, value, **kwargs):
        from sglang.srt.distributed import get_tensor_model_parallel_rank

        rank = get_tensor_model_parallel_rank()
        full_kwargs = dict(
            forward_pass_id=self.forward_pass_id,
            name=name,
            **kwargs,
        )
        full_filename = "___".join(f"{k}={v}" for k, v in full_kwargs.items()) + ".pt"
        path = (
            self._base_dir / f"sglang_dump_{self._partial_name}_{rank}" / full_filename
        )

        path.mkdir(parents=True, exist_ok=True)
        torch.save(value, str(name))


dumper = _Dumper()
