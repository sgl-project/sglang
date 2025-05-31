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

        sample_value = self._get_sample_value(name, value)

        print(
            f"[{rank}, {time.time()}] Dump {type(value)} to {path} (sample_value={sample_value})"
        )

        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(value, str(path))

    def _get_sample_value(self, name, value):
        if (value is None) or (not isinstance(value, torch.Tensor)):
            return None

        if "topk_idx" in name:
            return value

        if ("hidden_states" in name) or ("residual" in name):
            return value[:, :3]

        return None


dumper = _Dumper()
