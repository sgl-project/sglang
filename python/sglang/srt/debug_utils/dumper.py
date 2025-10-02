import os
import time
from pathlib import Path
from typing import Optional

import torch
import torch.distributed as dist


class _Dumper:
    """Utility to dump tensors, which can be useful when comparison checking models.

    Example usage:
    dumper.on_forward_pass_start()
    dumper.dump("layer_start__hidden_states", hidden_states, layer_id=self.layer_id)

    Import from non-SGLang system:
    ```
    import sys
    sys.path.append("/YOUR_PATH/sglang/python/sglang/srt/debug_utils")
    from dumper import dumper
    ```

    Related: `sglang.srt.debug_utils.dump_comparator` for dump comparison
    """

    def __init__(self):
        # Do not import `sglang` to make this file standalone
        self._enable = bool(int(os.environ.get("SGLANG_DUMPER_ENABLE", "1")))
        self._base_dir = Path(os.environ.get("SGLANG_DUMPER_DIR", "/tmp"))
        self._enable_write_file = bool(
            int(os.environ.get("SGLANG_DUMPER_WRITE_FILE", "1"))
        )
        self._partial_name: Optional[str] = None
        self._dump_index = 0
        self._forward_pass_id = 0

    def on_forward_pass_start(self):
        """This should be called on all ranks."""

        if not self._enable:
            return

        # Users may want to `dump` only on some ranks, thus determine name here
        if self._partial_name is None:
            self._partial_name = _get_partial_name()

        self._forward_pass_id += 1
        print(
            f"[Dumper] [{time.time()}] on_forward_pass_start id={self._forward_pass_id}"
        )

    def dump(self, name, value, **kwargs):
        if not self._enable:
            return

        assert (
            self._forward_pass_id >= 1
        ), "Do you forget to call `dumper.on_forward_pass_start()`?"
        assert self._partial_name is not None
        self._dump_index += 1

        rank = _get_rank()
        full_kwargs = dict(
            forward_pass_id=self._forward_pass_id,
            rank=rank,
            name=name,
            dump_index=self._dump_index,
            **kwargs,
        )
        full_filename = "___".join(f"{k}={v}" for k, v in full_kwargs.items()) + ".pt"
        path = self._base_dir / f"sglang_dump_{self._partial_name}" / full_filename

        sample_value = get_truncated_value(value)

        print(
            f"[Dumper] [{rank}, {time.time()}] {path} "
            f"type={type(value)} "
            f"shape={value.shape if isinstance(value, torch.Tensor) else None} "
            f"dtype={value.dtype if isinstance(value, torch.Tensor) else None} "
            f"sample_value={sample_value}"
        )

        if self._enable_write_file:
            path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(value, str(path))


def _get_partial_name():
    rank = _get_rank()
    object_list = [str(time.time()) if rank == 0 else None]
    if dist.is_initialized():
        dist.broadcast_object_list(object_list, device="cuda")
    return object_list[0]


def _get_rank():
    if dist.is_initialized():
        return dist.get_rank()
    else:
        return 0


def get_truncated_value(value):
    if value is None:
        return None

    if isinstance(value, tuple):
        return [get_truncated_value(x) for x in value]

    if not isinstance(value, torch.Tensor):
        return None

    if value.numel() < 200:
        return value

    slices = [
        slice(0, 5) if dim_size > 200 else slice(None) for dim_size in value.shape
    ]
    return value[tuple(slices)]


dumper = _Dumper()
