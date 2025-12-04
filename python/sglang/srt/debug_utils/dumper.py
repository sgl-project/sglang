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
        self._global_ctx = {}
        self._override_enable = None

    def on_forward_pass_start(self):
        """This should be called on all ranks."""

        if not self._enable:
            return

        # Users may want to `dump` only on some ranks, thus determine name here
        self._ensure_partial_name()

        self._forward_pass_id += 1
        print(
            f"[Dumper] [{time.time()}] on_forward_pass_start id={self._forward_pass_id}"
        )

    def _ensure_partial_name(self):
        if self._partial_name is None:
            self._partial_name = _get_partial_name()
            print(f"[Dumper] Choose partial_name={self._partial_name}")

    def set_ctx(self, **kwargs):
        """
        Example:

        dumper.override_enable(self.layer_id <= 3)
        dumper.set_ctx(layer_id=self.layer_id)
        ...
        dumper.set_ctx(layer_id=None)
        """
        self._global_ctx = {
            k: v for k, v in (self._global_ctx | kwargs).items() if v is not None
        }

    def override_enable(self, value: bool):
        self._override_enable = value

    def dump(self, name, value, save: bool = True, **kwargs):
        if not (self._enable and (self._override_enable is not False)):
            return

        if self._forward_pass_id < 1:
            print("Dump without on_forward_pass_start()")
        self._ensure_partial_name()
        self._dump_index += 1

        rank = _get_rank()
        full_kwargs = dict(
            forward_pass_id=self._forward_pass_id,
            rank=rank,
            name=name,
            dump_index=self._dump_index,
            **kwargs,
            **self._global_ctx,
        )
        full_filename = "___".join(f"{k}={v}" for k, v in full_kwargs.items()) + ".pt"
        path = self._base_dir / f"sglang_dump_{self._partial_name}" / full_filename

        sample_value = get_truncated_value(value)

        print(
            f"[Dumper] [{rank}, {time.time()}] {path} "
            f"type={type(value)} "
            f"shape={value.shape if isinstance(value, torch.Tensor) else None} "
            f"dtype={value.dtype if isinstance(value, torch.Tensor) else None} "
            f"device={value.device if isinstance(value, torch.Tensor) else None} "
            f"sample_value={sample_value}"
        )

        if self._enable_write_file and save:
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
        return value

    if value.numel() < 200:
        return value

    slices = [slice(0, 5) if dim_size > 50 else slice(None) for dim_size in value.shape]
    return value[tuple(slices)]


dumper = _Dumper()


def get_tensor_info(x):
    """
    from sglang.srt.debug_utils.dumper import get_tensor_info
    """
    if not isinstance(x, torch.Tensor):
        return f"type={type(x)} value={x}"
    min = x.float().min() if x.numel() > 0 else None
    max = x.float().max() if x.numel() > 0 else None
    mean = x.float().mean() if x.numel() > 0 else None
    torch.set_printoptions(precision=10)
    x_sample_head = str(x.flatten()[:5])
    x_sample_tail = str(x.flatten()[-5:])
    torch.set_printoptions(precision=4)
    return (
        f"type={type(x)} "
        f"shape={x.shape} "
        f"dtype={x.dtype} "
        f"device={x.device} "
        f"stride={x.stride()} "
        f"req_grad={x.requires_grad} "
        f"min={min} "
        f"max={max} "
        f"mean={mean} "
        f"x_sample_head={x_sample_head} "
        f"x_sample_tail={x_sample_tail}"
    )
