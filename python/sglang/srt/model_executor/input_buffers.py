from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Dict

import torch

_forward_input_buffer_pool: Dict[str, torch.Tensor] = {}


@dataclass
class ForwardInputBuffers:

    def _share_one_buffer(self, name: str, new_buffer: torch.Tensor) -> torch.Tensor:

        buffer_size = new_buffer.size()
        buffer_stride = new_buffer.stride()

        old_buffer = _forward_input_buffer_pool.get(name, None)
        if old_buffer is not None:
            assert (
                new_buffer.dtype == old_buffer.dtype
            ), f"Buffer {name} has different dtype than before."
            assert (
                new_buffer.device == old_buffer.device
            ), f"Buffer {name} has different device than before."
            if old_buffer.numel() > new_buffer.numel():
                new_buffer = old_buffer

        _forward_input_buffer_pool[name] = new_buffer
        return new_buffer.as_strided(buffer_size, buffer_stride)

    def share_buffers(self):

        for f in fields(self):
            name = f.name
            buffer = getattr(self, name)

            if buffer is None:
                continue
            elif isinstance(buffer, dict):
                for sub_name, sub_buffer in buffer.items():
                    assert isinstance(
                        sub_buffer, torch.Tensor
                    ), f"Field {name}.{sub_name} is expected to be a torch.Tensor, but got {type(sub_buffer)}."
                    new_buffer = self._share_one_buffer(
                        f"{name}.{sub_name}", sub_buffer
                    )
                    buffer[sub_name] = new_buffer
            else:
                assert isinstance(
                    buffer, torch.Tensor
                ), f"Field {name} is expected to be a torch.Tensor or a dict of torch.Tensor, but got {type(buffer)}."
                new_buffer = self._share_one_buffer(name, buffer)
                setattr(self, name, new_buffer)
