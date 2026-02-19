from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Dict

import torch

_forward_input_buffer_pool: Dict[str, torch.Tensor] = {}


@dataclass
class ForwardInputBuffers:

    def _build_one_buffer(self, name: str, input_buffer: torch.Tensor) -> torch.Tensor:

        assert input_buffer.is_contiguous()

        buffer_size = input_buffer.size()
        buffer_numel = input_buffer.numel()
        # buffer_stride = input_buffer.stride()

        old_buffer = _forward_input_buffer_pool.get(name, None)
        if old_buffer is not None:
            assert (
                input_buffer.dtype == old_buffer.dtype
            ), f"Buffer {name} has different dtype than before."
            assert (
                input_buffer.device == old_buffer.device
            ), f"Buffer {name} has different device than before."
            if old_buffer.numel() > input_buffer.numel():
                input_buffer = old_buffer.view(-1)
            else:
                input_buffer = input_buffer.view(-1)

        _forward_input_buffer_pool[name] = input_buffer
        # output_buffer = input_buffer.as_strided(buffer_size, buffer_stride)
        output_buffer = input_buffer[:buffer_numel].view(buffer_size)

        assert output_buffer.is_contiguous()

        return output_buffer

    def build(self):

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
                    new_buffer = self._build_one_buffer(
                        f"{name}.{sub_name}", sub_buffer
                    )
                    buffer[sub_name] = new_buffer
            else:
                assert isinstance(
                    buffer, torch.Tensor
                ), f"Field {name} is expected to be a torch.Tensor or a dict of torch.Tensor, but got {type(buffer)}."
                new_buffer = self._build_one_buffer(name, buffer)
                setattr(self, name, new_buffer)
