from __future__ import annotations

import dataclasses
from dataclasses import dataclass, fields
from typing import Dict, Tuple

import torch

from sglang.srt.utils import is_npu

# Keyed by ``(name, numel, dtype, device)``: buffers that mirror the same
# ForwardBatch field *and* have an identical size/dtype/device share one
# physical allocation. Keying on size — not name alone — makes the sharing
# outcome independent of registration order. The previous "reuse iff the
# existing buffer is strictly larger" rule was order-dependent two ways: a
# larger buffer arriving after a smaller one orphaned the smaller (no share),
# and equal sizes never shared at all. It could also have required repointing
# an already-registered buffer onto larger storage — unsafe, because cuda graph
# capture (which runs right after registration, per runner) burns the buffer's
# ``data_ptr`` into the graph. Size-keying guarantees an entry is never grown
# or replaced once created, so no captured buffer is ever repointed.
_PoolKey = Tuple[str, int, torch.dtype, torch.device]
_forward_input_buffer_pool: Dict[_PoolKey, torch.Tensor] = {}


def share_input_buffer(name: str, new_buffer: torch.Tensor) -> torch.Tensor:
    """Coalesce a buffer by ``(name, size, dtype, device)`` into the
    process-wide input-buffer pool.

    Distinct callers that request the same field ``name`` with the same
    size/dtype/device share one physical allocation (and therefore one
    ``data_ptr``): the first registrant's buffer becomes canonical and every
    later identical request is returned as a view aliased onto it. Requests
    that differ in size get their own allocation — they never reuse or displace
    an existing entry — so the sharing *structure* is independent of
    registration order and no already-captured buffer is ever repointed.
    """
    key: _PoolKey = (name, new_buffer.numel(), new_buffer.dtype, new_buffer.device)
    canonical = _forward_input_buffer_pool.get(key, None)
    if canonical is None:
        _forward_input_buffer_pool[key] = new_buffer
        canonical = new_buffer
    return canonical.as_strided(new_buffer.size(), new_buffer.stride())


@dataclass
class ForwardInputBuffers:

    def _share_one_buffer(self, name: str, new_buffer: torch.Tensor) -> torch.Tensor:
        return share_input_buffer(name, new_buffer)

    def share_buffers(self):
        # disable share input buffer on npu due to accuracy issue
        if is_npu():
            return

        for f in fields(self):
            name = f.name
            buffer = getattr(self, name)

            if buffer is None:
                continue

            if dataclasses.is_dataclass(buffer):
                buffer = vars(buffer)

            if isinstance(buffer, dict):
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
                ), f"Field {name} is expected to be a torch.Tensor, a dict of torch.Tensor, or a dataclass of torch.Tensor, but got {type(buffer)}."
                new_buffer = self._share_one_buffer(name, buffer)
                setattr(self, name, new_buffer)
