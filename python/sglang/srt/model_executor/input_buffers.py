from __future__ import annotations

import dataclasses
from dataclasses import dataclass, fields
from typing import Collection, Dict, Tuple

import torch

from sglang.srt.utils import is_npu

# Process-wide pool keyed by (name, numel, dtype, device); see share_input_buffer.
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

    This pool is process-wide and governs *every* ``share_buffers()`` caller —
    including graph runners not yet on the registry (the speculative draft /
    draft-extend / frozen-kv-mtp / multi-layer-eagle runners), which register
    identically-named ``input_ids`` / ``positions`` / ``out_cache_loc`` /
    ``mrope_positions``. Cross-runner sharing is safe because those buffers are
    filled immediately before each replay and the forwards that use them are
    sequential / mutually exclusive.
    """
    key: _PoolKey = (name, new_buffer.numel(), new_buffer.dtype, new_buffer.device)
    canonical = _forward_input_buffer_pool.get(key, None)
    if canonical is None:
        _forward_input_buffer_pool[key] = new_buffer
        canonical = new_buffer
    return canonical.as_strided(new_buffer.size(), new_buffer.stride())


# Values that index the rope table, the KV pool, req_to_token, or the mamba
# state pool, so stale content is unsafe to execute. build_decode_registry
# asserts its ZERO-policy slots against this set.
INDEX_SEMANTIC_BUFFERS = frozenset(
    {
        "positions",
        "mrope_positions",
        "out_cache_loc",
        "req_pool_indices",
        "mamba_track_indices",
        "mamba_track_mask",
    }
)


@dataclass
class ForwardInputBuffers:
    def reset_index_buffers(self) -> None:
        """Zero the index-semantic buffers this set declares."""
        for f in fields(self):
            if f.name not in INDEX_SEMANTIC_BUFFERS:
                continue
            buffer = getattr(self, f.name)
            if buffer is not None:
                buffer.zero_()

    def share_buffers(self, *, exclude: Collection[str] = ()):
        # disable share input buffer on npu due to accuracy issue
        if is_npu():
            return

        for f in fields(self):
            name = f.name
            if name in exclude:
                continue
            buffer = getattr(self, name)

            if buffer is None:
                continue

            if dataclasses.is_dataclass(buffer):
                buffer = vars(buffer)

            if isinstance(buffer, dict):
                for sub_name, sub_buffer in buffer.items():
                    assert isinstance(sub_buffer, torch.Tensor), (
                        f"Field {name}.{sub_name} is expected to be a torch.Tensor, but got {type(sub_buffer)}."
                    )
                    buffer[sub_name] = share_input_buffer(
                        f"{name}.{sub_name}", sub_buffer
                    )
            else:
                assert isinstance(buffer, torch.Tensor), (
                    f"Field {name} is expected to be a torch.Tensor, a dict of torch.Tensor, or a dataclass of torch.Tensor, but got {type(buffer)}."
                )
                setattr(self, name, share_input_buffer(name, buffer))
