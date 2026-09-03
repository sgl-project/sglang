from __future__ import annotations

import dataclasses
from dataclasses import dataclass, fields
from typing import Collection, Dict, Tuple

import torch

from sglang.srt.utils import is_npu

# Process-wide pool keyed by (name, dtype, device, row shape); see share_input_buffer.
_PoolKey = Tuple[str, torch.dtype, torch.device, Tuple[int, ...]]
_forward_input_buffer_pool: Dict[_PoolKey, torch.Tensor] = {}


def share_input_buffer(name: str, new_buffer: torch.Tensor) -> torch.Tensor:
    """Coalesce a buffer by ``(name, dtype, device, shape[1:])`` into the
    process-wide input-buffer pool.

    The pool keeps one canonical allocation per key: the registered buffer
    with the most rows. A request with no more rows is returned as a prefix
    of the canonical's rows (one ``data_ptr`` for every runner that fits), so
    runners captured at different widths -- the draft / draft-extend /
    target-verify runners of every adaptive speculative step -- share one
    physical buffer per field whose leading dim is the size axis (a ``(3, N)``
    ``mrope_positions`` only coalesces with an equal ``N``). A request with
    more rows keeps its own storage and becomes the new canonical; earlier
    registrants keep the tensors their graphs were captured against, so no
    already-captured buffer is ever repointed. Because a smaller request never
    allocates, the footprint depends on registration order: the speculative
    graph runners size their width-scaled token buffers at the widest
    candidate width so the initially captured state is already the widest
    (the draft-decode ``out_cache_loc`` and the verify ``custom_mask`` stay
    per-width; both are KB-scale), and the adaptive controller builds the
    remaining steps by descending batch size.

    This pool governs *every* ``share_buffers()`` caller. Cross-runner sharing
    is safe because these are per-replay inputs: each runner fills the region
    its graph reads immediately before every replay, and the forwards that use
    them are sequential / mutually exclusive. A field whose init-time contents
    must survive other runners (e.g. draft-extend ``select_index``) is excluded
    via ``share_buffers(exclude=...)``.
    """
    if new_buffer.dim() == 0:
        return new_buffer
    key: _PoolKey = (
        name,
        new_buffer.dtype,
        new_buffer.device,
        tuple(new_buffer.shape[1:]),
    )
    canonical = _forward_input_buffer_pool.get(key, None)
    if canonical is None or canonical.shape[0] < new_buffer.shape[0]:
        _forward_input_buffer_pool[key] = new_buffer
        canonical = new_buffer
    return canonical[: new_buffer.shape[0]]


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
                    assert isinstance(
                        sub_buffer, torch.Tensor
                    ), f"Field {name}.{sub_name} is expected to be a torch.Tensor, but got {type(sub_buffer)}."
                    buffer[sub_name] = share_input_buffer(
                        f"{name}.{sub_name}", sub_buffer
                    )
            else:
                assert isinstance(
                    buffer, torch.Tensor
                ), f"Field {name} is expected to be a torch.Tensor, a dict of torch.Tensor, or a dataclass of torch.Tensor, but got {type(buffer)}."
                setattr(self, name, share_input_buffer(name, buffer))
