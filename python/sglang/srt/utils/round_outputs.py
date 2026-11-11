"""Ring-held outputs of a speculative decode round's index kernels
(SGLANG_OPT_SPEC_INDEX_OUTPUT_RING): draft cache locations, chain-tree verify inputs
and the draft-extend layout."""

import torch

from sglang.srt.environ import envs

ENABLED = envs.SGLANG_OPT_SPEC_INDEX_OUTPUT_RING.get()
# A pooled tensor is handed out again 8 requests of its shape later; the overlap loop's
# readers of a round's index outputs finish within two rounds.
_RING_DEPTH = 8
_rings: dict = {}


def torch_empty(shape: tuple, dtype: torch.dtype, device) -> torch.Tensor:
    return torch.empty(shape, dtype=dtype, device=device)


def round_empty(shape: tuple, dtype: torch.dtype, device) -> torch.Tensor:
    """torch.empty for a decode round's index outputs (every element written by the
    producing kernel, read only within the round), from a ring per (shape, dtype, device)."""
    key = (shape, dtype, device)
    ring = _rings.get(key)
    if ring is None:
        ring = _rings[key] = [
            [
                torch.empty(shape, dtype=dtype, device=device)
                for _ in range(_RING_DEPTH)
            ],
            0,
        ]
    pos = ring[1]
    ring[1] = pos + 1 if pos + 1 < _RING_DEPTH else 0
    return ring[0][pos]


def empty_fn():
    return round_empty if ENABLED else torch_empty
