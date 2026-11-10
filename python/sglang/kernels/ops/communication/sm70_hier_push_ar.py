"""One-launch quad-then-pair fp16 all-reduce for the 8xV100 hybrid NVLink mesh,
bitwise equal to the custom-AR chain (quad 1stage, then pair 1stage)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Sequence

import torch

from sglang.kernels.jit.utils import cache_once, load_jit

if TYPE_CHECKING:
    from tvm_ffi.module import Module

# Must match kBlocks in sm70_hier_push_ar.cuh.
NUM_BLOCKS = 16
QUAD = 4
PAIR = 2
EMPTY_BYTE = 0xFF


@cache_once
def _module() -> Module:
    return load_jit(
        "sm70_hier_push_ar",
        cuda_files=["distributed/sm70_hier_push_ar.cuh"],
        cuda_wrappers=[("hier_push", "sglang::sm70_hier_push_ar::hier_push")],
    )


def quad_workspace_bytes(slot_bytes: int) -> int:
    """Bytes of one rank's quad receive buffer: [2 epochs][4 ranks][slot]."""
    return 2 * QUAD * slot_bytes


def pair_workspace_bytes(slot_bytes: int) -> int:
    return 2 * PAIR * slot_bytes


class Sm70HierPushAllReduce:
    """Holds the IPC receive pointers; the buffers must start filled with
    EMPTY_BYTE on every rank before the first call, and every rank of the 8
    must make the same sequence of calls."""

    def __init__(
        self,
        quad_ptrs: Sequence[int],
        pair_ptrs: Sequence[int],
        slot_bytes: int,
        quad_rank: int,
        pair_rank: int,
        device: torch.device,
    ):
        assert len(quad_ptrs) == QUAD and len(pair_ptrs) == PAIR
        assert slot_bytes % 16 == 0
        self.quad_ptrs = [int(p) for p in quad_ptrs]
        self.pair_ptrs = [int(p) for p in pair_ptrs]
        self.slot_bytes = slot_bytes
        self.quad_rank = quad_rank
        self.pair_rank = pair_rank
        self.epochs = torch.zeros(NUM_BLOCKS, dtype=torch.int32, device=device)
        _module()

    def covers(self, tensor: torch.Tensor) -> bool:
        nbytes = tensor.numel() * tensor.element_size()
        return (
            tensor.dtype == torch.float16
            and tensor.is_contiguous()
            and 0 < nbytes <= self.slot_bytes
            and nbytes % 16 == 0
            and tensor.data_ptr() % 16 == 0
        )

    def all_reduce(self, inp: torch.Tensor, out: torch.Tensor) -> None:
        """out = sum over the 8 ranks of inp; out may be inp."""
        _module().hier_push(
            inp.view(-1),
            out.view(-1),
            self.epochs,
            *self.quad_ptrs,
            *self.pair_ptrs,
            self.slot_bytes,
            self.quad_rank,
            self.pair_rank,
        )
