"""CustomAllReduceV2's ring all-reduce, ``ring_simple`` (ROCm gfx1250).

Each chunk goes to the next rank only, through a fixed-size FIFO in its memory,
in ``2 * (world_size - 1)`` hops; one ring per block. It runs over a ring plane
of its own (``csrc/distributed/ring_simple.cuh``), not the shared communicator.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, List

import torch
import tvm_ffi

from sglang.kernels.jit.utils import (
    cache_once,
    is_arch_support_pdl,
    lazy_register_class,
    load_jit,
    make_cpp_args,
)
from sglang.kernels.kernel_api_logging import debug_kernel_api

if TYPE_CHECKING:
    from tvm_ffi import Module


@cache_once
def _init_ring_plane() -> None:
    module = load_jit(
        "ring_simple_plane",
        cuda_files=["distributed/ring_simple.cuh"],
        cuda_wrappers=[("register_ring_plane", "register_ring_plane")],
    )
    module.register_ring_plane()


@lazy_register_class("sgl.distributed.RingPlane", _init_ring_plane)
class RingPlane(tvm_ffi.Object):
    """Ring FIFOs: a symmetric workspace + per-ring counter semaphores.

    Every block of the launch is one ring: it streams its packets into the
    next rank's FIFO row and waits on that rank's counter before reusing a
    slot. All buffers are owned by the caller; this object only validates and
    records them.
    """

    # C++ interface
    if TYPE_CHECKING:
        rank: int
        world_size: int
        num_rings: int
        num_slots: int
        slot_bytes: int

    def __init__(
        self,
        rank: int,
        world_size: int,
        *,
        workspaces: List[torch.Tensor],
        semaphores: List[torch.Tensor],
    ) -> None:
        """
        :param workspaces: per-rank ``[num_rings, num_slots, slot_bytes]``
                           uint8 views of symmetric memory. ``num_slots``
                           must be a power of two.
        :param semaphores: per-rank ``[num_rings, 128]`` uint8 views of
                           symmetric memory, zero-filled before first use.
        """
        self.__ffi_init__(rank, world_size, workspaces, semaphores)


@lazy_register_class("sgl.distributed.HierRingPlane", _init_ring_plane)
class HierRingPlane(tvm_ffi.Object):
    """Hier ring FIFOs over DPX pairs: a symmetric workspace + per-ring semaphores.

    Each SoC's two partitions split every chunk; half h runs a ring over
    partition h of every SoC, and the pair trades inputs and results through
    two more FIFOs. All buffers are owned by the caller; this object only
    validates and records them.
    """

    # C++ interface
    if TYPE_CHECKING:
        rank: int
        world_size: int
        num_socs: int
        num_rings: int
        num_slots: int
        slot_bytes: int

    def __init__(
        self,
        rank: int,
        world_size: int,
        *,
        peer: int,
        x_prev: int,
        x_next: int,
        half: int,
        soc: int,
        num_socs: int,
        workspaces: List[torch.Tensor],
        semaphores: List[torch.Tensor],
    ) -> None:
        """
        :param peer: the other DPX partition of this rank's SoC (group rank).
        :param x_prev, x_next: the same partition of the previous / next SoC.
        :param half: this rank's index in its pair; its ring reduces that half.
        :param soc: this rank's SoC's position in the cross ring of ``num_socs``.
        :param workspaces: per group rank ``[num_rings, 3, num_slots, slot_bytes]``
                           uint8 views of symmetric memory (cross, pair-in and
                           pair-out FIFOs). ``num_slots`` must be a power of two.
        :param semaphores: per group rank ``[num_rings, 128]`` uint8 views of
                           symmetric memory, zero-filled before first use.
        """
        self.__ffi_init__(
            rank,
            world_size,
            peer,
            x_prev,
            x_next,
            half,
            soc,
            num_socs,
            workspaces,
            semaphores,
        )


@cache_once
def get_ring_simple_module(dtype: torch.dtype, world_size: int) -> Module:
    args = make_cpp_args(dtype, world_size, is_arch_support_pdl())
    return load_jit(
        "ring_simple",
        *args,
        cuda_files=["distributed/ring_simple.cuh"],
        cuda_wrappers=[("run", f"RingSimpleKernel<{args}>::run")],
    )


@cache_once
def get_hier_ring_simple_module(dtype: torch.dtype, num_socs: int) -> Module:
    args = make_cpp_args(dtype, num_socs, is_arch_support_pdl())
    return load_jit(
        "hier_ring_simple",
        *args,
        cuda_files=["distributed/ring_simple.cuh"],
        cuda_wrappers=[("run", f"HierRingSimpleKernel<{args}>::run")],
    )


@debug_kernel_api
def custom_ring_simple(ring: RingPlane, input: torch.Tensor) -> torch.Tensor:
    """All-reduce ``input`` over ``ring``; returns a new tensor."""
    result = get_ring_simple_module(input.dtype, ring.world_size).run(ring, input)
    if isinstance(result, torch.Tensor):
        return result
    return torch.from_dlpack(result)


@debug_kernel_api
def custom_hier_ring_simple(ring: HierRingPlane, input: torch.Tensor) -> torch.Tensor:
    """All-reduce ``input`` over the hier ``ring``; returns a new tensor."""
    result = get_hier_ring_simple_module(input.dtype, ring.num_socs).run(ring, input)
    if isinstance(result, torch.Tensor):
        return result
    return torch.from_dlpack(result)
