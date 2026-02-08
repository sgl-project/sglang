# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# Use of this software is governed by the terms and conditions of the
# NVIDIA End User License Agreement (EULA), available at:
# https://docs.nvidia.com/cutlass/media/docs/pythonDSL/license.html
#
# Any use, reproduction, disclosure, or distribution of this software
# and related documentation outside the scope permitted by the EULA
# is strictly prohibited.

import enum
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Union
import warnings

import cutlass.cute as cute
from cutlass.cutlass_dsl import Boolean, Int32, Int64, if_generate
from cutlass._mlir.dialects import llvm
import cutlass._mlir.dialects.cute as _cute_ir


##############################################################################
# Agent class
##############################################################################


class Agent(enum.Enum):
    """
    Agent indicates what is participating in the pipeline synchronization.
    """

    # Arbitrary grouping of N threads
    Thread = enum.auto()
    # Same as AsyncThread, but includes all threads in the block
    ThreadBlock = enum.auto()
    # Same as AsyncThread, but includes all threads in the cluster
    ThreadBlockCluster = enum.auto()


class CooperativeGroup:
    """
    CooperativeGroup contains size and alignment restrictions for an Agent.
    """

    def __init__(self, agent: Agent, size: int = 1, alignment: int = 1):
        if agent is Agent.Thread:
            assert size > 0
            if size == 32:
                assert (
                    size == alignment
                ), "Error: Alignment does not match number of threads in a warp."
            elif size == 128:
                assert (
                    size == alignment
                ), "Error: Alignment does not match number of threads in a warpgroup."
        elif agent is Agent.ThreadBlock:
            raise NotImplementedError("Error: Not yet supported.")
        elif agent is Agent.ThreadBlockCluster:
            raise NotImplementedError("Error: Not yet supported.")
        else:
            # Should never reach this state
            size = 0

        if size <= 0:
            raise ValueError(
                "Error: The number of threads in a CooperativeGroup must be more than 0."
            )

        # Size indicates how many threads are participating in this CooperativeGroup
        self.size = size
        # Agent indicates the type of thread group
        self.agent = agent


class PipelineOp(enum.Enum):
    """
    PipelineOp assigns an operation to an agent corresponding to a specific hardware feature.
    """

    # async-threads
    AsyncThread = enum.auto()
    # Blackwell (SM100a) MMA instruction
    TCGen05Mma = enum.auto()
    # Tensor Memory Accelerator load
    TmaLoad = enum.auto()
    # TMA Store consuming smem produced by AsyncThread
    TmaStore = enum.auto()
    # Composite of multiple PipelineOps
    Composite = enum.auto()
    # Async load without TMA
    AsyncLoad = enum.auto()


def _get_pipeline_op(type_str):
    return PipelineOp(type_str)


##############################################################################
# SyncObject class
##############################################################################


class SyncObject(ABC):
    """Abstract base class for hardware synchronization primitives.

    This class defines the interface for different types of hardware synchronization
    mechanisms including shared memory barriers, named barriers, and fences.
    """

    @abstractmethod
    def arrive(self) -> None:
        pass

    @abstractmethod
    def wait(self) -> None:
        pass

    @abstractmethod
    def arrive_and_wait(self) -> None:
        pass

    @abstractmethod
    def arrive_and_drop(self) -> None:
        pass

    @abstractmethod
    def get_barrier(self) -> Union[cute.Pointer, int, None]:
        pass

    @abstractmethod
    def max(self) -> Union[int, None]:
        pass


class MbarrierArray(SyncObject):
    """
    MbarrierArray implements an abstraction for an array of smem barriers.
    """

    def __init__(
        self,
        barrier_storage: cute.Pointer,
        num_stages: int,
        agent: tuple[PipelineOp, CooperativeGroup],
        tx_count: int = 0,
    ) -> None:
        self.barrier_storage = barrier_storage
        self.tx_count = tx_count
        self.num_stages = num_stages
        self.op_type, self.cg = agent
        self.arrive_count = self.cg.size

        if self.num_stages <= 0:
            raise ValueError("Error: Mbarrier stage count must be greater than 0.")
        if self.arrive_count <= 0:
            raise ValueError("Error: Mbarrier arrive count must be greater than 0.")
        if self.op_type is PipelineOp.TmaLoad and self.tx_count < 0:
            raise ValueError(
                "Error: Mbarrier tx count must not be less than 0 for TMA ops."
            )

        # Store mbarrier base pointer
        self.mbarrier_base = self.barrier_storage

        # Mbarrier initialization in constructor
        self.mbarrier_init()

    def recast_to_new_op_type(self, new_op_type: PipelineOp) -> "MbarrierArray":
        """
        Creates a copy of MbarrierArray with a different op_type without re-initializing barriers
        """
        # Create new instance without initialization
        new_mbarrier_array = object.__new__(MbarrierArray)

        # Copy all attributes directly
        new_mbarrier_array.barrier_storage = self.barrier_storage
        new_mbarrier_array.op_type = new_op_type
        new_mbarrier_array.cg = self.cg
        new_mbarrier_array.num_stages = self.num_stages
        new_mbarrier_array.tx_count = self.tx_count
        new_mbarrier_array.arrive_count = self.arrive_count
        new_mbarrier_array.mbarrier_base = self.mbarrier_base
        return new_mbarrier_array

    # Mbarrier initialization
    def mbarrier_init(self) -> None:
        """
        Initializes an array of mbarriers using warp 0.
        """

        def then_body():
            for index in range(self.num_stages):
                cute.arch.mbarrier_init(self.get_barrier(index), self.arrive_count)

        warp_idx = cute.arch.warp_idx()
        warp_idx = cute.arch.make_warp_uniform(warp_idx)

        if_generate(warp_idx == 0, then_body)

    def arrive(
        self,
        index: int,
        dst: int,
        cta_group: Optional[cute.nvgpu.tcgen05.CtaGroup] = None,
    ) -> None:
        """Select the arrive corresponding to this MbarrierArray's PipelineOp.

        :param index: Index of the mbarrier in the array to arrive on
        :type index: int
        :param dst: Destination parameter for selective arrival, which can be either a mask or destination cta rank.
            When None, both ``TCGen05Mma`` and ``AsyncThread`` will arrive on their local mbarrier.
            - For ``TCGen05Mma``, ``dst`` serves as a multicast mask (e.g., 0b1011 allows arrive signal to be multicast to CTAs
            in the cluster with rank = 0, 1, and 3).
            - For ``AsyncThread``, ``dst`` serves as a destination cta rank (e.g., 3 means threads will arrive on
            the mbarrier with rank = 3 in the cluster).
        :type dst: int | None
        :param cta_group: CTA group for ``TCGen05Mma``, defaults to None for other op types
        :type cta_group: ``cute.nvgpu.tcgen05.CtaGroup``, optional
        """
        if self.op_type is PipelineOp.AsyncThread:
            self.arrive_mbarrier(index, dst)
        elif self.op_type is PipelineOp.TCGen05Mma:
            assert (
                cta_group is not None
            ), "Error: CTA group must be provided for TCGen05Mma."
            self.arrive_tcgen05mma(index, dst, cta_group)
        elif self.op_type in [PipelineOp.TmaLoad]:
            self.arrive_and_expect_tx(index, self.tx_count)
        elif self.op_type is PipelineOp.AsyncLoad:
            self.arrive_cp_async_mbarrier(index)
        else:
            assert (
                False
            ), f"Error: MbarrierArray is not supported for PipelineOp: {_get_pipeline_op(self.op_type)}."

    def arrive_mbarrier(self, index: int, dst_rank: Optional[int] = None) -> None:
        if dst_rank is None:
            cute.arch.mbarrier_arrive(self.get_barrier(index))
        else:
            cute.arch.mbarrier_arrive(self.get_barrier(index), dst_rank)

    def arrive_cp_async_mbarrier(self, index: int):
        cute.arch.cp_async_mbarrier_arrive_noinc(self.get_barrier(index))

    def arrive_tcgen05mma(
        self, index: int, mask: Optional[int], cta_group: cute.nvgpu.tcgen05.CtaGroup
    ) -> None:
        if mask is None:
            with cute.arch.elect_one():
                cute.nvgpu.tcgen05.commit(self.get_barrier(index))
        else:
            with cute.arch.elect_one():
                cute.nvgpu.tcgen05.commit(self.get_barrier(index), mask, cta_group)

    def arrive_and_expect_tx(self, index: int, tx_count: int) -> None:
        with cute.arch.elect_one():
            cute.arch.mbarrier_arrive_and_expect_tx(self.get_barrier(index), tx_count)

    def try_wait(self, index: int, phase: int) -> Boolean:
        return cute.arch.mbarrier_try_wait(self.get_barrier(index), phase)

    def wait(self, index: int, phase: int) -> None:
        cute.arch.mbarrier_wait(self.get_barrier(index), phase)

    def arrive_and_wait(
        self,
        index: int,
        phase: int,
        dst: int,
        cta_group: Optional[cute.nvgpu.tcgen05.CtaGroup] = None,
    ) -> None:
        arrive(index, dst, cta_group)
        wait(index, phase)

    def arrive_and_drop(self) -> None:
        raise NotImplementedError("Error: Not yet supported.")

    def get_barrier(self, index: int) -> cute.Pointer:
        return self.mbarrier_base + index

    def max(self) -> int:
        # Transaction barriers have a maximum arrive count of 511 (2^9 - 1).
        # Non-transaction barriers have a maximum arrive count of 1,048,575 (2^20 - 1).
        return 511

    def __extract_mlir_values__(self):
        return [self.barrier_storage]

    def __new_from_mlir_values__(self, values):
        return MbarrierArray(
            values[0], self.num_stages, (self.op_type, self.cg), self.tx_count
        )


@dataclass(frozen=True)
class NamedBarrier(SyncObject):
    """
    NamedBarrier is an abstraction for named barriers managed by hardware.
    There are 16 named barriers available, with barrier_ids 0-15.

    See the `PTX documentation <https://https://docs.nvidia.com/cuda/parallel-thread-execution/#parallel-synchronization-and-communication-instructions-bar>`__.
    """

    barrier_id: int
    num_threads: int

    def __post_init__(self) -> None:
        if self.barrier_id < 0 or self.barrier_id >= 16:
            raise ValueError("Error: NamedBarrier ID must be between 0 and 16.")
        if self.barrier_id == 0:
            warnings.warn(
                "NamedBarrier ID 0 is by other driver APIs (i.e. sync_threads()) and should not be used."
            )

    def arrive(self) -> None:
        """
        The aligned flavor of arrive is used when all threads in the CTA will execute the
        same instruction. See PTX documentation.
        """
        cute.arch.barrier_arrive(
            barrier_id=self.barrier_id, number_of_threads=self.num_threads
        )

    def arrive_unaligned(self) -> None:
        """
        The unaligned flavor of arrive can be used with an arbitrary number of threads in the CTA.
        """
        llvm.inline_asm(
            None,
            [Int32(self.barrier_id).ir_value(), Int32(self.num_threads).ir_value()],
            "barrier.arrive $0, $1;",
            "r,r",
            has_side_effects=True,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )

    def wait(self) -> None:
        """
        NamedBarriers do not have a standalone wait like mbarriers, only an arrive_and_wait.
        If synchronizing two warps in a producer/consumer pairing, the arrive count would be
        32 using mbarriers but 64 using NamedBarriers. Only threads from either the producer
        or consumer are counted for mbarriers, while all threads participating in the sync
        are counted for NamedBarriers.
        """
        warnings.warn(
            "NamedBarrier wait also arrives on the barrier. Routing call to NamedBarrier.arrive_and_wait()."
        )
        self.arrive_and_wait()

    def wait_unaligned(self) -> None:
        warnings.warn(
            "NamedBarrier wait also arrives on the barrier. Routing call to NamedBarrier.arrive_and_wait()."
        )
        llvm.inline_asm(
            None,
            [Int32(self.barrier_id).ir_value(), Int32(self.num_threads).ir_value()],
            "barrier.sync $0, $1;",
            "r,r",
            has_side_effects=True,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )

    def arrive_and_wait(self) -> None:
        cute.arch.barrier(
            barrier_id=self.barrier_id, number_of_threads=self.num_threads
        )

    def arrive_and_drop(self) -> None:
        raise NotImplementedError("Error: Not supported.")

    def sync(self) -> None:
        cute.arch.barrier(barrier_id=self.barrier_id)

    def get_barrier(self) -> int:
        return self.barrier_id

    def max(self) -> int:
        # Transaction barriers have a maximum arrive count of 4095 (2^12 - 1).
        return 4095


class TmaStoreFence(SyncObject):
    """
    TmaStoreFence is used for a multi-stage epilogue buffer.
    """

    def __init__(self, num_stages: int = 0) -> None:
        if num_stages <= 0:
            raise ValueError("Mbarrier stage count must be greater than 0.")

        self.num_stages = num_stages

    def arrive(self) -> None:
        cute.arch.cp_async_bulk_commit_group()

    def wait(self) -> None:
        cute.arch.cp_async_bulk_wait_group(self.num_stages - 1, read=True)

    def arrive_and_wait(self) -> None:
        self.arrive()
        self.wait()

    def arrive_and_drop(self) -> None:
        raise NotImplementedError("Error: Not supported.")

    # TmaStoreFence doesn't have mbarriers
    def get_barrier(self) -> None:
        assert (
            False
        ), "Error: TmaStoreFence doesn't use mbarriers and cannot return a barrier."

    def max(self) -> None:
        raise NotImplementedError("Error: Not supported.")

    def tail(self) -> None:
        cute.arch.cp_async_bulk_wait_group(0, read=True)


##############################################################################
# PipelineState class
##############################################################################


class PipelineUserType(enum.Enum):
    Producer = enum.auto()
    Consumer = enum.auto()


class PipelineState:
    """
    Pipeline state contains an index and phase bit corresponding to the current position in the circular buffer.
    """

    def __init__(self, stages: int, count, index, phase):
        self._stages = stages
        self._count = count
        self._index = index
        self._phase = phase

    def clone(self) -> "PipelineState":
        return PipelineState(self.stages, self._count, self.index, self.phase)

    @property
    def index(self) -> Int32:
        return self._index

    @property
    def count(self) -> Int32:
        return self._count

    @property
    def stages(self) -> int:
        return self._stages

    @property
    def phase(self) -> Int32:
        return self._phase

    def reset_count(self):
        self._count = Int32(0)

    def advance(self):
        self._index += 1
        self._count += 1

        def then_body(index, phase):
            new_index = Int32(0)
            new_phase = phase ^ 1
            return new_index, new_phase

        def else_body(index, phase):
            return index, phase

        self._index, self._phase = if_generate(
            self._index == self.stages,
            then_body,
            else_body,
            [self.index, self.phase],
            [Int32, Int32],
        )

    def reverse(self):
        self._index -= 1
        self._count -= 1

        def then_body(index, phase):
            new_index = Int32(self.stages - 1)
            new_phase = phase ^ 1
            return new_index, new_phase

        def else_body(index, phase):
            return index, phase

        self._index, self._phase = if_generate(
            self._index == -1,
            then_body,
            else_body,
            [self.index, self.phase],
            [Int32, Int32],
        )

    def __get_mlir_types__(self):
        return [self._count.type, self._index.type, self._phase.type]

    def __extract_mlir_values__(self):
        count = self._count
        index = self._index
        phase = self._phase
        return [count.ir_value(), index.ir_value(), phase.ir_value()]

    # This can be overridden by derived classes
    def __new_from_mlir_values__(self, values):
        return PipelineState(
            self.stages, Int32(values[0]), Int32(values[1]), Int32(values[2])
        )


def make_pipeline_state(type: PipelineUserType, stages: int):
    """
    Creates a pipeline state. Producers are assumed to start with an empty buffer and have a flipped phase bit of 1.
    """
    if type is PipelineUserType.Producer:
        return PipelineState(
            stages,
            Int32(0),
            Int32(0),
            Int32(1),
        )
    elif type is PipelineUserType.Consumer:
        return PipelineState(
            stages,
            Int32(0),
            Int32(0),
            Int32(0),
        )
    else:
        assert (
            False
        ), "Error: invalid PipelineUserType specified for make_pipeline_state."


##############################################################################
# Helper functions
##############################################################################


def pipeline_init_wait(cta_layout_vmnk: Optional[cute.Layout] = None):
    """
    Fences the mbarrier init and syncs the threadblock or cluster
    """
    cute.arch.mbarrier_init_fence()

    if cta_layout_vmnk is None or cute.size(cta_layout_vmnk) == 1:
        # If not using clusters, sync the threadblock
        _sync(Agent.ThreadBlock)
    else:
        # If using clusters, sync the cluster
        _sync(Agent.ThreadBlockCluster)


def _sync(group: Agent):
    """
    Syncs all threads within an agent.
    """
    if group is Agent.Thread:
        raise NotImplementedError("Error: Not supported.")
    elif group is Agent.ThreadBlock:
        cute.arch.sync_threads()
    elif group is Agent.ThreadBlockCluster:
        cute.arch.cluster_arrive()
        cute.arch.cluster_wait()
    else:
        assert (
            False
        ), "Error: No explicit sync instruction exists. Please use barriers (named / mbarrier) instead."


def _mbarrier_i64_to_ptr(val: Int64) -> cute.Pointer:
    """
    Converts a smem pointer of type Int64 to cute.Pointer with 8B alignment
    """
    return cute.make_ptr(
        Int64,
        val.ir_value(),
        mem_space=_cute_ir.AddressSpace.smem,
        assumed_align=8,
    )


# NamedBarrier free functions
def arrive(barrier_id: int, num_threads: int):
    """
    The aligned flavor of arrive is used when all threads in the CTA will execute the
    same instruction. See PTX documentation.
    """
    cute.arch.barrier_arrive(barrier_id=barrier_id, number_of_threads=num_threads)


def arrive_unaligned(barrier_id: int, num_threads: int):
    """
    The unaligned flavor of arrive can be used with an arbitrary number of threads in the CTA.
    """
    llvm.inline_asm(
        None,
        [Int32(barrier_id).ir_value(), Int32(num_threads).ir_value()],
        "barrier.arrive $0, $1;",
        "r,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )


def wait(barrier_id: int, num_threads: int):
    """
    NamedBarriers do not have a standalone wait like mbarriers, only an arrive_and_wait.
    If synchronizing two warps in a producer/consumer pairing, the arrive count would be
    32 using mbarriers but 64 using NamedBarriers. Only threads from either the producer
    or consumer are counted for mbarriers, while all threads participating in the sync
    are counted for NamedBarriers.
    """
    warnings.warn(
        "NamedBarrier wait also arrives on the barrier. Routing call to NamedBarrier.arrive_and_wait()."
    )
    arrive_and_wait()


def wait_unaligned(barrier_id: int, num_threads: int):
    warnings.warn(
        "NamedBarrier wait also arrives on the barrier. Routing call to NamedBarrier.arrive_and_wait()."
    )
    llvm.inline_asm(
        None,
        [Int32(barrier_id).ir_value(), Int32(num_threads).ir_value()],
        "barrier.sync $0, $1;",
        "r,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )


def arrive_and_wait(barrier_id: int, num_threads: int):
    cute.arch.barrier(barrier_id=barrier_id, number_of_threads=num_threads)


def sync(barrier_id: int = 0):
    cute.arch.barrier(barrier_id=barrier_id)
