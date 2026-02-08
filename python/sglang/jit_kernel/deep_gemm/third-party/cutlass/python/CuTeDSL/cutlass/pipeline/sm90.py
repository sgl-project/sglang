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
from typing import Type, Tuple
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Union
import warnings

import cutlass
import cutlass.cute as cute
from cutlass.cutlass_dsl import Boolean, Int32, if_generate

from cutlass.pipeline import (
    Agent,
    CooperativeGroup,
    PipelineOp,
    SyncObject,
    MbarrierArray,
    TmaStoreFence,
    PipelineUserType,
    PipelineState,
    make_pipeline_state,
    pipeline_init_wait,
)

##############################################################################
# Pipeline classes
##############################################################################


@dataclass(frozen=True)
class PipelineAsync:
    """PipelineAsync is a generic pipeline class where both the producer and consumer are
    AsyncThreads. It also serves as a base class for specialized pipeline classes.

    This class implements a producer-consumer pipeline pattern where both sides operate
    asynchronously. The pipeline maintains synchronization state using barrier objects
    to coordinate between producer and consumer threads.

    The pipeline state transitions of one pipeline entry(mbarrier) can be represented as:

    .. table:: Pipeline State Transitions
       :widths: auto

       +-----------+-----------+-----------+-----------+-----------+-----------+
       | Barrier   | State     | p.acquire | p.commit  | c.wait    | c.release |
       +===========+===========+===========+===========+===========+===========+
       | empty_bar | empty     | <Return>  | n/a       | n/a       | -         |
       +-----------+-----------+-----------+-----------+-----------+-----------+
       | empty_bar | wait      | <Block>   | n/a       | n/a       | -> empty  |
       +-----------+-----------+-----------+-----------+-----------+-----------+
       | full_bar  | wait      | n/a       | -> full   | <Block >  | n/a       |
       +-----------+-----------+-----------+-----------+-----------+-----------+
       | full_bar  | full      | n/a       | -         | <Return>  | n/a       |
       +-----------+-----------+-----------+-----------+-----------+-----------+

    Where:

    - p: producer
    - c: consumer
    - <Block>: This action is blocked until transition to a state allow it to proceed by other side
      - e.g. ``p.acquire()`` is blocked until ``empty_bar`` transition to ``empty`` state by ``c.release()``

    .. code-block:: text

        Array of mbarriers as circular buffer:

             Advance Direction
           <-------------------

            Producer   Consumer
                |         ^
                V         |
           +-----------------+
         --|X|X|W|D|D|D|D|R|X|<-.
        /  +-----------------+   \\
        |                        |
        `------------------------'

    Where:

    - X: Empty buffer (initial state)
    - W: Producer writing (producer is waiting for buffer to be empty)
    - D: Data ready (producer has written data to buffer)
    - R: Consumer reading (consumer is consuming data from buffer)

    **Example:**

    .. code-block:: python

        # Create pipeline with 5 stages
        pipeline = PipelineAsync.create(
            num_stages=5,                   # number of pipeline stages
            producer_group=producer_warp,
            consumer_group=consumer_warp
            barrier_storage=smem_ptr,       # smem pointer for array of mbarriers in shared memory
        )

        producer, consumer = pipeline.make_participants()
        # Producer side
        for i in range(num_iterations):
            handle = producer.acquire_and_advance()  # Wait for buffer to be empty & Move index to next stage
            # Write data to pipeline buffer
            handle.commit()   # Signal buffer is full

        # Consumer side
        for i in range(num_iterations):
            handle = consumer.wait_and_advance()     # Wait for buffer to be full & Move index to next stage
            # Read data from pipeline buffer
            handle.release()  # Signal buffer is empty
    """

    sync_object_full: SyncObject
    sync_object_empty: SyncObject
    num_stages: int
    producer_mask: Optional[Int32]
    consumer_mask: Optional[Int32]

    @staticmethod
    def _make_sync_object(
        barrier_storage: cute.Pointer,
        num_stages: int,
        agent: tuple[PipelineOp, CooperativeGroup],
        tx_count: int = 0,
    ) -> SyncObject:
        """
        Returns a SyncObject corresponding to an agent's PipelineOp.
        """
        if agent[0] in [
            PipelineOp.AsyncThread,
            PipelineOp.TmaLoad,
            PipelineOp.TCGen05Mma,
            PipelineOp.Composite,
            PipelineOp.AsyncLoad,
        ]:
            return MbarrierArray(
                barrier_storage=barrier_storage,
                num_stages=num_stages,
                agent=agent,
                tx_count=tx_count,
            )
        elif agent[0] is PipelineOp.TmaStore:
            # Path taken for AsyncTmaStore
            return TmaStoreFence(num_stages=num_stages)
        else:
            assert False, "Error: Invalid PipelineOp specified."

    @staticmethod
    def create(
        *,
        num_stages: int,
        producer_group: CooperativeGroup,
        consumer_group: CooperativeGroup,
        barrier_storage: cute.Pointer = None,
        producer_mask: Int32 = None,
        consumer_mask: Int32 = None,
    ):
        """Creates and initializes a new PipelineAsync instance.

        This helper function computes necessary attributes and returns an instance of PipelineAsync
        with the specified configuration for producer and consumer synchronization.

        :param barrier_storage: Pointer to the shared memory address for this pipeline's mbarriers
        :type barrier_storage: cute.Pointer
        :param num_stages: Number of buffer stages for this pipeline
        :type num_stages: int
        :param producer_group: `CooperativeGroup` for the producer agent
        :type producer_group: CooperativeGroup
        :param consumer_group: `CooperativeGroup` for the consumer agent
        :type consumer_group: CooperativeGroup
        :param producer_mask: Mask for signaling arrives for the producer agent, defaults to ``None``
        :type producer_mask: Int32, optional
        :param consumer_mask: Mask for signaling arrives for the consumer agent, defaults to ``None``
        :type consumer_mask: Int32, optional
        :return: A new PipelineAsync instance
        :rtype: PipelineAsync
        :raises ValueError: If barrier_storage is not a cute.Pointer instance
        """
        if not isinstance(barrier_storage, cute.Pointer):
            raise ValueError(
                f"Expected barrier_storage to be a cute.Pointer, but got {type(barrier_storage)}"
            )

        producer_type = PipelineOp.AsyncThread
        consumer_type = PipelineOp.AsyncThread

        producer = (producer_type, producer_group)
        consumer = (consumer_type, consumer_group)

        sync_object_full = PipelineAsync._make_sync_object(
            barrier_storage.align(min_align=8), num_stages, producer
        )
        sync_object_empty = PipelineAsync._make_sync_object(
            barrier_storage.align(min_align=8) + num_stages, num_stages, consumer
        )

        pipeline_init_wait()

        return PipelineAsync(
            sync_object_full,
            sync_object_empty,
            num_stages,
            producer_mask,
            consumer_mask,
        )

    def producer_acquire(
        self, state: PipelineState, try_acquire_token: Optional[Boolean] = None
    ):
        if_generate(
            try_acquire_token is None or try_acquire_token == 0,
            lambda: self.sync_object_empty.wait(state.index, state.phase),
        )

    def producer_try_acquire(self, state: PipelineState):
        return self.sync_object_empty.try_wait(state.index, state.phase)

    def producer_commit(self, state: PipelineState):
        self.sync_object_full.arrive(state.index, self.producer_mask)

    def consumer_wait(
        self, state: PipelineState, try_wait_token: Optional[Boolean] = None
    ):
        if_generate(
            try_wait_token is None or try_wait_token == 0,
            lambda: self.sync_object_full.wait(state.index, state.phase),
        )

    def consumer_try_wait(self, state: PipelineState):
        return self.sync_object_full.try_wait(state.index, state.phase)

    def consumer_release(self, state: PipelineState):
        self.sync_object_empty.arrive(state.index, self.consumer_mask)

    def producer_get_barrier(self, state: PipelineState) -> cute.Pointer:
        return self.sync_object_full.get_barrier(state.index)

    def producer_tail(self, state: PipelineState):
        """
        Make sure the last used buffer empty signal is visible to producer.
        Producer tail is usually executed by producer before exit, to avoid dangling
        mbarrier arrive signals after kernel exit.

        :param state: The pipeline state that points to next useful buffer
        :type state: PipelineState
        """
        # Assume state contains that next useful buffer
        # So we only need to advance to num_stages - 1 times to last used buffer
        for i in range(self.num_stages - 1):
            state.advance()
        self.producer_acquire(state)

    # Util methods to manage produer and consumer
    def make_producer(self):
        state = make_pipeline_state(PipelineUserType.Producer, self.num_stages)
        return PipelineProducer(self, state, self.sync_object_full.cg)

    def make_consumer(self):
        state = make_pipeline_state(PipelineUserType.Consumer, self.num_stages)
        return PipelineConsumer(self, state, self.sync_object_empty.cg)

    def make_participants(self):
        return self.make_producer(), self.make_consumer()



@dataclass(frozen=True)
class PipelineCpAsync(PipelineAsync):
    """
    PipelineCpAsync is used for CpAsync producers and AsyncThread consumers (e.g. Hopper non-TMA mainloops).
    """

    @staticmethod
    def create(
        barrier_storage: cute.Pointer,
        num_stages: Int32,
        producer_group: CooperativeGroup,
        consumer_group: CooperativeGroup,
        producer_mask: Int32 = None,
        consumer_mask: Int32 = None,
    ):
        """
        This helper function computes any necessary attributes and returns an instance of PipelineAsync.
        :param barrier_storage: Pointer to the smem address for this pipeline's mbarriers
        :type barrier_storage: cute.Pointer
        :param num_stages: Number of buffer stages for this pipeline
        :type num_stages: Int32
        :param producer_group: CooperativeGroup for the producer agent
        :type producer_group: CooperativeGroup
        :param consumer_group: CooperativeGroup for the consumer agent
        :type consumer_group: CooperativeGroup
        :param producer_mask: Mask for signaling arrives for the producer agent
        :type producer_mask: Int32 | None
        :param consumer_mask: Mask for signaling arrives for the consumer agent
        :type consumer_mask: Int32 | None
        """
        producer_type = PipelineOp.AsyncLoad
        consumer_type = PipelineOp.AsyncThread

        producer = (producer_type, producer_group)
        consumer = (consumer_type, consumer_group)

        sync_object_array_full = PipelineCpAsync._make_sync_object(
            barrier_storage.align(min_align=8), num_stages, producer
        )
        sync_object_array_empty = PipelineCpAsync._make_sync_object(
            barrier_storage.align(min_align=8) + num_stages, num_stages, consumer
        )

        pipeline_init_wait()

        return PipelineCpAsync(
            sync_object_array_full,
            sync_object_array_empty,
            num_stages,
            producer_mask,
            consumer_mask,
        )


@dataclass(frozen=True)
class PipelineTmaAsync(PipelineAsync):
    """
    PipelineTmaAsync is used for TMA producers and AsyncThread consumers (e.g. Hopper mainloops).
    """

    is_signalling_thread: Boolean

    @staticmethod
    @cute.jit
    def init_empty_barrier_arrive_signal(cta_layout_vmnk: cute.Layout, tidx: Int32):
        """
        Initialize the empty barrier arrive signal
        This function returns the destination cta rank and a boolean indicating if the signalling thread is the same as the current thread
        """
        # Logic to optimally schedule Empty Arrives
        cluster_shape_vmnk = cta_layout_vmnk.shape

        cta_rank_in_cluster = cute.arch.make_warp_uniform(
            cute.arch.block_idx_in_cluster()
        )

        tidx = tidx % 32
        is_signalling_thread = tidx < cute.size(cluster_shape_vmnk)
        dst_rank = tidx % cute.size(cluster_shape_vmnk)

        dst_cta_coord = cta_layout_vmnk.get_hier_coord(dst_rank)
        cur_cta_coord = cta_layout_vmnk.get_hier_coord(cta_rank_in_cluster)

        is_same_row = (
            dst_cta_coord[0] == cur_cta_coord[0]
            and dst_cta_coord[1] == cur_cta_coord[1]
            and dst_cta_coord[3] == cur_cta_coord[3]
        )
        is_same_col = (
            dst_cta_coord[0] == cur_cta_coord[0]
            and dst_cta_coord[2] == cur_cta_coord[2]
            and dst_cta_coord[3] == cur_cta_coord[3]
        )

        is_same_row_or_col = is_same_row or is_same_col
        is_signalling_thread_final = is_signalling_thread and is_same_row_or_col

        return dst_rank, is_signalling_thread_final

    @staticmethod
    def create(
        *,
        num_stages: int,
        producer_group: CooperativeGroup,
        consumer_group: CooperativeGroup,
        tx_count: int,
        barrier_storage: cute.Pointer = None,
        cta_layout_vmnk: Optional[cute.Layout] = None,
        tidx: Optional[Int32] = None,
    ):
        """
        This helper function computes any necessary attributes and returns an instance of PipelineTmaAsync.
        :param barrier_storage: Pointer to the smem address for this pipeline's mbarriers
        :type barrier_storage: cute.Pointer
        :param num_stages: Number of buffer stages for this pipeline
        :type num_stages: Int32
        :param producer_group: `CooperativeGroup` for the producer agent
        :type producer_group: CooperativeGroup
        :param consumer_group: `CooperativeGroup` for the consumer agent
        :type consumer_group: CooperativeGroup
        :param tx_count: Number of bytes expected to be written to the transaction barrier for one stage
        :type tx_count: int
        :param cta_layout_vmnk: Layout of the cluster shape
        :type cta_layout_vmnk: cute.Layout | None
        :param tidx: thread index to consumer async threads
        :type tidx: Int32 | None
        """
        if not isinstance(barrier_storage, cute.Pointer):
            raise ValueError(
                f"Expected barrier_storage to be a cute.Pointer, but got {type(barrier_storage)}"
            )

        producer_type = PipelineOp.TmaLoad
        consumer_type = PipelineOp.AsyncThread

        producer = (producer_type, producer_group)
        consumer = (consumer_type, consumer_group)

        sync_object_full = PipelineAsync._make_sync_object(
            barrier_storage.align(min_align=8), num_stages, producer, tx_count
        )
        sync_object_empty = PipelineAsync._make_sync_object(
            barrier_storage.align(min_align=8) + num_stages, num_stages, consumer
        )
        if tidx is None:
            tidx, _, _ = cute.arch.thread_idx()
        if cta_layout_vmnk is None:
            cta_layout_vmnk = cute.make_layout((1, 1, 1, 1))
        (
            dst_rank,
            is_signalling_thread,
        ) = PipelineTmaAsync.init_empty_barrier_arrive_signal(cta_layout_vmnk, tidx)
        if cta_layout_vmnk is None or cute.size(cta_layout_vmnk) == 1:
            dst_rank = None
        else:
            dst_rank = dst_rank

        producer_mask = None

        pipeline_init_wait(cta_layout_vmnk)

        return PipelineTmaAsync(
            sync_object_full,
            sync_object_empty,
            num_stages,
            producer_mask,
            dst_rank,
            is_signalling_thread,
        )

    def producer_acquire(
        self, state: PipelineState, try_acquire_token: Optional[Boolean] = None
    ):
        """
        TMA producer commit conditionally waits on buffer empty and sets the transaction barrier.
        """
        if_generate(
            try_acquire_token is None or try_acquire_token == 0,
            lambda: self.sync_object_empty.wait(state.index, state.phase),
        )
        self.sync_object_full.arrive(state.index, self.producer_mask)

    def producer_commit(self, state: PipelineState):
        """
        TMA producer commit is a noop since TMA instruction itself updates the transaction count.
        """
        pass

    def consumer_release(self, state: PipelineState):
        """
        TMA consumer release conditionally signals the empty buffer to the producer.
        """
        if_generate(
            self.is_signalling_thread,
            lambda: self.sync_object_empty.arrive(state.index, self.consumer_mask),
        )


@dataclass(frozen=True)
class PipelineTmaMultiConsumersAsync(PipelineAsync):
    """
    PipelineTmaMultiConsumersAsync is used for TMA producers and UMMA+Async consumers.
    """

    is_leader_cta: bool
    sync_object_empty_umma: SyncObject
    sync_object_empty_async: SyncObject
    cta_group: cute.nvgpu.tcgen05.CtaGroup

    @staticmethod
    def create(
        *,
        num_stages: int,
        producer_group: CooperativeGroup,
        consumer_group_umma: CooperativeGroup,
        consumer_group_async: CooperativeGroup,
        tx_count: int,
        barrier_storage: cute.Pointer = None,
        cta_layout_vmnk: Optional[cute.Layout] = None,
    ):
        """
        This helper function computes any necessary attributes and returns an instance of PipelineTmaMultiConsumersAsync.
        :param barrier_storage: Pointer to the smem address for this pipeline's mbarriers
        :type barrier_storage: cute.Pointer
        :param num_stages: Number of buffer stages for this pipeline
        :type num_stages: Int32
        :param producer_group: `CooperativeGroup` for the producer agent
        :type producer_group: CooperativeGroup
        :param consumer_group_umma: `CooperativeGroup` for the UMMA consumer agent
        :type consumer_group_umma: CooperativeGroup
        :param consumer_group_async: `CooperativeGroup` for the AsyncThread consumer agent
        :type consumer_group_async: CooperativeGroup
        :param tx_count: Number of bytes expected to be written to the transaction barrier for one stage
        :type tx_count: int
        :param cta_layout_vmnk: Layout of the cluster shape
        :type cta_layout_vmnk: cute.Layout | None
        """
        if not isinstance(barrier_storage, cute.Pointer):
            raise ValueError(
                f"Expected barrier_storage to be a cute.Pointer, but got {type(barrier_storage)}"
            )

        producer_type = PipelineOp.TmaLoad
        consumer_type = PipelineOp.Composite
        consumer_type_umma = PipelineOp.TCGen05Mma
        consumer_type_async = PipelineOp.AsyncThread

        if consumer_group_umma.agent != consumer_group_async.agent:
            raise ValueError(
                "UMMA and AsyncThread consumer groups must be the same agent"
            )

        if cta_layout_vmnk is not None and cute.size(cta_layout_vmnk) != 1:
            raise ValueError(
                f"PipelineTmaMultiConsumersAsync is not verified for cta_layout_vmnk != 1, cta_layout_vmnk:{cta_layout_vmnk}"
            )

        consumer_group = CooperativeGroup(
            consumer_group_umma.agent,
            consumer_group_umma.size + consumer_group_async.size,
        )

        producer = (producer_type, producer_group)
        consumer = (consumer_type, consumer_group)

        sync_object_full = PipelineAsync._make_sync_object(
            barrier_storage.align(min_align=8), num_stages, producer, tx_count
        )
        sync_object_empty = PipelineAsync._make_sync_object(
            barrier_storage.align(min_align=8) + num_stages, num_stages, consumer
        )
        sync_object_empty_umma = sync_object_empty.recast_to_new_op_type(
            consumer_type_umma
        )
        sync_object_empty_async = sync_object_empty.recast_to_new_op_type(
            consumer_type_async
        )

        # No mcast mask if not using clusters
        producer_mask = None
        consumer_mask = None
        # All threadblocks are leaders if not using clusters
        is_leader_cta = True
        cta_group = (
            cute.nvgpu.tcgen05.CtaGroup.ONE
            if cta_layout_vmnk is None or cute.size(cta_layout_vmnk, mode=[0]) == 1
            else cute.nvgpu.tcgen05.CtaGroup.TWO
        )

        pipeline_init_wait(cta_layout_vmnk)

        return PipelineTmaMultiConsumersAsync(
            sync_object_full,
            sync_object_empty,
            num_stages,
            producer_mask,
            consumer_mask,
            is_leader_cta,
            sync_object_empty_umma,
            sync_object_empty_async,
            cta_group,
        )

    def producer_acquire(
        self, state: PipelineState, try_acquire_token: Optional[Boolean] = None
    ):
        """
        TMA producer acquire waits on buffer empty and sets the transaction barrier for leader threadblocks.
        """
        if_generate(
            try_acquire_token is None or try_acquire_token == 0,
            lambda: self.sync_object_empty.wait(state.index, state.phase),
        )
        if_generate(
            self.is_leader_cta,
            lambda: self.sync_object_full.arrive(state.index, self.producer_mask),
        )

    def producer_commit(self, state: PipelineState):
        """
        TMA producer commit is a noop since TMA instruction itself updates the transaction count.
        """
        pass

    def consumer_release(self, state: PipelineState, op_type: PipelineOp):
        if op_type == PipelineOp.TCGen05Mma:
            self.sync_object_empty_umma.arrive(
                state.index, self.consumer_mask, self.cta_group
            )
        elif op_type == PipelineOp.AsyncThread:
            self.sync_object_empty_async.arrive(state.index, self.consumer_mask)
        else:
            raise ValueError(f"Invalid PipelineOp specified. op_type:{op_type}")


@dataclass(frozen=True)
class PipelineTmaStore(PipelineAsync):
    """
    PipelineTmaStore is used for synchronizing TMA stores in the epilogue. It does not use mbarriers.
    """

    @staticmethod
    def create(
        *,
        num_stages: int,
        producer_group: CooperativeGroup,
    ):
        """
        This helper function computes any necessary attributes and returns an instance of PipelineTmaStore.
        :param num_stages: Number of buffer stages for this pipeline
        :type num_stages: Int32
        :param producer_group: `CooperativeGroup` for the producer agent
        :type producer_group: CooperativeGroup
        """

        producer_type = PipelineOp.TmaStore

        producer = (producer_type, producer_group)

        sync_object_full = PipelineAsync._make_sync_object(None, num_stages, producer)

        return PipelineTmaStore(sync_object_full, None, num_stages, None, None)

    def producer_acquire(self):
        self.sync_object_full.wait()

    def producer_commit(self):
        self.sync_object_full.arrive()

    def consumer_wait(self):
        assert False, "Error: PipelineTmaStore does not have a consumer agent."

    def consumer_release(self):
        assert False, "Error: PipelineTmaStore does not have a consumer agent."

    def producer_tail(self):
        self.sync_object_full.tail()


#################################################################
# Utilities to help user of pipeline to simplify the workflow
#################################################################


class ImmutableResourceHandle:
    __origin: PipelineAsync
    __immutable_state: PipelineState

    def __init__(self, origin: PipelineAsync, immutable_state: PipelineState):
        self.__origin = origin
        self.__immutable_state = immutable_state

    @property
    def index(self):
        """Get the index of the current pipeline stage."""
        return self.__immutable_state.index

    @property
    def count(self):
        """Get the count of how many handles this producer has committed.
        This is useful for tracking the number of blocks that have been loaded from gmem.
        """
        return self.__immutable_state.count

    def get_origin(self):
        """Get the original pipeline this resource handle belongs to."""
        return self.__origin

    def __extract_mlir_values__(self):
        """Extract MLIR values from the current state.

        :return: List of MLIR values representing the current state
        :rtype: list
        """
        # TODO: need to handle pipeline as well
        return self.__immutable_state.__extract_mlir_values__()

    def __new_from_mlir_values__(self, values):
        """Create a new Producer instance from MLIR values.

        :param values: MLIR values to initialize the state
        :type values: Any
        :return: New Producer instance with state initialized from values
        :rtype: Producer
        """
        return self.__class__(
            self.__origin, self.__immutable_state.__new_from_mlir_values__(values)
        )

class PipelineProducer:
    """A class representing a producer in an asynchronous pipeline.

    The Producer class manages the producer side of an asynchronous pipeline, handling
    synchronization and state management for producing data. It provides methods for
    acquiring, committing, and advancing through pipeline stages.

    :ivar __pipeline: The asynchronous pipeline this producer belongs to
    :type __pipeline: PipelineAsync
    :ivar __state: The current state of the producer in the pipeline
    :type __state: PipelineState
    :ivar __group: The cooperative group this producer operates in
    :type __group: CooperativeGroup

    **Examples:**

        .. code-block:: python

            pipeline = PipelineAsync.create(...)
            producer = pipeline.create_producer(producer_group, stages)
            for i in range(iterations):
                handle = producer.acquire_and_advance()  # Wait for buffer to be empty
                # Produce data
                producer.commit(handle)   # Signal data is ready
                # An alternative way to do this is:
                # handle.commit()   # Signal data is ready
    """

    __pipeline: PipelineAsync
    __state: PipelineState
    __group: CooperativeGroup

    class ImmutableResourceHandle(ImmutableResourceHandle):
        @property
        def barrier(self):
            """Get the barrier pointer for the current pipeline stage.

            :return: Pointer to the barrier for the current stage
            :rtype: cute.Pointer
            """
            return self.get_origin().producer_get_barrier(
                self._ImmutableResourceHandle__immutable_state
            )

        def commit(self):
            """Signal that data production is complete for the current stage.
            This allows consumers to start processing the data.
            """
            self.get_origin().producer_commit(
                self._ImmutableResourceHandle__immutable_state
            )

    def __init__(self, pipeline, state, group: CooperativeGroup):
        """Initialize a new Producer instance.

        :param pipeline: The pipeline this producer belongs to
        :type pipeline: PipelineAsync
        :param state: Initial pipeline state
        :type state: PipelineState
        :param group: The cooperative group for synchronization
        :type group: CooperativeGroup
        """
        self.__pipeline = pipeline
        self.__state = state
        self.__group = group

    def acquire(
        self,
        try_acquire_token: Optional[Boolean] = None,
    ) -> ImmutableResourceHandle:
        """Wait for the current buffer to be empty before producing data.
        This is a blocking operation.

        :param try_acquire_token: Optional token to try to acquire the buffer
        :type try_acquire_token: Optional[Boolean]
        :return: A handle to the producer for committing the data
        :rtype: ImmutableResourceHandle
        """
        self.__pipeline.producer_acquire(self.__state, try_acquire_token)
        handle = PipelineProducer.ImmutableResourceHandle(
            self.__pipeline, self.__state.clone()
        )
        return handle

    def advance(self):
        """Move to the next pipeline stage."""
        self.__state.advance()

    def acquire_and_advance(
        self, try_acquire_token: Optional[Boolean] = None
    ) -> ImmutableResourceHandle:
        """Wait for the current buffer to be empty before producing data.
        Then advance to the next stage.
        This is a blocking operation.

        :param try_acquire_token: Optional token to try to acquire the buffer
        :type try_acquire_token: Optional[Boolean]
        :return: A handle to the producer for committing the data
        :rtype: ImmutableResourceHandle
        """
        handle = self.acquire(try_acquire_token)
        self.advance()
        return handle

    def try_acquire(self) -> Boolean:
        """Try to acquire the current buffer without blocking.

        :return: True if acquisition was successful, False otherwise
        :rtype: Boolean
        """
        return self.__pipeline.producer_try_acquire(self.__state)

    def commit(self, handle: Optional[ImmutableResourceHandle] = None):
        """Signal that data production is complete for the current stage.
        This allows consumers to start processing the data.
        """
        if handle is not None:
            assert (
                handle.get_origin() is self
            ), "ResourceHandle does not belong to this PipelineProducer instance"
            handle.commit()
        else:
            self.__pipeline.producer_commit(self.__state)

    def tail(self):
        """Ensure all used buffers are properly synchronized before producer exit.
        This should be called before the producer finishes to avoid dangling signals.
        """
        self.__pipeline.producer_tail(self.__state)

    def __extract_mlir_values__(self):
        """Extract MLIR values from the current state.

        :return: List of MLIR values representing the current state
        :rtype: list
        """
        # TODO: need to handle pipeline as well
        return self.__state.__extract_mlir_values__()

    def __new_from_mlir_values__(self, values):
        """Create a new Producer instance from MLIR values.

        :param values: MLIR values to initialize the state
        :type values: Any
        :return: New Producer instance with state initialized from values
        :rtype: Producer
        """
        return PipelineProducer(
            self.__pipeline, self.__state.__new_from_mlir_values__(values), self.__group
        )

class PipelineConsumer:
    """A class representing a consumer in an asynchronous pipeline.

    The Consumer class manages the consumer side of an asynchronous pipeline, handling
    synchronization and state management for consuming data. It provides methods for
    waiting, releasing, and advancing through pipeline stages.

    :ivar __pipeline: The asynchronous pipeline this consumer belongs to
    :type __pipeline: PipelineAsync
    :ivar __state: The current state of the consumer in the pipeline
    :type __state: PipelineState
    :ivar __group: The cooperative group this consumer operates in
    :type __group: CooperativeGroup

    **Examples:**
        .. code-block:: python

            pipeline = PipelineAsync.create(...)
            consumer = pipeline.create_consumer(consumer_group, stages)
            for i in range(iterations):
                handle = consumer.wait_and_advance()     # Wait for data to be ready
                # Consume data
                consumer.release(handle)  # Signal buffer is empty
                # An alternative way to do this is:
                # handle.release()  # Signal buffer is empty
    """

    __pipeline: PipelineAsync
    __state: PipelineState
    __group: CooperativeGroup

    class ImmutableResourceHandle(ImmutableResourceHandle):
        def release(self):
            """Signal that data production is complete for the current stage.
            This allows consumers to start processing the data.
            """
            self.get_origin().consumer_release(
                self._ImmutableResourceHandle__immutable_state
            )

    def __init__(self, pipeline, state: PipelineState, group: CooperativeGroup):
        """Initialize a new Consumer instance.

        :param pipeline: The pipeline this consumer belongs to
        :type pipeline: PipelineAsync
        :param state: Initial pipeline state
        :type state: PipelineState
        :param group: The cooperative group for synchronization
        :type group: CooperativeGroup
        """
        self.__pipeline = pipeline
        self.__group = group
        self.__state = state

    def wait(self, try_wait_token: Optional[Boolean] = None) -> ImmutableResourceHandle:
        """Wait for data to be ready in the current buffer.
        This is a blocking operation.

        :param try_wait_token: Optional token to try to wait for the buffer
        :type try_wait_token: Optional[Boolean]
        :return: A handle to the consumer for releasing the data
        :rtype: PipelineConsumerHandle
        """
        self.__pipeline.consumer_wait(self.__state, try_wait_token)
        handle = PipelineConsumer.ImmutableResourceHandle(
            self.__pipeline, self.__state.clone()
        )
        return handle

    def advance(self):
        """Move to the next pipeline stage."""
        self.__state.advance()

    def wait_and_advance(
        self, try_wait_token: Optional[Boolean] = None
    ) -> ImmutableResourceHandle:
        """Wait for data to be ready in the current buffer.
        Then advance to the next stage.
        This is a blocking operation.

        :param try_wait_token: Optional token to try to wait for the buffer
        :type try_wait_token: Optional[Boolean]
        :return: A handle to the consumer for releasing the data
        :rtype: PipelineConsumerHandle
        """
        handle = self.wait(try_wait_token)
        self.advance()
        return handle

    def try_wait(self) -> Boolean:
        """Try to check if data is ready without blocking.

        :return: True if data is ready, False otherwise
        :rtype: Boolean
        """
        return self.__pipeline.consumer_try_wait(self.__state)

    def release(self, handle: Optional[ImmutableResourceHandle] = None):
        """Signal that data consumption is complete for the current stage.
        This allows producers to start producing new data.
        """
        if handle is not None:
            assert (
                handle.get_origin() is self
            ), "ResourceHandle does not belong to this PipelineConsumer instance"
            handle.release()
        else:
            self.__pipeline.consumer_release(self.__state)

    def __extract_mlir_values__(self):
        """Extract MLIR values from the current state.

        :return: List of MLIR values representing the current state
        :rtype: list
        """
        return self.__state.__extract_mlir_values__()

    def __new_from_mlir_values__(self, values):
        """Create a new Consumer instance from MLIR values.

        :param values: MLIR values to initialize the state
        :type values: Any
        :return: New Consumer instance with state initialized from values
        :rtype: Consumer
        """
        # TODO: need to call pipeline.__new_from_mlir_values__ recursively
        return PipelineConsumer(
            self.__pipeline, self.__state.__new_from_mlir_values__(values), self.__group
        )
