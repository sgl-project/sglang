# Copyright (c) 2025, Tri Dao.

# import math
from typing import Optional
from dataclasses import dataclass

import cutlass
import cutlass.cute as cute
from cutlass.cutlass_dsl import Boolean, Int32, if_generate
from cutlass.pipeline import PipelineAsync, PipelineState, CooperativeGroup, pipeline_init_wait
from cutlass.pipeline import PipelineUserType, PipelineOp


class PipelineStateSimple:
    """
    Pipeline state contains an index and phase bit corresponding to the current position in the circular buffer.
    Use a single Int32 to store both the index and phase bit, then we use divmod to get the
    index and phase. If stages is a power of 2, divmod turns into bit twiddling.
    """

    def __init__(self, stages: int, phase_index: Int32):
        # assert stages < 2**16
        # self._log_stages = int(math.log2(stages))
        # assert 1 << self._log_stages == stages, "Number of stages must be a power of 2."
        self._stages = stages
        self._phase_index = phase_index

    def clone(self) -> "PipelineStateSimple":
        return PipelineStateSimple(self.stages, self._phase_index)

    @property
    def stages(self) -> int:
        # return 1 << self._log_stages
        return self._stages

    @property
    def index(self) -> Int32:
        # return self._phase_index & 0xFFFF
        # return self._phase_index & ((1 << self._log_stages) - 1)
        return self._phase_index % self._stages

    @property
    def phase(self) -> Int32:
        # return self._phase_index >> 16
        # PTX docs say that the phase parity needs to be 0 or 1, so by right we need to
        # take modulo 2. But in practice just passing the phase in without modulo works fine.
        # return (self._phase_index >> self._log_stages) % 2
        # return self._phase_index >> self._log_stages
        return self._phase_index // self._stages

    def advance(self):
        self._phase_index += 1

        # def then_body(phase_index):
        #     # XOR the phase bit and set the index to 0
        #     return (phase_index & 0xFFFF0000) ^ (1 << 16)

        # def else_body(phase_index):
        #     return phase_index

        # self._phase_index = if_generate(
        #     (self._phase_index & 0xFFFF) == self.stages,
        #     then_body,
        #     else_body,
        #     [self._phase_index],
        #     [Int32],
        # )

    def __extract_mlir_values__(self):
        phase_index = self._phase_index
        return [phase_index.ir_value()]

    def __new_from_mlir_values__(self, values):
        return PipelineStateSimple(self.stages, Int32(values[0]))


def make_pipeline_state(type: PipelineUserType, stages: int):
    """
    Creates a pipeline state. Producers are assumed to start with an empty buffer and have a flipped phase bit of 1.
    """
    if type is PipelineUserType.Producer:
        # return PipelineStateSimple(stages, Int32(1 << 16))
        return PipelineStateSimple(stages, Int32(stages))
    elif type is PipelineUserType.Consumer:
        return PipelineStateSimple(stages, Int32(0))
    else:
        assert False, "Error: invalid PipelineUserType specified for make_pipeline_state."


@dataclass(frozen=True)
class PipelineTmaAsyncNoCluster(PipelineAsync):
    """
    If size(ClusterShape) == 1, PipelineTmaAsync has all threads
    signaling the barrier during consumer_release. This causes a perf regression in FA3
    forward pass (especially hdim 128 causal). We instead implement a version of
    PipelineTmaAsync where only 1 out of 128 threads signals the barrier.

    Assumptions:
    (1) num_consumers % NumThreadsPerWarpGroup == 0
    (2) all 128 threads in the warp group are sync'ed right before calling consumer_release
    """

    @staticmethod
    def create(
        barrier_storage: cute.Pointer,
        num_stages: Int32,
        producer_group: CooperativeGroup,
        consumer_group: CooperativeGroup,
        tx_count: int,
        init_wait: cutlass.Constexpr[bool] = True,
    ):
        """
        This helper function computes any necessary attributes and returns an instance of PipelineTmaAsync.
        :param barrier_storage: Pointer to the smem address for this pipeline's mbarriers
        :type barrier_storage: cute.Pointer
        :param num_stages: Number of buffer stages for this pipeline
        :type num_stages: Int32
        :param producer_group: CooperativeGroup for the producer agent
        :type producer_group: CooperativeGroup
        :param consumer_group: CooperativeGroup for the consumer agent
        :type consumer_group: CooperativeGroup
        :param tx_count: Number of bytes expected to be written to the transaction barrier for one stage
        :type tx_count: int
        """
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
        dst_rank = None
        producer_mask = None
        if cutlass.const_expr(init_wait):
            pipeline_init_wait()
        return PipelineTmaAsyncNoCluster(
            sync_object_full,
            sync_object_empty,
            num_stages,
            producer_mask,
            dst_rank,
        )

    def producer_acquire(self, state: PipelineState, try_acquire_token: Optional[Boolean] = None):
        """
        TMA producer commit conditionally waits on buffer empty and sets the transaction barrier for leader threadblocks.
        """
        if_generate(
            try_acquire_token is None or try_acquire_token == 0,
            lambda: self.sync_object_empty.wait(state.index, state.phase),
        )
        self.sync_object_full.arrive(state.index, self.producer_mask)

    def producer_commit(self, state: PipelineState):
        """
        TMA producer commit is a NOP. The transaction barrier signals the commit upon completion of the TMA.
        """
        pass

    def consumer_release(self, state: PipelineState):
        """
        TMA consumer release conditionally signals the empty buffer to the producer.
        """
        # Only 1 thread per warp group signals the empty buffer.
        if_generate(
            cute.arch.thread_idx()[0] % 128 == 0,
            lambda: self.sync_object_empty.arrive(state.index, self.consumer_mask),
        )
