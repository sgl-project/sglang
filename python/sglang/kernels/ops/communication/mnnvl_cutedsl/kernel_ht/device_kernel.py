# Copyright (c) 2026 by FlashInfer team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Persistent BF16 MoE finalize, TP reduction, and RMSNorm for SM100."""

from __future__ import annotations

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline
import cutlass.utils as utils
from cutlass import BFloat16, Float32, Int32, Int64, Uint32

from ..cute_dsl_primitives import (
    VEC_BF16,
    WARP_SIZE,
    bf16x8_to_packed_u32x4,
    cpasync_bulk_g2s,
    fence_proxy_async_shared_cta,
    fragment_has_negative_zero,
    ldmc_bf16x8,
    load_global_bf16_as_f32,
    load_global_u32x4_address,
    load_shared_u32x4,
    packed_negative_zero_bf16x8,
    packed_u32x4_to_bf16x8,
    remote_release_add1_u32,
    sanitize_negative_zero_u32x4,
    stmc_bf16x8,
    store_global_u32x4,
    store_shared_u32x4,
)

SMEM_ALIGNMENT = 1024


class _MoeFinalizeAllReduceRMSNormHTDeviceKernel:
    def __init__(
        self,
        *,
        hidden: int,
        top_k: int,
        tp: int,
        rank: int,
        active_ctas: int,
        stages: int,
        consumer_threads: int,
        vectors_per_thread: int,
        reduction_warps: int,
        reduction_cta_groups: int | None,
        rms_token_groups: int,
        rms_pipeline_stages: int,
        rms_shard_major: bool,
        rms_epsilon: float,
        routed_scaling_factor: float,
        weight_bias: float,
        include_shared_expert: bool,
        add_residual: bool,
        write_residual_output: bool,
        enable_pdl: bool,
    ) -> None:
        if tp not in (2, 4, 8, 16):
            raise ValueError("tp must be 2, 4, 8, or 16")
        if rank < 0 or rank >= tp:
            raise ValueError("rank must be in [0, tp)")
        if hidden <= 0 or hidden % VEC_BF16:
            raise ValueError("hidden must be a positive multiple of 8")
        if top_k < 0:
            raise ValueError("top_k must be nonnegative")
        if active_ctas <= 0 or active_ctas % tp:
            raise ValueError("active_ctas must be positive and divisible by tp")
        if stages < 2:
            raise ValueError("stages must be at least 2")
        if consumer_threads <= 0 or consumer_threads % WARP_SIZE:
            raise ValueError("consumer_threads must be a positive warp multiple")
        if vectors_per_thread <= 0:
            raise ValueError("vectors_per_thread must be positive")
        if reduction_warps not in (1, 2, 4, 8):
            raise ValueError("reduction_warps must be 1, 2, 4, or 8")
        if rms_token_groups not in (1, 2, 4):
            raise ValueError("rms_token_groups must be 1, 2, or 4")
        if consumer_threads % rms_token_groups:
            raise ValueError("consumer threads must divide across RMS token groups")
        if rms_pipeline_stages not in (1, 2, 3):
            raise ValueError("rms_pipeline_stages must be 1, 2, or 3")
        block_threads = consumer_threads + (2 + reduction_warps) * WARP_SIZE
        if block_threads > 1024:
            raise ValueError("warp roles exceed the CUDA block limit")
        shard_elements = consumer_threads * VEC_BF16 * vectors_per_thread
        if hidden <= 0 or hidden % shard_elements:
            raise ValueError(f"hidden must be divisible by {shard_elements}")
        cta_groups = active_ctas // tp
        if reduction_cta_groups is None:
            reduction_cta_groups = active_ctas // tp
        if reduction_cta_groups <= 0 or reduction_cta_groups * tp > active_ctas:
            raise ValueError("reduction CTA groups and shards must fit the grid")
        contributions = top_k + int(include_shared_expert)
        if contributions <= 0:
            raise ValueError("at least one local contribution is required")
        self.hidden = hidden
        self.top_k = top_k
        self.tp = tp
        self.rank = rank
        self.active_ctas = active_ctas
        self.stages = stages
        self.vectors_per_thread = vectors_per_thread
        self.consumer_threads = consumer_threads
        self.reduction_warps = reduction_warps
        self.reduction_cta_groups = reduction_cta_groups
        self.reduction_ctas = reduction_cta_groups * tp
        self.rms_token_groups = rms_token_groups
        self.rms_pipeline_stages = rms_pipeline_stages
        self.rms_shard_major = rms_shard_major
        self.rms_epsilon = rms_epsilon
        self.routed_scaling_factor = routed_scaling_factor
        self.weight_bias = weight_bias
        self.include_shared_expert = include_shared_expert
        self.add_residual = add_residual
        self.write_residual_output = write_residual_output
        self.enable_pdl = enable_pdl
        self.metadata_chunks = (top_k + WARP_SIZE - 1) // WARP_SIZE
        self.metadata_slots = max(top_k, 1)
        self.consumer_warps = consumer_threads // WARP_SIZE
        self.rms_threads_per_token = consumer_threads // rms_token_groups
        self.rms_warps_per_token = self.rms_threads_per_token // WARP_SIZE
        self.rms_stage_slots = rms_token_groups * rms_pipeline_stages
        self.rms_warp_sum_slots = self.rms_stage_slots * self.rms_warps_per_token
        self.publisher_warp = 1 + self.consumer_warps
        self.reduction_warp_begin = self.publisher_warp + 1
        self.reduction_threads = reduction_warps * WARP_SIZE
        self.block_threads = block_threads
        self.shard_elements = shard_elements
        self.shard_bytes = shard_elements * 2
        self.hidden_shards = hidden // shard_elements
        self.contributions = contributions
        if (
            rms_pipeline_stages > 1
            and self.rms_stage_slots * hidden > self.shard_elements * stages
        ):
            raise ValueError("finalize stage storage cannot hold the RMS pipeline")
        self.cta_groups = cta_groups
        self.packs_per_token = hidden // VEC_BF16
        if self.packs_per_token % tp:
            raise ValueError("hidden vector count must be divisible by tp")
        if self.packs_per_token % consumer_threads:
            raise ValueError("token vectors must divide evenly across consumers")
        self.clear_vectors_per_thread = self.packs_per_token // consumer_threads
        self.copy_threads = self.rms_threads_per_token
        if self.packs_per_token % self.copy_threads:
            raise ValueError("token vectors must divide evenly across finalize threads")
        self.rms_vectors_per_thread = self.packs_per_token // self.copy_threads
        self.packs_per_reduction_shard = self.packs_per_token // tp
        if rms_shard_major:
            if tp < self.rms_warps_per_token or tp % self.rms_warps_per_token:
                raise ValueError(
                    "shard-major RMS requires an integer number of reduction "
                    "shards per RMS warp"
                )
            self.reduction_shards_per_rms_warp = tp // self.rms_warps_per_token
            if (
                self.rms_vectors_per_thread * WARP_SIZE
                != self.packs_per_reduction_shard * self.reduction_shards_per_rms_warp
            ):
                raise ValueError(
                    "shard-major RMS warp coverage must match its reduction shards"
                )
        else:
            self.reduction_shards_per_rms_warp = 0
        if self.packs_per_reduction_shard % self.reduction_threads:
            raise ValueError("the reduction shard must divide evenly across threads")
        self.reduction_vectors_per_thread = (
            self.packs_per_reduction_shard // self.reduction_threads
        )

    @cute.jit
    def _rms_arrive_and_wait(self, rms_group: Int32) -> None:
        barrier_0 = pipeline.NamedBarrier(
            barrier_id=2, num_threads=self.rms_threads_per_token
        )
        if cutlass.const_expr(self.rms_token_groups > 1):
            barrier_1 = pipeline.NamedBarrier(
                barrier_id=3, num_threads=self.rms_threads_per_token
            )
            if cutlass.const_expr(self.rms_token_groups == 4):
                barrier_2 = pipeline.NamedBarrier(
                    barrier_id=4, num_threads=self.rms_threads_per_token
                )
                barrier_3 = pipeline.NamedBarrier(
                    barrier_id=5, num_threads=self.rms_threads_per_token
                )
            if rms_group == 0:
                barrier_0.arrive_and_wait()
            elif cutlass.const_expr(self.rms_token_groups == 2):  # noqa: SIM114
                barrier_1.arrive_and_wait()
            elif rms_group == 1:
                barrier_1.arrive_and_wait()
            elif rms_group == 2:
                barrier_2.arrive_and_wait()
            else:
                barrier_3.arrive_and_wait()
        else:
            barrier_0.arrive_and_wait()

    @cute.jit
    def __call__(
        self,
        routed_output: cute.Tensor,
        expert_weights: cute.Tensor,
        permuted_indices: cute.Tensor,
        shared_output: cute.Tensor,
        residual_source: cute.Tensor,
        gamma: cute.Tensor,
        local_contributions: cute.Tensor,
        prenorm_mailbox: cute.Tensor,
        residual_output: cute.Tensor,
        norm_output: cute.Tensor,
        ready_counter_peer_addresses: cute.Tensor,
        ready_counters: cute.Tensor,
        processed_counters: cute.Tensor,
        local_contributions_multicast_address: Int64,
        prenorm_mailbox_multicast_address: Int64,
        m: Int32,
        stream: cuda.CUstream,
    ) -> None:
        smem_layout = cute.make_layout(
            (self.shard_elements * self.stages,), stride=(1,)
        )

        @cute.struct
        class SharedStorage:
            barriers: cute.struct.MemRange[Int64, 2 * self.stages]
            stage_probs: cute.struct.MemRange[Float32, self.stages]
            cached_rows: cute.struct.MemRange[Int32, self.metadata_slots]
            cached_probs: cute.struct.MemRange[Float32, self.metadata_slots]
            consumer_progress: cute.struct.MemRange[Int32, self.consumer_warps]
            norm_warp_sums: cute.struct.MemRange[Float32, self.rms_warp_sum_slots]
            norm_inv_rms: cute.struct.MemRange[Float32, self.rms_stage_slots]
            rows: cute.struct.Align[
                cute.struct.MemRange[BFloat16, cute.cosize(smem_layout)],
                SMEM_ALIGNMENT,
            ]

        self.shared_storage: type[cute.struct.Struct] = SharedStorage
        self.kernel(
            routed_output,
            shared_output,
            residual_source,
            gamma,
            expert_weights,
            permuted_indices,
            local_contributions,
            prenorm_mailbox,
            residual_output,
            norm_output,
            ready_counter_peer_addresses,
            ready_counters,
            processed_counters,
            local_contributions_multicast_address,
            prenorm_mailbox_multicast_address,
            m,
            smem_layout,
        ).launch(
            grid=(self.active_ctas, 1, 1),
            block=(self.block_threads, 1, 1),
            min_blocks_per_mp=1,
            use_pdl=self.enable_pdl,
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        routed_source: cute.Tensor,
        shared_source: cute.Tensor,
        residual_source: cute.Tensor,
        gamma: cute.Tensor,
        expert_weights: cute.Tensor,
        permuted_indices: cute.Tensor,
        local_contributions: cute.Tensor,
        prenorm_mailbox: cute.Tensor,
        residual_output: cute.Tensor,
        norm_output: cute.Tensor,
        ready_counter_peer_addresses: cute.Tensor,
        ready_counters: cute.Tensor,
        processed_counters: cute.Tensor,
        local_contributions_multicast_address: Int64,
        prenorm_mailbox_multicast_address: Int64,
        m: Int32,
        smem_layout: cute.Layout,
    ) -> None:
        block = cute.arch.block_idx()[0]
        tidx = cute.arch.thread_idx()[0]
        warp = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        lane = cute.arch.lane_idx()
        cta_group = block // self.tp
        cta_slot = block % self.tp
        wave = Int64(cta_group)
        token = wave * self.tp + cta_slot
        smem = utils.SmemAllocator()
        storage = smem.allocate(self.shared_storage)
        rows = storage.rows.get_tensor(smem_layout)
        barrier_storage = storage.barriers.data_ptr()
        stage_probs = storage.stage_probs.data_ptr()
        cached_rows = storage.cached_rows.data_ptr()
        cached_probs = storage.cached_probs.data_ptr()
        consumer_progress = storage.consumer_progress.data_ptr()
        norm_warp_sums = storage.norm_warp_sums.data_ptr()
        norm_inv_rms = storage.norm_inv_rms.data_ptr()
        if tidx < self.consumer_warps:
            cute.arch.store((consumer_progress + tidx).llvm_ptr, Int32(0))
        cute.arch.sync_threads()
        if cutlass.const_expr(self.enable_pdl):
            cute.arch.griddepcontrol_wait()
        load_pipeline = pipeline.PipelineTmaAsync.create(
            barrier_storage=barrier_storage,
            num_stages=self.stages,
            producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, 1),
            consumer_group=pipeline.CooperativeGroup(
                pipeline.Agent.Thread, self.consumer_warps
            ),
            tx_count=self.shard_bytes,
        )
        if warp == 0:
            producer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer, self.stages
            )
            peek_empty = cutlass.Boolean(1)
            if token < Int64(m):
                peek_empty = load_pipeline.producer_try_acquire(producer_state)
            while token < Int64(m):
                for metadata_chunk in cutlass.range_constexpr(self.metadata_chunks):
                    metadata_slot = metadata_chunk * WARP_SIZE + lane
                    if metadata_slot < self.top_k:
                        item = Int64(token) * self.top_k + metadata_slot
                        row = cute.arch.load(
                            (permuted_indices.iterator + item).llvm_ptr, Int32
                        )
                        prob = load_global_bf16_as_f32(
                            Int64((expert_weights.iterator + item).toint())
                        )
                        if cutlass.const_expr(self.routed_scaling_factor != 1.0):
                            prob = prob * Float32(self.routed_scaling_factor)
                        if row == Int32(-1):
                            row = Int32(0)
                            prob = Float32(0.0)
                        cute.arch.store((cached_rows + metadata_slot).llvm_ptr, row)
                        cute.arch.store((cached_probs + metadata_slot).llvm_ptr, prob)
                cute.arch.sync_warp()
                for shard in cutlass.range_constexpr(self.hidden_shards):
                    for contribution in cutlass.range_constexpr(self.contributions):
                        load_pipeline.producer_acquire(producer_state, peek_empty)
                        if lane == 0:
                            if cutlass.const_expr(contribution < self.top_k):
                                prob = cute.arch.load(
                                    (cached_probs + contribution).llvm_ptr,
                                    Float32,
                                )
                                row = cute.arch.load(
                                    (cached_rows + contribution).llvm_ptr,
                                    Int32,
                                )
                                source_element = (
                                    Int64(row) * self.hidden
                                    + shard * self.shard_elements
                                )
                                source = routed_source.iterator + source_element
                            else:
                                prob = Float32(1.0)
                                source_element = (
                                    Int64(token) * self.hidden
                                    + shard * self.shard_elements
                                )
                                source = shared_source.iterator + source_element
                            cute.arch.store(
                                (stage_probs + producer_state.index).llvm_ptr,
                                prob,
                            )
                            fence_proxy_async_shared_cta()
                            cpasync_bulk_g2s(
                                source,
                                rows.iterator
                                + producer_state.index * self.shard_elements,
                                load_pipeline.producer_get_barrier(producer_state),
                                Int32(self.shard_bytes),
                            )
                        producer_state.advance()
                        peek_empty = load_pipeline.producer_try_acquire(producer_state)
                wave += self.cta_groups
                token = wave * self.tp + cta_slot
            load_pipeline.producer_tail(producer_state)
        elif warp > 0 and warp <= self.consumer_warps:
            finalize_join = pipeline.NamedBarrier(
                barrier_id=6, num_threads=self.consumer_threads
            )
            consumer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, self.stages
            )
            consumer_tid = tidx - WARP_SIZE
            consumer_wave = Int64(cta_group)
            token = consumer_wave * self.tp + cta_slot
            consumer_token_progress = Int32(0)
            while token < Int64(m):
                for shard in cutlass.range_constexpr(self.hidden_shards):
                    if cutlass.const_expr(self.top_k == 0):
                        load_pipeline.consumer_wait(consumer_state)
                        for trip in cutlass.range_constexpr(self.vectors_per_thread):
                            output_element = (
                                Int64(token) * self.hidden
                                + shard * self.shard_elements
                                + trip * self.consumer_threads * VEC_BF16
                                + consumer_tid * VEC_BF16
                            )
                            store_global_u32x4(
                                Int64(
                                    (
                                        local_contributions.iterator + output_element
                                    ).toint()
                                ),
                                load_shared_u32x4(
                                    rows.iterator
                                    + consumer_state.index * self.shard_elements
                                    + trip * self.consumer_threads * VEC_BF16
                                    + consumer_tid * VEC_BF16
                                ),
                            )
                        load_pipeline.consumer_release(consumer_state)
                        consumer_state.advance()
                    accum = cute.make_rmem_tensor(
                        cute.make_layout(
                            (self.vectors_per_thread, VEC_BF16),
                            stride=(VEC_BF16, 1),
                        ),
                        Float32,
                    )
                    accum.fill(Float32(0.0))
                    for _ in cutlass.range_constexpr(
                        self.contributions if self.top_k > 0 else 0
                    ):
                        load_pipeline.consumer_wait(consumer_state)
                        prob = cute.arch.load(
                            (stage_probs + consumer_state.index).llvm_ptr,
                            Float32,
                        )
                        if prob != Float32(0.0):
                            for trip in cutlass.range_constexpr(
                                self.vectors_per_thread
                            ):
                                stage_ptr = (
                                    rows.iterator
                                    + consumer_state.index * self.shard_elements
                                    + trip * self.consumer_threads * VEC_BF16
                                    + consumer_tid * VEC_BF16
                                )
                                values = packed_u32x4_to_bf16x8(
                                    load_shared_u32x4(stage_ptr)
                                ).to(Float32)
                                accum[trip, None].store(
                                    accum[trip, None].load() + values * prob
                                )
                        load_pipeline.consumer_release(consumer_state)
                        consumer_state.advance()
                    for trip in cutlass.range_constexpr(
                        self.vectors_per_thread if self.top_k > 0 else 0
                    ):
                        output_element = (
                            Int64(token) * self.hidden
                            + shard * self.shard_elements
                            + trip * self.consumer_threads * VEC_BF16
                            + consumer_tid * VEC_BF16
                        )
                        store_global_u32x4(
                            Int64(
                                (local_contributions.iterator + output_element).toint()
                            ),
                            bf16x8_to_packed_u32x4(
                                accum[trip, None].load().to(BFloat16)
                            ),
                        )
                clear_value = packed_negative_zero_bf16x8()
                token_pack = token * self.packs_per_token
                for clear_item in cutlass.range_constexpr(
                    self.clear_vectors_per_thread
                ):
                    clear_pack = consumer_tid + clear_item * self.consumer_threads
                    clear_element = (token_pack + clear_pack) * VEC_BF16
                    store_global_u32x4(
                        Int64((prenorm_mailbox.iterator + clear_element).toint()),
                        clear_value,
                    )
                cute.arch.sync_warp()
                consumer_token_progress += 1
                if lane == 0:
                    cute.arch.store(
                        (consumer_progress + warp - 1).llvm_ptr,
                        consumer_token_progress,
                        sem="release",
                        scope="cta",
                    )
                consumer_wave += self.cta_groups
                token = consumer_wave * self.tp + cta_slot
            finalize_join.arrive_and_wait()
            if cutlass.const_expr(self.rms_token_groups > 1):
                rms_group = (warp - 1) // self.rms_warps_per_token
                rms_group_warp = warp - 1 - rms_group * self.rms_warps_per_token
                copy_tid = rms_group_warp * WARP_SIZE + lane
            else:
                rms_group = Int32(0)
                rms_group_warp = warp - 1
                copy_tid = consumer_tid
            rms_pack_base = copy_tid
            rms_pack_stride = self.copy_threads
            if cutlass.const_expr(self.rms_shard_major):
                rms_pack_base = (
                    rms_group_warp
                    * self.reduction_shards_per_rms_warp
                    * self.packs_per_reduction_shard
                    + lane
                )
                rms_pack_stride = WARP_SIZE
            copy_wave = Int64(cta_group) + Int64(rms_group) * self.cta_groups
            copy_token = copy_wave * self.tp + cta_slot
            if cutlass.const_expr(self.rms_pipeline_stages > 1):
                rms_wave_stride = self.cta_groups * self.rms_token_groups
                while copy_token < Int64(m):
                    for rms_stage in cutlass.range_constexpr(self.rms_pipeline_stages):
                        stage_wave = copy_wave + rms_stage * rms_wave_stride
                        stage_token = stage_wave * self.tp + cta_slot
                        if stage_token < Int64(m):
                            stage_slot = (
                                rms_group * self.rms_pipeline_stages + rms_stage
                            )
                            token_pack = stage_token * self.packs_per_token
                            copy_fragments = cute.make_rmem_tensor(
                                cute.make_layout(
                                    (self.rms_vectors_per_thread, 4),
                                    stride=(4, 1),
                                ),
                                Uint32,
                            )
                            all_ready = cutlass.Boolean(0)
                            while not all_ready:
                                all_ready = cutlass.Boolean(1)
                                for item in cutlass.range_constexpr(
                                    self.rms_vectors_per_thread
                                ):
                                    pack = rms_pack_base + item * rms_pack_stride
                                    linear_pack = token_pack + pack
                                    packed = load_global_u32x4_address(
                                        Int64(
                                            (
                                                prenorm_mailbox.iterator
                                                + linear_pack * VEC_BF16
                                            ).toint()
                                        ),
                                        volatile=True,
                                    )
                                    copy_fragments[item, None].store(packed)
                                    all_ready = all_ready and (
                                        not fragment_has_negative_zero(packed)
                                    )
                            thread_sum = Float32(0.0)
                            if cutlass.const_expr(self.write_residual_output):
                                prenorm_packed = []
                            for item in cutlass.range_constexpr(
                                self.rms_vectors_per_thread
                            ):
                                pack = rms_pack_base + item * rms_pack_stride
                                prenorm = packed_u32x4_to_bf16x8(
                                    copy_fragments[item, None].load()
                                )
                                packed_prenorm = bf16x8_to_packed_u32x4(prenorm)
                                if cutlass.const_expr(self.write_residual_output):
                                    prenorm_packed.append(packed_prenorm)
                                # Finalize has drained `rows`; __init__ verifies the RMS layout fits.
                                store_shared_u32x4(
                                    rows.iterator
                                    + stage_slot * self.hidden
                                    + pack * VEC_BF16,
                                    packed_prenorm,
                                )
                                prenorm_f32 = prenorm.to(Float32)
                                thread_sum = thread_sum + (
                                    prenorm_f32 * prenorm_f32
                                ).reduce(
                                    cute.ReductionOp.ADD,
                                    init_val=Float32(0.0),
                                    reduction_profile=0,
                                )
                            if cutlass.const_expr(self.write_residual_output):
                                for item in cutlass.range_constexpr(
                                    self.rms_vectors_per_thread
                                ):
                                    pack = rms_pack_base + item * rms_pack_stride
                                    linear_pack = token_pack + pack
                                    residual_address = Int64(
                                        (
                                            residual_output.iterator
                                            + linear_pack * VEC_BF16
                                        ).toint()
                                    )
                                    store_global_u32x4(
                                        residual_address, prenorm_packed[item]
                                    )
                            warp_sum = cute.arch.warp_reduction_sum(thread_sum)
                            if lane == 0:
                                cute.arch.store(
                                    (
                                        norm_warp_sums
                                        + stage_slot * self.rms_warps_per_token
                                        + rms_group_warp
                                    ).llvm_ptr,
                                    warp_sum,
                                )
                    self._rms_arrive_and_wait(rms_group)
                    if warp == 1 + rms_group * self.rms_warps_per_token:
                        for rms_stage in cutlass.range_constexpr(
                            self.rms_pipeline_stages
                        ):
                            stage_wave = copy_wave + rms_stage * rms_wave_stride
                            stage_token = stage_wave * self.tp + cta_slot
                            if stage_token < Int64(m):
                                stage_slot = (
                                    rms_group * self.rms_pipeline_stages + rms_stage
                                )
                                cta_sum = Float32(0.0)
                                if lane < self.rms_warps_per_token:
                                    cta_sum = cute.arch.load(
                                        (
                                            norm_warp_sums
                                            + stage_slot * self.rms_warps_per_token
                                            + lane
                                        ).llvm_ptr,
                                        Float32,
                                    )
                                cta_sum = cute.arch.warp_reduction_sum(cta_sum)
                                if lane == 0:
                                    inv_rms = cute.math.rsqrt(
                                        cta_sum / Float32(self.hidden)
                                        + Float32(self.rms_epsilon),
                                        fastmath=True,
                                    )
                                    cute.arch.store(
                                        (norm_inv_rms + stage_slot).llvm_ptr,
                                        inv_rms,
                                    )
                    self._rms_arrive_and_wait(rms_group)
                    gamma_values = []
                    for item in cutlass.range_constexpr(self.rms_vectors_per_thread):
                        pack = rms_pack_base + item * rms_pack_stride
                        gamma_value = packed_u32x4_to_bf16x8(
                            load_global_u32x4_address(
                                Int64((gamma.iterator + pack * VEC_BF16).toint())
                            )
                        ).to(Float32)
                        if cutlass.const_expr(self.weight_bias != 0.0):
                            gamma_value = gamma_value + Float32(self.weight_bias)
                        gamma_values.append(gamma_value)
                    for rms_stage in cutlass.range_constexpr(self.rms_pipeline_stages):
                        stage_wave = copy_wave + rms_stage * rms_wave_stride
                        stage_token = stage_wave * self.tp + cta_slot
                        if stage_token < Int64(m):
                            stage_slot = (
                                rms_group * self.rms_pipeline_stages + rms_stage
                            )
                            token_pack = stage_token * self.packs_per_token
                            inv_rms = cute.arch.load(
                                (norm_inv_rms + stage_slot).llvm_ptr, Float32
                            )
                            norm_packed = []
                            for item in cutlass.range_constexpr(
                                self.rms_vectors_per_thread
                            ):
                                pack = rms_pack_base + item * rms_pack_stride
                                prenorm = packed_u32x4_to_bf16x8(
                                    load_shared_u32x4(
                                        rows.iterator
                                        + stage_slot * self.hidden
                                        + pack * VEC_BF16
                                    )
                                ).to(Float32)
                                result = (prenorm * inv_rms * gamma_values[item]).to(
                                    BFloat16
                                )
                                norm_packed.append(bf16x8_to_packed_u32x4(result))
                            for item in cutlass.range_constexpr(
                                self.rms_vectors_per_thread
                            ):
                                pack = rms_pack_base + item * rms_pack_stride
                                linear_pack = token_pack + pack
                                store_global_u32x4(
                                    Int64(
                                        (
                                            norm_output.iterator
                                            + linear_pack * VEC_BF16
                                        ).toint()
                                    ),
                                    norm_packed[item],
                                )
                    copy_wave += rms_wave_stride * self.rms_pipeline_stages
                    copy_token = copy_wave * self.tp + cta_slot
            if cutlass.const_expr(self.rms_pipeline_stages == 1):
                while copy_token < Int64(m):
                    token_pack = copy_token * self.packs_per_token
                    copy_values = []
                    copy_sources = []
                    if cutlass.const_expr(self.write_residual_output):
                        copy_destinations = []
                    for item in cutlass.range_constexpr(self.rms_vectors_per_thread):
                        pack = rms_pack_base + item * rms_pack_stride
                        linear_pack = token_pack + pack
                        source_address = Int64(
                            (prenorm_mailbox.iterator + linear_pack * VEC_BF16).toint()
                        )
                        copy_sources.append(source_address)
                        if cutlass.const_expr(self.write_residual_output):
                            copy_destinations.append(
                                Int64(
                                    (
                                        residual_output.iterator
                                        + linear_pack * VEC_BF16
                                    ).toint()
                                )
                            )
                    copy_fragments = cute.make_rmem_tensor(
                        cute.make_layout(
                            (self.rms_vectors_per_thread, 4),
                            stride=(4, 1),
                        ),
                        Uint32,
                    )
                    all_ready = cutlass.Boolean(0)
                    while not all_ready:
                        all_ready = cutlass.Boolean(1)
                        for item in cutlass.range_constexpr(
                            self.rms_vectors_per_thread
                        ):
                            packed = load_global_u32x4_address(
                                copy_sources[item],
                                volatile=True,
                            )
                            copy_fragments[item, None].store(packed)
                            all_ready = all_ready and (
                                not fragment_has_negative_zero(packed)
                            )
                    for item in cutlass.range_constexpr(self.rms_vectors_per_thread):
                        copy_values.append(copy_fragments[item, None].load())
                    prenorm_fragments = cute.make_rmem_tensor(
                        cute.make_layout(
                            (self.rms_vectors_per_thread, VEC_BF16),
                            stride=(VEC_BF16, 1),
                        ),
                        BFloat16,
                    )
                    thread_sum = Float32(0.0)
                    if cutlass.const_expr(self.write_residual_output):
                        prenorm_packed = []
                    for item in cutlass.range_constexpr(self.rms_vectors_per_thread):
                        pack = rms_pack_base + item * rms_pack_stride
                        prenorm = packed_u32x4_to_bf16x8(copy_values[item])
                        prenorm_fragments[item, None].store(prenorm)
                        packed_prenorm = bf16x8_to_packed_u32x4(prenorm)
                        if cutlass.const_expr(self.write_residual_output):
                            prenorm_packed.append(packed_prenorm)
                        prenorm_f32 = prenorm.to(Float32)
                        thread_sum = thread_sum + (prenorm_f32 * prenorm_f32).reduce(
                            cute.ReductionOp.ADD,
                            init_val=Float32(0.0),
                            reduction_profile=0,
                        )
                    if cutlass.const_expr(self.write_residual_output):
                        for item in cutlass.range_constexpr(
                            self.rms_vectors_per_thread
                        ):
                            store_global_u32x4(
                                copy_destinations[item], prenorm_packed[item]
                            )
                    warp_sum = cute.arch.warp_reduction_sum(thread_sum)
                    if lane == 0:
                        cute.arch.store((norm_warp_sums + warp - 1).llvm_ptr, warp_sum)
                    self._rms_arrive_and_wait(rms_group)
                    if warp == 1 + rms_group * self.rms_warps_per_token:
                        cta_sum = Float32(0.0)
                        if lane < self.rms_warps_per_token:
                            cta_sum = cute.arch.load(
                                (
                                    norm_warp_sums
                                    + rms_group * self.rms_warps_per_token
                                    + lane
                                ).llvm_ptr,
                                Float32,
                            )
                        cta_sum = cute.arch.warp_reduction_sum(cta_sum)
                        if lane == 0:
                            inv_rms = cute.math.rsqrt(
                                cta_sum / Float32(self.hidden)
                                + Float32(self.rms_epsilon),
                                fastmath=True,
                            )
                            cute.arch.store(
                                (norm_inv_rms + rms_group).llvm_ptr, inv_rms
                            )
                    self._rms_arrive_and_wait(rms_group)
                    inv_rms = cute.arch.load(
                        (norm_inv_rms + rms_group).llvm_ptr, Float32
                    )
                    gamma_values = []
                    for item in cutlass.range_constexpr(self.rms_vectors_per_thread):
                        pack = rms_pack_base + item * rms_pack_stride
                        gamma_value = packed_u32x4_to_bf16x8(
                            load_global_u32x4_address(
                                Int64((gamma.iterator + pack * VEC_BF16).toint())
                            )
                        ).to(Float32)
                        if cutlass.const_expr(self.weight_bias != 0.0):
                            gamma_value = gamma_value + Float32(self.weight_bias)
                        gamma_values.append(gamma_value)
                    norm_packed = []
                    for item in cutlass.range_constexpr(self.rms_vectors_per_thread):
                        pack = rms_pack_base + item * rms_pack_stride
                        linear_pack = token_pack + pack
                        prenorm_for_norm = (
                            prenorm_fragments[item, None].load().to(Float32)
                        )
                        result = (prenorm_for_norm * inv_rms * gamma_values[item]).to(
                            BFloat16
                        )
                        norm_packed.append(bf16x8_to_packed_u32x4(result))
                    for item in cutlass.range_constexpr(self.rms_vectors_per_thread):
                        pack = rms_pack_base + item * rms_pack_stride
                        linear_pack = token_pack + pack
                        store_global_u32x4(
                            Int64(
                                (norm_output.iterator + linear_pack * VEC_BF16).toint()
                            ),
                            norm_packed[item],
                        )
                    copy_wave += self.cta_groups * self.rms_token_groups
                    copy_token = copy_wave * self.tp + cta_slot
        elif warp == self.publisher_warp:
            owner_ready_address = cute.arch.load(
                (ready_counter_peer_addresses.iterator + cta_slot).llvm_ptr,
                Int64,
            )
            first_token = Int64(cta_group) * self.tp + cta_slot
            token_count = Int32(0)
            if first_token < Int64(m):
                token_count = Int32(
                    (Int64(m) + self.active_ctas - 1 - first_token) // self.active_ctas
                )
            published = Int32(0)
            while published < token_count:
                observed = token_count
                if lane < self.consumer_warps:
                    observed = cute.arch.load(
                        (consumer_progress + lane).llvm_ptr,
                        Int32,
                        sem="relaxed",
                        scope="cta",
                    )
                frontier = cute.arch.warp_reduction(
                    observed, lambda x, y: cutlass.min(x, y)
                )
                if frontier > published:
                    acquired = token_count
                    if lane < self.consumer_warps:
                        acquired = cute.arch.load(
                            (consumer_progress + lane).llvm_ptr,
                            Int32,
                            sem="acquire",
                            scope="cta",
                        )
                    frontier = cute.arch.warp_reduction(
                        acquired, lambda x, y: cutlass.min(x, y)
                    )
                    cute.arch.sync_warp()
                    batch = cutlass.min(frontier - published, Int32(WARP_SIZE))
                    if lane < batch:
                        sequence = Int64(published + lane)
                        publish_token = (
                            Int64(cta_group) + sequence * self.cta_groups
                        ) * self.tp + cta_slot
                        owner_token = publish_token // self.tp
                        remote_release_add1_u32(owner_ready_address + owner_token * 4)
                    published += batch
        elif (
            block < self.reduction_ctas
            and warp >= self.reduction_warp_begin
            and (warp < self.reduction_warp_begin + self.reduction_warps)
        ):
            reduction_warp = warp - self.reduction_warp_begin
            reduction_tid = reduction_warp * WARP_SIZE + lane
            reduction_barrier = pipeline.NamedBarrier(
                barrier_id=1, num_threads=self.reduction_threads
            )
            reduction_shard = block % self.tp
            local_token = Int64(block // self.tp)
            token = local_token * self.tp + self.rank
            while token < Int64(m):
                processed_index = local_token * self.tp + reduction_shard
                target = Uint32(0)
                if reduction_tid == 0:
                    ready_counter_address = (
                        ready_counters.iterator + local_token
                    ).llvm_ptr
                    processed_counter_address = (
                        processed_counters.iterator + processed_index
                    ).llvm_ptr
                    target = cute.arch.load(processed_counter_address, Uint32) + Uint32(
                        self.tp
                    )
                    observed = Uint32(0)
                    while observed != target:
                        observed = cute.arch.load(
                            ready_counter_address,
                            Uint32,
                            sem="relaxed",
                            scope="sys",
                        )
                    cute.arch.load(
                        ready_counter_address,
                        Uint32,
                        sem="acquire",
                        scope="sys",
                    )
                reduction_barrier.arrive_and_wait()
                values = []
                addresses = []
                token_pack = token * self.packs_per_token
                shard_pack = reduction_shard * self.packs_per_reduction_shard
                for item in cutlass.range_constexpr(self.reduction_vectors_per_thread):
                    pack = shard_pack + reduction_tid + item * self.reduction_threads
                    input_address = (
                        local_contributions_multicast_address + (token_pack + pack) * 16
                    )
                    output_address = (
                        prenorm_mailbox_multicast_address + (token_pack + pack) * 16
                    )
                    reduced_packed = ldmc_bf16x8(input_address)
                    reduced_values = packed_u32x4_to_bf16x8(reduced_packed).to(Float32)
                    if cutlass.const_expr(self.add_residual):
                        residual_values = packed_u32x4_to_bf16x8(
                            load_global_u32x4_address(
                                Int64(
                                    (
                                        residual_source.iterator
                                        + (token_pack + pack) * VEC_BF16
                                    ).toint()
                                )
                            )
                        ).to(Float32)
                        reduced_values = reduced_values + residual_values
                    reduced_packed = bf16x8_to_packed_u32x4(reduced_values.to(BFloat16))
                    values.append(sanitize_negative_zero_u32x4(reduced_packed))
                    addresses.append(output_address)
                for item in cutlass.range_constexpr(self.reduction_vectors_per_thread):
                    stmc_bf16x8(addresses[item], values[item])
                if reduction_tid == 0:
                    cute.arch.store(
                        (processed_counters.iterator + processed_index).llvm_ptr,
                        target,
                    )
                local_token += self.reduction_cta_groups
                token = local_token * self.tp + self.rank
        cute.arch.sync_threads()
        if cutlass.const_expr(self.enable_pdl):
            cute.arch.griddepcontrol_launch_dependents()
