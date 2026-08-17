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

"""Split low-latency BF16 MoE finalize, TP reduction, and RMSNorm."""

from __future__ import annotations

import math

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
from cutlass import BFloat16, Float32, Int32, Int64, Uint32

from ..cute_dsl_primitives import (
    NEGATIVE_ZERO_BF16_BITS,
    QUAD_BF16,
    VEC_BF16,
    WARP_SIZE,
    bf16x4_to_packed_u32x2,
    bf16x8_to_packed_u32x4,
    f32_to_bf16_bits,
    fragment_has_negative_zero,
    load_global_bf16_as_f32,
    load_global_bf16_as_f32_predicated,
    load_global_u32x2,
    load_global_u32x4,
    load_volatile_u32,
    map_shared_to_peer,
    packed_u32x2_to_bf16x4,
    packed_u32x4_to_bf16x8,
    sanitize_negative_zero_u32x2,
    sanitize_negative_zero_u32x4,
    shuffle_sync_idx_u32,
    stmc_bf16x2,
    stmc_bf16x4,
    stmc_bf16x8,
    store_global_u32,
    store_global_u32x4,
    store_lamport_sentinel_u32x4,
    store_shared_cluster_f32,
)

LAMPORT_GENERATIONS = 3
NEXT_STAGE = 0
ACTIVE_STAGE = 1


@cute.jit
def _group_leader_block_sum(
    value: Float32,
    warp_sums: cute.Tensor,
    warps: cutlass.Constexpr[int],
    leader_stride: cutlass.Constexpr[int],
) -> Float32:
    lane = cute.arch.lane_idx()
    warp = cute.arch.warp_idx()
    for offset in cutlass.range_constexpr(1, WARP_SIZE):
        if cutlass.const_expr(offset >= leader_stride and (offset & (offset - 1)) == 0):
            value = value + cute.arch.shuffle_sync_bfly(
                value,
                offset=offset,
                mask=-1,
                mask_and_clamp=31,
            )
    if lane == 0:
        cute.arch.store((warp_sums + warp).llvm_ptr, value)
    cute.arch.barrier()

    result = Float32(0.0)
    if warp == 0:
        if lane < Int32(warps):
            result = cute.arch.load(
                (warp_sums + lane).llvm_ptr,
                Float32,
            )
        result = cute.arch.warp_reduction_sum(result)
        if lane == 0:
            cute.arch.store(warp_sums.llvm_ptr, result)
    cute.arch.barrier()
    return cute.arch.load(warp_sums.llvm_ptr, Float32)


class _ScalarFinalizePublishDeviceKernel:
    def __init__(
        self,
        *,
        hidden: int,
        top_k: int,
        tp: int,
        rank: int,
        capacity_m: int,
        threads: int,
        routed_scaling_factor: float,
        include_shared_expert: bool,
        load_shared_expert_before_pdl: bool,
        enable_pdl: bool,
        prefetch_group: int,
    ) -> None:
        if hidden <= 0 or hidden % 2:
            raise ValueError("hidden must be a positive multiple of 2")
        self.hidden = hidden
        self.top_k = top_k
        self.tp = tp
        self.rank = rank
        self.capacity_m = capacity_m
        self.threads = threads
        self.routed_scaling_factor = routed_scaling_factor
        self.ctas_per_token = math.ceil(hidden / threads)
        self.include_shared_expert = include_shared_expert
        self.load_shared_expert_before_pdl = load_shared_expert_before_pdl
        self.enable_pdl = enable_pdl
        self.prefetch_group = prefetch_group
        self.prefetch_groups = (top_k + prefetch_group - 1) // prefetch_group

    def smem_size_in_bytes(self) -> int:
        return self.top_k * 8

    @cute.jit
    def __call__(
        self,
        routed_output: cute.Tensor,
        expert_weights: cute.Tensor,
        permuted_indices: cute.Tensor,
        shared_output: cute.Tensor,
        stage_state: cute.Tensor,
        contribution_mailbox_multicast_address: Int64,
        m: Int32,
        stream: cuda.CUstream,
    ) -> None:
        self.kernel(
            routed_output,
            expert_weights,
            permuted_indices,
            shared_output,
            stage_state,
            contribution_mailbox_multicast_address,
        ).launch(
            grid=(m * self.ctas_per_token, 1, 1),
            block=(self.threads, 1, 1),
            smem=self.smem_size_in_bytes(),
            stream=stream,
            use_pdl=self.enable_pdl,
        )

    @cute.kernel
    def kernel(
        self,
        routed_output: cute.Tensor,
        expert_weights: cute.Tensor,
        permuted_indices: cute.Tensor,
        shared_output: cute.Tensor,
        stage_state: cute.Tensor,
        contribution_mailbox_multicast_address: Int64,
    ) -> None:
        block, _, _ = cute.arch.block_idx()
        tidx, _, _ = cute.arch.thread_idx()
        token = block // self.ctas_per_token
        cta_in_token = block % self.ctas_per_token
        hidden_index = cta_in_token * self.threads + tidx

        smem = cutlass.utils.SmemAllocator()
        staged_indices = smem.allocate_array(Int32, self.top_k)
        staged_weights = smem.allocate_array(Float32, self.top_k)
        metadata_index = Int32(tidx)
        while metadata_index < Int32(self.top_k):
            element = Int64(token) * self.top_k + Int64(metadata_index)
            row = cute.arch.load(
                (permuted_indices.iterator + element).llvm_ptr,
                Int32,
            )
            weight = load_global_bf16_as_f32(
                Int64((expert_weights.iterator + element).toint())
            )
            if cutlass.const_expr(self.routed_scaling_factor != 1.0):
                weight = weight * Float32(self.routed_scaling_factor)
            if row == Int32(-1):
                weight = Float32(0.0)
            cute.arch.store(
                (staged_indices + metadata_index).llvm_ptr,
                row,
            )
            cute.arch.store(
                (staged_weights + metadata_index).llvm_ptr,
                weight,
            )
            metadata_index = metadata_index + self.threads
        cute.arch.barrier()

        shared_value = Float32(0.0)
        if cutlass.const_expr(
            self.include_shared_expert and self.load_shared_expert_before_pdl
        ):
            if hidden_index < Int32(self.hidden):
                shared_element = Int64(token) * self.hidden + Int64(hidden_index)
                shared_value = load_global_bf16_as_f32(
                    Int64((shared_output.iterator + shared_element).toint())
                )

        if cutlass.const_expr(self.enable_pdl):
            cute.arch.griddepcontrol_wait()

        if cutlass.const_expr(
            self.include_shared_expert and not self.load_shared_expert_before_pdl
        ):
            if hidden_index < Int32(self.hidden):
                shared_element = Int64(token) * self.hidden + Int64(hidden_index)
                shared_value = load_global_bf16_as_f32(
                    Int64((shared_output.iterator + shared_element).toint())
                )

        stage = load_volatile_u32(stage_state.iterator + NEXT_STAGE)
        bits = Uint32(0)
        if hidden_index < Int32(self.hidden):
            accumulator = Float32(0.0)
            if cutlass.const_expr(self.prefetch_group == 1):
                for k in cutlass.range_constexpr(self.top_k):
                    row = cute.arch.load((staged_indices + k).llvm_ptr, Int32)
                    weight = cute.arch.load(
                        (staged_weights + k).llvm_ptr,
                        Float32,
                    )
                    source_element = Int64(row) * self.hidden + Int64(hidden_index)
                    accumulator = (
                        accumulator
                        + load_global_bf16_as_f32_predicated(
                            Int64((routed_output.iterator + source_element).toint()),
                            Int32(row != Int32(-1)),
                        )
                        * weight
                    )
            else:
                inputs = cute.make_rmem_tensor(
                    cute.make_layout((self.prefetch_group,)),
                    Float32,
                )
                inputs.fill(Float32(0.0))
                for group in cutlass.range_constexpr(self.prefetch_groups):
                    for item in cutlass.range_constexpr(self.prefetch_group):
                        k = group * self.prefetch_group + item
                        if cutlass.const_expr(k < self.top_k):
                            row = cute.arch.load(
                                (staged_indices + k).llvm_ptr,
                                Int32,
                            )
                            source_element = Int64(row) * self.hidden + Int64(
                                hidden_index
                            )
                            inputs[item] = load_global_bf16_as_f32_predicated(
                                Int64(
                                    (routed_output.iterator + source_element).toint()
                                ),
                                Int32(row != Int32(-1)),
                            )
                    for item in cutlass.range_constexpr(self.prefetch_group):
                        k = group * self.prefetch_group + item
                        if cutlass.const_expr(k < self.top_k):
                            accumulator = accumulator + inputs[item] * cute.arch.load(
                                (staged_weights + k).llvm_ptr,
                                Float32,
                            )
            if cutlass.const_expr(self.include_shared_expert):
                accumulator = accumulator + shared_value
            bits = f32_to_bf16_bits(accumulator)
            if bits == Uint32(NEGATIVE_ZERO_BF16_BITS):
                bits = Uint32(0)

        # Unlike the quad path, this publishes first; Lamport sentinels gate consumers.
        if block == 0 and tidx == 0:
            store_global_u32(
                stage_state.iterator + ACTIVE_STAGE,
                stage,
            )
        cute.arch.barrier()
        if cutlass.const_expr(self.enable_pdl):
            cute.arch.griddepcontrol_launch_dependents()

        lane = cute.arch.lane_idx()
        partner_bits = shuffle_sync_idx_u32(
            bits,
            Int32(lane | Int32(1)),
        )
        if (lane & Int32(1)) == Int32(0) and hidden_index < Int32(self.hidden):
            packed = bits | (partner_bits << Uint32(16))
            mailbox_element = (
                (Int64(stage) * self.tp + self.rank) * self.capacity_m + Int64(token)
            ) * self.hidden + Int64(hidden_index)
            stmc_bf16x2(
                contribution_mailbox_multicast_address + mailbox_element * 2,
                packed,
            )


class _QuadFinalizePublishDeviceKernel:
    def __init__(
        self,
        *,
        hidden: int,
        top_k: int,
        tp: int,
        rank: int,
        capacity_m: int,
        threads: int,
        routed_scaling_factor: float,
        include_shared_expert: bool,
        load_shared_expert_before_pdl: bool,
        enable_pdl: bool,
        prefetch_group: int,
    ) -> None:
        if hidden <= 0 or hidden % QUAD_BF16:
            raise ValueError("hidden must be a positive multiple of 4")
        self.hidden = hidden
        self.top_k = top_k
        self.tp = tp
        self.rank = rank
        self.capacity_m = capacity_m
        self.threads = threads
        self.routed_scaling_factor = routed_scaling_factor
        self.fragments = hidden // QUAD_BF16
        self.ctas_per_token = math.ceil(self.fragments / threads)
        self.include_shared_expert = include_shared_expert
        self.load_shared_expert_before_pdl = load_shared_expert_before_pdl
        self.enable_pdl = enable_pdl
        self.prefetch_group = prefetch_group
        self.prefetch_groups = (top_k + prefetch_group - 1) // prefetch_group

    def smem_size_in_bytes(self) -> int:
        return self.top_k * 8

    @cute.jit
    def __call__(
        self,
        routed_output: cute.Tensor,
        expert_weights: cute.Tensor,
        permuted_indices: cute.Tensor,
        shared_output: cute.Tensor,
        stage_state: cute.Tensor,
        contribution_mailbox_multicast_address: Int64,
        m: Int32,
        stream: cuda.CUstream,
    ) -> None:
        self.kernel(
            routed_output,
            expert_weights,
            permuted_indices,
            shared_output,
            stage_state,
            contribution_mailbox_multicast_address,
        ).launch(
            grid=(m * self.ctas_per_token, 1, 1),
            block=(self.threads, 1, 1),
            smem=self.smem_size_in_bytes(),
            stream=stream,
            use_pdl=self.enable_pdl,
        )

    @cute.kernel
    def kernel(
        self,
        routed_output: cute.Tensor,
        expert_weights: cute.Tensor,
        permuted_indices: cute.Tensor,
        shared_output: cute.Tensor,
        stage_state: cute.Tensor,
        contribution_mailbox_multicast_address: Int64,
    ) -> None:
        block, _, _ = cute.arch.block_idx()
        tidx, _, _ = cute.arch.thread_idx()
        token = block // self.ctas_per_token
        cta_in_token = block % self.ctas_per_token
        fragment = cta_in_token * self.threads + tidx

        smem = cutlass.utils.SmemAllocator()
        staged_indices = smem.allocate_array(Int32, self.top_k)
        staged_weights = smem.allocate_array(Float32, self.top_k)
        metadata_index = Int32(tidx)
        while metadata_index < Int32(self.top_k):
            element = Int64(token) * self.top_k + Int64(metadata_index)
            row = cute.arch.load(
                (permuted_indices.iterator + element).llvm_ptr,
                Int32,
            )
            weight = load_global_bf16_as_f32(
                Int64((expert_weights.iterator + element).toint())
            )
            if cutlass.const_expr(self.routed_scaling_factor != 1.0):
                weight = weight * Float32(self.routed_scaling_factor)
            if row == Int32(-1):
                weight = Float32(0.0)
            cute.arch.store(
                (staged_indices + metadata_index).llvm_ptr,
                row,
            )
            cute.arch.store(
                (staged_weights + metadata_index).llvm_ptr,
                weight,
            )
            metadata_index = metadata_index + self.threads
        cute.arch.barrier()

        if cutlass.const_expr(self.include_shared_expert):
            shared_values = cute.make_rmem_tensor(
                cute.make_layout((QUAD_BF16,)), BFloat16
            )
            shared_values.fill(BFloat16(0.0))
            if cutlass.const_expr(self.load_shared_expert_before_pdl):
                if fragment < self.fragments:
                    shared_element = (
                        Int64(token) * self.hidden + Int64(fragment) * QUAD_BF16
                    )
                    shared_pointer = cute.make_ptr(
                        BFloat16,
                        (shared_output.iterator + shared_element).llvm_ptr,
                        cute.AddressSpace.gmem,
                        assumed_align=8,
                    )
                    shared_values.store(
                        packed_u32x2_to_bf16x4(load_global_u32x2(shared_pointer))
                    )

        if cutlass.const_expr(self.enable_pdl):
            cute.arch.griddepcontrol_wait()

        if cutlass.const_expr(
            self.include_shared_expert and not self.load_shared_expert_before_pdl
        ):
            if fragment < self.fragments:
                shared_element = (
                    Int64(token) * self.hidden + Int64(fragment) * QUAD_BF16
                )
                shared_pointer = cute.make_ptr(
                    BFloat16,
                    (shared_output.iterator + shared_element).llvm_ptr,
                    cute.AddressSpace.gmem,
                    assumed_align=8,
                )
                shared_values.store(
                    packed_u32x2_to_bf16x4(load_global_u32x2(shared_pointer))
                )

        stage = load_volatile_u32(stage_state.iterator + NEXT_STAGE)
        if fragment < self.fragments:
            accumulator = cute.make_rmem_tensor(cute.make_layout((QUAD_BF16,)), Float32)
            accumulator.fill(Float32(0.0))
            if cutlass.const_expr(self.prefetch_group == 1):
                for k in cutlass.range_constexpr(self.top_k):
                    row = cute.arch.load((staged_indices + k).llvm_ptr, Int32)
                    weight = cute.arch.load(
                        (staged_weights + k).llvm_ptr,
                        Float32,
                    )
                    if row != Int32(-1):
                        source_element = (
                            Int64(row) * self.hidden + Int64(fragment) * QUAD_BF16
                        )
                        source_pointer = cute.make_ptr(
                            BFloat16,
                            (routed_output.iterator + source_element).llvm_ptr,
                            cute.AddressSpace.gmem,
                            assumed_align=8,
                        )
                        accumulator.store(
                            accumulator.load()
                            + packed_u32x2_to_bf16x4(
                                load_global_u32x2(source_pointer)
                            ).to(Float32)
                            * weight
                        )
            else:
                inputs = cute.make_rmem_tensor(
                    cute.make_layout((self.prefetch_group, 2)),
                    Uint32,
                )
                inputs.fill(Uint32(0))
                for group in cutlass.range_constexpr(self.prefetch_groups):
                    for item in cutlass.range_constexpr(self.prefetch_group):
                        k = group * self.prefetch_group + item
                        if cutlass.const_expr(k < self.top_k):
                            row = cute.arch.load(
                                (staged_indices + k).llvm_ptr,
                                Int32,
                            )
                            for word in cutlass.range_constexpr(2):
                                inputs[item, word] = Uint32(0)
                            if row != Int32(-1):
                                source_element = (
                                    Int64(row) * self.hidden
                                    + Int64(fragment) * QUAD_BF16
                                )
                                source_pointer = cute.make_ptr(
                                    BFloat16,
                                    (routed_output.iterator + source_element).llvm_ptr,
                                    cute.AddressSpace.gmem,
                                    assumed_align=8,
                                )
                                source = load_global_u32x2(source_pointer)
                                for word in cutlass.range_constexpr(2):
                                    inputs[item, word] = source[word]
                    for item in cutlass.range_constexpr(self.prefetch_group):
                        k = group * self.prefetch_group + item
                        if cutlass.const_expr(k < self.top_k):
                            source = cute.make_rmem_tensor(
                                cute.make_layout((2,)),
                                Uint32,
                            )
                            for word in cutlass.range_constexpr(2):
                                source[word] = inputs[item, word]
                            accumulator.store(
                                accumulator.load()
                                + packed_u32x2_to_bf16x4(source.load()).to(Float32)
                                * cute.arch.load(
                                    (staged_weights + k).llvm_ptr,
                                    Float32,
                                )
                            )
            result = accumulator.load()
            if cutlass.const_expr(self.include_shared_expert):
                result = result + shared_values.load().to(Float32)
            result_packed = bf16x4_to_packed_u32x2(result.to(BFloat16))
            packed = sanitize_negative_zero_u32x2(result_packed)
            mailbox_element = (
                (Int64(stage) * self.tp + self.rank) * self.capacity_m + Int64(token)
            ) * self.hidden + Int64(fragment) * QUAD_BF16
            stmc_bf16x4(
                contribution_mailbox_multicast_address + mailbox_element * 2,
                packed,
            )

        if block == 0 and tidx == 0:
            store_global_u32(
                stage_state.iterator + ACTIVE_STAGE,
                stage,
            )
        cute.arch.barrier()
        if cutlass.const_expr(self.enable_pdl):
            cute.arch.griddepcontrol_launch_dependents()


class _SharedOnlyPublishDeviceKernel:
    def __init__(
        self,
        *,
        hidden: int,
        tp: int,
        rank: int,
        capacity_m: int,
        elements_per_thread: int,
        threads: int,
        release_before_store: bool,
        enable_pdl: bool,
    ) -> None:
        if elements_per_thread not in (1, QUAD_BF16, VEC_BF16):
            raise ValueError("elements_per_thread must be 1, 4, or 8")
        if hidden <= 0 or hidden % elements_per_thread:
            raise ValueError("hidden must divide evenly across thread fragments")
        self.hidden = hidden
        self.tp = tp
        self.rank = rank
        self.capacity_m = capacity_m
        self.elements_per_thread = elements_per_thread
        self.threads = threads
        self.fragments = hidden // elements_per_thread
        self.ctas_per_token = math.ceil(self.fragments / threads)
        self.release_before_store = release_before_store
        self.enable_pdl = enable_pdl

    @cute.jit
    def __call__(
        self,
        shared_output: cute.Tensor,
        stage_state: cute.Tensor,
        contribution_mailbox_multicast_address: Int64,
        m: Int32,
        stream: cuda.CUstream,
    ) -> None:
        self.kernel(
            shared_output,
            stage_state,
            contribution_mailbox_multicast_address,
        ).launch(
            grid=(m * self.ctas_per_token, 1, 1),
            block=(self.threads, 1, 1),
            stream=stream,
            use_pdl=self.enable_pdl,
        )

    @cute.kernel
    def kernel(
        self,
        shared_output: cute.Tensor,
        stage_state: cute.Tensor,
        contribution_mailbox_multicast_address: Int64,
    ) -> None:
        block, _, _ = cute.arch.block_idx()
        tidx, _, _ = cute.arch.thread_idx()
        token = block // self.ctas_per_token
        cta_in_token = block % self.ctas_per_token
        fragment = cta_in_token * self.threads + tidx

        if cutlass.const_expr(self.enable_pdl):
            cute.arch.griddepcontrol_wait()

        if cutlass.const_expr(self.elements_per_thread == 1):
            bits = Uint32(0)
            if fragment < self.fragments:
                element = Int64(token) * self.hidden + Int64(fragment)
                bits = f32_to_bf16_bits(
                    load_global_bf16_as_f32(
                        Int64((shared_output.iterator + element).toint())
                    )
                )
                if bits == Uint32(NEGATIVE_ZERO_BF16_BITS):
                    bits = Uint32(0)
            partner_bits = shuffle_sync_idx_u32(
                bits,
                Int32(cute.arch.lane_idx() | Int32(1)),
            )
            packed_word = bits | (partner_bits << Uint32(16))
        elif cutlass.const_expr(self.elements_per_thread == QUAD_BF16):
            packed_words = cute.make_rmem_tensor(cute.make_layout((2,)), Uint32)
            packed_words.fill(Uint32(0))
            if fragment < self.fragments:
                element = Int64(token) * self.hidden + Int64(fragment) * QUAD_BF16
                pointer = cute.make_ptr(
                    BFloat16,
                    (shared_output.iterator + element).llvm_ptr,
                    cute.AddressSpace.gmem,
                    assumed_align=8,
                )
                packed_words.store(
                    sanitize_negative_zero_u32x2(load_global_u32x2(pointer))
                )
        else:
            packed_words = cute.make_rmem_tensor(cute.make_layout((4,)), Uint32)
            packed_words.fill(Uint32(0))
            if fragment < self.fragments:
                element = Int64(token) * self.hidden + Int64(fragment) * VEC_BF16
                pointer = cute.make_ptr(
                    BFloat16,
                    (shared_output.iterator + element).llvm_ptr,
                    cute.AddressSpace.gmem,
                    assumed_align=16,
                )
                packed_words.store(
                    sanitize_negative_zero_u32x4(load_global_u32x4(pointer))
                )

        stage = load_volatile_u32(stage_state.iterator + NEXT_STAGE)
        if cutlass.const_expr(self.release_before_store):
            if block == 0 and tidx == 0:
                store_global_u32(
                    stage_state.iterator + ACTIVE_STAGE,
                    stage,
                )
            cute.arch.barrier()
            if cutlass.const_expr(self.enable_pdl):
                cute.arch.griddepcontrol_launch_dependents()

        if cutlass.const_expr(self.elements_per_thread == 1):
            if (cute.arch.lane_idx() & Int32(1)) == Int32(
                0
            ) and fragment < self.fragments:
                mailbox_element = (
                    (Int64(stage) * self.tp + self.rank) * self.capacity_m
                    + Int64(token)
                ) * self.hidden + Int64(fragment)
                stmc_bf16x2(
                    contribution_mailbox_multicast_address + mailbox_element * 2,
                    packed_word,
                )
        elif cutlass.const_expr(self.elements_per_thread == QUAD_BF16):
            if fragment < self.fragments:
                mailbox_element = (
                    (Int64(stage) * self.tp + self.rank) * self.capacity_m
                    + Int64(token)
                ) * self.hidden + Int64(fragment) * QUAD_BF16
                stmc_bf16x4(
                    contribution_mailbox_multicast_address + mailbox_element * 2,
                    packed_words,
                )
        else:
            if fragment < self.fragments:
                mailbox_element = (
                    (Int64(stage) * self.tp + self.rank) * self.capacity_m
                    + Int64(token)
                ) * self.hidden + Int64(fragment) * VEC_BF16
                stmc_bf16x8(
                    contribution_mailbox_multicast_address + mailbox_element * 2,
                    packed_words,
                )

        if cutlass.const_expr(not self.release_before_store):
            if block == 0 and tidx == 0:
                store_global_u32(
                    stage_state.iterator + ACTIVE_STAGE,
                    stage,
                )
            cute.arch.barrier()
            if cutlass.const_expr(self.enable_pdl):
                cute.arch.griddepcontrol_launch_dependents()


class _LamportResidualRMSNormDeviceKernel:
    def __init__(
        self,
        *,
        hidden: int,
        tp: int,
        capacity_m: int,
        cluster_size: int,
        rank_lanes: int,
        threads: int,
        rms_epsilon: float,
        weight_bias: float,
        add_residual: bool,
        write_residual_output: bool,
        enable_pdl: bool,
    ) -> None:
        if rank_lanes not in (1, 2, 4, 8):
            raise ValueError("rank_lanes must be 1, 2, 4, or 8")
        if tp % rank_lanes:
            raise ValueError("tp must be divisible by rank_lanes")
        if threads <= 0 or threads % WARP_SIZE or threads % rank_lanes:
            raise ValueError("threads must be a positive warp and rank-lane multiple")
        if hidden <= 0 or hidden % VEC_BF16:
            raise ValueError("hidden must be a positive multiple of 8")
        self.hidden = hidden
        self.tp = tp
        self.capacity_m = capacity_m
        self.cluster_size = cluster_size
        self.rank_lanes = rank_lanes
        self.threads = threads
        self.rms_epsilon = rms_epsilon
        self.weight_bias = weight_bias
        self.add_residual = add_residual
        self.write_residual_output = write_residual_output
        self.enable_pdl = enable_pdl
        self.fragments = hidden // VEC_BF16
        self.groups_per_cta = threads // rank_lanes
        self.fragment_stride = cluster_size * self.groups_per_cta
        self.trips = math.ceil(self.fragments / self.fragment_stride)
        self.warps = threads // WARP_SIZE
        self.rank_waves = tp // rank_lanes

    def smem_size_in_bytes(self) -> int:
        return (self.warps + self.cluster_size) * 4

    @cute.jit
    def __call__(
        self,
        contribution_mailbox: cute.Tensor,
        residual_source: cute.Tensor,
        gamma: cute.Tensor,
        residual_output: cute.Tensor,
        norm_output: cute.Tensor,
        stage_state: cute.Tensor,
        m: Int32,
        stream: cuda.CUstream,
    ) -> None:
        self.kernel(
            contribution_mailbox,
            residual_source,
            gamma,
            residual_output,
            norm_output,
            stage_state,
        ).launch(
            grid=(m, self.cluster_size, 1),
            block=(self.threads, 1, 1),
            cluster=(1, self.cluster_size, 1),
            smem=self.smem_size_in_bytes(),
            stream=stream,
            use_pdl=self.enable_pdl,
        )

    @cute.kernel
    def kernel(
        self,
        contribution_mailbox: cute.Tensor,
        residual_source: cute.Tensor,
        gamma: cute.Tensor,
        residual_output: cute.Tensor,
        norm_output: cute.Tensor,
        stage_state: cute.Tensor,
    ) -> None:
        tidx, _, _ = cute.arch.thread_idx()
        token, _, _ = cute.arch.block_idx()
        cluster_rank = cute.arch.make_warp_uniform(cute.arch.block_idx_in_cluster())
        rank_lane = tidx % self.rank_lanes
        group = tidx // self.rank_lanes
        base_fragment = cluster_rank * self.groups_per_cta + group

        prenorm_fragments = cute.make_rmem_tensor(
            cute.make_layout(
                (self.trips, VEC_BF16),
                stride=(VEC_BF16, 1),
            ),
            BFloat16,
        )
        prenorm_fragments.fill(BFloat16(0.0))
        gamma_fragments = cute.make_rmem_tensor(
            cute.make_layout(
                (self.trips, VEC_BF16),
                stride=(VEC_BF16, 1),
            ),
            BFloat16,
        )
        gamma_fragments.fill(BFloat16(0.0))
        if cutlass.const_expr(self.add_residual):
            residual_fragments = cute.make_rmem_tensor(
                cute.make_layout(
                    (self.trips, VEC_BF16),
                    stride=(VEC_BF16, 1),
                ),
                BFloat16,
            )
            residual_fragments.fill(BFloat16(0.0))

        for trip in cutlass.range_constexpr(self.trips):
            fragment = base_fragment + trip * self.fragment_stride
            if fragment < self.fragments and rank_lane == 0:
                gamma_element = Int64(fragment) * VEC_BF16
                gamma_pointer = cute.make_ptr(
                    BFloat16,
                    (gamma.iterator + gamma_element).llvm_ptr,
                    cute.AddressSpace.gmem,
                    assumed_align=16,
                )
                gamma_fragments[trip, None].store(
                    packed_u32x4_to_bf16x8(load_global_u32x4(gamma_pointer))
                )
                if cutlass.const_expr(self.add_residual):
                    residual_element = Int64(token) * self.hidden + gamma_element
                    residual_pointer = cute.make_ptr(
                        BFloat16,
                        (residual_source.iterator + residual_element).llvm_ptr,
                        cute.AddressSpace.gmem,
                        assumed_align=16,
                    )
                    residual_fragments[trip, None].store(
                        packed_u32x4_to_bf16x8(load_global_u32x4(residual_pointer))
                    )

        if cutlass.const_expr(self.enable_pdl):
            cute.arch.griddepcontrol_wait()

        active_stage = load_volatile_u32(stage_state.iterator + ACTIVE_STAGE)
        for trip in cutlass.range_constexpr(self.trips):
            fragment = base_fragment + trip * self.fragment_stride
            lane_packed = cute.make_rmem_tensor(
                cute.make_layout((self.rank_waves, 4)),
                Uint32,
            )
            lane_packed.fill(Uint32(0))
            dirty = fragment < self.fragments
            while dirty:
                dirty = False
                for wave in cutlass.range_constexpr(self.rank_waves):
                    source_rank = wave * self.rank_lanes + rank_lane
                    if fragment < self.fragments:
                        source_element = (
                            (Int64(active_stage) * self.tp + Int64(source_rank))
                            * self.capacity_m
                            + Int64(token)
                        ) * self.hidden + Int64(fragment) * VEC_BF16
                        source_pointer = cute.make_ptr(
                            BFloat16,
                            (contribution_mailbox.iterator + source_element).llvm_ptr,
                            cute.AddressSpace.gmem,
                            assumed_align=16,
                        )
                        packed = load_global_u32x4(
                            source_pointer,
                            volatile=True,
                        )
                        dirty = dirty | fragment_has_negative_zero(packed)
                        for word in cutlass.range_constexpr(4):
                            lane_packed[wave, word] = packed[word]

            lane_sum = cute.make_rmem_tensor(cute.make_layout((VEC_BF16,)), Float32)
            lane_sum.fill(Float32(0.0))
            for wave in cutlass.range_constexpr(self.rank_waves):
                packed = cute.make_rmem_tensor(cute.make_layout((4,)), Uint32)
                for word in cutlass.range_constexpr(4):
                    packed[word] = lane_packed[wave, word]
                lane_sum.store(
                    lane_sum.load() + packed_u32x4_to_bf16x8(packed.load()).to(Float32)
                )
            for offset in cutlass.range_constexpr(1, 5):
                if cutlass.const_expr(offset < self.rank_lanes and offset in (1, 2, 4)):
                    for element in cutlass.range_constexpr(VEC_BF16):
                        lane_sum[element] = lane_sum[
                            element
                        ] + cute.arch.shuffle_sync_bfly(
                            lane_sum[element],
                            offset=offset,
                            mask=-1,
                            mask_and_clamp=31,
                        )

            if fragment < self.fragments and rank_lane == 0:
                prenorm = lane_sum.load()
                if cutlass.const_expr(self.add_residual):
                    prenorm = prenorm + residual_fragments[trip, None].load().to(
                        Float32
                    )
                prenorm_bf16 = prenorm.to(BFloat16)
                prenorm_fragments[trip, None].store(prenorm_bf16)
                if cutlass.const_expr(self.write_residual_output):
                    output_element = (
                        Int64(token) * self.hidden + Int64(fragment) * VEC_BF16
                    )
                    store_global_u32x4(
                        Int64((residual_output.iterator + output_element).toint()),
                        bf16x8_to_packed_u32x4(prenorm_bf16),
                    )

        if token == 0 and cluster_rank == 0 and tidx == 0:
            store_global_u32(
                stage_state.iterator + NEXT_STAGE,
                (active_stage + Uint32(1)) % Uint32(LAMPORT_GENERATIONS),
            )
        if cutlass.const_expr(self.enable_pdl):
            cute.arch.griddepcontrol_launch_dependents()

        for trip in cutlass.range_constexpr(self.trips):
            fragment = base_fragment + trip * self.fragment_stride
            for wave in cutlass.range_constexpr(self.rank_waves):
                source_rank = wave * self.rank_lanes + rank_lane
                if fragment < self.fragments:
                    source_element = (
                        (Int64(active_stage) * self.tp + Int64(source_rank))
                        * self.capacity_m
                        + Int64(token)
                    ) * self.hidden + Int64(fragment) * VEC_BF16
                    store_lamport_sentinel_u32x4(
                        Int64((contribution_mailbox.iterator + source_element).toint())
                    )

        thread_sum = Float32(0.0)
        for trip in cutlass.range_constexpr(self.trips):
            fragment = base_fragment + trip * self.fragment_stride
            if fragment < self.fragments and rank_lane == 0:
                values = prenorm_fragments[trip, None].load().to(Float32)
                thread_sum = thread_sum + (values * values).reduce(
                    cute.ReductionOp.ADD,
                    init_val=Float32(0.0),
                    reduction_profile=0,
                )

        smem = cutlass.utils.SmemAllocator()
        warp_sums = smem.allocate_array(Float32, self.warps)
        cluster_sums = smem.allocate_array(Float32, self.cluster_size)
        cta_sum = _group_leader_block_sum(
            thread_sum,
            warp_sums,
            self.warps,
            self.rank_lanes,
        )
        if tidx < self.cluster_size:
            local_slot = cluster_sums + cluster_rank
            remote_slot = map_shared_to_peer(local_slot, Int32(tidx))
            store_shared_cluster_f32(remote_slot, cta_sum)
        cute.arch.cluster_arrive()
        cute.arch.cluster_wait()

        full_sum = Float32(0.0)
        for peer in cutlass.range_constexpr(self.cluster_size):
            full_sum = full_sum + cute.arch.load(
                (cluster_sums + peer).llvm_ptr,
                Float32,
            )
        inv_rms = cute.math.rsqrt(
            full_sum / Float32(self.hidden) + Float32(self.rms_epsilon),
            fastmath=True,
        )
        for trip in cutlass.range_constexpr(self.trips):
            fragment = base_fragment + trip * self.fragment_stride
            if fragment < self.fragments and rank_lane == 0:
                gamma_values = gamma_fragments[trip, None].load().to(Float32)
                if cutlass.const_expr(self.weight_bias != 0.0):
                    gamma_values = gamma_values + Float32(self.weight_bias)
                result = (
                    prenorm_fragments[trip, None].load().to(Float32)
                    * inv_rms
                    * gamma_values
                ).to(BFloat16)
                output_element = Int64(token) * self.hidden + Int64(fragment) * VEC_BF16
                store_global_u32x4(
                    Int64((norm_output.iterator + output_element).toint()),
                    bf16x8_to_packed_u32x4(result),
                )
