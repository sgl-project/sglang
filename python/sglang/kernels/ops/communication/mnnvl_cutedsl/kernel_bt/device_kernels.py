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

"""Three-stage BF16 MoE finalize, TP reduction, and RMSNorm for SM100."""

from __future__ import annotations

import math

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
from cutlass import BFloat16, Float32, Int32, Int64, Uint32

from ..cute_dsl_primitives import (
    NEGATIVE_ZERO_BF16_BITS,
    VEC_BF16,
    WARP_SIZE,
    bf16x2_to_packed_u32,
    bf16x4_to_packed_u32x2,
    bf16x8_to_packed_u32x4,
    f32_to_bf16_bits,
    fragment_has_negative_zero,
    load_global_bf16_as_f32,
    load_global_bf16_as_f32_predicated,
    load_global_u32,
    load_global_u32_predicated,
    load_global_u32x2,
    load_global_u32x2_predicated,
    load_global_u32x4,
    load_volatile_u32,
    packed_u32_to_bf16x2,
    packed_u32x2_to_bf16x4,
    packed_u32x4_to_bf16x8,
    sanitize_negative_zero_u32,
    sanitize_negative_zero_u32x2,
    sanitize_negative_zero_u32x4,
    stmc_bf16x8,
    store_global_u16_bits,
    store_global_u32,
    store_global_u32_address,
    store_global_u32x2,
    store_global_u32x4,
    store_lamport_sentinel_u32x4,
)

LAMPORT_GENERATIONS = 3
NEXT_STAGE = 0
ACTIVE_STAGE = 1


@cute.jit
def _block_sum(
    value: Float32,
    warp_sums: cute.Tensor,
    warps: cutlass.Constexpr[int],
) -> Float32:
    lane = cute.arch.lane_idx()
    warp = cute.arch.warp_idx()
    value = cute.arch.warp_reduction_sum(value)
    if lane == 0:
        cute.arch.store((warp_sums + warp).llvm_ptr, value)
    cute.arch.barrier()

    result = Float32(0.0)
    if warp == 0:
        if lane < Int32(warps):
            result = cute.arch.load((warp_sums + lane).llvm_ptr, Float32)
        result = cute.arch.warp_reduction_sum(result)
        if lane == 0:
            cute.arch.store(warp_sums.llvm_ptr, result)
    cute.arch.barrier()
    return cute.arch.load(warp_sums.llvm_ptr, Float32)


class _ScalarFinalizeUnicastDeviceKernel:
    def __init__(
        self,
        *,
        hidden_size: int,
        top_k: int,
        tp_size: int,
        rank: int,
        local_capacity: int,
        threads: int,
        routed_scaling_factor: float,
        include_shared_expert: bool,
        load_shared_expert_before_pdl: bool,
        enable_pdl: bool,
        prefetch_group: int,
    ) -> None:
        self.hidden_size = hidden_size
        self.top_k = top_k
        self.tp_size = tp_size
        self.rank = rank
        self.local_capacity = local_capacity
        self.threads = threads
        self.routed_scaling_factor = routed_scaling_factor
        self.include_shared_expert = include_shared_expert
        self.load_shared_expert_before_pdl = load_shared_expert_before_pdl
        self.enable_pdl = enable_pdl
        self.prefetch_group = prefetch_group
        self.prefetch_groups = (top_k + prefetch_group - 1) // prefetch_group
        self.ctas_per_token = math.ceil(hidden_size / threads)

    @cute.jit
    def __call__(
        self,
        routed_output: cute.Tensor,
        expert_weights: cute.Tensor,
        permuted_indices: cute.Tensor,
        shared_output: cute.Tensor,
        stage_state: cute.Tensor,
        contribution_mailbox_peer_addresses: cute.Tensor,
        m: Int32,
        stream: cuda.CUstream,
    ) -> None:
        self.kernel(
            routed_output,
            expert_weights,
            permuted_indices,
            shared_output,
            stage_state,
            contribution_mailbox_peer_addresses,
        ).launch(
            grid=(m * self.ctas_per_token, 1, 1),
            block=(self.threads, 1, 1),
            smem=self.top_k * 8,
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
        contribution_mailbox_peer_addresses: cute.Tensor,
    ) -> None:
        block, _, _ = cute.arch.block_idx()
        tidx, _, _ = cute.arch.thread_idx()
        token = block // self.ctas_per_token
        hidden_index = (block % self.ctas_per_token) * self.threads + tidx

        smem = cutlass.utils.SmemAllocator()
        staged_indices = smem.allocate_array(Int32, self.top_k)
        staged_weights = smem.allocate_array(Float32, self.top_k)
        metadata_index = Int32(tidx)
        while metadata_index < Int32(self.top_k):
            metadata_offset = Int64(token) * self.top_k + Int64(metadata_index)
            routed_index = cute.arch.load(
                (permuted_indices.iterator + metadata_offset).llvm_ptr,
                Int32,
            )
            weight = load_global_bf16_as_f32(
                Int64((expert_weights.iterator + metadata_offset).toint())
            )
            if cutlass.const_expr(self.routed_scaling_factor != 1.0):
                weight = weight * Float32(self.routed_scaling_factor)
            if routed_index == Int32(-1):
                weight = Float32(0.0)
            cute.arch.store(
                (staged_indices + metadata_index).llvm_ptr,
                routed_index,
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
            if hidden_index < Int32(self.hidden_size):
                shared_offset = Int64(token) * self.hidden_size + Int64(hidden_index)
                shared_value = load_global_bf16_as_f32(
                    Int64((shared_output.iterator + shared_offset).toint())
                )

        if cutlass.const_expr(self.enable_pdl):
            cute.arch.griddepcontrol_wait()

        if cutlass.const_expr(
            self.include_shared_expert and not self.load_shared_expert_before_pdl
        ):
            if hidden_index < Int32(self.hidden_size):
                shared_offset = Int64(token) * self.hidden_size + Int64(hidden_index)
                shared_value = load_global_bf16_as_f32(
                    Int64((shared_output.iterator + shared_offset).toint())
                )

        stage = load_volatile_u32(stage_state.iterator + NEXT_STAGE)
        if hidden_index < Int32(self.hidden_size):
            accumulator = Float32(0.0)
            if cutlass.const_expr(self.prefetch_group == 1):
                for k in cutlass.range_constexpr(self.top_k):
                    routed_index = cute.arch.load(
                        (staged_indices + k).llvm_ptr,
                        Int32,
                    )
                    weight = cute.arch.load(
                        (staged_weights + k).llvm_ptr,
                        Float32,
                    )
                    source_offset = Int64(routed_index) * self.hidden_size + Int64(
                        hidden_index
                    )
                    accumulator = (
                        accumulator
                        + load_global_bf16_as_f32_predicated(
                            Int64((routed_output.iterator + source_offset).toint()),
                            Int32(routed_index != Int32(-1)),
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
                            routed_index = cute.arch.load(
                                (staged_indices + k).llvm_ptr,
                                Int32,
                            )
                            source_offset = Int64(
                                routed_index
                            ) * self.hidden_size + Int64(hidden_index)
                            inputs[item] = load_global_bf16_as_f32_predicated(
                                Int64((routed_output.iterator + source_offset).toint()),
                                Int32(routed_index != Int32(-1)),
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

            destination_rank = token % self.tp_size
            local_token = token // self.tp_size
            destination_base = cute.arch.load(
                (
                    contribution_mailbox_peer_addresses.iterator + destination_rank
                ).llvm_ptr,
                Int64,
            )
            destination_offset = (
                (Int64(stage) * self.tp_size + self.rank) * self.local_capacity
                + Int64(local_token)
            ) * self.hidden_size + Int64(hidden_index)
            store_global_u16_bits(
                destination_base + destination_offset * 2,
                bits,
            )

        if block == 0 and tidx == 0:
            store_global_u32(stage_state.iterator + ACTIVE_STAGE, stage)
        cute.arch.barrier()
        if cutlass.const_expr(self.enable_pdl):
            cute.arch.griddepcontrol_launch_dependents()


class _NarrowVectorFinalizeUnicastDeviceKernel:
    def __init__(
        self,
        *,
        hidden_size: int,
        top_k: int,
        tp_size: int,
        rank: int,
        local_capacity: int,
        threads: int,
        elements_per_thread: int,
        routed_scaling_factor: float,
        include_shared_expert: bool,
        load_shared_expert_before_pdl: bool,
        enable_pdl: bool,
        prefetch_group: int,
    ) -> None:
        self.hidden_size = hidden_size
        self.top_k = top_k
        self.tp_size = tp_size
        self.rank = rank
        self.local_capacity = local_capacity
        self.threads = threads
        self.elements_per_thread = elements_per_thread
        self.routed_scaling_factor = routed_scaling_factor
        self.include_shared_expert = include_shared_expert
        self.load_shared_expert_before_pdl = load_shared_expert_before_pdl
        self.enable_pdl = enable_pdl
        self.prefetch_group = prefetch_group
        self.prefetch_groups = (top_k + prefetch_group - 1) // prefetch_group
        self.words_per_fragment = elements_per_thread // 2
        self.fragments = hidden_size // elements_per_thread
        self.ctas_per_token = math.ceil(self.fragments / threads)

    @cute.jit
    def __call__(
        self,
        routed_output: cute.Tensor,
        expert_weights: cute.Tensor,
        permuted_indices: cute.Tensor,
        shared_output: cute.Tensor,
        stage_state: cute.Tensor,
        contribution_mailbox_peer_addresses: cute.Tensor,
        m: Int32,
        stream: cuda.CUstream,
    ) -> None:
        self.kernel(
            routed_output,
            expert_weights,
            permuted_indices,
            shared_output,
            stage_state,
            contribution_mailbox_peer_addresses,
        ).launch(
            grid=(m * self.ctas_per_token, 1, 1),
            block=(self.threads, 1, 1),
            smem=self.top_k * 8,
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
        contribution_mailbox_peer_addresses: cute.Tensor,
    ) -> None:
        block, _, _ = cute.arch.block_idx()
        tidx, _, _ = cute.arch.thread_idx()
        token = block // self.ctas_per_token
        fragment = (block % self.ctas_per_token) * self.threads + tidx

        smem = cutlass.utils.SmemAllocator()
        staged_indices = smem.allocate_array(Int32, self.top_k)
        staged_weights = smem.allocate_array(Float32, self.top_k)
        metadata_index = Int32(tidx)
        while metadata_index < Int32(self.top_k):
            metadata_offset = Int64(token) * self.top_k + Int64(metadata_index)
            routed_index = cute.arch.load(
                (permuted_indices.iterator + metadata_offset).llvm_ptr,
                Int32,
            )
            weight = load_global_bf16_as_f32(
                Int64((expert_weights.iterator + metadata_offset).toint())
            )
            if cutlass.const_expr(self.routed_scaling_factor != 1.0):
                weight = weight * Float32(self.routed_scaling_factor)
            if routed_index == Int32(-1):
                weight = Float32(0.0)
            cute.arch.store(
                (staged_indices + metadata_index).llvm_ptr,
                routed_index,
            )
            cute.arch.store(
                (staged_weights + metadata_index).llvm_ptr,
                weight,
            )
            metadata_index = metadata_index + self.threads
        cute.arch.barrier()

        if cutlass.const_expr(self.include_shared_expert):
            shared_values = cute.make_rmem_tensor(
                cute.make_layout((self.elements_per_thread,)),
                BFloat16,
            )
            shared_values.fill(BFloat16(0.0))
            if cutlass.const_expr(self.load_shared_expert_before_pdl):
                if fragment < self.fragments:
                    shared_offset = (
                        Int64(token) * self.hidden_size
                        + Int64(fragment) * self.elements_per_thread
                    )
                    shared_pointer = cute.make_ptr(
                        BFloat16,
                        (shared_output.iterator + shared_offset).llvm_ptr,
                        cute.AddressSpace.gmem,
                        assumed_align=self.elements_per_thread * 2,
                    )
                    if cutlass.const_expr(self.elements_per_thread == 2):
                        shared_values.store(
                            packed_u32_to_bf16x2(load_global_u32(shared_pointer))
                        )
                    else:
                        shared_values.store(
                            packed_u32x2_to_bf16x4(load_global_u32x2(shared_pointer))
                        )

        if cutlass.const_expr(self.enable_pdl):
            cute.arch.griddepcontrol_wait()

        if cutlass.const_expr(
            self.include_shared_expert and not self.load_shared_expert_before_pdl
        ):
            if fragment < self.fragments:
                shared_offset = (
                    Int64(token) * self.hidden_size
                    + Int64(fragment) * self.elements_per_thread
                )
                shared_pointer = cute.make_ptr(
                    BFloat16,
                    (shared_output.iterator + shared_offset).llvm_ptr,
                    cute.AddressSpace.gmem,
                    assumed_align=self.elements_per_thread * 2,
                )
                if cutlass.const_expr(self.elements_per_thread == 2):
                    shared_values.store(
                        packed_u32_to_bf16x2(load_global_u32(shared_pointer))
                    )
                else:
                    shared_values.store(
                        packed_u32x2_to_bf16x4(load_global_u32x2(shared_pointer))
                    )

        stage = load_volatile_u32(stage_state.iterator + NEXT_STAGE)
        if fragment < self.fragments:
            accumulator = cute.make_rmem_tensor(
                cute.make_layout((self.elements_per_thread,)),
                Float32,
            )
            accumulator.fill(Float32(0.0))
            if cutlass.const_expr(self.prefetch_group == 1):
                for k in cutlass.range_constexpr(self.top_k):
                    routed_index = cute.arch.load(
                        (staged_indices + k).llvm_ptr,
                        Int32,
                    )
                    weight = cute.arch.load(
                        (staged_weights + k).llvm_ptr,
                        Float32,
                    )
                    source_offset = (
                        Int64(routed_index) * self.hidden_size
                        + Int64(fragment) * self.elements_per_thread
                    )
                    source_pointer = cute.make_ptr(
                        BFloat16,
                        (routed_output.iterator + source_offset).llvm_ptr,
                        cute.AddressSpace.gmem,
                        assumed_align=self.elements_per_thread * 2,
                    )
                    if cutlass.const_expr(self.elements_per_thread == 2):
                        source = packed_u32_to_bf16x2(
                            load_global_u32_predicated(
                                source_pointer,
                                Int32(routed_index != Int32(-1)),
                            )
                        ).to(Float32)
                    else:
                        source = packed_u32x2_to_bf16x4(
                            load_global_u32x2_predicated(
                                source_pointer,
                                Int32(routed_index != Int32(-1)),
                            )
                        ).to(Float32)
                    accumulator.store(accumulator.load() + source * weight)
            else:
                inputs = cute.make_rmem_tensor(
                    cute.make_layout((self.prefetch_group, self.words_per_fragment)),
                    Uint32,
                )
                inputs.fill(Uint32(0))
                for group in cutlass.range_constexpr(self.prefetch_groups):
                    for item in cutlass.range_constexpr(self.prefetch_group):
                        k = group * self.prefetch_group + item
                        if cutlass.const_expr(k < self.top_k):
                            routed_index = cute.arch.load(
                                (staged_indices + k).llvm_ptr,
                                Int32,
                            )
                            source_offset = (
                                Int64(routed_index) * self.hidden_size
                                + Int64(fragment) * self.elements_per_thread
                            )
                            source_pointer = cute.make_ptr(
                                BFloat16,
                                (routed_output.iterator + source_offset).llvm_ptr,
                                cute.AddressSpace.gmem,
                                assumed_align=self.elements_per_thread * 2,
                            )
                            if cutlass.const_expr(self.elements_per_thread == 2):
                                inputs[item, 0] = load_global_u32_predicated(
                                    source_pointer,
                                    Int32(routed_index != Int32(-1)),
                                )
                            else:
                                source = load_global_u32x2_predicated(
                                    source_pointer,
                                    Int32(routed_index != Int32(-1)),
                                )
                                for word in cutlass.range_constexpr(
                                    self.words_per_fragment
                                ):
                                    inputs[item, word] = source[word]
                    for item in cutlass.range_constexpr(self.prefetch_group):
                        k = group * self.prefetch_group + item
                        if cutlass.const_expr(k < self.top_k):
                            if cutlass.const_expr(self.elements_per_thread == 2):
                                source_values = packed_u32_to_bf16x2(
                                    inputs[item, 0]
                                ).to(Float32)
                            else:
                                source = cute.make_rmem_tensor(
                                    cute.make_layout((self.words_per_fragment,)),
                                    Uint32,
                                )
                                for word in cutlass.range_constexpr(
                                    self.words_per_fragment
                                ):
                                    source[word] = inputs[item, word]
                                source_values = packed_u32x2_to_bf16x4(
                                    source.load()
                                ).to(Float32)
                            accumulator.store(
                                accumulator.load()
                                + source_values
                                * cute.arch.load(
                                    (staged_weights + k).llvm_ptr,
                                    Float32,
                                )
                            )

            result = accumulator.load()
            if cutlass.const_expr(self.include_shared_expert):
                result = result + shared_values.load().to(Float32)
            result = result.to(BFloat16)
            destination_rank = token % self.tp_size
            local_token = token // self.tp_size
            destination_base = cute.arch.load(
                (
                    contribution_mailbox_peer_addresses.iterator + destination_rank
                ).llvm_ptr,
                Int64,
            )
            destination_offset = (
                (Int64(stage) * self.tp_size + self.rank) * self.local_capacity
                + Int64(local_token)
            ) * self.hidden_size + Int64(fragment) * self.elements_per_thread
            if cutlass.const_expr(self.elements_per_thread == 2):
                word = sanitize_negative_zero_u32(bf16x2_to_packed_u32(result))
                store_global_u32_address(
                    destination_base + destination_offset * 2,
                    word,
                )
            else:
                half = sanitize_negative_zero_u32x2(bf16x4_to_packed_u32x2(result))
                store_global_u32x2(
                    destination_base + destination_offset * 2,
                    half,
                )

        if block == 0 and tidx == 0:
            store_global_u32(stage_state.iterator + ACTIVE_STAGE, stage)
        cute.arch.barrier()
        if cutlass.const_expr(self.enable_pdl):
            cute.arch.griddepcontrol_launch_dependents()


class _VectorFinalizeUnicastDeviceKernel:
    def __init__(
        self,
        *,
        hidden_size: int,
        top_k: int,
        tp_size: int,
        rank: int,
        local_capacity: int,
        threads: int,
        routed_scaling_factor: float,
        include_shared_expert: bool,
        load_shared_expert_before_pdl: bool,
        enable_pdl: bool,
        prefetch_group: int,
    ) -> None:
        self.hidden_size = hidden_size
        self.top_k = top_k
        self.tp_size = tp_size
        self.rank = rank
        self.local_capacity = local_capacity
        self.threads = threads
        self.routed_scaling_factor = routed_scaling_factor
        self.include_shared_expert = include_shared_expert
        self.load_shared_expert_before_pdl = load_shared_expert_before_pdl
        self.enable_pdl = enable_pdl
        self.prefetch_group = prefetch_group
        self.prefetch_groups = (top_k + prefetch_group - 1) // prefetch_group
        self.words_per_fragment = VEC_BF16 // 2
        self.fragments = hidden_size // VEC_BF16
        self.ctas_per_token = math.ceil(self.fragments / threads)

    @cute.jit
    def __call__(
        self,
        routed_output: cute.Tensor,
        expert_weights: cute.Tensor,
        permuted_indices: cute.Tensor,
        shared_output: cute.Tensor,
        stage_state: cute.Tensor,
        contribution_mailbox_peer_addresses: cute.Tensor,
        m: Int32,
        stream: cuda.CUstream,
    ) -> None:
        self.kernel(
            routed_output,
            expert_weights,
            permuted_indices,
            shared_output,
            stage_state,
            contribution_mailbox_peer_addresses,
        ).launch(
            grid=(m * self.ctas_per_token, 1, 1),
            block=(self.threads, 1, 1),
            smem=self.top_k * 8,
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
        contribution_mailbox_peer_addresses: cute.Tensor,
    ) -> None:
        block, _, _ = cute.arch.block_idx()
        tidx, _, _ = cute.arch.thread_idx()
        token = block // self.ctas_per_token
        fragment = (block % self.ctas_per_token) * self.threads + tidx

        smem = cutlass.utils.SmemAllocator()
        staged_indices = smem.allocate_array(Int32, self.top_k)
        staged_weights = smem.allocate_array(Float32, self.top_k)
        metadata_index = Int32(tidx)
        while metadata_index < Int32(self.top_k):
            metadata_offset = Int64(token) * self.top_k + Int64(metadata_index)
            routed_index = cute.arch.load(
                (permuted_indices.iterator + metadata_offset).llvm_ptr,
                Int32,
            )
            weight = load_global_bf16_as_f32(
                Int64((expert_weights.iterator + metadata_offset).toint())
            )
            if cutlass.const_expr(self.routed_scaling_factor != 1.0):
                weight = weight * Float32(self.routed_scaling_factor)
            if routed_index == Int32(-1):
                # Vector loads are unpredicated; row zero is safe and its weight is zero.
                routed_index = Int32(0)
                weight = Float32(0.0)
            cute.arch.store(
                (staged_indices + metadata_index).llvm_ptr,
                routed_index,
            )
            cute.arch.store(
                (staged_weights + metadata_index).llvm_ptr,
                weight,
            )
            metadata_index = metadata_index + self.threads
        cute.arch.barrier()

        if cutlass.const_expr(self.include_shared_expert):
            shared_values = cute.make_rmem_tensor(
                cute.make_layout((VEC_BF16,)),
                BFloat16,
            )
            shared_values.fill(BFloat16(0.0))
            if cutlass.const_expr(self.load_shared_expert_before_pdl):
                if fragment < self.fragments:
                    shared_offset = (
                        Int64(token) * self.hidden_size + Int64(fragment) * VEC_BF16
                    )
                    shared_pointer = cute.make_ptr(
                        BFloat16,
                        (shared_output.iterator + shared_offset).llvm_ptr,
                        cute.AddressSpace.gmem,
                        assumed_align=16,
                    )
                    shared_values.store(
                        packed_u32x4_to_bf16x8(load_global_u32x4(shared_pointer))
                    )

        if cutlass.const_expr(self.enable_pdl):
            cute.arch.griddepcontrol_wait()

        if cutlass.const_expr(
            self.include_shared_expert and not self.load_shared_expert_before_pdl
        ):
            if fragment < self.fragments:
                shared_offset = (
                    Int64(token) * self.hidden_size + Int64(fragment) * VEC_BF16
                )
                shared_pointer = cute.make_ptr(
                    BFloat16,
                    (shared_output.iterator + shared_offset).llvm_ptr,
                    cute.AddressSpace.gmem,
                    assumed_align=16,
                )
                shared_values.store(
                    packed_u32x4_to_bf16x8(load_global_u32x4(shared_pointer))
                )

        stage = load_volatile_u32(stage_state.iterator + NEXT_STAGE)
        if fragment < self.fragments:
            accumulator = cute.make_rmem_tensor(
                cute.make_layout((VEC_BF16,)),
                Float32,
            )
            accumulator.fill(Float32(0.0))
            if cutlass.const_expr(self.prefetch_group == 1):
                for k in cutlass.range_constexpr(self.top_k):
                    routed_index = cute.arch.load(
                        (staged_indices + k).llvm_ptr,
                        Int32,
                    )
                    weight = cute.arch.load(
                        (staged_weights + k).llvm_ptr,
                        Float32,
                    )
                    source_offset = (
                        Int64(routed_index) * self.hidden_size
                        + Int64(fragment) * VEC_BF16
                    )
                    source_pointer = cute.make_ptr(
                        BFloat16,
                        (routed_output.iterator + source_offset).llvm_ptr,
                        cute.AddressSpace.gmem,
                        assumed_align=16,
                    )
                    source_values = packed_u32x4_to_bf16x8(
                        load_global_u32x4(source_pointer)
                    ).to(Float32)
                    if weight != Float32(0.0):
                        accumulator.store(accumulator.load() + source_values * weight)
            else:
                inputs = cute.make_rmem_tensor(
                    cute.make_layout((self.prefetch_group, self.words_per_fragment)),
                    Uint32,
                )
                inputs.fill(Uint32(0))
                for group in cutlass.range_constexpr(self.prefetch_groups):
                    for item in cutlass.range_constexpr(self.prefetch_group):
                        k = group * self.prefetch_group + item
                        if cutlass.const_expr(k < self.top_k):
                            routed_index = cute.arch.load(
                                (staged_indices + k).llvm_ptr,
                                Int32,
                            )
                            source_offset = (
                                Int64(routed_index) * self.hidden_size
                                + Int64(fragment) * VEC_BF16
                            )
                            source_pointer = cute.make_ptr(
                                BFloat16,
                                (routed_output.iterator + source_offset).llvm_ptr,
                                cute.AddressSpace.gmem,
                                assumed_align=16,
                            )
                            source = load_global_u32x4(source_pointer)
                            for word in cutlass.range_constexpr(
                                self.words_per_fragment
                            ):
                                inputs[item, word] = source[word]
                    for item in cutlass.range_constexpr(self.prefetch_group):
                        k = group * self.prefetch_group + item
                        if cutlass.const_expr(k < self.top_k):
                            source = cute.make_rmem_tensor(
                                cute.make_layout((self.words_per_fragment,)),
                                Uint32,
                            )
                            for word in cutlass.range_constexpr(
                                self.words_per_fragment
                            ):
                                source[word] = inputs[item, word]
                            weight = cute.arch.load(
                                (staged_weights + k).llvm_ptr,
                                Float32,
                            )
                            if weight != Float32(0.0):
                                accumulator.store(
                                    accumulator.load()
                                    + packed_u32x4_to_bf16x8(source.load()).to(Float32)
                                    * weight
                                )

            result = accumulator.load()
            if cutlass.const_expr(self.include_shared_expert):
                result = result + shared_values.load().to(Float32)
            result = result.to(BFloat16)
            packed = sanitize_negative_zero_u32x4(bf16x8_to_packed_u32x4(result))
            destination_rank = token % self.tp_size
            local_token = token // self.tp_size
            destination_base = cute.arch.load(
                (
                    contribution_mailbox_peer_addresses.iterator + destination_rank
                ).llvm_ptr,
                Int64,
            )
            destination_offset = (
                (Int64(stage) * self.tp_size + self.rank) * self.local_capacity
                + Int64(local_token)
            ) * self.hidden_size + Int64(fragment) * VEC_BF16
            store_global_u32x4(
                destination_base + destination_offset * 2,
                packed,
            )

        if block == 0 and tidx == 0:
            store_global_u32(stage_state.iterator + ACTIVE_STAGE, stage)
        cute.arch.barrier()
        if cutlass.const_expr(self.enable_pdl):
            cute.arch.griddepcontrol_launch_dependents()


class _SharedOnlyPublishDeviceKernel:
    def __init__(
        self,
        *,
        hidden_size: int,
        tp_size: int,
        rank: int,
        local_capacity: int,
        threads: int,
        vectors_per_thread: int,
        enable_pdl: bool,
    ) -> None:
        self.hidden_size = hidden_size
        self.tp_size = tp_size
        self.rank = rank
        self.local_capacity = local_capacity
        self.threads = threads
        self.vectors_per_thread = vectors_per_thread
        self.enable_pdl = enable_pdl
        self.fragments = hidden_size // VEC_BF16
        self.fragments_per_cta = threads * vectors_per_thread
        self.ctas_per_token = math.ceil(self.fragments / self.fragments_per_cta)

    @cute.jit
    def __call__(
        self,
        shared_output: cute.Tensor,
        stage_state: cute.Tensor,
        contribution_mailbox_peer_addresses: cute.Tensor,
        m: Int32,
        stream: cuda.CUstream,
    ) -> None:
        self.kernel(
            shared_output,
            stage_state,
            contribution_mailbox_peer_addresses,
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
        contribution_mailbox_peer_addresses: cute.Tensor,
    ) -> None:
        block, _, _ = cute.arch.block_idx()
        tidx, _, _ = cute.arch.thread_idx()
        token = block // self.ctas_per_token
        fragment_base = (block % self.ctas_per_token) * self.fragments_per_cta + tidx

        inputs = cute.make_rmem_tensor(
            cute.make_layout(
                (self.vectors_per_thread, 4),
                stride=(4, 1),
            ),
            Uint32,
        )
        inputs.fill(Uint32(0))

        if cutlass.const_expr(self.enable_pdl):
            cute.arch.griddepcontrol_wait()

        for trip in cutlass.range_constexpr(self.vectors_per_thread):
            fragment = fragment_base + trip * self.threads
            if fragment < self.fragments:
                source_offset = (
                    Int64(token) * self.hidden_size + Int64(fragment) * VEC_BF16
                )
                source_pointer = cute.make_ptr(
                    BFloat16,
                    (shared_output.iterator + source_offset).llvm_ptr,
                    cute.AddressSpace.gmem,
                    assumed_align=16,
                )
                packed = load_global_u32x4(source_pointer)
                for word in cutlass.range_constexpr(4):
                    inputs[trip, word] = packed[word]

        stage = load_volatile_u32(stage_state.iterator + NEXT_STAGE)
        destination_rank = token % self.tp_size
        local_token = token // self.tp_size
        destination_base = cute.arch.load(
            (contribution_mailbox_peer_addresses.iterator + destination_rank).llvm_ptr,
            Int64,
        )
        for trip in cutlass.range_constexpr(self.vectors_per_thread):
            fragment = fragment_base + trip * self.threads
            if fragment < self.fragments:
                packed = cute.make_rmem_tensor(cute.make_layout((4,)), Uint32)
                for word in cutlass.range_constexpr(4):
                    packed[word] = inputs[trip, word]
                destination_offset = (
                    (Int64(stage) * self.tp_size + self.rank) * self.local_capacity
                    + Int64(local_token)
                ) * self.hidden_size + Int64(fragment) * VEC_BF16
                store_global_u32x4(
                    destination_base + destination_offset * 2,
                    sanitize_negative_zero_u32x4(packed.load()),
                )

        if block == 0 and tidx == 0:
            store_global_u32(stage_state.iterator + ACTIVE_STAGE, stage)
        cute.arch.barrier()
        if cutlass.const_expr(self.enable_pdl):
            cute.arch.griddepcontrol_launch_dependents()


class _OwnerReduceMulticastDeviceKernel:
    def __init__(
        self,
        *,
        hidden_size: int,
        tp_size: int,
        rank: int,
        capacity_m: int,
        local_capacity: int,
        threads: int,
        add_residual: bool,
        enable_pdl: bool,
    ) -> None:
        self.hidden_size = hidden_size
        self.tp_size = tp_size
        self.rank = rank
        self.capacity_m = capacity_m
        self.local_capacity = local_capacity
        self.threads = threads
        self.add_residual = add_residual
        self.enable_pdl = enable_pdl
        self.fragments = hidden_size // VEC_BF16
        self.ctas_per_token = math.ceil(self.fragments / threads)

    @cute.jit
    def __call__(
        self,
        contribution_mailbox: cute.Tensor,
        residual_source: cute.Tensor,
        stage_state: cute.Tensor,
        prenorm_mailbox_multicast_address: Int64,
        m: Int32,
        stream: cuda.CUstream,
    ) -> None:
        local_tokens = (m + Int32(self.tp_size - 1)) // Int32(self.tp_size)
        self.kernel(
            contribution_mailbox,
            residual_source,
            stage_state,
            prenorm_mailbox_multicast_address,
            m,
        ).launch(
            grid=(local_tokens * self.ctas_per_token, 1, 1),
            block=(self.threads, 1, 1),
            stream=stream,
            use_pdl=self.enable_pdl,
        )

    @cute.kernel
    def kernel(
        self,
        contribution_mailbox: cute.Tensor,
        residual_source: cute.Tensor,
        stage_state: cute.Tensor,
        prenorm_mailbox_multicast_address: Int64,
        m: Int32,
    ) -> None:
        block, _, _ = cute.arch.block_idx()
        tidx, _, _ = cute.arch.thread_idx()
        local_token = block // self.ctas_per_token
        fragment = (block % self.ctas_per_token) * self.threads + tidx
        token = local_token * self.tp_size + self.rank
        active = token < m and fragment < self.fragments

        if cutlass.const_expr(self.add_residual):
            residual = cute.make_rmem_tensor(
                cute.make_layout((VEC_BF16,)),
                BFloat16,
            )
            residual.fill(BFloat16(0.0))
            if active:
                residual_offset = (
                    Int64(token) * self.hidden_size + Int64(fragment) * VEC_BF16
                )
                residual_pointer = cute.make_ptr(
                    BFloat16,
                    (residual_source.iterator + residual_offset).llvm_ptr,
                    cute.AddressSpace.gmem,
                    assumed_align=16,
                )
                residual.store(
                    packed_u32x4_to_bf16x8(load_global_u32x4(residual_pointer))
                )

        if cutlass.const_expr(self.enable_pdl):
            cute.arch.griddepcontrol_wait()

        stage = load_volatile_u32(stage_state.iterator + ACTIVE_STAGE)
        rank_values = cute.make_rmem_tensor(
            cute.make_layout((self.tp_size, 4)),
            Uint32,
        )
        rank_values.fill(Uint32(0))
        dirty = active
        while dirty:
            dirty = False
            for source_rank in cutlass.range_constexpr(self.tp_size):
                source_offset = (
                    (Int64(stage) * self.tp_size + source_rank) * self.local_capacity
                    + Int64(local_token)
                ) * self.hidden_size + Int64(fragment) * VEC_BF16
                source_pointer = cute.make_ptr(
                    BFloat16,
                    (contribution_mailbox.iterator + source_offset).llvm_ptr,
                    cute.AddressSpace.gmem,
                    assumed_align=16,
                )
                packed = load_global_u32x4(source_pointer, volatile=True)
                dirty = dirty | fragment_has_negative_zero(packed)
                for word in cutlass.range_constexpr(4):
                    rank_values[source_rank, word] = packed[word]
        if active:
            reduced = cute.make_rmem_tensor(
                cute.make_layout((VEC_BF16,)),
                Float32,
            )
            reduced.fill(Float32(0.0))
            for source_rank in cutlass.range_constexpr(self.tp_size):
                packed = cute.make_rmem_tensor(cute.make_layout((4,)), Uint32)
                for word in cutlass.range_constexpr(4):
                    packed[word] = rank_values[source_rank, word]
                reduced.store(
                    reduced.load() + packed_u32x4_to_bf16x8(packed.load()).to(Float32)
                )

            prenorm = reduced.load()
            if cutlass.const_expr(self.add_residual):
                prenorm = prenorm + residual.load().to(Float32)
            prenorm_bf16 = prenorm.to(BFloat16)
            output_offset = (
                Int64(stage) * self.capacity_m + Int64(token)
            ) * self.hidden_size + Int64(fragment) * VEC_BF16
            stmc_bf16x8(
                prenorm_mailbox_multicast_address + output_offset * 2,
                sanitize_negative_zero_u32x4(bf16x8_to_packed_u32x4(prenorm_bf16)),
            )

        if block == 0 and tidx == 0:
            store_global_u32(
                stage_state.iterator + NEXT_STAGE,
                (stage + Uint32(1)) % Uint32(LAMPORT_GENERATIONS),
            )
        cute.arch.barrier()
        if cutlass.const_expr(self.enable_pdl):
            cute.arch.griddepcontrol_launch_dependents()

        if active:
            for source_rank in cutlass.range_constexpr(self.tp_size):
                source_offset = (
                    (Int64(stage) * self.tp_size + source_rank) * self.local_capacity
                    + Int64(local_token)
                ) * self.hidden_size + Int64(fragment) * VEC_BF16
                store_lamport_sentinel_u32x4(
                    Int64((contribution_mailbox.iterator + source_offset).toint())
                )


class _MaterializeRMSNormDeviceKernel:
    def __init__(
        self,
        *,
        hidden_size: int,
        capacity_m: int,
        threads: int,
        rms_epsilon: float,
        weight_bias: float,
        write_residual_output: bool,
        enable_pdl: bool,
    ) -> None:
        self.hidden_size = hidden_size
        self.capacity_m = capacity_m
        self.threads = threads
        self.rms_epsilon = rms_epsilon
        self.weight_bias = weight_bias
        self.write_residual_output = write_residual_output
        self.enable_pdl = enable_pdl
        self.fragments = hidden_size // VEC_BF16
        self.trips = math.ceil(self.fragments / threads)
        self.warps = threads // WARP_SIZE

    @cute.jit
    def __call__(
        self,
        prenorm_mailbox: cute.Tensor,
        residual_output: cute.Tensor,
        norm_output: cute.Tensor,
        gamma: cute.Tensor,
        stage_state: cute.Tensor,
        m: Int32,
        stream: cuda.CUstream,
    ) -> None:
        self.kernel(
            prenorm_mailbox,
            residual_output,
            norm_output,
            gamma,
            stage_state,
        ).launch(
            grid=(m, 1, 1),
            block=(self.threads, 1, 1),
            smem=self.warps * 4,
            stream=stream,
            use_pdl=self.enable_pdl,
        )

    @cute.kernel
    def kernel(
        self,
        prenorm_mailbox: cute.Tensor,
        residual_output: cute.Tensor,
        norm_output: cute.Tensor,
        gamma: cute.Tensor,
        stage_state: cute.Tensor,
    ) -> None:
        tidx, _, _ = cute.arch.thread_idx()
        token, _, _ = cute.arch.block_idx()

        gamma_fragments = cute.make_rmem_tensor(
            cute.make_layout(
                (self.trips, VEC_BF16),
                stride=(VEC_BF16, 1),
            ),
            BFloat16,
        )
        prenorm_fragments = cute.make_rmem_tensor(
            cute.make_layout(
                (self.trips, VEC_BF16),
                stride=(VEC_BF16, 1),
            ),
            BFloat16,
        )
        gamma_fragments.fill(BFloat16(0.0))
        prenorm_fragments.fill(BFloat16(0.0))
        for trip in cutlass.range_constexpr(self.trips):
            fragment = tidx + trip * self.threads
            if fragment < self.fragments:
                gamma_pointer = cute.make_ptr(
                    BFloat16,
                    (gamma.iterator + Int64(fragment) * VEC_BF16).llvm_ptr,
                    cute.AddressSpace.gmem,
                    assumed_align=16,
                )
                gamma_fragments[trip, None].store(
                    packed_u32x4_to_bf16x8(load_global_u32x4(gamma_pointer))
                )

        if cutlass.const_expr(self.enable_pdl):
            cute.arch.griddepcontrol_wait()

        stage = load_volatile_u32(stage_state.iterator + ACTIVE_STAGE)
        thread_sum = Float32(0.0)
        for trip in cutlass.range_constexpr(self.trips):
            fragment = tidx + trip * self.threads
            if fragment < self.fragments:
                mailbox_offset = (
                    Int64(stage) * self.capacity_m + Int64(token)
                ) * self.hidden_size + Int64(fragment) * VEC_BF16
                mailbox_pointer = cute.make_ptr(
                    BFloat16,
                    (prenorm_mailbox.iterator + mailbox_offset).llvm_ptr,
                    cute.AddressSpace.gmem,
                    assumed_align=16,
                )
                packed = load_global_u32x4(mailbox_pointer, volatile=True)
                while fragment_has_negative_zero(packed):
                    packed = load_global_u32x4(mailbox_pointer, volatile=True)
                prenorm = packed_u32x4_to_bf16x8(packed)
                prenorm_fragments[trip, None].store(prenorm)

                if cutlass.const_expr(self.write_residual_output):
                    output_offset = (
                        Int64(token) * self.hidden_size + Int64(fragment) * VEC_BF16
                    )
                    store_global_u32x4(
                        Int64((residual_output.iterator + output_offset).toint()),
                        packed,
                    )

                values = prenorm.to(Float32)
                thread_sum = thread_sum + (values * values).reduce(
                    cute.ReductionOp.ADD,
                    init_val=Float32(0.0),
                    reduction_profile=0,
                )

        smem = cutlass.utils.SmemAllocator()
        warp_sums = smem.allocate_array(Float32, self.warps)
        full_sum = _block_sum(thread_sum, warp_sums, self.warps)
        inverse_rms = cute.math.rsqrt(
            full_sum / Float32(self.hidden_size) + Float32(self.rms_epsilon),
            fastmath=True,
        )
        for trip in cutlass.range_constexpr(self.trips):
            fragment = tidx + trip * self.threads
            if fragment < self.fragments:
                gamma_value = gamma_fragments[trip, None].load().to(Float32)
                if cutlass.const_expr(self.weight_bias != 0.0):
                    gamma_value = gamma_value + Float32(self.weight_bias)
                result = (
                    prenorm_fragments[trip, None].load().to(Float32)
                    * inverse_rms
                    * gamma_value
                ).to(BFloat16)
                output_offset = (
                    Int64(token) * self.hidden_size + Int64(fragment) * VEC_BF16
                )
                store_global_u32x4(
                    Int64((norm_output.iterator + output_offset).toint()),
                    bf16x8_to_packed_u32x4(result),
                )

        cute.arch.barrier()
        if cutlass.const_expr(self.enable_pdl):
            cute.arch.griddepcontrol_launch_dependents()

        for trip in cutlass.range_constexpr(self.trips):
            fragment = tidx + trip * self.threads
            if fragment < self.fragments:
                mailbox_offset = (
                    Int64(stage) * self.capacity_m + Int64(token)
                ) * self.hidden_size + Int64(fragment) * VEC_BF16
                store_lamport_sentinel_u32x4(
                    Int64((prenorm_mailbox.iterator + mailbox_offset).toint())
                )
